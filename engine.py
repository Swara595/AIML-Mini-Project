# engine.py - all the fraud-checking logic in one place, used by api.py (the web frontend)

# It re-uses your existing files: explain.py, rules.py, scam_detector.py, mule_detector.py, retrain.py

import os
import threading
from datetime import datetime
import pandas as pd
import joblib
import mlflow
import explain
import rules
import scam_detector
import mule_detector
import retrain


# must match the order pandas used in features.py (alphabetical)

TYPE_CODES = {"COLLECT_REQUEST": 0, "MERCHANT": 1, "P2P": 2, "QR_PAYMENT": 3}

URL_WORDS = ["http", "bit.ly", "tinyurl",

             "cutt.ly", ".xyz", ".top", ".info", ".com"]

FEEDBACK_FILE = "feedback.csv"


def plain_number(x):
    """numpy number -> normal python number (so it can be sent as JSON)"""

    x = float(x)

    if x.is_integer():

        return int(x)

    return x


class Engine:

    def __init__(self):

        self.lock = threading.Lock()

        self.load()

     # ---------- load models and history ----------

    def load(self):

        model, features, explainer = explain.load_model()

        self.model = model

        self.features = features

        self.explainer = explainer

        self.lightgbm_model = joblib.load(

            os.path.join(

                os.path.dirname(os.path.abspath(__file__)),

                "models",

                "lightgbm_model.pkl"

            )

        )

        self.scam_model = joblib.load("models/scam_model.pkl")

        self.mules = joblib.load("models/mule_accounts.pkl")

        self.df = pd.read_csv("data/features.csv")

        self.avg_amount = self.df.groupby(

            "sender_id")["amount"].mean().to_dict()

        self.past_receivers = {}

        for s, r in zip(self.df["sender_id"], self.df["receiver_id"]):

            if s not in self.past_receivers:

                self.past_receivers[s] = set()

            self.past_receivers[s].add(r)

        self.senders = sorted(

            [s for s in self.avg_amount if s.startswith("U")])

     # ---------- sender list ----------

    def sender_list(self):

        out = []

        for s in self.senders:

            out.append({"id": s, "avg": int(round(self.avg_amount[s]))})

        return out

     # ---------- demo payments (fill the form) ----------

    def demos(self):

        s = self.senders

        safe_receiver = self.df[self.df["sender_id"] == s[0]]["receiver_id"].mode()[

            0]

        first_mule = sorted(self.mules.keys())[0]

        return [

            {"id": "safe", "label": "Everyday payment", "payload": {

                "sender": s[0], "receiver": safe_receiver,

                "amount": int(self.avg_amount[s[0]]), "txn_type": "MERCHANT", "hour": 14,

                "new_device": False, "recent": 0, "message": "Groceries"}},

            {"id": "collect", "label": "Fake refund request", "payload": {

                "sender": s[1], "receiver": "S005",

                "amount": int(self.avg_amount[s[1]] * 12), "txn_type": "COLLECT_REQUEST",

                "hour": 3, "new_device": False, "recent": 0,

                "message": "Approve to receive refund"}},

            {"id": "phishing", "label": "Phishing link", "payload": {

                "sender": s[2], "receiver": "S011", "amount": 1500, "txn_type": "P2P",

                "hour": 20, "new_device": False, "recent": 0,

                "message": "KYC expired, update at upi-kyc-update.xyz/login"}},

            {"id": "takeover", "label": "Hijacked account", "payload": {

                "sender": s[3], "receiver": "S021", "amount": 4500, "txn_type": "P2P",

                "hour": 2, "new_device": True, "recent": 8, "message": ""}},

            {"id": "mule", "label": "Mule account", "payload": {

                "sender": s[4], "receiver": first_mule, "amount": 2500, "txn_type": "P2P",

                "hour": 16, "new_device": False, "recent": 0, "message": "Registration fee"}},

        ]

     # ---------- the brain: screen one payment ----------

    def score(self, p):

        sender = p["sender"]

        receiver = p["receiver"].strip()

        message = p.get("message", "") or ""

        if sender not in self.avg_amount:

            raise ValueError(

                "Unknown sender %s. Pick a sender from the list." % sender)

        avg = self.avg_amount[sender]

        is_new_payee = 0 if receiver in self.past_receivers.get(

            sender, set()) else 1

        has_url = 0

        for w in URL_WORDS:

            if w in message.lower():

                has_url = 1

        row = pd.Series({

            "amount": p["amount"], "hour": p["hour"], "amount_vs_avg": p["amount"] / avg,

            "txn_last_2min": p["recent"], "is_new_payee": is_new_payee,

            "is_new_device": 1 if p["new_device"] else 0, "has_url": has_url,

            "type_code": TYPE_CODES[p["txn_type"]],

            "txn_type": p["txn_type"], "message": message})

        # check 1: main model + SHAP reasons

        model_prob, shap_reasons = explain.explain_transaction(

            self.model, self.features, self.explainer, row)

        # LightGBM prediction
        lightgbm_input = pd.DataFrame([
            {feature: row[feature] for feature in self.features}
        ])

        lightgbm_prob = float(
            self.lightgbm_model.predict_proba(lightgbm_input)[0][1]
        )

        lightgbm_prediction = (
            "FRAUD" if lightgbm_prob >= 0.5 else "NORMAL"
        )

        # check 2: scam message detector

        msg_present = message.strip() != ""

        msg_prob, msg_label, msg_reasons = 0.0, "SAFE", []

        if msg_present:

            msg_prob, msg_label, msg_reasons = scam_detector.check_message(

                message, self.scam_model)

         # check 3: mule accounts (look at both sides of the payment)

        mule_found, mule_text, mule_account = False, [], ""

        for acc in [receiver, sender]:

            flag, why = mule_detector.is_mule(acc, self.mules)

            if flag:

                mule_found, mule_text, mule_account = True, why, acc

                break

         # final score = the highest of the three checks

        final = model_prob

        reasons = []

        if mule_found:

            final = max(final, 0.95)

            reasons.append({"source": "Mule check",

                            "text": "%s is a known mule account: %s" % (mule_account, "; ".join(mule_text))})

        if msg_label == "SCAM":

            final = max(final, 0.5)

            reasons.append(

                {"source": "Message", "text": "The message looks like a scam"})

            for r in msg_reasons:

                reasons.append(

                    {"source": "Message", "text": r[0].upper() + r[1:]})

        if model_prob >= 0.3:

            for r in shap_reasons:

                reasons.append(

                    {"source": "Model", "text": r[0].upper() + r[1:]})

        label, emoji, advice, ftype = rules.assess(

            row, final, receiver_is_mule=mule_found)

        if ftype == "Suspicious (type unknown)" and msg_label == "SCAM":

            ftype = "Scam message"

        features = {}

        for f in self.features:

            features[f] = plain_number(row[f])

        return {

            "label": label, "advice": advice, "fraud_type": ftype,

            "final": round(float(final), 4),

            "model": {"prob": round(float(model_prob), 4)},
            "lightgbm": {
                "prob": round(lightgbm_prob, 4),
                "prediction": lightgbm_prediction,
            },

            "message": {"present": msg_present, "label": msg_label, "prob": round(float(msg_prob), 4)},

            "mule": {"found": mule_found, "account": mule_account},

            "reasons": reasons,

            "avg": int(round(avg)), "ratio": round(float(p["amount"] / avg), 1),

            "is_new_payee": bool(is_new_payee),

            "features": features,

        }

     # ---------- feedback ----------

    def save_feedback(self, fb):

        flagged = 1 if fb["system_said"] != "GREEN" else 0

        true_label = flagged if fb["was_right"] else 1 - flagged

        record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                  "sender": fb["sender"], "receiver": fb["receiver"]}

        for f in self.features:

            record[f] = fb["features"].get(f, 0)

        record["system_said"] = fb["system_said"]

        record["is_fraud"] = true_label

        with self.lock:

            pd.DataFrame([record]).to_csv(FEEDBACK_FILE, mode="a",

                                          header=not os.path.exists(FEEDBACK_FILE), index=False)

    def feedback_info(self):

        if not os.path.exists(FEEDBACK_FILE):

            return {"count": 0, "recent": []}

        fb = pd.read_csv(FEEDBACK_FILE)

        recent = []

        for i in range(len(fb) - 1, max(len(fb) - 11, -1), -1):

            r = fb.iloc[i]

            recent.append({"time": str(r["time"]), "sender": r["sender"], "receiver": r["receiver"],

                           "amount": plain_number(r["amount"]), "system_said": r["system_said"],

                           "is_fraud": int(r["is_fraud"])})

        return {"count": int(len(fb)), "recent": recent}

    def run_retrain(self):

        with self.lock:

            result = retrain.retrain()

            if result.get("ok") and result.get("accepted"):

                self.load()           # use the new model straight away*

        return result

     # ---------- transaction feed ----------

    def feed(self, seed):

        df = self.df

        part = pd.concat([df[df["is_fraud"] == 1].sample(8, random_state=seed),

                          df[df["is_fraud"] == 0].sample(22, random_state=seed)])

        part = part.sample(frac=1, random_state=seed)

        probs = self.model.predict_proba(part[self.features])[:, 1]

        rows = []

        for i in range(len(part)):

            r = part.iloc[i]

            label = rules.risk_label(probs[i])

            actual = int(r["is_fraud"])

            flagged = 1 if label != "GREEN" else 0

            if actual == 1 and flagged == 1:

                outcome = "caught"

            elif actual == 1 and flagged == 0:

                outcome = "missed"

            elif actual == 0 and flagged == 1:

                outcome = "false alarm"

            else:

                outcome = "clear"

            rows.append({"txn_id": r["txn_id"], "sender": r["sender_id"], "receiver": r["receiver_id"],

                         "amount": int(r["amount"]), "txn_type": r["txn_type"],

                         "label": label, "score": round(float(probs[i]), 2),

                         "actual": "fraud" if actual == 1 else "normal",

                         "fraud_type": r["fraud_type"], "outcome": outcome})

        return rows

     # ---------- mule network ----------

    def mule_list(self):

        out = []

        for acc in sorted(self.mules.keys()):

            out.append({"id": acc, "senders": self.mules[acc]["senders"]})

        return out

    def mule_detail(self, account):

        info = self.mules[account]

        inbound = self.df[self.df["receiver_id"] == account].groupby("sender_id")[

            "amount"].sum()

        inbound = inbound.sort_values(ascending=False)

        victims = []

        for k, v in inbound.head(12).items():

            victims.append({"id": k, "amount": int(v)})

        outbound = self.df[self.df["sender_id"] == account].groupby("receiver_id")[

            "amount"].sum()

        cashouts = []

        for k, v in outbound.items():

            cashouts.append({"id": k, "amount": int(v)})

        return {"id": account, "senders": int(info["senders"]), "total_in": int(info["total_in"]),

                "total_out": int(info["total_out"]), "pass_on": round(float(info["pass_on"]), 3),

                "delay_min": round(float(info["delay_min"]), 1),

                "victims": victims, "cashouts": cashouts}

     # ---------- model report (from MLflow) ----------

    def report(self):

        try:

            runs = mlflow.search_runs(experiment_names=["upi_fraud_detection"])

            runs = runs.drop_duplicates(subset="tags.mlflow.runName")

            runs = runs[runs["metrics.precision"].notna()]

        except Exception:

            return {"runs": [], "has_shap": os.path.exists("shap_summary.png")}

        out = []

        for i in range(len(runs)):

            r = runs.iloc[i]

            name = r["tags.mlflow.runName"]

            if name.startswith("Scam_"):

                group = "Scam message"

            elif name.startswith("Mule_"):

                group = "Mule accounts"

            elif name.startswith("AutoML"):

                group = "AutoML"

            elif name.startswith("Retrain"):

                group = "Retraining"

            else:

                group = "Fraud model"

            f1 = r.get("metrics.f1")

            out.append({"name": name.replace("Scam_", "").replace("_", " "), "group": group,

                        "precision": round(float(r["metrics.precision"]), 3),

                        "recall": round(float(r["metrics.recall"]), 3),

                        "f1": None if pd.isna(f1) else round(float(f1), 3)})

        return {"runs": out, "has_shap": os.path.exists("shap_summary.png")}
