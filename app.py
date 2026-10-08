# app.py - Phase 9: Streamlit dashboard
# Run with:  streamlit run app.py
import os
from datetime import datetime
import pandas as pd
import joblib
import mlflow
import streamlit as st

import explain
import rules
import scam_detector
import mule_detector
import retrain

FEEDBACK_FILE = "feedback.csv"

st.set_page_config(page_title="UPI Fraud Detection",
                   page_icon="🛡️", layout="wide")

# must match the order pandas used in features.py (alphabetical)
TYPE_CODES = {"COLLECT_REQUEST": 0, "MERCHANT": 1, "P2P": 2, "QR_PAYMENT": 3}
URL_WORDS = ["http", "bit.ly", "tinyurl",
             "cutt.ly", ".xyz", ".top", ".info", ".com"]


# ---------- load everything once ----------
@st.cache_resource
def load_everything():
    model, features, explainer = explain.load_model()
    scam_model = joblib.load("models/scam_model.pkl")
    mules = joblib.load("models/mule_accounts.pkl")
    df = pd.read_csv("data/features.csv")

    avg_amount = df.groupby("sender_id")["amount"].mean().to_dict()
    past_receivers = {}
    for s, r in zip(df["sender_id"], df["receiver_id"]):
        if s not in past_receivers:
            past_receivers[s] = set()
        past_receivers[s].add(r)
    return {"model": model, "features": features, "explainer": explainer,
            "scam_model": scam_model, "mules": mules, "df": df,
            "avg_amount": avg_amount, "past_receivers": past_receivers}


@st.cache_resource
def load_graph():
    raw = pd.read_csv("data/upi_transactions.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    return mule_detector.build_graph(raw)


ctx = load_everything()
df = ctx["df"]
senders = sorted([s for s in ctx["avg_amount"] if s.startswith("U")])


# ---------- the brain: score one payment ----------
def score_payment(sender, receiver, amount, txn_type, hour, new_device, recent, message):
    avg = ctx["avg_amount"][sender]
    is_new_payee = 0 if receiver in ctx["past_receivers"].get(
        sender, set()) else 1
    has_url = 0
    for w in URL_WORDS:
        if w in message.lower():
            has_url = 1

    row = pd.Series({
        "amount": amount, "hour": hour, "amount_vs_avg": amount / avg,
        "txn_last_2min": recent, "is_new_payee": is_new_payee,
        "is_new_device": 1 if new_device else 0, "has_url": has_url,
        "type_code": TYPE_CODES[txn_type], "txn_type": txn_type, "message": message})

    # check 1: main model + SHAP reasons
    model_prob, shap_reasons = explain.explain_transaction(
        ctx["model"], ctx["features"], ctx["explainer"], row)

    # check 2: scam message detector
    msg_prob, msg_label, msg_reasons = 0.0, "SAFE", []
    if message.strip() != "":
        msg_prob, msg_label, msg_reasons = scam_detector.check_message(
            message, ctx["scam_model"])

    # check 3: mule accounts (look at both sides of the payment)
    mule_found, mule_text, mule_account = False, [], ""
    for acc in [receiver, sender]:
        flag, why = mule_detector.is_mule(acc, ctx["mules"])
        if flag:
            mule_found, mule_text, mule_account = True, why, acc
            break

    # final score = the highest of the three checks
    final = model_prob
    reasons = []
    if mule_found:
        final = max(final, 0.95)
        reasons.append("%s is a known mule account (%s)" %
                       (mule_account, "; ".join(mule_text)))
    if msg_label == "SCAM":
        final = max(final, 0.5)
        reasons.append("message looks like a scam")
        reasons += msg_reasons
    if model_prob >= 0.3:
        reasons += shap_reasons

    label, emoji, advice, ftype = rules.assess(
        row, final, receiver_is_mule=mule_found)
    if ftype == "Suspicious (type unknown)" and msg_label == "SCAM":
        ftype = "Scam message"

    return {"label": label, "emoji": emoji, "advice": advice, "type": ftype,
            "final": final, "model_prob": model_prob, "msg_prob": msg_prob,
            "msg_label": msg_label, "mule": mule_found, "reasons": reasons,
            "is_new_payee": is_new_payee, "avg": avg, "row": row}


# ---------- feedback: save the user's answer to feedback.csv ----------
def save_feedback(was_right):
    last = st.session_state["last"]
    res = last["res"]
    # did the system call it fraud?
    flagged = 1 if res["label"] != "GREEN" else 0
    true_label = flagged if was_right else 1 - \
        flagged     # "wrong" flips the answer

    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "sender": last["sender"], "receiver": last["receiver"]}
    for f in ctx["features"]:
        record[f] = res["row"][f]
    record["system_said"] = res["label"]
    record["is_fraud"] = true_label

    pd.DataFrame([record]).to_csv(FEEDBACK_FILE, mode="a",
                                  header=not os.path.exists(FEEDBACK_FILE), index=False)
    last["feedback_done"] = True


# ---------- quick demo cases (fill the form) ----------
def fill(sender, receiver, amount, txn_type, hour, new_device, recent, message):
    st.session_state["sender"] = sender
    st.session_state["receiver"] = receiver
    st.session_state["amount"] = int(amount)
    st.session_state["txn_type"] = txn_type
    st.session_state["hour"] = hour
    st.session_state["new_device"] = new_device
    st.session_state["recent"] = recent
    st.session_state["message"] = message


def demo_safe():
    s = senders[0]
    r = df[df["sender_id"] == s]["receiver_id"].mode()[0]
    fill(s, r, ctx["avg_amount"][s], "MERCHANT", 14, False, 0, "Groceries")


def demo_collect():
    s = senders[1]
    fill(s, "S005", ctx["avg_amount"][s] * 12, "COLLECT_REQUEST", 3, False, 0,
         "Approve to receive refund")


def demo_phishing():
    s = senders[2]
    fill(s, "S011", 1500, "P2P", 20, False, 0,
         "KYC expired, update at upi-kyc-update.xyz/login")


def demo_takeover():
    s = senders[3]
    fill(s, "S021", 4500, "P2P", 2, True, 8, "")


def demo_mule():
    m = sorted(ctx["mules"].keys())[0]
    fill(senders[4], m, 2500, "P2P", 16, False, 0, "Registration fee")


if "sender" not in st.session_state:
    demo_safe()

# ---------- page ----------
st.title("🛡️ UPI Fraud Detection")
st.caption("Checks a payment with 3 tools: a machine learning model, a scam-message "
           "detector and a mule-account graph.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Check a payment", "Transaction feed", "Mule network", "Model report",
     "Feedback & retrain"])

# ================= TAB 1 =================
with tab1:
    st.write("**Quick demo cases:**")
    c = st.columns(5)
    c[0].button("🟢 Safe payment", on_click=demo_safe)
    c[1].button("Fake collect request", on_click=demo_collect)
    c[2].button("Phishing link", on_click=demo_phishing)
    c[3].button("Account takeover", on_click=demo_takeover)
    c[4].button("Mule account", on_click=demo_mule)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Payment details")
        sender = st.selectbox("Sender", senders, key="sender")
        receiver = st.text_input("Receiver ID", key="receiver")
        amount = st.number_input(
            "Amount (Rs)", min_value=1, step=100, key="amount")
        txn_type = st.selectbox("Payment type", list(
            TYPE_CODES.keys()), key="txn_type")
        hour = st.slider("Hour of the day", 0, 23, key="hour")
        new_device = st.checkbox("Paid from a NEW device", key="new_device")
        recent = st.number_input("Payments by this sender in the last 2 minutes",
                                 min_value=0, step=1, key="recent")
        message = st.text_input(
            "Message / note / link (optional)", key="message")
        go = st.button("Check this payment", type="primary")

    with right:
        st.subheader("Result")
        if go:
            # keep the result in session_state, so it stays on screen after a button click
            st.session_state["last"] = {
                "res": score_payment(sender, receiver, amount, txn_type, hour,
                                     new_device, recent, message),
                "sender": sender, "receiver": receiver, "amount": amount,
                "message": message, "feedback_done": False}

        last = st.session_state.get("last")
        if last is not None:
            res = last["res"]
            amount = last["amount"]
            message = last["message"]
            st.caption("Result for: %s -> %s, Rs %d" %
                       (last["sender"], last["receiver"], amount))
            text = "%s  %s RISK  |  %s" % (
                res["emoji"], res["label"], res["advice"])
            if res["label"] == "GREEN":
                st.success(text)
            elif res["label"] == "YELLOW":
                st.warning(text)
            else:
                st.error(text)

            m = st.columns(2)
            m[0].metric("Final risk score", "%.2f" % res["final"])
            m[1].metric("Amount vs usual", "%.1fx" % (amount / res["avg"]))
            st.write("**Fraud type:** " + res["type"])

            st.write("**Why:**")
            if len(res["reasons"]) == 0:
                st.write("- Nothing suspicious found")
            for r in res["reasons"]:
                st.write("- " + r)

            st.write("**The 3 checks:**")
            st.write("- Main model score: %.2f" % res["model_prob"])
            st.write("- Message check: %s (%.2f)" % (res["msg_label"], res["msg_prob"])
                     if message.strip() != "" else "- Message check: no message")
            st.write("- Mule check: %s" %
                     ("MULE ACCOUNT FOUND" if res["mule"] else "clear"))

            # ----- feedback (Phase 10) -----
            st.write("**Was this decision right?**")
            if last["feedback_done"]:
                st.success(
                    "Thanks! Saved to feedback.csv. Retrain from the last tab.")
            else:
                fb = st.columns(2)
                fb[0].button("✅ Correct decision",
                             on_click=save_feedback, args=(True,))
                fb[1].button("❌ Wrong decision",
                             on_click=save_feedback, args=(False,))
        else:
            st.info(
                "Fill the form (or click a demo case) and press **Check this payment**.")

# ================= TAB 2 =================
with tab2:
    st.subheader("Transaction feed (sample)")
    if "seed" not in st.session_state:
        st.session_state["seed"] = 1
    st.button("Load a new sample",
              on_click=lambda: st.session_state.update(seed=st.session_state["seed"] + 1))
    seed = st.session_state["seed"]

    fraud_part = df[df["is_fraud"] == 1].sample(8, random_state=seed)
    normal_part = df[df["is_fraud"] == 0].sample(22, random_state=seed)
    sample = pd.concat([fraud_part, normal_part]).sample(
        frac=1, random_state=seed)

    probs = ctx["model"].predict_proba(sample[ctx["features"]])[:, 1]
    risk, score = [], []
    for p in probs:
        lab = rules.risk_label(p)
        risk.append(rules.LABEL_EMOJI[lab] + " " + lab)
        score.append(round(float(p), 2))

    feed = pd.DataFrame({
        "Txn": sample["txn_id"].values, "Sender": sample["sender_id"].values,
        "Receiver": sample["receiver_id"].values, "Amount": sample["amount"].values,
        "Type": sample["txn_type"].values, "Risk": risk, "Score": score,
        "Actual": ["fraud" if x == 1 else "normal" for x in sample["is_fraud"]]})

    k = st.columns(3)
    k[0].metric("🟢 Green", sum(1 for x in risk if "GREEN" in x))
    k[1].metric("🟡 Yellow", sum(1 for x in risk if "YELLOW" in x))
    k[2].metric("🔴 Red", sum(1 for x in risk if "RED" in x))
    st.dataframe(feed, hide_index=True)
    st.caption(
        "Sample has 8 fraud and 22 normal payments. 'Actual' is the true answer from the data.")

# ================= TAB 3 =================
with tab3:
    st.subheader("Mule account network")
    mules = ctx["mules"]
    st.write("Accounts flagged as mules: **%d**" % len(mules))
    if len(mules) > 0:
        chosen = st.selectbox("Pick a mule account", sorted(mules.keys()))
        info = mules[chosen]
        a = st.columns(4)
        a[0].metric("Paid by (people)", info["senders"])
        a[1].metric("Money in (Rs)", int(info["total_in"]))
        a[2].metric("Passed on", "%.0f%%" % (100 * info["pass_on"]))
        a[3].metric("Delay (min)", "%.0f" % info["delay_min"])
        mule_detector.draw_mule(load_graph(), chosen, info, "mule_view.png")
        st.image("mule_view.png", width=700)
        st.caption(
            "Blue = people who paid, red = mule, orange = cash-out accounts.")

# ================= TAB 4 =================
with tab4:
    st.subheader("Model report (from MLflow)")
    try:
        runs = mlflow.search_runs(experiment_names=["upi_fraud_detection"])
        # newest run first, so keep only the latest run for each name
        runs = runs.drop_duplicates(subset="tags.mlflow.runName")
        runs = runs[runs["metrics.precision"].notna()]
        table = pd.DataFrame({
            "Run": runs["tags.mlflow.runName"].values,
            "Precision": runs["metrics.precision"].round(3).values,
            "Recall": runs["metrics.recall"].round(3).values,
            "F1": runs["metrics.f1"].round(3).values})
        st.dataframe(table, hide_index=True)
        st.caption("Fraud models: RandomForest, XGBoost, LogisticRegression. "
                   "Scam models: Scam_TFIDF_*. Mule_detection_graph = account-level result.")
    except Exception as e:
        st.info("No MLflow runs found yet. Run train_model.py first.")
    if os.path.exists("shap_summary.png"):
        st.write("**Which features matter most (SHAP):**")
        st.image("shap_summary.png", width=700)

# ================= TAB 5 =================
with tab5:
    st.subheader("Feedback and retraining")
    st.write("Every time you press **Correct** or **Wrong** on a result, it is saved. "
             "Retraining adds those payments to the training data, so the model learns from them.")

    if os.path.exists(FEEDBACK_FILE):
        fb_df = pd.read_csv(FEEDBACK_FILE)
    else:
        fb_df = pd.DataFrame()
    st.metric("Feedback collected", len(fb_df))
    if len(fb_df) > 0:
        show = fb_df[["time", "sender", "receiver",
                      "amount", "system_said", "is_fraud"]].tail(10)
        show = show.rename(columns={"is_fraud": "true label (1 = fraud)"})
        st.dataframe(show, hide_index=True)

    if st.button("🔁 Retrain model with feedback", type="primary"):
        with st.spinner("Retraining... (about 20 seconds)"):
            st.session_state["retrain_result"] = retrain.retrain()
        st.cache_resource.clear()      # so the dashboard loads the new model next time

    r = st.session_state.get("retrain_result")
    if r is not None:
        if not r["ok"]:
            st.info(r["message"])
        else:
            if r["accepted"]:
                st.success(
                    "New model accepted and saved. The old model was backed up.")
            else:
                st.warning(
                    "New model rejected: its test F1 dropped too much. Old model kept.")
            table = pd.DataFrame({
                "Before": [r["old"]["precision"], r["old"]["recall"], r["old"]["f1"]],
                "After": [r["new"]["precision"], r["new"]["recall"], r["new"]["f1"]]},
                index=["Precision", "Recall", "F1"]).round(3)
            st.dataframe(table)
            st.write("Feedback payments handled correctly: **%d -> %d** of %d"
                     % (r["old_fb_right"], r["new_fb_right"], r["feedback_rows"]))
            st.caption("Both models are tested on the same untouched test set. "
                       "The run is also logged in MLflow as 'Retrain_with_feedback'.")
