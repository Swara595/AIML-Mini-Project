# explain.py - Phase 5: explain WHY a transaction was flagged (SHAP)
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # save plots to a file, don't open a window

# type_code -> name (same order pandas used in features.py)
TYPE_NAMES = {0: "collect request", 1: "merchant payment",
              2: "P2P transfer", 3: "QR payment"}


def load_model():
    model = joblib.load("models/fraud_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, features, explainer


def get_shap_for_fraud(explainer, X):
    """Return SHAP values for the 'fraud' class as a 2D array (rows x features)."""
    values = explainer.shap_values(X)
    if isinstance(values, list):          # older shap: list of 2 arrays
        return values[1]
    if len(values.shape) == 3:            # newer shap: (rows, features, classes)
        return values[:, :, 1]
    return values


def make_sentence(feature, row):
    """Turn one feature into a plain-English reason."""
    if feature == "amount_vs_avg":
        return "amount is %.1fx higher than this sender's usual" % row["amount_vs_avg"]
    if feature == "amount":
        return "large amount (Rs %d)" % row["amount"]
    if feature == "is_new_payee":
        return "receiver is new for this sender"
    if feature == "is_new_device":
        return "payment made from a new device"
    if feature == "txn_last_2min":
        return "%d other payments in the last 2 minutes" % row["txn_last_2min"]
    if feature == "has_url":
        return "message contains a suspicious link"
    if feature == "hour":
        return "sent at an unusual hour (%d:00)" % row["hour"]
    if feature == "type_code":
        return "risky transaction type (%s)" % TYPE_NAMES.get(int(row["type_code"]), "unknown")
    return feature


def explain_transaction(model, features, explainer, row, top_n=3):
    """row = one transaction (a pandas row with all feature columns).
    Returns (fraud probability, list of reason sentences)."""
    X = pd.DataFrame([row[features]])
    prob = model.predict_proba(X)[0][1]
    shap_row = get_shap_for_fraud(explainer, X)[0]

    # collect (feature, shap value) pairs that push TOWARDS fraud
    pairs = []
    for i in range(len(features)):
        if shap_row[i] > 0:
            pairs.append((features[i], shap_row[i]))

    # sort biggest push first (simple bubble-style sort)
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            if pairs[b][1] > pairs[a][1]:
                pairs[a], pairs[b] = pairs[b], pairs[a]

    reasons = []
    for p in pairs[:top_n]:
        reasons.append(make_sentence(p[0], row))
    return prob, reasons


# ---------- demo: runs only when you run `python explain.py` ----------
if __name__ == "__main__":
    model, features, explainer = load_model()
    df = pd.read_csv("data/features.csv")

    # 1. Show reasons for 3 fraud and 2 normal transactions
    fraud_rows = df[df["is_fraud"] == 1].sample(3, random_state=1)
    normal_rows = df[df["is_fraud"] == 0].sample(2, random_state=1)
    demo = pd.concat([fraud_rows, normal_rows])

    for i in range(len(demo)):
        row = demo.iloc[i]
        prob, reasons = explain_transaction(model, features, explainer, row)
        print("\nTransaction", row["txn_id"], "| actual:", row["fraud_type"])
        print("  Fraud probability: %.2f" % prob)
        if len(reasons) == 0:
            print("  Reasons: nothing suspicious")
        for r in reasons:
            print("  -", r)

    # 2. Summary plot: which features matter most overall
    sample = df.sample(2000, random_state=42)[features]
    shap_vals = get_shap_for_fraud(explainer, sample)
    plt.figure()
    shap.summary_plot(shap_vals, sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120)
    plt.close()
    print("\nSaved shap_summary.png")

    # 3. Log the plot to MLflow so it sits next to the model runs
    mlflow.set_experiment("upi_fraud_detection")
    with mlflow.start_run(run_name="SHAP_explanation"):
        mlflow.log_artifact("shap_summary.png")
    print("Logged shap_summary.png to MLflow")
