# rules.py - Phase 6: risk labels (green/yellow/red) and fraud type naming
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split

# ---------- 1. risk label from the model's fraud probability ----------


def risk_label(prob):
    if prob < 0.3:
        return "GREEN"
    if prob < 0.7:
        return "YELLOW"
    return "RED"


LABEL_TEXT = {
    "GREEN": "Low risk - allow",
    "YELLOW": "Medium risk - ask user to confirm",
    "RED": "High risk - block and alert",
}
LABEL_EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


# ---------- 2. fraud type from simple rules ----------
def name_fraud_type(row, receiver_is_mule=False):
    """row = one transaction. Rules are checked from top to bottom."""
    message = str(row["message"]).lower()

    if row["txn_type"] == "COLLECT_REQUEST":
        return "Fake payment request"
    if row["has_url"] == 1:
        return "Phishing link"
    if row["txn_type"] == "QR_PAYMENT" and "scan" in message:
        return "QR scam"
    if row["is_new_device"] == 1 or row["txn_last_2min"] >= 2:
        return "Account takeover"
    if receiver_is_mule:                       # Phase 8 will fill this in
        return "Mule account"
    return "Suspicious (type unknown)"


def assess(row, prob, receiver_is_mule=False):
    """Combine everything: returns label, emoji, advice, fraud type."""
    label = risk_label(prob)
    if label == "GREEN":
        fraud_type = "None"
    else:
        fraud_type = name_fraud_type(row, receiver_is_mule)
    return label, LABEL_EMOJI[label], LABEL_TEXT[label], fraud_type


# ---------- demo + evaluation: runs only with `python rules.py` ----------
# true fraud_type (in the data) -> name our rules should give
EXPECTED = {
    "fake_collect_request": "Fake payment request",
    "phishing_link": "Phishing link",
    "qr_scam": "QR scam",
    "account_takeover_burst": "Account takeover",
    "mule_inbound": "Mule account",
    "mule_forward": "Mule account",
}

if __name__ == "__main__":
    model = joblib.load("models/fraud_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    df = pd.read_csv("data/features.csv")

    # same split as train_model.py, so we only judge rows the model never saw
    X = df[features]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    test = df.loc[X_test.index].copy()
    test["prob"] = model.predict_proba(X_test)[:, 1]

    labels = []
    for i in range(len(test)):
        labels.append(risk_label(test.iloc[i]["prob"]))
    test["label"] = labels

    # --- how are the labels spread? ---
    print("Risk labels for NORMAL transactions:")
    print(test[test["is_fraud"] == 0]["label"].value_counts())
    print("\nRisk labels for FRAUD transactions:")
    print(test[test["is_fraud"] == 1]["label"].value_counts())

    # --- are the fraud type names correct? (only for fraud rows) ---
    fraud_test = test[test["is_fraud"] == 1]
    right = 0
    stats = {}
    for i in range(len(fraud_test)):
        row = fraud_test.iloc[i]
        true_type = row["fraud_type"]
        # mule types need Phase 8, so we give the rule a fair chance by
        # pretending the mule check already ran
        is_mule = true_type in ["mule_inbound", "mule_forward"]
        guess = name_fraud_type(row, receiver_is_mule=is_mule)
        if true_type not in stats:
            stats[true_type] = [0, 0]
        stats[true_type][1] += 1
        if guess == EXPECTED[true_type]:
            stats[true_type][0] += 1
            right += 1

    print("\nFraud type naming (correct / total):")
    for t in stats:
        print("  %-24s %d / %d" % (t, stats[t][0], stats[t][1]))
    type_acc = right / len(fraud_test)
    print("Overall naming accuracy: %.3f" % type_acc)

    # --- show 4 example assessments ---
    print("\nExamples:")
    examples = pd.concat([fraud_test.sample(3, random_state=3),
                          test[test["is_fraud"] == 0].sample(1, random_state=3)])
    for i in range(len(examples)):
        row = examples.iloc[i]
        label, emoji, advice, ftype = assess(row, row["prob"])
        print(" ", row["txn_id"], emoji, label, "|", advice, "| type:", ftype)

    # --- log to MLflow ---
    fraud_red = (fraud_test["label"] == "RED").mean()
    fraud_green = (fraud_test["label"] == "GREEN").mean()
    normal_green = (test[test["is_fraud"] == 0]["label"] == "GREEN").mean()
    mlflow.set_experiment("upi_fraud_detection")
    with mlflow.start_run(run_name="Risk_labels_and_fraud_types"):
        mlflow.log_metric("fraud_type_accuracy", type_acc)
        mlflow.log_metric("fraud_rows_labelled_red", fraud_red)
        mlflow.log_metric("fraud_rows_labelled_green_missed", fraud_green)
        mlflow.log_metric("normal_rows_labelled_green", normal_green)
    print("\nLogged to MLflow")
