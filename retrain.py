# retrain.py - Phase 10: retrain the fraud model using the user's feedback
# Called by the dashboard's "Retrain" button, or run by hand:  python retrain.py
import os
import glob
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

FEEDBACK_FILE = "feedback.csv"
FEEDBACK_WEIGHT = 10      # one feedback row counts as 10 normal rows
TOLERANCE = 0.01          # accept the new model if its F1 drops by less than this


def evaluate(model, X, y):
    pred = model.predict(X)
    return {"precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def retrain():
    if not os.path.exists(FEEDBACK_FILE):
        return {"ok": False, "message": "No feedback yet. Use the Correct / Wrong buttons first."}
    fb = pd.read_csv(FEEDBACK_FILE)
    if len(fb) == 0:
        return {"ok": False, "message": "feedback.csv is empty."}

    features = joblib.load("models/feature_names.pkl")
    df = pd.read_csv("data/features.csv")
    X = df[features]
    y = df["is_fraud"]

    # SAME test split as train_model.py, so old and new models face the same exam
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    old_model = joblib.load("models/fraud_model.pkl")
    old = evaluate(old_model, X_test, y_test)
    old_fb_right = int(
        (old_model.predict(fb[features]) == fb["is_fraud"]).sum())

    # training data = original training rows + feedback rows (with extra weight)
    X_new = pd.concat([X_train, fb[features]])
    y_new = pd.concat([y_train, fb["is_fraud"]])
    weights = [1] * len(X_train) + [FEEDBACK_WEIGHT] * len(fb)

    new_model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=42)
    new_model.fit(X_new, y_new, sample_weight=weights)
    new = evaluate(new_model, X_test, y_test)
    new_fb_right = int(
        (new_model.predict(fb[features]) == fb["is_fraud"]).sum())

    accepted = new["f1"] >= old["f1"] - TOLERANCE
    if accepted:
        n = len(glob.glob("models/fraud_model_before_retrain_*.pkl")) + 1
        joblib.dump(
            old_model, "models/fraud_model_before_retrain_%d.pkl" % n)   # backup
        joblib.dump(new_model, "models/fraud_model.pkl")

    mlflow.set_experiment("upi_fraud_detection")
    with mlflow.start_run(run_name="Retrain_with_feedback"):
        mlflow.log_param("feedback_rows", len(fb))
        mlflow.log_param("feedback_weight", FEEDBACK_WEIGHT)
        mlflow.log_param("new_model_accepted", accepted)
        mlflow.log_metric("old_f1", old["f1"])
        mlflow.log_metric("precision", new["precision"])
        mlflow.log_metric("recall", new["recall"])
        mlflow.log_metric("f1", new["f1"])

    return {"ok": True, "accepted": accepted, "feedback_rows": len(fb),
            "old": old, "new": new,
            "old_fb_right": old_fb_right, "new_fb_right": new_fb_right}


if __name__ == "__main__":
    r = retrain()
    if not r["ok"]:
        print(r["message"])
    else:
        print("Feedback rows used:", r["feedback_rows"])
        print("              Before   After")
        for k in ["precision", "recall", "f1"]:
            print("%-12s  %.3f    %.3f" % (k, r["old"][k], r["new"][k]))
        print("Feedback payments handled correctly: %d -> %d of %d"
              % (r["old_fb_right"], r["new_fb_right"], r["feedback_rows"]))
        if r["accepted"]:
            print("New model ACCEPTED and saved (old one backed up).")
        else:
            print("New model REJECTED (test F1 dropped too much). Old model kept.")
