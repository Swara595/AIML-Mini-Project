# train_model.py - Phase 4: train 3 models, track them in MLflow, save the best
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ---------- load data ----------
df = pd.read_csv("data/features.csv")

features = ["amount", "hour", "amount_vs_avg", "txn_last_2min",
            "is_new_payee", "is_new_device", "has_url", "type_code"]
X = df[features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# how many normal rows per fraud row (used by XGBoost to handle imbalance)
normal_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()
ratio = normal_count / fraud_count

# ---------- the 3 models we compare ----------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100, scale_pos_weight=ratio, random_state=42,
        eval_metric="logloss"),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)),
}

# ---------- train, test, log ----------
mlflow.set_experiment("upi_fraud_detection")

best_name = ""
best_f1 = 0
best_model = None

for name in models:
    model = models[name]
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        print("\n=====", name, "=====")
        print("Precision:", round(precision, 3))
        print("Recall:   ", round(recall, 3))
        print("F1:       ", round(f1, 3))
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(confusion_matrix(y_test, pred))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

# ---------- save the best ----------
print("\nBest model:", best_name, "with F1 =", round(best_f1, 3))
joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(features, "models/feature_names.pkl")
print("Saved models/fraud_model.pkl")
