import os
import json
import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------- CONFIGURATION ----------------

DATA_PATH = "data/features.csv"
MODEL_PATH = "models/lightgbm_model.pkl"
METRICS_PATH = "outputs/lightgbm_metrics.json"

FEATURES = [
    "amount",
    "hour",
    "amount_vs_avg",
    "txn_last_2min",
    "is_new_payee",
    "is_new_device",
    "has_url",
    "type_code"
]

TARGET = "is_fraud"


# ---------------- LOAD DATA ----------------

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df = df.dropna(subset=[TARGET])

# Do not silently treat invalid labels as normal transactions.
if not df[TARGET].isin([0, 1]).all():
    raise ValueError(
        "is_fraud must contain only 0 and 1."
    )

df = df.dropna(subset=FEATURES)

X = df[FEATURES]
y = df[TARGET].astype(int)

if y.nunique() != 2:
    raise ValueError(
        "Training requires both normal and fraud examples."
    )

print("Dataset loaded:", len(df))
print("Normal transactions:", int((y == 0).sum()))
print("Fraud transactions:", int((y == 1).sum()))


# ---------------- TRAIN-TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------- TRAIN LIGHTGBM ----------------

model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nTraining LightGBM...")

model.fit(X_train, y_train)


# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# Probability of class 1: fraud
y_prob = model.predict_proba(X_test)[:, 1]


# ---------------- EVALUATION ----------------

metrics = {
    "model": "LightGBM",
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, y_prob)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist()
}

print("\n----- LIGHTGBM RESULTS -----")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


# ---------------- SAVE MODEL AND METRICS ----------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)


# ---------------- SAVE TEST PREDICTIONS ----------------

results = X_test.copy()

results["actual_fraud"] = y_test
results["predicted_fraud"] = y_pred
results["fraud_probability"] = y_prob

results.to_csv(
    "outputs/lightgbm_predictions.csv",
    index=False
)

print("Test predictions saved.")
