# anomaly_detection.py
# Unsupervised transaction anomaly detection using Isolation Forest

import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/features.csv"
MODEL_PATH = "models/isolation_forest.pkl"

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


def train_model():
    # Load transaction dataset
    df = pd.read_csv(DATA_PATH)

    # Validate required columns
    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing feature columns in dataset: {missing}"
        )

    # Do not include txn_id, sender_id, receiver_id,
    # or is_fraud as model input features.
    X = df[FEATURES].copy()

    # Convert features to numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove rows where every feature is missing
    X = X.dropna(how="all")

    if X.empty:
        raise ValueError("No usable transaction records found.")

    # Build unsupervised anomaly detection pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=100,
                contamination=0.04,
                random_state=42
            )
        )
    ])

    # Train without using fraud labels
    model.fit(X)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Isolation Forest training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Transactions used for training: {len(X)}")


def detect_anomalies():
    # Load trained model
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # Predict anomaly status
    predictions = model.predict(X)

    # Lower decision scores indicate more anomalous records.
    scores = model.decision_function(X)

    results = df.copy()

    results["anomaly_score"] = scores
    results["anomaly_status"] = [
        "ANOMALY" if prediction == -1 else "NORMAL"
        for prediction in predictions
    ]

    # Save results
    os.makedirs("outputs", exist_ok=True)

    results.to_csv(
        "outputs/anomaly_results.csv",
        index=False
    )

    print("\nAnomaly Detection Results")
    print("-" * 55)

    print(
        results[
            ["txn_id", "amount", "anomaly_score", "anomaly_status"]
        ].head(15).to_string(index=False)
    )

    print("\nStatus summary:")
    print(results["anomaly_status"].value_counts())

    print("\nFull results saved to outputs/anomaly_results.csv")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()

    detect_anomalies()
