# automl_baseline.py - OPTIONAL: let an AutoML tool build a model, then compare with ours
# Needs:  pip install "flaml[automl]"
# (FLAML is a light AutoML library that installs easily on Windows.
#  AutoGluon / auto-sklearn are heavier and auto-sklearn does not run on Windows.)
import pandas as pd
import joblib
import mlflow
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

TIME_BUDGET = 60       # seconds the AutoML tool may search for the best model

features = joblib.load("models/feature_names.pkl")
df = pd.read_csv("data/features.csv")
X = df[features]
y = df["is_fraud"]

# same split as train_model.py, so the comparison is fair
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

automl = AutoML()
automl.fit(X_train, y_train, task="classification", metric="f1",
           time_budget=TIME_BUDGET, estimator_list=["lgbm", "rf", "xgboost"],
           seed=42, verbose=0)

pred = automl.predict(X_test)
p = precision_score(y_test, pred)
r = recall_score(y_test, pred)
f = f1_score(y_test, pred)
print("AutoML picked:", automl.best_estimator)
print("AutoML    -> precision %.3f  recall %.3f  F1 %.3f" % (p, r, f))

# our own best model, on the same test set
ours = joblib.load("models/fraud_model.pkl")
pred2 = ours.predict(X_test)
print("Our model -> precision %.3f  recall %.3f  F1 %.3f"
      % (precision_score(y_test, pred2), recall_score(y_test, pred2), f1_score(y_test, pred2)))

mlflow.set_experiment("upi_fraud_detection")
with mlflow.start_run(run_name="AutoML_FLAML"):
    mlflow.log_param("model", automl.best_estimator)
    mlflow.log_param("time_budget_sec", TIME_BUDGET)
    mlflow.log_metric("precision", p)
    mlflow.log_metric("recall", r)
    mlflow.log_metric("f1", f)
print("Logged to MLflow")
