# UPI Fraud Detection

AIML mini project: checks a UPI payment with three tools and explains its decision.

1. **Fraud model** (Random Forest) gives a risk score, with SHAP reasons.
2. **Scam message detector** (TF-IDF + Logistic Regression + link rules).
3. **Mule account detector** (payment graph built with NetworkX).

The final risk is the highest of the three: 🟢 under 0.3, 🟡 0.3 to 0.7, 🔴 over 0.7.
The user can mark a decision Correct or Wrong, and the model retrains on that feedback.

## Setup
```
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Put `upi_transactions.csv` and `sms_scam_messages.csv` in `data/`.

## Run
```
python run_all.py              builds features, trains everything, saves models/
streamlit run app.py           opens the dashboard
mlflow ui                      compare all experiment runs at http://127.0.0.1:5000
```

## Files
| File | What it does |
|---|---|
| `features.py` | builds features (amount vs usual, speed check, new payee, link in message) |
| `train_model.py` | trains Random Forest, XGBoost, Logistic Regression; logs to MLflow |
| `explain.py` | SHAP reasons for each decision |
| `rules.py` | 🟢🟡🔴 labels and fraud type names |
| `scam_detector.py` | scam SMS / link detector |
| `mule_detector.py` | finds mule accounts in the payment graph |
| `app.py` | Streamlit dashboard (5 tabs) |
| `retrain.py` | retrains the model with user feedback |
| `automl_baseline.py` | optional AutoML comparison (FLAML) |
| `Dockerfile` | optional container for the dashboard |

## Results (synthetic data, 20% unseen test set)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest (used) | 0.875 | 0.889 | 0.882 |
| XGBoost | 0.821 | 0.918 | 0.867 |
| Logistic Regression | 0.449 | 0.923 | 0.604 |
| AutoML (FLAML, LightGBM) | 0.963 | 0.887 | 0.923 |

XGBoost numbers can change by a little from run to run, so yours may differ slightly.
Scam detector and mule detector both score 1.0 on the synthetic data (see limits below).

## Limits 
- The data is **synthetic**, because real UPI data is private. Perfect scores for the scam
  and mule detectors come from the generated patterns and will be lower on real data.
- Fraud type naming uses fixed rules written for this dataset.
- Feedback retraining has no human review, so false feedback could hurt the model.
  A check keeps the old model if the new one scores clearly worse.
