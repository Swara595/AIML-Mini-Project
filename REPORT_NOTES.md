# Report and viva notes (UPI Fraud Detection)

## 1. Report outline (matches the Mini Project / Mini-Project Lab components)
| Report section | Where it comes from |
|---|---|
| Problem domain study | what UPI fraud is: fake collect requests, QR scams, phishing links, account takeover, mule accounts |
| Literature and dataset survey | section 2 below, plus why we used synthetic data |
| Data preprocessing and feature engineering | `features.py` (feature table: what each means and why it helps) |
| Algorithm / model study | Random Forest vs XGBoost vs Logistic Regression; TF-IDF; graph rules |
| Tools and libraries | pandas, scikit-learn, XGBoost, SHAP, NetworkX, MLflow, Streamlit |
| Model development and experimentation | `train_model.py`, MLflow runs, `automl_baseline.py` |
| Performance evaluation and analysis | precision / recall / F1 / confusion matrix; before-after retraining |
| Documentation and presentation | README, MLflow screenshots, dashboard demo |

## 2. Literature survey (opened and checked; read them before you cite)
1. Lopez-Rojas, Elmir, Axelsson. *PaySim: A financial mobile money simulator for fraud detection.* EMSS 2016.
   https://msc-les.org/proceedings/emss/emss2016/emss2016_249.html
   Real financial data is private, so they generate synthetic transactions from a real sample. This is our reason for using synthetic data.
2. Sirisha, Jaheda, Tahaseen, Madhuri, Arshad. *Unified Payments Interface Fraud Detection Using Machine Learning.* ICRDICCT'25, SciTePress, 2025.
   https://scitepress.org/PublishedPapers/2025/139308
   Uses XGBoost, Decision Tree, Random Forest and Gradient Boosting on UPI fraud. The abstract gives no numbers.
3. Dahiphale et al. (Google). *Enhancing Trust and Safety in Digital Payments: An LLM-Powered Approach.* arXiv:2410.19845, Oct 2024.
   https://arxiv.org/abs/2410.19845
   Uses LLMs to classify UPI transactions (Google Pay) as scam or not and to give reasons for reviewers. Reported 93.33% scam classification accuracy with Gemini Ultra.
   Link to our project: it also gives *reasons*, like our SHAP explanations.

Add 1 or 2 more of your own after reading (search "mule account detection graph" and "SMOTE fraud detection").

## 3. Self-study topics: where each is used
| Topic from the self-study list | In our project |
|---|---|
| Model deployment (Streamlit) | `app.py` dashboard |
| Version control (Git, GitHub) | a commit after every phase |
| Experiment tracking (MLflow) | every model, scam model, mule run, retrain run |
| Data structures (graphs, sparse matrices) | NetworkX payment graph; sparse TF-IDF and adjacency matrices |
| AutoML | `automl_baseline.py` (FLAML) |
| MLOps (Docker) | `Dockerfile`; **DVC not done** (optional) |
| Computer vision, search space management | not applicable |

## 4. Demo script (about 3 minutes)
1. Run `streamlit run app.py`. Click **Safe payment**, then **Check**: 🟢.
2. Click **Fake collect request**: 🔴, with reasons (amount 12x usual, new payee, scam message).
3. Click **Mule account**: 🔴 and the mule graph in the **Mule network** tab.
4. Press **Wrong decision** on one result, open **Feedback & retrain**, press retrain, show Before/After.
5. Show the MLflow page with all runs side by side.

## 5. Likely viva questions
- **Why F1, not accuracy?** Only about 4% of payments are fraud. A model that always says "not fraud" gets 96% accuracy and catches nothing.
- **Why synthetic data?** Real UPI data is private (PaySim paper makes the same point).
- **What does SHAP do?** Splits a prediction among the features, so each decision has reasons.
- **What is a mule account?** An account that gets money from many people and quickly passes it on.
- **Why a graph?** Mules are visible only in the pattern of who pays whom, not in one payment.
- **Why is a sparse matrix used?** Only about 3% of the TF-IDF cells have values, so it uses far less memory.
- **How does feedback help?** Marked payments are added to the training data with extra weight. The new model is kept only if it does not get worse on the same test set.
- **Weakness?** Synthetic data, so scores are optimistic; fixed fraud-type rules; no human review of feedback.
- **AutoML beat our model. Why keep Random Forest?** Be honest: AutoML (LightGBM) got F1 0.923 vs 0.882. Random Forest was our first hand-built model and is easy to explain. Switching to LightGBM is the next improvement (SHAP works with it too).