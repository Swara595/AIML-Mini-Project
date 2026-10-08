# scam_detector.py - Phase 7: detect scam SMS / payment messages
# TF-IDF (turns text into numbers) + Logistic Regression, plus simple URL rules
import re
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# ---------- simple URL checks ----------
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "goo.gl", "t.co"]
BAD_ENDINGS = [".xyz", ".top", ".info", ".click", ".buzz"]


def check_url(text):
    """Return a list of warning sentences about links found in the message."""
    warnings = []
    words = text.split()
    for w in words:
        w_low = w.lower()
        looks_like_link = ("http" in w_low or "www." in w_low or "/" in w_low
                           and "." in w_low)
        for s in SHORTENERS:
            if s in w_low:
                warnings.append("shortened link (%s)" % s)
        for e in BAD_ENDINGS:
            if e in w_low:
                warnings.append("suspicious link ending (%s)" % e)
        if "@" in w_low and looks_like_link:
            warnings.append("'@' inside a link (hides the real site)")
        if looks_like_link and len(w_low) > 50:
            warnings.append("very long link")
    return warnings


# ---------- main function used by the dashboard (Phase 9) ----------
def check_message(text, model):
    """Returns (scam probability, label, list of reasons)."""
    prob = model.predict_proba([text])[0][1]
    reasons = []

    url_warnings = check_url(text)
    for w in url_warnings:
        reasons.append(w)

    low = text.lower()
    for word in ["urgent", "blocked", "kyc", "refund", "won", "claim", "verify", "suspended"]:
        if word in low:
            reasons.append("scam keyword: '%s'" % word)

    # a bad link alone is enough to push the score up
    if len(url_warnings) > 0 and prob < 0.5:
        prob = 0.5 + prob / 2

    if prob >= 0.5:
        label = "SCAM"
    else:
        label = "SAFE"
    return prob, label, reasons


# ---------- training + evaluation: runs only with `python scam_detector.py` ----------
if __name__ == "__main__":
    from sklearn.pipeline import make_pipeline

    df = pd.read_csv("data/sms_scam_messages.csv")
    y = (df["label"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], y, test_size=0.2, random_state=42, stratify=y)

    # ----- show the sparse matrix (PDF topic: Sparse Matrices) -----
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    M = vec.fit_transform(X_train)
    total_cells = M.shape[0] * M.shape[1]
    filled = M.nnz
    sparse_bytes = M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    dense_bytes = total_cells * 8
    print("TF-IDF matrix type:", type(M).__name__)
    print("Shape (messages x words):", M.shape)
    print("Non-zero cells: %d of %d (%.2f%% filled)" %
          (filled, total_cells, 100.0 * filled / total_cells))
    print("Memory sparse: %.2f MB  |  as a normal (dense) array: %.2f MB"
          % (sparse_bytes / 1e6, dense_bytes / 1e6))

    # ----- compare 2 models -----
    candidates = {
        "TFIDF_LogisticRegression": make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "TFIDF_NaiveBayes": make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2),
            MultinomialNB()),
    }

    mlflow.set_experiment("upi_fraud_detection")
    best_name = ""
    best_f1 = -1
    best_model = None
    for name in candidates:
        model = candidates[name]
        with mlflow.start_run(run_name="Scam_" + name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            p = precision_score(y_test, pred)
            r = recall_score(y_test, pred)
            f = f1_score(y_test, pred)
            mlflow.log_param("model", name)
            mlflow.log_metric("precision", p)
            mlflow.log_metric("recall", r)
            mlflow.log_metric("f1", f)
            mlflow.log_metric("matrix_filled_percent",
                              100.0 * filled / total_cells)
            print("\n=====", name, "=====")
            print("Precision: %.3f  Recall: %.3f  F1: %.3f" % (p, r, f))
            print(confusion_matrix(y_test, pred))
            if f > best_f1:
                best_f1 = f
                best_name = name
                best_model = model

    print("\nBest scam model:", best_name, "F1 = %.3f" % best_f1)
    joblib.dump(best_model, "models/scam_model.pkl")
    print("Saved models/scam_model.pkl")

    # ----- test on hand-written messages the model has never seen -----
    print("\n--- Hand-written tests ---")
    tests = [
        "Dear user, your SBI account is suspended. Verify now at sbi-secure@verify.info/login",
        "Congratulations!! You are selected for Rs 25000 cashback. Click bit.ly/getmoney",
        "Hi, I sent Rs 500 for the movie. Please check.",
        "Your Swiggy order is out for delivery. Pay Rs 340 on arrival.",
        "Your Amazon refund is pending. Approve the request to receive it today",
    ]
    for t in tests:
        prob, label, reasons = check_message(t, best_model)
        print("\n", label, "(%.2f)" % prob, "|", t)
        for r in reasons:
            print("    -", r)
