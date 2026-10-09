"""run_all.py - runs the whole pipeline in the right order.

Usage:  python run_all.py
Then:   streamlit run app.py
"""
import os
import subprocess
import sys

# a fresh clone has no models/ folder yet
os.makedirs("models", exist_ok=True)

steps = [
    ("features.py",      "Phase 3: build features"),
    ("train_model.py",   "Phase 4: train and compare fraud models (MLflow)"),
    ("explain.py",       "Phase 5: SHAP explanations"),
    ("rules.py",         "Phase 6: risk labels and fraud types"),
    ("scam_detector.py", "Phase 7: scam message detector"),
    ("mule_detector.py", "Phase 8: mule account detection"),
]

for script, title in steps:
    print("\n" + "=" * 60)
    print(title, "->", script)
    print("=" * 60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print("\nStopped: %s failed. Fix that error and run again." % script)
        sys.exit(1)

print("\nAll steps finished. Now run:  streamlit run app.py")
