"""Logistic Regression baseline (Ankit)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    run("logistic_regression", "Linear", "StandardScaler + L2, C=1.0, class_weight=None",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)))
