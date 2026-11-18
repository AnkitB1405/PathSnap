"""Gaussian Naive Bayes (Ankit)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    run("gaussian_nb", "Probabilistic", "default var_smoothing=1e-9", lambda: GaussianNB())
