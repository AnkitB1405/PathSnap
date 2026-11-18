"""Distance-weighted KNN (Aaron). Trained on a 150k-row sample because prediction cost grows with training size."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    run("knn", "Instance-based", "k=25, distance weights, scaled, 150k-row sample",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        fit_rows=150_000)
