"""Random Forest (Aaron)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    run("random_forest", "Ensemble (bagging)", "200 trees, max_depth=14, min_samples_leaf=20",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEED))
