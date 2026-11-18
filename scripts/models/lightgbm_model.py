"""LightGBM gradient boosting (Gaganaraj). Also saves the train/validation log loss per boosting round."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
import json
import lightgbm as lgb
from pathsnap.experiment import load_split
from pathsnap.features import FEATURES

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=1.0, colsample_bytree=1.0, random_state=SEED, verbose=-1)

if __name__ == "__main__":
    run("lightgbm", "Boosting", "400 rounds, lr=0.05, 31 leaves (defaults otherwise)",
        lambda: lgb.LGBMClassifier(**PARAMS))
    # training history: fit on train trips, watch student_test log loss per round
    tr, _ = load_split("train"); te, _ = load_split("student_test")
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[FEATURES], tr.y, eval_set=[(tr[FEATURES], tr.y), (te[FEATURES], te.y)],
          eval_names=["train", "student_test"], eval_metric="binary_logloss")
    json.dump({k: v["binary_logloss"] for k, v in m.evals_result_.items()},
              open("results/models/lightgbm_history.json", "w"))
