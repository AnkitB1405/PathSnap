"""Bayesian hyperparameter optimisation of LightGBM with Optuna TPE (Gaganaraj).
Objective: mean observation-level Top-1 accuracy over 3 group folds (whole trips per fold) on train only."""
import sys, os, json, time, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from pathsnap.experiment import load_split, run, SEED
from pathsnap.features import FEATURES
from pathsnap.metrics import top1_pick

N_TRIALS = 30
tr, st = load_split("train")
pts = st.set_index("trip")["n_points"]
folds = list(GroupKFold(n_splits=3).split(tr, groups=tr["trip"]))
X, y = tr[FEATURES].values, tr["y"].values


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=SEED, verbose=-1)
    accs = []
    for a, b in folds:
        m = lgb.LGBMClassifier(**params).fit(X[a], y[a])
        va = tr.iloc[b]
        pick = top1_pick(va, m.predict_proba(X[b])[:, 1])
        accs.append(pick.y.sum() / pts[va.trip.unique()].sum())
    return float(np.mean(accs))


if __name__ == "__main__":
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    cost = time.time() - t0
    study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": N_TRIALS,
               "search_time_s": cost}, open("results/optuna_best.json", "w"), indent=1)
    print("best", best, study.best_value, "time", cost)

    params = dict(best, subsample_freq=1, random_state=SEED, verbose=-1)
    run("lightgbm_tuned", "Boosting", "Optuna best: " + ", ".join(f"{k}={v:.3g}" for k, v in best.items()),
        lambda: lgb.LGBMClassifier(**params))
    shutil.copy("models/lightgbm_tuned.joblib", "models/final_model.joblib")

    # out-of-fold probabilities on train trips, used to tune the HMM decoder without leakage
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(n_splits=5).split(tr, groups=tr["trip"]):
        oof[b] = lgb.LGBMClassifier(**params).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    tr[["trip", "pt", "arc", "y"]].assign(p=oof).to_pickle("results/oof_train_probs.pkl")
