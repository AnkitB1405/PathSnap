"""Shared train / cross-validate / test routine so every model is evaluated the same way."""
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import load_trip
from .features import FEATURES
from .metrics import top1_pick, top1_accuracy, route_continuity, classification

SEED = 42
BAD_TRIP_OFFSET = 50.0  # trips whose GPS sits > 50 m (median) from the reference route are dropped


def load_split(part):
    df = pd.read_pickle(f"results/features_{part}.pkl")
    st = pd.read_csv("results/candidate_stats.csv", dtype={"trip": str})
    st = st[st.partition == part]
    st["trip"] = st["trip"].str.zfill(8)
    if part == "train":
        bad = st.loc[st.ref_dist_median > BAD_TRIP_OFFSET, "trip"]
        df = df[~df.trip.isin(bad)]
        st = st[~st.trip.isin(bad)]
    return df.reset_index(drop=True), st


def arcs_for(part, trips):
    return {t: load_trip(part, t, with_route=False)["arcs"] for t in trips}


def evaluate(df, prob, n_points, arcs):
    pick = top1_pick(df, prob)
    m = classification(df["y"].values, prob)
    m["top1"] = top1_accuracy(pick, n_points)
    m["continuity"] = route_continuity(pick, arcs)
    return m


def run(name, family, config, make_model, fit_rows=None, folds=5):
    """make_model() -> unfitted sklearn-style estimator. fit_rows: optional row cap for slow models."""
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    tr, st_tr = load_split("train")
    te, st_te = load_split("student_test")
    rng = np.random.RandomState(SEED)

    def fit(model, d):
        if fit_rows and len(d) > fit_rows:
            d = d.iloc[rng.choice(len(d), fit_rows, replace=False)]
        t0 = time.time()
        model.fit(d[FEATURES].values, d["y"].values)
        return time.time() - t0

    # group-aware CV: whole trips stay in one fold
    gkf = GroupKFold(n_splits=folds)
    pts = st_tr.set_index("trip")["n_points"]
    arcs_tr = arcs_for("train", st_tr.trip)
    cv = []
    for k, (a, b) in enumerate(gkf.split(tr, groups=tr["trip"])):
        m = make_model()
        fit(m, tr.iloc[a])
        va = tr.iloc[b]
        p = m.predict_proba(va[FEATURES].values)[:, 1]
        r = evaluate(va, p, pts[va.trip.unique()].sum(), arcs_tr)
        r["fold"] = k
        cv.append(r)
        print(f"{name} fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f}")
    cv = pd.DataFrame(cv)

    model = make_model()
    train_time = fit(model, tr)
    t0 = time.time()
    p_te = model.predict_proba(te[FEATURES].values)[:, 1]
    infer = time.time() - t0
    res = evaluate(te, p_te, st_te.n_points.sum(), arcs_for("student_test", st_te.trip))
    p_tr = model.predict_proba(tr[FEATURES].values)[:, 1]
    res_train = evaluate(tr, p_tr, st_tr.n_points.sum(), arcs_tr)

    out = {"model": name, "family": family, "config": config,
           "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
           "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
           "train_top1": res_train["top1"], "train_macro_f1": res_train["macro_f1"],
           "test_top1": res["top1"], "test_macro_f1": res["macro_f1"], "test_precision": res["precision"],
           "test_recall": res["recall"], "test_log_loss": res["log_loss"], "test_continuity": res["continuity"],
           "train_time_s": train_time, "infer_ms_per_point": 1000 * infer / st_te.n_points.sum(),
           "cv_folds": cv.to_dict("records")}
    json.dump(out, open(f"results/models/{name}.json", "w"), indent=1, default=float)
    pd.DataFrame({"trip": te.trip, "pt": te.pt, "arc": te.arc, "y": te.y, "p": p_te}).to_pickle(
        f"results/models/{name}_test_probs.pkl")
    joblib.dump(model, f"models/{name}.joblib", compress=3)
    print(json.dumps({k: v for k, v in out.items() if k != "cv_folds"}, indent=1, default=float))
    return model, out
