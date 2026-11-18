"""HMM / Viterbi experiments (Gaganaraj).
1. Tune decoder parameters (beta, emission weight) on 20 train trips using out-of-fold classifier probabilities.
2. Compare on student_test: nearest arc, classifier only, distance-only HMM, classifier + HMM."""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import joblib
from pathsnap.data import load_trip, prepare, trip_ids
from pathsnap.features import trip_features, FEATURES
from pathsnap.hmm import viterbi, RoadGraph, DEFAULTS
from pathsnap.pipeline import split_scores
from pathsnap.metrics import continuity, route_overlap
from pathsnap.experiment import SEED

oof = pd.read_pickle("results/oof_train_probs.pkl")
model = joblib.load("models/final_model.joblib")


def compress(seq):
    s = [a for a in seq if a >= 0]
    return [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]]


def scores_for(trip, part):
    df, info, cands = trip_features(trip)
    if part == "train":
        prob = oof[oof.trip == trip["id"]]["p"].values
        assert len(prob) == len(df)
    else:
        prob = model.predict_proba(df[FEATURES].values)[:, 1]
    return info, cands, split_scores(prob, cands)


def methods(trip, info, cands, scores, params, g):
    out = {"nearest": np.array([c[0][0] if len(c[0]) else -1 for c in cands]),
           "classifier": np.array([c[0][np.argmax(s)] if len(c[0]) else -1 for c, s in zip(cands, scores)])}
    t0 = time.time()
    out["hmm_distance"], _ = viterbi(trip, cands, None, params, g)
    out["classifier_hmm"], br = viterbi(trip, cands, scores, params, g)
    return out, br, time.time() - t0


if __name__ == "__main__":
    rng = np.random.RandomState(SEED)
    trips = sorted(oof.trip.unique())
    tune_trips = sorted(rng.choice(trips, 20, replace=False))
    cache = {}
    for t in tune_trips:
        tr = prepare(load_trip("train", t))
        cache[t] = (tr, *scores_for(tr, "train"), RoadGraph(tr))
    grid = []
    for beta, w in itertools.product([1, 2, 5, 10, 20, 50], [0.5, 1.0, 2.0, 4.0]):
        p = {**DEFAULTS, "beta": beta, "w_emit": w}
        ok = tot = 0
        for t, (tr, info, cands, scores, g) in cache.items():
            pred, _ = viterbi(tr, cands, scores, p, g)
            ok += (pred == info["ref_arc"]).sum(); tot += len(pred)
        grid.append({"beta": beta, "w_emit": w, "train_top1": ok / tot})
        print(grid[-1])
    grid = pd.DataFrame(grid)
    grid.to_csv("results/hmm_grid.csv", index=False)
    best = grid.loc[grid.train_top1.idxmax()]
    params = {**DEFAULTS, "beta": float(best.beta), "w_emit": float(best.w_emit)}
    json.dump(params, open("models/hmm_config.json", "w"), indent=1)

    # same grid for the distance-only HMM sigma (fair baseline)
    sg = []
    for sigma in [2, 3, 4, 6, 8, 12, 20]:
        ok = tot = 0
        for t, (tr, info, cands, scores, g) in cache.items():
            pred, _ = viterbi(tr, cands, None, {**params, "sigma": sigma}, g)
            ok += (pred == info["ref_arc"]).sum(); tot += len(pred)
        sg.append({"sigma": sigma, "train_top1": ok / tot})
    sg = pd.DataFrame(sg); sg.to_csv("results/hmm_sigma_grid.csv", index=False)
    params["sigma"] = float(sg.loc[sg.train_top1.idxmax(), "sigma"])
    json.dump(params, open("models/hmm_config.json", "w"), indent=1)
    print("decoder params", params)

    rows = []
    for t in trip_ids("student_test"):
        tr = prepare(load_trip("student_test", t))
        info, cands, scores = scores_for(tr, "student_test")
        g = RoadGraph(tr)
        out, br, hmm_time = methods(tr, info, cands, scores, params, g)
        for name, pred in out.items():
            pr, rc, f1 = route_overlap(compress(pred), tr["route"])
            rows.append({"trip": t, "method": name, "n_points": len(pred),
                         "top1": (pred == info["ref_arc"]).mean(),
                         "continuity": continuity(pred[pred >= 0], tr["arcs"]),
                         "route_precision": pr, "route_recall": rc, "route_f1": f1,
                         "breaks": br if "hmm" in name else 0, "hmm_time_s": hmm_time})
    res = pd.DataFrame(rows)
    res.to_csv("results/hmm_comparison_by_trip.csv", index=False)
    w = res.assign(c=res.top1 * res.n_points)
    summ = w.groupby("method").agg(points=("n_points", "sum"), correct=("c", "sum"),
                                   continuity=("continuity", "mean"), route_f1=("route_f1", "mean"))
    summ["top1"] = summ.correct / summ.points
    summ = summ.drop(columns="correct").loc[["nearest", "classifier", "hmm_distance", "classifier_hmm"]]
    summ.to_csv("results/hmm_comparison.csv")
    print(summ.round(4).to_string())
    print(res.pivot(index="trip", columns="method", values="top1").round(3).to_string())
