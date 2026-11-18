"""End-to-end prediction for one trip: candidates -> classifier -> HMM/Viterbi."""
import json
import os
import joblib
import numpy as np
from .features import trip_features, FEATURES
from .hmm import viterbi, RoadGraph, DEFAULTS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_final():
    model = joblib.load(os.path.join(MODEL_DIR, "final_model.joblib"))
    cfg = json.load(open(os.path.join(MODEL_DIR, "hmm_config.json")))
    return model, {**DEFAULTS, **cfg}


def split_scores(prob, cands):
    out, k = [], 0
    for ids, _, _ in cands:
        out.append(prob[k:k + len(ids)])
        k += len(ids)
    return out


def predict_trip(trip, model, params, labels=False):
    """trip must be prepared (pathsnap.data.prepare). Returns dict with per-point predictions."""
    df, info, cands = trip_features(trip, labels=labels)
    prob = model.predict_proba(df[FEATURES].values)[:, 1]
    scores = split_scores(prob, cands)
    clf = np.array([ids[np.argmax(s)] if len(ids) else -1 for (ids, _, _), s in zip(cands, scores)])
    hmm, breaks = viterbi(trip, cands, scores, params, RoadGraph(trip))
    return {"clf": clf, "hmm": hmm, "breaks": breaks, "cands": cands, "scores": scores, "info": info, "df": df}
