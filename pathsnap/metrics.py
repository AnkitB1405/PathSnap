"""Evaluation metrics used across all experiments."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1_pick(df, score):
    """Per GPS point, the candidate with the highest score -> DataFrame(trip, pt, arc, y)."""
    d = df[["trip", "pt", "arc", "y"]].copy()
    d["s"] = score
    i = d.groupby(["trip", "pt"])["s"].idxmax()
    return d.loc[i].sort_values(["trip", "pt"])


def top1_accuracy(pick, n_points):
    """Correct points / all GPS points (points with no candidates count as wrong)."""
    return pick["y"].sum() / n_points


def continuity(pred_arcs, arcs):
    """Share of consecutive predictions that are the same arc or physically connected."""
    p = np.asarray(pred_arcs)
    if len(p) < 2:
        return 1.0
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return ok.mean()


def route_continuity(pick, arcs_by_trip):
    vals = [continuity(g["arc"].values, arcs_by_trip[t]) for t, g in pick.groupby("trip")]
    return float(np.mean(vals))


def classification(y, prob, thr=0.5):
    yp = (prob >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yp, average="macro"),
        "precision": precision_score(y, yp, average="macro", zero_division=0),
        "recall": recall_score(y, yp, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


def route_overlap(pred_arcs, ref_route):
    """Sequence-level: precision / recall / F1 of the set of predicted arcs vs reference route arcs."""
    p, r = set(pred_arcs), set(ref_route)
    if not p or not r:
        return 0.0, 0.0, 0.0
    tp = len(p & r)
    pr, rc = tp / len(p), tp / len(r)
    return pr, rc, (2 * pr * rc / (pr + rc) if tp else 0.0)
