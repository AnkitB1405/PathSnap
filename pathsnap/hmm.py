"""HMM + Viterbi route reconstruction.

Hidden state at GPS point i = one of its candidate arcs.
Emission  = classifier probability of that candidate (renormalised over the point's candidates),
            or a Gaussian on GPS-to-arc distance for the distance-only baseline.
Transition = exp(-|network distance - straight-line GPS distance| / beta)   (Newson & Krumm style).
Network distance comes from Dijkstra on the trip's road graph (arcs are directed).
"""
import heapq
import numpy as np

DEFAULTS = {"beta": 10.0, "w_emit": 1.0, "sigma": 8.0, "max_gap_s": 60.0}


class RoadGraph:
    def __init__(self, trip):
        arcs = trip["arcs"]
        self.frm, self.to, self.len = arcs[:, 0], arcs[:, 1], trip["alen"]
        order = np.argsort(self.frm, kind="stable")
        self.adj_to = self.to[order]
        self.adj_len = self.len[order]
        self.ptr = np.searchsorted(self.frm[order], np.arange(len(trip["nodes"]) + 1))
        self.cache = {}

    def dists_from(self, src, cutoff):
        """Shortest road distance from node src to every node within cutoff metres."""
        c = self.cache.get(src)
        if c and c[0] >= cutoff:
            return c[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for k in range(self.ptr[u], self.ptr[u + 1]):
                v, nd = self.adj_to[k], d + self.adj_len[k]
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self.cache[src] = (cutoff, dist)
        return dist


def transition_logp(g, ids0, t0, ids1, t1, gc, beta):
    """K0 x K1 matrix of log transition probabilities between candidate arcs."""
    T = np.full((len(ids0), len(ids1)), -np.inf)
    cutoff = 2 * gc + 200.0
    for i, (a, ta) in enumerate(zip(ids0, t0)):
        La = g.len[a]
        dist = None
        for j, (b, tb) in enumerate(zip(ids1, t1)):
            if a == b:
                rd = abs(tb - ta) * La
            else:
                if dist is None:
                    dist = g.dists_from(g.to[a], cutoff)
                sp = dist.get(g.frm[b])
                if sp is None:
                    continue
                rd = (1 - ta) * La + sp + tb * g.len[b]
            T[i, j] = -abs(rd - gc) / beta
    return T


def viterbi(trip, cands, emit_scores=None, params=None, graph=None):
    """cands: list of (ids, dists, ts) per point. emit_scores: list of arrays aligned with ids
    (classifier probabilities) or None for distance-only emissions.
    Returns predicted arc per point (-1 when the point has no candidates) and number of chain breaks."""
    p = {**DEFAULTS, **(params or {})}
    g = graph or RoadGraph(trip)
    P, T = trip["gps"], trip["t"]
    n = len(cands)
    deltas, backs = [None] * n, [None] * n
    breaks = 0
    prev = None
    for i, (ids, d, t) in enumerate(cands):
        if len(ids) == 0:
            prev = None
            continue
        if emit_scores is None:
            e = -0.5 * (d / p["sigma"]) ** 2
        else:
            s = np.asarray(emit_scores[i], float) + 1e-6
            e = p["w_emit"] * np.log(s / s.sum())
        if prev is not None and T[i] - T[prev] <= p["max_gap_s"]:
            pids, _, pt = cands[prev]
            gc = float(np.hypot(*(P[i] - P[prev])))
            tr = transition_logp(g, pids, pt, ids, t, gc, p["beta"])
            score = deltas[prev][:, None] + tr
            best = score.argmax(0)
            new = score[best, np.arange(len(ids))] + e
            if np.isfinite(new).any():
                deltas[i], backs[i] = new, (prev, best)
                prev = i
                continue
        if prev is not None:
            breaks += 1  # gap, missing candidates or no connected path: start a new chain
        deltas[i], backs[i] = e, None
        prev = i

    pred = np.full(n, -1, dtype=np.int64)
    i = n - 1
    j = None
    while i >= 0:
        if deltas[i] is None:
            i -= 1
            j = None
            continue
        if j is None:
            j = int(np.argmax(deltas[i]))
        pred[i] = cands[i][0][j]
        if backs[i] is None:
            i, j = i - 1, None
        else:
            pi, best = backs[i]
            j = int(best[j])
            i = pi
    return pred, breaks
