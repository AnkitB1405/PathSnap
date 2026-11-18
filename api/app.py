"""PathSnap API (runs on Laptop 1).
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Test sets the UI can choose from:
  student_test    <PATHSNAP_DATA>/student_test           (default PATHSNAP_DATA = ../Student folder)
  evaluator_test  <PATHSNAP_DATA>/../evaluator_test
Override with PATHSNAP_SETS="name=/path/to/folder;name2=/path/to/folder2".
Only folders that exist are offered."""
import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.data import DATA_DIR, load_trip, prepare, trip_ids
from pathsnap.pipeline import load_final, predict_trip
from pathsnap.candidates import align_route
from pathsnap.metrics import continuity, route_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pathsnap")


def find_sets():
    env = os.environ.get("PATHSNAP_SETS")
    if env:
        pairs = [p.split("=", 1) for p in env.split(";") if "=" in p]
    else:
        pairs = [("student_test", os.path.join(DATA_DIR, "student_test")),
                 ("evaluator_test", os.path.join(DATA_DIR, "..", "evaluator_test"))]
    # an absolute folder works as a "partition" because os.path.join drops DATA_DIR before it
    return {n.strip(): os.path.abspath(p.strip()) for n, p in pairs if os.path.isdir(p.strip())}


SETS = find_sets()
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL, PARAMS = load_final()
log.info("model loaded: %s | hmm params %s", type(MODEL).__name__, PARAMS)
for n, p in SETS.items():
    log.info("test set %s: %d trips (%s)", n, len(trip_ids(p)), p)


class Req(BaseModel):
    trip_id: str
    set: str | None = None


def set_dir(name):
    name = name or next(iter(SETS))
    if name not in SETS:
        raise HTTPException(404, f"unknown test set {name}")
    return name, SETS[name]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sets")
def sets():
    return {"sets": [{"name": n, "n_trips": len(trip_ids(p))} for n, p in SETS.items()]}


@app.get("/trips")
def trips(set: str | None = None):
    name, d = set_dir(set)
    ids = trip_ids(d)
    pts = {t: sum(1 for _ in open(os.path.join(d, t, t + ".track"))) for t in ids}
    return {"set": name, "partition": name, "trips": ids, "points": pts}


def evaluate(d, trip, pred):
    """Server-side scoring against the reference route. The route never leaves this function."""
    ref = load_trip(d, trip["id"])["route"]
    ref_arc, _ = align_route(dict(trip, route=ref))
    seq = [a for a in pred if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    pr, rc, f1 = route_overlap(seq, ref)
    return {"point_accuracy": float((pred == ref_arc).mean()),
            "route_continuity": float(continuity(pred[pred >= 0], trip["arcs"])),
            "route_precision": pr, "route_recall": rc, "route_f1": f1}


def route_lines(trip, pred):
    nodes, arcs = trip["nodes"], trip["arcs"]
    seq = [int(a) for i, a in enumerate(pred) if a >= 0 and (i == 0 or a != pred[i - 1])]
    return [[[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]], [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]]
            for a in seq]


@app.post("/predict")
def predict(req: Req, request: Request):
    name, d = set_dir(req.set)
    if req.trip_id not in trip_ids(d):
        raise HTTPException(404, f"unknown trip {req.trip_id} in {name}")
    t0 = time.time()
    trip = prepare(load_trip(d, req.trip_id, with_route=False))   # model sees GPS + road network only
    out = predict_trip(trip, MODEL, PARAMS)
    t_pred = time.time() - t0
    m_hmm, m_clf = evaluate(d, trip, out["hmm"]), evaluate(d, trip, out["clf"])
    resp = {
        "set": name, "trip_id": req.trip_id, "n_points": len(trip["track"]), "chain_breaks": out["breaks"],
        "inference_time_s": round(t_pred, 3),
        "metrics": {"classifier_hmm": m_hmm, "classifier_only": m_clf},
        "gps": trip["track"][:, [1, 0]].round(6).tolist(),
        "predicted_route": route_lines(trip, out["hmm"]),
        "classifier_route": route_lines(trip, out["clf"]),
    }
    log.info("PREDICT from %s set=%s trip=%s points=%d time=%.2fs accuracy(hmm)=%.4f accuracy(clf)=%.4f "
             "continuity=%.4f route_f1=%.4f", request.client.host if request.client else "?", name, req.trip_id,
             resp["n_points"], t_pred, m_hmm["point_accuracy"], m_clf["point_accuracy"],
             m_hmm["route_continuity"], m_hmm["route_f1"])
    return resp
