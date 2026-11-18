"""PathSnap API (runs on Laptop 1).
    uvicorn api.app:app --host 0.0.0.0 --port 8000
Env: PATHSNAP_DATA (folder containing the partition folders), PATHSNAP_EVAL (partition with the 10 eval trips)."""
import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.data import load_trip, prepare, trip_ids
from pathsnap.pipeline import load_final, predict_trip
from pathsnap.candidates import align_route
from pathsnap.metrics import continuity, route_overlap

EVAL = os.environ.get("PATHSNAP_EVAL", "student_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL, PARAMS = load_final()
log.info("model loaded: %s | hmm params %s | eval partition %s (%d trips)",
         type(MODEL).__name__, PARAMS, EVAL, len(trip_ids(EVAL)))


class Req(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return {"partition": EVAL, "trips": trip_ids(EVAL)}


def evaluate(trip, pred):
    """Server-side scoring against the reference route. The route never leaves this function."""
    ref = load_trip(EVAL, trip["id"])["route"]
    gt = dict(trip, route=ref)
    ref_arc, _ = align_route(gt)
    seq = [a for a in pred if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    pr, rc, f1 = route_overlap(seq, ref)
    return {"point_accuracy": float((pred == ref_arc).mean()),
            "route_continuity": float(continuity(pred[pred >= 0], trip["arcs"])),
            "route_precision": pr, "route_recall": rc, "route_f1": f1}


@app.post("/predict")
def predict(req: Req, request: Request):
    if req.trip_id not in trip_ids(EVAL):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    trip = prepare(load_trip(EVAL, req.trip_id, with_route=False))   # model sees GPS + road network only
    out = predict_trip(trip, MODEL, PARAMS)
    t_pred = time.time() - t0
    m_hmm, m_clf = evaluate(trip, out["hmm"]), evaluate(trip, out["clf"])
    nodes, arcs = trip["nodes"], trip["arcs"]
    seq = [int(a) for i, a in enumerate(out["hmm"]) if a >= 0 and (i == 0 or a != out["hmm"][i - 1])]
    resp = {
        "trip_id": req.trip_id, "n_points": len(trip["track"]), "chain_breaks": out["breaks"],
        "inference_time_s": round(t_pred, 3),
        "metrics": {"classifier_hmm": m_hmm, "classifier_only": m_clf},
        "gps": trip["track"][:, [1, 0]].round(6).tolist(),
        "predicted_route": [[[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]],
                             [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]] for a in seq],
    }
    log.info("PREDICT from %s trip=%s points=%d time=%.2fs accuracy(hmm)=%.4f accuracy(clf)=%.4f continuity=%.4f route_f1=%.4f",
             request.client.host if request.client else "?", req.trip_id, resp["n_points"], t_pred,
             m_hmm["point_accuracy"], m_clf["point_accuracy"], m_hmm["route_continuity"], m_hmm["route_f1"])
    return resp
