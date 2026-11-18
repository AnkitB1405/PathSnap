"""Folium maps: raw GPS, road network, reference route and predicted routes (Ananya).
usage: python scripts/make_maps.py eda            -> maps/eda_train_00000097.html
       python scripts/make_maps.py results [ids]   -> maps/result_<id>.html for student_test trips"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import folium
from pathsnap.data import load_trip, prepare, trip_ids
from pathsnap.candidates import ArcIndex

os.makedirs("maps", exist_ok=True)

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #999;font:13px sans-serif">{}</div>"""


def arc_lines(trip, arc_ids):
    n = trip["nodes"]
    return [[(n[a, 1], n[a, 0]), (n[b, 1], n[b, 0])] for a, b in trip["arcs"][arc_ids]]


def base_map(trip):
    tr = trip["track"]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    near = ArcIndex(trip, margin=200).query(trip["gps"], 150, 50)
    net = np.unique(np.concatenate([c[0] for c in near]))
    folium.PolyLine(arc_lines(trip, net), color="#999999", weight=1.5, opacity=0.7).add_to(
        folium.FeatureGroup("Road network (within 150 m)").add_to(m))
    return m


def add_gps(m, trip, every=1):
    g = folium.FeatureGroup("GPS observations").add_to(m)
    for lon, lat, t in trip["track"][::every]:
        folium.CircleMarker((lat, lon), radius=2, color="#1f6feb", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(g)


def add_route(m, trip, arcs, name, color, weight):
    folium.PolyLine(arc_lines(trip, np.asarray(arcs)), color=color, weight=weight, opacity=0.85).add_to(
        folium.FeatureGroup(name).add_to(m))


def legend(m, items):
    html = "<br>".join(f'<span style="color:{c};font-size:18px">&#9644;</span> {n}' for n, c in items)
    m.get_root().html.add_child(folium.Element(LEGEND.format(html)))


def save_zoom(m, trip, i, path, half=40):
    """Also save a copy zoomed on GPS points i-half .. i+half."""
    tr = trip["track"][max(0, i - half): i + half]
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(path)


def eda_map(part="train", tid="00000097"):
    trip = prepare(load_trip(part, tid))
    m = base_map(trip)
    add_route(m, trip, trip["route"], "Reference route", "#2da44e", 6)
    add_gps(m, trip)
    folium.LayerControl(collapsed=False).add_to(m)
    legend(m, [("GPS observations", "#1f6feb"), ("Reference route", "#2da44e"), ("Road network", "#999999")])
    m.save(f"maps/eda_{part}_{tid}.html")
    save_zoom(m, trip, len(trip["track"]) // 2, f"maps/eda_{part}_{tid}_zoom.html")


def result_map(tid, model, params):
    from pathsnap.pipeline import predict_trip
    trip = prepare(load_trip("student_test", tid))
    out = predict_trip(trip, model, params)
    m = base_map(trip)
    add_route(m, trip, trip["route"], "Reference route", "#2da44e", 8)
    add_route(m, trip, np.unique(out["clf"][out["clf"] >= 0]), "Classifier only", "#fb8500", 4)
    add_route(m, trip, np.unique(out["hmm"][out["hmm"] >= 0]), "Classifier + HMM (predicted)", "#d62828", 3)
    add_gps(m, trip)
    folium.LayerControl(collapsed=False).add_to(m)
    legend(m, [("GPS observations", "#1f6feb"), ("Reference route", "#2da44e"),
               ("Classifier only", "#fb8500"), ("Classifier + HMM", "#d62828"), ("Road network", "#999999")])
    m.save(f"maps/result_{tid}.html")
    diff = np.where(out["clf"] != out["hmm"])[0]
    if len(diff):
        save_zoom(m, trip, int(diff[len(diff) // 2]), f"maps/result_{tid}_zoom.html", half=25)


if __name__ == "__main__":
    if sys.argv[1] == "eda":
        eda_map()
    else:
        from pathsnap.pipeline import load_final
        model, params = load_final()
        for t in (sys.argv[2:] or trip_ids("student_test")):
            result_map(t, model, params)
            print("saved", t)
