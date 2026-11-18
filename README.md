# PathSnap

Map-matching and route reconstruction for noisy GPS trajectories (UE24CS352A Machine Learning hackathon).

Problem statement and guidelines are in `docs/`.

## Team

- Ankit B (AnkitB1405)
- Aaron Immanuel (aaronimmanuel)
- Gaganaraj (gaganaraj)
- Ananya Dharani (Ananya-Dharani)

## Folder structure

```
pathsnap/        shared code: data loading, candidates, features, metrics, HMM
scripts/         EDA, dataset building, tuning, HMM experiments, maps
scripts/models/  one training script per model
api/             FastAPI backend (Laptop 1)
ui/              evaluator UI (Laptop 2)
models/          saved model and configs
results/         csv results, logs, plots
maps/            folium html maps
report/          technical report
docs/            problem statement pdfs
```

The dataset is not committed. Put the `train/` and `student_test/` folders in `Student folder/` next to this repo, or set `PATHSNAP_DATA` to their parent folder.

## Setup

Python 3.10+.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

Run from the repo root, in this order. Each step writes into `results/`, `models/` or `maps/`.

```
python scripts/eda.py                         # dataset inventory, checks and EDA plots
python scripts/build_dataset.py               # candidates + features for train and student_test
python scripts/models/logistic_regression.py  # each model: 5-fold group CV + student_test eval
python scripts/models/gaussian_nb.py
python scripts/models/knn.py
python scripts/models/random_forest.py
python scripts/models/lightgbm_model.py
python scripts/models/mlp.py
python scripts/tune_lightgbm.py               # Optuna TPE search, saves models/final_model.joblib
python scripts/hmm_eval.py                    # HMM decoder tuning + classifier vs classifier+HMM
python scripts/compare_models.py              # comparison table and plots
python scripts/make_maps.py eda               # folium maps
python scripts/make_maps.py results
```

## Deployment (two laptops)

Laptop 1 (model + API). Copy the repo, the trained `models/` folder and the evaluation trips. The eval trips go in a partition folder, for example `Student folder/eval_test/<trip_id>/<trip_id>.track|nodes|arcs|route`.

```
export PATHSNAP_DATA="/path/to/Student folder"   # Windows: set PATHSNAP_DATA=...
export PATHSNAP_EVAL=eval_test                    # default is student_test
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Find the Laptop 1 IP (`ipconfig` / `ip a`) and allow port 8000 through the firewall. Every request is printed in this terminal.

Laptop 2 (UI). Only the `ui/` folder is needed.

```
cd ui
python -m http.server 5500
```

Open `http://localhost:5500`, type `http://<laptop1-ip>:8000` in the API address box, press Load trips and click a trip.

API endpoints:

- `GET /health`
- `GET /trips` - the trip ids in the eval partition
- `POST /predict` with `{"trip_id": "00000040"}` - predicted route, GPS points and accuracy computed on the server

The reference route is only read inside the API for scoring. It is never sent to the model or the UI.
