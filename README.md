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
