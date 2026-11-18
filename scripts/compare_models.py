"""Collects every model's results into one table and draws the comparison plots (Aaron)."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)
ORDER = ["logistic_regression", "gaussian_nb", "knn", "random_forest", "lightgbm", "mlp", "lightgbm_tuned"]

res = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("results/models/*.json")
       if not f.endswith("_history.json")}
names = [n for n in ORDER if n in res]
tab = pd.DataFrame([{k: v for k, v in res[n].items() if k != "cv_folds"} for n in names])
tab.to_csv("results/model_comparison.csv", index=False)
cols = ["model", "family", "cv_macro_f1", "cv_macro_f1_std", "train_top1", "test_top1", "test_macro_f1",
        "test_precision", "test_recall", "test_log_loss", "test_continuity", "train_time_s", "infer_ms_per_point"]
print(tab[cols].round(4).to_string(index=False))

base = [n for n in names if n != "lightgbm_tuned"]
b = tab.set_index("model").loc[base]
x = np.arange(len(base))

plt.figure(figsize=(9, 4))
plt.bar(x, b.cv_macro_f1, yerr=b.cv_macro_f1_std, capsize=4, color="tab:blue")
plt.xticks(x, base, rotation=20); plt.ylim(0.8, 1); plt.ylabel("macro F1")
plt.title("5-fold group CV macro F1 (train trips)")
plt.tight_layout(); plt.savefig(f"{OUT}/cv_macro_f1.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
plt.bar(x - 0.2, b.test_top1, 0.4, label="Top-1 accuracy")
plt.bar(x + 0.2, b.test_macro_f1, 0.4, label="macro F1")
plt.xticks(x, base, rotation=20); plt.ylim(0.8, 1); plt.legend(); plt.title("student_test results")
plt.tight_layout(); plt.savefig(f"{OUT}/test_top1_f1.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
plt.bar(x, b.test_log_loss, color="tab:red")
plt.xticks(x, base, rotation=20); plt.ylabel("log loss"); plt.title("student_test log loss (lower is better)")
plt.tight_layout(); plt.savefig(f"{OUT}/log_loss.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
plt.bar(x - 0.2, b.train_top1, 0.4, label="train (fit data)")
plt.bar(x + 0.2, b.test_top1, 0.4, label="student_test")
plt.xticks(x, base, rotation=20); plt.ylim(0.8, 1.0); plt.legend(); plt.title("Top-1 accuracy: train vs student_test")
plt.tight_layout(); plt.savefig(f"{OUT}/train_vs_test_top1.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
for n in base:
    plt.plot([f["fold"] for f in res[n]["cv_folds"]], [f["top1"] for f in res[n]["cv_folds"]], "o-", label=n)
plt.xlabel("fold"); plt.ylabel("Top-1 accuracy"); plt.legend(fontsize=8); plt.title("Top-1 accuracy per CV fold")
plt.tight_layout(); plt.savefig(f"{OUT}/cv_folds_top1.png", dpi=120); plt.close()

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, n in zip(axs.ravel(), names):
    p = pd.read_pickle(f"results/models/{n}_test_probs.pkl")
    ConfusionMatrixDisplay(confusion_matrix(p.y, p.p >= 0.5), display_labels=["other", "true arc"]).plot(
        ax=ax, colorbar=False, values_format="d")
    ax.set_title(n)
for ax in axs.ravel()[len(names):]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=110); plt.close()

if os.path.exists("results/models/lightgbm_history.json"):
    h = json.load(open("results/models/lightgbm_history.json"))
    plt.figure(figsize=(7, 4))
    for k, v in h.items():
        plt.plot(v, label=k)
    plt.xlabel("boosting round"); plt.ylabel("log loss"); plt.legend(); plt.title("LightGBM training history")
    plt.tight_layout(); plt.savefig(f"{OUT}/lightgbm_history.png", dpi=120); plt.close()

if os.path.exists("results/models/mlp_history.json"):
    h = json.load(open("results/models/mlp_history.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["train_loss"], label="train loss", c="tab:blue"); ax.set_xlabel("epoch"); ax.set_ylabel("log loss")
    ax2 = ax.twinx(); ax2.plot(h["val_accuracy"], label="validation accuracy", c="tab:orange")
    ax2.set_ylabel("validation accuracy")
    fig.legend(loc="center right"); plt.title("MLP training history")
    plt.tight_layout(); plt.savefig(f"{OUT}/mlp_history.png", dpi=120); plt.close()

if "lightgbm_tuned" in res:
    t = tab.set_index("model").loc[["lightgbm", "lightgbm_tuned"]]
    m = ["cv_macro_f1", "test_top1", "test_macro_f1"]
    xx = np.arange(len(m))
    plt.figure(figsize=(7, 4))
    plt.bar(xx - 0.2, t.loc["lightgbm", m], 0.4, label="untuned")
    plt.bar(xx + 0.2, t.loc["lightgbm_tuned", m], 0.4, label="tuned")
    plt.xticks(xx, m); plt.ylim(0.85, 1); plt.legend(); plt.title("LightGBM untuned vs tuned")
    plt.tight_layout(); plt.savefig(f"{OUT}/tuned_vs_untuned.png", dpi=120); plt.close()

if os.path.exists("results/optuna_trials.csv"):
    tr = pd.read_csv("results/optuna_trials.csv")
    plt.figure(figsize=(7, 4))
    plt.plot(tr.number, tr.value, "o", label="trial")
    plt.plot(tr.number, tr.value.cummax(), "-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.legend(); plt.title("Optuna TPE search")
    plt.tight_layout(); plt.savefig(f"{OUT}/optuna_trials.png", dpi=120); plt.close()
