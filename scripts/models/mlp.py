"""Shallow MLP neural network (Ananya). Saves epoch-wise training loss and validation score."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathsnap.experiment import run, SEED
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

make = lambda: make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
    max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=SEED))

if __name__ == "__main__":
    model, _ = run("mlp", "Neural network", "64-32 ReLU, Adam lr=1e-3, early stopping (10% val)", make)
    mlp = model[-1]
    json.dump({"train_loss": mlp.loss_curve_, "val_accuracy": mlp.validation_scores_},
              open("results/models/mlp_history.json", "w"))
