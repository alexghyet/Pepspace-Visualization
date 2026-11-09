import numpy as np
import umap.umap_ as umap
import optuna
from sklearn.manifold import trustworthiness
import json
import os
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import validate_embeddings
from utils.projection_settings import DEFAULT_METRIC, DEFAULT_SEED

# Paths
EMB_PATH = PROJECT_ROOT / "embeddings/veltri/veltri_embeddings.npy"
RESULTS_DIR = PROJECT_ROOT / "optuna_results"

# Dimension to optimize — change this before each run
N_COMPONENTS = 3

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load embeddings 
print("Loading embeddings...")
X = validate_embeddings(np.load(EMB_PATH, allow_pickle=False))
print("Embedding shape:", X.shape)


# Optuna objective function 
def objective(trial):

    # Choose method
    method = trial.suggest_categorical("method", ["UMAP", "densMAP"])

    # Shared parameters
    max_neighbors = min(50, len(X) - 1)
    n_neighbors = trial.suggest_int("n_neighbors", min(5, max_neighbors), max_neighbors)
    min_dist    = trial.suggest_float("min_dist", 0.0, 0.5)

    reducer_kwargs = dict(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=N_COMPONENTS,
        metric=DEFAULT_METRIC,
        random_state=DEFAULT_SEED,
    )
    if len(X) <= N_COMPONENTS + 1:
        reducer_kwargs["init"] = "random"

    # densMAP-only parameter
    if method == "densMAP":
        dens_lambda = trial.suggest_float("dens_lambda", 0.1, 0.3)
        reducer_kwargs.update(
            densmap=True,
            dens_lambda=dens_lambda,
        )

    reducer = umap.UMAP(**reducer_kwargs)
    X_low   = reducer.fit_transform(X)
    quality_neighbors = min(5, (len(X) - 1) // 2)
    score   = trustworthiness(X, X_low, n_neighbors=quality_neighbors)

    return score


def progress_callback(study, trial):
    print(f"  Trial {trial.number + 1:>3}/75 | score={trial.value:.4f} | best={study.best_value:.4f} | {trial.params}", flush=True)

# Run optimizer
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"\nStarting Optuna optimization for {N_COMPONENTS}D...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=DEFAULT_SEED))
study.optimize(objective, n_trials=75, callbacks=[progress_callback])

print("\nBest parameters:")
print(study.best_params)
print("\nBest trustworthiness score:")
print(study.best_value)


# Write individual per-dimension file (read by app.py as optuna_results/{dim}d_params.json)
best_params = {**study.best_params, "method": f"{study.best_params['method']} {N_COMPONENTS}d"}

dim_file = os.path.join(RESULTS_DIR, f"{N_COMPONENTS}d_params.json")
with open(dim_file, "w") as f:
    json.dump({
        "best_params": best_params,
        "best_score":  study.best_value,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=4)

print(f"\nSaved {N_COMPONENTS}D results to: {dim_file}")
