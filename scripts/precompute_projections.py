"""Warm the app's exact projection cache before opening the visualization."""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.embedding_runtime import get_trustworthiness, get_umap_projection
from utils.dataset import validate_embeddings
from utils.projection_settings import load_optuna_parameters, projection_arguments, projection_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, choices=range(3, 9), default=3)
    parser.add_argument("--embeddings", type=Path, default=PROJECT_ROOT / "embeddings/veltri/veltri_embeddings.npy")
    parser.add_argument(
        "--params", type=Path, default=PROJECT_ROOT / "optuna_results/3d_params.json",
        help="Optimization settings file; defaults to the app's shared 3D baseline for every display dimension.",
    )
    parser.add_argument("--quality", action="store_true", help="Also cache exact 2D and nD trustworthiness scores.")
    args = parser.parse_args()
    try:
        X = validate_embeddings(np.load(args.embeddings, allow_pickle=False))
        best = load_optuna_parameters(args.params)
        parameters = projection_arguments(projection_defaults(len(X), best))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Preparing {len(X):,} rows for 2D and {args.dimensions}D using shared settings from {args.params.name}.", flush=True)
    for dimensions in (2, args.dimensions):
        coordinates, fit, elapsed, cached = get_umap_projection(X, n_components=dimensions, **parameters)
        print(f"{dimensions}D: {'cache hit' if cached else 'fit saved'}, {elapsed:.3f}s (original fit {fit:.2f}s)", flush=True)
        if args.quality:
            neighbors = min(parameters["n_neighbors"], (len(X) - 1) // 2)
            score = get_trustworthiness(X, coordinates, neighbors)
            print(f"{dimensions}D exact trustworthiness: {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
