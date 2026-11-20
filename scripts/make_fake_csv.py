"""Generate an optional synthetic dataset for visualization experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def main():
    output_directory = Path(__file__).resolve().parents[1] / "embeddings/fake"
    output_directory.mkdir(parents=True, exist_ok=True)

    existing = list(output_directory.glob("fake_embeddings*.npy"))
    idx = len(existing) + 1

    X, y = make_blobs(
        n_samples=800,
        centers=4,
        n_features=32,
        cluster_std=0.8,
        random_state=None
    )

    is_amp = y >= 2

    np.save(output_directory / f"fake_embeddings{idx}.npy", X.astype("float32"))

    ids = [f"pep_{i:04d}" for i in range(X.shape[0])]
    seqs = ["".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 12)) for _ in range(X.shape[0])]

    meta = pd.DataFrame({
        "id": ids,
        "sequence": seqs,
        "is_amp": is_amp
    })

    meta.to_csv(output_directory / f"fake_metadata{idx}.csv", index=False)

    print(f"Saved dataset #{idx}")


if __name__ == "__main__":
    main()
