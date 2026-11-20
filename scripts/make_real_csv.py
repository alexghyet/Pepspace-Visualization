"""Export the legacy solubility dataset as a combined metadata/vector CSV."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "embeddings/sol"


def main():
    embeddings = np.load(DATA_DIRECTORY / "peptides_embeddings_sol_v0.npy")
    meta = pd.read_csv(DATA_DIRECTORY / "peptides_metadata_sol_v0.csv")

    meta = meta.rename(columns={
        "peptide_id": "id",
        "source": "label"   # this becomes the color groups
    })

    # Create embedding column names
    embed_cols = [f"e{i}" for i in range(embeddings.shape[1])]

    embed_df = pd.DataFrame(embeddings, columns=embed_cols)

    final_df = pd.concat([meta, embed_df], axis=1)

    final_df.to_csv(DATA_DIRECTORY / "peptide_umap_sol_ready.csv", index=False)

    print("Saved embeddings/sol/peptide_umap_sol_ready.csv")


if __name__ == "__main__":
    main()
