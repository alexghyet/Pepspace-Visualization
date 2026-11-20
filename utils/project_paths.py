"""Default dataset locations, relative to the project root."""

EMBEDDING_DIRECTORY = "embeddings/veltri_peptidebert_sol_v2"
EMBEDDINGS_PATH = f"{EMBEDDING_DIRECTORY}/veltri_embeddings.npy"
METADATA_PATH = f"{EMBEDDING_DIRECTORY}/veltri_metadata.csv"
MANIFEST_PATH = f"{EMBEDDING_DIRECTORY}/manifest.json"
CHECKPOINT_PATH = ".cache/peptidebert/v2.0/sol/model.pt"
VELTRI_PATH = "data/veltri/all_veltri.csv"
