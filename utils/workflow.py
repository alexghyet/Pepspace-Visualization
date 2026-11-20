"""Dataset identity, provenance checks, and reproducible reference/query splits."""

from dataclasses import dataclass
import hashlib
import json

from joblib import hash as joblib_hash
import numpy as np
import pandas as pd

from utils.dataset import build_plot_dataframe, parse_amp_labels, pick_col
from utils.peptidebert import network_settings

SEQUENCE_COLUMNS = ["sequence", "aa_seq", "seq", "peptide", "peptide_sequence", "amp_sequence"]
ID_COLUMNS = ["peptide_id", "id", "identifier", "name", "entry", "accession", "sample_id", "sample_identifier", "peptide_name",
              "specimen_id", "specimen_no"]
ACTIVITY_COLUMNS = ["activity", "activity_type", "type_of_activity", "function", "label", "class"]
IDENTITY_FIELDS = ("model", "task", "checkpoint_sha256", "pooling", "encoding", "truncation")


def check_compatibility(reference, candidate, feature_count):
    """Matching dimensions alone cannot establish a common embedding space."""
    for name, manifest in (("reference", reference), ("uploaded", candidate)):
        if not isinstance(manifest, dict) or any(key not in manifest for key in IDENTITY_FIELDS):
            raise ValueError(f"The {name} manifest needs model, task, checkpoint_sha256, pooling, encoding, and truncation.")
        shape = manifest.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != feature_count:
            raise ValueError(f"The {name} manifest's embedding dimensions do not match the data.")
    mismatches = [field for field in IDENTITY_FIELDS if reference[field] != candidate[field]]
    if "config" in reference:
        try:
            if (network_settings(reference["config"]) != network_settings(candidate["config"])
                    or reference["config"]["vocab_size"] != candidate["config"]["vocab_size"]):
                mismatches.append("encoder configuration")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("The uploaded manifest needs the matching PeptideBERT encoder configuration.") from error
    if mismatches:
        raise ValueError(f"Embedding method differs from Veltri: {', '.join(mismatches)}. Use Explore a dataset for other encoders.")


def read_manifest(contents, embedding_bytes=None):
    manifest = json.loads(contents)
    if not isinstance(manifest, dict):
        raise ValueError("The embedding manifest must be a JSON object.")
    expected_hash = manifest.get("embeddings_sha256")
    if embedding_bytes is not None and expected_hash and hashlib.sha256(embedding_bytes).hexdigest() != expected_hash:
        raise ValueError("The embeddings do not match the SHA-256 hash in the manifest.")
    return manifest


def default_columns(meta):
    columns = dict(
        seq_col=pick_col(meta, SEQUENCE_COLUMNS), id_col=pick_col(meta, ID_COLUMNS),
        label_col=pick_col(meta, ["source", "label", "class", "family", "dataset", "type", "cohort"]),
        amp_col=pick_col(meta, ["is_amp", "AMP"]),
        activity_col=pick_col(meta, ACTIVITY_COLUMNS),
        structure_col=pick_col(meta, ["pdb_id", "structure_id", "pdb"]),
    )
    if columns["amp_col"] is None and columns["label_col"] is not None:
        labels = meta[columns["label_col"]]
        # A generic binary class can mean anything. Infer AMP only from an explicit AMP name.
        if labels.astype("string").str.strip().str.fullmatch(r"(?:non[\s_-]*)?amp", case=False).any():
            try:
                parse_amp_labels(labels)
                columns["amp_col"] = columns["label_col"]
            except ValueError:
                pass
    return columns


def plot_metadata(meta, *, columns=None, name="Veltri", role="Dataset", source_rows=None):
    columns = default_columns(meta) if columns is None else columns
    df = build_plot_dataframe(meta, **{key: columns[key] for key in ("seq_col", "id_col", "label_col", "amp_col")})
    for field, column, fallback in (
        ("activity", columns["activity_col"], "Unknown"),
        ("structure_id", columns["structure_col"], ""),
    ):
        df[field] = meta[column].reset_index(drop=True).astype("string").fillna(fallback).astype(str) if column else fallback
    df["class_label"] = df["is_amp"].map({True: "AMP", False: "Non-AMP"}).fillna("Unlabeled")
    df["dataset"] = name
    df["role"] = role
    df["source_row_index"] = np.arange(len(df)) if source_rows is None else source_rows
    return df


def holdout_indices(meta, count=30, seed=42):
    """Keep duplicate sequences together so a held-out sequence is truly unseen."""
    sequence_col = pick_col(meta, SEQUENCE_COLUMNS)
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= len(meta) - 3:
        raise ValueError("Holdout size must leave at least three reference rows.")
    groups = (meta[sequence_col].fillna("").astype(str).str.strip().str.upper() if sequence_col else pd.Series([""] * len(meta)))
    groups = pd.Series([value or f"missing-sequence-row-{index}" for index, value in enumerate(groups)])
    held_out = []
    for group in np.random.default_rng(seed).permutation(groups.unique()):
        indices = np.flatnonzero(groups.to_numpy() == group).tolist()
        if len(held_out) + len(indices) > len(meta) - 3:
            continue
        held_out.extend(indices)
        if len(held_out) >= count:
            break
    if len(held_out) < count:
        raise ValueError("Not enough distinct sequences to make this holdout while keeping three reference rows.")
    query = np.sort(held_out)
    reference = np.setdiff1d(np.arange(len(meta)), query)
    return reference, query


@dataclass
class Workflow:
    mode: str
    X: np.ndarray
    meta: pd.DataFrame
    df: pd.DataFrame
    embeddings_source: str
    metadata_source: str
    manifest: dict | None = None
    reference_count: int = 0
    query_manifest: dict | None = None

    @property
    def overlay(self):
        return self.reference_count > 0

    @property
    def reference(self):
        return self.X[:self.reference_count] if self.overlay else self.X

    @property
    def query(self):
        return self.X[self.reference_count:]

    @property
    def fingerprint(self):
        if not self.overlay:
            return joblib_hash((self.X, self.meta))
        return joblib_hash((self.mode, self.X, self.meta, self.df[["dataset", "source_row_index", "role"]]))


def combine_workflow(mode, reference, query, ref_meta, query_meta, ref_df, query_df, manifest, query_manifest):
    df = pd.concat([ref_df, query_df], ignore_index=True)
    df["row_index"] = np.arange(len(df))
    return Workflow(
        mode, np.vstack([reference, query]), pd.concat([ref_meta, query_meta], ignore_index=True), df,
        str(ref_df["dataset"].iloc[0]) + " reference + " + str(query_df["dataset"].iloc[0]), "Reference and query metadata",
        manifest, len(reference), query_manifest,
    )
