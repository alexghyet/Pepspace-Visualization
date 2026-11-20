"""Validate dataset inputs and keep source row identity independent of labels."""

import re

import numpy as np
import pandas as pd


def validate_embeddings(values, *, min_rows=3):
    """Return finite real-valued float32 embeddings without accepting text casts."""
    points = np.asarray(values)
    if points.ndim != 2 or points.shape[0] < min_rows or points.shape[1] < 2:
        raise ValueError(f"Embeddings must be a 2D numeric array with at least {min_rows} rows and two columns.")
    if points.dtype.kind not in "iuf":
        raise ValueError("Embeddings must contain real numeric values, not strings, objects, booleans, or complex numbers.")
    if not np.isfinite(points).all():
        raise ValueError("Embeddings must contain only finite values; remove NaN and infinity.")
    with np.errstate(over="ignore", invalid="ignore"):
        points = points.astype(np.float32, copy=False)
    if not np.isfinite(points).all():
        raise ValueError("Embedding values exceed the supported float32 range.")
    return points


def validate_row_alignment(embeddings, metadata):
    """Require one metadata row per embedding; truncation cannot prove alignment."""
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Row mismatch: embeddings have {len(embeddings):,} rows but metadata has {len(metadata):,}. "
            "Provide metadata in the same row order as the embeddings; no rows were truncated."
        )


def pick_col(frame, candidates):
    """Match column names without changing their source names or their order."""
    lookup = {}
    for column in frame.columns:
        normalized = str(column).strip().casefold()
        if normalized in lookup:
            raise ValueError(f"Metadata column names are ambiguous: {lookup[normalized]!r} and {column!r}.")
        lookup[normalized] = column
    for candidate in candidates:
        normalized = str(candidate).strip().casefold()
        if normalized in lookup:
            return lookup[normalized]
    return None


def parse_amp_labels(values):
    """Parse explicit binary labels; Python truthiness would turn 'False' true."""
    series = pd.Series(values, copy=False)
    parsed, invalid = [], []
    true_tokens = {"true", "1", "1.0", "amp"}
    false_tokens = {"false", "0", "0.0", "nonamp"}
    for position, value in enumerate(series):
        if pd.isna(value):
            invalid.append(position)
            parsed.append(False)
            continue
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            parsed.append(value == 1)
        elif isinstance(value, str):
            token = value.strip().casefold()
            if re.fullmatch(r"non[\s_\-]*amp", token):
                token = "nonamp"
            if token in true_tokens:
                parsed.append(True)
            elif token in false_tokens:
                parsed.append(False)
            else:
                invalid.append(position)
                parsed.append(False)
        else:
            invalid.append(position)
            parsed.append(False)
    if invalid:
        examples = ", ".join(str(position + 1) for position in invalid[:5])
        raise ValueError(
            f"AMP labels must be True/False, 1/0, or AMP/Non-AMP; missing or invalid values at metadata rows {examples}."
        )
    return pd.Series(parsed, index=series.index, dtype=bool, name=series.name)


def build_plot_dataframe(meta, *, seq_col=None, id_col=None, label_col=None, amp_col=None):
    """Normalize metadata without relying on duplicate IDs or incoming indices."""
    if not isinstance(meta, pd.DataFrame) or len(meta) == 0:
        raise ValueError("Metadata must contain at least one row.")
    # Validate normalized column uniqueness even when callers provide overrides.
    pick_col(meta, ())
    for column in (seq_col, id_col, label_col, amp_col):
        if column is not None and column not in meta.columns:
            raise ValueError(f"Metadata column {column!r} does not exist.")

    source = meta.reset_index(drop=True)
    row_indices = np.arange(len(source))
    fallback_ids = pd.Series(row_indices).astype(str)
    identifiers = fallback_ids
    if id_col is not None:
        supplied = source[id_col].astype("string")
        identifiers = supplied.where(supplied.str.strip().ne("").fillna(False), fallback_ids).astype(str)
    frame = pd.DataFrame({
        "row_index": row_indices,
        "id": identifiers,
        "label": source[label_col].astype("string").fillna("unknown").astype(str) if label_col is not None else "unknown",
        "sequence": source[seq_col].astype("string").fillna("").astype(str) if seq_col is not None else "",
        "is_amp": parse_amp_labels(source[amp_col]) if amp_col is not None else None,
    })
    frame.attrs["amp_labels_available"] = amp_col is not None
    return frame


def numeric_metadata_columns(meta):
    """Find potential size fields while keeping uploaded text IDs in their original form."""
    columns = []
    for name, values in meta.items():
        if values.dtype.kind in "iuf":
            columns.append(name)
        elif values.dtype.kind in "OUS":
            present = values.dropna()
            present = present[present.astype("string").str.strip().ne("")]
            if len(present) and pd.to_numeric(present, errors="coerce").notna().all():
                columns.append(name)
    return columns
