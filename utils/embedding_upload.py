"""Read embedding uploads and align optional metadata in embedding row order."""

from io import BytesIO
import re
import zipfile

import numpy as np
import pandas as pd

from utils.dataset import pick_col, validate_embeddings, validate_row_alignment


TABLE_FORMATS = ("csv", "tsv", "parquet")
_DIMENSION_NAME = re.compile(r"(?:emb(?:edding)?|dim(?:ension)?|vec(?:tor)?)[ _.-]?\d+", re.I)


def read_table(contents, file_format, has_header=True):
    if file_format == "parquet":
        frame = pd.read_parquet(BytesIO(contents), engine="pyarrow")
        if any(name is not None for name in frame.index.names):
            frame = frame.reset_index()
    elif file_format in ("csv", "tsv"):
        # Keep text IDs such as 0017 and NA intact through display and joining.
        # Read the header ourselves so pandas doesn't silently rename duplicate columns.
        frame = pd.read_csv(BytesIO(contents), sep="," if file_format == "csv" else "\t",
                            header=None, dtype=str, na_filter=False, skip_blank_lines=False)
        if has_header:
            frame.columns = frame.iloc[0].tolist()
            frame = frame.iloc[1:]
        else:
            frame.columns = [f"Column {i}" for i in range(1, len(frame.columns) + 1)]
    else:
        raise ValueError("Choose a CSV, TSV, or Parquet table.")
    if frame.empty:
        raise ValueError("The uploaded table needs at least one data row and one column.")
    frame.columns = [str(name) if str(name).strip() else f"Column {i}"
                     for i, name in enumerate(frame.columns, start=1)]
    pick_col(frame, ())  # Column mappings must identify exactly one source column.
    return frame.reset_index(drop=True)


def suggested_dimensions(columns):
    return [column for column in columns if _DIMENSION_NAME.fullmatch(column.strip())]


def split_embedding_table(frame, dimensions, min_rows=3):
    """Use only the chosen columns, in file order. Retain all others as metadata."""
    chosen = set(dimensions)
    if len(chosen) < 2 or not chosen.issubset(frame.columns):
        raise ValueError("Choose at least two embedding columns from the uploaded table.")
    names = [column for column in frame if column in chosen]
    points = np.empty((len(frame), len(names)), dtype=np.float32)
    for index, name in enumerate(names):
        values = frame[name]
        try:
            if values.dtype.kind not in "iufOSU":
                raise ValueError
            numeric = pd.to_numeric(values, errors="raise")
            if numeric.dtype.kind not in "iuf":
                raise ValueError
            with np.errstate(over="ignore", invalid="ignore"):
                points[:, index] = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"Embedding column {name!r} must contain a real number in every row.") from error
        if not np.isfinite(points[:, index]).all():
            raise ValueError(f"Embedding column {name!r} has missing, infinite, or out-of-range values.")
    return validate_embeddings(points, min_rows=min_rows), frame.drop(columns=names)


def npz_members(contents):
    try:
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            members = [name[:-4] for name in archive.namelist() if name.endswith(".npy")]
    except zipfile.BadZipFile as error:
        raise ValueError("Could not read the NPZ archive.") from error
    if not members or len(set(members)) != len(members):
        raise ValueError("The NPZ archive must contain NumPy arrays with distinct names.")
    return members


def npz_member_bytes(contents, member, max_bytes):
    with zipfile.ZipFile(BytesIO(contents)) as archive:
        info = archive.getinfo(member + ".npy")
        if info.file_size > max_bytes:
            raise ValueError(f"Array {member!r} exceeds the upload size limit when unpacked.")
        try:
            return archive.read(info)
        except zipfile.BadZipFile as error:
            raise ValueError(f"Could not read array {member!r} from the NPZ archive.") from error


def read_npy(contents):
    try:
        array = np.load(BytesIO(contents), allow_pickle=False)
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("Choose a valid NPY array saved without Python objects or pickle.") from error
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("Expected a single NPY array. Upload archives with the .npz extension.")
    return array


def array_identifiers(contents, name, row_count):
    values = read_npy(contents)
    if values.ndim != 1 or len(values) != row_count or values.dtype.kind not in "iufUS":
        raise ValueError("The identifier array must contain one text or numeric ID per embedding row.")
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8")
    return pd.DataFrame({name: values})


def _join_ids(frame, column, source):
    if column not in frame:
        raise ValueError(f"Choose an identifier column from the {source} file.")
    values = frame[column]
    if not values.map(pd.api.types.is_scalar).all():
        raise ValueError(f"Each {source} identifier in {column!r} must be a single value.")
    ids = pd.Index(values.astype("string").str.strip())
    if ids.isna().any() or (ids == "").any():
        raise ValueError(f"Missing identifiers in {source} column {column!r}. Choose a complete ID column.")
    duplicates = ids[ids.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate identifiers in {source} column {column!r}, including {duplicates[0]!r}. "
                         "Matching by identifier needs one row per ID in each file.")
    return ids


def attach_metadata(inline, metadata, embedding_key=None, metadata_key=None):
    """Preserve all rows and both files' metadata, including colliding column names."""
    if embedding_key is None and metadata_key is None:
        validate_row_alignment(inline, metadata)
    else:
        left = _join_ids(inline, embedding_key, "embedding")
        right = _join_ids(metadata, metadata_key, "metadata")
        missing, extra = left.difference(right, sort=False), right.difference(left, sort=False)
        if len(missing) or len(extra):
            problems = []
            if len(missing):
                problems.append(f"{len(missing):,} missing from metadata (e.g. {missing[0]!r})")
            if len(extra):
                problems.append(f"{len(extra):,} only in metadata (e.g. {extra[0]!r})")
            raise ValueError("Unmatched identifiers: " + ", ".join(problems) + ".")
        metadata = metadata.iloc[right.get_indexer(left)]

    used = {name.strip().casefold() for name in inline}
    names = []
    for name in metadata:
        while name.strip().casefold() in used:
            name = "attached." + name.strip()
        names.append(name)
        used.add(name.strip().casefold())
    return pd.concat([inline.reset_index(drop=True), metadata.set_axis(names, axis=1).reset_index(drop=True)], axis=1)
