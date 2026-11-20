"""Streamlit inputs for independent datasets, uploads, and reference overlays."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

from joblib import hash as joblib_hash
import numpy as np
import pandas as pd
import streamlit as st

from utils.dataset import validate_embeddings, validate_row_alignment
from utils.embedding_upload import (
    TABLE_FORMATS, array_identifiers, attach_metadata, npz_member_bytes, npz_members,
    read_npy, read_table, split_embedding_table, suggested_dimensions,
)
from utils.peptidebert import load_encoder, embed_sequences
from utils.project_paths import EMBEDDINGS_PATH, METADATA_PATH, MANIFEST_PATH, CHECKPOINT_PATH, VELTRI_PATH
from utils.workflow import (
    Workflow, check_compatibility, combine_workflow, default_columns,
    holdout_indices, plot_metadata, read_manifest,
)

EXPLORE = "Explore a dataset"
OVERLAY = "Project onto Veltri"
HOLDOUT = "Holdout demo"

read_upload_table = st.cache_data(max_entries=6, show_spinner=False)(read_table)
unpack_member = st.cache_data(max_entries=4, show_spinner=False)(npz_member_bytes)


@st.cache_data(max_entries=8)
def read_embeddings(contents, min_rows=3):
    return validate_embeddings(read_npy(contents), min_rows=min_rows)


@st.cache_data(max_entries=4, show_spinner=False)
def selected_table_embeddings(source_key, dimensions, min_rows, _table):
    """Cache by file digest and parsing options while reusing the table already open for preview."""
    return split_embedding_table(_table, dimensions, min_rows)


@st.cache_data(max_entries=8)
def read_metadata(contents):
    frame = pd.read_csv(BytesIO(contents))
    if frame.empty:
        raise ValueError("The metadata CSV needs at least one row.")
    return frame


@st.cache_resource(max_entries=1)
def cached_encoder(checkpoint, config, revisions, expected_hash, expected_config):
    return load_encoder(checkpoint, config, expected_hash=expected_hash, expected_config=expected_config)


@st.cache_data(max_entries=8)
def generated_embeddings(sequences, checkpoint, config, revisions, expected_hash, expected_config):
    encoder = cached_encoder(checkpoint, config, revisions, expected_hash, expected_config)
    progress = st.progress(0., text="Generating PeptideBERT embeddings on CPU…")
    try:
        return embed_sequences(sequences, encoder, progress=lambda done, total: progress.progress(done / total))
    finally:
        progress.empty()


def embedding_archive(X, meta, manifest):
    """A reusable upload pair and matching provenance, preserving CSV row order."""
    array = BytesIO()
    np.save(array, X)
    metadata = meta.to_csv(index=False).encode("utf-8")
    manifest = {**manifest, "shape": list(X.shape), "dataset_sha256": hashlib.sha256(metadata).hexdigest(),
                "embeddings_sha256": hashlib.sha256(array.getvalue()).hexdigest()}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("embeddings.npy", array.getvalue())
        archive.writestr("metadata.csv", metadata)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    return stream.getvalue()


def column_controls(meta, key, id_hint=None):
    columns = default_columns(meta)
    if not len(meta.columns):
        return columns
    if id_hint is not None:
        columns["id_col"] = id_hint
    options = [None, *meta.columns]
    with st.sidebar.expander("Metadata columns"):
        for field, label in (("id_col", "ID column"), ("seq_col", "Sequence column"), ("label_col", "Label column"),
                             ("amp_col", "AMP status column"), ("activity_col", "Activity column"),
                             ("structure_col", "Structure ID column")):
            fallback = "Use row numbers" if field == "id_col" else "None"
            columns[field] = st.selectbox(
                label, options, index=options.index(columns[field]), key=f"{key}_{field}",
                format_func=lambda value, empty=fallback: empty if value is None else str(value),
            )
    return columns


def load_veltri(root):
    X = read_embeddings((root / EMBEDDINGS_PATH).read_bytes())
    meta = read_metadata((root / METADATA_PATH).read_bytes())
    validate_row_alignment(X, meta)
    return X, meta


def reference_manifest(root):
    manifest = read_manifest((root / MANIFEST_PATH).read_bytes(), (root / EMBEDDINGS_PATH).read_bytes())
    return manifest


def table_input(upload, key):
    file_format = Path(upload.name).suffix.lower().lstrip(".")
    has_header = True
    if file_format in ("csv", "tsv"):
        has_header = st.sidebar.checkbox(f"First row of {upload.name} contains column names", value=True, key=f"{key}_header")
    frame = read_upload_table(upload.getvalue(), file_format, has_header)
    return frame, file_format, has_header


def embedding_input(upload, key, min_rows):
    contents = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix.lstrip(".") in TABLE_FORMATS:
        table, file_format, has_header = table_input(upload, key)
        with st.expander("Embedding file preview", expanded=True):
            st.dataframe(table.head(5), hide_index=True)
        suggestions = suggested_dimensions(table.columns)
        if st.sidebar.checkbox("Select columns to leave out instead", key=f"{key}_exclude"):
            omitted = st.sidebar.multiselect(
                "Columns to keep as metadata", table.columns,
                default=[c for c in table if c not in suggestions] if suggestions else [],
                key=f"{key}_{has_header}_omitted",
                help="All remaining columns will be used as embedding dimensions.")
            chosen = set(table.columns) - set(omitted)
        else:
            chosen = set(st.sidebar.multiselect(
                "Embedding columns", table.columns, default=suggestions,
                key=f"{key}_{has_header}_dimensions",
                help="Choose one numeric column per dimension. Other columns stay available as metadata."))
        dimensions = tuple(column for column in table if column in chosen)
        if len(dimensions) < 2:
            st.info("Choose at least two numeric embedding columns in the sidebar.")
            st.stop()
        X, inline = selected_table_embeddings((key, file_format, has_header), dimensions, min_rows, table)
        return X, inline, None, (has_header, dimensions)

    member = id_member = None
    npy_bytes = contents
    if suffix == ".npz":
        members = npz_members(contents)
        default = next((i for i, name in enumerate(members) if name.casefold() in ("embeddings", "x")), None)
        member = st.sidebar.selectbox("Embedding array", members, index=0 if len(members) == 1 else default,
                                      placeholder="Choose the 2D array", key=f"{key}_array")
        if member is None:
            st.info("Choose the array containing your embedding matrix.")
            st.stop()
        limit = st.get_option("server.maxUploadSize") * 1024**2
        npy_bytes = unpack_member(contents, member, limit)
        others = [name for name in members if name != member]
        if others:
            id_member = st.sidebar.selectbox("Identifier array (optional)", [None, *others], key=f"{key}_ids",
                                             help="Choose a one-dimensional array with one ID per embedding row.")
    X = read_embeddings(npy_bytes, min_rows=min_rows)
    inline = pd.DataFrame(index=pd.RangeIndex(len(X)))
    if id_member is not None:
        inline = array_identifiers(unpack_member(contents, id_member, limit), id_member, len(X))
    with st.expander("Embedding file preview"):
        st.dataframe(pd.DataFrame(X[:5, :8], columns=[f"Dimension {i + 1}" for i in range(min(8, X.shape[1]))]),
                     hide_index=True)
    return X, inline, npy_bytes, (member, id_member)


def metadata_input(inline, upload, key, id_hint):
    if upload is None:
        return inline, id_hint, None
    additional, _, has_header = table_input(upload, key)
    left_key = right_key = None
    if len(inline.columns):
        alignment = st.sidebar.radio("Match metadata by", ["Row order", "Identifier"], key=f"{key}_alignment")
        if alignment == "Identifier":
            left_default = id_hint or default_columns(inline)["id_col"] or inline.columns[0]
            left_options = list(inline.columns)
            left_key = st.sidebar.selectbox("ID in embeddings", left_options, index=left_options.index(left_default))
            right_options = list(additional.columns)
            right_default = left_key if left_key in right_options else default_columns(additional)["id_col"]
            right_key = st.sidebar.selectbox("ID in metadata", right_options,
                                             index=right_options.index(right_default) if right_default else 0)
            id_hint = left_key
    if left_key is None:
        st.sidebar.caption("Matching by row order pairs the first metadata row with the first embedding, and so on.")
    meta = attach_metadata(inline, additional, left_key, right_key)
    return meta, id_hint, (has_header, left_key, right_key)


def upload_manifest(X, npy_bytes, reference):
    upload = st.sidebar.file_uploader(
        "Embedding manifest (.json)" + (" — required for overlay" if reference is not None else " — optional"), type=["json"],
    )
    if upload is None:
        if reference is not None:
            st.info("Overlay needs a matching embedding manifest. To fit an independent map, choose Explore a dataset.")
            st.stop()
        return None
    contents = upload.getvalue()
    manifest = read_manifest(contents)
    if manifest.get("shape") != list(X.shape):
        raise ValueError("The manifest shape must match the selected embedding matrix.")
    if manifest.get("embeddings_sha256"):
        if npy_bytes is None:
            # A tabular export can be checked against the project's float32 NPY manifest.
            array = BytesIO()
            np.save(array, np.ascontiguousarray(X), allow_pickle=False)
            npy_bytes = array.getvalue()
        read_manifest(contents, npy_bytes)
    if reference is not None:
        check_compatibility(reference, manifest, X.shape[1])
    return manifest


def precomputed_dataset(reference):
    overlay = reference is not None
    if not overlay:
        sample = Path(__file__).resolve().parents[1] / "data/examples/sample_embeddings.csv"
        st.sidebar.download_button("Download example embeddings", sample.read_bytes(), file_name=sample.name, mime="text/csv")
    upload = st.sidebar.file_uploader("Upload embeddings", type=["npy", "npz", *TABLE_FORMATS])
    if upload is None:
        st.info("Upload a matrix or a table with one embedding vector per row.")
        st.stop()
    key = hashlib.sha256(upload.getvalue()).hexdigest()
    X, inline, npy_bytes, selection = embedding_input(upload, key, min_rows=1 if overlay else 3)
    sidecar = st.sidebar.file_uploader("Attach metadata (optional)", type=list(TABLE_FORMATS))
    metadata_key = hashlib.sha256(sidecar.getvalue()).hexdigest() if sidecar is not None else None
    id_hint = inline.columns[0] if Path(upload.name).suffix.lower() == ".npz" and len(inline.columns) else None
    meta, id_hint, alignment = metadata_input(inline, sidecar, f"metadata_{key}_{metadata_key}", id_hint)
    columns = column_controls(meta, f"map_{key}_{metadata_key}_{joblib_hash((tuple(meta.columns), id_hint))}", id_hint)
    name = st.sidebar.text_input("Dataset name", value=Path(upload.name).stem).strip() or "My dataset"
    manifest = upload_manifest(X, npy_bytes, reference)
    df = plot_metadata(meta, columns=columns, name=name, role="New" if overlay else "Dataset")

    metadata_file = (sidecar.name, metadata_key) if sidecar is not None else None
    signature = joblib_hash((overlay, upload.name, key, selection, metadata_file, alignment, columns, name, manifest))
    pending = st.session_state.get("projected_upload") != signature
    summary = f"{len(X):,} points × {X.shape[1]:,} embedding dimensions"
    st.sidebar.caption(summary)
    with st.expander("Point preview", expanded=pending):
        preview = meta.head(5).copy()
        preview.index = pd.Index(df["id"].head(5), name="Point ID")
        if len(preview.columns):
            st.dataframe(preview)
        else:
            st.dataframe(df[["id"]].head(5).rename(columns={"id": "Point ID"}), hide_index=True)
    if st.sidebar.button("Project dataset", type="primary"):
        st.session_state["projected_upload"] = signature
    elif pending:
        st.info("Check the point IDs and metadata in the preview, then select Project dataset.")
        st.stop()
    sources = ([upload.name] if len(inline.columns) else []) + ([sidecar.name] if sidecar is not None else [])
    return X, meta, df, manifest, upload.name, " + ".join(sources) or "Generated row IDs"


def uploaded_dataset(root, reference=None):
    overlay = reference is not None
    source = st.sidebar.radio("Upload format", ["Existing embeddings", "Peptide sequences (.csv)"],
                              index=1 if overlay else 0, key="upload_format")
    if source == "Existing embeddings":
        return precomputed_dataset(reference)
    st.sidebar.download_button(
        "Download CSV template", "id,sequence,activity\nexample-1,KWKLFKKIGAVLKVL,antibacterial\n",
        file_name="peptide_template.csv", mime="text/csv",
    )
    uploaded_csv = st.sidebar.file_uploader("Upload sequences and metadata (.csv)", type=["csv"])
    if uploaded_csv is None:
        st.info("Upload a CSV with a peptide sequence column.")
        st.stop()
    meta = read_upload_table(uploaded_csv.getvalue(), "csv")
    columns = column_controls(meta, "uploaded")
    name = st.sidebar.text_input("Dataset name", value="My peptides").strip() or "My peptides"
    if columns["seq_col"] is None:
        raise ValueError("Choose a sequence column under Metadata columns.")
    # Validate metadata before starting potentially expensive inference.
    plot_metadata(meta, columns=columns)
    manifest = reference if overlay else reference_manifest(root)
    checkpoint = root / CHECKPOINT_PATH
    config = checkpoint.with_name("config.yaml")
    if not checkpoint.is_file() or not config.is_file():
        raise ValueError("Restore the PeptideBERT solubility model.pt and config.yaml in .cache/peptidebert/v2.0/sol/ to embed sequences.")
    revisions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in (checkpoint, config))
    sequences = meta[columns["seq_col"]].tolist()
    signature = joblib_hash((sequences, manifest["checkpoint_sha256"], revisions))
    if st.sidebar.button("Generate embeddings", type="primary"):
        st.session_state["generate_request"] = signature
    if st.session_state.get("generate_request") != signature:
        st.info(f"Ready to embed {len(meta):,} peptides using PeptideBERT v2.0 solubility. Select Generate embeddings.")
        st.stop()
    with st.spinner("Loading PeptideBERT and generating embeddings…"):
        X = generated_embeddings(tuple(sequences), str(checkpoint), str(config), revisions, manifest["checkpoint_sha256"], manifest.get("config"))
    X = validate_embeddings(X, min_rows=1 if overlay else 3)
    description = "PeptideBERT embeddings generated from " + uploaded_csv.name
    manifest = {**manifest, "shape": list(X.shape), "dataset_sha256": hashlib.sha256(uploaded_csv.getvalue()).hexdigest()}
    manifest.pop("embeddings_sha256", None)
    st.sidebar.download_button("Download embeddings + metadata", embedding_archive(X, meta, manifest),
                               file_name="peptidebert_embeddings.zip", mime="application/zip")
    df = plot_metadata(meta, columns=columns, name=name, role="New" if overlay else "Dataset")
    return X, meta, df, manifest, description, uploaded_csv.name


def prepare_workflow(root):
    st.sidebar.header("Workflow")
    mode = st.sidebar.selectbox("Choose workflow", [EXPLORE, OVERLAY, HOLDOUT])
    try:
        if mode == EXPLORE:
            source = st.sidebar.radio("Dataset", ["Veltri (PeptideBERT)", "Upload my data"])
            if source == "Upload my data":
                X, meta, df, manifest, emb_source, meta_source = uploaded_dataset(root)
            else:
                X, meta = load_veltri(root)
                meta_choice = st.sidebar.radio("Metadata to use for labels/hover", ["Veltri metadata (recommended)", "Raw Veltri CSV (all_veltri.csv)"])
                meta_source = METADATA_PATH if meta_choice == "Veltri metadata (recommended)" else VELTRI_PATH
                if meta_source == VELTRI_PATH:
                    meta = read_metadata((root / meta_source).read_bytes())
                validate_row_alignment(X, meta)
                df = plot_metadata(meta, columns=column_controls(meta, "veltri"))
                manifest = reference_manifest(root) if (root / MANIFEST_PATH).is_file() else None
                emb_source = EMBEDDINGS_PATH
            return Workflow(mode, X, meta, df, emb_source, meta_source, manifest)

        holdout_source = st.sidebar.radio("Holdout dataset", ["Veltri (PeptideBERT)", "Upload my data"]) if mode == HOLDOUT else None
        if holdout_source == "Upload my data":
            reference, ref_meta, base_df, manifest, _, _ = uploaded_dataset(root)
        else:
            reference, ref_meta = load_veltri(root)
            manifest = reference_manifest(root)
            base_df = plot_metadata(ref_meta)
        if mode == HOLDOUT:
            if len(reference) < 4:
                raise ValueError("A holdout demo needs at least four rows: three to fit and one to transform.")
            count = st.sidebar.number_input("Held-out peptides", min_value=1, max_value=len(reference) - 3, value=min(30, len(reference) - 3))
            seed = st.sidebar.number_input("Split seed", min_value=0, max_value=2**31 - 1, value=42)
            ref_rows, query_rows = holdout_indices(base_df, count, seed)
            query = reference[query_rows]
            query_meta = ref_meta.iloc[query_rows].reset_index(drop=True)
            ref_meta = ref_meta.iloc[ref_rows].reset_index(drop=True)
            reference = reference[ref_rows]
            ref_df = base_df.iloc[ref_rows].copy()
            ref_df["role"] = "Reference"
            query_df = base_df.iloc[query_rows].copy()
            query_df["role"] = "Held-out"
            query_df["dataset"] = query_df["dataset"] + " holdout"
            query_manifest = manifest
            st.info(f"Fitting UMAP on {len(reference):,} reference points and transforming {len(query):,} held-out points. Exact duplicate sequences stay in the same split.")
        else:
            query, query_meta, query_df, query_manifest, _, _ = uploaded_dataset(root, reference=manifest)
            ref_df = plot_metadata(ref_meta, role="Reference")
        return combine_workflow(mode, reference, query, ref_meta, query_meta, ref_df, query_df, manifest, query_manifest)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        st.error(str(error))
        st.stop()
