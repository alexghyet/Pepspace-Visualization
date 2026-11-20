import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from joblib import hash as joblib_hash
from utils.dataset import numeric_metadata_columns
from utils.embedding_runtime import dependency_versions, get_umap_projection, get_trustworthiness, memory
from utils.globe_view import create_globe_scatter, parse_globe_selection, render_globe
from utils.projection_settings import load_optuna_parameters, projection_arguments as make_projection_arguments, projection_defaults
from utils.sphere_projection import fit_embeddings_in_sphere, project_embeddings_to_sphere
from utils.workflow_ui import prepare_workflow
from utils.reference_projection import get_reference_projection, fit_display, apply_display, reference_bundle
from utils.structure_view import show_structure_inspector

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data(max_entries=12)
def display_coordinates(coordinates, layout):
    if layout == "Inside globe":
        return fit_embeddings_in_sphere(coordinates)
    return project_embeddings_to_sphere(coordinates)


# --- Page setup ---
st.set_page_config(page_title="UMAP Peptide Embeddings", layout="wide")
st.title("PepSpace — UMAP / densMAP Peptide Embeddings")
st.sidebar.caption("Appearance: open ⋮ → Settings → Theme to choose Light, Dark, or System.")


workflow = prepare_workflow(PROJECT_ROOT)
X, meta, df = workflow.X, workflow.meta, workflow.df
emb_source, meta_source = workflow.embeddings_source, workflow.metadata_source
dataset_revision = workflow.fingerprint
fit_count = len(workflow.reference)
st.caption(f"Embeddings: {emb_source} | Metadata: {meta_source}")
if workflow.overlay:
    st.caption("Reference points stay fixed. New points are placed with the saved UMAP model. These coordinates describe embedding similarity.")
has_amp = df["is_amp"].notna().any()
if not has_amp:
    st.caption("Choose Label, Activity, or Dataset to color these points.")


st.sidebar.header("UMAP Settings")
sphere_dims = st.sidebar.selectbox(
    "Sphere Dimensions (3D plot)", [3, 4, 5, 6, 7, 8],
    help="The 2D plot uses its own UMAP fit. Outputs above 3D use PCA for the globe display.",
)
max_neighbors = min(200, fit_count - 1)
settings_context = (joblib_hash(workflow.reference), workflow.overlay, max_neighbors)
if st.session_state.get("settings_context") != settings_context:
    try:
        best = load_optuna_parameters(PROJECT_ROOT / "optuna_results/3d_params.json")
        defaults = projection_defaults(fit_count, best)
    except ValueError as error:
        st.sidebar.warning(f"{error}. Using the shared default settings.")
        defaults = projection_defaults(fit_count)
    if workflow.overlay:
        defaults["method"] = "UMAP"
    for name, value in defaults.items():
        st.session_state[f"setting_{name}"] = value
    st.session_state["applied_settings"] = defaults
    st.session_state["settings_context"] = settings_context

st.sidebar.caption("Changing dimensions keeps the same projection settings and selected peptides.")

with st.sidebar.form("projection_settings"):
    n_neighbors_input = st.number_input(
        "n_neighbors", min_value=2, max_value=max_neighbors, step=1,
        key="setting_n_neighbors",
    )
    min_dist_input = st.number_input(
        "min_dist", min_value=0.0, max_value=1.0, step=0.001,
        format="%.8f", key="setting_min_dist",
    )
    method_input = st.radio("Projection Type", ["UMAP"] if workflow.overlay else ["UMAP", "densMAP"], key="setting_method")
    st.caption("Reference overlays use UMAP because densMAP cannot transform unseen data." if workflow.overlay else "The next three settings apply to densMAP.")
    dens_lambda_input = st.number_input(
        "dens_lambda", min_value=0.0, step=0.01, format="%.8f", key="setting_dens_lambda", disabled=workflow.overlay,
    )
    dens_frac_input = st.number_input(
        "dens_frac", min_value=0.0, max_value=1.0, step=0.01, key="setting_dens_frac", disabled=workflow.overlay,
    )
    dens_var_shift_input = st.number_input(
        "dens_var_shift", min_value=0.000001, step=0.01, key="setting_dens_var_shift", disabled=workflow.overlay,
    )
    if st.form_submit_button("Apply projection settings"):
        st.session_state["applied_settings"] = dict(
            n_neighbors=n_neighbors_input, min_dist=min_dist_input, method=method_input,
            dens_lambda=dens_lambda_input, dens_frac=dens_frac_input,
            dens_var_shift=dens_var_shift_input,
        )

settings = st.session_state["applied_settings"]
projection_arguments = make_projection_arguments(settings)
method = settings["method"]

st.sidebar.header("3D Visualization")
sphere_layout = st.sidebar.radio("Point layout", ["Inside globe", "Sphere surface"])
show_globe = st.sidebar.checkbox("Show globe frame", value=True)
show_axes = st.sidebar.checkbox("Show axes", value=True)
show_hover_labels = st.sidebar.checkbox("Show hover labels", value=True)
point_size = st.sidebar.slider("Point size", 1, 10, 3)
opacity = 1. if workflow.overlay else st.sidebar.slider("Opacity", 0.1, 1.0, 0.7, 0.05)

st.sidebar.header("Region Selection (2D)")
select_mode = st.sidebar.radio(
    "Selection Mode", ["box", "lasso", "pan"],
    help="Select peptides across both views. Enable Filter globe to selection to restrict the globe to selected rows.",
)
calculate_quality = st.sidebar.checkbox(
    "Calculate exact quality scores", value=False,
    help="Full-data trustworthiness is expensive on its first calculation. Scores are cached; this does not change the embeddings.",
)
st.sidebar.caption("Unchanged projections are reused across app restarts. New settings require a full fit.")

st.sidebar.header("Layers and labels")
show_reference = show_new = True
reference_opacity, new_opacity = .2, 1.
if workflow.overlay:
    show_reference = st.sidebar.checkbox("Show reference data", value=True)
    show_new = st.sidebar.checkbox("Show new data", value=True)
    reference_opacity = st.sidebar.slider("Reference opacity", 0.05, 1.0, .2, .05)
    new_opacity = st.sidebar.slider("New data opacity", 0.05, 1.0, 1., .05)
    st.sidebar.caption("Visibility changes both views and keeps coordinates and selections.")
filter_globe_selection = st.sidebar.checkbox("Filter globe to selection", value=not workflow.overlay,
                                             key=f"filter_globe_{workflow.overlay}")
visible_mask = np.ones(len(df), dtype=bool)
if workflow.overlay:
    visible_mask = np.where(df["role"].eq("Reference"), show_reference, show_new)
if not visible_mask.any():
    st.info("Both layers are hidden. Turn on Show reference data or Show new data to view points.")
color_by = st.sidebar.selectbox("Color by", ["AMP status", "Label", "Activity", "Dataset"],
                                index=1 if not has_amp and df["label"].ne("unknown").any() else 0)
color_column = {"AMP status": "class_label", "Label": "label", "Activity": "activity", "Dataset": "dataset"}[color_by]
if color_by == "AMP status":
    color_map = {"AMP": "blue", "Non-AMP": "red", "Unlabeled": "#94a3b8"}
else:
    color_map = {value: px.colors.qualitative.Safe[index % len(px.colors.qualitative.Safe)]
                 for index, value in enumerate(sorted(df[color_column].unique()))}
numeric_columns = [column for column in numeric_metadata_columns(meta)
                   if column not in ("source_row_index", "row_index", "AMP", "is_amp")]
bubble_column = st.sidebar.selectbox("2D bubble size", [None] + numeric_columns,
                                     format_func=lambda column: "Uniform" if column is None else column)

try:
    projection_function = get_reference_projection if workflow.overlay else get_umap_projection
    projection_data = (workflow.reference, workflow.query) if workflow.overlay else (X,)
    with st.spinner("Loading or fitting the 2D projection..."):
        X_umap_2d, fit_2d, runtime_2d, cached_2d = projection_function(
            *projection_data, n_components=2, **projection_arguments,
        )
    with st.spinner(f"Loading or fitting the {sphere_dims}D projection..."):
        X_umap_nd, fit_nd, runtime_nd, cached_nd = projection_function(
            *projection_data, n_components=sphere_dims, **projection_arguments,
        )
    display_transform = fit_display(X_umap_nd[:workflow.reference_count]) if workflow.overlay else None
    X_sphere = (apply_display(X_umap_nd, display_transform, sphere_layout) if workflow.overlay
                else display_coordinates(X_umap_nd, sphere_layout))
except (ValueError, RuntimeError) as error:
    st.error(f"Could not project this dataset: {error}")
    st.stop()

# Keep selections tied to source rows, independently of any display projection.
plot_revision = joblib_hash((dataset_revision, df, projection_arguments))
if st.session_state.get("selection_dataset_revision") != dataset_revision:
    st.session_state.pop("selected_indices", None)
    st.session_state.pop("selected_bbox", None)
    st.session_state.pop("last_2d_selection", None)
    st.session_state.pop("last_globe_event", None)
    st.session_state["selection_generation"] = 0
    st.session_state["selection_dataset_revision"] = dataset_revision
elif st.session_state.get("plot_revision") != plot_revision:
    st.session_state.pop("selected_bbox", None)
st.session_state["plot_revision"] = plot_revision

export_context = {
    "schema_version": 2,
    "dataset": {
        "fingerprint": dataset_revision,
        "embeddings_source": emb_source,
        "metadata_source": meta_source,
        "embedding_shape": list(X.shape),
        "embedding_dtype": str(X.dtype),
    },
    "library_versions": dict(dependency_versions()),
    "workflow": workflow.mode,
    "embedding_manifest": workflow.manifest,
    "query_manifest": workflow.query_manifest,
    "reference_count": workflow.reference_count,
    "source_rows": df[["row_index", "dataset", "role", "source_row_index"]].to_dict(orient="records"),
    "display_transform": {key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in display_transform.items()} if display_transform else None,
    "visible_layers": {"reference": show_reference, "new": show_new},
}

st.subheader("Projection timings")
st.caption(f"{len(X):,} peptides | {X.shape[1]:,} embedding dimensions | {method}")
mc1, mc2 = st.columns(2)
for column, dimensions, fit_time, elapsed, cached in (
    (mc1, 2, fit_2d, runtime_2d, cached_2d),
    (mc2, sphere_dims, fit_nd, runtime_nd, cached_nd),
):
    column.metric(f"{dimensions}D {'cache load' if cached else 'fit'}", f"{elapsed:.3f}s")
    column.caption(f"Original fit: {fit_time:.2f}s | {'Saved result reused' if cached else 'Full fit saved'}")

if workflow.overlay:
    reference_rows = df.iloc[:workflow.reference_count][["dataset", "source_row_index", "id"]].to_dict(orient="records")
    bundle_key = joblib_hash((workflow.reference, reference_rows, sphere_dims, sphere_layout, projection_arguments, workflow.manifest))
    if st.button("Prepare fitted map download"):
        provenance = {"reference_fingerprint": joblib_hash(workflow.reference),
                      "embedding_manifest": workflow.manifest, "parameters": projection_arguments,
                      "display_layout": sphere_layout, "reference_rows": reference_rows}
        with st.spinner("Packaging the fitted UMAP models..."):
            st.session_state["map_download"] = (bundle_key, reference_bundle(
                workflow.reference, (2, sphere_dims), projection_arguments, provenance))
    saved_map = st.session_state.get("map_download")
    if saved_map and saved_map[0] == bundle_key:
        st.download_button("Download fitted map (.joblib)", saved_map[1], "pepspace_reference.joblib")
    st.caption("Fitted maps are saved locally and reused across restarts. The download includes both reducers, reference coordinates, provenance, and the globe transform.")

st.subheader("Embedding Quality")
if calculate_quality:
    # sklearn requires k < n_samples / 2. Keep the old score metric for comparison.
    quality_neighbors = min(settings["n_neighbors"], (len(X) - 1) // 2)
    with st.spinner("Calculating or loading exact quality scores..."):
        trust_2d = get_trustworthiness(X, X_umap_2d, quality_neighbors)
        trust_nd = get_trustworthiness(X, X_umap_nd, quality_neighbors)
        trust_display = (
            trust_nd if sphere_dims == 3 and sphere_layout == "Inside globe"
            else get_trustworthiness(X, X_sphere, quality_neighbors)
        )
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Trustworthiness 2D", f"{trust_2d:.4f}")
    tc2.metric(f"Trustworthiness {sphere_dims}D", f"{trust_nd:.4f}")
    tc3.metric("Trustworthiness displayed 3D", f"{trust_display:.4f}")
    st.caption(
        f"Exact descriptive scores on all {len(X):,} displayed-source rows, k={quality_neighbors}, original-space metric: Euclidean. "
        "The displayed score also measures loss introduced by PCA or surface projection. "
        "Trustworthiness measures local neighborhoods; it does not prove all structure is preserved."
    )
else:
    st.caption("Enable 'Calculate exact quality scores' in the sidebar to evaluate all rows. Skipping evaluation does not change the projections.")


# 2D PLOT — always rendered, acts as the selection tool
st.subheader("2D Projection — Selection Tool")
st.caption("Select peptides to inspect and export. Filter globe to selection restricts the globe to selected rows; selection stays active across 3D–8D.")

plot_df = df.assign(UMAP1=X_umap_2d[:, 0], UMAP2=X_umap_2d[:, 1], peptide_id=df["id"])
for column in meta.columns:
    plot_df[f"metadata.{column}"] = meta[column].to_numpy()
hover_fields = ["peptide_id", "sequence", "class_label", "activity", "dataset", "role", "source_row_index", "structure_id"]
if color_by == "Label":
    hover_fields.append("label")
if bubble_column is not None:
    hover_fields.append(f"metadata.{bubble_column}")
fig_2d = px.scatter(
    plot_df.loc[visible_mask], x="UMAP1", y="UMAP2", color=color_column,
    symbol="role" if workflow.overlay else None,
    color_discrete_map=color_map,
    category_orders={"class_label": ["AMP", "Non-AMP", "Unlabeled"]},
    hover_data=hover_fields,
    custom_data=["row_index"], title=f"{method} 2D projection",
)
marker_sizes = np.full(len(df), 6.)
if bubble_column is not None:
    values = pd.to_numeric(meta[bubble_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    if valid.any():
        largest = values[valid].max()
        marker_sizes[valid] = 6 + 16 * np.sqrt(values[valid] / largest) if largest > 0 else 6
        st.caption(f"2D bubble size: {bubble_column}. Missing, negative, or nonfinite values use the smallest marker.")
for trace in fig_2d.data:
    rows = np.array([int(item[0]) for item in trace.customdata])
    alpha = np.where(df.iloc[rows]["role"].eq("Reference"), reference_opacity, new_opacity) if workflow.overlay else .6
    trace.update(marker=dict(size=marker_sizes[rows], opacity=alpha))
# Native 3D selections highlight the same source rows in the 2D view.
active_selection = set(st.session_state.get("selected_indices", []))
for trace in fig_2d.data:
    trace.selectedpoints = (
        [i for i, values in enumerate(trace.customdata) if int(values[0]) in active_selection]
        if active_selection else None
    )

drag_map = {"pan": "pan", "box": "select", "lasso": "lasso"}
fig_2d.update_layout(
    dragmode=drag_map.get(select_mode, "select"),
    xaxis=dict(range=[float(X_umap_2d[:, 0].min()) - 1, float(X_umap_2d[:, 0].max()) + 1]),
    yaxis=dict(range=[float(X_umap_2d[:, 1].min()) - 1, float(X_umap_2d[:, 1].max()) + 1]),
    height=550,
    uirevision=plot_revision,
)

def record_2d_selection():
    # A fresh gesture can select the same rows as an older gesture after a 3D edit.
    st.session_state["selection_2d_event"] = st.session_state.get("selection_2d_event", 0) + 1


event_2d = st.plotly_chart(
    fig_2d,
    width="stretch",
    on_select=record_2d_selection,
    key=f"umap_2d_plot_{plot_revision}_{st.session_state.get('selection_epoch', 0)}",
)

# Extract selected indices and persist in session_state so the 3D plot
# can read them even after Streamlit reruns triggered by other widgets
selected_points_2d = []
if event_2d and hasattr(event_2d, "selection") and event_2d.selection:
    selected_points_2d = event_2d.selection.get("points", [])

# Only a new gesture may change the selected rows. Streamlit can register an
# empty chart state when selectedpoints changes its figure specification.
indices_2d = sorted({
    int(p["customdata"][0]) for p in selected_points_2d
    if p.get("customdata") and 0 <= int(p["customdata"][0]) < len(df) and visible_mask[int(p["customdata"][0])]
})
selection_signature = (
    dataset_revision, st.session_state.get("selection_epoch", 0),
    st.session_state.get("selection_2d_event", 0),
)
if st.session_state.get("last_2d_selection") != selection_signature:
    st.session_state["last_2d_selection"] = selection_signature
    if selected_points_2d:
        st.session_state["selected_bbox"] = {
            "x_min": round(min(p["x"] for p in selected_points_2d), 4),
            "x_max": round(max(p["x"] for p in selected_points_2d), 4),
            "y_min": round(min(p["y"] for p in selected_points_2d), 4),
            "y_max": round(max(p["y"] for p in selected_points_2d), 4),
        }
    else:
        st.session_state.pop("selected_bbox", None)
    if indices_2d != st.session_state.get("selected_indices", []):
        st.session_state["selected_indices"] = indices_2d
        st.session_state["selection_generation"] = st.session_state.get("selection_generation", 0) + 1
        st.rerun()

# Read persisted selection (survives reruns from dimension switcher etc.)
selected_indices = st.session_state.get("selected_indices", [])
selected_bbox = st.session_state.get("selected_bbox", None)

# Keep one block here so showing the summary cannot remount the globe iframe.
with st.container():
    if selected_indices:
        selected_df = df.iloc[selected_indices].copy()
        amp_count = int(selected_df["is_amp"].eq(True).sum())
        non_amp_count = int(selected_df["is_amp"].eq(False).sum())
        unlabeled_count = len(selected_df) - amp_count - non_amp_count

        st.subheader(f"Selected — {len(selected_indices)} peptides")
        st.caption(f"AMPs: {amp_count} | Non-AMPs: {non_amp_count} | Unlabeled: {unlabeled_count}")

        if selected_bbox:
            st.json(selected_bbox)

        col_save, col_clear = st.columns([1, 1])
        with col_save:
            if st.button("Save 2D Selection"):
                save_path = PROJECT_ROOT / "selections" / f"2d_selection_{time.time_ns()}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f_out:
                    json.dump({
                        **export_context,
                        "bbox": selected_bbox,
                        "params": projection_arguments,
                        "selected_ids": selected_df["id"].tolist(),
                        "selected_row_indices": selected_indices,
                        "projection_dimensions": 2,
                        "projection_coords": plot_df.iloc[selected_indices][["row_index", "UMAP1", "UMAP2"]].to_dict(orient="records"),
                    }, f_out, indent=2)
                st.success(f"Saved to {save_path}")
        with col_clear:
            if st.button("Clear Selection"):
                st.session_state.pop("selected_indices", None)
                st.session_state.pop("selected_bbox", None)
                st.session_state["selection_epoch"] = st.session_state.get("selection_epoch", 0) + 1
                st.rerun()

    elif select_mode in ["box", "lasso"]:
        st.info("Draw a region on the 2D plot above to select peptides.")



# 3D SPHERE PLOT — always rendered, highlights whatever is selected in 2D
st.divider()
st.subheader(f"3D Globe — {sphere_dims}D Projection · ScatterGL")
st.caption("Blue = AMP; red = non-AMP; grey = unlabeled." if color_by == "AMP status" else f"Colors show {color_by.lower()}, matching the 2D legend.")
if sphere_layout == "Inside globe":
    st.caption("Points fit inside the globe by centering and uniform scaling. In 3D, this preserves relative distances and neighbors.")
else:
    st.warning("Surface mode projects points radially onto the sphere and changes distances and potentially neighbors.")
if sphere_dims > 3:
    st.caption(f"The {sphere_dims}D result is displayed using PCA to 3D. This display projection loses some information; the original result is retained.")

sphere_df = pd.DataFrame({
    "row_index": np.arange(len(df)),
    "x":        X_sphere[:, 0],
    "y":        X_sphere[:, 1],
    "z":        X_sphere[:, 2],
    "id":       df["id"],
    "sequence": df["sequence"],
    "is_amp":   df["is_amp"],
})
for column in ("dataset", "role", "source_row_index", "activity", "structure_id"):
    sphere_df[column] = df[column]
sphere_df["visible"] = visible_mask
sphere_df["color"] = df[color_column].map(color_map)
if color_by == "Label":
    sphere_df["display_label"] = df["label"]
sphere_df["point_opacity"] = np.where(df["role"].eq("Reference"), reference_opacity, new_opacity) if workflow.overlay else opacity
sphere_df["selected"] = sphere_df.index.isin(selected_indices)

globe_title = f"{method} {sphere_dims}D sphere"
if selected_indices:
    globe_title += f" | {len(selected_indices)} selected peptides"
globe_revision = joblib_hash((plot_revision, sphere_dims, sphere_layout))
selection_revision = joblib_hash((
    globe_revision, st.session_state.get("selection_epoch", 0),
    st.session_state.get("selection_generation", 0), show_reference, show_new,
))
globe_payload = create_globe_scatter(
    sphere_df, title=globe_title, point_size=point_size,
    opacity=opacity, selected_only=filter_globe_selection,
    show_globe=show_globe, revision=globe_revision, selection_revision=selection_revision,
    show_axes=show_axes, show_hover_labels=show_hover_labels,
    axis_labels=[f"{method} {axis}" if sphere_dims == 3 else f"Display PC {axis}" for axis in (1, 2, 3)],
)
globe_event = render_globe(globe_payload)
globe_indices = parse_globe_selection(globe_event, revision=selection_revision, row_count=len(df))
if globe_indices is not None:
    event_key = (globe_event["revision"], globe_event["event_id"])
    globe_indices = [index for index in globe_indices if visible_mask[index]]
    if event_key != st.session_state.get("last_globe_event"):
        st.session_state["last_globe_event"] = event_key
        if globe_indices != selected_indices:
            st.session_state["selected_indices"] = globe_indices
            st.session_state.pop("selected_bbox", None)
            st.session_state["selection_generation"] = st.session_state.get("selection_generation", 0) + 1
            st.rerun()

# Show selected peptides table below the 3D plot if a selection is active
if selected_indices:
    st.subheader(f"Selected Peptides on Sphere ({sphere_dims}D)")
    selected_sphere = sphere_df.loc[sphere_df["selected"], ["row_index", "id", "sequence", "is_amp", "activity", "dataset", "role", "source_row_index", "structure_id", "x", "y", "z"]].copy()
    st.dataframe(selected_sphere.round({"x": 4, "y": 4, "z": 4}), width="stretch")

    if st.button("Save 3D Sphere Coordinates"):
        save_path = PROJECT_ROOT / "selections" / f"3d_sphere_{sphere_dims}d_{time.time_ns()}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f_out:
            json.dump({
                **export_context,
                "sphere_dims": sphere_dims,
                "point_layout": sphere_layout,
                "n_selected":  len(selected_indices),
                "params": projection_arguments,
                "selected_ids": df.iloc[selected_indices]["id"].tolist(),
                "selected_row_indices": selected_indices,
                "sphere_coords": selected_sphere[["row_index", "id", "x", "y", "z"]].to_dict(orient="records"),
                "projection_coords": [
                    {"row_index": index, "coordinates": X_umap_nd[index].tolist()}
                    for index in selected_indices
                ],
            }, f_out, indent=2)
        st.success(f"Saved to {save_path}")


projection_export = plot_df.copy()
for axis in range(sphere_dims):
    projection_export[f"UMAP_{sphere_dims}D_{axis + 1}"] = X_umap_nd[:, axis]
projection_export[["display_x", "display_y", "display_z"]] = X_sphere
st.download_button("Download all projected rows (.csv)", projection_export.to_csv(index=False),
                   file_name="pepspace_projection.csv", mime="text/csv")
show_structure_inspector()

# --- Clear cache ---
if st.sidebar.button("Clear Cache"):
    memory.clear(warn=False)
    st.cache_data.clear()
    st.rerun()
