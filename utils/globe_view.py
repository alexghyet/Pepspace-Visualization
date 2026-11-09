"""Streamlit bridge for the locally bundled ScatterGL peptide explorer."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def create_globe_scatter(
    sphere_df, *, title, point_size, opacity, selected_only=True,
    show_globe=True, revision="", selection_revision="",
    show_axes=True, show_hover_labels=True, axis_labels=None,
):
    """Send source identities and unchanged coordinates to the native renderer.

    Keep every peptide here: browser-only visibility changes must not change
    exported coordinates or the mapping between a point and its metadata.
    """
    metadata = []
    for row in sphere_df.itertuples(index=False):
        identifier = str(row.id)
        is_amp = None if row.is_amp is None else bool(row.is_amp)
        class_label = "Unlabeled" if is_amp is None else "AMP" if is_amp else "Non-AMP"
        metadata.append({
            "row_index": int(row.row_index),
            "id": identifier,
            "sequence": str(row.sequence),
            "is_amp": is_amp,
            "label": f"{identifier} | {class_label}",
        })
    return {
        "points": sphere_df[["x", "y", "z"]].to_numpy(dtype=float).tolist(),
        "metadata": metadata,
        "selected_indices": sphere_df.loc[sphere_df["selected"], "row_index"].astype(int).tolist(),
        "point_size": int(point_size),
        "opacity": float(opacity),
        "selected_only": bool(selected_only),
        "show_globe": bool(show_globe),
        "show_axes": bool(show_axes),
        "show_hover_labels": bool(show_hover_labels),
        "axis_labels": list(axis_labels) if axis_labels is not None else ["X", "Y", "Z"],
        "revision": str(revision),
        "selection_revision": str(selection_revision),
        "title": str(title),
    }


def parse_globe_selection(event, *, revision, row_count):
    """Reject stale or malformed browser events; an empty valid list clears."""
    if not isinstance(event, dict) or event.get("revision") != revision:
        return None
    event_id = event.get("event_id")
    if isinstance(event_id, bool) or not isinstance(event_id, (str, int)) or event_id == "":
        return None
    indices = event.get("selected_indices")
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < row_count
        for index in indices
    ):
        return None
    return sorted(set(indices))


@st.cache_resource
def _globe_component():
    # ScatterGL and Three.js are checked-in browser assets, served locally.
    source = Path(__file__).with_name("globe_frontend")
    return components.declare_component("pepspace_scattergl", path=str(source))


def render_globe(payload):
    return _globe_component()(payload=payload, key="peptide_scattergl", default=None)
