"""App rerun regressions with isolated files and inexpensive projections."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from joblib import hash as joblib_hash
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LinkedViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.embeddings = np.random.default_rng(17).normal(size=(12, 8)).astype("float32")
        self.metadata = pd.DataFrame({
            "id": [f"peptide-{index}" for index in range(12)],
            "sequence": [f"AAK{index}" for index in range(12)],
            "label": ["test"] * 12,
            "is_amp": [index % 2 == 0 for index in range(12)],
        })
        self.embedding_path = self.root / "embeddings/veltri/veltri_embeddings.npy"
        self.metadata_path = self.root / "embeddings/veltri/veltri_metadata.csv"
        self.embedding_path.parent.mkdir(parents=True)
        self.write_dataset()
        parameter_directory = self.root / "optuna_results"
        parameter_directory.mkdir()
        for dimensions, neighbors, distance in ((3, 4, .05), (8, 10, .75)):
            (parameter_directory / f"{dimensions}d_params.json").write_text(json.dumps({
                "best_params": {"n_neighbors": neighbors, "min_dist": distance, "method": "UMAP"},
            }), encoding="utf8")

        self.payloads = []
        self.figures = []
        self.projection_calls = []
        self.native_event = None
        self.plotly_points = []
        self.chart_states = {}
        self.chart_identity = None
        real_chart = st.plotly_chart

        def projection(points, *, n_components, **settings):
            self.projection_calls.append((n_components, settings.copy()))
            return points[:, :n_components].copy() * (1 + settings["min_dist"]), 0., 0., True

        def chart(figure, *args, **kwargs):
            self.figures.append(figure)
            real_chart(figure, *args, **kwargs)
            # Streamlit 1.53 includes the figure spec in the widget identity,
            # even with a key: changing selectedpoints creates empty state.
            self.chart_identity = (kwargs.get("key"), figure.to_json())
            return SimpleNamespace(selection={"points": self.chart_states.get(self.chart_identity, [])})

        def globe(payload):
            self.payloads.append(payload)
            return self.native_event

        for name, replacement in (
            ("utils.embedding_runtime.get_umap_projection", projection),
            ("utils.globe_view.render_globe", globe),
            ("streamlit.plotly_chart", chart),
        ):
            mocked = patch(name, side_effect=replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf8")
        original_root = "PROJECT_ROOT = Path(__file__).resolve().parent"
        self.assertIn(original_root, source)
        source = source.replace(original_root, f"PROJECT_ROOT = Path({str(self.root)!r})", 1)
        script_path = self.root / "app_test.py"
        script_path.write_text(source, encoding="utf8")
        self.app = AppTest.from_file(str(script_path), default_timeout=30).run()
        self.assert_selection([])

    def write_dataset(self):
        np.save(self.embedding_path, self.embeddings)
        self.metadata.to_csv(self.metadata_path, index=False)

    def widget(self, collection, label):
        return next(item for item in getattr(self.app, collection) if item.label == label)

    def assert_selection(self, expected):
        self.assertFalse(self.app.exception, [item.message for item in self.app.exception])
        self.assertEqual(self.app.session_state.filtered_state.get("selected_indices", []), expected)
        payload = self.payloads[-1]
        self.assertEqual(payload["selected_indices"], expected)
        self.assertIs(payload["selected_only"], True)
        self.assertEqual(len(payload["points"]), len(self.metadata))
        self.assertEqual([record["row_index"] for record in payload["metadata"]], list(range(12)))
        highlighted = sorted(
            int(trace.customdata[index][0])
            for trace in self.figures[-1].data
            for index in (trace.selectedpoints or [])
        )
        self.assertEqual(highlighted, expected)
        if expected:
            table = self.app.dataframe[0].value
            self.assertEqual(table["id"].tolist(), self.metadata.iloc[expected]["id"].tolist())
            self.assertEqual(table["is_amp"].tolist(), self.metadata.iloc[expected]["is_amp"].tolist())
        else:
            self.assertFalse(self.app.dataframe)

    def select_2d(self, indices):
        selected = set(indices)
        self.plotly_points = [
            {"customdata": values.tolist(), "x": float(trace.x[index]), "y": float(trace.y[index])}
            for trace in self.figures[-1].data
            for index, values in enumerate(trace.customdata)
            if int(values[0]) in selected
        ]
        self.app.session_state["selection_2d_event"] = (
            self.app.session_state.filtered_state.get("selection_2d_event", 0) + 1
        )
        self.chart_states[self.chart_identity] = self.plotly_points
        self.app.run()

    def select_native(self, indices, event_id):
        self.native_event = {
            "revision": self.payloads[-1]["selection_revision"],
            "event_id": event_id,
            "selected_indices": indices,
        }
        self.app.run()

    def test_2d_selection_filters_globe_and_survives_all_globe_dimensions(self):
        self.select_2d([1, 4, 9])
        self.assert_selection([1, 4, 9])
        original_settings = self.app.session_state["applied_settings"].copy()
        original_xy = [np.column_stack((trace.x, trace.y)) for trace in self.figures[-1].data]
        for dimensions in range(4, 9):
            with self.subTest(dimensions=dimensions):
                self.widget("selectbox", "Sphere Dimensions (3D plot)").set_value(dimensions).run()
                self.assert_selection([1, 4, 9])
                self.assertEqual(self.app.session_state["applied_settings"], original_settings)
                for trace, xy in zip(self.figures[-1].data, original_xy):
                    np.testing.assert_array_equal(np.column_stack((trace.x, trace.y)), xy)

    def test_native_retained_stale_and_repeated_2d_events_do_not_lose_source_ids(self):
        self.select_2d([1, 4])
        self.assert_selection([1, 4])
        self.select_native([2, 7], "native-1")
        self.assert_selection([2, 7])
        self.widget("slider", "Point size").set_value(5).run()
        self.assert_selection([2, 7])
        self.select_2d([1, 4])
        self.assert_selection([1, 4])
        # The retained native event predates the fresh 2D gesture.
        self.app.run()
        self.assert_selection([1, 4])
        self.select_native([], "native-clear")
        self.assert_selection([])

    def test_display_controls_keep_selection_and_describe_actual_projection(self):
        self.select_2d([1, 4])
        original_settings = self.app.session_state["applied_settings"].copy()
        self.assertEqual(self.payloads[-1]["axis_labels"], ["UMAP 1", "UMAP 2", "UMAP 3"])
        for label, flag in (("Show axes", "show_axes"), ("Show hover labels", "show_hover_labels")):
            self.widget("checkbox", label).uncheck().run()
            self.assert_selection([1, 4])
            self.assertIs(self.payloads[-1][flag], False)
            self.assertEqual(self.app.session_state["applied_settings"], original_settings)
        self.widget("selectbox", "Sphere Dimensions (3D plot)").set_value(8).run()
        self.assert_selection([1, 4])
        self.assertEqual(self.payloads[-1]["axis_labels"], ["Display PC 1", "Display PC 2", "Display PC 3"])

    def test_applying_settings_preserves_selection_but_new_dataset_clears_it(self):
        self.select_2d([0, 3])
        self.assert_selection([0, 3])
        self.widget("number_input", "min_dist").set_value(.2)
        self.widget("button", "Apply projection settings").click().run()
        self.assert_selection([0, 3])
        self.embeddings += .25
        self.metadata["id"] = self.metadata["id"].map(lambda value: value + "-replacement")
        self.write_dataset()
        self.plotly_points = []
        self.app.run()
        self.assert_selection([])
        self.assertEqual(self.payloads[-1]["metadata"][0]["id"], "peptide-0-replacement")

    def test_clear_button_ignores_retained_2d_state_and_allows_same_region_again(self):
        self.select_2d([1, 4])
        self.assert_selection([1, 4])
        self.widget("button", "Clear Selection").click().run()
        self.assert_selection([])
        self.app.run()
        self.assert_selection([])
        self.select_2d([1, 4])
        self.assert_selection([1, 4])

    def test_saved_selections_include_source_identity_and_original_projection_coordinates(self):
        selected = [1, 4]
        self.select_2d(selected)
        self.widget("button", "Save 2D Selection").click().run()
        self.assert_selection(selected)
        two_dimensional = json.loads(next((self.root / "selections").glob("2d_selection_*.json")).read_text())
        self.widget("selectbox", "Sphere Dimensions (3D plot)").set_value(8).run()
        self.widget("button", "Save 3D Sphere Coordinates").click().run()
        self.assert_selection(selected)
        eight_dimensional = json.loads(next((self.root / "selections").glob("3d_sphere_8d_*.json")).read_text())

        expected_fingerprint = joblib_hash((np.load(self.embedding_path), pd.read_csv(self.metadata_path)))
        expected_ids = self.metadata.iloc[selected]["id"].tolist()
        scale = 1 + self.app.session_state["applied_settings"]["min_dist"]
        for saved in (two_dimensional, eight_dimensional):
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["dataset"]["fingerprint"], expected_fingerprint)
            self.assertEqual(saved["dataset"]["embedding_shape"], [12, 8])
            self.assertEqual(saved["dataset"]["embedding_dtype"], "float32")
            self.assertEqual(saved["params"]["seed"], 42)
            self.assertEqual(saved["params"]["metric"], "cosine")
            self.assertEqual(saved["selected_row_indices"], selected)
            self.assertEqual(saved["selected_ids"], expected_ids)
            self.assertTrue({"numpy", "umap-learn", "scikit-learn"} <= saved["library_versions"].keys())
            self.assertEqual([row["row_index"] for row in saved["projection_coords"]], selected)
        self.assertEqual(two_dimensional["projection_dimensions"], 2)
        np.testing.assert_array_equal(
            [[row["UMAP1"], row["UMAP2"]] for row in two_dimensional["projection_coords"]],
            (self.embeddings[:, :2] * scale)[selected],
        )
        self.assertEqual(eight_dimensional["sphere_dims"], 8)
        np.testing.assert_array_equal(
            [row["coordinates"] for row in eight_dimensional["projection_coords"]],
            (self.embeddings * scale)[selected],
        )
        self.assertEqual([row["row_index"] for row in eight_dimensional["sphere_coords"]], selected)
        self.assertEqual([row["id"] for row in eight_dimensional["sphere_coords"]], expected_ids)
        np.testing.assert_array_equal(
            [[row["x"], row["y"], row["z"]] for row in eight_dimensional["sphere_coords"]],
            np.asarray(self.payloads[-1]["points"])[selected],
        )


if __name__ == "__main__":
    unittest.main()
