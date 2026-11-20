"""Exercise uploads, layer toggles, and source-row selection through Streamlit."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_reference_workflow import example_manifest


class Upload(BytesIO):
    def __init__(self, contents, name):
        super().__init__(contents)
        self.name = name


class OverlayUITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        directory = self.root / "embeddings/veltri_peptidebert_sol_v2"
        directory.mkdir(parents=True)
        self.X = np.random.default_rng(7).normal(size=(12, 8)).astype("float32")
        self.meta = pd.DataFrame({"id": ["same"] * 12, "sequence": ["A" * (i + 1) for i in range(12)],
                                  "is_amp": [bool(i % 2) for i in range(12)]})
        np.save(directory / "veltri_embeddings.npy", self.X)
        self.meta.to_csv(directory / "veltri_metadata.csv", index=False)
        self.manifest = example_manifest()
        (directory / "manifest.json").write_text(json.dumps(self.manifest))
        checkpoint = self.root / ".cache/peptidebert/v2.0/sol"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.pt").write_bytes(b"stub checkpoint")
        (checkpoint / "config.yaml").write_text("stub config")
        self.uploads, self.payloads, self.figures, self.projections = {}, [], [], []
        self.event = None
        real_chart = st.plotly_chart

        def chart(figure, *args, **kwargs):
            self.figures.append(figure)
            return real_chart(figure, *args, **kwargs)

        def globe(payload):
            self.payloads.append(payload)
            return self.event

        def projection(ref, query=None, *, n_components, **settings):
            self.projections.append((ref.copy(), None if query is None else query.copy(), settings))
            coords = ref[:, :n_components].copy()
            if query is not None:
                coords = np.vstack([coords, query[:, :n_components] + .1])
            return coords, 0., 0., True

        for name, replacement in (
            ("streamlit.delta_generator.DeltaGenerator.file_uploader", lambda label, **kwargs: self.uploads.get(label)),
            ("streamlit.plotly_chart", chart), ("utils.globe_view.render_globe", globe),
            ("utils.embedding_runtime.get_umap_projection", projection),
            ("utils.reference_projection.get_reference_projection", projection),
        ):
            mock = patch(name, side_effect=replacement)
            mock.start()
            self.addCleanup(mock.stop)
        root = Path(__file__).resolve().parents[1]
        source = (root / "app.py").read_text(encoding="utf-8").replace(
            "PROJECT_ROOT = Path(__file__).resolve().parent", f"PROJECT_ROOT = Path({str(self.root)!r})", 1)
        app_file = self.root / "app.py"
        app_file.write_text(source, encoding="utf-8")
        self.app = AppTest.from_file(str(app_file), default_timeout=30).run()
        self.assert_ok()

    def widget(self, kind, label):
        return next(widget for widget in getattr(self.app, kind) if widget.label == label)

    def assert_ok(self):
        self.assertFalse(self.app.exception, [item.message for item in self.app.exception])
        self.assertFalse(self.app.error, [item.value for item in self.app.error])

    def add_embedding_uploads(self, manifest=True):
        array = BytesIO()
        np.save(array, np.ones((1, 8), dtype="float32"))
        self.uploads["Upload embeddings"] = Upload(array.getvalue(), "new.npy")
        self.uploads["Attach metadata (optional)"] = Upload(b"id,sequence,activity,amount\nsame,WWW,antifungal,2.5\n", "new.csv")
        if manifest:
            self.uploads["Embedding manifest (.json) — required for overlay"] = Upload(
                json.dumps({**self.manifest, "shape": [1, 8]}).encode(), "manifest.json")
        self.widget("selectbox", "Choose workflow").set_value("Project onto Veltri").run()
        self.widget("radio", "Upload format").set_value("Existing embeddings").run()
        if manifest:
            self.widget("button", "Project dataset").click().run()

    def test_holdout_layers_preserve_coordinates_and_selection_when_hidden(self):
        self.widget("selectbox", "Choose workflow").set_value("Holdout demo").run()
        self.widget("number_input", "Held-out peptides").set_value(3).run()
        self.assert_ok()
        original = np.asarray(self.payloads[-1]["points"])
        original_xy = {int(item[0]): (x, y) for trace in self.figures[-1].data
                       for item, x, y in zip(trace.customdata, trace.x, trace.y)}
        self.assertEqual(self.widget("radio", "Projection Type").options, ["UMAP"])
        self.event = {"revision": self.payloads[-1]["selection_revision"], "event_id": "new-row", "selected_indices": [10]}
        self.app.run()
        self.assertEqual(self.app.session_state["selected_indices"], [10])
        self.widget("checkbox", "Show reference data").uncheck().run()
        self.assert_ok()
        np.testing.assert_array_equal(self.payloads[-1]["points"], original)
        self.assertEqual(sum(record["visible"] for record in self.payloads[-1]["metadata"]), 3)
        for trace in self.figures[-1].data:
            for row, x, y in zip(trace.customdata, trace.x, trace.y):
                self.assertEqual((x, y), original_xy[int(row[0])])
        self.widget("checkbox", "Show new data").uncheck().run()
        self.assert_ok()
        self.assertEqual(self.app.session_state["selected_indices"], [10])
        self.assertFalse(any(row["visible"] for row in self.payloads[-1]["metadata"]))
        self.widget("checkbox", "Show reference data").check().run()
        self.widget("checkbox", "Show new data").check().run()
        np.testing.assert_array_equal(self.payloads[-1]["points"], original)
        self.widget("selectbox", "Sphere Dimensions (3D plot)").set_value(8).run()
        self.assert_ok()
        self.assertEqual(self.app.session_state["selected_indices"], [10])
        for reference, query, settings in self.projections[-2:]:
            self.assertEqual(reference.shape, (9, 8))
            self.assertEqual(query.shape, (3, 8))
            self.assertEqual(settings["method"], "UMAP")

    def test_single_embedding_upload_uses_reference_fit_and_activity_in_both_views(self):
        self.add_embedding_uploads()
        self.assert_ok()
        self.assertEqual(len(self.payloads[-1]["points"]), 13)
        new = self.payloads[-1]["metadata"][-1]
        self.assertEqual((new["id"], new["role"], new["source_row_index"]), ("same", "New", 0))
        self.assertIsNone(new["is_amp"])
        self.widget("selectbox", "Color by").set_value("Activity").run()
        self.widget("selectbox", "2D bubble size").set_value("amount").run()
        self.assert_ok()
        self.assertEqual(self.payloads[-1]["metadata"][-1]["activity"], "antifungal")
        new_trace = next(trace for trace in self.figures[-1].data if "antifungal" in trace.name)
        self.assertEqual(new_trace.marker.color, self.payloads[-1]["metadata"][-1]["color"])
        self.assertEqual(list(new_trace.marker.opacity), [1.])

    def test_missing_or_mismatched_manifest_prevents_overlay(self):
        self.add_embedding_uploads(manifest=False)
        before = len(self.projections)
        self.assertTrue(any("manifest" in item.value for item in self.app.info))
        wrong = {**self.manifest, "shape": [1, 8], "pooling": "CLS"}
        self.uploads["Embedding manifest (.json) — required for overlay"] = Upload(json.dumps(wrong).encode(), "wrong.json")
        self.app.run()
        self.assertTrue(any("pooling" in item.value for item in self.app.error))
        self.assertEqual(len(self.projections), before)

    def test_overlay_hash_follows_selected_npz_matrix(self):
        original = BytesIO()
        np.save(original, self.X[:1])
        archive = BytesIO()
        np.savez_compressed(archive, embeddings=self.X[:1], alternate=self.X[1:2])
        manifest = {**self.manifest, "shape": [1, 8],
                    "embeddings_sha256": hashlib.sha256(original.getvalue()).hexdigest()}
        self.uploads["Upload embeddings"] = Upload(archive.getvalue(), "query.npz")
        self.uploads["Embedding manifest (.json) — required for overlay"] = Upload(json.dumps(manifest).encode(), "query.json")
        self.widget("selectbox", "Choose workflow").set_value("Project onto Veltri").run()
        self.widget("radio", "Upload format").set_value("Existing embeddings").run()
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        np.testing.assert_array_equal(self.projections[-1][1], self.X[:1])
        before = len(self.projections)
        self.widget("selectbox", "Embedding array").set_value("alternate").run()
        self.assertTrue(any("SHA-256" in error.value for error in self.app.error))
        self.assertEqual(len(self.projections), before)

    def test_raw_sequence_upload_requires_explicit_generation_and_keeps_labels_optional(self):
        self.uploads["Upload sequences and metadata (.csv)"] = Upload(b"sequence,activity\nWWW,antifungal\n", "sequences.csv")
        with patch("utils.workflow_ui.generated_embeddings", return_value=np.ones((1, 8), dtype="float32")) as generate:
            self.widget("selectbox", "Choose workflow").set_value("Project onto Veltri").run()
            generate.assert_not_called()
            self.widget("button", "Generate embeddings").click().run()
            self.assert_ok()
            self.assertTrue(generate.called)
            self.assertEqual(generate.call_args.args[0], ("WWW",))
            self.assertEqual(generate.call_args.args[-2], self.manifest["checkpoint_sha256"])
            self.assertIsNone(self.payloads[-1]["metadata"][-1]["is_amp"])

    def test_independent_upload_and_custom_holdout_work_without_a_manifest(self):
        array = BytesIO()
        np.save(array, self.X[:8])
        self.uploads["Upload embeddings"] = Upload(array.getvalue(), "custom.npy")
        self.uploads["Attach metadata (optional)"] = Upload(self.meta.iloc[:8].to_csv(index=False).encode(), "custom.csv")
        self.widget("radio", "Dataset").set_value("Upload my data").run()
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        self.assertEqual(len(self.payloads[-1]["points"]), 8)
        self.assertIsNone(self.projections[-1][1])
        self.widget("selectbox", "Choose workflow").set_value("Holdout demo").run()
        self.widget("radio", "Holdout dataset").set_value("Upload my data").run()
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        self.widget("number_input", "Held-out peptides").set_value(2).run()
        self.assert_ok()
        self.assertEqual(self.projections[-1][0].shape, (6, 8))
        self.assertEqual(self.projections[-1][1].shape, (2, 8))

    def test_pair_row_mismatch_stops_before_projection(self):
        self.add_embedding_uploads()
        before = len(self.projections)
        self.uploads["Attach metadata (optional)"] = Upload(b"sequence\nAAA\nCCC\n", "wrong.csv")
        self.app.run()
        self.assertTrue(any("Row mismatch" in error.value for error in self.app.error))
        self.assertEqual(len(self.projections), before)

    def test_sample_maps_numeric_ids_and_labels_only_after_confirmation(self):
        sample = Path(__file__).resolve().parents[1] / "data/examples/sample_embeddings.csv"
        self.uploads["Upload embeddings"] = Upload(sample.read_bytes(), sample.name)
        table = pd.read_csv(sample)
        dimensions = [f"emb_{i}" for i in range(1, 9)]
        before = len(self.projections)
        self.widget("radio", "Dataset").set_value("Upload my data").run()
        self.assertEqual(self.widget("multiselect", "Embedding columns").value, dimensions)
        self.assertEqual(self.widget("selectbox", "ID column").value, "specimen_no")
        self.assertEqual(self.widget("selectbox", "Label column").value, "cohort")
        self.assertIsNone(self.widget("selectbox", "Sequence column").value)
        self.assertEqual(len(self.projections), before)
        self.widget("button", "Project dataset").click().run()
        self.assertEqual(self.widget("selectbox", "Color by").value, "Label")
        self.assert_ok()
        np.testing.assert_array_equal(self.projections[-1][0], table[dimensions].to_numpy(dtype=np.float32))
        self.assertEqual([row["id"] for row in self.payloads[-1]["metadata"]], table.specimen_no.astype(str).tolist())
        self.assertEqual([row["display_label"] for row in self.payloads[-1]["metadata"]], table.cohort.tolist())
        self.assertEqual({trace.name for trace in self.figures[-1].data}, set(table.cohort))
        before = len(self.projections)
        self.widget("multiselect", "Embedding columns").set_value(dimensions[:-1]).run()
        self.assertEqual(len(self.projections), before)
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        self.assertEqual(self.projections[-1][0].shape, (len(table), 7))

    def test_npz_with_shuffled_tsv_matches_distinct_keys_and_blocks_ambiguous_ids(self):
        ids = ["0004", "0002", "0001", "0003"]
        array = BytesIO()
        np.savez_compressed(array, vectors=self.X[:4], catalog=ids)
        self.uploads["Upload embeddings"] = Upload(array.getvalue(), "vectors.npz")
        self.uploads["Attach metadata (optional)"] = Upload(
            b"activity\tsample_id\nfour\t0004\none\t0001\nthree\t0003\ntwo\t0002\n", "details.tsv")
        self.widget("radio", "Dataset").set_value("Upload my data").run()
        self.widget("selectbox", "Embedding array").set_value("vectors").run()
        self.widget("selectbox", "Identifier array (optional)").set_value("catalog").run()
        self.widget("radio", "Match metadata by").set_value("Identifier").run()
        self.assertEqual(self.widget("selectbox", "ID in metadata").value, "sample_id")
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        np.testing.assert_array_equal(self.projections[-1][0], self.X[:4])
        records = self.payloads[-1]["metadata"]
        self.assertEqual([row["id"] for row in records], ids)
        self.assertEqual([row["activity"] for row in records], ["four", "two", "one", "three"])
        before = len(self.projections)
        self.uploads["Attach metadata (optional)"] = Upload(
            b"activity\tsample_id\nfour\t0004\none\t0001\nthree\t0003\nextra\t0004\n", "details.tsv")
        self.app.run()
        self.widget("radio", "Match metadata by").set_value("Identifier").run()
        self.assertTrue(any("Duplicate identifiers" in error.value for error in self.app.error))
        self.assertEqual(len(self.projections), before)

    def test_standalone_numpy_matrix_generates_row_ids_without_metadata(self):
        buffer = BytesIO()
        np.save(buffer, self.X)
        self.uploads["Upload embeddings"] = Upload(buffer.getvalue(), "model_output.npy")
        self.widget("radio", "Dataset").set_value("Upload my data").run()
        self.widget("button", "Project dataset").click().run()
        self.assert_ok()
        np.testing.assert_array_equal(self.projections[-1][0], self.X)
        self.assertEqual([row["id"] for row in self.payloads[-1]["metadata"]], [str(i) for i in range(len(self.X))])


if __name__ == "__main__":
    unittest.main()
