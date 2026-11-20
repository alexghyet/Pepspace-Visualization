import copy
import hashlib
from io import BytesIO
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
from joblib import Memory
import numpy as np
import pandas as pd

from utils import reference_projection as projection
from utils.dataset import validate_embeddings
from utils.workflow import check_compatibility, holdout_indices, plot_metadata, combine_workflow, read_manifest
from utils.structure_view import read_pdb_trace


class FakeReducer:
    fits = []
    transforms = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, reference):
        self.reference = reference.copy()
        self.embedding_ = reference[:, :self.parameters["n_components"]].copy()
        self.fits.append(reference.copy())
        return self

    def transform(self, query):
        self.transforms.append(query.copy())
        return query[:, :self.parameters["n_components"]] + .125


def example_manifest():
    return dict(model="ChakradharG/PeptideBERT", task="sol", checkpoint_sha256="a" * 64,
                pooling="residue mean", encoding="no special tokens", truncation=False, shape=[12, 8])


class ReferenceWorkflowTests(unittest.TestCase):
    def test_reference_is_fitted_once_queries_never_enter_fit_and_model_survives_export(self):
        FakeReducer.fits, FakeReducer.transforms = [], []
        module = types.ModuleType("umap.umap_")
        module.UMAP = FakeReducer
        package = types.ModuleType("umap")
        package.umap_ = module
        reference = np.random.default_rng(42).normal(size=(20, 8)).astype("float32")
        query = np.full((1, 8), 50, dtype="float32")
        settings = dict(n_neighbors=5, min_dist=.1, metric="cosine", seed=42, method="UMAP")
        with tempfile.TemporaryDirectory() as directory:
            cache = Memory(directory, verbose=0)
            with patch.object(projection, "fit_reference", cache.cache(projection.fit_reference.func)), patch.object(
                projection, "transform_query", cache.cache(projection.transform_query.func)
            ), patch.dict("sys.modules", {"umap": package, "umap.umap_": module}):
                first = projection.get_reference_projection(reference, query, n_components=3, **settings)
                repeated = projection.get_reference_projection(reference, query.copy(), n_components=3, **settings)
                changed = projection.get_reference_projection(reference, query + 1, n_components=3, **settings)
                self.assertFalse(first[3])
                self.assertTrue(repeated[3])
                self.assertEqual(len(FakeReducer.fits), 1)
                self.assertEqual(len(FakeReducer.transforms), 2)
                np.testing.assert_array_equal(FakeReducer.fits[0], reference)
                np.testing.assert_array_equal(first[0][:-1], changed[0][:-1])
                bundle = joblib.load(BytesIO(projection.reference_bundle(reference, (2, 3), settings, {"test": True})))
                np.testing.assert_array_equal(bundle["models"][3]["reducer"].transform(query), first[0][-1:])
                self.assertIn("display", bundle["models"][3])
                self.assertEqual(bundle["provenance"], {"test": True})

    def test_overlay_rejects_other_models_and_dimensions_even_when_numeric(self):
        ref = example_manifest()
        check_compatibility(ref, copy.deepcopy(ref), 8)
        for field in ("checkpoint_sha256", "pooling", "encoding", "task", "truncation"):
            bad = {**ref, field: "different"}
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_compatibility(ref, bad, 8)
        for bad in (None, {}, {**ref, "shape": [4, 9]}):
            with self.assertRaises(ValueError):
                check_compatibility(ref, bad, 8)

    def test_single_query_is_allowed_but_a_fit_still_requires_three_rows(self):
        one = np.ones((1, 8))
        self.assertEqual(validate_embeddings(one, min_rows=1).shape, (1, 8))
        with self.assertRaises(ValueError):
            validate_embeddings(one)
        with self.assertRaisesRegex(ValueError, "densMAP"):
            projection.reference_parameters(3, 5, .1, "cosine", 42, "densMAP")

    def test_manifest_hash_and_attention_configuration_are_checked(self):
        import json

        manifest = {**example_manifest(), "embeddings_sha256": hashlib.sha256(b"original").hexdigest()}
        self.assertEqual(read_manifest(json.dumps(manifest), b"original"), manifest)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_manifest(json.dumps(manifest), b"changed")
        ref = {**manifest, "config": {"vocab_size": 25, "network": {"hidden_size": 8, "attn_heads": 2}}}
        other = copy.deepcopy(ref)
        other["config"]["network"]["attn_heads"] = 4
        with self.assertRaisesRegex(ValueError, "encoder configuration"):
            check_compatibility(ref, other, 8)

    def test_holdout_is_reproducible_and_duplicate_sequences_do_not_leak(self):
        meta = pd.DataFrame({"sequence": ["AAA", " AAA ", "aaa", "CCC", "DDD", "EEE", "FFF", "GGG"]})
        ref, query = holdout_indices(meta, 2, 42)
        ref2, query2 = holdout_indices(meta, 2, 42)
        np.testing.assert_array_equal(ref, ref2)
        np.testing.assert_array_equal(query, query2)
        normalized = meta.sequence.str.strip().str.upper()
        self.assertFalse(set(normalized.iloc[ref]) & set(normalized.iloc[query]))
        self.assertEqual(sorted([*ref, *query]), list(range(len(meta))))

    def test_reference_pca_center_and_scale_are_stable_for_extreme_queries(self):
        reference = np.random.default_rng(1).normal(size=(15, 6))
        transform = projection.fit_display(reference)
        original = projection.apply_display(reference, transform)
        combined = np.vstack([reference, np.full((2, 6), 1000)])
        transformed = projection.apply_display(combined, transform)
        np.testing.assert_allclose(original, transformed[:len(reference)])
        self.assertAlmostEqual(np.linalg.norm(original, axis=1).max(), .9)
        self.assertGreater(np.linalg.norm(transformed[-1]), 1)

    def test_duplicate_ids_still_have_distinct_dataset_and_source_rows(self):
        reference_meta = pd.DataFrame({"id": ["same"] * 3, "sequence": ["AAA"] * 3, "is_amp": [1, 0, 1]})
        query_meta = pd.DataFrame({"id": ["same"], "sequence": ["CCC"], "activity": ["antifungal"]})
        ref = plot_metadata(reference_meta, role="Reference")
        query = plot_metadata(query_meta, name="User", role="New")
        result = combine_workflow("overlay", np.ones((3, 8)), np.zeros((1, 8)), reference_meta, query_meta, ref, query, {}, {})
        self.assertEqual(result.df.row_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.df.source_row_index.tolist(), [0, 1, 2, 0])
        self.assertEqual(result.df.iloc[-1].class_label, "Unlabeled")
        self.assertEqual(result.df.iloc[-1].activity, "antifungal")

    def test_generic_binary_labels_do_not_imply_amp_status(self):
        generic = plot_metadata(pd.DataFrame({"class": ["0", "1", "0"]}))
        self.assertEqual(generic.label.tolist(), ["0", "1", "0"])
        self.assertTrue(generic.is_amp.isna().all())
        named = plot_metadata(pd.DataFrame({"label": ["amp", "Non-AMP"]}))
        self.assertEqual(named.is_amp.tolist(), [True, False])

    def test_pdb_parser_uses_first_model_and_only_alpha_carbons(self):
        atom = "ATOM      1  CA  ALA A   1      10.000  11.000  12.000  1.00 20.00           C"
        trace = read_pdb_trace(atom + "\nENDMDL\n" + atom.replace("10.000", "99.000"))
        self.assertEqual(trace["A"][0][0], [10., 11., 12.])
        with self.assertRaises(ValueError):
            read_pdb_trace("not a PDB")


if __name__ == "__main__":
    unittest.main()
