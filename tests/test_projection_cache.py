import tempfile
import types
import unittest
from unittest.mock import patch

from joblib import Memory
import numpy as np

from utils import embedding_runtime as runtime


class ProjectionCacheTests(unittest.TestCase):
    def test_cache_reuses_exact_result_and_invalidates_on_inputs_settings_versions(self):
        calls = []
        module = types.ModuleType("umap.umap_")

        class FakeUMAP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_transform(self, X):
                calls.append(self.kwargs)
                return X[:, :self.kwargs["n_components"]].copy()

        module.UMAP = FakeUMAP
        package = types.ModuleType("umap")
        package.umap_ = module
        with tempfile.TemporaryDirectory() as directory:
            cache = Memory(directory, verbose=0).cache(runtime._fit_umap.func)
            with patch.object(runtime, "_fit_umap", cache), patch.dict(
                "sys.modules", {"umap": package, "umap.umap_": module},
            ):
                points = np.arange(90, dtype=np.float32).reshape(15, 6)
                arguments = dict(n_neighbors=5, min_dist=0.001, n_components=3, metric="cosine", seed=42, method="UMAP")
                first = runtime.get_umap_projection(points, **arguments)
                second = runtime.get_umap_projection(points.copy(), **arguments)
                np.testing.assert_array_equal(first[0], second[0])
                self.assertFalse(first[3])
                self.assertTrue(second[3])
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["random_state"], 42)
                self.assertNotIn("n_epochs", calls[0])
                changed = points.copy()
                changed[-1, -1] += 1
                self.assertFalse(runtime.get_umap_projection(changed, **arguments)[3])
                self.assertFalse(runtime.get_umap_projection(points, **{**arguments, "min_dist": 0.1})[3])
                with patch.object(runtime, "dependency_versions", return_value=(("numpy", "changed"),)):
                    self.assertFalse(runtime.get_umap_projection(points, **arguments)[3])
                self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
