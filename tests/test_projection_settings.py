import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import precompute_projections

from utils.projection_settings import (
    load_optuna_parameters, projection_arguments, projection_defaults,
)


class ProjectionSettingsTests(unittest.TestCase):
    def test_same_settings_produce_identical_app_and_cli_arguments(self):
        best = {"method": "densMAP 5d", "n_neighbors": 300, "min_dist": 0.02, "dens_lambda": 0.4}
        settings = projection_defaults(400, best)
        app_arguments = projection_arguments(settings)
        cli_arguments = projection_arguments(projection_defaults(400, best.copy()))
        self.assertEqual(app_arguments, cli_arguments)
        self.assertEqual(app_arguments["n_neighbors"], 200)
        self.assertEqual(app_arguments["method"], "densMAP")
        self.assertEqual(app_arguments["seed"], 42)
        self.assertEqual(app_arguments["metric"], "cosine")
        self.assertEqual(app_arguments["dens_lambda"], 0.4)
        self.assertEqual(app_arguments["dens_frac"], 0.3)
        self.assertEqual(app_arguments["dens_var_shift"], 0.1)

    def test_neighbor_bounds_and_umap_defaults_are_shared(self):
        self.assertEqual(projection_defaults(3)["n_neighbors"], 2)
        self.assertEqual(projection_defaults(20, {"n_neighbors": -1})["n_neighbors"], 2)
        self.assertEqual(projection_defaults(20, {"n_neighbors": 100})["n_neighbors"], 19)
        arguments = projection_arguments(projection_defaults(20))
        self.assertEqual(arguments["n_neighbors"], 5)
        self.assertEqual(arguments["method"], "UMAP")
        for name in ("dens_lambda", "dens_frac", "dens_var_shift"):
            self.assertIsNone(arguments[name])

    def test_bad_optimization_values_fail_before_fitting(self):
        for best in (
            {"method": "PCA"}, {"n_neighbors": 2.5}, {"n_neighbors": True},
            {"min_dist": float("nan")}, {"min_dist": -0.01}, {"min_dist": 2},
            {"dens_lambda": -1}, {"dens_frac": 1.1}, {"dens_var_shift": 0},
        ):
            with self.subTest(best=best):
                with self.assertRaises(ValueError):
                    projection_defaults(20, best)
        with self.assertRaisesRegex(ValueError, "three samples"):
            projection_defaults(2)

    def test_missing_and_malformed_optimization_files_have_clear_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "3d_params.json"
            self.assertEqual(load_optuna_parameters(path), {})
            expected = {"n_neighbors": 8, "min_dist": 0.01}
            path.write_text(json.dumps({"best_params": expected}), encoding="utf-8")
            self.assertEqual(load_optuna_parameters(path), expected)
            for contents in ("", "{}", '{"best_params": []}', "not json"):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "3d_params.json"):
                    load_optuna_parameters(path)

    def test_cli_dimension_switch_keeps_shared_baseline_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "optuna_results").mkdir()
            (root / "optuna_results/3d_params.json").write_text(
                json.dumps({"best_params": {"n_neighbors": 4, "min_dist": 0.02}}), encoding="utf-8",
            )
            (root / "optuna_results/5d_params.json").write_text(
                json.dumps({"best_params": {"n_neighbors": 12, "min_dist": 0.8}}), encoding="utf-8",
            )
            embeddings = root / "points.npy"
            np.save(embeddings, np.arange(160).reshape(20, 8))
            with patch.object(precompute_projections, "PROJECT_ROOT", root), patch(
                "sys.argv", ["precompute_projections.py", "--dimensions", "5", "--embeddings", str(embeddings)],
            ), patch.object(precompute_projections, "get_umap_projection", return_value=(None, 0, 0, True)) as fit, redirect_stdout(StringIO()):
                precompute_projections.main()
            self.assertEqual([call.kwargs["n_components"] for call in fit.call_args_list], [2, 5])
            for call in fit.call_args_list:
                self.assertEqual(call.kwargs["n_neighbors"], 4)
                self.assertEqual(call.kwargs["min_dist"], 0.02)
                self.assertEqual(call.kwargs["seed"], 42)


if __name__ == "__main__":
    unittest.main()
