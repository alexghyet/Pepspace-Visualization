import unittest

import numpy as np

from utils.sphere_projection import (
    centered_3d_coordinates, fit_embeddings_in_sphere, project_embeddings_to_sphere,
)


class GlobeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(42).normal(size=(30, 3)) + [15, 20, 10]

    def test_inside_globe_preserves_every_relative_pairwise_distance(self):
        original = self.points.copy()
        globe = fit_embeddings_in_sphere(self.points)
        distances = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1)
        display_distances = np.linalg.norm(globe[:, None] - globe[None, :], axis=-1)
        scale = display_distances[0, 1] / distances[0, 1]
        np.testing.assert_allclose(display_distances, distances * scale, atol=1e-12)
        np.testing.assert_allclose(globe.mean(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(globe, axis=1).max(), 0.9)
        np.testing.assert_array_equal(self.points, original)

    def test_surface_does_not_depend_on_umap_translation(self):
        surface = project_embeddings_to_sphere(self.points)
        translated = project_embeddings_to_sphere(self.points + [100, 200, -300])
        np.testing.assert_allclose(surface, translated, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(surface, axis=1), 1, atol=1e-12)
        self.assertTrue((surface < 0).any(axis=0).all())
        self.assertTrue((surface > 0).any(axis=0).all())

    def test_centroid_and_coincident_points_are_finite(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
        surface = project_embeddings_to_sphere(points)
        np.testing.assert_array_equal(surface[0], [0, 0, 0])
        for project in (fit_embeddings_in_sphere, project_embeddings_to_sphere):
            result = project(np.ones((5, 3)))
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result, np.zeros((5, 3)))

    def test_higher_dimensions_have_deterministic_finite_display(self):
        for dimensions in range(4, 9):
            points = np.random.default_rng(1).normal(size=(30, dimensions))
            first = centered_3d_coordinates(points)
            np.testing.assert_array_equal(first, centered_3d_coordinates(points))
            self.assertEqual(first.shape, (30, 3))
            self.assertTrue(np.isfinite(project_embeddings_to_sphere(points)).all())
            self.assertLessEqual(np.linalg.norm(fit_embeddings_in_sphere(points), axis=1).max(), 0.90000001)

    def test_invalid_coordinates_are_rejected(self):
        for points in (np.ones((5, 2)), np.ones((2, 3)), np.full((5, 3), np.nan)):
            with self.assertRaises(ValueError):
                fit_embeddings_in_sphere(points)


if __name__ == "__main__":
    unittest.main()
