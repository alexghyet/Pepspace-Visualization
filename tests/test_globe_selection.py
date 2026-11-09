import json
import unittest

import numpy as np
import pandas as pd

from utils.globe_view import create_globe_scatter, parse_globe_selection


class GlobeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.points = pd.DataFrame({
            "row_index": np.arange(4),
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.4, 0.3, 0.2, 0.1],
            "z": [-0.1, -0.2, -0.3, -0.4],
            "id": ["amp0", "amp1", "non2", "non3"],
            "sequence": ["AAA", "CCC", "DDD", "EEE"],
            "is_amp": [True, True, False, False],
            "selected": [False, False, False, False],
        })

    def render(self, points=None, selected=(), selected_only=False):
        points = self.points.copy(deep=True) if points is None else points.copy(deep=True)
        points["selected"] = points["row_index"].isin(selected)
        before = points.copy(deep=True)
        payload = create_globe_scatter(
            points, title="Selection regression", point_size=3, opacity=0.8,
            selected_only=selected_only, show_globe=True,
            revision="projection-a", selection_revision="selection-a",
        )
        pd.testing.assert_frame_equal(points, before)
        return payload

    def assert_source_points(self, payload, points=None, selected=()):
        points = self.points if points is None else points
        np.testing.assert_array_equal(payload["points"], points[["x", "y", "z"]])
        self.assertEqual(len(payload["metadata"]), len(points))
        for record, (_, source) in zip(payload["metadata"], points.iterrows()):
            self.assertEqual(record["row_index"], int(source.row_index))
            self.assertEqual(record["id"], source.id)
            self.assertEqual(record["sequence"], source.sequence)
            self.assertIs(record["is_amp"], bool(source.is_amp))
            expected_class = "AMP" if source.is_amp else "Non-AMP"
            self.assertEqual(record["label"], f"{source.id} | {expected_class}")
        self.assertCountEqual(payload["selected_indices"], selected)

    def test_mixed_selection_preserves_both_classes_and_source_rows(self):
        self.assert_source_points(self.render(selected=(1, 2)), selected=(1, 2))

    def test_non_amp_selection_keeps_non_amp_metadata(self):
        self.assert_source_points(self.render(selected=(2, 3)), selected=(2, 3))

    def test_default_filters_selection_and_preserves_unlabeled_status(self):
        points = self.points.copy()
        points["is_amp"] = None
        payload = create_globe_scatter(points, title="Unlabeled", point_size=3, opacity=0.8)
        self.assertIs(payload["selected_only"], True)
        self.assertIsNone(payload["metadata"][0]["is_amp"])
        self.assertEqual(payload["metadata"][0]["label"], "amp0 | Unlabeled")

    def test_orientation_and_hover_options_preserve_dataset_and_selection(self):
        points = self.points.copy()
        points.loc[1, "selected"] = True
        payload = create_globe_scatter(
            points, title="Display controls", point_size=3, opacity=0.8,
            show_axes=False, show_hover_labels=False,
            axis_labels=["Display PC 1", "Display PC 2", "Display PC 3"],
        )
        self.assertIs(payload["show_axes"], False)
        self.assertIs(payload["show_hover_labels"], False)
        self.assertEqual(payload["axis_labels"], ["Display PC 1", "Display PC 2", "Display PC 3"])
        self.assert_source_points(payload, selected=(1,))

    def test_selected_only_never_filters_or_renumbers_the_source_payload(self):
        # ScatterGL filters in the browser, where display indices differ from source IDs.
        for selected in ((), (1, 2), (0, 1, 2, 3)):
            with self.subTest(selected=selected):
                payload = self.render(selected=selected, selected_only=True)
                self.assert_source_points(payload, selected=selected)
                self.assertIs(payload["selected_only"], True)

    def test_projection_coordinates_and_metadata_keep_the_same_order(self):
        # Keep the input layout across every supported globe dimension.
        for dimensions in range(3, 9):
            with self.subTest(dimensions=dimensions):
                points = self.points.iloc[[2, 0, 3, 1]].copy()
                points.loc[:, ["x", "y", "z"]] *= dimensions
                payload = self.render(points=points, selected=(0, 2))
                self.assert_source_points(payload, points=points, selected=(0, 2))

    def test_payload_is_json_serializable_and_keeps_display_settings(self):
        payload = self.render()
        restored = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(restored, payload)
        self.assertEqual(payload["title"], "Selection regression")
        self.assertEqual(payload["point_size"], 3)
        self.assertEqual(payload["opacity"], 0.8)
        self.assertIs(payload["show_globe"], True)
        self.assertIs(payload["selected_only"], False)
        self.assertEqual(payload["revision"], "projection-a")
        self.assertEqual(payload["selection_revision"], "selection-a")
        self.assert_source_points(payload)


class GlobeSelectionEventTests(unittest.TestCase):
    def event(self, selected_indices, **overrides):
        return {
            "revision": "projection-a",
            "selected_indices": selected_indices,
            "event_id": "event-1",
            **overrides,
        }

    def parse(self, event):
        return parse_globe_selection(event, revision="projection-a", row_count=4)

    def test_selection_deduplicates_and_sorts_source_ids(self):
        self.assertEqual(self.parse(self.event([3, 1, 3, 0])), [0, 1, 3])
        self.assertEqual(self.parse(self.event([2], event_id=0)), [2])

    def test_empty_selection_clears_the_previous_selection(self):
        self.assertEqual(self.parse(self.event([])), [])

    def test_invalid_or_stale_events_are_ignored(self):
        invalid_events = [
            None,
            [],
            {},
            self.event([1], revision="old-projection"),
            self.event([1], event_id=""),
            self.event([1], event_id=None),
            self.event([1], event_id=True),
            self.event(None),
            self.event("1"),
            self.event([-1]),
            self.event([4]),
            self.event([True]),
            self.event([1.0]),
            self.event(["1"]),
            self.event([1, None]),
            {"revision": "projection-a", "selected_indices": [1]},
        ]
        for event in invalid_events:
            with self.subTest(event=event):
                self.assertIsNone(self.parse(event))


if __name__ == "__main__":
    unittest.main()
