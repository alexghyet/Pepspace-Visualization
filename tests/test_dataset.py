import unittest

import numpy as np
import pandas as pd

from utils.dataset import (
    build_plot_dataframe, parse_amp_labels, pick_col, validate_embeddings,
    validate_row_alignment,
)


class DatasetValidationTests(unittest.TestCase):
    def test_embeddings_keep_rows_and_values_in_supported_numeric_format(self):
        original = np.arange(24, dtype=np.float64).reshape(4, 6)
        result = validate_embeddings(original)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(original, np.arange(24).reshape(4, 6))

    def test_invalid_embeddings_cannot_be_silently_cast(self):
        invalid = [
            np.ones((2, 3)), np.ones((3, 1)), np.ones(8),
            np.full((3, 2), np.nan), np.full((3, 2), np.inf),
            np.full((3, 2), 1e100), np.ones((3, 2), dtype=complex),
            np.ones((3, 2), dtype=bool), np.full((3, 2), "1"),
            np.ones((3, 2), dtype=object),
        ]
        for points in invalid:
            with self.subTest(dtype=points.dtype, shape=points.shape):
                with self.assertRaises(ValueError):
                    validate_embeddings(points)

    def test_row_mismatch_is_rejected_instead_of_truncating(self):
        embeddings = np.ones((4, 2))
        metadata = pd.DataFrame({"sequence": ["AAA", "CCC", "DDD"]})
        with self.assertRaisesRegex(ValueError, "no rows were truncated"):
            validate_row_alignment(embeddings, metadata)
        self.assertEqual(len(embeddings), 4)
        self.assertEqual(len(metadata), 3)
        validate_row_alignment(embeddings[:3], metadata)

    def test_column_detection_accepts_case_and_outer_whitespace(self):
        metadata = pd.DataFrame({" Sequence ": ["AAA"], "IS_AMP ": [True]})
        self.assertEqual(pick_col(metadata, ["sequence"]), " Sequence ")
        self.assertEqual(pick_col(metadata, ["is_amp"]), "IS_AMP ")
        self.assertIsNone(pick_col(metadata, ["id"]))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            pick_col(pd.DataFrame(columns=["AMP", " amp "]), ["amp"])

    def test_explicit_labels_preserve_false_strings_and_binary_values(self):
        labels = pd.Series([
            True, False, 1, 0, 1.0, 0.0, "TRUE", " False ",
            "1", "0", "1.0", "0.0", "AMP", "non-amp", "Non_AMP", "non AMP",
        ], index=range(20, 36), name="is_amp")
        expected = [True, False] * 6 + [True, False, False, False]
        result = parse_amp_labels(labels)
        self.assertEqual(result.tolist(), expected)
        self.assertTrue(result.index.equals(labels.index))
        self.assertEqual(result.name, labels.name)
        for invalid in (None, np.nan, pd.NA, "unknown", "", 2, -1, "-1", "-True", "-AMP", "maybe"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "metadata rows 2"):
                    parse_amp_labels(pd.Series([True, invalid]))

    def test_source_rows_remain_distinct_with_duplicate_ids_and_custom_index(self):
        metadata = pd.DataFrame({
            "name": ["same", "same", None],
            "seq": ["AAA", None, "CCC"],
            "status": ["AMP", "non-AMP", "False"],
        }, index=[4, 4, 10])
        before = metadata.copy(deep=True)
        frame = build_plot_dataframe(metadata, id_col="name", seq_col="seq", amp_col="status")
        self.assertEqual(frame.row_index.tolist(), [0, 1, 2])
        self.assertEqual(frame.id.tolist(), ["same", "same", "2"])
        self.assertEqual(frame.sequence.tolist(), ["AAA", "", "CCC"])
        self.assertEqual(frame.is_amp.tolist(), [True, False, False])
        self.assertTrue(frame.attrs["amp_labels_available"])
        pd.testing.assert_frame_equal(metadata, before)

    def test_optional_fields_are_explicit_and_empty_metadata_fails(self):
        frame = build_plot_dataframe(pd.DataFrame({"value": [1, 2, 3]}))
        self.assertEqual(frame.id.tolist(), ["0", "1", "2"])
        self.assertEqual(frame.sequence.tolist(), ["", "", ""])
        self.assertFalse(frame.attrs["amp_labels_available"])
        self.assertEqual(frame.is_amp.tolist(), [None, None, None])
        with self.assertRaisesRegex(ValueError, "at least one column"):
            build_plot_dataframe(pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_plot_dataframe(pd.DataFrame({"value": [1]}), seq_col="missing")


if __name__ == "__main__":
    unittest.main()
