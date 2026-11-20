from io import BytesIO
import unittest

import numpy as np
import pandas as pd

from utils.embedding_upload import (
    array_identifiers, attach_metadata, npz_member_bytes, npz_members, read_npy,
    read_table, split_embedding_table, suggested_dimensions,
)


class EmbeddingUploadTests(unittest.TestCase):
    def test_table_formats_preserve_ids_feature_order_and_nonfeature_columns(self):
        table = pd.DataFrame({"registration": ["0007", "NA", "0003"], "dim_2": [1., 2., 3.],
                              "measurement": [7, 2, 4], "dim_1": [-3, -2, -1]})
        for kind, contents in (
            ("csv", table.to_csv(index=False).encode()),
            ("tsv", table.to_csv(index=False, sep="\t").encode()),
            ("parquet", table.to_parquet(index=False)),
        ):
            with self.subTest(kind=kind):
                frame = read_table(contents, kind)
                self.assertEqual(suggested_dimensions(frame.columns), ["dim_2", "dim_1"])
                X, metadata = split_embedding_table(frame, ["dim_1", "dim_2"])
                np.testing.assert_array_equal(X, table[["dim_2", "dim_1"]].to_numpy())
                self.assertEqual(metadata.columns.tolist(), ["registration", "measurement"])
                self.assertEqual(metadata.registration.tolist(), ["0007", "NA", "0003"])

    def test_headerless_tables_keep_first_vector_and_reject_ambiguous_headers(self):
        frame = read_table(b"1\t2\n3\t4\n5\t6\n", "tsv", has_header=False)
        self.assertEqual(suggested_dimensions(frame.columns), [])
        X, metadata = split_embedding_table(frame, frame.columns)
        np.testing.assert_array_equal(X, [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(metadata.shape, (3, 0))
        for header in ("id,id", "ID, id "):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                read_table((header + "\n1,2\n").encode(), "csv")

    def test_numeric_validation_applies_only_to_selected_dimensions(self):
        for bad in ("", "text", "inf", "1e90", "False"):
            with self.subTest(bad=bad):
                frame = read_table(f"id,v1,v2\na,1,2\nb,3,{bad}\nc,4,5\n".encode(), "csv")
                with self.assertRaisesRegex(ValueError, "v2"):
                    split_embedding_table(frame, ["v1", "v2"])
        frame = pd.DataFrame({"a": [1, 2, 3], "flag": [False, True, False]})
        with self.assertRaisesRegex(ValueError, "flag"):
            split_embedding_table(read_table(frame.to_parquet(index=False), "parquet"), frame.columns)

    def test_identifier_match_preserves_original_rows_and_both_files_fields(self):
        inline = pd.DataFrame({"code": ["01", "1", "NA"], "batch": ["x", "y", "z"]}, index=[2, 1, 0])
        attached = pd.DataFrame({"source": ["NA", "01", "1"], "Batch": [3, 1, 2]})
        result = attach_metadata(inline, attached, "code", "source")
        self.assertEqual(result.code.tolist(), ["01", "1", "NA"])
        self.assertEqual(result.batch.tolist(), ["x", "y", "z"])
        self.assertEqual(result["attached.Batch"].tolist(), [1, 2, 3])
        self.assertEqual(result.index.tolist(), [0, 1, 2])
        self.assertEqual(attach_metadata(inline, attached).source.tolist(), ["NA", "01", "1"])

    def test_identifier_and_position_matches_reject_ambiguous_or_unpaired_rows(self):
        for left, right, message in (
            (["A", "A"], ["A", "B"], "Duplicate identifiers in embedding"),
            (["A", "B"], ["A", " A "], "Duplicate identifiers in metadata"),
            (["A", " "], ["A", "B"], "Missing identifiers in embedding"),
            (["A", "B"], ["A", None], "Missing identifiers in metadata"),
            (["A", "B"], ["C", "A"], "missing from metadata.*only in metadata"),
            (["A"], ["A", "B"], "only in metadata"),
        ):
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, message):
                    attach_metadata(pd.DataFrame({"x": left}), pd.DataFrame({"y": right}), "x", "y")
        with self.assertRaisesRegex(ValueError, "Row mismatch"):
            attach_metadata(pd.DataFrame(index=range(3)), pd.DataFrame({"code": ["A", "B"]}))

    def test_npz_only_loads_chosen_members_and_rejects_pickled_arrays(self):
        output = BytesIO()
        X = np.arange(12).reshape(3, 4)
        np.savez_compressed(output, X=X, ids=[b"004", b"002", b"001"], objects=np.array([{}], dtype=object))
        contents = output.getvalue()
        self.assertEqual(npz_members(contents), ["X", "ids", "objects"])
        np.testing.assert_array_equal(read_npy(npz_member_bytes(contents, "X", 1024)), X)
        ids = npz_member_bytes(contents, "ids", 1024)
        self.assertEqual(array_identifiers(ids, "ids", 3).ids.tolist(), ["004", "002", "001"])
        with self.assertRaisesRegex(ValueError, "one text or numeric ID"):
            array_identifiers(ids, "ids", 4)
        with self.assertRaisesRegex(ValueError, "pickle"):
            read_npy(npz_member_bytes(contents, "objects", 1024))
        with self.assertRaisesRegex(ValueError, "size limit"):
            npz_member_bytes(contents, "X", 16)
