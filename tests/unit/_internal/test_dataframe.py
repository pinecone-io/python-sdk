"""Unit tests for the shared DataFrame record extractor."""

from __future__ import annotations

import pytest

from pinecone._internal.dataframe import extract_records
from pinecone.errors.exceptions import PineconeValueError

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


class TestRequiredColumns:
    def test_id_and_values_only(self) -> None:
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1, 0.2], [0.3, 0.4]]})

        assert extract_records(df) == [
            {"id": "v1", "values": [0.1, 0.2]},
            {"id": "v2", "values": [0.3, 0.4]},
        ]

    def test_empty_frame_yields_no_records(self) -> None:
        df = pd.DataFrame({"id": [], "values": []})

        assert extract_records(df) == []

    def test_missing_id_column_names_the_schema(self) -> None:
        df = pd.DataFrame({"values": [[0.1]], "metadata": [{}]})

        with pytest.raises(PineconeValueError) as excinfo:
            extract_records(df)

        message = str(excinfo.value)
        assert "['id']" in message
        assert "upsert_from_dataframe" in message
        assert "values" in message and "metadata" in message

    def test_missing_values_column_names_the_schema(self) -> None:
        df = pd.DataFrame({"id": ["v1"]})

        with pytest.raises(PineconeValueError, match=r"\['values'\]"):
            extract_records(df)

    def test_both_missing_are_reported_together(self) -> None:
        df = pd.DataFrame({"vector": [[0.1]]})

        with pytest.raises(PineconeValueError, match=r"\['id', 'values'\]"):
            extract_records(df)


class TestOptionalColumns:
    def test_metadata_and_sparse_included_when_present(self) -> None:
        sparse = {"indices": [0, 2], "values": [0.5, 0.8]}
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "values": [[0.1, 0.2]],
                "metadata": [{"genre": "rock"}],
                "sparse_values": [sparse],
            }
        )

        (record,) = extract_records(df)

        assert record["metadata"] == {"genre": "rock"}
        assert record["sparse_values"] == sparse

    def test_absent_columns_are_not_invented(self) -> None:
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})

        (record,) = extract_records(df)

        assert set(record) == {"id", "values"}


class TestUnrecognizedColumns:
    def test_extra_columns_are_ignored_not_rejected(self) -> None:
        """Long-standing behavior on both transports: only the four columns count."""
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]], "colour": ["red"]})

        (record,) = extract_records(df)

        assert set(record) == {"id", "values"}


class TestMissingValues:
    """A frame built from row dicts leaves NaN where a key was omitted."""

    def test_nan_metadata_cell_is_treated_as_absent(self) -> None:
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "metadata": {"genre": "rock"}},
                {"id": "v2", "values": [0.2]},
            ]
        )
        assert df["metadata"].isna()[1]

        first, second = extract_records(df)

        assert first["metadata"] == {"genre": "rock"}
        assert "metadata" not in second

    def test_nan_sparse_values_cell_is_treated_as_absent(self) -> None:
        sparse = {"indices": [1], "values": [0.5]}
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "sparse_values": sparse},
                {"id": "v2", "values": [0.2]},
            ]
        )

        first, second = extract_records(df)

        assert first["sparse_values"] == sparse
        assert "sparse_values" not in second

    @pytest.mark.parametrize(
        "missing",
        [None, float("nan"), np.float64("nan"), np.float32("nan"), pd.NA, pd.NaT],
        ids=["none", "float-nan", "float64-nan", "float32-nan", "pd-NA", "pd-NaT"],
    )
    def test_every_missing_marker_is_treated_as_absent(self, missing: object) -> None:
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]], "metadata": [missing]})

        (record,) = extract_records(df)

        assert "metadata" not in record

    def test_falsy_but_present_metadata_is_kept(self) -> None:
        """An empty dict is data, not a missing value."""
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]], "metadata": [{}]})

        (record,) = extract_records(df)

        assert record["metadata"] == {}


class TestNoDtypeCoercion:
    def test_numpy_values_pass_through_without_conversion(self) -> None:
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        df = pd.DataFrame({"id": ["v1"], "values": [arr]})

        (record,) = extract_records(df)

        assert record["values"] is arr

    def test_ids_stay_strings_alongside_a_numeric_column(self) -> None:
        """iterrows() would coerce the row to one dtype; columnar access does not."""
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]], "rank": [1, 2]})

        records = extract_records(df)

        assert [r["id"] for r in records] == ["v1", "v2"]
        assert all(isinstance(r["id"], str) for r in records)
