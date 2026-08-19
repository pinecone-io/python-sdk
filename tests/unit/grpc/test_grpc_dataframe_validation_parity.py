"""The gRPC DataFrame path must reject what VectorFactory rejects.

That path hands array values straight to the Rust layer instead of building
Vector objects, which means it also skips VectorFactory's validation. These
tests are the guarantee that skipping the conversion did not quietly widen what
the SDK accepts: for the same malformed row, both routes must raise the same
exception type.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.vector_factory import VectorFactory, validate_vector_dict
from pinecone.grpc import GrpcIndex

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


def _raised_by(fn: Any, *args: Any) -> type[BaseException] | None:
    try:
        fn(*args)
    except BaseException as exc:
        return type(exc)
    return None


_MALFORMED_ROWS: list[dict[str, Any]] = [
    {"id": 42, "values": [0.1]},
    {"id": b"bytes", "values": [0.1]},
    {"id": None, "values": [0.1]},
    {"id": "café", "values": [0.1]},
    {"id": "nul\x00byte", "values": [0.1]},
    {"id": "v1", "values": []},
    {"id": "v1", "values": np.array([], dtype=np.float32)},
    {"id": "v1", "values": np.float32(0.5).reshape(())},
    {"id": "v1", "values": [0.1], "metadata": 1.5},
    {"id": "v1", "values": [0.1], "metadata": "not-a-dict"},
    {"id": "v1", "values": [0.1], "metadata": [1, 2]},
    {"id": "v1", "values": [0.1], "sparse_values": "not-a-dict"},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": [0]}},
    {"id": "v1", "values": [0.1], "sparse_values": {"values": [0.5]}},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": [0, 1], "values": [0.5]}},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": ["x"], "values": [0.5]}},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": [0], "values": ["x"]}},
]

_WELL_FORMED_ROWS: list[dict[str, Any]] = [
    {"id": "v1", "values": [0.1, 0.2]},
    {"id": "v1", "values": np.array([0.1, 0.2], dtype=np.float32)},
    {"id": "v1", "values": np.array([0.1], dtype=np.float64)},
    {"id": "v1", "values": [0.1], "metadata": {}},
    {"id": "v1", "values": [0.1], "metadata": {"genre": "rock", "year": 2024}},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": [0, 2], "values": [0.5, 0.8]}},
    {"id": "v1", "values": [0.1], "sparse_values": {"indices": [], "values": []}},
]


class TestValidationParity:
    @pytest.mark.parametrize("row", _MALFORMED_ROWS, ids=lambda r: repr(r)[:60])
    def test_same_exception_type_as_vector_factory(self, row: dict[str, Any]) -> None:
        from_factory = _raised_by(VectorFactory.build, dict(row))
        from_validator = _raised_by(validate_vector_dict, dict(row))

        assert from_factory is not None, "fixture is not actually malformed"
        assert from_validator is from_factory

    @pytest.mark.parametrize("row", _WELL_FORMED_ROWS, ids=lambda r: repr(r)[:60])
    def test_well_formed_rows_are_accepted_by_both(self, row: dict[str, Any]) -> None:
        assert _raised_by(VectorFactory.build, dict(row)) is None
        assert _raised_by(validate_vector_dict, dict(row)) is None


class TestParityThroughTheDataFrameMethod:
    """Same parity, reached the way a caller reaches it."""

    @pytest.mark.parametrize("row", _MALFORMED_ROWS, ids=lambda r: repr(r)[:60])
    def test_upsert_from_dataframe_raises_what_the_factory_would(self, row: dict[str, Any]) -> None:
        expected = _raised_by(VectorFactory.build, dict(row))
        assert expected is not None

        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 0}
        index = _grpc_index(mock_channel)
        df = pd.DataFrame([row])

        with pytest.raises(expected):
            index.upsert_from_dataframe(df, show_progress=False)

        assert mock_channel.upsert.call_count == 0, (
            "a malformed row must be rejected before anything is sent"
        )


class TestValidationDoesNotMaterializeValues:
    def test_numpy_values_are_not_converted(self) -> None:
        """The whole point of the bypass: the array reaches the channel as-is."""
        arr = np.arange(8, dtype=np.float32)
        mock_channel = MagicMock()
        mock_channel.upsert.return_value = {"upserted_count": 1}
        index = _grpc_index(mock_channel)
        df = pd.DataFrame({"id": ["v1"], "values": [arr]})

        index.upsert_from_dataframe(df, show_progress=False)

        (sent,) = mock_channel.upsert.call_args[0][0]
        assert sent["values"] is arr


_ids = st.text(min_size=1, max_size=8)
_values = st.lists(st.floats(-10, 10, allow_nan=False), max_size=4)
_metadata = st.one_of(
    st.none(),
    st.dictionaries(st.sampled_from(["a", "b"]), st.integers(-5, 5), max_size=2),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(max_size=3),
)


class TestValidationParityProperties:
    @settings(max_examples=400, deadline=None)
    @given(id_=_ids, values=_values, metadata=_metadata)
    def test_agreement_over_generated_rows(
        self, id_: str, values: list[float], metadata: object
    ) -> None:
        row: dict[str, Any] = {"id": id_, "values": values}
        if metadata is not None:
            row["metadata"] = metadata

        assert _raised_by(validate_vector_dict, dict(row)) is _raised_by(
            VectorFactory.build, dict(row)
        )

    @settings(max_examples=200, deadline=None)
    @given(
        id_=_ids,
        values=st.lists(st.floats(-10, 10, allow_nan=False), max_size=4),
        dtype=st.sampled_from(["float32", "float64"]),
    )
    def test_numpy_and_list_values_agree(self, id_: str, values: list[float], dtype: str) -> None:
        """An array and the list it would convert to must be judged the same."""
        as_list: dict[str, Any] = {"id": id_, "values": values}
        as_array: dict[str, Any] = {"id": id_, "values": np.array(values, dtype=dtype)}

        assert _raised_by(validate_vector_dict, as_array) is _raised_by(
            validate_vector_dict, as_list
        )
