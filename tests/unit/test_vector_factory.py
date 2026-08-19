"""Tests for VectorFactory — vector input format parsing."""

from __future__ import annotations

import pytest

from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector


class TestVectorPassthrough:
    """unified-vecfmt-0001: Accept vectors as dataclass objects."""

    def test_vector_object_returned_as_is(self) -> None:
        v = Vector(id="v1", values=[0.1, 0.2])
        result = VectorFactory.build(v)
        assert result is v

    def test_vector_with_metadata_passthrough(self) -> None:
        v = Vector(id="v1", values=[0.1], metadata={"k": "v"})
        result = VectorFactory.build(v)
        assert result is v

    def test_vector_with_sparse_passthrough(self) -> None:
        sv = SparseValues(indices=[0], values=[1.0])
        v = Vector(id="v1", values=[], sparse_values=sv)
        result = VectorFactory.build(v)
        assert result is v

    def test_vector_passthrough_empty_rejected(self) -> None:
        # Vector's __post_init__ validates at construction time, so callers
        # cannot build a fully-empty vector to pass through the factory.
        with pytest.raises(ValueError, match="values or sparse_values"):
            VectorFactory.build(Vector(id="v1"))

    def test_vector_passthrough_sparse_only_accepted(self) -> None:
        sv = SparseValues(indices=[0], values=[1.0])
        v = Vector(id="v1", sparse_values=sv)
        result = VectorFactory.build(v)
        assert result is v


class TestTupleFormat:
    """unified-vecfmt-0002, 0003, 0006, 0007, 0008."""

    def test_two_tuple(self) -> None:
        result = VectorFactory.build(("id1", [0.1, 0.2]))
        assert isinstance(result, Vector)
        assert result.id == "id1"
        assert result.values == [0.1, 0.2]
        assert result.metadata is None
        assert result.sparse_values is None

    def test_three_tuple(self) -> None:
        result = VectorFactory.build(("id1", [0.1, 0.2], {"key": "val"}))
        assert result.id == "id1"
        assert result.values == [0.1, 0.2]
        assert result.metadata == {"key": "val"}

    def test_three_tuple_none_metadata(self) -> None:
        result = VectorFactory.build(("id1", [0.1], None))
        assert result.metadata is None

    def test_values_converted_to_list(self) -> None:
        """unified-vecfmt-0006: iterable values auto-converted to list."""
        result = VectorFactory.build(("id1", (0.1, 0.2, 0.3)))
        assert result.values == [0.1, 0.2, 0.3]
        assert isinstance(result.values, list)

    def test_four_tuple_rejected(self) -> None:
        """unified-vecfmt-0007/0008: no 4-element tuples."""
        with pytest.raises(PineconeValueError, match="2 or 3 elements"):
            VectorFactory.build(("id", [0.1], {}, "extra"))

    def test_one_tuple_rejected(self) -> None:
        """unified-vecfmt-0008: 1-element tuple rejected."""
        with pytest.raises(PineconeValueError, match="2 or 3 elements"):
            VectorFactory.build(("id",))

    def test_empty_tuple_rejected(self) -> None:
        with pytest.raises(PineconeValueError, match="2 or 3 elements"):
            VectorFactory.build(())


class TestDictFormat:
    """unified-vecfmt-0004, 0005, 0009, 0010, 0011."""

    def test_dict_with_id_and_values(self) -> None:
        """unified-vecfmt-0004."""
        result = VectorFactory.build({"id": "v1", "values": [0.1, 0.2]})
        assert result.id == "v1"
        assert result.values == [0.1, 0.2]

    def test_dict_with_sparse_only(self) -> None:
        """unified-vecfmt-0005: sparse without dense."""
        result = VectorFactory.build(
            {
                "id": "v1",
                "sparse_values": {"indices": [0, 2], "values": [1.0, 0.5]},
            }
        )
        assert result.id == "v1"
        assert result.values == []
        assert result.sparse_values is not None
        assert result.sparse_values.indices == [0, 2]
        assert result.sparse_values.values == [1.0, 0.5]

    def test_dict_with_all_fields(self) -> None:
        result = VectorFactory.build(
            {
                "id": "v1",
                "values": [0.1],
                "sparse_values": {"indices": [0], "values": [1.0]},
                "metadata": {"key": "val"},
            }
        )
        assert result.id == "v1"
        assert result.values == [0.1]
        assert result.sparse_values is not None
        assert result.metadata == {"key": "val"}

    def test_dict_values_converted_to_list(self) -> None:
        """unified-vecfmt-0006: iterable values converted."""
        result = VectorFactory.build({"id": "v1", "values": (0.1, 0.2)})
        assert isinstance(result.values, list)

    def test_dict_missing_id_rejected(self) -> None:
        """unified-vecfmt-0009."""
        with pytest.raises(PineconeValueError, match="'id' key"):
            VectorFactory.build({"values": [0.1]})

    def test_dict_extra_keys_rejected(self) -> None:
        """unified-vecfmt-0010."""
        with pytest.raises(PineconeValueError, match="unrecognized keys"):
            VectorFactory.build({"id": "v1", "values": [0.1], "extra": 42})

    def test_dict_metadata_non_dict_rejected(self) -> None:
        """unified-vecfmt-0011."""
        with pytest.raises(PineconeTypeError, match="metadata must be a dict"):
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": "bad"})


class TestSparseValuesValidation:
    """unified-vecfmt-0012, 0013, 0014, 0015."""

    def test_sparse_mismatched_lengths(self) -> None:
        """unified-vecfmt-0012."""
        with pytest.raises(PineconeValueError, match="same length"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": {"indices": [0, 1], "values": [1.0]},
                }
            )

    def test_sparse_missing_indices(self) -> None:
        """unified-vecfmt-0013."""
        with pytest.raises(PineconeValueError, match="missing required keys"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": {"values": [1.0]},
                }
            )

    def test_sparse_missing_values(self) -> None:
        """unified-vecfmt-0013."""
        with pytest.raises(PineconeValueError, match="missing required keys"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": {"indices": [0]},
                }
            )

    def test_sparse_non_dict_rejected(self) -> None:
        """unified-vecfmt-0014."""
        with pytest.raises(PineconeTypeError, match="sparse_values must be a dict"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": [0, 1.0],
                }
            )

    def test_sparse_string_index_rejected(self) -> None:
        """unified-vecfmt-0015: first-element type check."""
        with pytest.raises(PineconeTypeError, match="indices must be integers"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": {"indices": ["a"], "values": [1.0]},
                }
            )

    def test_sparse_string_value_rejected(self) -> None:
        """unified-vecfmt-0015: first-element type check on values."""
        with pytest.raises(PineconeTypeError, match="values must be floats"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "sparse_values": {"indices": [0], "values": ["bad"]},
                }
            )

    def test_sparse_string_value_in_3key_dict_rejected(self) -> None:
        """Fast-path 3-key dict must raise PineconeTypeError, not ValueError."""
        with pytest.raises(PineconeTypeError, match="values must be floats"):
            VectorFactory.build(
                {
                    "id": "x",
                    "values": [1.0],
                    "sparse_values": {"indices": [0], "values": ["bad"]},
                }
            )

    def test_sparse_string_index_in_3key_dict_rejected(self) -> None:
        """Fast-path 3-key dict must raise PineconeTypeError for non-int indices."""
        with pytest.raises(PineconeTypeError, match="indices must be integers"):
            VectorFactory.build(
                {
                    "id": "x",
                    "values": [1.0],
                    "sparse_values": {"indices": ["a"], "values": [1.0]},
                }
            )


class TestGeneralValidation:
    """unified-vecfmt-0016, 0017."""

    def test_dict_empty_values_no_sparse_rejected(self) -> None:
        """unified-vecfmt-0016."""
        with pytest.raises(PineconeValueError, match="at least one of"):
            VectorFactory.build({"id": "v1", "values": []})

    def test_dict_no_values_no_sparse_rejected(self) -> None:
        """unified-vecfmt-0016."""
        with pytest.raises(PineconeValueError, match="at least one of"):
            VectorFactory.build({"id": "v1"})

    def test_tuple_empty_values_rejected(self) -> None:
        """unified-vecfmt-0016: tuple with empty values."""
        with pytest.raises(PineconeValueError, match="at least one of"):
            VectorFactory.build(("v1", []))

    def test_integer_id_rejected(self) -> None:
        """unified-vecfmt-0017."""
        with pytest.raises(PineconeTypeError, match="must be a string"):
            VectorFactory.build((123, [0.1, 0.2]))

    def test_integer_id_in_dict_rejected(self) -> None:
        """unified-vecfmt-0017."""
        with pytest.raises(PineconeTypeError, match="must be a string"):
            VectorFactory.build({"id": 123, "values": [0.1]})

    def test_validate_id_rejects_non_ascii(self) -> None:
        """unified-ids-0002: non-ASCII IDs rejected."""
        with pytest.raises(PineconeValueError, match="ASCII"):
            VectorFactory.build(("café", [0.1, 0.2]))

    def test_validate_id_rejects_null_char(self) -> None:
        """unified-ids-0002: null characters in ID rejected."""
        with pytest.raises(PineconeValueError, match="null"):
            VectorFactory.build(("hello\x00world", [0.1, 0.2]))

    def test_validate_id_accepts_ascii(self) -> None:
        """unified-ids-0002: valid ASCII IDs pass validation."""
        result = VectorFactory.build(("valid-id_123", [0.1, 0.2]))
        assert result.id == "valid-id_123"


class TestUnsupportedTypes:
    """Reject non-Vector, non-tuple, non-dict inputs."""

    def test_list_rejected(self) -> None:
        with pytest.raises(PineconeTypeError, match="got list"):
            VectorFactory.build([0.1, 0.2])

    def test_string_rejected(self) -> None:
        with pytest.raises(PineconeTypeError, match="got str"):
            VectorFactory.build("bad")

    def test_int_rejected(self) -> None:
        with pytest.raises(PineconeTypeError, match="got int"):
            VectorFactory.build(42)


class TestArrayLikeValues:
    """Array-like values are unboxed in C, not one Python object per element."""

    def test_numpy_values_become_real_python_floats(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = VectorFactory.build({"id": "v1", "values": arr})

        assert all(type(v) is float for v in result.values)

    def test_numpy_values_are_json_serializable(self) -> None:
        """`list(arr)` yielded np.float32 objects, which orjson refuses outright."""
        np = pytest.importorskip("numpy")
        import orjson

        arr = np.array([0.1, 0.2], dtype=np.float32)
        with pytest.raises(TypeError, match=r"numpy\.float32"):
            orjson.dumps({"values": list(arr)})

        result = VectorFactory.build({"id": "v1", "values": arr})

        assert orjson.dumps({"values": result.values})

    def test_plain_lists_are_still_passed_through_untouched(self) -> None:
        values = [0.1, 0.2]

        result = VectorFactory.build({"id": "v1", "values": values})

        assert result.values is values

    def test_tuple_values_still_convert(self) -> None:
        result = VectorFactory.build({"id": "v1", "values": (0.1, 0.2)})

        assert result.values == [0.1, 0.2]

    def test_numpy_values_in_a_tuple_input(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([0.1, 0.2], dtype=np.float32)

        result = VectorFactory.build(("v1", arr))

        assert all(type(v) is float for v in result.values)

    def test_zero_dimensional_array_still_rejected(self) -> None:
        """A 0-d array is not a sequence of values; tolist() gives a bare scalar."""
        np = pytest.importorskip("numpy")

        with pytest.raises(TypeError):
            VectorFactory.build({"id": "v1", "values": np.float32(0.5).reshape(())})
