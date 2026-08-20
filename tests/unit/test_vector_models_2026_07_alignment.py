"""2026-07 alignment for the db_data vector models.

Three spec changes land here, and each one is pinned below.

``QueryRequest.queries`` and the whole ``QueryVector`` schema are gone in 2026-07
(they were ``db_data_2025-10.oas.yaml:1412-1419`` and ``:1482-1531``). The SDK never
exposed either, so there is nothing to delete — only something to keep deleted, which
``TestQueriesRemoved`` does by asserting the names are absent from the public surface
and that a dict carrying ``queries`` is refused rather than forwarded.

``IndexDescription``'s fullness fields are respelled ``memoryFullness`` /
``storageFullness`` (``db_data_2026-07.oas.yaml:1949,1953``). This corrects the OAS to
match the wire: they are proto fields 7 and 8, ``memory_fullness`` / ``storage_fullness``
(pinecone-db ``pc-outer-protos/src/pinecone-specs/vector_service.proto:626-630`` @
``f6fd0a40``), and JSON transcoding camelCases them. ``TestFullnessCasing`` asserts both
directions so the ``rename="camel"`` that already made this work cannot be dropped
silently.

``Vector.metadata`` becomes a map of ``MetadataValue``: string, number, boolean, or list
of strings (``db_data_2026-07.oas.yaml:2662-2669``, ``:2714-2724``). The server has always
enforced this; 2026-07 writes it down. ``TestMetadataValueTyping`` pins the client-side
check, including that its message is the server's own.
"""

from __future__ import annotations

import json
from typing import Any

import msgspec
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.vector_factory import VectorFactory, validate_vector_dict
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.vectors.responses import DescribeIndexStatsResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import ScoredVector, Vector

_SERVER_MESSAGE = "Metadata value must be a string, number, boolean or list of strings"


class TestQueriesRemoved:
    """2026-07 deletes ``QueryRequest.queries`` and the ``QueryVector`` schema."""

    def test_query_vector_is_not_a_public_name(self) -> None:
        import pinecone

        assert not hasattr(pinecone, "QueryVector")

    def test_query_vector_is_not_importable_from_the_vector_models(self) -> None:
        import pinecone.models.vectors as vectors

        assert "QueryVector" not in vectors.__all__
        with pytest.raises(AttributeError, match="QueryVector"):
            vectors.QueryVector  # type: ignore[attr-defined]

    def test_queries_is_not_a_recognized_vector_key(self) -> None:
        """The failure mode for the removed field: refused here, not forwarded as a 400."""
        with pytest.raises(PineconeValueError, match=r"unrecognized keys: \['queries'\]"):
            VectorFactory.build({"id": "v1", "values": [0.1], "queries": [{"values": [0.1]}]})


class TestFullnessCasing:
    """``memoryFullness`` / ``storageFullness`` — camelCase on the wire, snake on the model."""

    def test_decodes_camel_case_into_snake_case_attributes(self) -> None:
        decoded = msgspec.json.decode(
            json.dumps({"memoryFullness": 0.3, "storageFullness": 0.4}).encode(),
            type=DescribeIndexStatsResponse,
        )

        assert decoded.memory_fullness == pytest.approx(0.3)
        assert decoded.storage_fullness == pytest.approx(0.4)

    def test_encodes_snake_case_attributes_back_to_camel_case(self) -> None:
        encoded = json.loads(
            msgspec.json.encode(
                DescribeIndexStatsResponse(memory_fullness=0.3, storage_fullness=0.4)
            )
        )

        assert encoded["memoryFullness"] == pytest.approx(0.3)
        assert encoded["storageFullness"] == pytest.approx(0.4)
        assert "memory_fullness" not in encoded
        assert "storage_fullness" not in encoded

    def test_the_2025_10_snake_spelling_no_longer_decodes(self) -> None:
        """The documented failure mode for the rename: the old spelling is simply ignored."""
        decoded = msgspec.json.decode(
            json.dumps({"memory_fullness": 0.3, "storage_fullness": 0.4}).encode(),
            type=DescribeIndexStatsResponse,
        )

        assert decoded.memory_fullness is None
        assert decoded.storage_fullness is None

    def test_both_fields_absent_round_trips_as_none(self) -> None:
        decoded = msgspec.json.decode(b"{}", type=DescribeIndexStatsResponse)

        assert decoded.memory_fullness is None
        assert decoded.storage_fullness is None

        reencoded = json.loads(msgspec.json.encode(decoded))

        assert reencoded["memoryFullness"] is None
        assert reencoded["storageFullness"] is None
        assert "memory_fullness" not in reencoded
        assert "storage_fullness" not in reencoded


class TestSparseOnlyVector:
    """2026-07 ``Vector`` is ``anyOf: [required values, required sparseValues]``."""

    def test_sparse_only_vector_serializes_an_empty_values_array(self) -> None:
        encoded = json.loads(
            msgspec.json.encode(
                Vector(id="v1", values=[], sparse_values=SparseValues(indices=[3], values=[0.5]))
            )
        )

        assert encoded["values"] == []
        assert encoded["sparseValues"] == {"indices": [3], "values": [0.5]}

    def test_the_serialized_form_satisfies_the_anyof(self) -> None:
        encoded = json.loads(
            msgspec.json.encode(
                Vector(id="v1", values=[], sparse_values=SparseValues(indices=[3], values=[0.5]))
            )
        )

        assert "values" in encoded or "sparseValues" in encoded

    def test_dense_only_vector_omits_sparse_values(self) -> None:
        encoded = json.loads(msgspec.json.encode(Vector(id="v1", values=[0.1, 0.2])))

        assert encoded["values"] == [0.1, 0.2]
        assert encoded["sparseValues"] is None

    def test_a_vector_with_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="either values or sparse_values"):
            Vector(id="v1", values=[])

    def test_sparse_only_survives_a_round_trip(self) -> None:
        original = Vector(id="v1", values=[], sparse_values=SparseValues(indices=[3], values=[0.5]))

        decoded = msgspec.json.decode(msgspec.json.encode(original), type=Vector)

        assert decoded.values == []
        assert decoded.sparse_values == SparseValues(indices=[3], values=[0.5])
        assert decoded.metadata is None

    def test_scored_vector_round_trips_with_metadata_absent(self) -> None:
        decoded = msgspec.json.decode(
            json.dumps({"id": "v1", "score": 0.9}).encode(), type=ScoredVector
        )

        assert decoded.values == []
        assert decoded.sparse_values is None
        assert decoded.metadata is None


_LEGAL_VALUES: list[Any] = [
    "a string",
    "",
    0,
    -1,
    42,
    0.0,
    1.5,
    True,
    False,
    [],
    ["one"],
    ["one", "two", ""],
    ("one", "two"),
    {"only", "strings"},
    frozenset({"one"}),
]

_ILLEGAL_VALUES: list[Any] = [
    {"nested": "object"},
    {},
    [1, 2],
    [10],
    ["a", 1],
    ["a", None],
    [["nested"]],
    [{"k": "v"}],
    (1, 2),
    {1, 2},
    object(),
    b"bytes",
]


class TestMetadataValueTyping:
    """``MetadataValue``: string, number, boolean, or list of strings — nothing else."""

    @pytest.mark.parametrize("value", _LEGAL_VALUES, ids=repr)
    def test_legal_values_are_accepted(self, value: Any) -> None:
        result = VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"f": value}})

        assert result.metadata == {"f": value}

    @pytest.mark.parametrize("value", _ILLEGAL_VALUES, ids=repr)
    def test_illegal_values_are_rejected(self, value: Any) -> None:
        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"f": value}})

    def test_the_message_is_the_servers_message_verbatim(self) -> None:
        """pinecone-db ``pc-metadata-filtering/src/metadata/compiler.rs:1750`` @ ``f6fd0a40``."""
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build(
                {"id": "v1", "values": [0.1], "metadata": {"price": {"color": "red"}}}
            )

        assert str(excinfo.value) == (
            "Metadata value must be a string, number, boolean or list of strings, "
            "got '{\"color\":\"red\"}' for field 'price'"
        )

    def test_a_list_of_numbers_renders_the_way_the_server_renders_it(self) -> None:
        """``[10]`` prints as ``[10.0]`` — compiler.rs:1761 @ ``f6fd0a40``."""
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"price": [10]}})

        assert str(excinfo.value) == (
            "Metadata value must be a string, number, boolean or list of strings, "
            "got '[10.0]' for field 'price'"
        )

    def test_a_multi_element_list_of_numbers_renders_without_spaces(self) -> None:
        """pc-validation ``src/data_plane/tests.rs:302`` @ ``f6fd0a40``."""
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"price": [3, 4]}})

        assert "got '[3.0,4.0]' for field 'price'" in str(excinfo.value)

    def test_a_long_value_is_truncated_at_sixteen_bytes(self) -> None:
        """compiler.rs:1776 @ ``f6fd0a40`` — 16 bytes, then ``...``."""
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build(
                {
                    "id": "v1",
                    "values": [0.1],
                    "metadata": {"price": {"abcdefghijklmnopqrstuvwxyz": 1}},
                }
            )

        assert str(excinfo.value) == (
            "Metadata value must be a string, number, boolean or list of strings, "
            "got '{\"abcdefghijklmn...' for field 'price'"
        )

    def test_truncation_does_not_split_a_multibyte_character(self) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"f": {"k": "é" * 20}}})

        str(excinfo.value).encode("utf-8")

    def test_the_offending_field_is_named(self) -> None:
        with pytest.raises(PineconeTypeError, match="for field 'bad'"):
            VectorFactory.build(
                {
                    "id": "v1",
                    "values": [0.1],
                    "metadata": {"good": "ok", "bad": {"nested": True}},
                }
            )

    def test_a_none_value_is_accepted_because_the_server_strips_it(self) -> None:
        """``NullHandling::Strip`` — pinecone-db ``pc-utils/src/prost_util.rs:75-126`` @
        ``f6fd0a40``. Rejecting it here would break callers whose key is silently dropped
        today. See the SPEC-vs-BACKEND question filed alongside this change.
        """
        result = VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"tag": None}})

        assert result.metadata == {"tag": None}

    def test_metadata_is_validated_on_the_tuple_form(self) -> None:
        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            VectorFactory.build(("v1", [0.1], {"f": {"nested": 1}}))

    def test_metadata_is_validated_on_a_prebuilt_vector(self) -> None:
        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            VectorFactory.build(Vector(id="v1", values=[0.1], metadata={"f": {"nested": 1}}))

    def test_metadata_is_validated_on_a_vector_subclass(self) -> None:
        class Subclass(Vector):
            pass

        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            VectorFactory.build(Subclass(id="v1", values=[0.1], metadata={"f": {"nested": 1}}))

    def test_metadata_is_validated_on_the_dataframe_path(self) -> None:
        """The gRPC DataFrame path skips ``build``, so it carries its own copy of the checks."""
        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            validate_vector_dict({"id": "v1", "values": [0.1], "metadata": {"f": {"nested": 1}}})

    def test_the_dataframe_path_accepts_legal_metadata(self) -> None:
        validate_vector_dict(
            {"id": "v1", "values": [0.1], "metadata": {"a": "s", "b": 1, "c": True, "d": ["x"]}}
        )

    def test_a_non_dict_metadata_is_still_a_type_error(self) -> None:
        with pytest.raises(PineconeTypeError, match="metadata must be a dict"):
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": "bad"})

    def test_a_rejected_value_never_reaches_the_encoder(self) -> None:
        """The whole point: msgspec would happily encode a nested object and let the server
        reject the entire batch."""
        assert msgspec.json.encode({"f": {"nested": 1}}) == b'{"f":{"nested":1}}'

        with pytest.raises(PineconeTypeError):
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {"f": {"nested": 1}}})


_legal_metadata_values = st.one_of(
    st.text(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.lists(st.text()),
)

_illegal_metadata_values = st.one_of(
    st.dictionaries(st.text(), st.text()),
    st.lists(st.integers(), min_size=1),
    st.lists(st.none(), min_size=1),
    st.lists(st.dictionaries(st.text(), st.text()), min_size=1),
    st.lists(st.one_of(st.text(), st.integers()), min_size=2).filter(
        lambda items: not all(isinstance(i, str) for i in items)
    ),
)

_metadata_keys = st.text().filter(lambda k: not k.startswith("$"))


class TestMetadataValueProperties:
    """Every value drawn from the legal grammar passes; every illegal one raises."""

    @given(st.dictionaries(_metadata_keys, _legal_metadata_values))
    def test_legal_metadata_always_builds(self, metadata: dict[str, Any]) -> None:
        result = VectorFactory.build({"id": "v1", "values": [0.1], "metadata": metadata})

        assert result.metadata == metadata

    @given(_metadata_keys, _illegal_metadata_values)
    def test_illegal_metadata_always_raises(self, key: str, value: Any) -> None:
        with pytest.raises(PineconeTypeError, match=_SERVER_MESSAGE):
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {key: value}})

    @given(st.dictionaries(_metadata_keys, _legal_metadata_values))
    def test_legal_metadata_always_survives_a_vector_round_trip(
        self, metadata: dict[str, Any]
    ) -> None:
        built = VectorFactory.build({"id": "v1", "values": [0.1], "metadata": metadata})

        decoded = msgspec.json.decode(msgspec.json.encode(built), type=Vector)

        assert decoded.metadata == metadata

    @given(_metadata_keys, _illegal_metadata_values)
    def test_the_message_always_names_the_field_and_carries_the_servers_text(
        self, key: str, value: Any
    ) -> None:
        with pytest.raises(PineconeTypeError) as excinfo:
            VectorFactory.build({"id": "v1", "values": [0.1], "metadata": {key: value}})

        message = str(excinfo.value)
        assert message.startswith(_SERVER_MESSAGE + ", got '")
        assert message.endswith(f"for field '{key}'")
