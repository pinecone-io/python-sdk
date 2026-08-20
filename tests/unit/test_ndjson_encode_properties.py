"""Property-based tests for the NDJSON request-body encoder (issue #196).

``upsert_records`` serializes one JSON object per record, so unlike the single
``_encode_json`` body there is no enclosing document to give a rejected value a
path. Three invariants, checked over the input space:

*Every encodable record list still produces the same bytes.* The typed-error
wrapper must be invisible on the success path — one line per record, in order,
byte-identical to a plain per-record ``orjson.dumps``.

*Every unencodable value is reported as a Pinecone error.* Out-of-range
integers, non-string keys, and types JSON has no representation for, at any
depth inside any record.

*The reported path starts at the record the caller passed.* ``records[2]`` is
the index in the caller's own list, which is the only handle they have on it —
a byte offset into the joined blob would not be actionable. As in #195's
property tests, generated dict keys are restricted to an unambiguous alphabet
so the dotted path notation stays resolvable.
"""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.http_client import _JSON_INT_MAX, _JSON_INT_MIN, _encode_ndjson
from pinecone.errors.exceptions import PineconeError, PineconeTypeError

_SAFE_KEYS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)
_IN_RANGE = st.integers(min_value=_JSON_INT_MIN, max_value=_JSON_INT_MAX)
_TOO_LARGE = st.integers(min_value=_JSON_INT_MAX + 1, max_value=_JSON_INT_MAX + 10**40)
_TOO_SMALL = st.integers(min_value=_JSON_INT_MIN - 10**40, max_value=_JSON_INT_MIN - 1)
_OUT_OF_RANGE = st.one_of(_TOO_LARGE, _TOO_SMALL)

_BENIGN = st.one_of(
    st.none(),
    st.booleans(),
    _IN_RANGE,
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=10),
)
_ENCODABLE = st.recursive(
    _BENIGN,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_SAFE_KEYS, children, max_size=3),
    ),
    max_leaves=8,
)
_RECORD = st.dictionaries(_SAFE_KEYS, _ENCODABLE, max_size=4)
_RECORDS = st.lists(_RECORD, min_size=1, max_size=5)

_STEPS = st.lists(
    st.one_of(_SAFE_KEYS, st.integers(min_value=0, max_value=3)),
    max_size=3,
)


@st.composite
def planted_records(draw: st.DrawFn) -> tuple[list[dict[str, Any]], str, int]:
    """A record list with exactly one out-of-range integer, plus its full path.

    Encodable records are planted before and after the offender so the encoder
    has to reach the right line rather than fail on the first one.
    """
    key = draw(_SAFE_KEYS)
    steps = draw(_STEPS)
    planted = draw(_OUT_OF_RANGE)
    node: Any = planted
    for step in reversed(steps):
        if isinstance(step, str):
            node = {step: node}
        else:
            node = [draw(_BENIGN) for _ in range(step)] + [node] + [draw(_BENIGN)]

    leading = draw(st.integers(min_value=0, max_value=2))
    records: list[dict[str, Any]] = [{"_id": f"pre-{i}"} for i in range(leading)]
    records.append({"_id": "bad", key: node})
    records.append({"_id": "post"})

    parts = [f"records[{leading}]", f".{key}"]
    for step in steps:
        parts.append(f".{step}" if isinstance(step, str) else f"[{step}]")
    return records, "".join(parts), planted


def _resolve(records: list[dict[str, Any]], path: str) -> Any:
    node: Any = records
    for segment in path.replace("[", ".[").split("."):
        if not segment:
            continue
        if segment.startswith("["):
            node = node[int(segment[1:-1])]
        elif segment != "records":
            node = node[segment]
    return node


@given(_RECORDS)
@settings(max_examples=200, deadline=None)
def test_encodable_records_are_one_line_each(records: list[dict[str, Any]]) -> None:
    assert _encode_ndjson(records) == b"".join(orjson.dumps(r) + b"\n" for r in records)


@given(_RECORDS)
@settings(max_examples=200, deadline=None)
def test_every_line_round_trips_to_its_record(records: list[dict[str, Any]]) -> None:
    lines = _encode_ndjson(records).split(b"\n")
    assert lines[-1] == b""
    assert [orjson.loads(line) for line in lines[:-1]] == records


@given(st.text(alphabet='\n\r\\"{}', min_size=1, max_size=6))
def test_control_characters_in_a_value_cannot_forge_a_line_break(text: str) -> None:
    """A newline inside a string value is escaped, so the line count is the
    record count no matter what the caller puts in a field."""
    assert _encode_ndjson([{"_id": "a", "text": text}, {"_id": "b"}]).count(b"\n") == 2


@given(_SAFE_KEYS, _OUT_OF_RANGE, st.integers(min_value=0, max_value=4))
@settings(max_examples=100, deadline=None)
def test_out_of_range_int_names_the_record_index(key: str, value: int, index: int) -> None:
    records: list[dict[str, Any]] = [{"_id": f"r{i}"} for i in range(index + 2)]
    records[index][key] = value
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson(records)
    exc = excinfo.value
    assert isinstance(exc, PineconeError)
    assert isinstance(exc, TypeError)
    assert exc.path == f"records[{index}].{key}"
    assert f"records[{index}].{key}" in str(exc)


@given(planted_records())
@settings(max_examples=200, deadline=None)
def test_reported_path_leads_back_to_the_offending_value(
    case: tuple[list[dict[str, Any]], str, int],
) -> None:
    records, expected_path, planted = case
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson(records)
    reported = excinfo.value.path
    assert reported == expected_path
    assert _resolve(records, expected_path) == planted


@given(_SAFE_KEYS, _OUT_OF_RANGE, st.integers(min_value=0, max_value=3))
@settings(max_examples=50, deadline=None)
def test_wide_int_in_record_metadata_names_the_field(key: str, value: int, index: int) -> None:
    records: list[dict[str, Any]] = [
        {"_id": f"r{i}", "metadata": {"ok": 1}} for i in range(index + 2)
    ]
    records[index]["metadata"][key] = value
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson(records)
    assert excinfo.value.path == f"records[{index}].metadata.{key}"


@given(st.integers(min_value=0, max_value=3))
def test_unserializable_type_in_a_record_names_the_field(index: int) -> None:
    records: list[dict[str, Any]] = [{"_id": f"r{i}"} for i in range(index + 2)]
    records[index]["tags"] = {1, 2}
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson(records)
    assert excinfo.value.path == f"records[{index}].tags"
    assert "set" in str(excinfo.value)


@given(st.integers(min_value=0, max_value=3))
def test_non_string_key_in_a_record_names_the_container(index: int) -> None:
    records: list[dict[str, Any]] = [{"_id": f"r{i}"} for i in range(index + 2)]
    records[index]["filter"] = {1: "x"}
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson(records)
    assert excinfo.value.path == f"records[{index}].filter"
    assert "key" in str(excinfo.value)


def test_deeply_nested_record_still_names_the_record() -> None:
    """Nesting depth is not attributable to one value, so #195 reports no path.
    The record index is still known, and stays on the error."""
    node: dict[str, Any] = {}
    deep: dict[str, Any] = {"_id": "deep", "n": node}
    for _ in range(400):
        node["n"] = {}
        node = node["n"]
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson([{"_id": "ok"}, deep])
    assert excinfo.value.path == "records[1]"


def test_empty_record_list_encodes_to_an_empty_body() -> None:
    assert _encode_ndjson([]) == b""


def test_original_orjson_error_is_chained() -> None:
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_ndjson([{"_id": "a", "n": 2**64}])
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert type(excinfo.value.__cause__) is not PineconeTypeError
