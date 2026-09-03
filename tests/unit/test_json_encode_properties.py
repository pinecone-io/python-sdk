"""Property-based tests for the request-body JSON encoder (issue #187).

Three invariants, checked over the input space rather than a handful of examples:

*Every encodable body still encodes.* The typed-error wrapper added around
``orjson.dumps`` must be invisible on the success path — same bytes, no new
rejections. The int64 boundary values themselves are part of that space.

*Every out-of-range integer is reported as a Pinecone error.* Both signs, at any
depth, whether it sits in a dict value, a list element, or vector metadata. The
error is a ``PineconeTypeError``, which is simultaneously a ``PineconeError``
and a built-in ``TypeError`` — the swap only widens what a caller can catch.

*The reported path leads back to the offending value.* Naming the field is the
whole point of the ticket, so the path is not merely asserted to be non-empty:
it is walked back through the generated body and must land on the planted
integer. Generated dict keys are restricted to an unambiguous alphabet for that
reason — a key containing ``.`` or ``[`` would make any dotted path notation
ambiguous, which is a limitation of the notation, not of the encoder.
"""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.http_client import _JSON_INT_MAX, _JSON_INT_MIN, _encode_json
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
        st.lists(children, max_size=4),
        st.dictionaries(_SAFE_KEYS, children, max_size=4),
    ),
    max_leaves=12,
)

_STEPS = st.lists(
    st.one_of(_SAFE_KEYS, st.integers(min_value=0, max_value=3)),
    max_size=4,
)


@st.composite
def planted_bodies(draw: st.DrawFn) -> tuple[Any, str, int]:
    """A body with exactly one out-of-range integer, plus its expected path.

    Benign siblings are planted alongside so the walk has to skip encodable
    values to reach the offender instead of stopping at the first thing it sees.
    """
    steps = draw(_STEPS)
    planted = draw(_OUT_OF_RANGE)
    node: Any = planted
    for step in reversed(steps):
        if isinstance(step, str):
            sibling = draw(_SAFE_KEYS)
            wrapper: dict[str, Any] = {}
            if sibling != step:
                wrapper[sibling] = draw(_BENIGN)
            wrapper[step] = node
            node = wrapper
        else:
            node = [draw(_BENIGN) for _ in range(step)] + [node] + [draw(_BENIGN)]
    parts: list[str] = []
    for step in steps:
        if isinstance(step, str):
            parts.append(f".{step}" if parts else step)
        else:
            parts.append(f"[{step}]")
    return node, "".join(parts) or "<body>", planted


def _resolve(body: Any, path: str) -> Any:
    if path == "<body>":
        return body
    node = body
    for segment in path.replace("[", ".[").split("."):
        if not segment:
            continue
        if segment.startswith("["):
            node = node[int(segment[1:-1])]
        else:
            node = node[segment]
    return node


@given(_ENCODABLE)
@settings(max_examples=200, deadline=None)
def test_encodable_bodies_are_unchanged(body: Any) -> None:
    assert _encode_json(body) == orjson.dumps(body)


@given(st.sampled_from([_JSON_INT_MIN, _JSON_INT_MIN + 1, _JSON_INT_MAX - 1, _JSON_INT_MAX]))
def test_int64_boundary_values_encode(value: int) -> None:
    assert orjson.loads(_encode_json({"n": value}))["n"] == value


@given(_OUT_OF_RANGE)
@settings(max_examples=100, deadline=None)
def test_out_of_range_raises_pinecone_typed_error(value: int) -> None:
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_json({"n": value})
    exc = excinfo.value
    assert isinstance(exc, PineconeError)
    assert isinstance(exc, TypeError)
    assert exc.path == "n"


@given(planted_bodies())
@settings(max_examples=200, deadline=None)
def test_reported_path_leads_back_to_the_offending_value(
    case: tuple[Any, str, int],
) -> None:
    body, expected_path, planted = case
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_json(body)
    exc = excinfo.value
    reported = exc.path if exc.path is not None else "<body>"
    assert reported == expected_path
    assert _resolve(body, reported) == planted
    assert reported in str(exc)


@given(_SAFE_KEYS, _OUT_OF_RANGE)
@settings(max_examples=50, deadline=None)
def test_wide_int_in_vector_metadata_names_the_field(key: str, value: int) -> None:
    body = {
        "vectors": [
            {"id": "a", "values": [0.1], "metadata": {"ok": 1}},
            {"id": "b", "values": [0.2], "metadata": {key: value}},
        ]
    }
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_json(body)
    assert excinfo.value.path == f"vectors[1].metadata.{key}"


@given(_OUT_OF_RANGE, st.integers(min_value=0, max_value=3))
@settings(max_examples=50, deadline=None)
def test_wide_int_in_a_list_element_names_the_index(value: int, index: int) -> None:
    documents: list[dict[str, Any]] = [{"text": "hi"} for _ in range(index + 2)]
    documents[index]["count"] = value
    with pytest.raises(PineconeTypeError) as excinfo:
        _encode_json({"documents": documents})
    assert excinfo.value.path == f"documents[{index}].count"
