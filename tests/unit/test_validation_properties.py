"""Property-based tests for pinecone._internal.validation.

Uses Hypothesis to generate whole classes of valid and invalid inputs, rather
than the hand-picked examples in test_require_valid_resource_name.py, so both
sides of each validator's accept/reject boundary are exercised exhaustively.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.validation import (
    require_in_range,
    require_max_length,
    require_non_empty,
    require_one_of,
    require_positive,
    require_valid_resource_name,
)
from pinecone.errors.exceptions import ValidationError

_NAME_EDGE_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
_NAME_INNER_CHARS = _NAME_EDGE_CHARS + "-"
_INVALID_NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_./ +!@#"


@st.composite
def valid_resource_names(draw: st.DrawFn) -> str:
    length = draw(st.integers(min_value=1, max_value=45))
    if length == 1:
        return draw(st.sampled_from(_NAME_EDGE_CHARS))
    first = draw(st.sampled_from(_NAME_EDGE_CHARS))
    last = draw(st.sampled_from(_NAME_EDGE_CHARS))
    middle = draw(st.text(alphabet=_NAME_INNER_CHARS, min_size=length - 2, max_size=length - 2))
    return first + middle + last


@st.composite
def one_of_cases(draw: st.DrawFn) -> tuple[list[str], str]:
    allowed = draw(st.lists(st.text(max_size=6), min_size=1, max_size=6, unique=True))
    value = draw(st.one_of(st.sampled_from(allowed), st.text(max_size=6)))
    return allowed, value


@given(name=valid_resource_names())
def test_valid_resource_names_are_accepted(name: str) -> None:
    require_valid_resource_name("field", name)


@given(name=st.text(alphabet=_NAME_INNER_CHARS, min_size=46, max_size=100))
def test_over_length_resource_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        require_valid_resource_name("field", name)


@given(
    base=st.text(alphabet=_NAME_INNER_CHARS, max_size=30),
    bad=st.sampled_from(_INVALID_NAME_CHARS),
)
def test_resource_names_with_invalid_chars_are_rejected(base: str, bad: str) -> None:
    with pytest.raises(ValidationError):
        require_valid_resource_name("field", base + bad)


@given(body=st.text(alphabet=_NAME_INNER_CHARS, max_size=43))
def test_resource_names_with_edge_hyphens_are_rejected(body: str) -> None:
    with pytest.raises(ValidationError):
        require_valid_resource_name("field", f"-{body}")
    with pytest.raises(ValidationError):
        require_valid_resource_name("field", f"{body}-")


@given(value=st.text(max_size=20))
def test_require_non_empty_string_contract(value: str) -> None:
    if value.strip():
        require_non_empty("field", value)
    else:
        with pytest.raises(ValidationError):
            require_non_empty("field", value)


@given(value=st.lists(st.integers(), max_size=10))
def test_require_non_empty_list_contract(value: list[int]) -> None:
    if value:
        require_non_empty("field", value)
    else:
        with pytest.raises(ValidationError):
            require_non_empty("field", value)


@given(value=st.integers(min_value=1))
def test_require_positive_accepts_positive(value: int) -> None:
    require_positive("field", value)


@given(value=st.integers(max_value=0))
def test_require_positive_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        require_positive("field", value)


@given(lo=st.integers(-1000, 1000), span=st.integers(0, 1000), value=st.integers(-3000, 3000))
def test_require_in_range_matches_inclusive_bounds(lo: int, span: int, value: int) -> None:
    hi = lo + span
    if lo <= value <= hi:
        require_in_range("field", value, lo, hi)
    else:
        with pytest.raises(ValidationError):
            require_in_range("field", value, lo, hi)


@given(value=st.text(max_size=50), max_length=st.integers(min_value=0, max_value=50))
def test_require_max_length_contract(value: str, max_length: int) -> None:
    if len(value) <= max_length:
        require_max_length("field", value, max_length)
    else:
        with pytest.raises(ValidationError):
            require_max_length("field", value, max_length)


@given(case=one_of_cases())
def test_require_one_of_contract(case: tuple[list[str], str]) -> None:
    allowed, value = case
    if value in allowed:
        require_one_of("field", value, allowed)
    else:
        with pytest.raises(ValidationError):
            require_one_of("field", value, allowed)
