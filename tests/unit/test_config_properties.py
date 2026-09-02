"""Property-based tests for the normalization helpers in pinecone._internal.config.

Both ``normalize_host`` and ``normalize_source_tag`` are pure string
transforms whose outputs are constrained regardless of input, and both are
idempotent: normalizing an already-normalized value changes nothing.
"""

from __future__ import annotations

from hypothesis import example, given
from hypothesis import strategies as st

from pinecone._internal.config import normalize_host, normalize_source_tag

_SOURCE_TAG_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_:")
_SCHEMES = ("http://", "https://")


@given(host=st.text())
def test_normalize_host_nonempty_result_has_scheme(host: str) -> None:
    result = normalize_host(host)
    assert result == "" or result.startswith(_SCHEMES)


@given(host=st.text())
def test_normalize_host_result_has_no_stacked_scheme(host: str) -> None:
    result = normalize_host(host)
    for scheme in _SCHEMES:
        if result.startswith(scheme):
            assert not result[len(scheme) :].startswith(_SCHEMES)


@given(host=st.text())
@example(host="https://https://https://foo.io")
@example(host="http://https://http://foo.io")
def test_normalize_host_is_idempotent(host: str) -> None:
    once = normalize_host(host)
    assert normalize_host(once) == once


@given(tag=st.text())
def test_normalize_source_tag_output_charset(tag: str) -> None:
    assert set(normalize_source_tag(tag)) <= _SOURCE_TAG_ALLOWED


@given(tag=st.text())
def test_normalize_source_tag_is_idempotent(tag: str) -> None:
    once = normalize_source_tag(tag)
    assert normalize_source_tag(once) == once
