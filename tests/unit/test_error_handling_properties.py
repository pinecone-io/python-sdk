"""Property-based tests for HTTP error handling in pinecone._internal.http_client.

Covers the three pure functions on the error path — ``_raise_for_status``,
``_extract_message_and_error_code``, ``_extract_request_id`` — plus the
exception classes' string rendering. The recurring properties are *totality*
(these run against untrusted server responses, so they must never crash or leak
a non-Pinecone exception, and the renderers must never raise), *correct
status-to-type mapping*, and *bounded/normalized output* (truncated messages,
non-negative retry-after). Complements test_retry_after_fuzz.py, which fuzzes
the retry-after delay computation specifically.
"""

from __future__ import annotations

import httpx
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from pinecone._internal.http_client import (
    _TEXT_BODY_MAX_LEN,
    _extract_message_and_error_code,
    _extract_request_id,
    _raise_for_status,
)
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeError,
    PineconeTypeError,
    PineconeValueError,
    RateLimitError,
    ServiceError,
    UnauthorizedError,
)

_TRUNCATION_SUFFIX = "... (truncated)"

_json_text = st.text(alphabet=st.characters(codec="utf-8"), max_size=64)
_header_value = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=32
)
_header_name = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=16)
_SPECIAL_HEADERS = frozenset({"x-request-id", "x-pinecone-request-id"})


def _expected_type(status: int) -> type[ApiError]:
    specific = {
        401: UnauthorizedError,
        403: ForbiddenError,
        404: NotFoundError,
        409: ConflictError,
        429: RateLimitError,
    }.get(status)
    if specific is not None:
        return specific
    return ServiceError if 500 <= status <= 599 else ApiError


@given(status=st.integers(min_value=400, max_value=599))
@example(status=400)
@example(status=401)
@example(status=403)
@example(status=404)
@example(status=409)
@example(status=429)
@example(status=500)
def test_raise_for_status_maps_each_status_to_its_type(status: int) -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(httpx.Response(status, json={}))
    assert type(exc_info.value) is _expected_type(status)
    assert exc_info.value.status_code == status


@given(status=st.integers(min_value=200, max_value=299))
def test_raise_for_status_does_not_raise_on_success(status: int) -> None:
    _raise_for_status(httpx.Response(status))


@given(
    status=st.integers(min_value=400, max_value=599),
    content=st.binary(max_size=200),
    headers=st.dictionaries(_header_name, _header_value, max_size=5),
)
def test_raise_for_status_always_raises_pinecone_error(
    status: int, content: bytes, headers: dict[str, str]
) -> None:
    resp = httpx.Response(status, content=content, headers=headers)
    with pytest.raises(PineconeError):
        _raise_for_status(resp)


@given(
    status=st.sampled_from([400, 401, 403, 404, 409, 429, 500, 503]),
    message=st.text(alphabet=st.characters(codec="utf-8"), min_size=1, max_size=64),
    code=st.one_of(st.none(), _json_text),
)
def test_error_message_and_code_propagated_from_body(
    status: int, message: str, code: str | None
) -> None:
    resp = httpx.Response(status, json={"error": {"message": message, "code": code}})
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(resp)
    assert exc_info.value.message == message
    assert exc_info.value.error_code == (code if isinstance(code, str) else None)


@given(
    retry_after=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=32)
)
def test_rate_limit_retry_after_is_none_or_non_negative(retry_after: str) -> None:
    resp = httpx.Response(429, headers={"retry-after": retry_after}, json={})
    with pytest.raises(RateLimitError) as exc_info:
        _raise_for_status(resp)
    ra = exc_info.value.retry_after
    assert ra is None or (isinstance(ra, float) and ra >= 0.0 and ra == ra)


@given(body_text=st.text(alphabet=st.characters(codec="utf-8"), min_size=1, max_size=2000))
def test_extract_message_is_length_bounded_in_text_fallback(body_text: str) -> None:
    resp = httpx.Response(400, content=body_text.encode("utf-8"))
    message, code = _extract_message_and_error_code(None, resp)
    assert isinstance(message, str)
    assert len(message) <= _TEXT_BODY_MAX_LEN + len(_TRUNCATION_SUFFIX)
    assert code is None


@given(
    body=st.one_of(
        st.none(),
        st.integers(),
        st.text(max_size=32),
        st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=4),
    ),
    content=st.binary(max_size=64),
)
def test_extract_message_is_total(body: object, content: bytes) -> None:
    message, code = _extract_message_and_error_code(body, httpx.Response(400, content=content))
    assert isinstance(message, str)
    assert code is None or isinstance(code, str)


@given(
    pinecone_id=st.one_of(st.none(), _header_value),
    request_id=st.one_of(st.none(), _header_value),
    extra=st.dictionaries(
        _header_name.filter(lambda k: k not in _SPECIAL_HEADERS), _header_value, max_size=3
    ),
)
def test_extract_request_id_precedence_and_totality(
    pinecone_id: str | None, request_id: str | None, extra: dict[str, str]
) -> None:
    headers = dict(extra)
    if pinecone_id is not None:
        headers["x-pinecone-request-id"] = pinecone_id
    if request_id is not None:
        headers["x-request-id"] = request_id

    result = _extract_request_id(httpx.Headers(headers))

    if pinecone_id:
        assert result == pinecone_id
    elif request_id:
        assert result == request_id
    else:
        assert result is None


@given(
    status=st.integers(),
    message=st.text(max_size=128),
    error_code=st.one_of(st.none(), st.text(max_size=32)),
    request_id=st.one_of(st.none(), st.text(max_size=32)),
    body=st.one_of(st.none(), st.dictionaries(st.text(max_size=8), st.integers(), max_size=4)),
)
def test_apierror_str_and_repr_are_total(
    status: int,
    message: str,
    error_code: str | None,
    request_id: str | None,
    body: dict[str, int] | None,
) -> None:
    exc = ApiError(message, status, body, error_code=error_code, request_id=request_id)
    rendered = str(exc)
    represented = repr(exc)
    assert isinstance(rendered, str)
    assert isinstance(represented, str)
    assert str(status) in rendered
    assert message in rendered


@given(
    message=st.text(max_size=64),
    path=st.one_of(st.none(), st.text(min_size=1, max_size=32)),
)
def test_value_and_type_error_str_includes_path(message: str, path: str | None) -> None:
    for cls in (PineconeValueError, PineconeTypeError):
        rendered = str(cls(message, path))
        assert isinstance(rendered, str)
        if path:
            assert rendered == f"at {path}: {message}"
        else:
            assert rendered == message
