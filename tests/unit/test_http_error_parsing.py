"""Tests for _raise_for_status body extraction, error_code, and request_id."""

from __future__ import annotations

import httpx
import pytest

from pinecone._internal.http_client import (
    _TEXT_BODY_MAX_LEN,
    _extract_message_and_error_code,
    _extract_request_id,
    _raise_for_status,
)
from pinecone.errors.exceptions import (
    ApiError,
    NotFoundError,
    RateLimitError,
    ServiceError,
)


def _make_response(
    status_code: int,
    *,
    body: bytes | None = None,
    json: object | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    content = body
    if content is None and json is not None:
        import orjson

        content = orjson.dumps(json)
        base_headers = {"content-type": "application/json"}
    else:
        base_headers = {}
    if headers:
        base_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        content=content or b"",
        headers=base_headers,
    )


class TestMessageExtraction:
    def test_pinecone_canonical_shape(self) -> None:
        body = {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "$match_phras is not a valid operator",
            },
            "status": 400,
        }
        response = _make_response(400, json=body)
        msg, error_code = _extract_message_and_error_code(body, response)
        assert msg == "$match_phras is not a valid operator"
        assert error_code == "INVALID_ARGUMENT"

    def test_top_level_message(self) -> None:
        body = {"message": "Bad request"}
        response = _make_response(400, json=body)
        msg, error_code = _extract_message_and_error_code(body, response)
        assert msg == "Bad request"
        assert error_code is None

    def test_top_level_detail(self) -> None:
        body = {"detail": "Index does not exist"}
        response = _make_response(404, json=body)
        msg, error_code = _extract_message_and_error_code(body, response)
        assert msg == "Index does not exist"
        assert error_code is None

    def test_top_level_description(self) -> None:
        body = {"description": "Something went wrong"}
        response = _make_response(500, json=body)
        msg, error_code = _extract_message_and_error_code(body, response)
        assert msg == "Something went wrong"
        assert error_code is None

    def test_priority_canonical_beats_top_level(self) -> None:
        body = {"error": {"code": "X", "message": "from error"}, "message": "from top"}
        response = _make_response(400, json=body)
        msg, error_code = _extract_message_and_error_code(body, response)
        assert msg == "from error"
        assert error_code == "X"

    def test_plain_text_body(self) -> None:
        response = _make_response(503, body=b"Service Unavailable")
        msg, error_code = _extract_message_and_error_code(None, response)
        assert msg == "Service Unavailable"
        assert error_code is None

    def test_long_text_body_truncated(self) -> None:
        long_text = "x" * 1000
        response = _make_response(503, body=long_text.encode())
        msg, _error_code = _extract_message_and_error_code(None, response)
        assert msg.endswith("... (truncated)")
        assert len(msg) <= _TEXT_BODY_MAX_LEN + len("... (truncated)")

    def test_empty_body_uses_reason_phrase(self) -> None:
        response = _make_response(404, body=b"")
        msg, error_code = _extract_message_and_error_code(None, response)
        # reason_phrase for 404 is "Not Found" in httpx
        assert msg == "Not Found"
        assert error_code is None

    def test_empty_body_no_reason_phrase(self) -> None:
        # Use a non-standard status code with no standard reason
        response = httpx.Response(status_code=599, content=b"", headers={})
        msg, error_code = _extract_message_and_error_code(None, response)
        assert msg == ""
        assert error_code is None

    def test_malformed_json_body(self) -> None:
        response = _make_response(500, body=b"not valid json at all")
        # body is None because JSON parse failed; falls through to text
        msg, error_code = _extract_message_and_error_code(None, response)
        assert "not valid json" in msg
        assert error_code is None

    def test_html_error_page(self) -> None:
        html = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
        response = _make_response(502, body=html)
        msg, error_code = _extract_message_and_error_code(None, response)
        assert msg != ""  # does not crash; returns something
        assert error_code is None


class TestRequestIdExtraction:
    def test_pinecone_header(self) -> None:
        headers = httpx.Headers({"x-pinecone-request-id": "abc"})
        assert _extract_request_id(headers) == "abc"

    def test_x_request_id_fallback(self) -> None:
        headers = httpx.Headers({"x-request-id": "xyz"})
        assert _extract_request_id(headers) == "xyz"

    def test_pinecone_header_wins_when_both_present(self) -> None:
        headers = httpx.Headers(
            {"x-pinecone-request-id": "pinecone-id", "x-request-id": "generic-id"}
        )
        assert _extract_request_id(headers) == "pinecone-id"

    def test_no_headers(self) -> None:
        headers = httpx.Headers({})
        assert _extract_request_id(headers) is None

    def test_empty_string_value(self) -> None:
        headers = httpx.Headers({"x-request-id": ""})
        assert _extract_request_id(headers) is None


class TestRaiseForStatus:
    def test_404_with_canonical_body(self) -> None:
        response = _make_response(
            404,
            json={
                "error": {"code": "NOT_FOUND", "message": "index 'foo' does not exist"},
                "status": 404,
            },
            headers={"x-pinecone-request-id": "req-abc"},
        )
        with pytest.raises(NotFoundError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.error_code == "NOT_FOUND"
        assert err.message == "index 'foo' does not exist"
        assert err.request_id == "req-abc"
        assert str(err) == "[404 NOT_FOUND] index 'foo' does not exist (request_id: req-abc)"

    def test_500_no_body(self) -> None:
        response = _make_response(500, body=b"")
        with pytest.raises(ServiceError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.error_code is None
        assert err.request_id is None

    def test_400_generic_apierror_html_body(self) -> None:
        response = _make_response(400, body=b"<html><body>Bad Request</body></html>")
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.message  # non-empty, sensible
        assert err.error_code is None


class TestRateLimitError:
    def test_429_maps_to_rate_limit_error(self) -> None:
        response = _make_response(
            429,
            json={
                "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests"},
                "status": 429,
            },
            headers={"x-pinecone-request-id": "req-rl-1"},
        )
        with pytest.raises(RateLimitError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.status_code == 429
        assert err.error_code == "RATE_LIMIT_EXCEEDED"
        assert err.request_id == "req-rl-1"

    def test_429_parses_retry_after_integer(self) -> None:
        response = _make_response(
            429,
            body=b"",
            headers={"retry-after": "60"},
        )
        with pytest.raises(RateLimitError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.retry_after == 60.0

    def test_429_handles_missing_retry_after(self) -> None:
        response = _make_response(429, body=b"")
        with pytest.raises(RateLimitError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.retry_after is None

    def test_429_handles_http_date_retry_after(self) -> None:
        response = _make_response(
            429,
            body=b"",
            headers={"retry-after": "Fri, 31 Dec 2026 23:59:59 GMT"},
        )
        with pytest.raises(RateLimitError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.retry_after is None


class TestExtractionResilience:
    def test_body_is_not_a_dict(self) -> None:
        response = _make_response(400, body=b"[1,2,3]")
        msg, code = _extract_message_and_error_code([1, 2, 3], response)
        assert isinstance(msg, str)
        assert code is None

    def test_body_error_is_not_a_dict(self) -> None:
        body = {"error": "string not dict"}
        response = _make_response(400, json=body)
        msg, code = _extract_message_and_error_code(body, response)
        assert isinstance(msg, str)
        assert code is None

    def test_body_error_message_is_not_a_string(self) -> None:
        body = {"error": {"code": "X", "message": 12345}}
        response = _make_response(400, json=body)
        msg, _code = _extract_message_and_error_code(body, response)
        assert isinstance(msg, str)
        # error_code may still be extracted; should not crash regardless

    def test_body_error_code_is_not_a_string(self) -> None:
        body = {"error": {"code": ["not", "a", "string"], "message": "msg"}}
        response = _make_response(400, json=body)
        msg, code = _extract_message_and_error_code(body, response)
        assert msg == "msg"
        assert code is None

    def test_unicode_in_message(self) -> None:
        body = {"error": {"code": "X", "message": "エラー 🚨 données"}}
        response = _make_response(400, json=body)
        msg, _code = _extract_message_and_error_code(body, response)
        assert "エラー" in msg
        assert "🚨" in msg

    def test_extraction_never_raises_under_fuzz(self) -> None:
        weird_inputs: list[object] = [
            None,
            [],
            {},
            set(),
            object(),
            {"error": None},
            {"error": {"code": None, "message": None}},
            {"error": {"code": None, "message": None, "extra": {"deep": "nest"}}},
            {"a": {"b": {"c": {"d": {"e": "deeply nested"}}}}},
        ]
        response = _make_response(400, body=b"")
        for inp in weird_inputs:
            result = _extract_message_and_error_code(inp, response)
            assert isinstance(result, tuple)
            assert len(result) == 2
            msg, code = result
            assert isinstance(msg, str)
            assert code is None or isinstance(code, str)
