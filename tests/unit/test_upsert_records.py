"""Unit tests for Index.upsert_records() method."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.responses import UpsertRecordsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/records/namespaces/test-ns/upsert"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


class TestUpsertRecords:
    """Tests for Index.upsert_records() — integrated inference upsert via NDJSON."""

    @respx.mock
    def test_upsert_records_basic(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        assert isinstance(result, UpsertRecordsResponse)
        assert result.record_count == 2

    @respx.mock
    def test_upsert_records_ndjson_content_type(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}],
        )

        request = route.calls.last.request
        assert request.headers["Content-Type"] == "application/x-ndjson"

    @respx.mock
    def test_upsert_records_ndjson_body_format(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        assert parsed_0["_id"] == "r1"
        assert parsed_1["_id"] == "r2"

    @respx.mock
    def test_upsert_records_id_alias(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"id": "r1", "text": "hello"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert "_id" in parsed
        assert "id" not in parsed
        assert parsed["_id"] == "r1"

    def test_upsert_records_namespace_not_string(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a string"):
            idx.upsert_records(namespace=123, records=[{"_id": "r1"}])  # type: ignore[arg-type]

    def test_upsert_records_namespace_empty_string(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            idx.upsert_records(namespace="", records=[{"_id": "r1"}])

    def test_upsert_records_namespace_whitespace_only(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            idx.upsert_records(namespace="   ", records=[{"_id": "r1"}])

    def test_upsert_records_empty_list(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="test-ns", records=[])

    def test_upsert_records_missing_id(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="'_id' or 'id'"):
            idx.upsert_records(
                namespace="test-ns",
                records=[{"text": "no id"}],
            )

    @respx.mock
    def test_upsert_records_mixed_id_formats(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "a"}, {"id": "r2", "text": "b"}],
        )

        assert result.record_count == 2
        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        # First record keeps _id as-is
        assert parsed_0["_id"] == "r1"
        # Second record has id normalized to _id
        assert parsed_1["_id"] == "r2"
        assert "id" not in parsed_1

    @respx.mock
    def test_upsert_records_both_id_fields_strips_id(self) -> None:
        """When both '_id' and 'id' are present, 'id' is dropped and '_id' wins."""
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "wins", "id": "loses", "text": "hello"}],
        )
        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert parsed["_id"] == "wins"
        assert "id" not in parsed

    def test_upsert_records_id_must_be_string(self) -> None:
        """Records where '_id' is not a string raise ValidationError."""
        idx = _make_index()
        with pytest.raises(ValidationError, match="'_id' must be a string"):
            idx.upsert_records(
                namespace="test-ns",
                records=[{"_id": 123, "text": "hello"}],
            )

    def test_upsert_records_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.upsert_records([{"_id": "r1"}], "test-ns")  # type: ignore[misc]

    @respx.mock
    def test_upsert_records_preserves_metadata_fields(self) -> None:
        """Non-field_map fields (e.g. 'category') are forwarded in the NDJSON payload.

        Regression guard for CI-0058: if the SDK were to strip non-embedded fields
        before sending, metadata filters would silently return zero hits because the
        backend would have nothing to filter on.
        """
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[
                {"_id": "sci-1", "text": "Quantum mechanics.", "category": "science"},
                {"_id": "hist-1", "text": "The Roman Empire.", "category": "history"},
            ],
        )

        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        assert parsed_0["category"] == "science"
        assert parsed_0["text"] == "Quantum mechanics."
        assert parsed_1["category"] == "history"
        assert parsed_1["text"] == "The Roman Empire."

    @respx.mock
    def test_upsert_records_response_bracket_access(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}],
        )

        assert result["record_count"] == 1
