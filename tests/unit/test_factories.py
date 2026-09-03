"""Smoke tests verifying each factory returns the expected top-level keys."""

from __future__ import annotations

from tests.factories import (
    make_backup_response,
    make_collection_response,
    make_describe_index_stats_response,
    make_embed_response,
    make_error_response,
    make_fetch_response,
    make_index_list_response,
    make_index_response,
    make_query_response,
    make_rerank_response,
    make_upsert_response,
)


def test_make_index_response_keys() -> None:
    result = make_index_response()
    assert set(result.keys()) >= {
        "name",
        "host",
        "deployment",
        "schema",
        "status",
        "deletion_protection",
    }


def test_make_index_list_response_keys() -> None:
    result = make_index_list_response()
    assert "indexes" in result
    assert isinstance(result["indexes"], list)


def test_make_collection_response_keys() -> None:
    result = make_collection_response()
    assert set(result.keys()) >= {
        "name",
        "size",
        "status",
        "dimension",
        "vector_count",
        "environment",
    }


def test_make_backup_response_keys() -> None:
    result = make_backup_response()
    assert set(result.keys()) >= {
        "backup_id",
        "source_index_name",
        "source_index_id",
        "status",
        "cloud",
        "region",
    }


def test_make_upsert_response_keys() -> None:
    result = make_upsert_response()
    assert "upsertedCount" in result
    assert isinstance(result["upsertedCount"], int)


def test_make_query_response_keys() -> None:
    result = make_query_response()
    assert "matches" in result
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) > 0
    match = result["matches"][0]
    assert "id" in match
    assert "score" in match


def test_make_fetch_response_keys() -> None:
    result = make_fetch_response()
    assert "vectors" in result
    assert isinstance(result["vectors"], dict)
    assert "namespace" in result
    assert "usage" in result


def test_make_embed_response_keys() -> None:
    result = make_embed_response()
    assert set(result.keys()) >= {"model", "vector_type", "data", "usage"}
    assert isinstance(result["data"], list)


def test_make_rerank_response_keys() -> None:
    result = make_rerank_response()
    assert set(result.keys()) >= {"model", "data", "usage"}
    assert isinstance(result["data"], list)
    assert len(result["data"]) > 0
    doc = result["data"][0]
    assert "index" in doc
    assert "score" in doc


def test_make_error_response_keys() -> None:
    result = make_error_response(status_code=404, message="Not found")
    assert result["status"] == 404
    assert result["error"]["code"] == "NOT_FOUND"
    assert result["error"]["message"] == "Not found"


def test_make_error_response_default() -> None:
    result = make_error_response()
    assert result["status"] == 500
    assert result["error"]["code"] == "UNKNOWN"


def test_make_describe_index_stats_response_keys() -> None:
    result = make_describe_index_stats_response()
    assert "namespaces" in result
    assert isinstance(result["namespaces"], dict)
    assert "dimension" in result
    assert "indexFullness" in result
    assert "totalVectorCount" in result


def test_factory_overrides() -> None:
    """Factories accept **overrides to customise fields."""
    result = make_index_response(name="custom-name", dimension=768)
    assert result["name"] == "custom-name"
    assert result["dimension"] == 768
