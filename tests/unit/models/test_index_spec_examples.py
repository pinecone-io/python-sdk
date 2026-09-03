"""Round-trip tests for the 2026-07 db_control spec response examples.

Each example is copied verbatim from
``apis/_build/2026-07/db_control_2026-07.oas.yaml``.  Decoding must succeed,
and re-encoding must reproduce every key the example carries (host values
gain the ``https://`` scheme the SDK normalizes onto them).
"""

from __future__ import annotations

from typing import Any

import msgspec
import orjson

from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList

LIST_INDEXES_MULTIPLE = {
    "indexes": [
        {
            "deletion_protection": "disabled",
            "deployment": {
                "cloud": "aws",
                "deployment_type": "managed",
                "region": "us-east-1",
            },
            "host": "movie-recommendations-c01b5b5.svc.us-east1-gcp.pinecone.io",
            "name": "movie-recommendations",
            "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
            "schema": {
                "fields": {
                    "embedding": {
                        "dimension": 1536,
                        "metric": "cosine",
                        "type": "dense_vector",
                    }
                }
            },
            "status": {"ready": True, "state": "Ready"},
        }
    ]
}

DESCRIBE_SERVERLESS_DENSE = {
    "deletion_protection": "disabled",
    "deployment": {
        "cloud": "aws",
        "deployment_type": "managed",
        "region": "us-east-1",
    },
    "host": "movie-recommendations-c01b5b5.svc.us-east1-gcp.pinecone.io",
    "name": "movie-recommendations",
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
    "schema": {
        "fields": {"embedding": {"dimension": 1536, "metric": "cosine", "type": "dense_vector"}}
    },
    "status": {"ready": True, "state": "Ready"},
}

DESCRIBE_SERVERLESS_FTS = {
    "deletion_protection": "disabled",
    "deployment": {
        "cloud": "aws",
        "deployment_type": "managed",
        "region": "us-east-1",
    },
    "host": "article-search-d12e6e6.svc.us-east1-gcp.pinecone.io",
    "name": "article-search",
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
    "schema": {
        "fields": {
            "body": {
                "full_text_search": {"language": "en", "stemming": True, "stop_words": True},
                "type": "string",
            },
            "title": {"full_text_search": {"language": "en"}, "type": "string"},
        }
    },
    "status": {"ready": True, "state": "Ready"},
}

CREATE_FOR_MODEL_CREATED = {
    "deletion_protection": "enabled",
    "deployment": {
        "cloud": "aws",
        "deployment_type": "managed",
        "region": "us-east-1",
    },
    "host": "multilingual-e5-large-index-c01b5b5.svc.us-east1.pinecone.io",
    "name": "multilingual-e5-large-index",
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
    "schema": {
        "fields": {
            "content": {
                "metric": "cosine",
                "model": "multilingual-e5-large",
                "read_parameters": {"input_type": "query", "truncate": "NONE"},
                "type": "semantic_text",
                "write_parameters": {"input_type": "passage"},
            }
        }
    },
    "status": {"ready": False, "state": "Initializing"},
}


def _assert_reencodes_known_fields(example: dict[str, Any], model: IndexModel) -> None:
    reencoded = msgspec.to_builtins(model)
    _assert_subset(example, reencoded, path="$")


def _assert_subset(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected object, got {type(actual)}"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key}: missing on re-encode"
            if key in ("host", "private_host") and isinstance(value, str):
                assert actual[key] == f"https://{value}", f"{path}.{key}: {actual[key]!r}"
            else:
                _assert_subset(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: list mismatch"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_subset(e, a, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_describe_index_serverless_dense_example_roundtrip() -> None:
    model = msgspec.json.decode(orjson.dumps(DESCRIBE_SERVERLESS_DENSE), type=IndexModel)
    assert model.name == "movie-recommendations"
    _assert_reencodes_known_fields(DESCRIBE_SERVERLESS_DENSE, model)


def test_describe_index_serverless_fts_example_roundtrip() -> None:
    model = msgspec.json.decode(orjson.dumps(DESCRIBE_SERVERLESS_FTS), type=IndexModel)
    assert model.name == "article-search"
    title = model.schema.fields["title"]
    assert title.full_text_search is not None  # type: ignore[union-attr]
    _assert_reencodes_known_fields(DESCRIBE_SERVERLESS_FTS, model)


def test_create_index_for_model_created_example_roundtrip() -> None:
    model = msgspec.json.decode(orjson.dumps(CREATE_FOR_MODEL_CREATED), type=IndexModel)
    assert model.name == "multilingual-e5-large-index"
    content = model.schema.fields["content"]
    assert content.model == "multilingual-e5-large"  # type: ignore[union-attr]
    assert content.read_parameters == {"input_type": "query", "truncate": "NONE"}  # type: ignore[union-attr]
    _assert_reencodes_known_fields(CREATE_FOR_MODEL_CREATED, model)


def test_list_indexes_example_roundtrip() -> None:
    from pinecone._internal.adapters.indexes_adapter import IndexesAdapter

    result = IndexesAdapter.to_index_list(orjson.dumps(LIST_INDEXES_MULTIPLE))
    assert isinstance(result, IndexList)
    assert result.names() == ["movie-recommendations"]
    _assert_reencodes_known_fields(LIST_INDEXES_MULTIPLE["indexes"][0], result[0])


def test_list_indexes_empty_example() -> None:
    from pinecone._internal.adapters.indexes_adapter import IndexesAdapter

    result = IndexesAdapter.to_index_list(b'{"indexes": []}')
    assert len(result) == 0
