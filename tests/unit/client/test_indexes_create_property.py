"""Property-based tests for the 2026-07 create-index request body (#131).

For arbitrary valid (schema, deployment, read_capacity, tags) inputs:

* the serialized request body contains no null-valued keys at any depth, and
* the body round-trips through CreateIndexRequest without loss.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone.models.indexes.requests import CreateIndexRequest

_field_names = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=16
).filter(lambda s: not s.startswith(("_", "$")))

_dense_fields = st.fixed_dictionaries(
    {
        "type": st.just("dense_vector"),
        "dimension": st.integers(min_value=1, max_value=20000),
        "metric": st.sampled_from(["cosine", "euclidean", "dotproduct"]),
    }
)
_sparse_fields = st.fixed_dictionaries({"type": st.just("sparse_vector")})
_fts_fields = st.fixed_dictionaries(
    {
        "type": st.just("string"),
        "full_text_search": st.fixed_dictionaries(
            {"language": st.sampled_from(["en", "de", "fr"])},
            optional={"stemming": st.booleans(), "stop_words": st.booleans()},
        ),
    }
)

_schemas = st.dictionaries(
    _field_names, st.one_of(_dense_fields, _sparse_fields, _fts_fields), min_size=1, max_size=4
).map(lambda fields: {"fields": fields})

_managed = st.fixed_dictionaries(
    {
        "deployment_type": st.just("managed"),
        "cloud": st.sampled_from(["aws", "gcp", "azure"]),
        "region": st.sampled_from(["us-east-1", "us-central1", "eu-west-1"]),
    }
)
_pod = st.fixed_dictionaries(
    {
        "deployment_type": st.just("pod"),
        "environment": st.just("us-east1-gcp"),
        "pod_type": st.sampled_from(["s1.x1", "p1.x2", "p2.x8"]),
        "replicas": st.integers(min_value=1, max_value=8),
        "shards": st.integers(min_value=1, max_value=8),
    }
)
_byoc = st.fixed_dictionaries(
    {"deployment_type": st.just("byoc"), "environment": st.just("aws-us-east-1-b921")}
)
_deployments = st.one_of(st.none(), _managed, _pod, _byoc)

_on_demand = st.fixed_dictionaries({"mode": st.just("OnDemand")})
_dedicated = st.fixed_dictionaries(
    {
        "mode": st.just("Dedicated"),
        "dedicated": st.fixed_dictionaries(
            {
                "node_type": st.sampled_from(["b1", "t1"]),
                "scaling": st.just("Manual"),
                "manual": st.fixed_dictionaries(
                    {
                        "replicas": st.integers(min_value=1, max_value=4),
                        "shards": st.integers(min_value=1, max_value=4),
                    }
                ),
            }
        ),
    }
)
_read_capacities = st.one_of(st.none(), _on_demand, _dedicated)

_tag_keys = st.text(
    alphabet=st.characters(whitelist_categories=(), whitelist_characters="abcdefgh_-123"),
    min_size=1,
    max_size=80,
)
_tag_values = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=120
)
_tags = st.one_of(st.none(), st.dictionaries(_tag_keys, _tag_values, min_size=1, max_size=20))


def _assert_no_nulls(obj: Any, crumb: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert value is not None, f"null value at {crumb}.{key}"
            _assert_no_nulls(value, f"{crumb}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _assert_no_nulls(value, f"{crumb}[{i}]")


@given(schema=_schemas, deployment=_deployments, read_capacity=_read_capacities, tags=_tags)
def test_create_body_has_no_nulls_and_roundtrips(
    schema: dict[str, Any],
    deployment: dict[str, Any] | None,
    read_capacity: dict[str, Any] | None,
    tags: dict[str, str] | None,
) -> None:
    request = CreateIndexRequest(
        schema=schema,
        name="prop-index",
        deployment=deployment,
        read_capacity=read_capacity,
        tags=tags,
    )

    body = json.loads(IndexesAdapter.to_create_request(request))

    _assert_no_nulls(body)

    assert body["schema"] == schema
    assert body["name"] == "prop-index"
    if deployment is None:
        assert "deployment" not in body
    else:
        assert body["deployment"] == deployment
    if read_capacity is None:
        assert "read_capacity" not in body
    else:
        assert body["read_capacity"] == read_capacity
    if tags is None:
        assert "tags" not in body
    else:
        assert body["tags"] == tags

    decoded = CreateIndexRequest(
        schema=body["schema"],
        name=body.get("name"),
        deployment=body.get("deployment"),
        read_capacity=body.get("read_capacity"),
        tags=body.get("tags"),
    )
    reencoded = json.loads(IndexesAdapter.to_create_request(decoded))
    assert reencoded == body
