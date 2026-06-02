"""Shared helpers for sync and async data plane clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pinecone._internal.config import normalize_host
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.search import SearchQuery
from pinecone.models.vectors.vector import Vector


def _validate_host(host: str) -> str:
    """Validate and normalize an index host URL.

    Raises:
        ValidationError: If the host is empty or does not look like a real hostname.
    """
    if not host or not host.strip():
        raise ValidationError("host must be a non-empty string")
    normalized = normalize_host(host.strip())
    # Strip scheme for the dot/localhost check
    bare = normalized
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if "." not in bare and "localhost" not in bare.lower():
        raise ValidationError(
            f"host {host!r} does not appear to be a valid URL (must contain a dot or 'localhost')"
        )
    return normalized


_ALLOWED_SEARCH_VECTOR_KEYS: frozenset[str] = frozenset(
    {"values", "sparse_indices", "sparse_values"}
)


def _normalize_search_vector_dict(vector: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a sparse/hybrid search vector dict.

    Raises:
        PineconeValueError: when *vector* contains a key outside the supported set
            (``values``, ``sparse_indices``, ``sparse_values``).
    """
    unknown = set(vector.keys()) - _ALLOWED_SEARCH_VECTOR_KEYS
    if unknown:
        raise PineconeValueError(
            f"Unsupported keys in search vector dict: {sorted(unknown)}. "
            f"Allowed keys are {sorted(_ALLOWED_SEARCH_VECTOR_KEYS)}."
        )
    result: dict[str, Any] = {}
    if "values" in vector:
        result["values"] = list(vector["values"])
    if "sparse_indices" in vector:
        result["sparse_indices"] = list(vector["sparse_indices"])
    if "sparse_values" in vector:
        result["sparse_values"] = list(vector["sparse_values"])
    return result


def _legacy_search_query_to_dict(query: Any) -> dict[str, Any]:
    if hasattr(query, "to_dict") and callable(query.to_dict):
        raw = query.to_dict()
    elif hasattr(query, "as_dict") and callable(query.as_dict):
        raw = query.as_dict()
    else:
        raw = dict(query)
    return dict(raw)


def _build_search_records_body(
    *,
    method_name: str = "search",
    top_k: int | None,
    inputs: Mapping[str, Any] | None,
    vector: Sequence[float] | Mapping[str, Any] | None,
    id: str | None,
    filter: Mapping[str, Any] | None,
    fields: Sequence[str] | None,
    rerank: Mapping[str, Any] | None,
    match_terms: Mapping[str, Any] | None,
    query: Any | None,
    wrap_dense_vector: bool = True,
) -> dict[str, Any]:
    if rerank is not None:
        if "model" not in rerank:
            raise ValidationError("rerank requires 'model' to be specified")
        if "rank_fields" not in rerank:
            raise ValidationError("rerank requires 'rank_fields' to be specified")

    if query is not None:
        conflicting = [
            name
            for name, value in (
                ("top_k", top_k),
                ("inputs", inputs),
                ("vector", vector),
                ("id", id),
                ("filter", filter),
                ("match_terms", match_terms),
            )
            if value is not None
        ]
        if conflicting:
            raise TypeError(
                f"{method_name}() received both 'query=' and {conflicting!r}. "
                "Pass either the legacy 'query=SearchQuery(...)' form OR the new "
                "flat keyword arguments, not both."
            )
        if not isinstance(query, (SearchQuery, Mapping)):
            raise TypeError(
                f"{method_name}() 'query=' must be a SearchQuery or Mapping, "
                f"got {type(query).__name__}"
            )
        query_body = _legacy_search_query_to_dict(query)
        if "vector" in query_body and query_body["vector"] is not None:
            query_vector = query_body["vector"]
            if isinstance(query_vector, Mapping):
                query_body["vector"] = _normalize_search_vector_dict(query_vector)
            else:
                values = list(query_vector)
                query_body["vector"] = {"values": values} if wrap_dense_vector else values
    else:
        if top_k is None:
            raise ValidationError("top_k is required unless query is provided")
        query_body = {"top_k": top_k}
        if inputs is not None:
            query_body["inputs"] = inputs
        if vector is not None:
            if isinstance(vector, Mapping):
                query_body["vector"] = _normalize_search_vector_dict(vector)
            else:
                values = list(vector)
                query_body["vector"] = {"values": values} if wrap_dense_vector else values
        if id is not None:
            query_body["id"] = id
        if filter is not None:
            query_body["filter"] = filter
        if match_terms is not None:
            query_body["match_terms"] = match_terms

    top_k_value = query_body.get("top_k")
    if not isinstance(top_k_value, int) or top_k_value < 1:
        raise ValidationError(f"top_k must be a positive integer, got {top_k_value}")
    if (
        query_body.get("inputs") is None
        and query_body.get("vector") is None
        and query_body.get("id") is None
    ):
        raise ValidationError(
            "At least one of inputs, vector, or id must be provided as a query source"
        )

    body: dict[str, Any] = {"query": query_body}
    if fields is not None:
        body["fields"] = fields
    if rerank is not None:
        body["rerank"] = rerank
    return body


def _vector_to_dict(v: Vector) -> dict[str, Any]:
    """Serialize a Vector to a dict matching the API wire format."""
    id_ = v.id
    vals = v.values
    if v.sparse_values is not None:
        sv = v.sparse_values
        sv_dict: dict[str, Any] = {"indices": sv.indices, "values": sv.values}
        if v.metadata is not None:
            return {"id": id_, "values": vals, "sparseValues": sv_dict, "metadata": v.metadata}
        return {"id": id_, "values": vals, "sparseValues": sv_dict}
    if v.metadata is not None:
        return {"id": id_, "values": vals, "metadata": v.metadata}
    return {"id": id_, "values": vals}
