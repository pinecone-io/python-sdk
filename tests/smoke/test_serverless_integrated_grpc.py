"""Priority-4 smoke test — gRPC serverless integrated (auto-embed).

GrpcIndex's ``upsert_records``, ``search``, and ``search_records`` delegate
to the REST endpoint under the hood (the gRPC API does not expose records
ops), but the surface is exercised here for parity coverage.
"""

from __future__ import annotations

import pytest

from pinecone import EmbedConfig, GrpcIndex, Pinecone
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import wait_for_vector_count

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "multilingual-e5-large"
NS = "docs"


@pytest.mark.smoke
def test_serverless_integrated_grpc_smoke(client: Pinecone) -> None:
    name = unique_name(f"{SMOKE_PREFIX}-srv-int-grpc")
    try:
        client.indexes.create_for_model(
            name=name,
            cloud=CLOUD,
            region=REGION,
            embed=EmbedConfig(
                model=EMBED_MODEL,
                field_map={"text": "chunk_text"},
            ),
        )

        idx = client.index(name=name, grpc=True)
        assert isinstance(idx, GrpcIndex)
        try:
            records = [
                {"_id": "g1", "chunk_text": "Vector databases enable similarity search at scale."},
                {"_id": "g2", "chunk_text": "RAG combines retrieval with large language models."},
                {"_id": "g3", "chunk_text": "Embeddings represent text as high-dim vectors."},
            ]
            r = idx.upsert_records(namespace=NS, records=records)
            assert r.record_count == 3
            known_ids = {record["_id"] for record in records}

            wait_for_vector_count(idx, NS, expected=3)

            response = idx.search(
                namespace=NS,
                top_k=2,
                inputs={"text": "vector search"},
            )
            assert response.result is not None
            assert len(response.result.hits) > 0
            assert {hit.id for hit in response.result.hits} <= known_ids

            alias = idx.search_records(
                namespace=NS,
                top_k=2,
                inputs={"text": "vector search"},
            )
            assert alias.result is not None
            assert len(alias.result.hits) > 0
            assert {hit.id for hit in alias.result.hits} <= known_ids
        finally:
            idx.close()
    finally:
        ensure_index_deleted(client, name)
        client.close()
