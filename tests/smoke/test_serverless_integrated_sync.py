"""Priority-4 smoke test — sync serverless integrated (auto-embed).

Punchlist coverage (sync):

- pc.indexes.create_for_model
- Index.upsert_records
- Index.search
- Index.search_records (alias of search)
"""

from __future__ import annotations

import pytest

from pinecone import EmbedConfig, Pinecone
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
def test_serverless_integrated_smoke(client: Pinecone) -> None:
    """Create an integrated index, upsert text records, search."""
    name = unique_name(f"{SMOKE_PREFIX}-srv-int")

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

        idx = client.index(name=name)
        try:
            records = [
                {"_id": "d1", "chunk_text": "Vector databases enable similarity search at scale."},
                {"_id": "d2", "chunk_text": "RAG combines retrieval with large language models."},
                {"_id": "d3", "chunk_text": "Cooking pasta requires boiling water properly."},
                {
                    "_id": "d4",
                    "chunk_text": "Embeddings represent text as high-dimensional vectors.",
                },
            ]
            r = idx.upsert_records(namespace=NS, records=records)
            assert r.record_count == 4
            known_ids = {record["_id"] for record in records}

            wait_for_vector_count(idx, NS, expected=4)

            # search via inputs (auto-embed)
            response = idx.search(
                namespace=NS,
                top_k=3,
                inputs={"text": "vector search and retrieval augmented generation"},
            )
            assert response.result is not None
            hits = response.result.hits
            assert len(hits) > 0
            # Never assert order (#368: query results are unsorted server-side) —
            # just that every returned id is one we actually upserted.
            assert {hit.id for hit in hits} <= known_ids

            # search_records is an alias — same behavior
            alias = idx.search_records(
                namespace=NS,
                top_k=2,
                inputs={"text": "what is RAG?"},
            )
            assert alias.result is not None
            assert len(alias.result.hits) > 0
            assert {hit.id for hit in alias.result.hits} <= known_ids
        finally:
            idx.close()
    finally:
        ensure_index_deleted(client, name)
        client.close()
