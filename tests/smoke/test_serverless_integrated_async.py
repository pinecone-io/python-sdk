"""Priority-4 smoke test — async serverless integrated (auto-embed).

Mirror of ``test_serverless_integrated_sync.py`` against AsyncPinecone /
AsyncIndex.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone, EmbedConfig
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import async_wait_for_vector_count

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "multilingual-e5-large"
NS = "docs"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_serverless_integrated_smoke_async(api_key: str) -> None:
    pc = AsyncPinecone(api_key=api_key)
    name = unique_name(f"{SMOKE_PREFIX}-srv-int-async")

    try:
        await pc.indexes.create_for_model(
            name=name,
            cloud=CLOUD,
            region=REGION,
            embed=EmbedConfig(
                model=EMBED_MODEL,
                field_map={"text": "chunk_text"},
            ),
        )

        idx = await pc.index(name=name)
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
            r = await idx.upsert_records(namespace=NS, records=records)
            assert r.record_count == 4
            known_ids = {record["_id"] for record in records}

            await async_wait_for_vector_count(idx, NS, expected=4)

            response = await idx.search(
                namespace=NS,
                top_k=3,
                inputs={"text": "vector search and retrieval augmented generation"},
            )
            assert response.result is not None
            assert len(response.result.hits) > 0
            assert {hit.id for hit in response.result.hits} <= known_ids

            alias = await idx.search_records(
                namespace=NS,
                top_k=2,
                inputs={"text": "what is RAG?"},
            )
            assert alias.result is not None
            assert len(alias.result.hits) > 0
            assert {hit.id for hit in alias.result.hits} <= known_ids
        finally:
            await idx.close()
    finally:
        await async_ensure_index_deleted(pc, name)
        await pc.close()
