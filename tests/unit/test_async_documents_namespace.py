"""Async mirror of ``tests/unit/test_documents_namespace.py`` (#494).

Same coverage, on ``AsyncIndex``/``AsyncDocuments``: lazy import of the
namespace-implementation modules and caching of ``.documents``. See the sync
file's module docstring for the fuller rationale.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator

import pytest

from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.documents import AsyncDocuments

INDEX_HOST = "async-documents-namespace-abc123.svc.us-east-1-aws.pinecone.io"


@pytest.fixture
async def index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


class TestLazyImport:
    def test_async_index_construction_does_not_import_documents_modules(self) -> None:
        script = (
            "import sys\n"
            "from pinecone.async_client.async_index import AsyncIndex\n"
            f"AsyncIndex(host={INDEX_HOST!r}, api_key='test-key')\n"
            "hits = [m for m in sys.modules if 'documents' in m]\n"
            "print(','.join(sorted(hits)))\n"
        )
        result = subprocess.run(  # noqa: S603 — fixed argv, sys.executable is trusted
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "", (
            "AsyncIndex() must not import any documents-implementation module; "
            f"found: {result.stdout.strip()}"
        )

    def test_documents_access_imports_the_namespace_module(self) -> None:
        script = (
            "import sys\n"
            "from pinecone.async_client.async_index import AsyncIndex\n"
            f"idx = AsyncIndex(host={INDEX_HOST!r}, api_key='test-key')\n"
            "idx.documents\n"
            "print('pinecone.async_client.documents' in sys.modules)\n"
        )
        result = subprocess.run(  # noqa: S603 — fixed argv, sys.executable is trusted
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "True"

    async def test_documents_property_is_cached(self, index: AsyncIndex) -> None:
        first = index.documents
        second = index.documents
        assert first is second
        assert isinstance(first, AsyncDocuments)
