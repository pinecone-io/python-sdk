"""Tests for the restored ``Index.documents`` / ``AsyncIndex.documents`` namespace (#494).

Preview shipped documents as a namespace (``index.documents.upsert(...)``). The
2026-07 graduation flattened it into suffixed methods (``index.upsert_documents(...)``)
and dropped the namespace entirely, which turned out to be an accidental side effect
of retiring the ``PreviewIndex`` wrapper rather than a deliberate design choice. This
suite pins the fix: the namespace is restored as the sole surface, lazily constructed
like every other resource namespace on the client (``pinecone/_client.py``'s
``indexes``/``inference``/etc.).

Covers:
    * The namespace-implementation modules are not imported until ``.documents`` is
      first accessed — not at ``Index``/``AsyncIndex`` construction.
    * ``.documents`` is cached — the same instance comes back on repeat access.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import pytest

from pinecone.client.documents import Documents
from pinecone.index import Index

INDEX_HOST = "documents-namespace-abc123.svc.us-east-1-aws.pinecone.io"


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


class TestLazyImport:
    """The documents-implementation modules load only on first ``.documents`` access."""

    def test_index_construction_does_not_import_documents_modules(self) -> None:
        script = (
            "import sys\n"
            "from pinecone.index import Index\n"
            f"Index(host={INDEX_HOST!r}, api_key='test-key')\n"
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
            "Index() must not import any documents-implementation module; "
            f"found: {result.stdout.strip()}"
        )

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
            "from pinecone.index import Index\n"
            f"idx = Index(host={INDEX_HOST!r}, api_key='test-key')\n"
            "idx.documents\n"
            "print('pinecone.client.documents' in sys.modules)\n"
        )
        result = subprocess.run(  # noqa: S603 — fixed argv, sys.executable is trusted
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "True"

    def test_documents_property_is_cached(self, index: Index) -> None:
        first = index.documents
        second = index.documents
        assert first is second
        assert isinstance(first, Documents)
