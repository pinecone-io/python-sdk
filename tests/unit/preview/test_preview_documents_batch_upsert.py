"""Unit tests for PreviewDocuments.batch_upsert and AsyncPreviewDocuments.batch_upsert."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ValidationError
from pinecone.models.batch import BatchResult
from pinecone.models.response_info import ResponseInfo
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.documents import PreviewDocuments
from pinecone.preview.models.documents import PreviewDocumentUpsertResponse

INDEX_HOST = "https://idx-abc.svc.pinecone.io"

_UPSERT_RESPONSE = PreviewDocumentUpsertResponse(upserted_count=100)


def _make_docs(n: int) -> list[dict[str, Any]]:
    return [{"_id": str(i)} for i in range(n)]


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key")


@pytest.fixture
def docs(config: PineconeConfig) -> PreviewDocuments:
    return PreviewDocuments(config=config, host=INDEX_HOST)


@pytest.fixture
def async_docs(config: PineconeConfig) -> AsyncPreviewDocuments:
    return AsyncPreviewDocuments(config=config, host=INDEX_HOST)


# ---------------------------------------------------------------------------
# Sync — batching
# ---------------------------------------------------------------------------


def test_batch_upsert_250_calls_upsert_three_times(docs: PreviewDocuments) -> None:
    documents = _make_docs(250)

    with patch.object(docs, "upsert", return_value=_UPSERT_RESPONSE) as mock_upsert:
        result = docs.batch_upsert(
            namespace="ns",
            documents=documents,
            batch_size=100,
            max_concurrency=1,
            show_progress=False,
        )

    assert mock_upsert.call_count == 3
    sizes = sorted(len(c.kwargs["documents"]) for c in mock_upsert.call_args_list)
    assert sizes == [50, 100, 100]
    assert isinstance(result, BatchResult)
    assert result.total_item_count == 250
    assert result.total_batch_count == 3


def test_batch_upsert_show_progress_propagated(docs: PreviewDocuments) -> None:
    with (
        patch.object(docs, "upsert", return_value=_UPSERT_RESPONSE),
        patch("pinecone._internal.bulk.engine._create_progress_bar") as mock_bar,
    ):
        bar = MagicMock()
        mock_bar.return_value = bar
        docs.batch_upsert(
            namespace="ns",
            documents=_make_docs(10),
            show_progress=False,
        )
        mock_bar.assert_called_once_with(1, "Upserting", False)


def test_batch_upsert_partial_failure_captured(docs: PreviewDocuments) -> None:
    documents = _make_docs(30)
    call_count_holder = [0]

    def _side_effect(
        *, namespace: str, documents: list[dict[str, Any]]
    ) -> PreviewDocumentUpsertResponse:
        call_count_holder[0] += 1
        if call_count_holder[0] == 2:
            raise Exception("server error")
        return PreviewDocumentUpsertResponse(upserted_count=len(documents))

    with patch.object(docs, "upsert", side_effect=_side_effect):
        result = docs.batch_upsert(
            namespace="ns",
            documents=documents,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )

    assert result.has_errors is True
    assert result.failed_item_count == 10
    assert len(result.failed_items) == 10


# ---------------------------------------------------------------------------
# Sync — validation
# ---------------------------------------------------------------------------


def test_batch_upsert_empty_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.batch_upsert(namespace="", documents=[{"_id": "a"}])


def test_batch_upsert_whitespace_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.batch_upsert(namespace="   ", documents=[{"_id": "a"}])


def test_batch_upsert_empty_documents_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="documents"):
        docs.batch_upsert(namespace="ns", documents=[])


def test_batch_upsert_batch_size_zero_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError):
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], batch_size=0)


def test_batch_upsert_max_concurrency_zero_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError):
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_concurrency=0)


def test_batch_upsert_max_concurrency_65_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError):
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_concurrency=65)


# ---------------------------------------------------------------------------
# Async — batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_upsert_250_calls_upsert_three_times(
    async_docs: AsyncPreviewDocuments,
) -> None:
    documents = _make_docs(250)

    with patch.object(
        async_docs, "upsert", new_callable=AsyncMock, return_value=_UPSERT_RESPONSE
    ) as mock_upsert:
        result = await async_docs.batch_upsert(
            namespace="ns",
            documents=documents,
            batch_size=100,
            max_concurrency=1,
            show_progress=False,
        )

    assert mock_upsert.call_count == 3
    sizes = sorted(len(c.kwargs["documents"]) for c in mock_upsert.call_args_list)
    assert sizes == [50, 100, 100]
    assert isinstance(result, BatchResult)
    assert result.total_item_count == 250
    assert result.total_batch_count == 3


@pytest.mark.asyncio
async def test_async_batch_upsert_show_progress_propagated(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with (
        patch.object(async_docs, "upsert", new_callable=AsyncMock, return_value=_UPSERT_RESPONSE),
        patch("pinecone._internal.batch._create_progress_bar") as mock_bar,
    ):
        bar = MagicMock()
        mock_bar.return_value = bar
        await async_docs.batch_upsert(
            namespace="ns",
            documents=_make_docs(10),
            show_progress=False,
        )
        mock_bar.assert_called_once_with(1, "Upserting", False)


@pytest.mark.asyncio
async def test_async_batch_upsert_partial_failure_captured(
    async_docs: AsyncPreviewDocuments,
) -> None:
    documents = _make_docs(30)
    call_count_holder = [0]

    async def _side_effect(
        *, namespace: str, documents: list[dict[str, Any]]
    ) -> PreviewDocumentUpsertResponse:
        call_count_holder[0] += 1
        if call_count_holder[0] == 2:
            raise Exception("server error")
        return PreviewDocumentUpsertResponse(upserted_count=len(documents))

    with patch.object(async_docs, "upsert", side_effect=_side_effect):
        result = await async_docs.batch_upsert(
            namespace="ns",
            documents=documents,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )

    assert result.has_errors is True
    assert result.failed_item_count == 10
    assert len(result.failed_items) == 10


# ---------------------------------------------------------------------------
# Async — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_upsert_empty_namespace_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        await async_docs.batch_upsert(namespace="", documents=[{"_id": "a"}])


@pytest.mark.asyncio
async def test_async_batch_upsert_empty_documents_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="documents"):
        await async_docs.batch_upsert(namespace="ns", documents=[])


@pytest.mark.asyncio
async def test_async_batch_upsert_max_concurrency_zero_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError):
        await async_docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_concurrency=0)


# ---------------------------------------------------------------------------
# max_workers alias — sync
# ---------------------------------------------------------------------------


def test_batch_upsert_max_workers_alias_accepted_sync(docs: PreviewDocuments) -> None:
    with patch("pinecone.preview.documents.bulk_execute_sync") as mock_execute:
        mock_execute.return_value = BatchResult(
            total_item_count=1,
            successful_item_count=1,
            failed_item_count=0,
            total_batch_count=1,
            successful_batch_count=1,
            failed_batch_count=0,
            errors=[],
            response_info=None,
        )
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workers=8)
    mock_execute.assert_called_once()
    assert mock_execute.call_args.kwargs["max_concurrency"] == 8


def test_batch_upsert_max_workers_and_max_concurrency_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="both"):
        docs.batch_upsert(
            namespace="ns",
            documents=[{"_id": "a"}],
            max_concurrency=4,
            max_workers=8,
        )


def test_batch_upsert_unknown_kwarg_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="max_workerz"):
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workerz=8)


def test_batch_upsert_max_workers_alias_zero_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="max_concurrency"):
        docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workers=0)


# ---------------------------------------------------------------------------
# max_workers alias — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_upsert_max_workers_alias_accepted(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with patch(
        "pinecone.preview.async_documents.async_batch_execute", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = BatchResult(
            total_item_count=1,
            successful_item_count=1,
            failed_item_count=0,
            total_batch_count=1,
            successful_batch_count=1,
            failed_batch_count=0,
            errors=[],
            response_info=None,
        )
        await async_docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workers=8)
    mock_execute.assert_called_once()
    assert mock_execute.call_args.kwargs["max_concurrency"] == 8


@pytest.mark.asyncio
async def test_async_batch_upsert_max_workers_and_max_concurrency_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="both"):
        await async_docs.batch_upsert(
            namespace="ns",
            documents=[{"_id": "a"}],
            max_concurrency=4,
            max_workers=8,
        )


@pytest.mark.asyncio
async def test_async_batch_upsert_unknown_kwarg_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="max_workerz"):
        await async_docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workerz=8)


@pytest.mark.asyncio
async def test_async_batch_upsert_max_workers_alias_zero_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="max_concurrency"):
        await async_docs.batch_upsert(namespace="ns", documents=[{"_id": "a"}], max_workers=0)


# ---------------------------------------------------------------------------
# response_info aggregation — sync
# ---------------------------------------------------------------------------


class TestBatchUpsertResponseInfo:
    def test_batch_upsert_aggregates_response_info(self, docs: PreviewDocuments) -> None:
        counter = [0]

        def _side_effect(
            *, namespace: str, documents: list[dict[str, Any]]
        ) -> PreviewDocumentUpsertResponse:
            counter[0] += 1
            i = counter[0]
            return PreviewDocumentUpsertResponse(
                upserted_count=len(documents),
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": str(i * 10)}),
            )

        with patch.object(docs, "upsert", side_effect=_side_effect):
            result = docs.batch_upsert(
                namespace="ns",
                documents=_make_docs(30),
                batch_size=10,
                max_concurrency=1,
                show_progress=False,
            )

        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.is_reconciled(30) is True
        assert result.response_info.is_reconciled(31) is False

    def test_batch_upsert_response_info_none_when_no_lsn_headers(
        self, docs: PreviewDocuments
    ) -> None:
        def _side_effect(
            *, namespace: str, documents: list[dict[str, Any]]
        ) -> PreviewDocumentUpsertResponse:
            return PreviewDocumentUpsertResponse(
                upserted_count=len(documents),
                response_info=ResponseInfo(raw_headers={"x-pinecone-request-id": "req-123"}),
            )

        with patch.object(docs, "upsert", side_effect=_side_effect):
            result = docs.batch_upsert(
                namespace="ns",
                documents=_make_docs(20),
                batch_size=10,
                max_concurrency=1,
                show_progress=False,
            )

        assert result.response_info is None


# ---------------------------------------------------------------------------
# response_info aggregation — async
# ---------------------------------------------------------------------------


class TestAsyncBatchUpsertResponseInfo:
    @pytest.mark.asyncio
    async def test_async_batch_upsert_aggregates_response_info(
        self, async_docs: AsyncPreviewDocuments
    ) -> None:
        counter = [0]

        async def _side_effect(
            *, namespace: str, documents: list[dict[str, Any]]
        ) -> PreviewDocumentUpsertResponse:
            counter[0] += 1
            i = counter[0]
            return PreviewDocumentUpsertResponse(
                upserted_count=len(documents),
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": str(i * 10)}),
            )

        with patch.object(async_docs, "upsert", side_effect=_side_effect):
            result = await async_docs.batch_upsert(
                namespace="ns",
                documents=_make_docs(30),
                batch_size=10,
                max_concurrency=1,
                show_progress=False,
            )

        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.is_reconciled(30) is True
        assert result.response_info.is_reconciled(31) is False

    @pytest.mark.asyncio
    async def test_async_batch_upsert_response_info_none_when_no_lsn_headers(
        self, async_docs: AsyncPreviewDocuments
    ) -> None:
        async def _side_effect(
            *, namespace: str, documents: list[dict[str, Any]]
        ) -> PreviewDocumentUpsertResponse:
            return PreviewDocumentUpsertResponse(
                upserted_count=len(documents),
                response_info=ResponseInfo(raw_headers={"x-pinecone-request-id": "req-123"}),
            )

        with patch.object(async_docs, "upsert", side_effect=_side_effect):
            result = await async_docs.batch_upsert(
                namespace="ns",
                documents=_make_docs(20),
                batch_size=10,
                max_concurrency=1,
                show_progress=False,
            )

        assert result.response_info is None
