"""Unit tests for Index.upsert_from_dataframe() method."""

from __future__ import annotations

import inspect
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone import Index
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.vectors.responses import UpsertResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _make_upsert_response(*, upserted_count: int = 3) -> UpsertResponse:
    return UpsertResponse(upserted_count=upserted_count)


class TestUpsertFromDataframeBasic:
    """Basic upsert_from_dataframe functionality."""

    def test_upsert_from_dataframe_basic(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3"],
                "values": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=3))  # type: ignore[method-assign]

        result = idx.upsert_from_dataframe(df)

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
        idx.upsert.assert_called_once()
        call_kwargs = idx.upsert.call_args[1]
        assert len(call_kwargs["vectors"]) == 3
        assert call_kwargs["vectors"][0] == {"id": "v1", "values": [0.1, 0.2]}
        assert call_kwargs["vectors"][1] == {"id": "v2", "values": [0.3, 0.4]}
        assert call_kwargs["vectors"][2] == {"id": "v3", "values": [0.5, 0.6]}

    def test_upsert_from_dataframe_with_metadata(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2"],
                "values": [[0.1, 0.2], [0.3, 0.4]],
                "metadata": [{"genre": "rock"}, {"genre": "pop"}],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=2))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df)

        call_kwargs = idx.upsert.call_args[1]
        assert call_kwargs["vectors"][0]["metadata"] == {"genre": "rock"}
        assert call_kwargs["vectors"][1]["metadata"] == {"genre": "pop"}

    def test_upsert_from_dataframe_with_sparse_values(self) -> None:
        pd = pytest.importorskip("pandas")
        sparse = {"indices": [0, 2], "values": [0.5, 0.8]}
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "values": [[0.1, 0.2]],
                "sparse_values": [sparse],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=1))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df)

        call_kwargs = idx.upsert.call_args[1]
        assert call_kwargs["vectors"][0]["sparse_values"] == sparse


class TestUpsertFromDataframeMissingCells:
    """Rows that omit an optional key leave NaN behind, not None."""

    def test_nan_metadata_is_treated_as_absent(self) -> None:
        """A NaN metadata cell used to surface as `metadata must be a dict, got float`."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "metadata": {"genre": "rock"}},
                {"id": "v2", "values": [0.2]},
            ]
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=2))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df, show_progress=False)

        first, second = idx.upsert.call_args[1]["vectors"]
        assert first["metadata"] == {"genre": "rock"}
        assert "metadata" not in second

    def test_nan_sparse_values_is_treated_as_absent(self) -> None:
        pd = pytest.importorskip("pandas")
        sparse = {"indices": [1], "values": [0.5]}
        df = pd.DataFrame(
            [
                {"id": "v1", "values": [0.1], "sparse_values": sparse},
                {"id": "v2", "values": [0.2]},
            ]
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=2))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df, show_progress=False)

        first, second = idx.upsert.call_args[1]["vectors"]
        assert first["sparse_values"] == sparse
        assert "sparse_values" not in second


class TestUpsertFromDataframeBatching:
    """Batching behavior."""

    def test_upsert_from_dataframe_delegates_default_batch_size(self) -> None:
        """Batching is delegated to upsert(batch_size=...), not done here."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(1200)],
                "values": [[float(i)] for i in range(1200)],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=1200))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df, show_progress=False)

        idx.upsert.assert_called_once()
        call_kwargs = idx.upsert.call_args[1]
        assert len(call_kwargs["vectors"]) == 1200
        assert call_kwargs["batch_size"] == 500

    def test_upsert_from_dataframe_custom_batch_size(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(10)],
                "values": [[float(i)] for i in range(10)],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=10))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df, batch_size=3, show_progress=False)

        idx.upsert.assert_called_once()
        call_kwargs = idx.upsert.call_args[1]
        assert call_kwargs["batch_size"] == 3
        assert len(call_kwargs["vectors"]) == 10

    def test_upsert_from_dataframe_namespace(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "values": [[0.1, 0.2]],
            }
        )
        idx = _make_index()
        idx.upsert = MagicMock(return_value=_make_upsert_response(upserted_count=1))  # type: ignore[method-assign]

        idx.upsert_from_dataframe(df, namespace="my-ns")

        call_kwargs = idx.upsert.call_args[1]
        assert call_kwargs["namespace"] == "my-ns"


class TestUpsertFromDataframeDefaults:
    """Default values and aggregation."""

    def test_upsert_from_dataframe_default_batch_500(self) -> None:
        sig = inspect.signature(Index.upsert_from_dataframe)
        assert sig.parameters["batch_size"].default == 500

    def test_upsert_from_dataframe_timeout_defaults_to_none(self) -> None:
        sig = inspect.signature(Index.upsert_from_dataframe)
        assert sig.parameters["timeout"].default is None

    def test_signature_parity_across_transports(self) -> None:
        """All three variants expose the same parameter names in the same order."""
        sync_params = list(inspect.signature(Index.upsert_from_dataframe).parameters)
        async_params = list(inspect.signature(AsyncIndex.upsert_from_dataframe).parameters)
        assert sync_params == async_params

    def test_upsert_from_dataframe_returns_upsert_result(self) -> None:
        """The aggregate upsert() computes across batches is returned unchanged."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(100)],
                "values": [[float(i)] for i in range(100)],
            }
        )
        idx = _make_index()
        aggregate = UpsertResponse(
            upserted_count=80,
            total_item_count=100,
            failed_item_count=20,
        )
        idx.upsert = MagicMock(return_value=aggregate)  # type: ignore[method-assign]

        result = idx.upsert_from_dataframe(df, batch_size=50, show_progress=False)

        assert result is aggregate
        assert result.upserted_count == 80
        assert result.failed_item_count == 20


class TestUpsertFromDataframeErrors:
    """Error handling."""

    def test_upsert_from_dataframe_not_a_dataframe(self) -> None:
        pytest.importorskip("pandas")
        idx = _make_index()

        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            idx.upsert_from_dataframe([1, 2, 3])

    def test_batch_size_zero_raises_value_error(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1, 0.2]]})
        idx = _make_index()

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            idx.upsert_from_dataframe(df, batch_size=0)

    def test_batch_size_negative_raises_value_error(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1, 0.2]]})
        idx = _make_index()

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            idx.upsert_from_dataframe(df, batch_size=-5)

    def test_batch_size_float_raises_value_error(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1, 0.2]]})
        idx = _make_index()

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            idx.upsert_from_dataframe(df, batch_size=3.5)  # type: ignore[arg-type]

    def test_upsert_from_dataframe_no_pandas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        idx = _make_index()

        # Make 'import pandas' fail inside the method
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr("builtins.__import__", _fake_import)

        with pytest.raises(RuntimeError, match="pip install pandas") as excinfo:
            idx.upsert_from_dataframe("not-a-df")

        message = str(excinfo.value)
        assert "pandas is required" in message
        assert "not a dependency of this SDK" in message


class TestAsyncUpsertFromDataframe:
    """AsyncIndex.upsert_from_dataframe delegates to async upsert (#5)."""

    def _make_async_index(self) -> AsyncIndex:
        return AsyncIndex(host=INDEX_HOST, api_key="test-key")

    @pytest.mark.asyncio
    async def test_async_upsert_from_dataframe_basic(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3"],
                "values": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        )
        async_idx = self._make_async_index()
        async_idx.upsert = AsyncMock(return_value=_make_upsert_response(upserted_count=3))  # type: ignore[method-assign]

        result = await async_idx.upsert_from_dataframe(df)

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
        call_kwargs = async_idx.upsert.call_args[1]
        assert len(call_kwargs["vectors"]) == 3
        assert call_kwargs["vectors"][0] == {"id": "v1", "values": [0.1, 0.2]}
        assert call_kwargs["batch_size"] == 500

    @pytest.mark.asyncio
    async def test_async_forwards_all_knobs(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        async_idx = self._make_async_index()
        async_idx.upsert = AsyncMock(return_value=_make_upsert_response(upserted_count=1))  # type: ignore[method-assign]

        await async_idx.upsert_from_dataframe(
            df,
            namespace="my-ns",
            batch_size=7,
            show_progress=False,
            timeout=30.0,
            max_concurrency=3,
            total_timeout=120.0,
        )

        call_kwargs = async_idx.upsert.call_args[1]
        assert call_kwargs["namespace"] == "my-ns"
        assert call_kwargs["batch_size"] == 7
        assert call_kwargs["show_progress"] is False
        assert call_kwargs["timeout"] == 30.0
        assert call_kwargs["max_concurrency"] == 3
        assert call_kwargs["total_timeout"] == 120.0

    @pytest.mark.asyncio
    async def test_async_max_concurrency_defaults_to_flat_constant(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        async_idx = self._make_async_index()
        async_idx.upsert = AsyncMock(return_value=_make_upsert_response(upserted_count=1))  # type: ignore[method-assign]

        await async_idx.upsert_from_dataframe(df)

        assert async_idx.upsert.call_args[1]["max_concurrency"] == DEFAULT_MAX_CONCURRENCY

    @pytest.mark.asyncio
    async def test_async_not_a_dataframe_raises(self) -> None:
        pytest.importorskip("pandas")
        async_idx = self._make_async_index()

        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            await async_idx.upsert_from_dataframe("dummy")

    @pytest.mark.asyncio
    async def test_async_batch_size_zero_raises(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        async_idx = self._make_async_index()

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            await async_idx.upsert_from_dataframe(df, batch_size=0)

    @pytest.mark.asyncio
    async def test_async_on_error_raise_rethrows_lowest_batch_index(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1], [0.2]]})
        async_idx = self._make_async_index()
        err_low = RuntimeError("batch 1 failed")
        err_high = RuntimeError("batch 3 failed")
        response = UpsertResponse(
            upserted_count=1,
            total_item_count=2,
            failed_item_count=1,
            errors=[
                SimpleNamespace(batch_index=3, error=err_high),
                SimpleNamespace(batch_index=1, error=err_low),
            ],
        )
        async_idx.upsert = AsyncMock(return_value=response)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="batch 1 failed") as excinfo:
            await async_idx.upsert_from_dataframe(df, on_error="raise")

        assert excinfo.value.response is response  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_async_on_error_default_collects(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        async_idx = self._make_async_index()
        response = UpsertResponse(
            upserted_count=0,
            total_item_count=1,
            failed_item_count=1,
            errors=[SimpleNamespace(batch_index=0, error=RuntimeError("boom"))],
        )
        async_idx.upsert = AsyncMock(return_value=response)  # type: ignore[method-assign]

        result = await async_idx.upsert_from_dataframe(df)

        assert result is response


class TestUpsertFromDataframeTransportParity:
    """One signature across REST sync, asyncio, and gRPC (#5, the parity effort)."""

    def test_signature_parity_all_three_transports(self) -> None:
        from pinecone.grpc import GrpcIndex

        sync_params = list(inspect.signature(Index.upsert_from_dataframe).parameters)
        async_params = list(inspect.signature(AsyncIndex.upsert_from_dataframe).parameters)
        grpc_params = list(inspect.signature(GrpcIndex.upsert_from_dataframe).parameters)
        assert sync_params == async_params == grpc_params
