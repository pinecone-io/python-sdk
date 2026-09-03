"""on_error, and the one-time warning that announces the 10.0.0 change.

The warning is the whole point of "making the break loud": gRPC callers who
relied on the raise get told once, in the run where it first matters, with the
one-word fix in the message.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pinecone.grpc as grpc_module
from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex

pd = pytest.importorskip("pandas")

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Any:
    grpc_module._warned_about_grpc_partial_failure = False
    yield
    grpc_module._warned_about_grpc_partial_failure = False


def _frame(rows: int) -> Any:
    return pd.DataFrame(
        {"id": [f"v{i}" for i in range(rows)], "values": [[float(i)] for i in range(rows)]}
    )


def _index(channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(host="test-index-abc123.svc.pinecone.io", api_key="test-api-key")


def _failing_channel(fail_batch: int = 1) -> MagicMock:
    channel = MagicMock()

    def _upsert(vectors: list[dict[str, Any]], namespace: str | None, **_: Any) -> dict[str, Any]:
        if int(vectors[0]["id"][1:]) // 2 == fail_batch:
            raise RuntimeError("boom")
        return {"upserted_count": len(vectors)}

    channel.upsert.side_effect = _upsert
    return channel


class TestDefault:
    def test_partial_failure_is_aggregated_not_raised(self) -> None:
        index = _index(_failing_channel())

        with pytest.warns(UserWarning, match="aggregates partial failures"):
            response = index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)

        assert response.upserted_count == 4
        assert response.failed_item_count == 2
        assert {item["id"] for item in response.failed_items} == {"v2", "v3"}

    def test_signature_default_is_none(self) -> None:
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)
        param = sig.parameters["on_error"]

        assert param.default is None
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestOnErrorRaise:
    def test_raises_the_lowest_indexed_failure(self) -> None:
        channel = MagicMock()

        def _upsert(vectors: list[dict[str, Any]], namespace: str | None, **_: Any) -> Any:
            batch = int(vectors[0]["id"][1:]) // 2
            if batch in (1, 2):
                raise RuntimeError(f"batch {batch} boom")
            return {"upserted_count": len(vectors)}

        channel.upsert.side_effect = _upsert
        index = _index(channel)

        with pytest.raises(RuntimeError, match="batch 1 boom"):
            index.upsert_from_dataframe(
                _frame(6), batch_size=2, show_progress=False, on_error="raise"
            )

    def test_raise_does_not_warn(self) -> None:
        index = _index(_failing_channel())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeError):
                index.upsert_from_dataframe(
                    _frame(6), batch_size=2, show_progress=False, on_error="raise"
                )


class TestOnErrorValidation:
    @pytest.mark.parametrize("bad", ["collect_all", "RAISE", "", "true"])
    def test_unknown_value_raises(self, bad: str) -> None:
        index = _index(MagicMock())

        with pytest.raises(PineconeValueError, match="on_error"):
            index.upsert_from_dataframe(_frame(2), show_progress=False, on_error=bad)  # type: ignore[arg-type]


class TestTheWarning:
    def test_fires_once_per_process(self) -> None:
        index = _index(_failing_channel())

        with pytest.warns(UserWarning, match="aggregates partial failures") as first:
            index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)

        assert len(first) == 1
        assert second == []

    def test_names_the_escape_hatch_and_the_inspection_point(self) -> None:
        index = _index(_failing_channel())

        with pytest.warns(UserWarning, match="aggregates partial failures") as record:
            index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)

        message = str(record[0].message)
        assert 'on_error="raise"' in message
        assert "response.errors" in message
        assert "10.0.0" in message

    def test_does_not_fire_when_nothing_failed(self) -> None:
        channel = MagicMock()
        channel.upsert.return_value = {"upserted_count": 2}
        index = _index(channel)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)

        assert record == []

    def test_does_not_fire_when_on_error_was_passed_explicitly(self) -> None:
        index = _index(_failing_channel())

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            index.upsert_from_dataframe(
                _frame(6), batch_size=2, show_progress=False, on_error="collect"
            )

        assert record == []


class TestRestDoesNotWarn:
    @staticmethod
    def _rest_index_with_partial_failure() -> Any:
        from pinecone import Index
        from pinecone.models.batch import BatchError
        from pinecone.models.vectors.responses import UpsertResponse

        index = Index(host="test-index-abc1234.svc.us-east1-gcp.pinecone.io", api_key="test-key")
        index.upsert = MagicMock(  # type: ignore[method-assign]
            return_value=UpsertResponse(
                upserted_count=4,
                total_item_count=6,
                failed_item_count=2,
                errors=[
                    BatchError(
                        batch_index=1,
                        items=[{"id": "v2"}, {"id": "v3"}],
                        error=RuntimeError("boom"),
                        error_message="boom",
                    )
                ],
            )
        )
        return index

    def test_rest_partial_failure_is_silent(self) -> None:
        """REST has aggregated since v9.0.0; warning there is noise."""
        index = self._rest_index_with_partial_failure()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            response = index.upsert_from_dataframe(_frame(6), batch_size=2, show_progress=False)

        assert record == []
        assert response.failed_item_count == 2

    def test_rest_on_error_raise_still_raises(self) -> None:
        index = self._rest_index_with_partial_failure()

        with pytest.raises(RuntimeError, match="boom") as excinfo:
            index.upsert_from_dataframe(
                _frame(6), batch_size=2, show_progress=False, on_error="raise"
            )

        assert excinfo.value.response.upserted_count == 4


class TestSignatureParity:
    def test_all_three_transports_expose_on_error(self) -> None:
        from pinecone import Index
        from pinecone.async_client.async_index import AsyncIndex

        for method in (
            Index.upsert_from_dataframe,
            AsyncIndex.upsert_from_dataframe,
            GrpcIndex.upsert_from_dataframe,
        ):
            assert "on_error" in inspect.signature(method).parameters
