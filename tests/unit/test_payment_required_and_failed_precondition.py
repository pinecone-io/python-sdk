"""Tests for the 402 PaymentRequiredError / 412 FailedPreconditionError taxonomy.

Both statuses used to fall through to a bare ``ApiError``. These tests pin the
typed mapping, the verbatim server message, and identical behavior on the sync
and async lanes.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient, _raise_for_status
from pinecone.errors.exceptions import (
    ApiError,
    FailedPreconditionError,
    PaymentRequiredError,
    PineconeError,
)

BASE_URL = "https://api.pinecone.io"

PAYMENT_MESSAGE = (
    "Organization org-abc123 has no active payment method. Add one before creating a project."
)
PRECONDITION_MESSAGE = (
    "Unable to delete project. It still has 2 indexes, 1 assistant, and 1 backup."
)


def _make_sync_client() -> HTTPClient:
    return HTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), api_version="2025-10")


def _make_async_client() -> AsyncHTTPClient:
    return AsyncHTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), api_version="2025-10")


def _error_body(code: str, message: str, status: int) -> dict[str, object]:
    return {"error": {"code": code, "message": message}, "status": status}


class TestClassShape:
    def test_payment_required_defaults(self) -> None:
        err = PaymentRequiredError()
        assert err.status_code == 402
        assert err.message == "Payment required"
        assert str(err) == "[402] Payment required"

    def test_failed_precondition_defaults(self) -> None:
        err = FailedPreconditionError()
        assert err.status_code == 412
        assert err.message == "Precondition failed"
        assert str(err) == "[412] Precondition failed"

    @pytest.mark.parametrize(
        ("cls", "status"),
        [(PaymentRequiredError, 402), (FailedPreconditionError, 412)],
    )
    def test_inherits_api_error_and_pinecone_error(self, cls: type[ApiError], status: int) -> None:
        err = cls()
        assert isinstance(err, ApiError)
        assert isinstance(err, PineconeError)
        assert err.status_code == status

    @pytest.mark.parametrize("cls", [PaymentRequiredError, FailedPreconditionError])
    def test_repr_shows_own_class_name(self, cls: type[ApiError]) -> None:
        assert repr(cls()).startswith(f"{cls.__name__}(")

    @pytest.mark.parametrize(
        ("cls", "status"),
        [(PaymentRequiredError, 402), (FailedPreconditionError, 412)],
    )
    def test_propagates_structured_context(self, cls: type[ApiError], status: int) -> None:
        headers = {"x-request-id": "req-1"}
        err = cls(
            message="server said so",
            body={"error": {"message": "server said so"}},
            reason="Test Reason",
            headers=headers,
            error_code="SOME_CODE",
            request_id="req-1",
        )
        assert err.status_code == status
        assert err.reason == "Test Reason"
        assert err.headers == headers
        assert err.error_code == "SOME_CODE"
        assert err.request_id == "req-1"
        assert err.body == {"error": {"message": "server said so"}}


class TestRaiseForStatusMapping:
    def test_402_maps_to_payment_required(self) -> None:
        response = httpx.Response(
            402,
            json=_error_body("PAYMENT_REQUIRED", PAYMENT_MESSAGE, 402),
            headers={"x-pinecone-request-id": "req-402"},
        )
        with pytest.raises(PaymentRequiredError) as exc:
            _raise_for_status(response)
        err = exc.value
        assert err.status_code == 402
        assert err.message == PAYMENT_MESSAGE
        assert err.error_code == "PAYMENT_REQUIRED"
        assert err.request_id == "req-402"
        assert err.reason == "Payment Required"

    def test_412_maps_to_failed_precondition(self) -> None:
        response = httpx.Response(
            412,
            json=_error_body("FAILED_PRECONDITION", PRECONDITION_MESSAGE, 412),
            headers={"x-pinecone-request-id": "req-412"},
        )
        with pytest.raises(FailedPreconditionError) as exc:
            _raise_for_status(response)
        err = exc.value
        assert err.status_code == 412
        assert err.message == PRECONDITION_MESSAGE
        assert err.error_code == "FAILED_PRECONDITION"
        assert err.request_id == "req-412"
        assert err.reason == "Precondition Failed"

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(402, PaymentRequiredError), (412, FailedPreconditionError)],
    )
    def test_still_catchable_as_api_error(self, status: int, expected: type[ApiError]) -> None:
        response = httpx.Response(status, json={"message": "boom"})
        with pytest.raises(ApiError) as exc:
            _raise_for_status(response)
        assert type(exc.value) is expected

    @pytest.mark.parametrize("status", [402, 412])
    def test_plain_text_body_message_preserved_verbatim(self, status: int) -> None:
        response = httpx.Response(status, content=b"upstream billing service says no")
        with pytest.raises(ApiError) as exc:
            _raise_for_status(response)
        assert exc.value.message == "upstream billing service says no"

    @pytest.mark.parametrize(
        ("status", "expected_message"),
        [(402, "Payment Required"), (412, "Precondition Failed")],
    )
    def test_empty_body_falls_back_to_reason_phrase(
        self, status: int, expected_message: str
    ) -> None:
        response = httpx.Response(status, content=b"")
        with pytest.raises(ApiError) as exc:
            _raise_for_status(response)
        assert exc.value.message == expected_message

    def test_neighbouring_statuses_unchanged(self) -> None:
        for status in (400, 405, 411, 413):
            response = httpx.Response(status, json={"message": "generic"})
            with pytest.raises(ApiError) as exc:
                _raise_for_status(response)
            assert type(exc.value) is ApiError
            assert exc.value.status_code == status


class TestSyncLane:
    @respx.mock
    def test_project_create_402(self) -> None:
        route = respx.post(f"{BASE_URL}/admin/projects").mock(
            return_value=httpx.Response(
                402, json=_error_body("PAYMENT_REQUIRED", PAYMENT_MESSAGE, 402)
            )
        )
        client = _make_sync_client()
        try:
            with pytest.raises(PaymentRequiredError) as exc:
                client.post("/admin/projects", json={"name": "p"})
        finally:
            client.close()
        assert exc.value.message == PAYMENT_MESSAGE
        assert route.call_count == 1

    @respx.mock
    def test_project_delete_412(self) -> None:
        route = respx.delete(f"{BASE_URL}/admin/projects/proj-abc123").mock(
            return_value=httpx.Response(
                412, json=_error_body("FAILED_PRECONDITION", PRECONDITION_MESSAGE, 412)
            )
        )
        client = _make_sync_client()
        try:
            with pytest.raises(FailedPreconditionError) as exc:
                client.delete("/admin/projects/proj-abc123")
        finally:
            client.close()
        assert exc.value.message == PRECONDITION_MESSAGE
        assert route.call_count == 1


class TestAsyncLane:
    @respx.mock
    @pytest.mark.asyncio
    async def test_project_create_402(self) -> None:
        route = respx.post(f"{BASE_URL}/admin/projects").mock(
            return_value=httpx.Response(
                402, json=_error_body("PAYMENT_REQUIRED", PAYMENT_MESSAGE, 402)
            )
        )
        client = _make_async_client()
        try:
            with pytest.raises(PaymentRequiredError) as exc:
                await client.post("/admin/projects", json={"name": "p"})
        finally:
            await client.close()
        assert exc.value.message == PAYMENT_MESSAGE
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_project_delete_412(self) -> None:
        route = respx.delete(f"{BASE_URL}/admin/projects/proj-abc123").mock(
            return_value=httpx.Response(
                412, json=_error_body("FAILED_PRECONDITION", PRECONDITION_MESSAGE, 412)
            )
        )
        client = _make_async_client()
        try:
            with pytest.raises(FailedPreconditionError) as exc:
                await client.delete("/admin/projects/proj-abc123")
        finally:
            await client.close()
        assert exc.value.message == PRECONDITION_MESSAGE
        assert route.call_count == 1


class TestExports:
    def test_importable_from_top_level(self) -> None:
        import pinecone

        assert pinecone.PaymentRequiredError is PaymentRequiredError
        assert pinecone.FailedPreconditionError is FailedPreconditionError
        assert "PaymentRequiredError" in pinecone.__all__
        assert "FailedPreconditionError" in pinecone.__all__

    def test_importable_from_errors_package(self) -> None:
        import pinecone.errors as errors

        assert errors.PaymentRequiredError is PaymentRequiredError
        assert errors.FailedPreconditionError is FailedPreconditionError
        assert "PaymentRequiredError" in errors.__all__
        assert "FailedPreconditionError" in errors.__all__
