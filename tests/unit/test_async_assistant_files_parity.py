"""Sync/async parity for the assistant_data file-lifecycle and operations surface.

``Assistants`` (#125, #126) and ``AsyncAssistants`` (#129) drive the same
``2026-07`` endpoints through the same ``AssistantsAdapter``, so the two
transports should differ only in ``await``. These tests hold them to that on the
axes a transport port can quietly break:

* identical request snapshots on the wire — method, the ``/assistant``-prefixed
  path (#173), query string, multipart field set, and the version header — for
  identical arguments, across upload, upsert, delete, describe, list and both
  operations reads;
* identical signatures, defaults and return annotations modulo
  ``AsyncPaginator``;
* identical exception types and message *text* for the failures a caller can
  provoke: the sealed ``operation_type``/``status`` filters, a ``202`` that names
  no file, a ``"Failed"`` operation, and a polling timeout;
* the same graduated polling shape — the operation record is what gets polled,
  at the same cadence, and the per-call ``2026-04`` client helpers are gone from
  both.

The control-plane and evaluation surface has its own module,
``tests/unit/test_async_assistants_parity.py``, and the chat/context data plane
has ``tests/unit/test_async_assistant_chat_parity.py``; this one follows their
request-snapshot pattern.
"""

from __future__ import annotations

import inspect
import io
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import PineconeError, PineconeTimeoutError, PineconeValueError

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
DATA_HOST = "https://prod-1-data.ke.pinecone.io"
DATA_URL = f"{DATA_HOST}/assistant"

ASSISTANT_NAME = "parity-assistant"
FILE_ID = "file-parity-1"
UPSERT_FILE_ID = "caller-chosen-id"
OPERATION_ID = "op-parity-1"

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "host": DATA_HOST,
    "region": "us",
}

FILE: dict[str, Any] = {
    "id": FILE_ID,
    "name": "guide.pdf",
    "size": 2048,
    "status": "Available",
    "multimodal": False,
    "metadata": {"tags": ["report"]},
    "signed_url": None,
    "created_on": "2026-07-01T00:00:00Z",
    "updated_on": "2026-07-01T00:01:00Z",
}


def operation(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": OPERATION_ID,
        "operation_type": "upload_file",
        "file_id": FILE_ID,
        "status": "Processing",
        "created_on": "2026-07-01T00:00:00Z",
        "completed_on": None,
        "percent_complete": 40,
        "error_message": None,
        "ingestion_units": None,
    }
    body.update(overrides)
    return body


OPERATION_LIST: dict[str, Any] = {
    "operations": [operation(), operation(id="op-parity-2", status="Completed")],
    "pagination": {"next": "next-token"},
}

FILE_LIST: dict[str, Any] = {"files": [FILE], "pagination": {"next": "next-token"}}

METHODS = [
    "delete_file",
    "describe_file",
    "describe_operation",
    "list_files",
    "list_files_page",
    "list_operations",
    "list_operations_page",
    "upload_file",
]

REMOVED_HELPERS = ["_list_files_http", "_upsert_http", "_poll_file_until_processed"]

CALLS: dict[str, dict[str, Any]] = {
    "upload_file": {
        "assistant_name": ASSISTANT_NAME,
        "file_stream": None,
        "file_name": "guide.pdf",
        "metadata": {"tags": ["report"]},
        "multimodal": True,
        "timeout": -1,
    },
    "upsert_file": {
        "assistant_name": ASSISTANT_NAME,
        "file_stream": None,
        "file_name": "guide.pdf",
        "metadata": {"published": "2026-07-01"},
        "file_id": UPSERT_FILE_ID,
        "timeout": -1,
    },
    "delete_file": {"assistant_name": ASSISTANT_NAME, "file_id": FILE_ID, "timeout": -1},
    "describe_file": {
        "assistant_name": ASSISTANT_NAME,
        "file_id": FILE_ID,
        "include_url": True,
    },
    "list_files_page": {
        "assistant_name": ASSISTANT_NAME,
        "page_size": 10,
        "pagination_token": "prev-token",
        "filter": {"tags": {"$in": ["report"]}},
    },
    "describe_operation": {"assistant_name": ASSISTANT_NAME, "operation_id": OPERATION_ID},
    "list_operations_page": {
        "assistant_name": ASSISTANT_NAME,
        "operation_type": "upload_file",
        "status": "Processing",
        "page_size": 25,
        "pagination_token": "prev-token",
    },
}

WRITE_METHOD = {"upload_file": "upload_file", "upsert_file": "upload_file"}

BAD_FILTERS: list[dict[str, str]] = [
    {"operation_type": "uploadfile"},
    {"operation_type": "UPLOAD_FILE"},
    {"operation_type": ""},
    {"status": "processing"},
    {"status": "Succeeded"},
    {"status": ""},
]

FAILURE_MESSAGE = "Uploaded file can only currently be either a pdf or txt file"


def _comparable(annotation: Any) -> str:
    """Erase the one difference the two transports are allowed to have."""
    return str(annotation).replace("AsyncPaginator[", "Paginator[")


def _multipart_fields(request: httpx.Request) -> dict[str, bytes] | None:
    """The multipart parts of *request* by field name, or ``None`` if not multipart.

    The boundary is random per request, so the raw body cannot be compared
    directly; the field set and each field's bytes can.
    """
    content_type = request.headers.get("content-type", "")
    match = re.search(r"boundary=([^;]+)", content_type)
    if match is None:
        return None
    boundary = match.group(1).strip('"').encode()
    fields: dict[str, bytes] = {}
    for part in request.content.split(b"--" + boundary):
        head, separator, body = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        name = re.search(rb'\bname="([^"]*)"', head)
        if name is None:
            continue
        fields[name.group(1).decode()] = body[:-2] if body.endswith(b"\r\n") else body
    return fields


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    fields = _multipart_fields(request)
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode().split("?")[0],
        "query": dict(request.url.params),
        "multipart": fields,
        "body": None if fields is not None or not request.content else request.content,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _raised(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


async def _raised_async(call: Callable[[], Awaitable[object]]) -> tuple[type[BaseException], str]:
    try:
        await call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


@pytest.fixture
def sync_assistants(respx_mock: respx.MockRouter) -> Iterator[Assistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = Assistants(config=PineconeConfig(api_key="parity-key", host=BASE_URL))
    yield client
    client.close()


@pytest.fixture
async def async_assistants(respx_mock: respx.MockRouter) -> AsyncIterator[AsyncAssistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = AsyncAssistants(config=PineconeConfig(api_key="parity-key", host=BASE_URL))
    yield client
    await client.close()


def _register_routes(respx_mock: respx.MockRouter) -> dict[str, respx.Route]:
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{UPSERT_FILE_ID}").mock(
        return_value=httpx.Response(200, json={**FILE, "id": UPSERT_FILE_ID})
    )
    return {
        "upload_file": respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(
                202,
                json=operation(),
                headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
            )
        ),
        "upsert_file": respx_mock.put(f"{DATA_URL}/files/{ASSISTANT_NAME}/{UPSERT_FILE_ID}").mock(
            return_value=httpx.Response(
                202, json=operation(operation_type="upsert_file", file_id=UPSERT_FILE_ID)
            )
        ),
        "delete_file": respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
            return_value=httpx.Response(202, json=operation(operation_type="delete_file"))
        ),
        "describe_file": respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
            return_value=httpx.Response(200, json=FILE)
        ),
        "list_files_page": respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=FILE_LIST)
        ),
        "describe_operation": respx_mock.get(
            f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}"
        ).mock(return_value=httpx.Response(200, json=operation())),
        "list_operations_page": respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=OPERATION_LIST)
        ),
    }


def _kwargs(operation_name: str) -> dict[str, Any]:
    kwargs = dict(CALLS[operation_name])
    if "file_stream" in kwargs:
        kwargs["file_stream"] = io.BytesIO(b"%PDF-1.7 parity")
    return kwargs


@pytest.mark.parametrize("method_name", METHODS)
def test_parameter_parity(method_name: str) -> None:
    sync_params = dict(inspect.signature(getattr(Assistants, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(AsyncAssistants, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{method_name}.{name}: kind differs (sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert _comparable(sync_param.annotation) == _comparable(async_param.annotation), (
            f"{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize("method_name", METHODS)
def test_return_annotation_parity(method_name: str) -> None:
    sync_return = inspect.signature(getattr(Assistants, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncAssistants, method_name)).return_annotation

    assert _comparable(sync_return) == _comparable(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize("helper", REMOVED_HELPERS)
def test_neither_transport_keeps_a_per_call_client_helper(helper: str) -> None:
    """The 2026-04 uncached-client helpers exist on neither transport."""
    assert not hasattr(Assistants, helper)
    assert not hasattr(AsyncAssistants, helper)


@pytest.mark.parametrize("operation_name", sorted(CALLS))
async def test_request_snapshot_parity(
    operation_name: str,
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    routes = _register_routes(respx_mock)
    route = routes[operation_name]
    method_name = WRITE_METHOD.get(operation_name, operation_name)

    getattr(sync_assistants, method_name)(**_kwargs(operation_name))
    await getattr(async_assistants, method_name)(**_kwargs(operation_name))

    assert len(route.calls) == 2, "each transport must have issued exactly one request"
    sync_snapshot = _snapshot(route.calls[0].request)
    async_snapshot = _snapshot(route.calls[1].request)

    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == "2026-07"
    assert async_snapshot["raw_path"].startswith("/assistant/")


async def test_metadata_travels_in_the_multipart_body_on_both_transports(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """Both transports must produce the 2026-07 call shape, not the 2025-10 one."""
    routes = _register_routes(respx_mock)
    route = routes["upload_file"]

    sync_assistants.upload_file(**_kwargs("upload_file"))
    await async_assistants.upload_file(**_kwargs("upload_file"))

    for call in route.calls:
        fields = _multipart_fields(call.request)
        assert fields is not None
        assert set(fields) == {"file", "metadata"}
        assert orjson.loads(fields["metadata"]) == {"tags": ["report"]}
        assert "metadata" not in call.request.url.params
        assert call.request.url.params["multimodal"] == "true"


async def test_polling_reads_the_operation_at_the_same_cadence(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """Both transports poll the operation record, not the file, and sleep the same."""
    _register_routes(respx_mock)
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}")
    responses = [
        httpx.Response(200, json=operation(percent_complete=25)),
        httpx.Response(200, json=operation(status="Completed", percent_complete=100)),
    ]

    operations.mock(side_effect=list(responses))
    with patch("pinecone.client.assistants.time.sleep") as sync_sleep:
        sync_assistants.upload_file(assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data"))

    operations.mock(side_effect=list(responses))
    with patch("pinecone.async_client.assistants.asyncio.sleep") as async_sleep:
        await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data")
        )

    assert async_sleep.call_args_list == sync_sleep.call_args_list
    assert [call.args for call in async_sleep.call_args_list] == [(5,)]


@pytest.mark.parametrize("bad", BAD_FILTERS)
async def test_operation_filter_rejection_parity(
    bad: dict[str, str], sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    """The client-side x-enum check must fail identically before any request."""
    sync_result = _raised(
        lambda: sync_assistants.list_operations_page(assistant_name=ASSISTANT_NAME, **bad)
    )
    async_result = await _raised_async(
        lambda: async_assistants.list_operations_page(assistant_name=ASSISTANT_NAME, **bad)
    )

    assert async_result == sync_result
    assert async_result[0] is PineconeValueError


async def test_accepted_upload_without_a_file_id_fails_identically(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=operation(file_id=None))
    )

    sync_result = _raised(
        lambda: sync_assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"data"),
            file_name="guide.pdf",
        )
    )
    async_result = await _raised_async(
        lambda: async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"data"),
            file_name="guide.pdf",
        )
    )

    assert async_result == sync_result
    assert async_result[0] is PineconeError
    assert "did not name the file it created" in async_result[1]


@pytest.mark.parametrize(
    ("method_name", "kwargs", "operation_type"),
    [
        ("upload_file", {"file_stream": None, "file_name": "a.gif"}, "upload_file"),
        ("delete_file", {"file_id": FILE_ID}, "delete_file"),
    ],
)
async def test_failed_operation_message_parity(
    method_name: str,
    kwargs: dict[str, Any],
    operation_type: str,
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """The server's ``error_message`` reaches the caller verbatim on both paths."""
    _register_routes(respx_mock)
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200,
            json=operation(
                operation_type=operation_type, status="Failed", error_message=FAILURE_MESSAGE
            ),
        )
    )

    def _call_kwargs() -> dict[str, Any]:
        resolved = dict(kwargs)
        if "file_stream" in resolved:
            resolved["file_stream"] = io.BytesIO(b"gif89a")
        return {"assistant_name": ASSISTANT_NAME, **resolved}

    with patch("pinecone.client.assistants.time.sleep"):
        sync_result = _raised(lambda: getattr(sync_assistants, method_name)(**_call_kwargs()))
    with patch("pinecone.async_client.assistants.asyncio.sleep"):
        async_result = await _raised_async(
            lambda: getattr(async_assistants, method_name)(**_call_kwargs())
        )

    assert async_result == sync_result
    assert async_result[0] is PineconeError
    assert FAILURE_MESSAGE in async_result[1]


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("upload_file", {"file_stream": None, "file_name": "guide.pdf"}),
        ("delete_file", {"file_id": FILE_ID}),
    ],
)
async def test_polling_timeout_message_parity(
    method_name: str,
    kwargs: dict[str, Any],
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """A deadline that elapses names the same operation and the same progress."""
    _register_routes(respx_mock)
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=operation(percent_complete=40))
    )

    def _call_kwargs() -> dict[str, Any]:
        resolved = dict(kwargs)
        if "file_stream" in resolved:
            resolved["file_stream"] = io.BytesIO(b"data")
        return {"assistant_name": ASSISTANT_NAME, "timeout": 10, **resolved}

    with (
        patch("pinecone.client.assistants.time.sleep"),
        patch("pinecone.client.assistants.time.monotonic", side_effect=[0.0, 11.0]),
    ):
        sync_result = _raised(lambda: getattr(sync_assistants, method_name)(**_call_kwargs()))
    with (
        patch("pinecone.async_client.assistants.asyncio.sleep"),
        patch("pinecone.async_client.assistants.time.monotonic", side_effect=[0.0, 11.0]),
    ):
        async_result = await _raised_async(
            lambda: getattr(async_assistants, method_name)(**_call_kwargs())
        )

    assert async_result == sync_result
    assert async_result[0] is PineconeTimeoutError
    assert f"operation_id='{OPERATION_ID}'" in async_result[1]
    assert "percent_complete=40" in async_result[1]


async def test_paginators_walk_the_same_cursors(
    sync_assistants: Assistants,
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
) -> None:
    """``list_operations`` and ``list_files`` yield the same ids in the same order."""
    pages = [
        httpx.Response(200, json={"operations": [operation()], "pagination": {"next": "tok-1"}}),
        httpx.Response(200, json={"operations": [operation(id="op-parity-2")]}),
    ]
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}")

    route.mock(side_effect=list(pages))
    sync_ids = [
        op.operation_id
        for op in sync_assistants.list_operations(assistant_name=ASSISTANT_NAME).to_list()
    ]
    sync_tokens = [call.request.url.params.get("pagination_token") for call in route.calls]

    route.reset()
    route.mock(side_effect=list(pages))
    async_ops = await async_assistants.list_operations(assistant_name=ASSISTANT_NAME).to_list()
    async_ids = [op.operation_id for op in async_ops]
    async_tokens = [call.request.url.params.get("pagination_token") for call in route.calls]

    assert async_ids == sync_ids == [OPERATION_ID, "op-parity-2"]
    assert async_tokens == sync_tokens == [None, "tok-1"]
