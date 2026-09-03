"""2026-07 conformance for the asyncio transport of the five assistant_data
file-lifecycle operations.

The sync variants live in ``test_assistant_data_files_2026_07.py``; both may
claim the same operation (see README, "Additional rules"), and these add no
operation ids to the coverage numerator. What they add is the guarantee that
``AsyncAssistants`` puts the same method, the same ``/assistant``-prefixed path
(#173), the same sealed multipart field set and the same
``X-Pinecone-Api-Version`` on the wire; that it reads the ``202`` envelope and
the ``204`` immediate delete the same way; and — the part only an async test can
prove — that the graduated upload handshake survives ``await``, polling the
operation record rather than the file.

Every payload and expected string is imported from the sync module rather than
restated, so the two transports cannot drift apart in the fixtures.
"""

from __future__ import annotations

import io
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.errors.exceptions import ApiError, PineconeError
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListFilesResponse
from pinecone.models.assistant.operation import OperationModel
from tests.unit.conformance import api_op
from tests.unit.conformance.test_assistant_data_files_2026_07 import (
    ACCEPTED_DELETE,
    ACCEPTED_UPLOAD,
    ACCEPTED_UPSERT,
    ASSISTANT,
    ASSISTANT_NAME,
    BASE_URL,
    COMPLETED_UPLOAD,
    CONTROL_URL,
    DATA_URL,
    FILE,
    FILE_ID,
    FILE_LIST,
    METADATA_QUERY_PARAM_ERROR,
    OPERATION_ID,
    UPSERT_FILE_ID,
    multipart_fields,
    operation,
)


@pytest.fixture
async def async_assistants(respx_mock: respx.MockRouter) -> AsyncGenerator[AsyncAssistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    await client.close()


def mock_upload_completion(respx_mock: respx.MockRouter, file_id: str = FILE_ID) -> None:
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=COMPLETED_UPLOAD)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{file_id}").mock(
        return_value=httpx.Response(200, json={**FILE, "id": file_id})
    )


@api_op("assistant_data:upload_file")
async def test_async_upload_file(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_UPLOAD,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    mock_upload_completion(respx_mock)

    result = await async_assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"%PDF-1.7 pinecone"),
        file_name="pinecone-guide.pdf",
        metadata={"tags": ["report", "Q4"]},
        multimodal=True,
        timeout=-1,
    )
    assert isinstance(result, AssistantFileModel)
    assert result.id == FILE_ID
    assert result.size == 25000

    request = route.calls.last.request
    fields = multipart_fields(request)
    assert set(fields) == {"file", "metadata"}
    assert fields["file"] == b"%PDF-1.7 pinecone"
    assert orjson.loads(fields["metadata"]) == {"tags": ["report", "Q4"]}
    assert "metadata" not in request.url.params
    assert request.url.params["multimodal"] == "true"

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_UPLOAD,
        optional_absent=["file_id", "percent_complete", "operation_type"],
    )


@api_op("assistant_data:upsert_file")
async def test_async_upsert_file(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.put(f"{DATA_URL}/files/{ASSISTANT_NAME}/{UPSERT_FILE_ID}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_UPSERT,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    mock_upload_completion(respx_mock, file_id=UPSERT_FILE_ID)

    with patch("pinecone.async_client.assistants.asyncio.sleep"):
        result = await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"plain text"),
            file_name="notes.txt",
            file_id=UPSERT_FILE_ID,
            metadata={"published": "2025-10-01"},
        )
    assert isinstance(result, AssistantFileModel)
    assert result.id == UPSERT_FILE_ID

    request = route.calls.last.request
    fields = multipart_fields(request)
    assert set(fields) == {"file", "metadata"}
    assert orjson.loads(fields["metadata"]) == {"published": "2025-10-01"}
    assert "metadata" not in request.url.params

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_UPSERT,
        optional_absent=["file_id", "created_on", "percent_complete"],
    )


@api_op("assistant_data:delete_file")
async def test_async_delete_file(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_DELETE,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200, json=operation(operation_type="delete_file", status="Completed")
        )
    )

    assert (
        await async_assistants.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID) is None
    )

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_DELETE,
        optional_absent=["file_id", "operation_type", "percent_complete"],
    )


@api_op("assistant_data:describe_file")
async def test_async_describe_file(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )

    result = await async_assistants.describe_file(
        assistant_name=ASSISTANT_NAME, file_id=FILE_ID, include_url=True
    )
    assert isinstance(result, AssistantFileModel)
    assert result.size == 25000
    assert result.metadata == {"tags": ["report", "Q4"]}

    request = route.calls.last.request
    assert request.url.params["include_url"] == "true"
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantFileModel, FILE, optional_absent=["size", "status", "metadata"])


@api_op("assistant_data:list_files")
async def test_async_list_files(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=FILE_LIST)
    )

    page = await async_assistants.list_files_page(
        assistant_name=ASSISTANT_NAME,
        page_size=10,
        filter={"tags": {"$in": ["report"]}},
    )
    assert isinstance(page, ListFilesResponse)
    assert [f.id for f in page.files] == [FILE_ID]
    assert page.next == FILE_LIST["pagination"]["next"]

    request = route.calls.last.request
    assert request.url.params["limit"] == "10"
    assert orjson.loads(request.url.params["filter"]) == {"tags": {"$in": ["report"]}}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ListFilesResponse, FILE_LIST, optional_absent=["pagination"])


async def test_async_delete_file_204_is_success_with_no_body(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """A file the backend could remove at once answers 204 and never polls."""
    route = respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(204)
    )
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=COMPLETED_UPLOAD)
    )

    assert (
        await async_assistants.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID) is None
    )

    assert route.call_count == 1
    assert operations.call_count == 0
    assert route.calls.last.request.headers["x-pinecone-api-version"] == "2026-07"


async def test_async_metadata_is_never_sent_as_a_query_parameter(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The 2025-10 call shape is a 400 on 2026-07, so the async path must not produce it."""
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    mock_upload_completion(respx_mock)

    await async_assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"data"),
        file_name="report.pdf",
        metadata={"created_by": "Jane Doe"},
        timeout=-1,
    )

    assert "metadata=" not in str(route.calls.last.request.url)


async def test_async_metadata_query_param_rejection_reaches_the_caller_verbatim(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """A caller pinned to the old shape gets the backend's remediation text."""
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            400,
            json={
                "status": 400,
                "error": {"code": "INVALID_ARGUMENT", "message": METADATA_QUERY_PARAM_ERROR},
            },
        )
    )

    with pytest.raises(ApiError) as excinfo:
        await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"data"),
            file_name="report.pdf",
            timeout=-1,
        )

    assert excinfo.value.message == METADATA_QUERY_PARAM_ERROR
    assert "multipart form field" in str(excinfo.value)


async def test_async_a_2025_10_file_body_on_upload_fails_pointing_at_the_operations_api(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The old 200 ``AssistantFileModel`` upload body no longer yields the file."""
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={**FILE, "status": "Processing"})
    )

    with pytest.raises(PineconeError) as excinfo:
        await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"data"),
            file_name="pinecone-guide.pdf",
        )

    message = str(excinfo.value)
    assert "did not name the file it created" in message
    assert "describe_operation()" in message
    assert FILE["id"] in message


async def test_async_a_failed_upload_operation_quotes_the_server_message(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The 400-example text from the spec must survive to the raised error."""
    server_message = "Uploaded file can only currently be either a pdf or txt file"
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200,
            json=operation(
                status="Failed",
                completed_on="2026-07-01T00:00:05Z",
                error_message=server_message,
            ),
        )
    )

    with pytest.raises(PineconeError) as excinfo:
        await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"gif89a"), file_name="a.gif"
        )

    message = str(excinfo.value)
    assert server_message in message
    assert OPERATION_ID in message
    assert FILE_ID in message


async def test_async_the_failure_path_does_not_read_the_removed_file_fields(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """``AssistantFileModel`` no longer carries the failure detail at all."""
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json={**FILE, "status": "ProcessingFailed"})
    )

    file = await async_assistants.describe_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
    for removed in ("error_message", "percent_done"):
        with pytest.raises(AttributeError, match="describe_operation"):
            getattr(file, removed)


async def test_async_no_metadata_means_no_metadata_part(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """Omitting metadata must not send an empty part — the backend seals the set."""
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    mock_upload_completion(respx_mock)

    await async_assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"data"),
        file_name="report.pdf",
        timeout=-1,
    )

    request = route.calls.last.request
    assert set(multipart_fields(request)) == {"file"}
    assert "multimodal" not in request.url.params


async def test_async_upload_polls_the_operation_not_the_file(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """Progress comes from the operation record, which reports percent_complete."""
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        side_effect=[
            httpx.Response(200, json=operation(percent_complete=25)),
            httpx.Response(200, json=operation(percent_complete=80)),
            httpx.Response(200, json=COMPLETED_UPLOAD),
        ]
    )
    describe = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )

    with patch("pinecone.async_client.assistants.asyncio.sleep"):
        await async_assistants.upload_file(
            assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data"), file_name="report.pdf"
        )

    assert operations.call_count == 3
    assert describe.call_count == 1
    for call in operations.calls:
        assert call.request.headers["x-pinecone-api-version"] == "2026-07"


async def test_async_one_cached_data_plane_client_serves_every_file_operation(
    respx_mock: respx.MockRouter,
) -> None:
    """List, describe and delete share the cached client — no per-call rebuild."""
    describe_assistant = respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=FILE_LIST)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )
    respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(204)
    )

    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    try:
        await client.list_files_page(assistant_name=ASSISTANT_NAME)
        await client.list_files_page(assistant_name=ASSISTANT_NAME)
        await client.describe_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
        await client.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
    finally:
        await client.close()

    assert describe_assistant.call_count == 1
