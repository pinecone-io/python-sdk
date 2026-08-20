"""2026-07 conformance for the asyncio transport of the two assistant_data
operations endpoints.

The sync variants live in ``test_assistant_data_operations_2026_07.py``; both
may claim the same operation (see README, "Additional rules"), and these add no
operation ids to the coverage numerator. What they add is the guarantee that
``AsyncAssistants`` puts the same method, the same ``/assistant``-prefixed path
(#173), the same sealed filter spellings and the same
``X-Pinecone-Api-Version`` on the wire, decodes ``OperationList`` and
``OperationModel`` through the shared adapter, and — the part only an async test
can prove — that ``AsyncPaginator`` walks the cursor the server hands back
without replaying a spent token.

Every payload and expected string is imported from the sync module rather than
restated, so the two transports cannot drift apart in the fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.errors.exceptions import NotFoundError, PineconeValueError
from pinecone.models.assistant.list import ListOperationsResponse
from pinecone.models.assistant.operation import OperationModel
from tests.unit.conformance import api_op
from tests.unit.conformance.test_assistant_data_operations_2026_07 import (
    ASSISTANT,
    ASSISTANT_NAME,
    BASE_URL,
    COMPLETED,
    CONTROL_URL,
    DATA_URL,
    FAILED,
    FILE_ID,
    NEXT_TOKEN,
    OPERATION_ID,
    OPERATION_LIST,
    PROCESSING,
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


@api_op("assistant_data:list_operations")
async def test_async_list_operations(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    page = await async_assistants.list_operations_page(
        assistant_name=ASSISTANT_NAME,
        operation_type="upload_file",
        status="Processing",
        page_size=100,
        pagination_token="eyJza2lwX3Bhc3QiOiI5OTNlNzRhIn0=",
    )
    assert isinstance(page, ListOperationsResponse)
    assert [op.operation_id for op in page.operations] == [
        OPERATION_ID,
        "op-8765-dcba-4321",
        "op-5555-eeee-9999",
    ]
    assert page.next == NEXT_TOKEN
    assert page.next_token == NEXT_TOKEN

    request = route.calls.last.request
    assert request.url.params["operation_type"] == "upload_file"
    assert request.url.params["status"] == "Processing"
    assert request.url.params["limit"] == "100"
    assert request.url.params["pagination_token"] == "eyJza2lwX3Bhc3QiOiI5OTNlNzRhIn0="

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ListOperationsResponse, OPERATION_LIST, optional_absent=["pagination"])


@api_op("assistant_data:describe_operation")
async def test_async_describe_operation(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=PROCESSING)
    )

    result = await async_assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID
    )
    assert isinstance(result, OperationModel)
    assert result.operation_id == OPERATION_ID
    assert result.operation_type == "upload_file"
    assert result.file_id == FILE_ID
    assert result.status == "Processing"
    assert result.percent_complete == 42
    assert result.created_at == "2025-10-01T12:30:00Z"
    assert result.completed_on is None
    assert result.ingestion_units is None
    assert result.error is None

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        PROCESSING,
        optional_absent=["file_id", "percent_complete", "completed_on"],
    )


async def test_async_a_completed_operation_reports_what_the_upload_cost(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """``ingestion_units`` and ``completed_on`` only appear once it finished."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{COMPLETED['id']}").mock(
        return_value=httpx.Response(200, json=COMPLETED)
    )

    result = await async_assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=str(COMPLETED["id"])
    )
    assert result.status == "Completed"
    assert result.percent_complete == 100
    assert result.completed_on == "2025-10-01T12:35:00Z"
    assert result.ingestion_units == 50.0
    assert result.error is None


async def test_async_a_failed_operation_carries_the_reason(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The failure text is the whole point of describing a failed operation."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{FAILED['id']}").mock(
        return_value=httpx.Response(200, json=FAILED)
    )

    result = await async_assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=str(FAILED["id"])
    )
    assert result.status == "Failed"
    assert result.error == "File processing failed: unsupported file format."
    assert result.ingestion_units is None


async def test_async_a_2026_04_minimal_operation_body_still_decodes(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """The enriched fields are all optional, so the older body is not a break."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200, json={"id": OPERATION_ID, "status": "Succeeded", "created_on": None}
        )
    )

    result = await async_assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID
    )
    assert result.status == "Succeeded"
    assert result.operation_type is None
    assert result.file_id is None
    assert result.percent_complete is None


async def test_async_describe_operation_404_reaches_the_caller(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """Operations are retained 30 days; after that the spec's 404 is what you get."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            404,
            json={
                "status": 404,
                "error": {"code": "NOT_FOUND", "message": f"Operation {OPERATION_ID} not found."},
            },
        )
    )

    with pytest.raises(NotFoundError) as excinfo:
        await async_assistants.describe_operation(
            assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID
        )

    assert OPERATION_ID in str(excinfo.value)


async def test_async_list_operations_404_when_the_assistant_is_gone(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            404,
            json={
                "status": 404,
                "error": {
                    "code": "NOT_FOUND",
                    "message": f'Assistant "{ASSISTANT_NAME}" not found.',
                },
            },
        )
    )

    with pytest.raises(NotFoundError):
        await async_assistants.list_operations_page(assistant_name=ASSISTANT_NAME)


async def test_async_no_filters_means_no_query_parameters(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """An unfiltered listing must not send empty filters the backend would parse."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={"operations": []})
    )

    page = await async_assistants.list_operations_page(assistant_name=ASSISTANT_NAME)
    assert page.operations == []
    assert page.next is None

    assert str(route.calls.last.request.url) == f"{DATA_URL}/operations/{ASSISTANT_NAME}"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"operation_type": "uploadfile"},
            "operation_type must be one of ('upload_file', 'upsert_file', "
            "'update_file_metadata', 'delete_file'), got 'uploadfile'",
        ),
        (
            {"operation_type": "UPLOAD_FILE"},
            "operation_type must be one of ('upload_file', 'upsert_file', "
            "'update_file_metadata', 'delete_file'), got 'UPLOAD_FILE'",
        ),
        (
            {"status": "processing"},
            "status must be one of ('Processing', 'Completed', 'Failed'), got 'processing'",
        ),
        (
            {"status": "Succeeded"},
            "status must be one of ('Processing', 'Completed', 'Failed'), got 'Succeeded'",
        ),
    ],
)
async def test_async_an_invalid_filter_never_reaches_the_wire(
    async_assistants: AsyncAssistants,
    respx_mock: respx.MockRouter,
    kwargs: dict[str, str],
    expected: str,
) -> None:
    """The enum is sealed backend-side; failing locally names the alternatives."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    with pytest.raises(PineconeValueError) as excinfo:
        await async_assistants.list_operations_page(assistant_name=ASSISTANT_NAME, **kwargs)

    assert str(excinfo.value) == expected
    assert route.call_count == 0


async def test_async_an_invalid_filter_on_the_paginator_raises_on_first_fetch(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """``list_operations`` is lazy, so the check lands when a page is fetched."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    paginator = async_assistants.list_operations(assistant_name=ASSISTANT_NAME, status="done")
    with pytest.raises(PineconeValueError, match=r"status must be one of"):
        await paginator.to_list()

    assert route.call_count == 0


async def test_async_the_paginator_walks_the_cursor_the_response_hands_back(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """Three pages, two tokens: every operation arrives once, in order."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        side_effect=[
            httpx.Response(
                200, json={"operations": [PROCESSING], "pagination": {"next": "token-1"}}
            ),
            httpx.Response(
                200, json={"operations": [COMPLETED], "pagination": {"next": "token-2"}}
            ),
            httpx.Response(200, json={"operations": [FAILED]}),
        ]
    )

    paginator = async_assistants.list_operations(assistant_name=ASSISTANT_NAME)
    assert [op.operation_id async for op in paginator] == [
        OPERATION_ID,
        "op-8765-dcba-4321",
        "op-5555-eeee-9999",
    ]
    assert paginator.pagination_token is None


async def test_async_the_paginator_forwards_the_filters_to_every_page(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """A filtered listing stays filtered after the first page, or it lies."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        side_effect=[
            httpx.Response(200, json={"operations": [FAILED], "pagination": {"next": "token-1"}}),
            httpx.Response(200, json={"operations": [FAILED]}),
        ]
    )

    await async_assistants.list_operations(
        assistant_name=ASSISTANT_NAME, operation_type="delete_file", status="Failed"
    ).to_list()

    assert route.call_count == 2
    for call in route.calls:
        assert call.request.url.params["operation_type"] == "delete_file"
        assert call.request.url.params["status"] == "Failed"
    assert "pagination_token" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["pagination_token"] == "token-1"


async def test_async_limit_bounds_the_items_not_the_pages(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    """``limit`` is the caller's item budget, matching ``list_files``."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            200,
            json={
                "operations": [PROCESSING, COMPLETED, FAILED],
                "pagination": {"next": "token-1"},
            },
        )
    )

    result = await async_assistants.list_operations(
        assistant_name=ASSISTANT_NAME, limit=2
    ).to_list()

    assert [op.operation_id for op in result] == [OPERATION_ID, "op-8765-dcba-4321"]
    assert route.call_count == 1


async def test_async_pagination_token_resumes_where_a_previous_call_stopped(
    async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={"operations": [FAILED]})
    )

    await async_assistants.list_operations(
        assistant_name=ASSISTANT_NAME, pagination_token=NEXT_TOKEN
    ).to_list()

    assert route.calls.last.request.url.params["pagination_token"] == NEXT_TOKEN


async def test_async_describe_operation_and_the_polling_loop_share_one_client(
    respx_mock: respx.MockRouter,
) -> None:
    """The public method is the polling loop's code path, not a parallel one."""
    describe_assistant = respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(
            202, json=operation(operation_type="delete_file", percent_complete=0)
        )
    )
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200,
            json=operation(
                operation_type="delete_file",
                status="Completed",
                percent_complete=100,
                completed_on="2025-10-01T12:35:00Z",
            ),
        )
    )

    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    try:
        await client.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
        result = await client.describe_operation(
            assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID
        )
    finally:
        await client.close()

    assert result.status == "Completed"
    assert describe_assistant.call_count == 1
    assert operations.call_count == 2
    for call in operations.calls:
        assert call.request.headers["x-pinecone-api-version"] == "2026-07"
        assert str(call.request.url) == f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}"
