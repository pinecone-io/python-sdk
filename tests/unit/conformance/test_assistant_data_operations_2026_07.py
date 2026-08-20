"""2026-07 conformance for the two assistant_data operations endpoints.

``assistant_data_2026-07.oas.yaml`` documents the pair that turns the file
write path's ``202`` envelopes into something a caller can follow:

* ``list_operations`` (line 549) — ``GET /operations/{assistant_name}`` with
  ``operation_type``, ``status``, ``limit`` and ``pagination_token``, answering
  ``OperationList`` (line 1779): ``operations`` plus a ``PaginationResponse``.
* ``describe_operation`` (line 697) — ``GET
  /operations/{assistant_name}/{operation_id}``, answering ``OperationModel``
  (line 1790).

Backend behavior is authoritative and matches. The operations router serves
these only on ``2026-04`` and later — every version up to ``2025-10`` falls to a
``not_found_router``
(``svc-knowledge-engine/src/ingest/service/routes/mod.rs:38-58``). The
``2026-04`` handler seals the filters to the four file operation types and the
three file statuses (``routes/v202604/mod.rs:508-552``), deserializing them from
the query string with exactly the spec's wire spellings, which is why the SDK
checks them client-side rather than spending a round trip on a typo. ``limit``
defaults to 50 and is rejected above 100
(``handler.rs:66-67``, ``:445-451``), and ``pagination`` is
``skip_serializing_if = Option::is_none`` (``routes/v202604/mod.rs:554-560``),
so an exhausted listing omits the key rather than sending a null. All at
pinecone-db@f6fd0a4019.

Paths carry the ``/assistant`` prefix the SDK really sends, which the spec's
``servers`` URL omits (#173, registered as a ``base_path_overrides`` entry).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import NotFoundError, PineconeValueError
from pinecone.models.assistant.list import ListOperationsResponse
from pinecone.models.assistant.operation import OperationModel
from tests.unit.conformance import api_op

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
DATA_HOST = "https://prod-1-data.ke.pinecone.io"
DATA_URL = f"{DATA_HOST}/assistant"

ASSISTANT_NAME = "operations-conformance"
FILE_ID = "my-report-2025"
OPERATION_ID = "op-1234-abcd-5678"
NEXT_TOKEN = "dXNlcl9pZD11c2VyXzE="

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "host": DATA_HOST,
    "region": "us",
}


def operation(**overrides: Any) -> dict[str, Any]:
    """A 2026-07 ``OperationModel`` body, with every field the schema declares."""
    body: dict[str, Any] = {
        "id": OPERATION_ID,
        "operation_type": "upload_file",
        "file_id": FILE_ID,
        "status": "Processing",
        "created_on": "2025-10-01T12:30:00Z",
        "completed_on": None,
        "percent_complete": 42,
        "error_message": None,
        "ingestion_units": None,
    }
    body.update(overrides)
    return body


PROCESSING = operation()
COMPLETED = operation(
    id="op-8765-dcba-4321",
    status="Completed",
    percent_complete=100,
    completed_on="2025-10-01T12:35:00Z",
    ingestion_units=50.0,
)
FAILED = operation(
    id="op-5555-eeee-9999",
    status="Failed",
    percent_complete=15,
    completed_on="2025-10-01T11:32:00Z",
    error_message="File processing failed: unsupported file format.",
)

OPERATION_LIST: dict[str, Any] = {
    "operations": [PROCESSING, COMPLETED, FAILED],
    "pagination": {"next": NEXT_TOKEN},
}


@pytest.fixture
def assistants(respx_mock: respx.MockRouter) -> Iterator[Assistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    client.close()


@api_op("assistant_data:list_operations")
def test_list_operations(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    page = assistants.list_operations_page(
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
def test_describe_operation(
    claim: Any, assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=PROCESSING)
    )

    result = assistants.describe_operation(assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID)
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


def test_a_completed_operation_reports_what_the_upload_cost(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """``ingestion_units`` and ``completed_on`` only appear once it finished."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{COMPLETED['id']}").mock(
        return_value=httpx.Response(200, json=COMPLETED)
    )

    result = assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=str(COMPLETED["id"])
    )
    assert result.status == "Completed"
    assert result.percent_complete == 100
    assert result.completed_on == "2025-10-01T12:35:00Z"
    assert result.ingestion_units == 50.0
    assert result.error is None


def test_a_failed_operation_carries_the_reason_and_no_ingestion_units(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """The failure text is the whole point of describing a failed operation."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{FAILED['id']}").mock(
        return_value=httpx.Response(200, json=FAILED)
    )

    result = assistants.describe_operation(
        assistant_name=ASSISTANT_NAME, operation_id=str(FAILED["id"])
    )
    assert result.status == "Failed"
    assert result.error == "File processing failed: unsupported file format."
    assert result.ingestion_units is None


def test_a_2026_04_minimal_operation_body_still_decodes(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """The enriched fields are all optional, so the older body is not a break."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200, json={"id": OPERATION_ID, "status": "Succeeded", "created_on": None}
        )
    )

    result = assistants.describe_operation(assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID)
    assert result.status == "Succeeded"
    assert result.operation_type is None
    assert result.file_id is None
    assert result.percent_complete is None


def test_describe_operation_404_reaches_the_caller(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """Operations are retained 30 days; after that the spec's 404 is what you get."""
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            404,
            json={
                "status": 404,
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Operation {OPERATION_ID} not found.",
                },
            },
        )
    )

    with pytest.raises(NotFoundError) as excinfo:
        assistants.describe_operation(assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID)

    assert OPERATION_ID in str(excinfo.value)


def test_list_operations_404_when_the_assistant_is_gone(
    assistants: Assistants, respx_mock: respx.MockRouter
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
        assistants.list_operations_page(assistant_name=ASSISTANT_NAME)


def test_no_filters_means_no_query_parameters(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """An unfiltered listing must not send empty filters the backend would parse."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={"operations": []})
    )

    page = assistants.list_operations_page(assistant_name=ASSISTANT_NAME)
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
def test_an_invalid_filter_never_reaches_the_wire(
    assistants: Assistants,
    respx_mock: respx.MockRouter,
    kwargs: dict[str, str],
    expected: str,
) -> None:
    """The enum is sealed backend-side; failing locally names the alternatives."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    with pytest.raises(PineconeValueError) as excinfo:
        assistants.list_operations_page(assistant_name=ASSISTANT_NAME, **kwargs)

    assert str(excinfo.value) == expected
    assert route.call_count == 0


def test_an_invalid_filter_on_the_paginator_raises_on_first_fetch(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """``list_operations`` is lazy, so the check lands when a page is fetched."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=OPERATION_LIST)
    )

    paginator = assistants.list_operations(assistant_name=ASSISTANT_NAME, status="done")
    with pytest.raises(PineconeValueError, match=r"status must be one of"):
        paginator.to_list()

    assert route.call_count == 0


def test_the_paginator_walks_the_cursor_the_response_hands_back(
    assistants: Assistants, respx_mock: respx.MockRouter
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

    paginator = assistants.list_operations(assistant_name=ASSISTANT_NAME)
    assert [op.operation_id for op in paginator] == [
        OPERATION_ID,
        "op-8765-dcba-4321",
        "op-5555-eeee-9999",
    ]
    assert paginator.pagination_token is None


def test_the_paginator_forwards_the_filters_to_every_page(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """A filtered listing stays filtered after the first page, or it lies."""
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        side_effect=[
            httpx.Response(200, json={"operations": [FAILED], "pagination": {"next": "token-1"}}),
            httpx.Response(200, json={"operations": [FAILED]}),
        ]
    )

    assistants.list_operations(
        assistant_name=ASSISTANT_NAME, operation_type="delete_file", status="Failed"
    ).to_list()

    assert route.call_count == 2
    for call in route.calls:
        assert call.request.url.params["operation_type"] == "delete_file"
        assert call.request.url.params["status"] == "Failed"
    assert "pagination_token" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["pagination_token"] == "token-1"


def test_limit_bounds_the_items_not_the_pages(
    assistants: Assistants, respx_mock: respx.MockRouter
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

    result = assistants.list_operations(assistant_name=ASSISTANT_NAME, limit=2).to_list()

    assert [op.operation_id for op in result] == [OPERATION_ID, "op-8765-dcba-4321"]
    assert route.call_count == 1


def test_pagination_token_resumes_where_a_previous_call_stopped(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={"operations": [FAILED]})
    )

    assistants.list_operations(assistant_name=ASSISTANT_NAME, pagination_token=NEXT_TOKEN).to_list()

    assert route.calls.last.request.url.params["pagination_token"] == NEXT_TOKEN


def test_describe_operation_and_the_polling_loop_share_one_client(
    respx_mock: respx.MockRouter,
) -> None:
    """The public method is the polling loop's code path, not a parallel one.

    ``_poll_operation_until_done`` calls ``describe_operation``, which resolves
    the data-plane host through the cached client — so an upload that polls
    twice still costs exactly one control-plane describe, and both the poll and
    a later public call hit the same URL with the same version header.
    """
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

    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    try:
        client.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
        result = client.describe_operation(assistant_name=ASSISTANT_NAME, operation_id=OPERATION_ID)
    finally:
        client.close()

    assert result.status == "Completed"
    assert describe_assistant.call_count == 1
    assert operations.call_count == 2
    for call in operations.calls:
        assert call.request.headers["x-pinecone-api-version"] == "2026-07"
        assert str(call.request.url) == f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}"


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    page_sizes=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=6),
    tokens=st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=12),
        min_size=6,
        max_size=6,
        unique=True,
    ),
)
def test_the_paginator_terminates_and_never_replays_a_consumed_token(
    page_sizes: list[int], tokens: list[str]
) -> None:
    """Any cursor sequence the server can hand back must terminate exactly once.

    Empty pages that still carry a token are the interesting case: the paginator
    has to keep walking without yielding anything, and it must never re-request
    a token it has already spent — that is the shape of an infinite loop against
    a live index.
    """
    cursors = tokens[: max(len(page_sizes) - 1, 0)]
    pages = [
        [operation(id=f"op-{index}-{item}") for item in range(size)]
        for index, size in enumerate(page_sizes)
    ]
    requested: list[str | None] = []

    def respond(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pagination_token")
        requested.append(token)
        index = 0 if token is None else cursors.index(token) + 1
        body: dict[str, Any] = {"operations": pages[index]}
        if index < len(cursors):
            body["pagination"] = {"next": cursors[index]}
        return httpx.Response(200, json=body)

    with respx.mock:
        respx.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=ASSISTANT)
        )
        respx.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}").mock(side_effect=respond)
        client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
        try:
            collected = client.list_operations(assistant_name=ASSISTANT_NAME).to_list()
        finally:
            client.close()

    assert [op.operation_id for op in collected] == [
        str(body["id"]) for page in pages for body in page
    ]
    assert requested == [None, *cursors]
    assert len(requested) == len(set(requested))
