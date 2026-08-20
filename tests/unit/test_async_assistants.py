"""Unit tests for AsyncAssistants namespace — create, describe, list, update."""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL
from pinecone.async_client.assistants import (
    _CREATE_POLL_INTERVAL_SECONDS,
    _DELETE_POLL_INTERVAL_SECONDS,
    _UPLOAD_POLL_INTERVAL_SECONDS,
    AsyncAssistants,
)
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
)
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.pagination import AsyncPaginator, Page
from tests.factories import (
    make_alignment_response,
    make_assistant_file_response,
    make_assistant_response,
    make_context_response,
    make_operation_response,
)

BASE_URL = "https://api.test.pinecone.io"
DATA_PLANE_HOST = "test-assistant-abc123.svc.pinecone.io"
DATA_PLANE_URL = f"https://{DATA_PLANE_HOST}/assistant"


@pytest.fixture
def async_assistants() -> AsyncAssistants:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncAssistants(config=config)


# ---------------------------------------------------------------------------
# create() — validation
# ---------------------------------------------------------------------------


async def test_create_assistant_region_validation(async_assistants: AsyncAssistants) -> None:
    """Invalid region raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="region") as exc_info:
        await async_assistants.create(name="test-assistant", region="ap-southeast-1")
    message = str(exc_info.value)
    assert "ap-southeast-1" in message
    assert "'us'" in message
    assert "'eu'" in message


async def test_create_assistant_region_case_sensitive(async_assistants: AsyncAssistants) -> None:
    """Uppercase 'US' and 'EU' are rejected — validation is case-sensitive."""
    with pytest.raises(PineconeValueError, match="region"):
        await async_assistants.create(name="test-assistant", region="US")

    with pytest.raises(PineconeValueError, match="region"):
        await async_assistants.create(name="test-assistant", region="EU")


# ---------------------------------------------------------------------------
# create() — defaults
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_defaults(async_assistants: AsyncAssistants) -> None:
    """Default region is 'us', metadata is omitted from body, instructions is None."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "test-assistant"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["region"] == "us"
    assert "metadata" not in body
    assert body["instructions"] is None


# ---------------------------------------------------------------------------
# create() — success with immediate return (timeout=-1)
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_immediate_return(async_assistants: AsyncAssistants) -> None:
    """timeout=-1 returns immediately without polling."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = await async_assistants.create(name="test-assistant", timeout=-1)

    assert isinstance(result, AssistantModel)
    assert result.status == "Initializing"
    assert route.call_count == 1


@respx.mock
async def test_create_assistant_with_all_params(async_assistants: AsyncAssistants) -> None:
    """Create with instructions, metadata, and region sends correct body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = await async_assistants.create(
        name="research-bot",
        instructions="You are a research assistant.",
        metadata={"team": "engineering", "version": "1"},
        region="eu",
        timeout=-1,
    )

    assert isinstance(result, AssistantModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "research-bot"
    assert body["instructions"] == "You are a research assistant."
    assert body["metadata"] == {"team": "engineering", "version": "1"}
    assert body["region"] == "eu"


# ---------------------------------------------------------------------------
# create() — polling
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_until_ready(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Polling loop calls GET until status is 'Ready'."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    poll_route = respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Ready")),
        ]
    )

    result = await async_assistants.create(name="test-assistant")

    assert result.status == "Ready"
    assert poll_route.call_count == 3


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_with_correct_interval(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Polling sleeps with the correct interval between polls."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant")

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_CREATE_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# create() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError with helpful message."""
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    with pytest.raises(PineconeTimeoutError, match="not ready") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=5)

    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_timeout_zero_polls_once(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """timeout=0 polls once, then raises if not ready."""
    mock_monotonic.side_effect = [0.0, 0.0, 0.1]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    poll_route = respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
        ]
    )

    with pytest.raises(PineconeTimeoutError):
        await async_assistants.create(name="test-assistant", timeout=0)

    assert poll_route.call_count >= 1


# ---------------------------------------------------------------------------
# create() — status check is case-sensitive
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_status_case_sensitive(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Status check uses exact 'Ready' — 'ready' or 'READY' does not match."""
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="ready")),
    )

    with pytest.raises(PineconeTimeoutError):
        await async_assistants.create(name="test-assistant", timeout=5)


# ---------------------------------------------------------------------------
# create() — indefinite polling (timeout=None)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """When timeout is None, polling continues until Ready with no deadline."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Ready")),
        ]
    )

    result = await async_assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


# ---------------------------------------------------------------------------
# create() — terminal error states
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_create_assistant_failed_status(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """'Failed' status raises PineconeError immediately, not a timeout."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Failed")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=None)

    assert "Failed" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_create_assistant_initialization_failed_status(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """'InitializationFailed' status raises PineconeError immediately, not a timeout."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(
            200, json=make_assistant_response(status="InitializationFailed")
        ),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=None)

    assert "InitializationFailed" in str(exc_info.value)


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_create_assistant_poll_terminal_terminated(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """'Terminated' status raises PineconeError immediately instead of looping."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Terminated")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=None)

    assert "Terminated" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_create_assistant_poll_terminal_terminating(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """'Terminating' status raises PineconeError immediately instead of looping."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Terminating")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=None)

    assert "Terminating" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


# ---------------------------------------------------------------------------
# create() — valid regions accepted
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_accepts_us_region(async_assistants: AsyncAssistants) -> None:
    """Region 'us' is accepted."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant", region="us")
    assert isinstance(result, AssistantModel)


@respx.mock
async def test_create_assistant_accepts_eu_region(async_assistants: AsyncAssistants) -> None:
    """Region 'eu' is accepted."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant", region="eu")
    assert isinstance(result, AssistantModel)


@respx.mock
async def test_create_assistant_metadata_default_omitted(
    async_assistants: AsyncAssistants,
) -> None:
    """When metadata is not provided, the 'metadata' key is absent from the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body


@respx.mock
async def test_create_assistant_metadata_explicit_dict_included(
    async_assistants: AsyncAssistants,
) -> None:
    """When metadata is provided, it is sent in the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant", metadata={"key": "value"}, timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metadata"] == {"key": "value"}


@respx.mock
async def test_create_assistant_environment_omitted_by_default(
    async_assistants: AsyncAssistants,
) -> None:
    """When environment is not provided, the 'environment' key is absent from the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "environment" not in body


@respx.mock
async def test_create_assistant_environment_included_when_provided(
    async_assistants: AsyncAssistants,
) -> None:
    """When environment is provided, it is sent in the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant", environment="prod-us", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["environment"] == "prod-us"


@respx.mock
async def test_create_assistant_environment_403_propagates(
    async_assistants: AsyncAssistants,
) -> None:
    """A 403 from the backend when environment is set propagates as ApiError."""
    from pinecone.errors.exceptions import ApiError

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            403, json={"error": {"code": "FORBIDDEN", "message": "Not authorized"}}
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        await async_assistants.create(name="test-assistant", environment="prod-us", timeout=-1)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# describe() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_assistant(async_assistants: AsyncAssistants) -> None:
    """describe() sends GET /assistants/{name} and returns AssistantModel."""
    response_data = make_assistant_response(
        name="my-assistant",
        status="Ready",
        instructions="Be helpful.",
        metadata={"team": "ml"},
        host="my-assistant-abc.svc.pinecone.io",
    )
    route = respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    result = await async_assistants.describe(name="my-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.instructions == "Be helpful."
    assert result.metadata == {"team": "ml"}
    assert result.host == "https://my-assistant-abc.svc.pinecone.io"
    assert result.created_at == "2025-01-15T12:00:00Z"
    assert result.updated_at == "2025-01-15T12:00:00Z"
    assert route.call_count == 1


@respx.mock
async def test_describe_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """describe() lets 404 errors propagate from the HTTP client."""
    respx.get(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.describe(name="nonexistent")


@respx.mock
async def test_describe_assistant_minimal_response(async_assistants: AsyncAssistants) -> None:
    """describe() handles response with optional fields absent."""
    response_data = make_assistant_response(
        name="minimal",
        metadata=None,
        instructions=None,
        host=None,
    )
    respx.get(f"{BASE_URL}/assistant/assistants/minimal").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    result = await async_assistants.describe(name="minimal")

    assert result.name == "minimal"
    assert result.metadata is None
    assert result.instructions is None
    assert result.host is None


# ---------------------------------------------------------------------------
# list() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_assistants_empty(async_assistants: AsyncAssistants) -> None:
    """list() returns an AsyncPaginator that yields nothing when no assistants exist."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    result = async_assistants.list()

    assert isinstance(result, AsyncPaginator)
    assert await result.to_list() == []


@respx.mock
async def test_list_assistants_single_page(async_assistants: AsyncAssistants) -> None:
    """list() returns an AsyncPaginator over all assistants from a single-page response."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                ],
            },
        ),
    )

    result = await async_assistants.list().to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"
    assert all(isinstance(a, AssistantModel) for a in result)


@respx.mock
async def test_list_assistants_multi_page(async_assistants: AsyncAssistants) -> None:
    """list() auto-paginates through multiple pages via AsyncPaginator."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a1")],
                    "pagination": {"next": "token-page2"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a2")],
                    "pagination": {"next": "token-page3"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a3")],
                },
            ),
        ]
    )

    result = await async_assistants.list().to_list()

    assert len(result) == 3
    assert [a.name for a in result] == ["a1", "a2", "a3"]


@respx.mock
async def test_list_assistants_to_list(async_assistants: AsyncAssistants) -> None:
    """list().to_list() collects all assistants into a list."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                ],
            },
        ),
    )

    items = await async_assistants.list().to_list()

    assert len(items) == 2
    assert all(isinstance(a, AssistantModel) for a in items)


@respx.mock
async def test_list_assistants_iteration(async_assistants: AsyncAssistants) -> None:
    """list() supports direct async for-loop iteration."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                ],
            },
        ),
    )

    names = [a.name async for a in async_assistants.list()]

    assert names == ["a1", "a2"]


@respx.mock
async def test_list_assistants_with_limit(async_assistants: AsyncAssistants) -> None:
    """list(limit=N) yields at most N items across all pages."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                    make_assistant_response(name="a3"),
                ],
                "next": "more",
            },
        ),
    )

    result = await async_assistants.list(limit=2).to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"


@respx.mock
async def test_list_assistants_pages(async_assistants: AsyncAssistants) -> None:
    """list().pages() yields Page objects with items and has_more."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a1")],
                    "pagination": {"next": "token-next"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a2")],
                },
            ),
        ]
    )

    pages = [p async for p in async_assistants.list().pages()]

    assert len(pages) == 2
    assert pages[0].has_more is True
    assert pages[0].items[0].name == "a1"
    assert pages[1].has_more is False
    assert pages[1].items[0].name == "a2"
    for page in pages:
        assert isinstance(page, Page)


@respx.mock
async def test_list_assistants_with_pagination_token(async_assistants: AsyncAssistants) -> None:
    """list(pagination_token=...) starts from the given token."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={"assistants": [make_assistant_response(name="a2")]},
        ),
    )

    result = await async_assistants.list(pagination_token="tok-page2").to_list()

    assert len(result) == 1
    assert result[0].name == "a2"
    request = route.calls.last.request
    assert "pagination_token=tok-page2" in str(request.url)


# ---------------------------------------------------------------------------
# list_page() — single page with pagination control
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_assistants_page(async_assistants: AsyncAssistants) -> None:
    """list_page() returns single page with next token."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
                "pagination": {"next": "token-next"},
            },
        ),
    )

    result = await async_assistants.list_page()

    assert isinstance(result, ListAssistantsResponse)
    assert len(result.assistants) == 1
    assert result.assistants[0].name == "a1"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
async def test_list_assistants_page_last_page(async_assistants: AsyncAssistants) -> None:
    """list_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
            },
        ),
    )

    result = await async_assistants.list_page()

    assert result.next is None


@respx.mock
async def test_list_assistants_page_with_page_size(async_assistants: AsyncAssistants) -> None:
    """list_page() sends limit query param when provided."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page(page_size=5)

    request = route.calls.last.request
    assert "limit=5" in str(request.url)


@respx.mock
async def test_list_assistants_page_with_pagination_token(
    async_assistants: AsyncAssistants,
) -> None:
    """list_page() sends pagination_token query param when provided."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page(pagination_token="abc123")

    request = route.calls.last.request
    assert "pagination_token=abc123" in str(request.url)


@respx.mock
async def test_list_assistants_page_omits_none_params(
    async_assistants: AsyncAssistants,
) -> None:
    """list_page() does not send params that are None."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page()

    request = route.calls.last.request
    assert "limit" not in str(request.url)
    assert "pagination_token" not in str(request.url)


# ---------------------------------------------------------------------------
# update() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_update_assistant_instructions(async_assistants: AsyncAssistants) -> None:
    """update() sends PATCH /assistants/{name} with instructions."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="Updated instructions.",
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(
        name="my-assistant", instructions="Updated instructions."
    )

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.instructions == "Updated instructions."
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "Updated instructions."}


@respx.mock
async def test_update_assistant_metadata(async_assistants: AsyncAssistants) -> None:
    """update() sends PATCH with metadata."""
    new_metadata = {"team": "ml", "version": "2"}
    updated_response = make_assistant_response(
        name="my-assistant",
        metadata=new_metadata,
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(name="my-assistant", metadata=new_metadata)

    assert isinstance(result, AssistantModel)
    assert result.metadata == new_metadata
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"metadata": new_metadata}


@respx.mock
async def test_update_assistant_both_fields(async_assistants: AsyncAssistants) -> None:
    """update() sends both instructions and metadata when provided."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )

    assert result.instructions == "New instructions."
    assert result.metadata == {"env": "prod"}

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "New instructions.", "metadata": {"env": "prod"}}


@respx.mock
async def test_update_assistant_omits_none_fields(async_assistants: AsyncAssistants) -> None:
    """update() only includes provided fields in the request body."""
    updated_response = make_assistant_response(name="my-assistant")
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    await async_assistants.update(name="my-assistant", instructions="Only this.")

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body
    assert body == {"instructions": "Only this."}


@respx.mock
async def test_update_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """update() lets 404 errors propagate from the HTTP client."""
    respx.patch(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.update(name="nonexistent", instructions="test")


# ---------------------------------------------------------------------------
# delete() — polls until gone
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_delete_assistant(mock_sleep: object, async_assistants: AsyncAssistants) -> None:
    """delete() sends DELETE then polls describe until 404 confirms deletion."""
    route = respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    result = await async_assistants.delete(name="my-assistant")

    assert result is None
    assert route.call_count == 1

    request = route.calls.last.request
    assert request.method == "DELETE"
    assert str(request.url) == f"{BASE_URL}/assistant/assistants/my-assistant"


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_delete_assistant_polls_until_gone(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """delete() polls describe every 5s; returns when 404 is received."""
    respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    describe_route = respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        side_effect=[
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(404, json={"error": "Not found"}),
        ],
    )

    result = await async_assistants.delete(name="my-assistant")

    assert result is None
    assert describe_route.call_count == 3

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
async def test_delete_assistant_timeout_minus_one_skips_polling(
    async_assistants: AsyncAssistants,
) -> None:
    """delete(timeout=-1) returns immediately without polling."""
    delete_route = respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )

    result = await async_assistants.delete(name="my-assistant", timeout=-1)

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_delete_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200, json=make_assistant_response(name="my-assistant", status="Terminating")
        ),
    )

    with pytest.raises(PineconeTimeoutError, match="still exists after 10"):
        await async_assistants.delete(name="my-assistant", timeout=10)


@respx.mock
async def test_delete_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """delete() lets 404 errors from the initial DELETE propagate."""
    respx.delete(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.delete(name="nonexistent")


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_repr(async_assistants: AsyncAssistants) -> None:
    assert repr(async_assistants) == "AsyncAssistants()"


CONTROL_PLANE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# AsyncPinecone.assistant() convenience method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_pinecone_assistant_convenience_method() -> None:
    """pc.assistant(name=...) delegates to GET /assistants/{name} and returns AssistantModel."""
    from pinecone.async_client.pinecone import AsyncPinecone

    response_data = make_assistant_response(
        name="my-assistant",
        status="Ready",
        instructions="Be helpful.",
        metadata={"team": "ml"},
        host="my-assistant-abc.svc.pinecone.io",
    )
    route = respx.get(f"{CONTROL_PLANE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    pc = AsyncPinecone(api_key="test-key")
    result = await pc.assistant("my-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.instructions == "Be helpful."
    assert result.metadata == {"team": "ml"}
    assert result.host == "https://my-assistant-abc.svc.pinecone.io"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_pinecone_assistant_not_found() -> None:
    """pc.assistant(name=...) raises NotFoundError when assistant does not exist."""
    from pinecone.async_client.pinecone import AsyncPinecone

    respx.get(f"{CONTROL_PLANE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    pc = AsyncPinecone(api_key="test-key")
    with pytest.raises(NotFoundError):
        await pc.assistant("nonexistent")


# ---------------------------------------------------------------------------
# AsyncPinecone.assistants property
# ---------------------------------------------------------------------------


def test_async_pinecone_assistants_property() -> None:
    from pinecone.async_client.pinecone import AsyncPinecone

    pc = AsyncPinecone(api_key="test-key")
    assistants = pc.assistants
    assert isinstance(assistants, AsyncAssistants)
    # Verify lazy caching — same instance returned
    assert pc.assistants is assistants


# ---------------------------------------------------------------------------
# __init__ — data-plane and eval HTTP infrastructure
# ---------------------------------------------------------------------------


def test_init_creates_eval_http(async_assistants: AsyncAssistants) -> None:
    """Constructor creates _eval_http targeting ASSISTANT_EVALUATION_BASE_URL."""
    from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL
    from pinecone._internal.http_client import AsyncHTTPClient

    assert hasattr(async_assistants, "_eval_http")
    assert isinstance(async_assistants._eval_http, AsyncHTTPClient)
    # _eval_http should be configured with the evaluation base URL
    assert async_assistants._eval_http._config.host == ASSISTANT_EVALUATION_BASE_URL


def test_init_uses_data_host_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PINECONE_PLUGIN_ASSISTANT_DATA_HOST is set, async eval client targets it."""
    from pinecone import AsyncPinecone

    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_DATA_HOST",
        "https://staging-data.ke.pinecone.io",
    )
    pc = AsyncPinecone(api_key="test-key", host="https://api.test.pinecone.io")
    assert pc.assistants._eval_http._config.host == "https://staging-data.ke.pinecone.io/assistant"


def test_init_data_host_env_var_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slash on the async env var is normalized."""
    from pinecone import AsyncPinecone

    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_DATA_HOST",
        "https://staging-data.ke.pinecone.io/",
    )
    pc = AsyncPinecone(api_key="test-key", host="https://api.test.pinecone.io")
    assert pc.assistants._eval_http._config.host == "https://staging-data.ke.pinecone.io/assistant"


def test_init_uses_control_host_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When config.host is empty, async PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST is used."""
    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST",
        "https://api-staging.pinecone.io",
    )
    monkeypatch.delenv("PINECONE_HOST", raising=False)
    config = PineconeConfig(api_key="test-key", host="")
    assistants = AsyncAssistants(config=config)
    assert assistants._http._config.host == "https://api-staging.pinecone.io/assistant"


def test_init_explicit_host_overrides_control_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config.host wins over PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST in async."""
    from pinecone import AsyncPinecone

    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST",
        "https://api-staging.pinecone.io",
    )
    pc = AsyncPinecone(api_key="test-key", host="https://custom.example.com")
    assert pc.assistants._http._config.host == "https://custom.example.com/assistant"


def test_init_creates_empty_data_plane_clients(async_assistants: AsyncAssistants) -> None:
    """Constructor initializes _data_plane_clients as an empty dict."""
    assert hasattr(async_assistants, "_data_plane_clients")
    assert isinstance(async_assistants._data_plane_clients, dict)
    assert len(async_assistants._data_plane_clients) == 0


# ---------------------------------------------------------------------------
# close() — eval_http and data-plane clients
# ---------------------------------------------------------------------------


@respx.mock
async def test_close_closes_eval_http(async_assistants: AsyncAssistants) -> None:
    """close() closes _eval_http in addition to _http."""
    from unittest.mock import AsyncMock

    async_assistants._eval_http.close = AsyncMock()  # type: ignore[method-assign]

    await async_assistants.close()

    async_assistants._eval_http.close.assert_called_once()  # type: ignore[union-attr]


@respx.mock
async def test_close_closes_data_plane_clients(async_assistants: AsyncAssistants) -> None:
    """close() closes all cached data-plane clients and clears the cache."""
    from unittest.mock import AsyncMock

    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient

    # Inject a fake data-plane client into the cache
    fake_client = AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host="https://fake.host"), "2025-10"
    )
    fake_client.close = AsyncMock()  # type: ignore[method-assign]
    async_assistants._data_plane_clients["my-assistant"] = fake_client

    await async_assistants.close()

    fake_client.close.assert_called_once()  # type: ignore[union-attr]
    assert len(async_assistants._data_plane_clients) == 0


# ---------------------------------------------------------------------------
# _data_plane_http() — caching and error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_data_plane_http_returns_client_for_host(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() returns an AsyncHTTPClient configured with the assistant's host."""
    from pinecone._internal.http_client import AsyncHTTPClient

    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(
                name="my-assistant", host="my-assistant-abc.svc.pinecone.io"
            ),
        ),
    )

    client = await async_assistants._data_plane_http("my-assistant")

    assert isinstance(client, AsyncHTTPClient)
    assert client._config.host == "https://my-assistant-abc.svc.pinecone.io/assistant"


@respx.mock
async def test_data_plane_http_caches_client(async_assistants: AsyncAssistants) -> None:
    """_data_plane_http() returns the same client on repeated calls (no extra describe)."""
    describe_route = respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(
                name="my-assistant", host="my-assistant-abc.svc.pinecone.io"
            ),
        ),
    )

    client1 = await async_assistants._data_plane_http("my-assistant")
    client2 = await async_assistants._data_plane_http("my-assistant")

    assert client1 is client2
    assert describe_route.call_count == 1


@respx.mock
async def test_data_plane_http_different_assistants_get_different_clients(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() caches separately per assistant name."""
    respx.get(f"{BASE_URL}/assistant/assistants/assistant-a").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="assistant-a", host="host-a.svc.pinecone.io"),
        ),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/assistant-b").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="assistant-b", host="host-b.svc.pinecone.io"),
        ),
    )

    client_a = await async_assistants._data_plane_http("assistant-a")
    client_b = await async_assistants._data_plane_http("assistant-b")

    assert client_a is not client_b
    assert client_a._config.host == "https://host-a.svc.pinecone.io/assistant"
    assert client_b._config.host == "https://host-b.svc.pinecone.io/assistant"


@respx.mock
async def test_data_plane_http_raises_when_host_is_none(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() raises PineconeValueError when assistant has no host."""
    respx.get(f"{BASE_URL}/assistant/assistants/no-host-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="no-host-assistant", host=None),
        ),
    )

    with pytest.raises(PineconeValueError, match="no data-plane host"):
        await async_assistants._data_plane_http("no-host-assistant")


@respx.mock
async def test_data_plane_http_raises_when_host_is_empty_string(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() raises PineconeValueError when assistant host is empty string."""
    respx.get(f"{BASE_URL}/assistant/assistants/empty-host-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="empty-host-assistant", host=""),
        ),
    )

    with pytest.raises(PineconeValueError, match="no data-plane host"):
        await async_assistants._data_plane_http("empty-host-assistant")


# ---------------------------------------------------------------------------
# describe_file() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_describe_file_success(async_assistants: AsyncAssistants) -> None:
    """describe_file() sends GET /files/{name}/{id} via data-plane and returns AssistantFileModel."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    result = await async_assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123"
    )

    assert isinstance(result, AssistantFileModel)
    assert result.id == "file-abc123"
    assert result.name == "test-file.pdf"
    assert route.call_count == 1


@respx.mock
async def test_async_describe_file_without_url(async_assistants: AsyncAssistants) -> None:
    """describe_file() does not send include_url param by default."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    await async_assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    request = route.calls.last.request
    assert "include_url" not in str(request.url)


@respx.mock
async def test_async_describe_file_with_url(async_assistants: AsyncAssistants) -> None:
    """describe_file(include_url=True) sends include_url=true query param."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(signed_url="https://storage.example.com/file-abc123"),
        ),
    )

    result = await async_assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123", include_url=True
    )

    request = route.calls.last.request
    assert "include_url=true" in str(request.url)
    assert result.signed_url == "https://storage.example.com/file-abc123"


@respx.mock
async def test_async_describe_file_not_found(async_assistants: AsyncAssistants) -> None:
    """describe_file() raises NotFoundError when file does not exist."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.describe_file(assistant_name="test-assistant", file_id="nonexistent")


# ---------------------------------------------------------------------------
# list_files_page() — single page
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_list_files_page_success(async_assistants: AsyncAssistants) -> None:
    """list_files_page() returns ListFilesResponse with files and next token."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [make_assistant_file_response()],
                "pagination": {"next": "token-next"},
            },
        ),
    )

    result = await async_assistants.list_files_page(assistant_name="test-assistant")

    assert isinstance(result, ListFilesResponse)
    assert len(result.files) == 1
    assert result.files[0].id == "file-abc123"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
async def test_async_list_files_page_last_page(async_assistants: AsyncAssistants) -> None:
    """list_files_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": [make_assistant_file_response()]}),
    )

    result = await async_assistants.list_files_page(assistant_name="test-assistant")

    assert result.next is None


@respx.mock
async def test_async_list_files_page_no_page_size_param(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() does not send a limit query param when page_size is omitted."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "limit" not in str(request.url)


@respx.mock
async def test_async_list_files_page_sends_page_size(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() sends limit query param when page_size is provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(assistant_name="test-assistant", page_size=20)

    request = route.calls.last.request
    assert "limit=20" in str(request.url)


@respx.mock
async def test_async_list_files_page_with_pagination_token(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() sends pagination_token query param when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(
        assistant_name="test-assistant", pagination_token="tok123"
    )

    request = route.calls.last.request
    assert "pagination_token=tok123" in str(request.url)


@respx.mock
async def test_async_list_files_page_with_filter(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() serializes filter dict to JSON string query param."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str


@respx.mock
async def test_async_list_files_page_omits_none_params(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() does not send params that are None."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "limit" not in str(request.url)
    assert "pagination_token" not in str(request.url)
    assert "filter" not in str(request.url)


# ---------------------------------------------------------------------------
# list_files() — AsyncPaginator
# ---------------------------------------------------------------------------


def test_async_list_files_returns_async_paginator(async_assistants: AsyncAssistants) -> None:
    """list_files() returns an AsyncPaginator instance (not a list)."""
    result = async_assistants.list_files(assistant_name="test-assistant")

    assert isinstance(result, AsyncPaginator)


@respx.mock
async def test_async_list_files_empty(async_assistants: AsyncAssistants) -> None:
    """list_files() yields no items when no files exist."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    result = await async_assistants.list_files(assistant_name="test-assistant").to_list()

    assert result == []


@respx.mock
async def test_async_list_files_single_page(async_assistants: AsyncAssistants) -> None:
    """list_files() yields all files from a single-page response."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [
                    make_assistant_file_response(id="f1", name="file1.pdf"),
                    make_assistant_file_response(id="f2", name="file2.pdf"),
                ],
            },
        ),
    )

    result = await async_assistants.list_files(assistant_name="test-assistant").to_list()

    assert len(result) == 2
    assert all(isinstance(f, AssistantFileModel) for f in result)
    assert result[0].id == "f1"
    assert result[1].id == "f2"


@respx.mock
async def test_async_list_files_multi_page(async_assistants: AsyncAssistants) -> None:
    """list_files() auto-paginates through multiple pages collecting all files."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f1", name="file1.pdf")],
                    "pagination": {"next": "token-page2"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f2", name="file2.pdf")],
                    "pagination": {"next": "token-page3"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f3", name="file3.pdf")],
                },
            ),
        ]
    )

    result = await async_assistants.list_files(assistant_name="test-assistant").to_list()

    assert len(result) == 3
    assert [f.id for f in result] == ["f1", "f2", "f3"]


@respx.mock
async def test_async_list_files_with_filter(async_assistants: AsyncAssistants) -> None:
    """list_files() passes filter to list_files_page on each request."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={"files": [make_assistant_file_response()]},
        ),
    )

    result = await async_assistants.list_files(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    ).to_list()

    assert len(result) == 1
    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str


@respx.mock
async def test_async_list_files_limit_accepted(async_assistants: AsyncAssistants) -> None:
    """list_files() accepts a limit parameter and stops after that many items."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [
                    make_assistant_file_response(id="f1", name="file1.pdf"),
                    make_assistant_file_response(id="f2", name="file2.pdf"),
                    make_assistant_file_response(id="f3", name="file3.pdf"),
                ],
                "pagination": {"next": "token-page2"},
            },
        ),
    )

    result = await async_assistants.list_files(assistant_name="test-assistant", limit=2).to_list()

    assert len(result) == 2
    assert result[0].id == "f1"
    assert result[1].id == "f2"


@respx.mock
async def test_async_list_files_pagination_token_accepted(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files() accepts a pagination_token to resume from a previous page."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={"files": [make_assistant_file_response(id="f2", name="file2.pdf")]},
        ),
    )

    result = await async_assistants.list_files(
        assistant_name="test-assistant", pagination_token="token-page2"
    ).to_list()

    assert len(result) == 1
    assert result[0].id == "f2"
    request = route.calls.last.request
    assert "pagination_token=token-page2" in str(request.url)


@respx.mock
async def test_async_list_files_async_for_loop(async_assistants: AsyncAssistants) -> None:
    """list_files() supports async for iteration."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={"files": [make_assistant_file_response(id="f1", name="file1.pdf")]},
        ),
    )

    collected = [f async for f in async_assistants.list_files(assistant_name="test-assistant")]

    assert len(collected) == 1
    assert isinstance(collected[0], AssistantFileModel)
    assert collected[0].id == "f1"


# ---------------------------------------------------------------------------
# upload_file() — validation
# ---------------------------------------------------------------------------


async def test_async_upload_file_neither_path_nor_stream(async_assistants: AsyncAssistants) -> None:
    """Providing neither file_path nor file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        await async_assistants.upload_file(assistant_name="test-assistant")


async def test_async_upload_file_both_path_and_stream(async_assistants: AsyncAssistants) -> None:
    """Providing both file_path and file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        await async_assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/some/path.pdf",
            file_stream=io.BytesIO(b"data"),
        )


async def test_async_upload_file_path_not_found(async_assistants: AsyncAssistants) -> None:
    """Uploading from a nonexistent local path raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="File not found"):
        await async_assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/nonexistent/path/document.pdf",
        )


# ---------------------------------------------------------------------------
# upload_file() — success from stream
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_from_stream(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """upload_file with file_stream uploads and polls until Available."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Available")),
        ]
    )

    stream = io.BytesIO(b"test file content")
    result = await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="report.pdf",
    )

    assert isinstance(result, AssistantFileModel)
    assert result.status == "Available"
    assert result.id == "file-abc123"
    assert upload_route.call_count == 1
    assert poll_route.call_count == 2


# ---------------------------------------------------------------------------
# upload_file() — multimodal serialization
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_multimodal_true(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """multimodal=True is serialized as 'true' in query params."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"content")
    await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        multimodal=True,
    )

    request = upload_route.calls.last.request
    assert "multimodal=true" in str(request.url)


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_multimodal_false(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """multimodal=False is serialized as 'false' in query params."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"content")
    await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        multimodal=False,
    )

    request = upload_route.calls.last.request
    assert "multimodal=false" in str(request.url)


# ---------------------------------------------------------------------------
# upload_file() — metadata and file_id
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_with_metadata_and_file_id(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """upload_file(file_id=...) uses PUT on the 2026-04 upsert endpoint and polls the operation."""
    # _upsert_http calls describe() to get the data-plane host
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    # Upsert: PUT /files/{assistant_name}/{file_id} (2026-04 API)
    upsert_route = respx.put(f"{DATA_PLANE_URL}/files/test-assistant/custom-file-id").mock(
        return_value=httpx.Response(202, json=make_operation_response(status="Succeeded")),
    )
    # Poll: GET /operations/{assistant_name}/{operation_id}
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_operation_response(status="Succeeded")),
    )
    # describe_file after upsert
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/custom-file-id").mock(
        return_value=httpx.Response(
            200, json=make_assistant_file_response(id="custom-file-id", status="Available")
        ),
    )

    stream = io.BytesIO(b"content")
    result = await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        metadata={"genre": "comedy"},
        file_id="custom-file-id",
    )

    # Must use PUT, not POST
    assert upsert_route.call_count == 1
    request = upsert_route.calls.last.request
    url_str = str(request.url)
    # file_id is in the path, not a query param
    assert "/files/test-assistant/custom-file-id" in url_str
    assert "file_id=" not in url_str
    # v202604 rejects metadata as a query param; it must be in the multipart body
    assert "metadata=" not in url_str
    body = request.content.decode("latin-1")
    assert "genre" in body and "comedy" in body
    # Returned model has the caller-specified file id
    assert isinstance(result, AssistantFileModel)
    assert result.id == "custom-file-id"


# ---------------------------------------------------------------------------
# upload_file() — polling interval
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_polls_with_correct_interval(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """upload_file polls every _UPLOAD_POLL_INTERVAL_SECONDS seconds."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Available")),
        ]
    )

    stream = io.BytesIO(b"data")
    await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
    )

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_UPLOAD_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# upload_file() — processing failure
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_processing_failed(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """If processing fails, raises PineconeError pointing at the operations API.

    ``2026-07`` dropped ``AssistantFileModel.error_message``, so the failure
    reason is no longer on the file payload to quote.
    """
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(status="ProcessingFailed"),
        ),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeError, match="describe_operation"):
        await async_assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
        )


# ---------------------------------------------------------------------------
# upload_file() — upsert operation failure (error_message field name)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_upsert_error_message(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Upsert polling surfaces backend error_message, not the fallback 'Unknown operation error'."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.put(f"{DATA_PLANE_URL}/files/test-assistant/custom-file-id").mock(
        return_value=httpx.Response(202, json=make_operation_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_operation_response(
                status="Failed", error_message="Conflict: file already being processed"
            ),
        ),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeError, match="Conflict: file already being processed"):
        await async_assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
            file_id="custom-file-id",
        )


# ---------------------------------------------------------------------------
# upload_file() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_upload_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Timeout raises PineconeTimeoutError with operation ID in message."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeTimeoutError, match="file-abc123") as exc_info:
        await async_assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
            timeout=10,
        )

    assert "operation_id" in str(exc_info.value)


@respx.mock
async def test_async_upload_timeout_negative_one(async_assistants: AsyncAssistants) -> None:
    """upload_file(timeout=-1) calls describe_file once without polling."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    describe_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )

    stream = io.BytesIO(b"data")
    result = await async_assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        timeout=-1,
    )

    assert upload_route.call_count == 1
    assert describe_route.call_count == 1
    assert isinstance(result, AssistantFileModel)
    assert result.status == "Processing"


# ---------------------------------------------------------------------------
# delete_file() — success
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_file_success(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """delete_file() sends DELETE then polls until 404 confirms deletion."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    result = await async_assistants.delete_file(
        assistant_name="test-assistant", file_id="file-abc123"
    )

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_file_polls_until_gone(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """delete_file() polls describe_file every 5s; returns when 404 received."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
            httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
            httpx.Response(404, json={"error": "Not found"}),
        ]
    )

    result = await async_assistants.delete_file(
        assistant_name="test-assistant", file_id="file-abc123"
    )

    assert result is None
    assert poll_route.call_count == 3

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
async def test_async_delete_file_timeout_minus_one_skips_polling(
    async_assistants: AsyncAssistants,
) -> None:
    """delete_file(timeout=-1) returns immediately without polling."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )

    result = await async_assistants.delete_file(
        assistant_name="test-assistant", file_id="file-abc123", timeout=-1
    )

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
    )

    with pytest.raises(PineconeTimeoutError, match="still exists after 10"):
        await async_assistants.delete_file(
            assistant_name="test-assistant", file_id="file-abc123", timeout=10
        )


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_file_server_error_raises(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """delete_file() raises PineconeError if server-side deletion fails."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(status="ProcessingFailed"),
        ),
    )

    with pytest.raises(PineconeError, match="'ProcessingFailed'"):
        await async_assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")


# ---------------------------------------------------------------------------
# context() — validation
# ---------------------------------------------------------------------------


async def test_async_context_both_query_and_messages(
    async_assistants: AsyncAssistants,
) -> None:
    """Providing both query and messages raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="not both"):
        await async_assistants.context(
            assistant_name="test-assistant",
            query="What is Pinecone?",
            messages=[{"content": "Hello"}],
        )


async def test_async_context_neither_query_nor_messages(
    async_assistants: AsyncAssistants,
) -> None:
    """Providing neither query nor messages raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        await async_assistants.context(assistant_name="test-assistant")


async def test_async_context_empty_string_query(async_assistants: AsyncAssistants) -> None:
    """Empty string query is treated as not provided — raises if messages also absent."""
    with pytest.raises(PineconeValueError):
        await async_assistants.context(assistant_name="test-assistant", query="")


async def test_async_context_empty_list_messages(async_assistants: AsyncAssistants) -> None:
    """Empty list messages is treated as not provided — raises if query also absent."""
    with pytest.raises(PineconeValueError):
        await async_assistants.context(assistant_name="test-assistant", messages=[])


async def test_async_context_top_k_negative_raises(async_assistants: AsyncAssistants) -> None:
    """Negative top_k raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="top_k"):
        await async_assistants.context(assistant_name="test-assistant", query="test", top_k=-1)


async def test_async_context_snippet_size_negative_raises(
    async_assistants: AsyncAssistants,
) -> None:
    """Negative snippet_size raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="snippet_size"):
        await async_assistants.context(
            assistant_name="test-assistant", query="test", snippet_size=-1
        )


# ---------------------------------------------------------------------------
# context() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_context_with_query(async_assistants: AsyncAssistants) -> None:
    """context() with query POSTs to /chat/{name}/context and returns ContextResponse."""
    from pinecone.models.assistant.context import ContextResponse

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    result = await async_assistants.context(
        assistant_name="test-assistant",
        query="What is Pinecone?",
    )

    assert isinstance(result, ContextResponse)
    assert len(result.snippets) == 1

    request_body = json.loads(context_route.calls.last.request.content)
    assert request_body["query"] == "What is Pinecone?"
    assert "messages" not in request_body


@respx.mock
async def test_async_context_with_messages(async_assistants: AsyncAssistants) -> None:
    """context() with messages parses and sends them; does not send query."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    await async_assistants.context(
        assistant_name="test-assistant",
        messages=[{"content": "Tell me about vector databases."}],
    )

    request_body = json.loads(context_route.calls.last.request.content)
    assert "query" not in request_body
    assert request_body["messages"] == [
        {"role": "user", "content": "Tell me about vector databases."}
    ]


@respx.mock
async def test_async_context_optional_params_included_when_provided(
    async_assistants: AsyncAssistants,
) -> None:
    """Optional parameters are sent in the request body when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    await async_assistants.context(
        assistant_name="test-assistant",
        query="What is Pinecone?",
        filter={"genre": {"$ne": "documentary"}},
        top_k=5,
        snippet_size=1024,
        multimodal=True,
        include_binary_content=False,
    )

    request_body = json.loads(context_route.calls.last.request.content)
    assert request_body["filter"] == {"genre": {"$ne": "documentary"}}
    assert request_body["top_k"] == 5
    assert request_body["snippet_size"] == 1024
    assert request_body["multimodal"] is True
    assert request_body["include_binary_content"] is False


@respx.mock
async def test_async_context_optional_params_omitted_when_absent(
    async_assistants: AsyncAssistants,
) -> None:
    """Optional parameters are not included in the request body when not provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    await async_assistants.context(
        assistant_name="test-assistant",
        query="What is Pinecone?",
    )

    request_body = json.loads(context_route.calls.last.request.content)
    assert "filter" not in request_body
    assert "top_k" not in request_body
    assert "snippet_size" not in request_body
    assert "multimodal" not in request_body


# ---------------------------------------------------------------------------
# evaluate_alignment() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_evaluate_alignment(async_assistants: AsyncAssistants) -> None:
    """evaluate_alignment() POSTs to evaluation endpoint and returns AlignmentResult."""
    from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL
    from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores

    eval_route = respx.post(f"{ASSISTANT_EVALUATION_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    result = await async_assistants.evaluate_alignment(
        question="What is the capital of Spain?",
        answer="Barcelona.",
        ground_truth_answer="Madrid.",
    )

    assert isinstance(result, AlignmentResult)
    assert isinstance(result.scores, AlignmentScores)
    assert result.scores.correctness == 0.0
    assert result.scores.completeness == 1.0
    assert result.scores.alignment == 0.0
    assert len(result.facts) == 1
    assert result.facts[0].fact == "Madrid is the capital of Spain."
    assert result.facts[0].entailment == "entailed"
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 40
    assert result.usage.total_tokens == 160

    request = eval_route.calls.last.request
    body = json.loads(request.content)
    assert body["question"] == "What is the capital of Spain?"
    assert body["answer"] == "Barcelona."
    assert body["ground_truth_answer"] == "Madrid."


@respx.mock
async def test_async_evaluate_alignment_multiple_facts(async_assistants: AsyncAssistants) -> None:
    """evaluate_alignment() correctly maps multiple evaluated_facts from the response."""
    from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL

    multi_fact_response = make_alignment_response(
        reasoning={
            "evaluated_facts": [
                {"fact": {"content": "Fact one."}, "entailment": "entailed"},
                {"fact": {"content": "Fact two."}, "entailment": "contradicted"},
                {"fact": {"content": "Fact three."}, "entailment": "neutral"},
            ]
        }
    )
    respx.post(f"{ASSISTANT_EVALUATION_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=multi_fact_response),
    )

    result = await async_assistants.evaluate_alignment(
        question="Q?",
        answer="A.",
        ground_truth_answer="GT.",
    )

    assert len(result.facts) == 3
    assert result.facts[0].fact == "Fact one."
    assert result.facts[0].entailment == "entailed"
    assert result.facts[1].fact == "Fact two."
    assert result.facts[1].entailment == "contradicted"
    assert result.facts[2].fact == "Fact three."
    assert result.facts[2].entailment == "neutral"


@respx.mock
async def test_async_evaluate_alignment_uses_api_key(async_assistants: AsyncAssistants) -> None:
    """evaluate_alignment() sends the Api-Key header in the request."""
    from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL

    route = respx.post(f"{ASSISTANT_EVALUATION_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    await async_assistants.evaluate_alignment(
        question="Q?",
        answer="A.",
        ground_truth_answer="GT.",
    )

    request = route.calls.last.request
    assert request.headers.get("Api-Key") == "test-key"


# ---------------------------------------------------------------------------
# chat() — validation
# ---------------------------------------------------------------------------


async def test_async_chat_json_streaming_validation(async_assistants: AsyncAssistants) -> None:
    """Requesting json_response=True with stream=True raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="json_response"):
        await async_assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
            json_response=True,
        )


# ---------------------------------------------------------------------------
# chat() — defaults
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_default_model(async_assistants: AsyncAssistants) -> None:
    """chat() defaults model to 'gpt-4o' when not specified."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
                "citations": [],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatResponse

    result = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    assert isinstance(result, ChatResponse)
    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# chat() — message parsing
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_message_parsing(async_assistants: AsyncAssistants) -> None:
    """Dicts are converted to Message objects; missing role defaults to 'user'."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
                "citations": [],
            },
        )
    )

    await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[
            {"content": "No role here"},
            {"content": "Explicit role", "role": "user"},
        ],
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    msgs = request_body["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "No role here"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Explicit role"


# ---------------------------------------------------------------------------
# chat_completions() — defaults
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_completions_default_stream_false(
    async_assistants: AsyncAssistants,
) -> None:
    """chat_completions() defaults stream to False and posts to the completions endpoint."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from completions!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatCompletionResponse

    result = await async_assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    assert isinstance(result, ChatCompletionResponse)
    request_body = json.loads(completions_route.calls.last.request.content)
    assert request_body["stream"] is False
    assert request_body["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# _chat_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_sse_parsing(async_assistants: AsyncAssistants) -> None:
    """Empty SSE lines are skipped and 'data:' prefix is stripped before JSON parsing."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"type": "message_start", "model": "gpt-4o", "role": "assistant"}\n'
        b"\n"
        b'data: {"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}\n'
        b"\n"
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import StreamMessageEnd, StreamMessageStart

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamMessageStart)
    assert isinstance(chunks[1], StreamMessageEnd)


# ---------------------------------------------------------------------------
# _chat_streaming — chunk dispatch
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_chunk_dispatch(async_assistants: AsyncAssistants) -> None:
    """Correct chunk types are yielded based on the 'type' field in each SSE event."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"type": "message_start", "model": "gpt-4o", "role": "assistant"}\n'
        b'data: {"type": "content_chunk", "id": "c1",'
        b' "delta": {"content": "Hello"}}\n'
        b'data: {"type": "citation", "id": "cit1",'
        b' "citation": {"position": 5, "references": []}}\n'
        b'data: {"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import (
        StreamCitationChunk,
        StreamContentChunk,
        StreamMessageEnd,
        StreamMessageStart,
    )

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 4
    assert isinstance(chunks[0], StreamMessageStart)
    assert chunks[0].model == "gpt-4o"
    assert chunks[0].role == "assistant"
    assert isinstance(chunks[1], StreamContentChunk)
    assert chunks[1].delta.content == "Hello"
    assert isinstance(chunks[2], StreamCitationChunk)
    assert isinstance(chunks[3], StreamMessageEnd)
    assert chunks[3].usage.total_tokens == 15


# ---------------------------------------------------------------------------
# _chat_streaming — request shape
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_request_body(async_assistants: AsyncAssistants) -> None:
    """Streaming chat always includes stream=True and include_highlights in body."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"type": "message_start", "model": "gpt-4o", "role": "assistant"}\n'
        b'data: {"type": "message_end", "id": "e1",'
        b' "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}\n'
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    async for _ in async_iter:  # type: ignore[union-attr]
        pass

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["stream"] is True
    assert "include_highlights" in request_body
    assert request_body["include_highlights"] is False


# ---------------------------------------------------------------------------
# streaming bodies — unencodable values (issue #196)
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_unencodable_filter_raises_typed_error(
    async_assistants: AsyncAssistants,
) -> None:
    """The async streaming body reports the same path as the sync one."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=b""),
    )

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
        filter={"views": {"$gt": 2**64}},
    )
    with pytest.raises(PineconeTypeError) as excinfo:
        async for _ in async_iter:  # type: ignore[union-attr]
            pass
    assert excinfo.value.path == "filter.views.$gt"
    assert isinstance(excinfo.value, TypeError)
    assert not chat_route.called


@respx.mock
async def test_async_chat_completions_streaming_unencodable_filter_raises_typed_error(
    async_assistants: AsyncAssistants,
) -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=b"")
    )

    async_iter = await async_assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
        filter={"views": {"$gt": 2**64}},
    )
    with pytest.raises(PineconeTypeError) as excinfo:
        async for _ in async_iter:  # type: ignore[union-attr]
            pass
    assert excinfo.value.path == "filter.views.$gt"
    assert not completions_route.called


# ---------------------------------------------------------------------------
# _chat_completions_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_completions_streaming_sse_parsing(
    async_assistants: AsyncAssistants,
) -> None:
    """chat_completions streaming parses SSE lines as ChatCompletionStreamChunk."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"id": "cmpl1", "choices": [{"index": 0,'
        b' "delta": {"role": "assistant", "content": null},'
        b' "finish_reason": null}]}\n'
        b"\n"
        b'data: {"id": "cmpl2", "choices": [{"index": 0,'
        b' "delta": {"content": "Hello"},'
        b' "finish_reason": null}]}\n'
        b"\n"
        b'data: {"id": "cmpl3", "choices": [{"index": 0,'
        b' "delta": {}, "finish_reason": "stop"}]}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    async_iter = await async_assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 3
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[2].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# _chat_streaming — [DONE] sentinel
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_handles_done_sentinel(
    async_assistants: AsyncAssistants,
) -> None:
    """data: [DONE] terminates the stream cleanly without raising JSONDecodeError."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"type": "message_start", "model": "gpt-4o", "role": "assistant"}\n'
        b"data: [DONE]\n"
        b'data: {"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import StreamMessageStart

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamMessageStart)


# ---------------------------------------------------------------------------
# _chat_completions_streaming — [DONE] sentinel
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_completions_streaming_handles_done_sentinel(
    async_assistants: AsyncAssistants,
) -> None:
    """data: [DONE] terminates the completions stream cleanly without raising."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b'data: {"id": "cmpl1", "choices": [{"index": 0,'
        b' "delta": {"role": "assistant", "content": null},'
        b' "finish_reason": null}]}\n'
        b"data: [DONE]\n"
        b'data: {"id": "cmpl2", "choices": [{"index": 0,'
        b' "delta": {"content": "After DONE - should not appear"},'
        b' "finish_reason": null}]}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    async_iter = await async_assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatCompletionStreamChunk)
    assert chunks[0].choices[0].delta.role == "assistant"


# ---------------------------------------------------------------------------
# create() — timeout=-1 and timeout=None
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_create_timeout_none(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """timeout=None waits indefinitely until Ready."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


@respx.mock
async def test_async_create_timeout_negative_one(
    async_assistants: AsyncAssistants,
) -> None:
    """timeout=-1 returns immediately without polling."""
    create_route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = await async_assistants.create(name="test-assistant", timeout=-1)

    assert result.status == "Initializing"
    assert create_route.call_count == 1


# ---------------------------------------------------------------------------
# delete() — timeout exceeded, indefinite polling, and error propagation
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_timeout_exceeded(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Exceeding timeout during delete raises PineconeTimeoutError."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200, json=make_assistant_response(name="my-assistant", status="Terminating")
        ),
    )

    with pytest.raises(PineconeTimeoutError):
        await async_assistants.delete(name="my-assistant", timeout=10)


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """When timeout is None, polling continues until 404 with no deadline."""
    respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        side_effect=[
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(404, json={"error": "Not found"}),
        ],
    )

    result = await async_assistants.delete(name="my-assistant")

    assert result is None


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_async_delete_assistant_propagates_non_404_errors_during_poll(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Non-404 errors during polling propagate instead of being swallowed."""
    from pinecone.errors.exceptions import ServiceError

    respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(500, json={"error": "Internal server error"}),
    )

    with pytest.raises(ServiceError):
        await async_assistants.delete(name="my-assistant")


# ---------------------------------------------------------------------------
# _chat_streaming() — transport error wrapping
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_timeout_raises_pinecone_timeout_error(
    async_assistants: AsyncAssistants,
) -> None:
    """httpx.ReadTimeout during streaming is wrapped as PineconeTimeoutError."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        side_effect=httpx.ReadTimeout("test"),
    )

    with pytest.raises(PineconeTimeoutError):
        async for _ in await async_assistants.chat(  # type: ignore[union-attr]
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        ):
            pass


@respx.mock
async def test_async_chat_streaming_connect_error_raises_pinecone_connection_error(
    async_assistants: AsyncAssistants,
) -> None:
    """httpx.ConnectError during streaming is wrapped as PineconeConnectionError."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        side_effect=httpx.ConnectError("Connection refused"),
    )

    with pytest.raises(PineconeConnectionError):
        async for _ in await async_assistants.chat(  # type: ignore[union-attr]
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        ):
            pass


# ---------------------------------------------------------------------------
# _chat_streaming() — config timeout propagation
# ---------------------------------------------------------------------------


async def test_async_chat_streaming_uses_config_timeout() -> None:
    """Custom PineconeConfig.timeout is forwarded to the underlying httpx stream call."""
    from pinecone._internal.constants import ASSISTANT_API_VERSION
    from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

    config = PineconeConfig(api_key="test-key", host=BASE_URL, timeout=300.0)
    custom_assistants = AsyncAssistants(config=config)

    # Pre-populate the data plane client cache to avoid needing a describe() mock.
    data_config = PineconeConfig(api_key="test-key", host=DATA_PLANE_URL, timeout=300.0)
    data_plane_client = _AsyncHTTPClient(data_config, ASSISTANT_API_VERSION)
    custom_assistants._data_plane_clients["test-assistant"] = data_plane_client

    captured_timeout: list[float | None] = []

    @contextlib.asynccontextmanager  # type: ignore[misc]
    async def _mock_httpx_stream(
        method: str,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):  # type: ignore[misc]
        captured_timeout.append(timeout)
        mock_resp = MagicMock()
        mock_resp.is_success = True

        async def _aiter_lines():  # type: ignore[misc]
            yield (
                'data: {"type": "message_end", "id": "e1",'
                ' "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}'
            )

        mock_resp.aiter_lines.return_value = _aiter_lines()
        yield mock_resp

    # Force-initialize the underlying httpx client, then swap in the mock.
    mock_httpx_client = MagicMock()
    mock_httpx_client.stream = _mock_httpx_stream
    data_plane_client._client = mock_httpx_client

    async_iter = await custom_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    async for _ in async_iter:  # type: ignore[union-attr]
        pass

    assert len(captured_timeout) == 1
    assert captured_timeout[0] == 300.0


# ---------------------------------------------------------------------------
# _chat_streaming — SSE comment guards
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_streaming_skips_sse_comments(
    async_assistants: AsyncAssistants,
) -> None:
    """SSE comment lines (:...) and event: lines are silently skipped."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b": this is a comment\n"
        b"event: ping\n"
        b'data: {"type": "message_start", "model": "gpt-4o", "role": "assistant"}\n'
        b"retry: 3000\n"
        b'data: {"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import StreamMessageEnd, StreamMessageStart

    async_iter = await async_assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    # Only the two data: lines produce chunks; comment, event:, retry: are skipped
    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamMessageStart)
    assert isinstance(chunks[1], StreamMessageEnd)


# ---------------------------------------------------------------------------
# _chat_completions_streaming — SSE comment guards
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_chat_completions_streaming_skips_sse_comments(
    async_assistants: AsyncAssistants,
) -> None:
    """SSE comment lines (:...) and event: lines are silently skipped in completions stream."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b": keep-alive\n"
        b"event: message\n"
        b'data: {"id": "cmpl1", "choices": [{"index": 0,'
        b' "delta": {"content": "Hello"}, "finish_reason": null}]}\n'
        b"retry: 1000\n"
        b'data: {"id": "cmpl2", "choices": [{"index": 0,'
        b' "delta": {}, "finish_reason": "stop"}]}\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    async_iter = await async_assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in async_iter]  # type: ignore[union-attr]

    assert len(chunks) == 2
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in chunks)
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# X-Pinecone-Api-Version — one ASSISTANT_API_VERSION for every assistant client
# ---------------------------------------------------------------------------

_API_VERSION_HEADER = "X-Pinecone-Api-Version"

_CONTROL_PLANE_CALLS: dict[str, Callable[[AsyncAssistants], Awaitable[Any]]] = {
    "create": lambda a: a.create(name="test-assistant", timeout=-1),
    "delete": lambda a: a.delete(name="test-assistant", timeout=-1),
    "describe": lambda a: a.describe(name="test-assistant"),
    "evaluate_alignment": lambda a: a.evaluate_alignment(
        question="What is the capital of Spain?",
        answer="Barcelona.",
        ground_truth_answer="Madrid.",
    ),
    "list_page": lambda a: a.list_page(),
    "update": lambda a: a.update(name="test-assistant", instructions="Be brief."),
}


def _mock_control_plane_routes() -> None:
    assistant = make_assistant_response(status="Ready")
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": [assistant]}),
    )
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=assistant),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=assistant),
    )
    respx.patch(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=assistant),
    )
    respx.delete(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200),
    )
    respx.post(f"{ASSISTANT_EVALUATION_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )


@pytest.mark.parametrize("operation", sorted(_CONTROL_PLANE_CALLS))
@respx.mock
async def test_async_control_plane_sends_2026_07_api_version(
    operation: str, async_assistants: AsyncAssistants
) -> None:
    """Every control-plane op and evaluate_alignment send X-Pinecone-Api-Version: 2026-07."""
    _mock_control_plane_routes()

    await _CONTROL_PLANE_CALLS[operation](async_assistants)

    request = respx.calls.last.request
    assert request.headers[_API_VERSION_HEADER] == "2026-07"


_DATA_PLANE_CALLS: dict[str, Callable[[AsyncAssistants], Awaitable[Any]]] = {
    "chat": lambda a: a.chat(assistant_name="test-assistant", messages=[{"content": "Hi"}]),
    "chat_completions": lambda a: a.chat_completions(
        assistant_name="test-assistant", messages=[{"content": "Hi"}]
    ),
    "context": lambda a: a.context(assistant_name="test-assistant", query="Hi"),
}

_STREAMING_DATA_PLANE_CALLS: dict[str, Callable[[AsyncAssistants], Awaitable[Any]]] = {
    "chat": lambda a: a.chat(
        assistant_name="test-assistant", messages=[{"content": "Hi"}], stream=True
    ),
    "chat_completions": lambda a: a.chat_completions(
        assistant_name="test-assistant", messages=[{"content": "Hi"}], stream=True
    ),
}


def _mock_chat_routes(*, sse: bool) -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    chat_body = (
        b'data: {"type": "message_end", "id": "e1", "model": "gpt-4o",'
        b' "finish_reason": "tool_calls"}\n\ndata: [DONE]\n\n'
    )
    completion_body = (
        b'data: {"id": "k1", "model": "gpt-4o", "choices": [{"index": 0,'
        b' "delta": {}, "finish_reason": "tool_calls"}]}\n\ndata: [DONE]\n\n'
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=chat_body)
        if sse
        else httpx.Response(
            200,
            json={
                "id": "c1",
                "model": "gpt-4o",
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": "Hi"},
                "citations": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "context_snippet_count": 3,
            },
        ),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=completion_body)
        if sse
        else httpx.Response(
            200,
            json={
                "id": "k1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "Hi"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )


@pytest.mark.parametrize("operation", sorted(_DATA_PLANE_CALLS))
@respx.mock
async def test_async_chat_surface_sends_2026_07_api_version(
    operation: str, async_assistants: AsyncAssistants
) -> None:
    """chat, chat_completions and context all send X-Pinecone-Api-Version: 2026-07."""
    _mock_chat_routes(sse=False)

    await _DATA_PLANE_CALLS[operation](async_assistants)

    assert respx.calls.last.request.headers[_API_VERSION_HEADER] == "2026-07"


@pytest.mark.parametrize("operation", sorted(_STREAMING_DATA_PLANE_CALLS))
@respx.mock
async def test_async_streaming_chat_sends_2026_07_api_version(
    operation: str, async_assistants: AsyncAssistants
) -> None:
    """The streaming variants carry the version header on the SSE request too."""
    _mock_chat_routes(sse=True)

    stream = await _STREAMING_DATA_PLANE_CALLS[operation](async_assistants)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1, "the [DONE] sentinel must not surface as a chunk"
    assert respx.calls.last.request.headers[_API_VERSION_HEADER] == "2026-07"


@respx.mock
async def test_async_list_page_sends_limit_and_pagination_token_together(
    async_assistants: AsyncAssistants,
) -> None:
    """Paginated list still passes both limit and pagination_token on the 2026-07 client."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response()],
                "pagination": {"next": "tok-page3"},
            },
        ),
    )

    page = await async_assistants.list_page(page_size=20, pagination_token="tok-page2")

    assert page.next == "tok-page3"
    request = route.calls.last.request
    assert dict(request.url.params) == {"limit": "20", "pagination_token": "tok-page2"}
    assert request.headers[_API_VERSION_HEADER] == "2026-07"
