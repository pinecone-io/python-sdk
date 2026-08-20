"""Unit tests for Assistants namespace — create_assistant."""

from __future__ import annotations

import contextlib
import io
import json
from unittest.mock import MagicMock, patch

import httpx
import msgspec
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ASSISTANT_API_VERSION, ASSISTANT_EVALUATION_BASE_URL
from pinecone._internal.http_client import HTTPClient
from pinecone.client.assistants import (
    _CREATE_POLL_INTERVAL_SECONDS,
    _DELETE_POLL_INTERVAL_SECONDS,
    _UPLOAD_POLL_INTERVAL_SECONDS,
    Assistants,
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
from pinecone.models.assistant.list import ListAssistantsResponse, ListOperationsResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.pagination import Page, Paginator
from tests.factories import (
    make_alignment_response,
    make_assistant_file_response,
    make_assistant_response,
    make_context_response,
    make_file_operation_response,
    make_operation_list_response,
    make_operation_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ASSISTANT_API_VERSION)


@pytest.fixture
def assistants() -> Assistants:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return Assistants(config=config)


def test_init_uses_data_host_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PINECONE_PLUGIN_ASSISTANT_DATA_HOST is set, eval client targets it."""
    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_DATA_HOST",
        "https://staging-data.ke.pinecone.io",
    )
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    assistants = Assistants(config=config)
    assert assistants._eval_http._config.host == "https://staging-data.ke.pinecone.io/assistant"


def test_init_data_host_env_var_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slash on the env var is normalized."""
    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_DATA_HOST",
        "https://staging-data.ke.pinecone.io/",
    )
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    assistants = Assistants(config=config)
    assert assistants._eval_http._config.host == "https://staging-data.ke.pinecone.io/assistant"


def test_init_uses_control_host_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When config.host is empty, PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST is used."""
    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST",
        "https://api-staging.pinecone.io",
    )
    # Must NOT also set PINECONE_HOST or pass host= — empty host triggers fallback.
    monkeypatch.delenv("PINECONE_HOST", raising=False)
    config = PineconeConfig(api_key="test-key", host="")
    assistants = Assistants(config=config)
    assert assistants._http._config.host == "https://api-staging.pinecone.io/assistant"


def test_init_explicit_host_overrides_control_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config.host wins over PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST."""
    monkeypatch.setenv(
        "PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST",
        "https://api-staging.pinecone.io",
    )
    config = PineconeConfig(api_key="test-key", host="https://custom.example.com")
    assistants = Assistants(config=config)
    assert assistants._http._config.host == "https://custom.example.com/assistant"


# ---------------------------------------------------------------------------
# create() — validation
# ---------------------------------------------------------------------------


def test_create_assistant_region_validation(assistants: Assistants) -> None:
    """Invalid region raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="region") as exc_info:
        assistants.create(name="test-assistant", region="ap-southeast-1")
    message = str(exc_info.value)
    assert "ap-southeast-1" in message
    assert "'us'" in message
    assert "'eu'" in message


def test_create_assistant_region_case_sensitive(assistants: Assistants) -> None:
    """Uppercase 'US' and 'EU' are rejected — validation is case-sensitive."""
    with pytest.raises(PineconeValueError, match="region"):
        assistants.create(name="test-assistant", region="US")

    with pytest.raises(PineconeValueError, match="region"):
        assistants.create(name="test-assistant", region="EU")


# ---------------------------------------------------------------------------
# create() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_create_assistant_defaults(assistants: Assistants) -> None:
    """Default region is 'us', metadata is omitted from body, instructions is None."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant")

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
def test_create_assistant_immediate_return(assistants: Assistants) -> None:
    """timeout=-1 returns immediately without polling."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = assistants.create(name="test-assistant", timeout=-1)

    assert isinstance(result, AssistantModel)
    assert result.status == "Initializing"
    assert route.call_count == 1


@respx.mock
def test_create_assistant_with_all_params(assistants: Assistants) -> None:
    """Create with instructions, metadata, and region sends correct body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = assistants.create(
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
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_until_ready(mock_sleep: object, assistants: Assistants) -> None:
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

    result = assistants.create(name="test-assistant")

    assert result.status == "Ready"
    assert poll_route.call_count == 3


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_with_correct_interval(
    mock_sleep: object, assistants: Assistants
) -> None:
    """Polling sleeps with the correct interval between polls."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant")

    # sleep is not called after the final successful poll
    # but it may be called before the first poll depending on flow
    # The key assertion is the interval value
    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_CREATE_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# create() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError with helpful message."""
    # Simulate time progression: start=0, after first poll=6 (exceeds timeout=5)
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    with pytest.raises(PineconeTimeoutError, match="not ready") as exc_info:
        assistants.create(name="test-assistant", timeout=5)

    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_timeout_zero_polls_once(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """timeout=0 polls once, then raises if not ready."""
    # start=0, first elapsed check=0.0 (not >= 0, so one iteration), second=0.1 (>= 0)
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
        assistants.create(name="test-assistant", timeout=0)

    # At least one poll was made
    assert poll_route.call_count >= 1


# ---------------------------------------------------------------------------
# create() — status check is case-sensitive
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_status_case_sensitive(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
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
        assistants.create(name="test-assistant", timeout=5)


# ---------------------------------------------------------------------------
# create() — indefinite polling (timeout=None)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, assistants: Assistants
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

    result = assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_timeout_none(mock_sleep: object, assistants: Assistants) -> None:
    """timeout=None waits indefinitely until Ready."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


@respx.mock
def test_create_timeout_negative_one(assistants: Assistants) -> None:
    """timeout=-1 returns immediately without polling."""
    create_route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = assistants.create(name="test-assistant", timeout=-1)

    assert result.status == "Initializing"
    assert create_route.call_count == 1


# ---------------------------------------------------------------------------
# create() — terminal error states
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_failed_status(mock_sleep: object, assistants: Assistants) -> None:
    """'Failed' status raises PineconeError immediately, not a timeout."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Failed")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        assistants.create(name="test-assistant", timeout=None)

    assert "Failed" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_initialization_failed_status(
    mock_sleep: object, assistants: Assistants
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
        assistants.create(name="test-assistant", timeout=None)

    assert "InitializationFailed" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_poll_terminal_terminated(
    mock_sleep: object, assistants: Assistants
) -> None:
    """'Terminated' status raises PineconeError immediately instead of looping."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Terminated")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        assistants.create(name="test-assistant", timeout=None)

    assert "Terminated" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_poll_terminal_terminating(
    mock_sleep: object, assistants: Assistants
) -> None:
    """'Terminating' status raises PineconeError immediately instead of looping."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Terminating")),
    )

    with pytest.raises(PineconeError, match="terminal state") as exc_info:
        assistants.create(name="test-assistant", timeout=None)

    assert "Terminating" in str(exc_info.value)
    assert "pc.assistants.describe" in str(exc_info.value)


# ---------------------------------------------------------------------------
# create() — valid regions accepted
# ---------------------------------------------------------------------------


@respx.mock
def test_create_assistant_accepts_us_region(assistants: Assistants) -> None:
    """Region 'us' is accepted."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant", region="us")
    assert isinstance(result, AssistantModel)


@respx.mock
def test_create_assistant_accepts_eu_region(assistants: Assistants) -> None:
    """Region 'eu' is accepted."""
    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant", region="eu")
    assert isinstance(result, AssistantModel)


@respx.mock
def test_create_assistant_metadata_default_omitted(assistants: Assistants) -> None:
    """When metadata is not provided, the 'metadata' key is absent from the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body


@respx.mock
def test_create_assistant_metadata_explicit_dict_included(assistants: Assistants) -> None:
    """When metadata is provided, it is sent in the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant", metadata={"key": "value"}, timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metadata"] == {"key": "value"}


@respx.mock
def test_create_assistant_environment_omitted_by_default(assistants: Assistants) -> None:
    """When environment is not provided, the 'environment' key is absent from the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "environment" not in body


@respx.mock
def test_create_assistant_environment_included_when_provided(assistants: Assistants) -> None:
    """When environment is provided, it is sent in the request body."""
    route = respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant", environment="prod-us", timeout=-1)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["environment"] == "prod-us"


@respx.mock
def test_create_assistant_environment_403_propagates(assistants: Assistants) -> None:
    """A 403 from the backend when environment is set propagates as ApiError."""
    from pinecone.errors.exceptions import ApiError

    respx.post(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            403, json={"error": {"code": "FORBIDDEN", "message": "Not authorized"}}
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        assistants.create(name="test-assistant", environment="prod-us", timeout=-1)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# describe() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_assistant(assistants: Assistants) -> None:
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

    result = assistants.describe(name="my-assistant")

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
def test_describe_assistant_not_found(assistants: Assistants) -> None:
    """describe() lets 404 errors propagate from the HTTP client."""
    respx.get(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.describe(name="nonexistent")


@respx.mock
def test_describe_assistant_minimal_response(assistants: Assistants) -> None:
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

    result = assistants.describe(name="minimal")

    assert result.name == "minimal"
    assert result.metadata is None
    assert result.instructions is None
    assert result.host is None


# ---------------------------------------------------------------------------
# list() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
def test_list_assistants_empty(assistants: Assistants) -> None:
    """list() returns a Paginator that yields nothing when no assistants exist."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    result = assistants.list()

    assert isinstance(result, Paginator)
    assert result.to_list() == []


@respx.mock
def test_list_assistants_single_page(assistants: Assistants) -> None:
    """list() returns a Paginator over all assistants from a single-page response."""
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

    result = assistants.list().to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"
    assert all(isinstance(a, AssistantModel) for a in result)


@respx.mock
def test_list_assistants_multi_page(assistants: Assistants) -> None:
    """list() auto-paginates through multiple pages via Paginator."""
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

    result = assistants.list().to_list()

    assert len(result) == 3
    assert [a.name for a in result] == ["a1", "a2", "a3"]


@respx.mock
def test_list_assistants_to_list(assistants: Assistants) -> None:
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

    items = assistants.list().to_list()

    assert len(items) == 2
    assert all(isinstance(a, AssistantModel) for a in items)


@respx.mock
def test_list_assistants_iteration(assistants: Assistants) -> None:
    """list() supports direct for-loop iteration."""
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

    names = [a.name for a in assistants.list()]

    assert names == ["a1", "a2"]


@respx.mock
def test_list_assistants_with_limit(assistants: Assistants) -> None:
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

    result = assistants.list(limit=2).to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"


@respx.mock
def test_list_assistants_pages(assistants: Assistants) -> None:
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

    pages = list(assistants.list().pages())

    assert len(pages) == 2
    assert pages[0].has_more is True
    assert pages[0].items[0].name == "a1"
    assert pages[1].has_more is False
    assert pages[1].items[0].name == "a2"
    for page in pages:
        assert isinstance(page, Page)


@respx.mock
def test_list_assistants_with_pagination_token(assistants: Assistants) -> None:
    """list(pagination_token=...) starts from the given token."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={"assistants": [make_assistant_response(name="a2")]},
        ),
    )

    result = assistants.list(pagination_token="tok-page2").to_list()

    assert len(result) == 1
    assert result[0].name == "a2"
    request = route.calls.last.request
    assert "pagination_token=tok-page2" in str(request.url)


# ---------------------------------------------------------------------------
# list_page() — single page with pagination control
# ---------------------------------------------------------------------------


@respx.mock
def test_list_assistants_page(assistants: Assistants) -> None:
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

    result = assistants.list_page()

    assert isinstance(result, ListAssistantsResponse)
    assert len(result.assistants) == 1
    assert result.assistants[0].name == "a1"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
def test_list_assistants_page_last_page(assistants: Assistants) -> None:
    """list_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
            },
        ),
    )

    result = assistants.list_page()

    assert result.next is None


@respx.mock
def test_list_assistants_page_with_page_size(assistants: Assistants) -> None:
    """list_page() sends limit query param when provided."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page(page_size=5)

    request = route.calls.last.request
    assert "limit=5" in str(request.url)


@respx.mock
def test_list_assistants_page_with_pagination_token(assistants: Assistants) -> None:
    """list_page() sends pagination_token query param when provided."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page(pagination_token="abc123")

    request = route.calls.last.request
    assert "pagination_token=abc123" in str(request.url)


# ---------------------------------------------------------------------------
# update() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_update_assistant_instructions(assistants: Assistants) -> None:
    """update() sends PATCH /assistants/{name} with instructions and returns AssistantModel."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="Updated instructions.",
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", instructions="Updated instructions.")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.instructions == "Updated instructions."
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "Updated instructions."}


@respx.mock
def test_update_assistant_metadata(assistants: Assistants) -> None:
    """update() sends PATCH with metadata and returns AssistantModel."""
    new_metadata = {"team": "ml", "version": "2"}
    updated_response = make_assistant_response(
        name="my-assistant",
        metadata=new_metadata,
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", metadata=new_metadata)

    assert isinstance(result, AssistantModel)
    assert result.metadata == new_metadata
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"metadata": new_metadata}


@respx.mock
def test_update_assistant_both_fields(assistants: Assistants) -> None:
    """update() sends both instructions and metadata when provided."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(
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
def test_update_assistant_clear_instructions(assistants: Assistants) -> None:
    """update() can clear instructions by setting them to empty string."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="",
    )
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", instructions="")

    assert result.instructions == ""

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": ""}


@respx.mock
def test_update_assistant_omits_none_fields(assistants: Assistants) -> None:
    """update() only includes provided fields in the request body."""
    updated_response = make_assistant_response(name="my-assistant")
    route = respx.patch(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    assistants.update(name="my-assistant", instructions="Only this.")

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body
    assert body == {"instructions": "Only this."}


@respx.mock
def test_update_assistant_not_found(assistants: Assistants) -> None:
    """update() lets 404 errors propagate from the HTTP client."""
    respx.patch(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.update(name="nonexistent", instructions="test")


# ---------------------------------------------------------------------------
# list_page() — omission
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# delete() — polls until gone
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant(mock_sleep: object, assistants: Assistants) -> None:
    """delete() sends DELETE then polls describe until 404 confirms deletion."""
    delete_route = respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    result = assistants.delete(name="my-assistant")

    assert result is None
    assert delete_route.call_count == 1

    request = delete_route.calls.last.request
    assert request.method == "DELETE"
    assert str(request.url) == f"{BASE_URL}/assistant/assistants/my-assistant"


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_polls_until_gone(mock_sleep: object, assistants: Assistants) -> None:
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

    result = assistants.delete(name="my-assistant")

    assert result is None
    assert describe_route.call_count == 3

    from unittest.mock import call

    from pinecone.client.assistants import _DELETE_POLL_INTERVAL_SECONDS

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
def test_delete_assistant_timeout_minus_one_skips_polling(assistants: Assistants) -> None:
    """delete(timeout=-1) returns immediately without polling."""
    delete_route = respx.delete(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )

    result = assistants.delete(name="my-assistant", timeout=-1)

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
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
        assistants.delete(name="my-assistant", timeout=10)


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_delete_timeout_exceeded(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
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
        assistants.delete(name="my-assistant", timeout=10)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, assistants: Assistants
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

    result = assistants.delete(name="my-assistant")

    assert result is None


@respx.mock
def test_delete_assistant_not_found(assistants: Assistants) -> None:
    """delete() lets 404 errors propagate from the initial DELETE request."""
    respx.delete(f"{BASE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.delete(name="nonexistent")


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_propagates_non_404_errors_during_poll(
    mock_sleep: object, assistants: Assistants
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
        assistants.delete(name="my-assistant")


# ---------------------------------------------------------------------------
# list_page() — omission
# ---------------------------------------------------------------------------


@respx.mock
def test_list_assistants_page_omits_none_params(assistants: Assistants) -> None:
    """list_page() does not send params that are None."""
    route = respx.get(f"{BASE_URL}/assistant/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page()

    request = route.calls.last.request
    assert "limit" not in str(request.url)
    assert "pagination_token" not in str(request.url)


CONTROL_PLANE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# Pinecone.assistant() convenience method
# ---------------------------------------------------------------------------


@respx.mock
def test_pinecone_assistant_convenience_method() -> None:
    """pc.assistant(name=...) delegates to GET /assistants/{name} and returns AssistantModel."""
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

    pc = Pinecone(api_key="test-key")
    result = pc.assistant(name="my-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.instructions == "Be helpful."
    assert result.metadata == {"team": "ml"}
    assert result.host == "https://my-assistant-abc.svc.pinecone.io"
    assert route.call_count == 1


@respx.mock
def test_pinecone_assistant_not_found() -> None:
    """pc.assistant(name=...) raises NotFoundError when assistant does not exist."""
    respx.get(f"{CONTROL_PLANE_URL}/assistant/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    pc = Pinecone(api_key="test-key")
    with pytest.raises(NotFoundError):
        pc.assistant(name="nonexistent")


# ---------------------------------------------------------------------------
# upload_file() — validation
# ---------------------------------------------------------------------------

DATA_PLANE_HOST = "test-assistant-abc123.svc.pinecone.io"
DATA_PLANE_URL = f"https://{DATA_PLANE_HOST}/assistant"


def test_upload_file_path_not_found(assistants: Assistants) -> None:
    """Uploading from a nonexistent local path raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="File not found"):
        assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/nonexistent/path/document.pdf",
        )


def test_upload_file_neither_path_nor_stream(assistants: Assistants) -> None:
    """Providing neither file_path nor file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        assistants.upload_file(assistant_name="test-assistant")


def test_upload_file_both_path_and_stream(assistants: Assistants) -> None:
    """Providing both file_path and file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/some/path.pdf",
            file_stream=io.BytesIO(b"data"),
        )


# ---------------------------------------------------------------------------
# upload_file() — multimodal serialization
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_multimodal_serialization(mock_sleep: object, assistants: Assistants) -> None:
    """multimodal bool is converted to lowercase string in query params."""
    # Mock describe call for data-plane host resolution
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )

    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response(status="Completed")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"file content")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="doc.pdf",
        multimodal=True,
    )

    request = upload_route.calls.last.request
    assert "multimodal=true" in str(request.url)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_multimodal_false_serialization(
    mock_sleep: object, assistants: Assistants
) -> None:
    """multimodal=False is sent as 'false' string."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response(status="Completed")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"file content")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="doc.pdf",
        multimodal=False,
    )

    request = upload_route.calls.last.request
    assert "multimodal=false" in str(request.url)


# ---------------------------------------------------------------------------
# upload_file() — success from stream
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_from_stream(mock_sleep: object, assistants: Assistants) -> None:
    """upload_file resolves the file id from the 202 operation and polls it."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        side_effect=[
            httpx.Response(
                200, json=make_file_operation_response(status="Processing", percent_complete=40)
            ),
            httpx.Response(
                200,
                json=make_file_operation_response(
                    status="Completed",
                    percent_complete=100,
                    completed_on="2026-07-01T12:00:30Z",
                    ingestion_units=3.5,
                ),
            ),
        ]
    )
    describe_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"test file content")
    result = assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="report.pdf",
    )

    assert isinstance(result, AssistantFileModel)
    assert result.status == "Available"
    assert result.id == "file-abc123"
    assert upload_route.call_count == 1
    assert poll_route.call_count == 2
    assert describe_route.call_count == 1


# ---------------------------------------------------------------------------
# upload_file() — success with metadata and file_id
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_with_metadata_and_file_id(mock_sleep: object, assistants: Assistants) -> None:
    """upload_file(file_id=...) upserts with PUT and polls the operation.

    The 202 and poll bodies here are the *minimal* ``2026-04`` operation shape,
    so this doubles as the backward-compatibility case: a server that omits
    ``operation_type``/``file_id`` and reports the older ``"Succeeded"``
    terminal status must still be understood.
    """
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upsert_route = respx.put(f"{DATA_PLANE_URL}/files/test-assistant/custom-file-id").mock(
        return_value=httpx.Response(202, json=make_operation_response(status="Succeeded")),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_operation_response(status="Succeeded")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/custom-file-id").mock(
        return_value=httpx.Response(
            200, json=make_assistant_file_response(id="custom-file-id", status="Available")
        ),
    )

    stream = io.BytesIO(b"content")
    result = assistants.upload_file(
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
# upload_file() — polling
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_polls_with_correct_interval(
    mock_sleep: object, assistants: Assistants
) -> None:
    """upload_file polls every 5 seconds while processing."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_file_operation_response(status="Processing")),
            httpx.Response(200, json=make_file_operation_response(status="Processing")),
            httpx.Response(200, json=make_file_operation_response(status="Completed")),
        ]
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"data")
    assistants.upload_file(
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
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_processing_failed(mock_sleep: object, assistants: Assistants) -> None:
    """A failed upload operation raises with the server's error_message verbatim.

    ``2026-07`` dropped ``AssistantFileModel.error_message``; the failure
    reason now comes from the operation record, so the message an agent reads
    must name the file, the operation, and what the server actually said.
    """
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_file_operation_response(
                status="Failed",
                completed_on="2026-07-01T12:00:05Z",
                error_message="Uploaded file can only currently be either a pdf or txt file",
            ),
        ),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeError) as exc_info:
        assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
        )

    message = str(exc_info.value)
    assert "Uploaded file can only currently be either a pdf or txt file" in message
    assert "op-abc123" in message
    assert "file-abc123" in message


# ---------------------------------------------------------------------------
# upload_file() — upsert operation failure (error_message field name)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_upsert_error_message(mock_sleep: object, assistants: Assistants) -> None:
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
        assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
            file_id="custom-file-id",
        )


# ---------------------------------------------------------------------------
# upload_file() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Timeout raises PineconeTimeoutError with operation ID in message."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            200, json=make_file_operation_response(status="Processing", percent_complete=42)
        ),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeTimeoutError, match="file-abc123") as exc_info:
        assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
            timeout=10,
        )

    message = str(exc_info.value)
    assert "operation_id='op-abc123'" in message
    assert "percent_complete=42" in message


@respx.mock
def test_upload_timeout_negative_one(assistants: Assistants) -> None:
    """upload_file(timeout=-1) calls describe_file once without polling."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    operation_route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response()),
    )
    describe_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )

    stream = io.BytesIO(b"data")
    result = assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        timeout=-1,
    )

    assert upload_route.call_count == 1
    assert describe_route.call_count == 1
    assert operation_route.call_count == 0
    assert isinstance(result, AssistantFileModel)
    assert result.status == "Processing"


# ---------------------------------------------------------------------------
# upload_file() — data-plane client caching
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_caches_data_plane_client(mock_sleep: object, assistants: Assistants) -> None:
    """Second upload to same assistant reuses cached data-plane client."""
    describe_route = respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response(status="Completed")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    # First upload — triggers describe
    stream1 = io.BytesIO(b"data1")
    assistants.upload_file(assistant_name="test-assistant", file_stream=stream1)

    # Second upload — should reuse cached client
    stream2 = io.BytesIO(b"data2")
    assistants.upload_file(assistant_name="test-assistant", file_stream=stream2)

    # Describe should only be called once (for the first upload)
    assert describe_route.call_count == 1


# ---------------------------------------------------------------------------
# describe_file() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_file_success(assistants: Assistants) -> None:
    """describe_file() sends GET /files/{name}/{id} and returns AssistantFileModel."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    result = assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    assert isinstance(result, AssistantFileModel)
    assert result.id == "file-abc123"
    assert result.name == "test-file.pdf"
    assert route.call_count == 1


@respx.mock
def test_describe_file_without_url(assistants: Assistants) -> None:
    """describe_file() does not send include_url param by default."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    request = route.calls.last.request
    assert "include_url" not in str(request.url)


@respx.mock
def test_describe_file_with_url(assistants: Assistants) -> None:
    """describe_file(include_url=True) sends include_url=true query param."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(signed_url="https://storage.example.com/file-abc123"),
        ),
    )

    result = assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123", include_url=True
    )

    request = route.calls.last.request
    assert "include_url=true" in str(request.url)
    assert result.signed_url == "https://storage.example.com/file-abc123"


@respx.mock
def test_describe_file_not_found(assistants: Assistants) -> None:
    """describe_file() raises NotFoundError when file does not exist."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.describe_file(assistant_name="test-assistant", file_id="nonexistent")


# ---------------------------------------------------------------------------
# list_files_page() — single page
# ---------------------------------------------------------------------------


@respx.mock
def test_list_files_page_success(assistants: Assistants) -> None:
    """list_files_page() returns ListFilesResponse with files and next token."""
    from pinecone.models.assistant.list import ListFilesResponse

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files_page(assistant_name="test-assistant")

    assert isinstance(result, ListFilesResponse)
    assert len(result.files) == 1
    assert result.files[0].id == "file-abc123"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
def test_list_files_page_last_page(assistants: Assistants) -> None:
    """list_files_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": [make_assistant_file_response()]}),
    )

    result = assistants.list_files_page(assistant_name="test-assistant")

    assert result.next is None


@respx.mock
def test_list_files_page_no_page_size_param(assistants: Assistants) -> None:
    """list_files_page() does not send a limit query param when page_size is omitted."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "limit" not in str(request.url)


@respx.mock
def test_list_files_page_sends_page_size(assistants: Assistants) -> None:
    """list_files_page() sends limit query param when page_size is provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant", page_size=20)

    request = route.calls.last.request
    assert "limit=20" in str(request.url)


@respx.mock
def test_list_files_page_with_pagination_token(assistants: Assistants) -> None:
    """list_files_page() sends pagination_token query param when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant", pagination_token="tok123")

    request = route.calls.last.request
    assert "pagination_token=tok123" in str(request.url)


@respx.mock
def test_list_files_page_with_filter(assistants: Assistants) -> None:
    """list_files_page() serializes filter dict to JSON string query param."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str


@respx.mock
def test_list_files_page_omits_none_params(assistants: Assistants) -> None:
    """list_files_page() does not send params that are None."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "limit" not in str(request.url)
    assert "pagination_token" not in str(request.url)
    assert "filter" not in str(request.url)


# ---------------------------------------------------------------------------
# list_files() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
def test_list_files_empty(assistants: Assistants) -> None:
    """list_files() returns a Paginator that yields nothing when no files exist."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    result = assistants.list_files(assistant_name="test-assistant")

    assert isinstance(result, Paginator)
    assert result.to_list() == []


@respx.mock
def test_list_files_single_page(assistants: Assistants) -> None:
    """list_files() returns a Paginator over files from a single-page response."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files(assistant_name="test-assistant").to_list()

    assert len(result) == 2
    assert all(isinstance(f, AssistantFileModel) for f in result)


@respx.mock
def test_list_files_multi_page(assistants: Assistants) -> None:
    """list_files() auto-paginates through multiple pages via Paginator."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files(assistant_name="test-assistant").to_list()

    assert len(result) == 3
    assert [f.id for f in result] == ["f1", "f2", "f3"]


@respx.mock
def test_list_files_to_list(assistants: Assistants) -> None:
    """list_files().to_list() collects all files into a list."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    items = assistants.list_files(assistant_name="test-assistant").to_list()

    assert len(items) == 2
    assert all(isinstance(f, AssistantFileModel) for f in items)


@respx.mock
def test_list_files_iteration(assistants: Assistants) -> None:
    """list_files() supports direct for-loop iteration."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    names = [f.name for f in assistants.list_files(assistant_name="test-assistant")]

    assert names == ["file1.pdf", "file2.pdf"]


@respx.mock
def test_list_files_with_limit(assistants: Assistants) -> None:
    """list_files(limit=N) yields at most N items across all pages."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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
                "pagination": {"next": "more"},
            },
        ),
    )

    result = assistants.list_files(assistant_name="test-assistant", limit=2).to_list()

    assert len(result) == 2
    assert result[0].id == "f1"
    assert result[1].id == "f2"


@respx.mock
def test_list_files_pages(assistants: Assistants) -> None:
    """list_files().pages() yields Page objects with items and has_more."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f1", name="file1.pdf")],
                    "pagination": {"next": "token-next"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f2", name="file2.pdf")],
                },
            ),
        ]
    )

    pages = list(assistants.list_files(assistant_name="test-assistant").pages())

    assert len(pages) == 2
    assert pages[0].has_more is True
    assert pages[0].items[0].id == "f1"
    assert pages[1].has_more is False
    assert pages[1].items[0].id == "f2"
    for page in pages:
        assert isinstance(page, Page)


@respx.mock
def test_list_files_with_pagination_token(assistants: Assistants) -> None:
    """list_files(pagination_token=...) starts from the given token."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={"files": [make_assistant_file_response(id="f2", name="file2.pdf")]},
        ),
    )

    result = assistants.list_files(
        assistant_name="test-assistant", pagination_token="tok-page2"
    ).to_list()

    assert len(result) == 1
    assert result[0].id == "f2"
    request = route.calls.last.request
    assert "pagination_token=tok-page2" in str(request.url)


@respx.mock
def test_list_files_propagates_filter_through_pages(assistants: Assistants) -> None:
    """list_files() sends the same filter on every paginated request."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f1", name="file1.pdf")],
                    "pagination": {"next": "token-p2"},
                },
            ),
            httpx.Response(
                200,
                json={"files": [make_assistant_file_response(id="f2", name="file2.pdf")]},
            ),
        ]
    )

    assistants.list_files(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    ).to_list()

    assert route.call_count == 2
    for call_obj in route.calls:
        assert "filter=" in str(call_obj.request.url)
        assert "genre" in str(call_obj.request.url)


# ---------------------------------------------------------------------------
# delete_file() — success
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_204_returns_without_polling(
    mock_sleep: object, assistants: Assistants
) -> None:
    """A 204 means the file was already gone: nothing to poll, no error."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(204)
    )
    operation_route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response())
    )

    result = assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")

    assert result is None
    assert delete_route.call_count == 1
    assert operation_route.call_count == 0
    assert mock_sleep.call_count == 0  # type: ignore[union-attr]


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_202_polls_the_operation(mock_sleep: object, assistants: Assistants) -> None:
    """A 202 means deletion is pending: poll the returned operation every 5s."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            202,
            json=make_file_operation_response(operation_type="delete_file"),
            headers={"Location": "/assistant/operations/test-assistant/op-abc123"},
        )
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_file_operation_response(operation_type="delete_file")),
            httpx.Response(
                200,
                json=make_file_operation_response(
                    operation_type="delete_file", status="Completed", percent_complete=100
                ),
            ),
        ]
    )

    result = assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")

    assert result is None
    assert delete_route.call_count == 1
    assert poll_route.call_count == 2

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
def test_delete_file_timeout_minus_one_skips_polling(assistants: Assistants) -> None:
    """delete_file(timeout=-1) returns as soon as the 202 is accepted."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            202, json=make_file_operation_response(operation_type="delete_file")
        )
    )
    operation_route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response())
    )

    result = assistants.delete_file(
        assistant_name="test-assistant", file_id="file-abc123", timeout=-1
    )

    assert result is None
    assert delete_route.call_count == 1
    assert operation_route.call_count == 0


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError naming the operation."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            202, json=make_file_operation_response(operation_type="delete_file")
        )
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            200, json=make_file_operation_response(operation_type="delete_file")
        ),
    )

    with pytest.raises(PineconeTimeoutError, match="did not finish within 10") as exc_info:
        assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123", timeout=10)

    assert "operation_id='op-abc123'" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_server_error_raises(mock_sleep: object, assistants: Assistants) -> None:
    """A failed deletion operation surfaces the server's error_message verbatim."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            202, json=make_file_operation_response(operation_type="delete_file")
        )
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_file_operation_response(
                operation_type="delete_file",
                status="Failed",
                error_message="File is referenced by an in-flight ingestion",
            ),
        ),
    )

    with pytest.raises(
        PineconeError, match="File is referenced by an in-flight ingestion"
    ) as exc_info:
        assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")

    assert "Deletion of file 'file-abc123' failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# chat() — validation
# ---------------------------------------------------------------------------


def test_chat_json_streaming_validation(assistants: Assistants) -> None:
    """Requesting json_response=True with stream=True raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="json_response"):
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
            json_response=True,
        )


# ---------------------------------------------------------------------------
# chat() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_default_model(assistants: Assistants) -> None:
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

    result = assistants.chat(
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
def test_chat_message_parsing(assistants: Assistants) -> None:
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

    assistants.chat(
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
# chat() — optional body parameters
# ---------------------------------------------------------------------------

_CHAT_RESPONSE = {
    "id": "chat-abc123",
    "model": "gpt-4o",
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop",
    "citations": [],
}


@respx.mock
def test_chat_forwards_temperature(assistants: Assistants) -> None:
    """chat() includes 'temperature' in the request body when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        temperature=0.7,
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["temperature"] == 0.7


@respx.mock
def test_chat_forwards_filter(assistants: Assistants) -> None:
    """chat() includes 'filter' in the request body when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        filter={"doc_type": "pdf"},
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["filter"] == {"doc_type": "pdf"}


@respx.mock
def test_chat_forwards_json_response(assistants: Assistants) -> None:
    """chat() includes 'json_response': true in the request body when json_response=True."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        json_response=True,
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["json_response"] is True


@respx.mock
def test_chat_omits_defaults(assistants: Assistants) -> None:
    """chat() omits optional keys from the body when using default values."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert "temperature" not in request_body
    assert "filter" not in request_body
    assert "json_response" not in request_body
    assert "context_options" not in request_body


@respx.mock
def test_chat_forwards_context_options_dict(assistants: Assistants) -> None:
    """chat() passes a dict context_options directly into the request body."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        context_options={"top_k": 5},
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["context_options"] == {"top_k": 5}


@respx.mock
def test_chat_forwards_context_options_struct(assistants: Assistants) -> None:
    """chat() serializes a ContextOptions struct to dict, omitting None-valued fields."""
    from pinecone.models.assistant.options import ContextOptions

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        context_options=ContextOptions(top_k=5),
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["context_options"] == {"top_k": 5}
    assert "snippet_size" not in request_body["context_options"]
    assert "multimodal" not in request_body["context_options"]
    assert "include_binary_content" not in request_body["context_options"]


# ---------------------------------------------------------------------------
# chat_completions() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_default_stream_false(assistants: Assistants) -> None:
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

    result = assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    assert isinstance(result, ChatCompletionResponse)
    request_body = json.loads(completions_route.calls.last.request.content)
    assert request_body["stream"] is False
    assert request_body["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# chat_completions() — no model validation
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_no_model_validation(assistants: Assistants) -> None:
    """chat_completions() accepts any model string without client-side validation."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-xyz789",
                "model": "some-unknown-model-v99",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatCompletionResponse

    result = assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        model="some-unknown-model-v99",
    )

    assert isinstance(result, ChatCompletionResponse)
    request_body = json.loads(completions_route.calls.last.request.content)
    assert request_body["model"] == "some-unknown-model-v99"


# ---------------------------------------------------------------------------
# _chat_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_sse_parsing(assistants: Assistants) -> None:
    """Empty SSE lines are skipped and 'data:' prefix is stripped before JSON parsing."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    # SSE payload: empty lines between events, data: prefix on each event
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

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamMessageStart)
    assert isinstance(chunks[1], StreamMessageEnd)


# ---------------------------------------------------------------------------
# _chat_streaming — chunk dispatch
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_chunk_dispatch(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

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
def test_chat_streaming_request_body(assistants: Assistants) -> None:
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

    list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["stream"] is True
    assert "include_highlights" in request_body
    assert request_body["include_highlights"] is False


# ---------------------------------------------------------------------------
# streaming bodies — unencodable values (issue #196)
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_unencodable_filter_raises_typed_error(assistants: Assistants) -> None:
    """A streaming body is encoded at the call site rather than by the HTTP
    client, so it needs its own route to the typed error."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=b""),
    )

    stream = assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
        filter={"views": {"$gt": 2**64}},
    )
    with pytest.raises(PineconeTypeError) as excinfo:
        list(stream)  # type: ignore[call-overload]
    assert excinfo.value.path == "filter.views.$gt"
    assert isinstance(excinfo.value, TypeError)
    assert not chat_route.called


@respx.mock
def test_chat_completions_streaming_unencodable_filter_raises_typed_error(
    assistants: Assistants,
) -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=b"")
    )

    stream = assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        stream=True,
        filter={"views": {"$gt": 2**64}},
    )
    with pytest.raises(PineconeTypeError) as excinfo:
        list(stream)  # type: ignore[call-overload]
    assert excinfo.value.path == "filter.views.$gt"
    assert not completions_route.called


# ---------------------------------------------------------------------------
# _chat_completions_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_streaming_sse_parsing(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat_completions(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 3
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[2].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# _chat_streaming — [DONE] sentinel and SSE comment guards
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_handles_done_sentinel(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    # Stream stops at [DONE]; the message_end line after it is never yielded
    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamMessageStart)


@respx.mock
def test_chat_streaming_skips_sse_comments(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    # Only the two data: lines produce chunks; comment, event:, retry: are skipped
    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamMessageStart)
    assert isinstance(chunks[1], StreamMessageEnd)


# ---------------------------------------------------------------------------
# _chat_completions_streaming — [DONE] sentinel and SSE comment guards
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_streaming_handles_done_sentinel(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat_completions(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatCompletionStreamChunk)
    assert chunks[0].choices[0].delta.role == "assistant"


@respx.mock
def test_chat_completions_streaming_skips_sse_comments(assistants: Assistants) -> None:
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

    chunks = list(
        assistants.chat_completions(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 2
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in chunks)
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# context() — validation
# ---------------------------------------------------------------------------


def test_context_both_query_and_messages(assistants: Assistants) -> None:
    """Providing both query and messages raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="not both"):
        assistants.context(
            assistant_name="test-assistant",
            query="What is Pinecone?",
            messages=[{"content": "Hello"}],
        )


def test_context_neither_query_nor_messages(assistants: Assistants) -> None:
    """Providing neither query nor messages raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        assistants.context(assistant_name="test-assistant")


def test_context_empty_string_query(assistants: Assistants) -> None:
    """Empty string query is treated as not provided — raises if messages also absent."""
    with pytest.raises(PineconeValueError):
        assistants.context(assistant_name="test-assistant", query="")


def test_context_empty_list_messages(assistants: Assistants) -> None:
    """Empty list messages is treated as not provided — raises if query also absent."""
    with pytest.raises(PineconeValueError):
        assistants.context(assistant_name="test-assistant", messages=[])


def test_context_top_k_negative_raises(assistants: Assistants) -> None:
    """Negative top_k raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="top_k"):
        assistants.context(assistant_name="test-assistant", query="test", top_k=-1)


def test_context_snippet_size_negative_raises(assistants: Assistants) -> None:
    """Negative snippet_size raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="snippet_size"):
        assistants.context(assistant_name="test-assistant", query="test", snippet_size=-1)


# ---------------------------------------------------------------------------
# context() — success with query
# ---------------------------------------------------------------------------


@respx.mock
def test_context_with_query(assistants: Assistants) -> None:
    """context() with query POSTs to /chat/{name}/context and returns ContextResponse."""
    from pinecone.models.assistant.context import ContextResponse

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    result = assistants.context(
        assistant_name="test-assistant",
        query="What is Pinecone?",
    )

    assert isinstance(result, ContextResponse)
    assert len(result.snippets) == 1

    request_body = json.loads(context_route.calls.last.request.content)
    assert request_body["query"] == "What is Pinecone?"
    assert "messages" not in request_body


@respx.mock
def test_context_with_messages(assistants: Assistants) -> None:
    """context() with messages parses and sends them; does not send query."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    assistants.context(
        assistant_name="test-assistant",
        messages=[{"content": "Tell me about vector databases."}],
    )

    request_body = json.loads(context_route.calls.last.request.content)
    assert "query" not in request_body
    assert request_body["messages"] == [
        {"role": "user", "content": "Tell me about vector databases."}
    ]


@respx.mock
def test_context_optional_params_included_when_provided(assistants: Assistants) -> None:
    """Optional parameters are sent in the request body when provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    assistants.context(
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
def test_context_optional_params_omitted_when_absent(assistants: Assistants) -> None:
    """Optional parameters are not included in the request body when not provided."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    context_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/context").mock(
        return_value=httpx.Response(200, json=make_context_response()),
    )

    assistants.context(
        assistant_name="test-assistant",
        query="What is Pinecone?",
    )

    request_body = json.loads(context_route.calls.last.request.content)
    assert "filter" not in request_body
    assert "top_k" not in request_body
    assert "snippet_size" not in request_body
    assert "multimodal" not in request_body


# ---------------------------------------------------------------------------
# Optional field decoding tests (claims: quality)
# ---------------------------------------------------------------------------


def test_context_response_decodes_without_id() -> None:
    """ContextResponse decodes correctly when id is absent; id is None."""
    from pinecone.models.assistant.context import ContextResponse

    payload = json.dumps(
        {
            "snippets": [
                {
                    "type": "text",
                    "content": "Pinecone is a vector database.",
                    "score": 0.95,
                    "reference": {
                        "file": make_context_response()["snippets"][0]["reference"]["file"]
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
    ).encode()

    result = msgspec.json.decode(payload, type=ContextResponse)

    assert result.id is None
    assert len(result.snippets) == 1
    assert result.usage.total_tokens == 10


def test_context_image_block_image_data_populated_from_json_image_key() -> None:
    """ContextImageBlock.image_data is populated from the API's 'image' JSON key (IT-0019)."""
    from pinecone.models.assistant.context import ContextResponse

    file_fixture = make_context_response()["snippets"][0]["reference"]["file"]
    payload = json.dumps(
        {
            "snippets": [
                {
                    "type": "multimodal",
                    "content": [
                        {
                            "type": "image",
                            "caption": "A chart showing sales data",
                            "image": {
                                "type": "base64",
                                "mime_type": "image/jpeg",
                                "data": "abc123==",
                            },
                        }
                    ],
                    "score": 0.88,
                    "reference": {"file": file_fixture},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            "id": "ctx-img-test",
        }
    ).encode()

    result = msgspec.json.decode(payload, type=ContextResponse)

    from pinecone.models.assistant.context import ContextImageBlock, MultimodalSnippet

    assert len(result.snippets) == 1
    snippet = result.snippets[0]
    assert isinstance(snippet, MultimodalSnippet)
    assert len(snippet.content) == 1
    block = snippet.content[0]
    assert isinstance(block, ContextImageBlock)
    assert block.caption == "A chart showing sales data"
    assert block.image_data is not None
    assert block.image_data.type == "base64"
    assert block.image_data.mime_type == "image/jpeg"
    assert block.image_data.data == "abc123=="


def test_context_image_block_image_data_none_when_image_absent() -> None:
    """ContextImageBlock.image_data is None when 'image' key is absent (binary content excluded)."""
    from pinecone.models.assistant.context import ContextResponse

    file_fixture = make_context_response()["snippets"][0]["reference"]["file"]
    payload = json.dumps(
        {
            "snippets": [
                {
                    "type": "multimodal",
                    "content": [
                        {
                            "type": "image",
                            "caption": "A chart without binary content",
                        }
                    ],
                    "score": 0.75,
                    "reference": {"file": file_fixture},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
    ).encode()

    result = msgspec.json.decode(payload, type=ContextResponse)

    from pinecone.models.assistant.context import ContextImageBlock, MultimodalSnippet

    snippet = result.snippets[0]
    assert isinstance(snippet, MultimodalSnippet)
    block = snippet.content[0]
    assert isinstance(block, ContextImageBlock)
    assert block.caption == "A chart without binary content"
    assert block.image_data is None


def test_assistant_model_decodes_without_timestamps() -> None:
    """AssistantModel decodes correctly when created_at and updated_at are absent; both are None."""
    payload = json.dumps({"name": "my-assistant", "status": "Ready"}).encode()

    result = msgspec.json.decode(payload, type=AssistantModel)

    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.created_at is None
    assert result.updated_at is None
    assert result.region is None


def test_assistant_model_decodes_region_and_timestamps() -> None:
    """A 2026-07 assistant payload round-trips region and both timestamps."""
    payload = {
        "name": "my-assistant",
        "status": "Ready",
        "instructions": "Be helpful.",
        "metadata": {"team": "Operations"},
        "host": "https://prod-1-data.ke.pinecone.io",
        "region": "eu",
        "created_at": "2026-07-01T12:30:00Z",
        "updated_at": "2026-07-01T12:45:00Z",
    }

    result = msgspec.json.decode(json.dumps(payload).encode(), type=AssistantModel)

    assert result.region == "eu"
    assert result.created_at == "2026-07-01T12:30:00Z"
    assert result.updated_at == "2026-07-01T12:45:00Z"
    assert msgspec.to_builtins(result) == payload


def test_assistant_model_region_us() -> None:
    """The other documented region value decodes unchanged."""
    payload = json.dumps({"name": "a", "status": "Ready", "region": "us"}).encode()

    assert msgspec.json.decode(payload, type=AssistantModel).region == "us"


# ---------------------------------------------------------------------------
# Dict-like access tests (claims: unified-model-0001 through unified-model-0009)
# ---------------------------------------------------------------------------


def _make_assistant_model(**overrides: object) -> AssistantModel:
    data = {
        "name": "test-assistant",
        "status": "Ready",
        "created_at": "2025-01-15T12:00:00Z",
        "updated_at": "2025-01-15T12:00:00Z",
        "metadata": {"key": "value"},
        "instructions": "Be helpful.",
        "host": "test-assistant.svc.pinecone.io",
    }
    data.update(overrides)
    return AssistantModel(**data)  # type: ignore[arg-type]


def _make_file_model(**overrides: object) -> AssistantFileModel:
    data = {
        "name": "test-file.pdf",
        "id": "file-abc123",
        "metadata": None,
        "created_on": "2025-01-15T12:00:00Z",
        "updated_on": "2025-01-15T12:00:00Z",
        "status": "Available",
        "size": 1024,
        "multimodal": False,
        "signed_url": None,
        "content_hash": None,
    }
    data.update(overrides)
    return AssistantFileModel(**data)  # type: ignore[arg-type]


def test_model_dict_access_assistant() -> None:
    """AssistantModel supports subscript, in, len, keys, values, items, get."""
    model = _make_assistant_model()

    # __getitem__
    assert model["name"] == "test-assistant"
    assert model["status"] == "Ready"
    assert model["instructions"] == "Be helpful."

    # __contains__
    assert "name" in model
    assert "status" in model
    assert "nonexistent" not in model

    # __len__
    assert len(model) == len(model.__struct_fields__)

    # keys()
    assert "name" in model
    assert "status" in model

    # values()
    vals = model.values()
    assert "test-assistant" in vals
    assert "Ready" in vals

    # items()
    items = dict(model.items())
    assert items["name"] == "test-assistant"
    assert items["status"] == "Ready"

    # get() — existing key
    assert model.get("name") == "test-assistant"
    # get() — missing key with default
    assert model.get("nonexistent") is None
    assert model.get("nonexistent", "fallback") == "fallback"


def test_model_dict_access_file() -> None:
    """AssistantFileModel supports subscript, in, len, keys, values, items, get."""
    model = _make_file_model()

    assert model["name"] == "test-file.pdf"
    assert model["id"] == "file-abc123"
    assert "name" in model
    assert "nonexistent" not in model
    assert len(model) == len(model.__struct_fields__)
    assert "name" in model
    assert "test-file.pdf" in model.values()
    assert dict(model.items())["id"] == "file-abc123"
    assert model.get("size") == 1024
    assert model.get("missing", 0) == 0


def test_model_getitem_keyerror_assistant() -> None:
    """AssistantModel raises KeyError for unknown key subscript."""
    model = _make_assistant_model()
    with pytest.raises(KeyError):
        _ = model["nonexistent_field"]


def test_model_getitem_keyerror_file() -> None:
    """AssistantFileModel raises KeyError for unknown key subscript."""
    model = _make_file_model()
    with pytest.raises(KeyError):
        _ = model["nonexistent_field"]


def test_model_to_dict_assistant() -> None:
    """AssistantModel.to_dict() returns a plain dict with correct values."""
    model = _make_assistant_model(metadata={"env": "prod", "version": "1"})
    result = model.to_dict()

    assert isinstance(result, dict)
    assert result["name"] == "test-assistant"
    assert result["status"] == "Ready"
    assert result["metadata"] == {"env": "prod", "version": "1"}
    assert result["instructions"] == "Be helpful."


def test_model_to_dict_recursive() -> None:
    """AssistantModel.to_dict() works when retrieved from ListAssistantsResponse."""
    from pinecone.models.assistant.list import ListAssistantsResponse

    assistant = _make_assistant_model(metadata=None)
    response = ListAssistantsResponse(assistants=[assistant])

    # Attribute access still works; nested entity models still have to_dict()
    assert len(response.assistants) == 1
    nested = response.assistants[0].to_dict()
    assert isinstance(nested, dict)
    assert nested["name"] == "test-assistant"
    assert nested["status"] == "Ready"


def test_list_assistants_response_empty_attribute_access() -> None:
    """ListAssistantsResponse with no items has empty assistants list via attribute access."""
    from pinecone.models.assistant.list import ListAssistantsResponse

    response = ListAssistantsResponse(assistants=[])
    assert response.assistants == []


def test_model_str_repr_dict_form_assistant() -> None:
    """AssistantModel __str__ and __repr__ show class name."""
    model = _make_assistant_model()

    assert "AssistantModel(" in repr(model)
    assert "AssistantModel(" in str(model)


def test_model_str_repr_dict_form_file() -> None:
    """AssistantFileModel __str__ and __repr__ show class name."""
    model = _make_file_model()

    assert "AssistantFileModel(" in repr(model)
    assert "AssistantFileModel(" in str(model)


# ---------------------------------------------------------------------------
# evaluate_alignment() — success
# ---------------------------------------------------------------------------

EVAL_BASE_URL = ASSISTANT_EVALUATION_BASE_URL


@respx.mock
def test_evaluate_alignment(assistants: Assistants) -> None:
    """evaluate_alignment() POSTs to the evaluation endpoint and returns AlignmentResult."""
    from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores

    eval_route = respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    result = assistants.evaluate_alignment(
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
    assert result.facts[0].reasoning == "The answer states Barcelona but Madrid is the capital."
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 40
    assert result.usage.total_tokens == 160

    request = eval_route.calls.last.request
    body = json.loads(request.content)
    assert body["question"] == "What is the capital of Spain?"
    assert body["answer"] == "Barcelona."
    assert body["ground_truth_answer"] == "Madrid."


@respx.mock
def test_evaluate_alignment_request_body_fields(assistants: Assistants) -> None:
    """evaluate_alignment() sends all three required fields in the request body."""
    route = respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    assistants.evaluate_alignment(
        question="Q",
        answer="A",
        ground_truth_answer="GT",
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["question"] == "Q"
    assert body["answer"] == "A"
    assert body["ground_truth_answer"] == "GT"


@respx.mock
def test_evaluate_alignment_multiple_facts(assistants: Assistants) -> None:
    """evaluate_alignment() correctly maps multiple evaluated_facts from the response."""
    multi_fact_response = make_alignment_response(
        reasoning={
            "evaluated_facts": [
                {"fact": {"content": "Fact one."}, "entailment": "entailed"},
                {"fact": {"content": "Fact two."}, "entailment": "contradicted"},
                {"fact": {"content": "Fact three."}, "entailment": "neutral"},
            ]
        }
    )
    respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=multi_fact_response),
    )

    result = assistants.evaluate_alignment(
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
def test_evaluate_alignment_uses_api_key(assistants: Assistants) -> None:
    """evaluate_alignment() sends the Api-Key header in the request."""
    route = respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    assistants.evaluate_alignment(
        question="Q?",
        answer="A.",
        ground_truth_answer="GT.",
    )

    request = route.calls.last.request
    assert request.headers.get("Api-Key") == "test-key"


@respx.mock
def test_evaluate_alignment_reasoning_field(assistants: Assistants) -> None:
    """evaluate_alignment() populates EntailmentResult.reasoning from the API response."""
    from pinecone.models.assistant.evaluation import AlignmentResult

    respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=make_alignment_response()),
    )

    result = assistants.evaluate_alignment(
        question="What is the capital of Spain?",
        answer="Barcelona.",
        ground_truth_answer="Madrid.",
    )

    assert isinstance(result, AlignmentResult)
    assert all(isinstance(f.reasoning, str) for f in result.facts)
    # The factory includes a non-empty reasoning string
    assert result.facts[0].reasoning == "The answer states Barcelona but Madrid is the capital."


@respx.mock
def test_evaluate_alignment_reasoning_defaults_to_empty(assistants: Assistants) -> None:
    """evaluate_alignment() uses empty string for reasoning when API omits the field."""
    from pinecone.models.assistant.evaluation import AlignmentResult

    response_without_reasoning = make_alignment_response(
        reasoning={
            "evaluated_facts": [
                {"fact": {"content": "Fact without reasoning."}, "entailment": "entailed"}
            ]
        }
    )
    respx.post(f"{EVAL_BASE_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=response_without_reasoning),
    )

    result = assistants.evaluate_alignment(
        question="Q?",
        answer="A.",
        ground_truth_answer="GT.",
    )

    assert isinstance(result, AlignmentResult)
    assert result.facts[0].reasoning == ""


# ---------------------------------------------------------------------------
# _chat_streaming() — transport error wrapping
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_timeout_raises_pinecone_timeout_error(
    assistants: Assistants,
) -> None:
    """httpx.ReadTimeout during streaming is wrapped as PineconeTimeoutError."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        side_effect=httpx.ReadTimeout("test"),
    )

    with pytest.raises(PineconeTimeoutError):
        list(
            assistants.chat(
                assistant_name="test-assistant",
                messages=[{"content": "Hello"}],
                stream=True,
            )
        )


@respx.mock
def test_chat_streaming_connect_error_raises_pinecone_connection_error(
    assistants: Assistants,
) -> None:
    """httpx.ConnectError during streaming is wrapped as PineconeConnectionError."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        side_effect=httpx.ConnectError("Connection refused"),
    )

    with pytest.raises(PineconeConnectionError):
        list(
            assistants.chat(
                assistant_name="test-assistant",
                messages=[{"content": "Hello"}],
                stream=True,
            )
        )


# ---------------------------------------------------------------------------
# _chat_streaming() — config timeout propagation
# ---------------------------------------------------------------------------


def test_chat_streaming_uses_config_timeout() -> None:
    """Custom PineconeConfig.timeout is forwarded to the underlying httpx stream call."""
    config = PineconeConfig(api_key="test-key", host=BASE_URL, timeout=300.0)
    custom_assistants = Assistants(config=config)

    # Pre-populate the data plane client cache to avoid needing a describe() mock.
    data_config = PineconeConfig(api_key="test-key", host=DATA_PLANE_URL, timeout=300.0)
    data_plane_client = HTTPClient(data_config, ASSISTANT_API_VERSION)
    custom_assistants._data_plane_clients["test-assistant"] = data_plane_client

    captured_timeout: list[float | None] = []

    @contextlib.contextmanager
    def _mock_httpx_stream(
        method: str,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        captured_timeout.append(timeout)
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.iter_lines.return_value = iter(
            [
                'data: {"type": "message_end", "id": "e1",'
                ' "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}',
            ]
        )
        yield mock_resp

    with patch.object(data_plane_client._client, "stream", _mock_httpx_stream):
        list(
            custom_assistants.chat(
                assistant_name="test-assistant",
                messages=[{"content": "Hello"}],
                stream=True,
            )
        )

    assert len(captured_timeout) == 1
    assert captured_timeout[0] == 300.0


# ---------------------------------------------------------------------------
# ChatStreamChunk — tag-based dispatch
# ---------------------------------------------------------------------------


def test_stream_chunk_tag_dispatch() -> None:
    """msgspec.convert dispatches to StreamMessageStart by tag when type='message_start'."""
    from pinecone.models.assistant.streaming import ChatStreamChunk, StreamMessageStart

    chunk = msgspec.convert(
        {"type": "message_start", "model": "gpt-4o", "role": "assistant"},
        ChatStreamChunk,
    )
    assert isinstance(chunk, StreamMessageStart)
    assert chunk.model == "gpt-4o"
    assert chunk.role == "assistant"


def test_stream_chunk_unknown_type() -> None:
    """msgspec.convert raises ValidationError when the type tag is not recognised."""
    from pinecone.models.assistant.streaming import ChatStreamChunk

    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"type": "unknown", "data": "foo"}, ChatStreamChunk)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestAssistantsClose:
    def test_close_calls_main_http_close(self, assistants: Assistants) -> None:
        """close() calls close() on the main _http client."""
        mock_http = MagicMock()
        assistants._http = mock_http

        assistants.close()

        mock_http.close.assert_called_once()

    def test_close_calls_eval_http_close(self, assistants: Assistants) -> None:
        """close() calls close() on the _eval_http client."""
        mock_eval_http = MagicMock()
        assistants._eval_http = mock_eval_http

        assistants.close()

        mock_eval_http.close.assert_called_once()

    def test_close_closes_cached_data_plane_clients(self, assistants: Assistants) -> None:
        """close() closes all cached data-plane clients and clears the dict."""
        mock_a = MagicMock()
        mock_b = MagicMock()
        assistants._data_plane_clients = {"a": mock_a, "b": mock_b}

        assistants.close()

        mock_a.close.assert_called_once()
        mock_b.close.assert_called_once()
        assert assistants._data_plane_clients == {}

    def test_close_is_idempotent(self, assistants: Assistants) -> None:
        """close() on an empty _data_plane_clients is a no-op and does not raise."""
        assistants._data_plane_clients = {}

        assistants.close()
        assistants.close()


# ---------------------------------------------------------------------------
# ChatCompletionMessage optional fields
# ---------------------------------------------------------------------------


def test_chat_completion_choice_message_optional_fields() -> None:
    """ChatCompletionMessage decodes successfully when role and content are absent."""
    import msgspec

    from pinecone.models.assistant.chat import ChatCompletionMessage, ChatCompletionResponse

    payload = msgspec.json.encode(
        {
            "id": "chatcmpl-empty",
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            "choices": [
                {
                    "index": 0,
                    "message": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    response = msgspec.json.decode(payload, type=ChatCompletionResponse)
    assert response.choices[0].message.role is None
    assert response.choices[0].message.content is None
    assert isinstance(response.choices[0].message, ChatCompletionMessage)


# ---------------------------------------------------------------------------
# ChatCompletionStreamChunk usage field
# ---------------------------------------------------------------------------


def test_chat_completion_stream_chunk_usage() -> None:
    """ChatCompletionStreamChunk captures usage when present in an SSE data line."""
    import orjson

    from pinecone.models.assistant.chat import ChatUsage
    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    raw = orjson.loads(
        b'{"id": "cmpl-end", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'
    )
    chunk = msgspec.convert(raw, ChatCompletionStreamChunk)

    assert chunk.usage is not None
    assert isinstance(chunk.usage, ChatUsage)
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5
    assert chunk.usage.total_tokens == 15


def test_chat_completion_stream_chunk_usage_absent() -> None:
    """ChatCompletionStreamChunk.usage is None when the SSE data line omits the field."""
    import orjson

    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    raw = orjson.loads(
        b'{"id": "cmpl-mid", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}'
    )
    chunk = msgspec.convert(raw, ChatCompletionStreamChunk)

    assert chunk.usage is None


# ---------------------------------------------------------------------------
# 2026-07 error codes
# ---------------------------------------------------------------------------


@respx.mock
def test_too_many_requests_surfaces_server_message() -> None:
    """A 429 TOO_MANY_REQUESTS keeps the server's code and message."""
    from pinecone._internal.config import RetryConfig
    from pinecone.errors.exceptions import RateLimitError

    config = PineconeConfig(
        api_key="test-key", host=BASE_URL, retry_config=RetryConfig(max_retries=0)
    )
    client = Assistants(config=config)
    respx.get(f"{BASE_URL}/assistant/assistants/my-assistant").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "code": "TOO_MANY_REQUESTS",
                    "message": "Too many get or list assistant requests, try again later",
                },
                "status": 429,
            },
        )
    )

    try:
        with pytest.raises(RateLimitError) as exc_info:
            client.describe(name="my-assistant")
    finally:
        client.close()

    err = exc_info.value
    assert err.status_code == 429
    assert err.error_code == "TOO_MANY_REQUESTS"
    assert err.message == "Too many get or list assistant requests, try again later"
    assert "Too many get or list assistant requests" in str(err)


# ---------------------------------------------------------------------------
# describe_operation() / list_operations()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_operation_targets_the_data_plane(assistants: Assistants) -> None:
    """The operations endpoints live on the assistant's host, not the control plane."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response()),
    )

    result = assistants.describe_operation(
        assistant_name="test-assistant", operation_id="op-abc123"
    )

    assert result.operation_id == "op-abc123"
    assert result.operation_type == "upload_file"
    assert result.file_id == "file-abc123"
    request = route.calls.last.request
    assert request.method == "GET"
    assert str(request.url) == f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123"
    assert request.headers["x-pinecone-api-version"] == ASSISTANT_API_VERSION


@respx.mock
def test_describe_operation_reuses_the_cached_data_plane_client(assistants: Assistants) -> None:
    """Repeated describes cost one control-plane host lookup, not one each."""
    describe_assistant = respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(200, json=make_file_operation_response()),
    )

    for _ in range(3):
        assistants.describe_operation(assistant_name="test-assistant", operation_id="op-abc123")

    assert describe_assistant.call_count == 1


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_the_polling_loop_goes_through_describe_operation(
    mock_sleep: object, assistants: Assistants
) -> None:
    """#125's internal poll is the public method now, so patching it is observable."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(202, json=make_file_operation_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    with patch.object(
        Assistants,
        "describe_operation",
        wraps=assistants.describe_operation,
    ) as spy:
        respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
            side_effect=[
                httpx.Response(200, json=make_file_operation_response(status="Processing")),
                httpx.Response(200, json=make_file_operation_response(status="Completed")),
            ],
        )
        assistants.upload_file(assistant_name="test-assistant", file_stream=io.BytesIO(b"data"))

    assert spy.call_count == 2
    for call in spy.call_args_list:
        assert call.kwargs == {
            "assistant_name": "test-assistant",
            "operation_id": "op-abc123",
        }


@respx.mock
def test_describe_operation_surfaces_a_server_error(assistants: Assistants) -> None:
    """A 500 from the operations endpoint is not swallowed into a fake status."""
    from pinecone.errors.exceptions import ApiError

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant/op-abc123").mock(
        return_value=httpx.Response(
            500,
            json={"status": 500, "error": {"code": "UNKNOWN", "message": "Internal server error"}},
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        assistants.describe_operation(assistant_name="test-assistant", operation_id="op-abc123")

    assert exc_info.value.status_code == 500


@respx.mock
def test_list_operations_page_returns_the_model_and_the_token(assistants: Assistants) -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_operation_list_response(
                [make_file_operation_response(), make_file_operation_response(id="op-def456")],
                next_token="tok-2",
            ),
        ),
    )

    page = assistants.list_operations_page(assistant_name="test-assistant")

    assert isinstance(page, ListOperationsResponse)
    assert [op.operation_id for op in page.operations] == ["op-abc123", "op-def456"]
    assert page.next == "tok-2"
    request = route.calls.last.request
    assert request.headers["x-pinecone-api-version"] == ASSISTANT_API_VERSION


@respx.mock
def test_list_operations_page_omits_unset_parameters(assistants: Assistants) -> None:
    """Only what the caller asked for goes on the wire."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant").mock(
        return_value=httpx.Response(200, json=make_operation_list_response()),
    )

    assistants.list_operations_page(assistant_name="test-assistant", status="Completed")

    params = route.calls.last.request.url.params
    assert dict(params) == {"status": "Completed"}


@respx.mock
def test_list_operations_returns_a_paginator_with_pages(assistants: Assistants) -> None:
    """``pages()`` exposes the raw pages, matching ``list_files``."""
    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/operations/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_operation_list_response(
                    [make_file_operation_response()], next_token="tok-2"
                ),
            ),
            httpx.Response(
                200, json=make_operation_list_response([make_file_operation_response(id="op-def")])
            ),
        ],
    )

    paginator = assistants.list_operations(assistant_name="test-assistant")
    assert isinstance(paginator, Paginator)

    pages = list(paginator.pages())
    assert [p.pagination_token for p in pages] == ["tok-2", None]
    assert [op.operation_id for page in pages for op in page.items] == ["op-abc123", "op-def"]


@respx.mock
def test_list_operations_page_surfaces_the_backend_limit_ceiling(
    assistants: Assistants,
) -> None:
    """page_size is not validated client-side; the backend's 400 says the ceiling."""
    from pinecone.errors.exceptions import ApiError

    respx.get(f"{BASE_URL}/assistant/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/operations/test-assistant").mock(
        return_value=httpx.Response(
            400,
            json={
                "status": 400,
                "error": {"code": "INVALID_ARGUMENT", "message": "Limit cannot exceed 100"},
            },
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        assistants.list_operations_page(assistant_name="test-assistant", page_size=101)

    assert exc_info.value.message == "Limit cannot exceed 100"
    assert route.calls.last.request.url.params["limit"] == "101"


@pytest.mark.parametrize("bad", ["upload", "UploadFile", "", "upload_file "])
def test_list_operations_rejects_an_unknown_operation_type(
    assistants: Assistants, bad: str
) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        assistants.list_operations_page(assistant_name="test-assistant", operation_type=bad)

    message = str(exc_info.value)
    assert message.startswith("operation_type must be one of (")
    for valid in ("upload_file", "upsert_file", "update_file_metadata", "delete_file"):
        assert repr(valid) in message
    assert repr(bad) in message


@pytest.mark.parametrize("bad", ["completed", "Succeeded", "PartiallyFailed", ""])
def test_list_operations_rejects_an_unknown_status(assistants: Assistants, bad: str) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        assistants.list_operations_page(assistant_name="test-assistant", status=bad)

    message = str(exc_info.value)
    assert message.startswith("status must be one of (")
    for valid in ("Processing", "Completed", "Failed"):
        assert repr(valid) in message
    assert repr(bad) in message
