"""2026-07 conformance for the asyncio transport of the five assistant
control-plane operations and the single evaluation operation.

The sync variants live in ``test_assistant_2026_07.py``; both may claim the
same operation (see README, "Additional rules"), and these add no operation
ids to the coverage numerator. What they add is the guarantee that the async
client puts the same method, path and ``X-Pinecone-Api-Version`` on the wire
now that ``AsyncAssistants`` has a single ``ASSISTANT_API_VERSION`` client —
the payloads and expected shapes are imported from the sync module rather
than restated, so the two transports cannot drift apart in the fixtures.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.adapters.assistants_adapter import _AlignmentResponse
from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel
from tests.unit.conformance import api_op
from tests.unit.conformance.test_assistant_2026_07 import (
    ALIGNMENT,
    ASSISTANT,
    ASSISTANT_NAME,
    ASSISTANT_OPTIONAL,
    BASE_URL,
    CONTROL_URL,
    EVAL_URL,
)


@pytest.fixture
async def async_assistants() -> AsyncGenerator[AsyncAssistants]:
    client = AsyncAssistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    await client.close()


@api_op("assistant_control:list_assistants")
async def test_async_list_assistants(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    payload = {"assistants": [ASSISTANT], "pagination": {"next": "dXNlcl9pZD11c2VyXzI="}}
    route = respx_mock.get(f"{CONTROL_URL}/assistants").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = await async_assistants.list_page(page_size=20, pagination_token="dXNlcl9pZD11c2VyXzE=")
    assert [a.name for a in result.assistants] == [ASSISTANT_NAME]
    assert [a.region for a in result.assistants] == ["eu"]
    assert result.next == "dXNlcl9pZD11c2VyXzI="

    request = route.calls.last.request
    assert dict(request.url.params) == {
        "limit": "20",
        "pagination_token": "dXNlcl9pZD11c2VyXzE=",
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ListAssistantsResponse, payload, optional_absent=["pagination"])


@api_op("assistant_control:create_assistant")
async def test_async_create_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{CONTROL_URL}/assistants").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = await async_assistants.create(
        name=ASSISTANT_NAME,
        instructions=ASSISTANT["instructions"],
        metadata=ASSISTANT["metadata"],
        region="eu",
        timeout=-1,
    )
    assert result.name == ASSISTANT_NAME
    assert result.region == "eu"

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "name": ASSISTANT_NAME,
        "instructions": ASSISTANT["instructions"],
        "metadata": ASSISTANT["metadata"],
        "region": "eu",
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantModel, ASSISTANT, optional_absent=ASSISTANT_OPTIONAL)


@api_op("assistant_control:get_assistant")
async def test_async_get_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = await async_assistants.describe(name=ASSISTANT_NAME)
    assert result.status == "Ready"
    assert result.region == "eu"
    assert result.created_at == "2026-07-01T12:30:00Z"
    assert result.updated_at == "2026-07-01T12:45:00Z"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantModel, ASSISTANT, optional_absent=ASSISTANT_OPTIONAL)


@api_op("assistant_control:update_assistant")
async def test_async_update_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.patch(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = await async_assistants.update(
        name=ASSISTANT_NAME,
        instructions="Keep the tone friendly.",
        metadata={"team": "Operations"},
    )
    assert result.name == ASSISTANT_NAME

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "instructions": "Keep the tone friendly.",
        "metadata": {"team": "Operations"},
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantModel, ASSISTANT, optional_absent=ASSISTANT_OPTIONAL)


@api_op("assistant_control:delete_assistant")
async def test_async_delete_assistant(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200)
    )

    returned = await async_assistants.delete(name=ASSISTANT_NAME, timeout=-1)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("assistant_evaluation:metrics_alignment")
async def test_async_metrics_alignment(
    claim: Any, async_assistants: AsyncAssistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{EVAL_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=ALIGNMENT)
    )

    result = await async_assistants.evaluate_alignment(
        question="What is the capital city of Spain?",
        answer="Barcelona.",
        ground_truth_answer="Madrid.",
    )
    assert result.scores.completeness == 1.0
    assert [f.entailment for f in result.facts] == ["entailed", "contradicted"]
    assert result.usage.total_tokens == 160

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "question": "What is the capital city of Spain?",
        "answer": "Barcelona.",
        "ground_truth_answer": "Madrid.",
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(_AlignmentResponse, ALIGNMENT, optional_absent=[])
