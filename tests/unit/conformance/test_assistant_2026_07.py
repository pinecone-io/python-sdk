"""2026-07 conformance for the five assistant control-plane operations and
the single evaluation operation.

``assistant_control_2026-07.oas.yaml`` and ``assistant_evaluation_2026-07.oas.yaml``
are byte-for-byte their 2026-04 predecessors apart from ``info.version``, and
the backend routes 2026-07 to the same handlers as 2026-04
(``svc-knowledge-engine/src/control/service/routes/mod.rs:21``). These tests
pin method, path, the ``X-Pinecone-Api-Version`` header now that a single
``ASSISTANT_API_VERSION`` feeds every assistant client, and the response
schemas.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.adapters.assistants_adapter import _AlignmentResponse
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL
from pinecone.client.assistants import Assistants
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel
from tests.unit.conformance import api_op

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
EVAL_URL = ASSISTANT_EVALUATION_BASE_URL

ASSISTANT_NAME = "conformance-assistant"

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "instructions": "Answer questions with clear, helpful answers.",
    "metadata": {"role": "Customer Support Helper", "team": "Operations"},
    "host": "https://prod-1-data.ke.pinecone.io",
    "region": "eu",
    "created_at": "2026-07-01T12:30:00Z",
    "updated_at": "2026-07-01T12:45:00Z",
}

ASSISTANT_OPTIONAL = [
    "instructions",
    "metadata",
    "host",
    "region",
    "created_at",
    "updated_at",
]

ALIGNMENT: dict[str, Any] = {
    "metrics": {"correctness": 0.5, "completeness": 1.0, "alignment": 0.667},
    "reasoning": {
        "evaluated_facts": [
            {"fact": {"content": "Madrid is the capital of Spain."}, "entailment": "entailed"},
            {
                "fact": {"content": "Barcelona is the capital of Spain."},
                "entailment": "contradicted",
            },
        ]
    },
    "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
}


@pytest.fixture
def assistants() -> Iterator[Assistants]:
    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    client.close()


@api_op("assistant_control:list_assistants")
def test_list_assistants(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    payload = {"assistants": [ASSISTANT], "pagination": {"next": "dXNlcl9pZD11c2VyXzI="}}
    route = respx_mock.get(f"{CONTROL_URL}/assistants").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = assistants.list_page(page_size=20, pagination_token="dXNlcl9pZD11c2VyXzE=")
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
def test_create_assistant(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{CONTROL_URL}/assistants").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = assistants.create(
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
def test_get_assistant(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = assistants.describe(name=ASSISTANT_NAME)
    assert result.status == "Ready"
    assert result.region == "eu"
    assert result.created_at == "2026-07-01T12:30:00Z"
    assert result.updated_at == "2026-07-01T12:45:00Z"

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantModel, ASSISTANT, optional_absent=ASSISTANT_OPTIONAL)


@api_op("assistant_control:update_assistant")
def test_update_assistant(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )

    result = assistants.update(
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
def test_delete_assistant(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200)
    )

    returned = assistants.delete(name=ASSISTANT_NAME, timeout=-1)

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


# assistant_evaluation is one operation and this is it, so the surface's whole
# version claim rests here — and it is empty: the knowledge-engine evaluation
# router mounts no api_versioning layer, so the version leg below passes
# vacuously (#348). Annotated in VACUOUS_VERSION_HEADER (scripts/api_coverage.py);
# the assertion stays so it starts meaning something the day the router gates.
@api_op("assistant_evaluation:metrics_alignment")
def test_metrics_alignment(
    claim: Any, assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{EVAL_URL}/evaluation/metrics/alignment").mock(
        return_value=httpx.Response(200, json=ALIGNMENT)
    )

    result = assistants.evaluate_alignment(
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
