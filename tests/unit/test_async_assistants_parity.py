"""Sync/async parity for the assistant control-plane and evaluation surface.

Now that ``Assistants`` and ``AsyncAssistants`` speak one
``ASSISTANT_API_VERSION`` (2026-07) against the same handlers, the two
clients differ only in ``await``. These tests hold them to that: identical
parameter names, kinds, defaults and annotations, identical return
annotations modulo ``AsyncPaginator``, and identical exception types and
messages for the failures a caller can trigger.

Follows the pattern of ``tests/unit/preview/test_async_parity.py``.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import NotFoundError, PineconeError, PineconeValueError
from tests.factories import make_assistant_response

BASE_URL = "https://api.test.pinecone.io"

_METHODS = [
    "create",
    "delete",
    "describe",
    "evaluate_alignment",
    "list",
    "list_page",
    "update",
]


def _comparable(annotation: Any) -> str:
    return str(annotation).replace("AsyncPaginator[", "Paginator[")


@pytest.fixture
def sync_assistants() -> Assistants:
    return Assistants(config=PineconeConfig(api_key="test-key", host=BASE_URL))


@pytest.fixture
def async_assistants() -> AsyncAssistants:
    return AsyncAssistants(config=PineconeConfig(api_key="test-key", host=BASE_URL))


@pytest.mark.parametrize("method_name", _METHODS)
def test_assistants_parameter_parity(method_name: str) -> None:
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


@pytest.mark.parametrize("method_name", _METHODS)
def test_assistants_return_annotation_parity(method_name: str) -> None:
    sync_return = inspect.signature(getattr(Assistants, method_name)).return_annotation
    async_return = inspect.signature(getattr(AsyncAssistants, method_name)).return_annotation

    assert _comparable(sync_return) == _comparable(async_return), (
        f"{method_name}: return annotation differs (sync={sync_return}, async={async_return})"
    )


async def test_invalid_region_error_parity(
    sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    with pytest.raises(PineconeValueError) as sync_exc:
        sync_assistants.create(name="test-assistant", region="ap-southeast-1")
    with pytest.raises(PineconeValueError) as async_exc:
        await async_assistants.create(name="test-assistant", region="ap-southeast-1")

    assert type(sync_exc.value) is type(async_exc.value)
    assert str(sync_exc.value) == str(async_exc.value)


@pytest.mark.parametrize("method_name", ["create", "delete", "describe", "update"])
async def test_missing_name_error_parity(
    method_name: str, sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    with pytest.raises(PineconeValueError) as sync_exc:
        getattr(sync_assistants, method_name)()
    with pytest.raises(PineconeValueError) as async_exc:
        await getattr(async_assistants, method_name)()

    assert type(sync_exc.value) is type(async_exc.value)
    assert str(sync_exc.value) == str(async_exc.value)


async def test_empty_update_error_parity(
    sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    """update() with neither field refuses on both transports with one message."""
    with pytest.raises(PineconeValueError) as sync_exc:
        sync_assistants.update(name="test-assistant")
    with pytest.raises(PineconeValueError) as async_exc:
        await async_assistants.update(name="test-assistant")

    assert type(sync_exc.value) is type(async_exc.value)
    assert str(sync_exc.value) == str(async_exc.value)


@pytest.mark.parametrize("status", ["Failed", "InitializationFailed"])
@respx.mock
async def test_delete_terminal_state_error_parity(
    status: str, sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    """A delete parked in a terminal failure state reports identically on both."""
    respx.delete(f"{BASE_URL}/assistant/assistants/stuck-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistant/assistants/stuck-assistant").mock(
        return_value=httpx.Response(
            200, json=make_assistant_response(name="stuck-assistant", status=status)
        ),
    )

    with (
        patch("pinecone.client.assistants.time.sleep"),
        pytest.raises(PineconeError) as sync_exc,
    ):
        sync_assistants.delete(name="stuck-assistant")
    with (
        patch("pinecone.async_client.assistants.asyncio.sleep"),
        pytest.raises(PineconeError) as async_exc,
    ):
        await async_assistants.delete(name="stuck-assistant")

    assert type(sync_exc.value) is type(async_exc.value)
    assert str(sync_exc.value) == str(async_exc.value)


@respx.mock
async def test_describe_not_found_error_parity(
    sync_assistants: Assistants, async_assistants: AsyncAssistants
) -> None:
    respx.get(f"{BASE_URL}/assistant/assistants/missing-assistant").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}, "status": 404}),
    )

    with pytest.raises(NotFoundError) as sync_exc:
        sync_assistants.describe(name="missing-assistant")
    with pytest.raises(NotFoundError) as async_exc:
        await async_assistants.describe(name="missing-assistant")

    assert type(sync_exc.value) is type(async_exc.value)
    assert str(sync_exc.value) == str(async_exc.value)
