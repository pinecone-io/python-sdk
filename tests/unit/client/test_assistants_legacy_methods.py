"""Unit tests for AssistantsLegacyNamespaceMixin shim methods."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.client.assistants import Assistants
from pinecone.models.assistant.list import ListAssistantsResponse, _Pagination
from pinecone.models.assistant.model import AssistantModel

_CANNED_ASSISTANT = AssistantModel(
    name="legacy-name",
    status="Ready",
    created_at="2025-01-15T12:00:00Z",
    updated_at="2025-01-15T12:00:00Z",
    metadata={},
    instructions=None,
    host="test-assistant-abc123.svc.pinecone.io",
)

# ---------------------------------------------------------------------------
# list_assistants shim tests
# ---------------------------------------------------------------------------


def test_list_assistants_returns_list(mock_assistants: Assistants) -> None:
    """list_assistants() materializes the paginator and returns a plain list."""
    mock_assistants._adapter.to_assistant_list.return_value = ListAssistantsResponse(  # type: ignore[attr-defined]
        assistants=[_CANNED_ASSISTANT]
    )
    result = mock_assistants.list_assistants()
    assert isinstance(result, list)
    assert all(isinstance(a, AssistantModel) for a in result)


def test_list_assistants_returns_all_pages(mock_assistants: Assistants) -> None:
    """list_assistants() follows pagination and returns items from all pages."""
    page1 = ListAssistantsResponse(
        assistants=[_CANNED_ASSISTANT], pagination=_Pagination(next="token-1")
    )
    page2 = ListAssistantsResponse(assistants=[_CANNED_ASSISTANT])
    mock_assistants._adapter.to_assistant_list.side_effect = [page1, page2]  # type: ignore[attr-defined]
    result = mock_assistants.list_assistants()
    assert len(result) == 2


# ---------------------------------------------------------------------------
# list_assistants_paginated shim tests
# ---------------------------------------------------------------------------


def test_list_assistants_paginated_returns_response_shape(
    mock_assistants: Assistants,
) -> None:
    """list_assistants_paginated() returns a ListAssistantsResponse."""
    mock_assistants._adapter.to_assistant_list.return_value = ListAssistantsResponse(  # type: ignore[attr-defined]
        assistants=[_CANNED_ASSISTANT]
    )
    resp = mock_assistants.list_assistants_paginated(limit=2)
    assert isinstance(resp, ListAssistantsResponse)
    assert hasattr(resp, "assistants")
    assert hasattr(resp, "next")


def test_list_assistants_paginated_legacy_limit_alias(
    mock_assistants: Assistants,
) -> None:
    """list_assistants_paginated(limit=...) passes page_size= to list_page."""
    canned_response = ListAssistantsResponse(assistants=[])
    original_list_page = mock_assistants.list_page
    spy_list_page = MagicMock(side_effect=original_list_page)
    mock_assistants.list_page = spy_list_page  # type: ignore[method-assign]
    mock_assistants._adapter.to_assistant_list.return_value = canned_response  # type: ignore[attr-defined]

    mock_assistants.list_assistants_paginated(limit=5)

    spy_list_page.assert_called_once_with(page_size=5, pagination_token=None)


def test_list_assistants_paginated_page_size_kwarg(
    mock_assistants: Assistants,
) -> None:
    """list_assistants_paginated(page_size=...) also passes page_size= to list_page."""
    canned_response = ListAssistantsResponse(assistants=[])
    original_list_page = mock_assistants.list_page
    spy_list_page = MagicMock(side_effect=original_list_page)
    mock_assistants.list_page = spy_list_page  # type: ignore[method-assign]
    mock_assistants._adapter.to_assistant_list.return_value = canned_response  # type: ignore[attr-defined]

    mock_assistants.list_assistants_paginated(page_size=10)

    spy_list_page.assert_called_once_with(page_size=10, pagination_token=None)


def test_list_assistants_paginated_with_pagination_token(
    mock_assistants: Assistants,
) -> None:
    """list_assistants_paginated() forwards pagination_token to list_page."""
    canned_response = ListAssistantsResponse(assistants=[])
    original_list_page = mock_assistants.list_page
    spy_list_page = MagicMock(side_effect=original_list_page)
    mock_assistants.list_page = spy_list_page  # type: ignore[method-assign]
    mock_assistants._adapter.to_assistant_list.return_value = canned_response  # type: ignore[attr-defined]

    mock_assistants.list_assistants_paginated(pagination_token="some-token")

    spy_list_page.assert_called_once_with(page_size=None, pagination_token="some-token")


def test_create_assistant_legacy_method(mock_assistants: Assistants) -> None:
    """create_assistant() with assistant_name= delegates to create() and returns a model."""
    result = mock_assistants.create_assistant(assistant_name="foo", instructions="be nice")
    # Response is the canned fixture value from mock_assistants
    assert result.name == "legacy-name"


def test_create_assistant_legacy_forwards_to_create(
    mock_assistants: Assistants, spy_create: MagicMock
) -> None:
    """create_assistant() forwards the right arguments to create()."""
    mock_assistants.create_assistant(assistant_name="foo")
    spy_create.assert_called_once_with(
        name="foo",
        instructions=None,
        metadata=None,
        region="us",
        timeout=None,
    )


def test_create_assistant_with_name_kwarg(mock_assistants: Assistants) -> None:
    """create_assistant() accepts the current-style name= kwarg too."""
    result = mock_assistants.create_assistant(name="bar")
    assert result.name == "legacy-name"


def test_create_assistant_name_kwarg_forwards_correctly(
    mock_assistants: Assistants, spy_create: MagicMock
) -> None:
    """create_assistant(name=...) resolves to create(name=...)."""
    mock_assistants.create_assistant(name="bar")
    spy_create.assert_called_once_with(
        name="bar",
        instructions=None,
        metadata=None,
        region="us",
        timeout=None,
    )


def test_create_assistant_no_name_propagates_error(
    mock_assistants: Assistants,
) -> None:
    """create_assistant() with no name propagates the error from create()."""
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.create_assistant()


# ---------------------------------------------------------------------------
# describe_assistant shim tests
# ---------------------------------------------------------------------------


def test_describe_assistant_legacy_method(mock_assistants: Assistants) -> None:
    """describe_assistant(assistant_name=...) delegates to describe() and returns a model."""
    result = mock_assistants.describe_assistant(assistant_name="foo")
    # Returns the canned fixture value from mock_assistants
    assert result.name == "legacy-name"


def test_describe_assistant_legacy_with_name_kwarg(mock_assistants: Assistants) -> None:
    """describe_assistant(name=...) accepts the current-style kwarg."""
    result = mock_assistants.describe_assistant(name="foo")
    assert result.name == "legacy-name"


def test_describe_assistant_rejects_both_names(mock_assistants: Assistants) -> None:
    """describe_assistant() raises PineconeValueError when both assistant_name and name are given.

    Matches the canonical ``describe()``/``create()``/etc. convention for the identical
    "received both the legacy and current name" condition (see
    ``pinecone/_internal/kwargs_aliases.py``), rather than the bare ``TypeError`` this
    shim previously raised.
    """
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="Pass only one") as exc_info:
        mock_assistants.describe_assistant("a", name="b")
    assert type(exc_info.value) is PineconeValueError
    assert isinstance(exc_info.value, ValueError)


def test_describe_assistant_legacy_forwards_to_describe(
    mock_assistants: Assistants,
) -> None:
    """describe_assistant() forwards assistant_name as name= to describe()."""
    spy = MagicMock(side_effect=mock_assistants.describe)
    mock_assistants.describe = spy  # type: ignore[method-assign]

    mock_assistants.describe_assistant(assistant_name="my-assistant")
    spy.assert_called_once_with(name="my-assistant")


# ---------------------------------------------------------------------------
# update_assistant shim tests
# ---------------------------------------------------------------------------


def test_update_assistant_legacy_method(mock_assistants: Assistants) -> None:
    """update_assistant(assistant_name=...) delegates to update() and returns a model."""
    result = mock_assistants.update_assistant(
        assistant_name="foo",
        instructions="be precise",
        metadata={"k": "v"},
    )
    # Returns the canned fixture value from mock_assistants
    assert result.name == "legacy-name"


def test_update_assistant_legacy_positional_args(mock_assistants: Assistants) -> None:
    """Legacy SDK supported positional: update_assistant("foo", "inst", {"k": "v"})."""
    result = mock_assistants.update_assistant("foo", "be nice", {"k": "v"})
    assert result.name == "legacy-name"


def test_update_assistant_with_name_kwarg(mock_assistants: Assistants) -> None:
    """update_assistant(name=...) accepts the current-style kwarg too."""
    result = mock_assistants.update_assistant(name="bar", instructions="new instructions")
    assert result.name == "legacy-name"


def test_update_assistant_forwards_to_update(mock_assistants: Assistants) -> None:
    """update_assistant() forwards assistant_name as name= to update()."""
    spy = MagicMock(side_effect=mock_assistants.update)
    mock_assistants.update = spy  # type: ignore[method-assign]

    mock_assistants.update_assistant(
        assistant_name="foo",
        instructions="be precise",
        metadata={"k": "v"},
    )
    spy.assert_called_once_with(name="foo", instructions="be precise", metadata={"k": "v"})


# ---------------------------------------------------------------------------
# delete_assistant shim tests
# ---------------------------------------------------------------------------


def test_delete_assistant_legacy_method(mock_assistants: Assistants) -> None:
    """delete_assistant(assistant_name=...) delegates to delete()."""
    spy = MagicMock()
    mock_assistants.delete = spy  # type: ignore[method-assign]

    mock_assistants.delete_assistant(assistant_name="foo", timeout=30)
    spy.assert_called_once_with(name="foo", timeout=30)


def test_delete_assistant_legacy_positional(mock_assistants: Assistants) -> None:
    """delete_assistant("foo", 30) using positional legacy-style call."""
    spy = MagicMock()
    mock_assistants.delete = spy  # type: ignore[method-assign]

    mock_assistants.delete_assistant("foo", 30)
    spy.assert_called_once_with(name="foo", timeout=30)


def test_delete_assistant_with_name_kwarg(mock_assistants: Assistants) -> None:
    """delete_assistant(name=...) accepts the current-style kwarg."""
    spy = MagicMock()
    mock_assistants.delete = spy  # type: ignore[method-assign]

    mock_assistants.delete_assistant(name="bar")
    spy.assert_called_once_with(name="bar", timeout=None)


def test_delete_assistant_no_timeout(mock_assistants: Assistants) -> None:
    """delete_assistant() without timeout passes timeout=None to delete()."""
    spy = MagicMock()
    mock_assistants.delete = spy  # type: ignore[method-assign]

    mock_assistants.delete_assistant(assistant_name="baz")
    spy.assert_called_once_with(name="baz", timeout=None)
