"""Unit tests for legacy kwarg aliases on Assistants.create."""

from __future__ import annotations

import pytest

from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import PineconeValueError


def test_create_accepts_legacy_assistant_name_kwarg(mock_assistants: Assistants) -> None:
    """assistant_name= is accepted as a legacy alias for name=."""
    result = mock_assistants.create(assistant_name="legacy-name")
    assert result.name == "legacy-name"


def test_create_rejects_both_name_and_assistant_name(mock_assistants: Assistants) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.create(name="a", assistant_name="b")


def test_create_rejects_unknown_kwarg(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.create(name="foo", bogus_param=1)


def test_create_missing_name_raises(mock_assistants: Assistants) -> None:
    """Calling create() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.create()


def test_create_with_name_still_works(mock_assistants: Assistants) -> None:
    """The canonical name= parameter continues to work as before."""
    result = mock_assistants.create(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


def test_describe_accepts_legacy_assistant_name(mock_assistants: Assistants) -> None:
    """assistant_name= is accepted as a legacy alias for name= on describe."""
    result = mock_assistants.describe(assistant_name="foo")
    assert result.name == "legacy-name"  # canned response from fixture


def test_describe_rejects_both(mock_assistants: Assistants) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.describe(name="a", assistant_name="b")


def test_describe_rejects_unknown(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.describe(name="foo", random_arg=1)


def test_describe_missing_name_raises(mock_assistants: Assistants) -> None:
    """Calling describe() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.describe()


def test_describe_with_name_still_works(mock_assistants: Assistants) -> None:
    """The canonical name= parameter continues to work as before."""
    result = mock_assistants.describe(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


def test_update_accepts_legacy_assistant_name(mock_assistants: Assistants) -> None:
    """assistant_name= is accepted as a legacy alias for name= on update."""
    result = mock_assistants.update(assistant_name="foo", instructions="be polite")
    assert result.name == "legacy-name"  # canned response from fixture


def test_update_rejects_both(mock_assistants: Assistants) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.update(name="a", assistant_name="b")


def test_update_rejects_unknown(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.update(name="foo", bogus=1)


def test_update_missing_name_raises(mock_assistants: Assistants) -> None:
    """Calling update() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.update()


def test_update_with_name_still_works(mock_assistants: Assistants) -> None:
    """The canonical name= parameter continues to work as before."""
    result = mock_assistants.update(name="my-assistant", instructions="be precise")
    assert result.name == "legacy-name"  # canned response from fixture


def test_delete_accepts_legacy_assistant_name(mock_assistants: Assistants) -> None:
    """assistant_name= is accepted as a legacy alias for name= on delete."""
    mock_assistants.delete(assistant_name="foo", timeout=-1)
    mock_assistants._http.delete.assert_called_once_with("/assistants/foo")  # type: ignore[attr-defined]


def test_delete_rejects_both(mock_assistants: Assistants) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.delete(name="a", assistant_name="b")


def test_delete_rejects_unknown(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.delete(name="foo", bogus=1)


def test_delete_missing_name_raises(mock_assistants: Assistants) -> None:
    """Calling delete() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.delete()


def test_delete_with_name_still_works(mock_assistants: Assistants) -> None:
    """The canonical name= parameter continues to work as before."""
    mock_assistants.delete(name="my-assistant", timeout=-1)
    mock_assistants._http.delete.assert_called_once_with("/assistants/my-assistant")  # type: ignore[attr-defined]


def test_list_page_accepts_legacy_limit(mock_assistants: Assistants) -> None:
    """limit= is accepted as a legacy alias for page_size= on list_page."""
    from pinecone.models.assistant.list import ListAssistantsResponse

    mock_assistants._adapter.to_assistant_list.return_value = ListAssistantsResponse(  # type: ignore[attr-defined]
        assistants=[]
    )
    mock_assistants.list_page(limit=5)
    mock_assistants._http.get.assert_called_once_with(  # type: ignore[attr-defined]
        "/assistants", params={"limit": 5}
    )


def test_list_page_rejects_both_limit_and_page_size(mock_assistants: Assistants) -> None:
    """Passing both limit= and page_size= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.list_page(limit=5, page_size=5)


def test_list_page_rejects_unknown(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.list_page(page_size=5, bogus=1)
