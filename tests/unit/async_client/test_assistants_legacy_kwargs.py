"""Unit tests for legacy kwarg aliases on AsyncAssistants.create."""

from __future__ import annotations

import pytest

from pinecone.async_client.assistants import AsyncAssistants
from pinecone.errors.exceptions import PineconeValueError


@pytest.mark.asyncio
async def test_async_create_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async create."""
    result = await mock_async_assistants.create(assistant_name="legacy-name")
    assert result.name == "legacy-name"


@pytest.mark.asyncio
async def test_async_create_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.create(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_create_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.create(name="foo", bogus_param=1)


@pytest.mark.asyncio
async def test_async_create_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling create() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.create()


@pytest.mark.asyncio
async def test_async_create_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    result = await mock_async_assistants.create(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_describe_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async describe."""
    result = await mock_async_assistants.describe(assistant_name="foo")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_describe_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.describe(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_describe_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.describe(name="foo", random=1)


@pytest.mark.asyncio
async def test_async_describe_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling describe() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.describe()


@pytest.mark.asyncio
async def test_async_describe_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    result = await mock_async_assistants.describe(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_update_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling update() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.update()


@pytest.mark.asyncio
async def test_async_update_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    result = await mock_async_assistants.update(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_update_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async update."""
    result = await mock_async_assistants.update(assistant_name="foo", instructions="be precise")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_update_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.update(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_update_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.update(name="foo", bogus=1)


@pytest.mark.asyncio
async def test_async_delete_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling delete() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.delete()


@pytest.mark.asyncio
async def test_async_delete_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    await mock_async_assistants.delete(name="my-assistant", timeout=-1)
    mock_async_assistants._http.delete.assert_called_once_with("/assistants/my-assistant")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_delete_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async delete."""
    await mock_async_assistants.delete(assistant_name="foo", timeout=-1)


@pytest.mark.asyncio
async def test_async_delete_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.delete(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_delete_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.delete(name="foo", bogus=1)


@pytest.mark.asyncio
async def test_async_list_page_accepts_legacy_limit(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """limit= is accepted as a legacy alias for page_size= on async list_page."""
    from pinecone.models.assistant.list import ListAssistantsResponse

    mock_async_assistants._adapter.to_assistant_list.return_value = (  # type: ignore[attr-defined]
        ListAssistantsResponse(assistants=[])
    )
    await mock_async_assistants.list_page(limit=5)
    mock_async_assistants._http.get.assert_called_once_with(  # type: ignore[attr-defined]
        "/assistants", params={"limit": 5}
    )


@pytest.mark.asyncio
async def test_async_list_page_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both limit= and page_size= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.list_page(limit=5, page_size=5)


@pytest.mark.asyncio
async def test_async_list_page_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.list_page(page_size=5, bogus=1)


@pytest.mark.asyncio
async def test_async_list_files_page_accepts_legacy_limit(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """limit= is accepted as a legacy alias for page_size= on async list_files_page."""
    from unittest.mock import AsyncMock, MagicMock

    from pinecone.models.assistant.list import ListFilesResponse

    mock_data_http = AsyncMock()
    mock_data_response = MagicMock()
    mock_data_response.content = b"{}"
    mock_data_http.get.return_value = mock_data_response
    mock_async_assistants._data_plane_http = AsyncMock(return_value=mock_data_http)  # type: ignore[method-assign]
    mock_async_assistants._adapter.to_file_list.return_value = ListFilesResponse(  # type: ignore[attr-defined]
        files=[]
    )

    await mock_async_assistants.list_files_page(assistant_name="foo", limit=5)

    mock_data_http.get.assert_called_once_with("/files/foo", params={"limit": 5})
