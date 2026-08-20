"""Unit tests for the assistant list response models."""

from __future__ import annotations

from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import (
    ListAssistantsResponse,
    ListFilesResponse,
    ListOperationsResponse,
    _Pagination,
)
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant(name: str = "test-asst") -> AssistantModel:
    return AssistantModel(name=name, status="Ready")


def _make_file(file_id: str = "file-abc") -> AssistantFileModel:
    return AssistantFileModel(id=file_id, name="test.txt", status="Available", size=0)


def _make_operation(operation_id: str = "op-abc") -> OperationModel:
    return OperationModel(
        operation_id=operation_id, status="Processing", operation_type="upload_file"
    )


# ---------------------------------------------------------------------------
# ListAssistantsResponse — next_token alias
# ---------------------------------------------------------------------------


def test_list_assistants_response_next_token_alias_none() -> None:
    """next_token returns None when pagination is absent."""
    resp = ListAssistantsResponse(assistants=[])
    assert resp.next_token is None
    assert resp.next_token == resp.next


def test_list_assistants_response_next_token_alias_with_value() -> None:
    """next_token returns the same string as next when pagination is present."""
    token = "some-pagination-token"
    resp = ListAssistantsResponse(assistants=[], pagination=_Pagination(next=token))
    assert resp.next_token == token
    assert resp.next_token == resp.next


def test_list_assistants_response_next_token_alias_with_assistants() -> None:
    """next_token alias works when assistants list is non-empty."""
    assistants = [_make_assistant("a1"), _make_assistant("a2")]
    resp = ListAssistantsResponse(assistants=assistants, pagination=_Pagination(next="tok-2"))
    assert resp.next_token == "tok-2"
    assert resp.next_token == resp.next


# ---------------------------------------------------------------------------
# ListFilesResponse — next_token alias
# ---------------------------------------------------------------------------


def test_list_files_response_next_token_alias_none() -> None:
    """next_token returns None when pagination is absent."""
    resp = ListFilesResponse(files=[])
    assert resp.next_token is None
    assert resp.next_token == resp.next


def test_list_files_response_next_token_alias_with_value() -> None:
    """next_token returns the same string as next when pagination is present."""
    token = "files-pagination-token"
    resp = ListFilesResponse(files=[], pagination=_Pagination(next=token))
    assert resp.next_token == token
    assert resp.next_token == resp.next


def test_list_files_response_next_token_alias_with_files() -> None:
    """next_token alias works when files list is non-empty."""
    files = [_make_file("id-1"), _make_file("id-2")]
    resp = ListFilesResponse(files=files, pagination=_Pagination(next="tok-files-2"))
    assert resp.next_token == "tok-files-2"
    assert resp.next_token == resp.next


# ---------------------------------------------------------------------------
# ListOperationsResponse — next_token alias
# ---------------------------------------------------------------------------


def test_list_operations_response_next_token_alias_none() -> None:
    """next_token returns None when pagination is absent."""
    resp = ListOperationsResponse(operations=[])
    assert resp.next_token is None
    assert resp.next_token == resp.next


def test_list_operations_response_next_token_alias_with_value() -> None:
    """next_token returns the same string as next when pagination is present."""
    token = "operations-pagination-token"
    resp = ListOperationsResponse(operations=[], pagination=_Pagination(next=token))
    assert resp.next_token == token
    assert resp.next_token == resp.next


def test_list_operations_response_next_token_alias_with_operations() -> None:
    """next_token alias works when operations list is non-empty."""
    operations = [_make_operation("op-1"), _make_operation("op-2")]
    resp = ListOperationsResponse(operations=operations, pagination=_Pagination(next="tok-ops-2"))
    assert resp.next_token == "tok-ops-2"
    assert resp.next_token == resp.next
