"""Tests for to_dict() on assistant context and evaluation model classes."""

from __future__ import annotations

from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.context import (
    ContextResponse,
    FileReference,
    TextSnippet,
)
from pinecone.models.assistant.evaluation import (
    AlignmentResult,
    AlignmentScores,
    EntailmentResult,
)
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import (
    ListAssistantsResponse,
    ListFilesResponse,
    ListOperationsResponse,
    _Pagination,
)
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel


def _usage() -> ChatUsage:
    return ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _file() -> AssistantFileModel:
    return AssistantFileModel(id="file-1", name="doc.txt", status="Available", size=100)


def _file_ref() -> FileReference:
    return FileReference(file=_file())


def _text_snippet() -> TextSnippet:
    return TextSnippet(content="hello world", score=0.9, reference=_file_ref())


def _context_response() -> ContextResponse:
    return ContextResponse(snippets=[_text_snippet()], usage=_usage())


def _operation() -> OperationModel:
    return OperationModel(
        operation_id="op-1",
        status="Completed",
        operation_type="upload_file",
        file_id="file-1",
        created_at="2026-07-01T12:30:00Z",
        completed_on="2026-07-01T12:35:00Z",
        percent_complete=100,
        ingestion_units=50.0,
    )


class TestContextResponseToDict:
    def test_context_response_to_dict(self) -> None:
        result = _context_response().to_dict()
        assert isinstance(result, dict)
        assert "snippets" in result
        assert "usage" in result
        assert "id" in result

    def test_context_response_nested_snippets(self) -> None:
        resp = ContextResponse(snippets=[_text_snippet()], usage=_usage())
        result = resp.to_dict()
        assert isinstance(result["snippets"][0], dict)
        assert not isinstance(result["snippets"][0], TextSnippet)
        assert result["snippets"][0]["content"] == "hello world"
        assert result["snippets"][0]["score"] == 0.9

    def test_context_response_snippet_reference_nested(self) -> None:
        resp = ContextResponse(snippets=[_text_snippet()], usage=_usage())
        result = resp.to_dict()
        ref = result["snippets"][0]["reference"]
        assert isinstance(ref, dict)
        assert not isinstance(ref, FileReference)
        assert ref["file"]["name"] == "doc.txt"

    def test_context_response_usage_nested(self) -> None:
        resp = ContextResponse(snippets=[], usage=_usage())
        result = resp.to_dict()
        assert isinstance(result["usage"], dict)
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["total_tokens"] == 15

    def test_context_response_id_none(self) -> None:
        resp = ContextResponse(snippets=[], usage=_usage())
        result = resp.to_dict()
        assert result["id"] is None

    def test_context_response_id_populated(self) -> None:
        resp = ContextResponse(snippets=[], usage=_usage(), id="ctx-123")
        result = resp.to_dict()
        assert result["id"] == "ctx-123"


class TestAlignmentResultToDict:
    def test_alignment_result_to_dict(self) -> None:
        result = AlignmentResult(
            scores=AlignmentScores(correctness=0.8, completeness=0.9, alignment=0.85),
            facts=[EntailmentResult(fact="sky is blue", entailment="entailed")],
            usage=_usage(),
        ).to_dict()
        assert isinstance(result, dict)
        assert "scores" in result
        assert "facts" in result
        assert "usage" in result

    def test_alignment_result_nested_entailment(self) -> None:
        ar = AlignmentResult(
            scores=AlignmentScores(correctness=0.8, completeness=0.9, alignment=0.85),
            facts=[
                EntailmentResult(fact="fact1", entailment="entailed"),
                EntailmentResult(fact="fact2", entailment="contradicted", reasoning="wrong"),
            ],
            usage=_usage(),
        )
        result = ar.to_dict()
        for item in result["facts"]:
            assert isinstance(item, dict)
            assert not isinstance(item, EntailmentResult)
        assert result["facts"][0]["fact"] == "fact1"
        assert result["facts"][0]["entailment"] == "entailed"
        assert result["facts"][1]["reasoning"] == "wrong"

    def test_alignment_result_scores_nested(self) -> None:
        ar = AlignmentResult(
            scores=AlignmentScores(correctness=0.7, completeness=0.8, alignment=0.75),
            facts=[],
            usage=_usage(),
        )
        result = ar.to_dict()
        assert isinstance(result["scores"], dict)
        assert not isinstance(result["scores"], AlignmentScores)
        assert result["scores"]["correctness"] == 0.7

    def test_alignment_result_empty_facts(self) -> None:
        ar = AlignmentResult(
            scores=AlignmentScores(correctness=1.0, completeness=1.0, alignment=1.0),
            facts=[],
            usage=_usage(),
        )
        result = ar.to_dict()
        assert result["facts"] == []


class TestListAssistantsResponseToDict:
    def test_list_assistants_response_to_dict(self) -> None:
        resp = ListAssistantsResponse(assistants=[])
        result = resp.to_dict()
        assert isinstance(result, dict)
        assert "assistants" in result
        assert "pagination" in result
        assert result["assistants"] == []
        assert result["pagination"] is None

    def test_list_assistants_response_nested_models(self) -> None:
        asst = AssistantModel(name="my-asst", status="Ready")
        resp = ListAssistantsResponse(assistants=[asst], pagination=_Pagination(next="tok"))
        result = resp.to_dict()
        assert isinstance(result["assistants"][0], dict)
        assert not isinstance(result["assistants"][0], AssistantModel)
        assert result["assistants"][0]["name"] == "my-asst"
        assert result["pagination"]["next"] == "tok"


class TestListFilesResponseToDict:
    def test_list_files_response_to_dict(self) -> None:
        resp = ListFilesResponse(files=[])
        result = resp.to_dict()
        assert isinstance(result, dict)
        assert "files" in result
        assert "pagination" in result
        assert result["files"] == []
        assert result["pagination"] is None

    def test_list_files_response_nested_files(self) -> None:
        resp = ListFilesResponse(files=[_file()], pagination=_Pagination(next="page2"))
        result = resp.to_dict()
        assert isinstance(result["files"][0], dict)
        assert not isinstance(result["files"][0], AssistantFileModel)
        assert result["files"][0]["name"] == "doc.txt"
        assert result["pagination"]["next"] == "page2"


class TestOperationModelToDict:
    def test_operation_to_dict_uses_python_field_names(self) -> None:
        result = _operation().to_dict()
        assert result == {
            "operation_id": "op-1",
            "status": "Completed",
            "operation_type": "upload_file",
            "file_id": "file-1",
            "created_at": "2026-07-01T12:30:00Z",
            "completed_on": "2026-07-01T12:35:00Z",
            "percent_complete": 100,
            "error": None,
            "ingestion_units": 50.0,
        }

    def test_operation_to_dict_with_all_optional_fields_absent(self) -> None:
        result = OperationModel(operation_id="op-2", status="Processing").to_dict()
        assert result == {
            "operation_id": "op-2",
            "status": "Processing",
            "operation_type": None,
            "file_id": None,
            "created_at": None,
            "completed_on": None,
            "percent_complete": None,
            "error": None,
            "ingestion_units": None,
        }


class TestListOperationsResponseToDict:
    def test_list_operations_response_to_dict(self) -> None:
        resp = ListOperationsResponse(operations=[])
        result = resp.to_dict()
        assert isinstance(result, dict)
        assert "operations" in result
        assert "pagination" in result
        assert result["operations"] == []
        assert result["pagination"] is None

    def test_list_operations_response_nested_operations(self) -> None:
        resp = ListOperationsResponse(
            operations=[_operation()], pagination=_Pagination(next="ops2")
        )
        result = resp.to_dict()
        assert isinstance(result["operations"][0], dict)
        assert not isinstance(result["operations"][0], OperationModel)
        assert result["operations"][0]["operation_id"] == "op-1"
        assert result["pagination"]["next"] == "ops2"


class TestToDictIsPureRead:
    def test_operation_to_dict_is_pure_read(self) -> None:
        op = _operation()
        result = op.to_dict()
        result["status"] = "mutated"
        assert op.status == "Completed"

    def test_context_response_to_dict_is_pure_read(self) -> None:
        resp = ContextResponse(snippets=[_text_snippet()], usage=_usage())
        result = resp.to_dict()
        result["id"] = "mutated"
        assert resp.id is None

    def test_alignment_result_to_dict_is_pure_read(self) -> None:
        ar = AlignmentResult(
            scores=AlignmentScores(correctness=0.8, completeness=0.9, alignment=0.85),
            facts=[EntailmentResult(fact="fact", entailment="entailed")],
            usage=_usage(),
        )
        result = ar.to_dict()
        result["facts"] = []
        assert len(ar.facts) == 1
