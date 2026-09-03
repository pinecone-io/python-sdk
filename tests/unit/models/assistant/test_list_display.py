from __future__ import annotations

import builtins
from unittest.mock import patch

from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import (
    ListAssistantsResponse,
    ListFilesResponse,
    ListOperationsResponse,
    _Pagination,
)
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel


def mk_assist(n: int) -> list[AssistantModel]:
    return [AssistantModel(name=f"a{i}", status="Ready") for i in range(n)]


class TestListAssistantsRepr:
    def test_empty(self) -> None:
        r = repr(ListAssistantsResponse(assistants=[]))
        assert "count=0" in r

    def test_populated(self) -> None:
        r = repr(ListAssistantsResponse(assistants=mk_assist(3)))
        assert "count=3" in r

    def test_large(self) -> None:
        r = repr(ListAssistantsResponse(assistants=mk_assist(500)))
        assert len(r) < 500
        assert "count=500" in r

    def test_with_next_token(self) -> None:
        assert "tok" in repr(
            ListAssistantsResponse(assistants=[], pagination=_Pagination(next="tok"))
        )

    def test_safe_on_malformed(self) -> None:
        m = ListAssistantsResponse(assistants=mk_assist(1))
        with patch.object(builtins, "len", side_effect=RuntimeError("corrupted")):
            result = repr(m)
        assert isinstance(result, str)


class TestListAssistantsHtml:
    def test_populated(self) -> None:
        h = ListAssistantsResponse(assistants=mk_assist(3))._repr_html_()
        assert "<div" in h
        assert "a0" in h

    def test_large_shows_overflow(self) -> None:
        h = ListAssistantsResponse(assistants=mk_assist(500))._repr_html_()
        assert "more" in h
        assert len(h) < 10_000

    def test_empty(self) -> None:
        assert "<div" in ListAssistantsResponse(assistants=[])._repr_html_()

    def test_safe_on_malformed(self) -> None:
        m = ListAssistantsResponse(assistants=mk_assist(1))
        with patch("pinecone.models.assistant.list.HtmlBuilder", side_effect=RuntimeError("error")):
            result = m._repr_html_()
        assert isinstance(result, str)


class TestListAssistantsPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        result = pretty(ListAssistantsResponse(assistants=mk_assist(3)))
        assert "count" in result.lower() or "3" in result


def mk_files(n: int) -> list[AssistantFileModel]:
    return [
        AssistantFileModel(name=f"f{i}.txt", id=f"id-{i}", status="Available") for i in range(n)
    ]


class TestListFilesRepr:
    def test_empty(self) -> None:
        assert "count=0" in repr(ListFilesResponse(files=[]))

    def test_populated(self) -> None:
        assert "count=3" in repr(ListFilesResponse(files=mk_files(3)))

    def test_large(self) -> None:
        r = repr(ListFilesResponse(files=mk_files(1000)))
        assert len(r) < 500

    def test_with_next_token(self) -> None:
        assert "tok" in repr(ListFilesResponse(files=[], pagination=_Pagination(next="tok")))

    def test_safe_on_malformed(self) -> None:
        m = ListFilesResponse(files=mk_files(1))
        with patch.object(builtins, "len", side_effect=RuntimeError("corrupted")):
            result = repr(m)
        assert isinstance(result, str)


class TestListFilesHtml:
    def test_populated(self) -> None:
        h = ListFilesResponse(files=mk_files(3))._repr_html_()
        assert "f0.txt" in h

    def test_large_shows_overflow(self) -> None:
        h = ListFilesResponse(files=mk_files(1000))._repr_html_()
        assert "more" in h
        assert len(h) < 10_000

    def test_empty(self) -> None:
        assert "<div" in ListFilesResponse(files=[])._repr_html_()


class TestListFilesPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert isinstance(pretty(ListFilesResponse(files=mk_files(3))), str)


def mk_ops(n: int) -> list[OperationModel]:
    return [
        OperationModel(
            operation_id=f"op-{i}",
            status="Processing",
            operation_type="upload_file",
            percent_complete=i,
        )
        for i in range(n)
    ]


class TestListOperationsRepr:
    def test_empty(self) -> None:
        assert "count=0" in repr(ListOperationsResponse(operations=[]))

    def test_populated(self) -> None:
        assert "count=3" in repr(ListOperationsResponse(operations=mk_ops(3)))

    def test_large(self) -> None:
        r = repr(ListOperationsResponse(operations=mk_ops(1000)))
        assert len(r) < 500
        assert "count=1000" in r

    def test_with_next_token(self) -> None:
        assert "tok" in repr(
            ListOperationsResponse(operations=[], pagination=_Pagination(next="tok"))
        )

    def test_safe_on_malformed(self) -> None:
        m = ListOperationsResponse(operations=mk_ops(1))
        with patch.object(builtins, "len", side_effect=RuntimeError("corrupted")):
            result = repr(m)
        assert isinstance(result, str)


class TestListOperationsHtml:
    def test_populated(self) -> None:
        h = ListOperationsResponse(operations=mk_ops(3))._repr_html_()
        assert "op-0" in h
        assert "Operations" in h

    def test_large_shows_overflow(self) -> None:
        h = ListOperationsResponse(operations=mk_ops(1000))._repr_html_()
        assert "more" in h
        assert len(h) < 10_000

    def test_empty(self) -> None:
        assert "<div" in ListOperationsResponse(operations=[])._repr_html_()

    def test_safe_on_malformed(self) -> None:
        m = ListOperationsResponse(operations=mk_ops(1))
        with patch("pinecone.models.assistant.list.HtmlBuilder", side_effect=RuntimeError("error")):
            result = m._repr_html_()
        assert isinstance(result, str)


class TestListOperationsPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        rendered = pretty(ListOperationsResponse(operations=mk_ops(3)))
        assert "count=3" in rendered
        assert "op-0" in rendered
