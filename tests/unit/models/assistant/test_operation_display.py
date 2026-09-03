from __future__ import annotations

import pytest

from pinecone.models.assistant.operation import OperationModel


def make_ok() -> OperationModel:
    return OperationModel(
        operation_id="op-1",
        status="Completed",
        operation_type="upload_file",
        file_id="file-1",
        created_at="2026-01-01T00:00:00Z",
        completed_on="2026-01-01T00:05:00Z",
        percent_complete=100,
        ingestion_units=50.0,
    )


def make_failed() -> OperationModel:
    return OperationModel(
        operation_id="op-2",
        status="Failed",
        operation_type="upsert_file",
        percent_complete=0,
        error="boom " * 100,
    )


def make_minimal() -> OperationModel:
    return OperationModel(operation_id="op-3", status="Processing")


class TestRepr:
    def test_ok(self) -> None:
        assert "op-1" in repr(make_ok())

    def test_ok_includes_progress_fields(self) -> None:
        r = repr(make_ok())
        assert "upload_file" in r
        assert "file-1" in r
        assert "percent_complete=100" in r

    def test_failed_includes_error(self) -> None:
        r = repr(make_failed())
        assert "Failed" in r
        assert len(r) < 500

    def test_minimal(self) -> None:
        r = repr(make_minimal())
        assert "None" not in r

    def test_safe_on_malformed(self) -> None:
        m = make_ok()
        m.status = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)


class TestReprHtml:
    def test_ok(self) -> None:
        assert "op-1" in make_ok()._repr_html_()

    def test_ok_shows_progress_rows(self) -> None:
        h = make_ok()._repr_html_()
        assert "upload_file" in h
        assert "file-1" in h
        assert "100%" in h
        assert "Ingestion units" in h

    def test_minimal_omits_absent_rows(self) -> None:
        h = make_minimal()._repr_html_()
        assert "Progress" not in h
        assert "File ID" not in h

    def test_failed_has_error_section(self) -> None:
        assert "#991b1b" in make_failed()._repr_html_()

    def test_minimal(self) -> None:
        assert "<div" in make_minimal()._repr_html_()

    def test_safe_on_malformed(self) -> None:
        m = make_ok()
        m.operation_id = object()  # type: ignore[assignment]
        assert isinstance(m._repr_html_(), str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "op-1" in pretty(make_ok())

    def test_populated_includes_new_fields(self) -> None:
        from IPython.lib.pretty import pretty

        rendered = pretty(make_ok())
        assert "operation_type" in rendered
        assert "completed_on" in rendered
        assert "ingestion_units" in rendered

    def test_minimal_omits_absent_fields(self) -> None:
        from IPython.lib.pretty import pretty

        rendered = pretty(make_minimal())
        assert "operation_type" not in rendered
        assert "percent_complete" not in rendered


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_never_raises(method: str) -> None:
    assert isinstance(getattr(make_failed(), method)(), str)
