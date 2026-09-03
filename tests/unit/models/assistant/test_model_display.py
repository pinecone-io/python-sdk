from __future__ import annotations

import pytest

from pinecone.models.assistant.model import AssistantModel


def make_full() -> AssistantModel:
    return AssistantModel(
        name="my-assistant",
        status="Ready",
        metadata={"k1": "v1", "k2": "v2"},
        instructions="Be helpful.",
        host="https://x.pinecone.io",
        region="eu",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )


def make_minimal() -> AssistantModel:
    return AssistantModel(name="a", status="Initializing")


class TestRepr:
    def test_populated(self) -> None:
        r = repr(make_full())
        assert isinstance(r, str)
        assert "my-assistant" in r

    def test_region_shown(self) -> None:
        assert "region='eu'" in repr(make_full())

    def test_region_absent_when_none(self) -> None:
        assert "region" not in repr(make_minimal())

    def test_all_optional_none(self) -> None:
        r = repr(make_minimal())
        assert isinstance(r, str)
        assert "None" not in r

    def test_large_instructions_truncated(self) -> None:
        m = AssistantModel(name="a", status="Ready", instructions="x" * 5000)
        r = repr(m)
        assert len(r) < 500
        assert "..." in r

    def test_large_metadata_truncated(self) -> None:
        m = AssistantModel(name="a", status="Ready", metadata={f"k{i}": i for i in range(100)})
        r = repr(m)
        assert len(r) < 500

    def test_safe_on_malformed(self) -> None:
        m = make_minimal()
        m.name = None
        r = repr(m)
        assert isinstance(r, str)


class TestReprHtml:
    def test_populated(self) -> None:
        h = make_full()._repr_html_()
        assert "<div" in h
        assert "my-assistant" in h

    def test_region_row(self) -> None:
        h = make_full()._repr_html_()
        assert "Region:" in h
        assert "eu" in h

    def test_optional_none(self) -> None:
        h = make_minimal()._repr_html_()
        assert "<div" in h
        assert "Region:" not in h

    def test_large_metadata_truncated(self) -> None:
        m = AssistantModel(name="a", status="Ready", metadata={f"k{i}": i for i in range(100)})
        h = m._repr_html_()
        assert "more" in h

    def test_safe_on_malformed(self) -> None:
        m = make_minimal()
        m.name = object()
        h = m._repr_html_()
        assert isinstance(h, str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "my-assistant" in pretty(make_full())

    def test_cycle(self) -> None:
        from IPython.lib.pretty import pretty

        assert isinstance(pretty(make_minimal()), str)


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_display_never_raises(method: str) -> None:
    m = make_full()
    assert isinstance(getattr(m, method)(), str)
