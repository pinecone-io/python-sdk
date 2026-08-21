"""Executes ``docs/migration/v10-2026-07-db-data-breaking-changes.md`` (#138).

Same discipline as ``test_docs_migration_query_param_enums_371.py``: the guide's
code blocks and the error text they print are read out of the published file and
run, never transcribed here, so a transcription cannot drift from what a reader
copies.

Seven claims the guide makes and this file holds to:

1. Every ``PineconeValueError`` the guide prints is the message the SDK really
   raises, character for character, and it is raised before any request goes out.
2. Every ``# now:`` shape the guide prints is what ``SchemaBuilder.build()``
   really emits, and the paired ``# before:`` shape differs from it by exactly
   the one key the surrounding section names.
3. ``start_import`` sends no ``errorMode`` when ``error_mode`` is omitted, and
   the enum member and the string produce byte-identical bodies — the guide's
   row-8 claim that no ``db_data`` call was affected by the enum mangling.
4. ``create_namespace`` omits ``schema`` from the body when the argument is
   omitted, which is what makes "inherit" rather than "index everything" the
   observable behaviour.
5. ``describe_index_stats`` still forwards a filter unvalidated — the guide says
   no behaviour changed there, only the documentation.
6. Both operation families in the opening table name methods that really exist,
   and the document family really is absent from ``GrpcIndex``.
7. The guide does not describe the vector operations as deprecated or removed.
   That framing is the resolution of #322 and it is the thing on this page most
   expensive to get wrong, so it is pinned rather than trusted.

The server wording quoted for a vectors-API write against a document-schema
index comes from the backend, not from this SDK; its citation is in the PR body.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncIndex, GrpcIndex, Index, SchemaBuilder
from pinecone._internal.validation import QUERY_TOP_K_MAX
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.imports.error_mode import ImportErrorMode

GUIDE = (
    Path(__file__).resolve().parents[2] / "docs/migration/v10-2026-07-db-data-breaking-changes.md"
)
TEXT = GUIDE.read_text()

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"


def _blocks() -> list[str]:
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", TEXT, re.DOTALL)]
    assert sources, f"no python blocks found in {GUIDE}"
    return sources


def _block(needle: str) -> str:
    matches = [s for s in _blocks() if needle in s]
    assert len(matches) == 1, f"expected one block containing {needle!r}, got {len(matches)}"
    return matches[0]


def _printed_error(source: str) -> str:
    """The ``# PineconeValueError: ...`` message a block prints, unwrapped.

    The guide wraps a long message across several comment lines, so continuation
    lines are rejoined with single spaces to recover the one-line message.
    """
    lines = source.strip().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# PineconeValueError:"))
    parts = [lines[start].split(":", 1)[1].strip()]
    for line in lines[start + 1 :]:
        if not line.startswith("#"):
            break
        parts.append(line.lstrip("#").strip())
    return " ".join(parts)


def _code(source: str) -> str:
    """A block with its comment lines stripped, so it can be executed."""
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))


def _shape(source: str, label: str) -> dict[str, Any]:
    """The dict literal a ``# before:`` or ``# now:`` comment line carries."""
    match = re.search(rf"^#\s*{label}:\s*(\{{.*\}})\s*$", source, re.MULTILINE)
    assert match, f"no '# {label}:' shape in block:\n{source}"
    return ast.literal_eval(match.group(1))


def _index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _normalized() -> str:
    """Whitespace, smart quotes and dashes flattened, so a reflow cannot hide a phrase."""
    quotes = dict.fromkeys("\u2018\u2019", "'") | dict.fromkeys("\u201c\u201d", '"')
    dashes = dict.fromkeys("\u2013\u2014\u2212", "-")
    return re.sub(r"\s+", " ", TEXT.translate(str.maketrans(quotes | dashes)))


def _table_row_methods(row_prefix: str) -> list[str]:
    line = next(line for line in TEXT.splitlines() if line.startswith(row_prefix))
    return re.findall(r"`(\w+)`", line.split("|")[2])


@respx.mock
def test_the_top_k_block_raises_the_message_the_guide_prints() -> None:
    source = _block("top_k=20000")
    route = respx.post(f"{BASE_URL}/query").mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(PineconeValueError) as excinfo:
        exec(_code(source), {"idx": _index()})  # noqa: S102

    assert str(excinfo.value) == _printed_error(source)
    assert not route.calls, "the guide says the call fails locally, so nothing may be sent"


def test_the_sparse_metric_block_raises_the_message_the_guide_prints() -> None:
    source = _block('add_sparse_vector_field("sparse_terms", metric=')

    with pytest.raises(PineconeValueError) as excinfo:
        exec(_code(source), {"SchemaBuilder": SchemaBuilder})  # noqa: S102

    assert str(excinfo.value) == _printed_error(source)


@pytest.mark.parametrize("top_k", [1, QUERY_TOP_K_MAX])
@respx.mock
def test_both_ends_of_the_range_the_guide_names_are_accepted(top_k: int) -> None:
    route = respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json={"matches": [], "namespace": "", "usage": {}})
    )
    _index().query(vector=[0.1, 0.2], top_k=top_k, namespace="movies-en")
    assert json.loads(route.calls.last.request.content)["topK"] == top_k


@pytest.mark.parametrize(
    "module", ["pinecone.index", "pinecone.async_client.async_index", "pinecone.grpc"]
)
def test_the_bound_the_guide_names_is_the_one_all_three_lanes_use(module: str) -> None:
    assert f"between 1 and {QUERY_TOP_K_MAX}" in _printed_error(_block("top_k=20000"))
    source = Path(__import__(module, fromlist=["__file__"]).__file__ or "").read_text()
    assert "QUERY_TOP_K_MAX" in source


@pytest.mark.parametrize(
    ("needle", "delta_key"),
    [("add_boolean_field", "filterable"), ('add_sparse_vector_field("sparse_terms")', "metric")],
    ids=["filterable", "sparse-metric"],
)
def test_the_now_shape_is_what_the_builder_really_emits(needle: str, delta_key: str) -> None:
    source = _block(needle)
    namespace: dict[str, Any] = {"SchemaBuilder": SchemaBuilder}
    body, expression = _code(source).rstrip().rsplit("\n", 1)
    exec(body, namespace)  # noqa: S102

    assert eval(expression, namespace) == _shape(source, "now")  # noqa: S307


@pytest.mark.parametrize(
    ("needle", "delta_key"),
    [("add_boolean_field", "filterable"), ('add_sparse_vector_field("sparse_terms")', "metric")],
    ids=["filterable", "sparse-metric"],
)
def test_the_before_shape_differs_by_exactly_the_key_the_section_names(
    needle: str, delta_key: str
) -> None:
    source = _block(needle)
    before, now = _shape(source, "before"), _shape(source, "now")

    assert before != now
    assert set(before) ^ set(now) == {delta_key}
    assert {k: v for k, v in before.items() if k != delta_key} == {
        k: v for k, v in now.items() if k != delta_key
    }


def test_the_builder_still_emits_filterable_when_it_is_true() -> None:
    """The guide says ``filterable=True`` was unaffected."""
    field = SchemaBuilder().add_boolean_field("b", filterable=True).build()["fields"]["b"]
    assert field == {"type": "boolean", "filterable": True}


@respx.mock
def test_the_start_import_block_sends_what_the_guide_describes() -> None:
    source = _block("start_import")
    route = respx.post(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json={"id": "import-1"})
    )
    exec(_code(source), {"idx": _index()})  # noqa: S102

    default_body, opted_in_body = (json.loads(call.request.content) for call in route.calls)
    assert "errorMode" not in default_body, "the omitted form must not pin a mode client-side"
    assert opted_in_body["errorMode"] == {"onError": "continue"}


@respx.mock
def test_the_enum_member_and_the_string_produce_identical_bodies() -> None:
    """Row 8: ``db_data``'s one enum-valued argument was never mangled."""
    route = respx.post(f"{BASE_URL}/bulk/imports").mock(
        return_value=httpx.Response(200, json={"id": "import-1"})
    )
    idx = _index()
    idx.start_import(uri="s3://b/v/", error_mode="continue")
    from_string = route.calls.last.request.content
    idx.start_import(uri="s3://b/v/", error_mode=ImportErrorMode.CONTINUE)

    assert route.calls.last.request.content == from_string
    assert b"ImportErrorMode" not in from_string


def test_a_db_data_query_parameter_carrying_an_enum_would_be_resolved() -> None:
    """The other half of row 8: the encoder fix covers parameters added later."""
    from pinecone._internal.http_client import _prepare_params

    params = _prepare_params({"params": {"errorMode": ImportErrorMode.CONTINUE}})
    assert str(httpx.QueryParams(params["params"])) == "errorMode=continue"


@respx.mock
def test_the_create_namespace_block_sends_what_the_guide_describes() -> None:
    source = _block("create_namespace")
    route = respx.post(f"{BASE_URL}/namespaces").mock(
        return_value=httpx.Response(
            200, json={"name": "movies-en", "record_count": 0, "size_bytes": 0}
        )
    )
    exec(_code(source), {"idx": _index()})  # noqa: S102

    inherited, overridden = (json.loads(call.request.content) for call in route.calls)
    assert "schema" not in inherited, "the inheriting form must send no schema at all"
    assert overridden["schema"] == {"fields": {"genre": {"filterable": True}}}


@respx.mock
def test_describe_index_stats_still_forwards_a_filter_unvalidated() -> None:
    """The guide says row 2 changed documentation, not behaviour."""
    source = _block("describe_index_stats(filter=")
    route = respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
    )
    exec(_code(source), {"idx": _index()})  # noqa: S102

    assert json.loads(route.calls.last.request.content)["filter"] == {"genre": {"$eq": "action"}}


@respx.mock
def test_the_unfiltered_block_the_guide_recommends_runs() -> None:
    source = _block("stats.total_vector_count")
    route = respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(
            200, json={"namespaces": {}, "totalVectorCount": 7, "dimension": 3}
        )
    )
    exec(_code(source), {"idx": _index()})  # noqa: S102

    assert json.loads(route.calls.last.request.content) == {}


def test_the_vector_family_row_names_methods_that_exist_on_every_lane() -> None:
    names = _table_row_methods("| Created under an API version earlier")
    assert names, "the vector-family row lists no methods"
    for name in names:
        for cls in (Index, AsyncIndex, GrpcIndex):
            assert callable(getattr(cls, name, None)), f"{cls.__name__} has no {name}"


def test_the_document_family_row_names_rest_only_methods() -> None:
    names = _table_row_methods("| Created with `2026-07`")
    assert names, "the document-family row lists no methods"
    for name in names:
        assert callable(getattr(Index, name, None)), f"Index has no {name}"
        assert callable(getattr(AsyncIndex, name, None)), f"AsyncIndex has no {name}"
        assert not hasattr(GrpcIndex, name), f"the guide says gRPC serves no documents, but {name}"


def test_the_guide_states_the_vector_operations_are_kept_on_purpose() -> None:
    normalized = _normalized()
    assert "serve indexes created under earlier API versions, and they are meant to" in normalized
    assert "None of them is deprecated, none is scheduled for removal" in normalized
    assert "deliberately provides no way to create an index for the vector operations" in normalized


@pytest.mark.parametrize(
    "forbidden",
    [
        "vector operations are deprecated",
        "vector operations will be removed",
        "vector operations are going away",
        "vector operations are being removed",
        "vector api is deprecated",
    ],
)
def test_the_guide_never_calls_the_vector_operations_deprecated(forbidden: str) -> None:
    assert forbidden not in _normalized().lower()


def test_every_relative_link_the_guide_makes_resolves() -> None:
    """MyST link syntax with a live target — #402 is rST syntax in a MyST file."""
    targets = set(re.findall(r"\]\((v10-[\w.-]+\.md)(?:#[\w-]+)?\)", TEXT))
    assert targets, "the guide claims to defer to other guides but links none"
    for target in targets:
        assert (GUIDE.parent / target).is_file(), f"dead link: {target}"
