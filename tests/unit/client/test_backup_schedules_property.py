"""Property tests for the backup-schedule endpoints.

The pagination fan-out property is the one the ticket names: for an arbitrary
sequence of pages ending in a ``null``/absent pagination envelope, the
auto-paginating iterators must yield every row exactly once, in order, and
must follow the token each page handed back.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.backup_schedules import BackupSchedules

BASE_URL = "https://api.test.pinecone.io"
SCHEDULE_ID = "e88f7273-42aa-47e9-af73-593827136867"

_SCHEDULE_TEMPLATE: dict[str, Any] = {
    "name": "daily-compliance-backup",
    "index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "project_id": "71ce31ea-75f7-45d6-a147-ef67f661a1b0",
    "schedule_type": "time-based",
    "frequency": "daily",
    "retention_expire_after_days": 90,
    "enabled": True,
    "next_scheduled_run": "2026-04-03T06:00:00+00:00",
    "created_at": "2026-04-02T18:22:56.712605+00:00",
}

_HISTORY_TEMPLATE: dict[str, Any] = {
    "source_index_id": "8cbf7ba6-4135-438e-a3c3-4a89a3298905",
    "source_index_name": "my-index",
    "name": "daily-compliance-backup-20260403T060000Z",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "record_count": 1,
    "namespace_count": 1,
    "size_bytes": 1,
    "created_at": "2026-04-03T06:00:00+00:00",
}

page_sizes = st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=5)

final_envelope = st.sampled_from(["absent", "null"])


def _client() -> BackupSchedules:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return BackupSchedules(http=HTTPClient(config, CONTROL_PLANE_API_VERSION))


def _paged_responses(
    sizes: list[int], *, template: dict[str, Any], id_field: str, final: str
) -> tuple[list[httpx.Response], list[str]]:
    """Build one response per page plus the flat list of ids they carry.

    Every page but the last hands back a token; the last page either omits
    the ``pagination`` key entirely or sends it as ``null`` -- both spellings
    appear in the spec's own examples, and both must terminate iteration.
    """
    responses: list[httpx.Response] = []
    expected: list[str] = []
    counter = 0
    last = len(sizes) - 1
    for index, size in enumerate(sizes):
        rows: list[dict[str, Any]] = []
        for _ in range(size):
            row_id = f"row-{counter}"
            counter += 1
            expected.append(row_id)
            rows.append({**template, id_field: row_id})
        body: dict[str, Any] = {"data": rows}
        if index != last:
            body["pagination"] = {"next": f"tok-{index + 1}"}
        elif final == "null":
            body["pagination"] = None
        responses.append(httpx.Response(200, json=body))
    return responses, expected


@given(sizes=page_sizes, final=final_envelope)
@settings(max_examples=60, deadline=None)
def test_iter_schedules_yields_every_row_exactly_once(sizes: list[int], final: str) -> None:
    responses, expected = _paged_responses(
        sizes, template=_SCHEDULE_TEMPLATE, id_field="schedule_id", final=final
    )

    with respx.mock:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            side_effect=responses
        )

        yielded = [s.schedule_id for s in _client().iter_schedules(index_name="my-index")]

    assert yielded == expected
    assert len(yielded) == len(set(yielded))
    assert route.call_count == len(sizes)


@given(sizes=page_sizes, final=final_envelope)
@settings(max_examples=60, deadline=None)
def test_iter_history_yields_every_row_exactly_once(sizes: list[int], final: str) -> None:
    responses, expected = _paged_responses(
        sizes, template=_HISTORY_TEMPLATE, id_field="backup_id", final=final
    )

    with respx.mock:
        route = respx.get(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}/history").mock(
            side_effect=responses
        )

        yielded = [r.backup_id for r in _client().iter_history(schedule_id=SCHEDULE_ID)]

    assert yielded == expected
    assert len(yielded) == len(set(yielded))
    assert route.call_count == len(sizes)


@given(sizes=page_sizes, final=final_envelope)
@settings(max_examples=60, deadline=None)
def test_each_request_carries_the_previous_pages_token(sizes: list[int], final: str) -> None:
    responses, _ = _paged_responses(
        sizes, template=_SCHEDULE_TEMPLATE, id_field="schedule_id", final=final
    )

    with respx.mock:
        route = respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            side_effect=responses
        )

        list(_client().iter_schedules(index_name="my-index"))

    assert "paginationToken" not in route.calls[0].request.url.params
    for index in range(1, len(sizes)):
        params = route.calls[index].request.url.params
        assert params["paginationToken"] == f"tok-{index}"


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=4),
    cap=st.integers(min_value=1, max_value=12),
)
@settings(max_examples=60, deadline=None)
def test_a_limit_never_yields_more_than_it_allows(sizes: list[int], cap: int) -> None:
    responses, expected = _paged_responses(
        sizes, template=_SCHEDULE_TEMPLATE, id_field="schedule_id", final="null"
    )

    with respx.mock:
        respx.get(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            side_effect=responses + [responses[-1]] * len(sizes)
        )

        yielded = [
            s.schedule_id for s in _client().iter_schedules(index_name="my-index", limit=cap)
        ]

    assert len(yielded) <= cap
    assert yielded == expected[: len(yielded)]


@given(
    frequency=st.one_of(st.none(), st.sampled_from(["daily", "weekly", "monthly"])),
    retention_days=st.one_of(st.none(), st.integers(min_value=1, max_value=3650)),
    enabled=st.one_of(st.none(), st.booleans()),
)
@settings(max_examples=100, deadline=None)
def test_the_patch_body_holds_exactly_the_fields_that_were_set(
    frequency: str | None, retention_days: int | None, enabled: bool | None
) -> None:
    """PATCH sparseness: an unset argument must never reach the wire.

    Anything that leaked would silently reset a field the caller did not
    mention, which is the one failure mode a sparse update must not have.
    """
    expected: dict[str, Any] = {}
    if frequency is not None:
        expected["frequency"] = frequency
    if retention_days is not None:
        expected["retention"] = {"expire_after_days": retention_days}
    if enabled is not None:
        expected["enabled"] = enabled

    with respx.mock:
        route = respx.patch(f"{BASE_URL}/backup-schedules/{SCHEDULE_ID}").mock(
            return_value=httpx.Response(
                200, json={**_SCHEDULE_TEMPLATE, "schedule_id": SCHEDULE_ID}
            )
        )

        _client().update(
            schedule_id=SCHEDULE_ID,
            frequency=frequency,
            retention_days=retention_days,
            enabled=enabled,
        )

    assert json.loads(route.calls.last.request.content) == expected


@given(
    name=st.text(min_size=1, max_size=40).filter(lambda s: s.strip() != ""),
    frequency=st.sampled_from(["daily", "weekly", "monthly"]),
    retention_days=st.integers(min_value=1, max_value=3650),
)
@settings(max_examples=100, deadline=None)
def test_the_create_body_is_always_the_nested_wire_shape(
    name: str, frequency: str, retention_days: int
) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes/my-index/backup-schedules").mock(
            return_value=httpx.Response(
                201, json={**_SCHEDULE_TEMPLATE, "schedule_id": SCHEDULE_ID}
            )
        )

        _client().create(
            index_name="my-index",
            name=name,
            frequency=frequency,
            retention_days=retention_days,
        )

    assert json.loads(route.calls.last.request.content) == {
        "name": name,
        "schedule": {"type": "time-based", "frequency": frequency},
        "retention": {"expire_after_days": retention_days},
    }
