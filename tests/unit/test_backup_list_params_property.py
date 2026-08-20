"""Property-based tests for the list-backups query string.

Pins the mapping every backup listing depends on: SDK keyword ->
wire parameter name. ``limit`` and ``paginationToken`` differ in case
convention, ``include_deleted`` is a lowercase JSON-style boolean, and
``None`` must never reach the wire as the string ``"None"``.
"""

from __future__ import annotations

import httpx
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.backups_helpers import backup_list_params

_limit = st.one_of(st.none(), st.integers(min_value=1, max_value=100))
_token = st.one_of(st.none(), st.text(min_size=1, max_size=40))
_include_deleted = st.one_of(st.none(), st.booleans())


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_only_the_three_documented_keys_are_ever_emitted(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )

    assert set(params) <= {"limit", "paginationToken", "include_deleted"}


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_none_is_omitted_and_set_values_are_present(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )

    assert ("limit" in params) is (limit is not None)
    assert ("paginationToken" in params) is (pagination_token is not None)
    assert ("include_deleted" in params) is (include_deleted is not None)


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_no_value_ever_serialises_to_none(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )

    assert None not in params.values()
    assert "None" not in {str(v) for v in params.values()}


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_the_encoded_query_string_round_trips_every_value(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )
    encoded = httpx.URL("https://api.test.pinecone.io/backups", params=params).params

    if limit is not None:
        assert encoded["limit"] == str(limit)
    if pagination_token is not None:
        assert encoded["paginationToken"] == pagination_token
    if include_deleted is not None:
        assert encoded["include_deleted"] == ("true" if include_deleted else "false")
