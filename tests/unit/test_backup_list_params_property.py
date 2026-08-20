"""Property-based tests for the list-backups query string.

Pins the mapping every backup listing depends on: SDK keyword ->
wire parameter name. ``limit`` and ``paginationToken`` differ in case
convention, ``include_deleted`` is a lowercase JSON-style boolean, and an
argument left at ``None`` must never reach the wire as its ``repr``.

That last one is a claim about *omitted arguments*, not about the text
``None``: ``paginationToken`` is an opaque server-minted string, so
``"None"``, ``"null"`` and ``""`` are all legal tokens a caller may be
holding and every one of them has to survive to the wire byte-for-byte
(#273).

Also pins the mutual exclusion the offset-token backend forces (#252):
``limit`` and ``paginationToken`` may never appear together, for any pair of
values. ``include_deleted`` is exempt — it is a filter, not a window.
"""

from __future__ import annotations

import httpx
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from pinecone._internal.backups_helpers import backup_list_params

_limit = st.one_of(st.none(), st.integers(min_value=1, max_value=100))
# min_size=0: the empty token is a legal value distinct from an absent one.
_token_text = st.text(min_size=0, max_size=40)
_token = st.one_of(st.none(), _token_text)
_include_deleted = st.one_of(st.none(), st.booleans())

_ABSENT_LOOKALIKE_TOKENS = ("None", "null", "NULL", "nil", "undefined", "")


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

    assert ("limit" in params) is (limit is not None and pagination_token is None)
    assert ("paginationToken" in params) is (pagination_token is not None)
    assert ("include_deleted" in params) is (include_deleted is not None)


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_limit_and_pagination_token_are_never_sent_together(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )

    assert not ("limit" in params and "paginationToken" in params)


@given(limit=st.integers(min_value=1, max_value=100), token=_token_text)
def test_a_token_suppresses_the_limit_it_was_paired_with(limit: int, token: str) -> None:
    with_token = backup_list_params(limit=limit, pagination_token=token)
    without_token = backup_list_params(limit=limit)

    assert "limit" not in with_token
    assert with_token["paginationToken"] == token
    assert without_token["limit"] == limit


def _verbatim_wire_values(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> dict[str, str | int | None]:
    return {
        "limit": limit,
        "paginationToken": pagination_token,
        "include_deleted": None if include_deleted is None else str(include_deleted).lower(),
    }


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
@example(limit=None, pagination_token="None", include_deleted=None)
@example(limit=7, pagination_token="None", include_deleted=True)
@example(limit=None, pagination_token="", include_deleted=False)
def test_no_omitted_argument_reaches_the_wire_as_its_repr(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    """Every emitted value is its own argument, so no ``None`` is stringified.

    Stated this way rather than as "the text ``None`` never appears anywhere":
    a caller holding the opaque token ``"None"`` has to be able to send it,
    and the old spelling rejected that (#273). Requiring each present key to
    carry its argument verbatim is the stronger claim -- it rules out
    ``str(None)`` for *every* key instead of pattern-matching the one
    placeholder, and it forbids a key whose argument was ``None`` from
    appearing at all.
    """
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )
    verbatim = _verbatim_wire_values(limit, pagination_token, include_deleted)

    assert None not in params.values()
    for key, value in params.items():
        assert verbatim[key] is not None
        assert value == verbatim[key]


@given(limit=_limit, pagination_token=_token, include_deleted=_include_deleted)
def test_the_encoded_query_string_round_trips_every_value(
    limit: int | None, pagination_token: str | None, include_deleted: bool | None
) -> None:
    params = backup_list_params(
        limit=limit, pagination_token=pagination_token, include_deleted=include_deleted
    )
    encoded = httpx.URL("https://api.test.pinecone.io/backups", params=params).params

    if limit is not None and pagination_token is None:
        assert encoded["limit"] == str(limit)
    if pagination_token is not None:
        assert encoded["paginationToken"] == pagination_token
        assert "limit" not in encoded
    if include_deleted is not None:
        assert encoded["include_deleted"] == ("true" if include_deleted else "false")


@pytest.mark.parametrize("token", _ABSENT_LOOKALIKE_TOKENS)
def test_a_token_that_looks_like_an_absent_value_survives_to_the_wire(token: str) -> None:
    """Regression for #273: these are opaque tokens, not stand-ins for absence.

    The server mints ``paginationToken`` and the SDK never interprets it, so a
    token that happens to read ``None`` -- or ``null``, or empty -- is as
    forwardable as any other. Pinned as literals because the property test
    only reaches them on a lucky draw.
    """
    params = backup_list_params(limit=10, pagination_token=token, include_deleted=True)

    assert params == {"paginationToken": token, "include_deleted": "true"}

    encoded = httpx.URL("https://api.test.pinecone.io/backups", params=params).params
    assert encoded["paginationToken"] == token
    assert "limit" not in encoded


def test_an_absent_token_is_not_the_same_as_an_empty_one() -> None:
    """``""`` is a value the caller chose; ``None`` is the caller staying silent."""
    assert "paginationToken" not in backup_list_params(pagination_token=None)
    assert backup_list_params(pagination_token="") == {"paginationToken": ""}
