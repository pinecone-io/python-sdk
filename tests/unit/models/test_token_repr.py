"""``TokenResponse`` must never leak the live Bearer token through ``repr()``/``str()``."""

from __future__ import annotations

from pinecone.models.admin.token import TokenResponse

SECRET = "SUPERSECRET_MARKER_abcd"


def test_repr_masks_access_token() -> None:
    token = TokenResponse(access_token=SECRET, token_type="Bearer", expires_in=3600)

    assert SECRET not in repr(token)
    assert "...abcd" in repr(token)
    assert repr(token) == (
        "TokenResponse(access_token='...abcd', token_type='Bearer', expires_in=3600)"
    )


def test_str_matches_repr() -> None:
    token = TokenResponse(access_token=SECRET, token_type="Bearer", expires_in=3600)

    assert SECRET not in str(token)
    assert str(token) == repr(token)
    assert SECRET not in f"{token}"


def test_repr_defined_directly_on_token_response() -> None:
    assert "__repr__" in TokenResponse.__dict__


def test_repr_short_token_falls_back_to_stars() -> None:
    token = TokenResponse(access_token="ab", token_type=None, expires_in=None)

    assert repr(token) == "TokenResponse(access_token='***', token_type=None, expires_in=None)"
