"""OAuth token response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class TokenResponse(StructDictMixin, Struct, kw_only=True):
    """Response model for the OAuth2 client-credentials token exchange.

    Attributes:
        access_token: The Bearer token used to authorize Admin API requests.
        token_type: The type of token issued. ``"Bearer"`` in practice.
        expires_in: Seconds until the token expires.
    """

    access_token: str
    token_type: str | None = None
    expires_in: int | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. token['access_token'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'access_token' in token``)."""
        return key in self.__struct_fields__
