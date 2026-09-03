"""OAuth token response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class TokenResponse(StructDictMixin, Struct, kw_only=True):
    """Response model for the OAuth2 client-credentials token exchange.

    :class:`~pinecone.Admin` performs this exchange itself and refreshes the
    token as needed, so callers do not normally handle one of these. It is
    documented because the token is what a service account's ``client_id`` and
    ``client_secret`` are traded for.

    Attributes:
        access_token (str): The Bearer token used to authorize Admin API
            requests. Treat as a credential.
        token_type (str | None): The type of token issued. ``"Bearer"`` in
            practice, and ``None`` when the server omits the field.
        expires_in (int | None): Seconds until the token expires, or ``None``
            when the server omits it. Deleting the service account behind the
            token cuts it short of this; rotating that account's secret does
            not — the token already issued stays valid until it expires.

    Examples:
        >>> from pinecone.models.admin.token import TokenResponse
        >>> token = TokenResponse(
        ...     access_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9", token_type="Bearer"
        ... )
        >>> token["token_type"]
        'Bearer'
        >>> token.expires_in is None
        True
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
