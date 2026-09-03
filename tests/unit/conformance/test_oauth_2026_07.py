"""2026-07 conformance for the single oauth operation, ``get_token``.

``oauth_2026-07.oas.yaml`` is byte-for-byte its 2025-10 predecessor apart from
``info.version``, so this pins the request the SDK actually makes during
:class:`Admin` construction: POST /oauth/token on ``login.pinecone.io``, the
2026-07 version header, the ``TokenRequest`` body, and a ``TokenResponse``
round-trip.
"""

from __future__ import annotations

from typing import Any

import httpx
import orjson
import respx

from pinecone.admin.admin import _OAUTH_AUDIENCE, _OAUTH_URL, Admin
from pinecone.models.admin.token import TokenResponse
from tests.unit.conformance import api_op

TOKEN: dict[str, Any] = {
    "access_token": "conformance-access-token",
    "token_type": "Bearer",
    "expires_in": 1800,
}


@api_op("oauth:get_token")
def test_get_token(claim: Any, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(_OAUTH_URL).mock(return_value=httpx.Response(200, json=TOKEN))

    admin = Admin(client_id="conformance-id", client_secret="conformance-secret")
    try:
        assert admin._http._headers["Authorization"] == "Bearer conformance-access-token"
    finally:
        admin.close()

    request = route.calls.last.request
    assert orjson.loads(request.content) == {
        "client_id": "conformance-id",
        "client_secret": "conformance-secret",
        "grant_type": "client_credentials",
        "audience": _OAUTH_AUDIENCE,
    }
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(TokenResponse, TOKEN, optional_absent=["expires_in"])
