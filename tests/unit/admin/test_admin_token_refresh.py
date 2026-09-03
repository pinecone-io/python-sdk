"""OAuth token expiry tracking and transparent re-fetch for :class:`Admin` (#243).

The backend provisions 10-hour tokens. These tests pin the two recovery paths
the SDK now has — a proactive re-fetch a margin ahead of the stated
``expires_in``, and a single retry against a fresh token when a request comes
back 401 anyway — plus the cases where neither should fire.

Every test drives a fake clock patched over ``pinecone.admin.admin.time``;
nothing here sleeps.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.admin import admin as admin_module
from pinecone.admin.admin import _OAUTH_URL, _TOKEN_REFRESH_MARGIN_SECONDS, Admin
from pinecone.errors.exceptions import ForbiddenError, NotFoundError, UnauthorizedError

ORGS_URL = f"{DEFAULT_BASE_URL}/admin/organizations"
KEYS_URL = f"{DEFAULT_BASE_URL}/admin/projects/p1/api-keys"
KEY_URL = f"{DEFAULT_BASE_URL}/admin/api-keys/k1"
EXPIRES_IN = 36000

_API_KEY_WITH_SECRET: dict[str, Any] = {
    "key": {
        "id": "key-abc123",
        "name": "mykey",
        "project_id": "p1",
        "roles": ["ProjectEditor"],
    },
    "value": "pckey_abc_123",
}


class _Clock:
    """Stand-in for the :mod:`time` module, exposing only what admin.py uses."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(admin_module, "time", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_CONTROLLER_HOST", raising=False)


def _token_payload(token: str, expires_in: int | None = EXPIRES_IN) -> dict[str, Any]:
    body: dict[str, Any] = {"access_token": token, "token_type": "Bearer"}
    if expires_in is not None:
        body["expires_in"] = expires_in
    return body


def _rotating_tokens(expires_in: int | None = EXPIRES_IN) -> Any:
    """Mint ``token-1``, ``token-2``, ... so every exchange is distinguishable."""
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, json=_token_payload(f"token-{counter['n']}", expires_in))

    return handler


def _bearers(route: respx.Route) -> list[str]:
    return [call.request.headers["Authorization"] for call in route.calls]


class TestFreshTokenIsReused:
    """A token nowhere near expiry must be reused, with no extra exchanges."""

    def test_repeated_calls_make_one_token_exchange(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                for _ in range(5):
                    admin.organizations.list()

            assert oauth.call_count == 1
            assert _bearers(orgs) == ["Bearer token-1"] * 5

    def test_no_refetch_just_inside_the_margin(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN - _TOKEN_REFRESH_MARGIN_SECONDS - 1)
                admin.organizations.list()

            assert oauth.call_count == 1

    def test_short_lived_token_is_not_born_stale(self, clock: _Clock) -> None:
        """``expires_in`` under twice the margin halves the margin instead of expiring at once."""
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens(expires_in=60))
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                admin.organizations.list()
                assert oauth.call_count == 1

                clock.advance(29)
                admin.organizations.list()
                assert oauth.call_count == 1

                clock.advance(1)
                admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-1", "Bearer token-1", "Bearer token-2"]


class TestExpiryTriggersRefetch:
    """Crossing the refresh deadline re-mints the token before the request goes out."""

    def test_refetch_at_the_margin(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                admin.organizations.list()
                clock.advance(EXPIRES_IN - _TOKEN_REFRESH_MARGIN_SECONDS)
                admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-1", "Bearer token-2"]

    def test_refetch_past_the_ten_hour_mark(self, clock: _Clock) -> None:
        """The reported failure: an Admin still working past the backend's 10h lifetime."""
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN + 3_600)
                admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-2"]

    def test_expired_burst_refetches_once(self, clock: _Clock) -> None:
        """A refreshed token resets the deadline, so a burst does not storm the endpoint."""
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN)
                for _ in range(6):
                    admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-2"] * 6

    def test_concurrent_expired_callers_share_one_exchange(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN)
                barrier = threading.Barrier(4)

                def call() -> None:
                    barrier.wait()
                    admin.organizations.list()

                threads = [threading.Thread(target=call) for _ in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-2"] * 4

    def test_post_path_picks_up_the_refreshed_token(self, clock: _Clock) -> None:
        """The POST fast path keeps its own header cache; it must be rewritten too."""
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            keys = router.post(KEYS_URL).mock(
                return_value=httpx.Response(201, json=_API_KEY_WITH_SECRET)
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN)
                admin.api_keys.create(project_id="p1", name="mykey")

            assert oauth.call_count == 2
            assert _bearers(keys) == ["Bearer token-2"]

    def test_delete_path_picks_up_the_refreshed_token(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            deletion = router.delete(KEY_URL).mock(return_value=httpx.Response(204))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN)
                admin.api_keys.delete(api_key_id="k1")

            assert oauth.call_count == 2
            assert _bearers(deletion) == ["Bearer token-2"]


class TestNoDeadlineWithoutExpiresIn:
    """Without a usable ``expires_in`` there is nothing to count down from."""

    @pytest.mark.parametrize("expires_in", [None, 0, -1])
    def test_absent_or_nonpositive_expires_in_never_refetches(
        self, clock: _Clock, expires_in: int | None
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens(expires_in))
            orgs = router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(10 * EXPIRES_IN)
                admin.organizations.list()

            assert oauth.call_count == 1
            assert _bearers(orgs) == ["Bearer token-1"]


class TestRetryOnUnauthorized:
    """A 401 the deadline did not predict is recovered once, then surfaced."""

    def test_401_refetches_and_retries_once(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(
                side_effect=[
                    httpx.Response(401, json={"error": {"message": "token expired"}}),
                    httpx.Response(200, json={"data": []}),
                ]
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                result = admin.organizations.list()

            assert list(result) == []
            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-1", "Bearer token-2"]

    def test_401_on_the_retry_is_raised(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(
                return_value=httpx.Response(401, json={"error": {"message": "nope"}})
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                with pytest.raises(UnauthorizedError, match="nope"):
                    admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-1", "Bearer token-2"]

    def test_401_not_retried_when_the_endpoint_repeats_the_token(self, clock: _Clock) -> None:
        """An unchanged token means the 401 was not about expiry — do not resend."""
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(
                return_value=httpx.Response(200, json=_token_payload("same-token"))
            )
            orgs = router.get(ORGS_URL).mock(
                return_value=httpx.Response(401, json={"error": {"message": "wrong scope"}})
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                with pytest.raises(UnauthorizedError, match="wrong scope"):
                    admin.organizations.list()

            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer same-token"]

    def test_401_on_post_refetches_and_retries_once(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            keys = router.post(KEYS_URL).mock(
                side_effect=[
                    httpx.Response(401, json={"error": {"message": "token expired"}}),
                    httpx.Response(201, json=_API_KEY_WITH_SECRET),
                ]
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                created = admin.api_keys.create(project_id="p1", name="mykey")

            assert created.value == "pckey_abc_123"
            assert oauth.call_count == 2
            assert _bearers(keys) == ["Bearer token-1", "Bearer token-2"]

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(403, ForbiddenError), (404, NotFoundError)],
    )
    def test_other_failures_do_not_refetch(
        self, clock: _Clock, status: int, expected: type[Exception]
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(
                return_value=httpx.Response(status, json={"error": {"message": "denied"}})
            )

            with Admin(client_id="id", client_secret="secret") as admin:
                with pytest.raises(expected):
                    admin.organizations.list()

            assert oauth.call_count == 1
            assert _bearers(orgs) == ["Bearer token-1"]


class TestCallerPinnedAuthorizationOptsOut:
    """``additional_headers={"Authorization": ...}`` means the caller owns the token."""

    def test_pinned_header_is_never_refreshed(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            orgs = router.get(ORGS_URL).mock(
                return_value=httpx.Response(401, json={"error": {"message": "expired"}})
            )

            admin = Admin(
                client_id="id",
                client_secret="secret",
                additional_headers={"Authorization": "Bearer caller-owned"},
            )
            try:
                clock.advance(10 * EXPIRES_IN)
                with pytest.raises(UnauthorizedError):
                    admin.organizations.list()
            finally:
                admin.close()

            assert oauth.call_count == 1
            assert _bearers(orgs) == ["Bearer caller-owned"]


class TestRefreshRepeatsTheOriginalExchange:
    """A refresh must reissue the exact exchange construction made, overrides included."""

    def test_refresh_reuses_oauth_url_override_and_credentials(self, clock: _Clock) -> None:
        override = "http://localhost:5080/oauth/token"
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(override).mock(side_effect=_rotating_tokens())
            production = router.post(_OAUTH_URL).mock(
                return_value=httpx.Response(200, json=_token_payload("production"))
            )
            orgs = router.get("http://localhost:5080/admin/organizations").mock(
                return_value=httpx.Response(200, json={"data": []})
            )

            with Admin(
                client_id="sim-id",
                client_secret="sim-secret",
                host="http://localhost:5080",
                oauth_url=override,
            ) as admin:
                clock.advance(EXPIRES_IN)
                admin.organizations.list()

            assert not production.called
            assert oauth.call_count == 2
            assert _bearers(orgs) == ["Bearer token-2"]

            body = orjson.loads(oauth.calls.last.request.content)
            assert body["client_id"] == "sim-id"
            assert body["client_secret"] == "sim-secret"
            assert body["grant_type"] == "client_credentials"
            assert body["audience"] == "https://api.pinecone.io/"

    def test_refresh_keeps_the_source_tag_user_agent(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            oauth = router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret", source_tag="myapp") as admin:
                clock.advance(EXPIRES_IN)
                admin.organizations.list()

            assert oauth.call_count == 2
            assert "source_tag=myapp" in oauth.calls.last.request.headers["User-Agent"]


class TestRefreshedHeaderIsVisibleEverywhere:
    """Every header copy ``HTTPClient`` keeps must agree after a refresh."""

    def test_all_header_caches_are_rewritten(self, clock: _Clock) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())
            router.get(ORGS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

            with Admin(client_id="id", client_secret="secret") as admin:
                clock.advance(EXPIRES_IN)
                admin.organizations.list()

                http = admin._http
                expected = "Bearer token-2"
                assert http._headers["Authorization"] == expected
                assert http._post_default_headers["Authorization"] == expected
                assert http._post_default_headers_obj["Authorization"] == expected
                assert http._client.headers["Authorization"] == expected


class TestMintClosureDoesNotLeakTheClientSecret:
    """``_mint`` must not be a ``functools.partial`` whose repr bares the secret."""

    def test_mint_repr_omits_the_client_secret(self, clock: _Clock) -> None:
        secret = "SUPERSECRET_MARKER_abcd"
        with respx.mock(assert_all_called=False) as router:
            router.post(_OAUTH_URL).mock(side_effect=_rotating_tokens())

            with Admin(client_id="id", client_secret=secret) as admin:
                assert secret not in repr(admin._http._mint)
