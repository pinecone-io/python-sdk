"""Admin client for Pinecone organization and project management."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from pinecone import __version__
from pinecone._internal.adapters._decode import decode_response
from pinecone._internal.config import PineconeConfig, normalize_host
from pinecone._internal.constants import ADMIN_API_VERSION, API_VERSION_HEADER, DEFAULT_BASE_URL
from pinecone._internal.http_client import HTTPClient, _build_socket_options, _RetryTransport
from pinecone._internal.user_agent import build_user_agent
from pinecone.errors.exceptions import (
    ApiError,
    PineconeConnectionError,
    PineconeTimeoutError,
    ResponseParsingError,
    UnauthorizedError,
    ValidationError,
)
from pinecone.models.admin.token import TokenResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinecone.admin.api_keys import ApiKeys
    from pinecone.admin.invites import Invites
    from pinecone.admin.organizations import Organizations
    from pinecone.admin.projects import Projects
    from pinecone.admin.role_bindings import RoleBindings
    from pinecone.admin.service_accounts import ServiceAccounts
    from pinecone.admin.users import Users

_OAUTH_URL: str = "https://login.pinecone.io/oauth/token"
_OAUTH_AUDIENCE: str = "https://api.pinecone.io/"
_TOKEN_REFRESH_MARGIN_SECONDS: float = 300.0


def _refresh_deadline(expires_in: int | None) -> float | None:
    """Monotonic time at which a token minted now should be replaced.

    Replacement happens ``_TOKEN_REFRESH_MARGIN_SECONDS`` ahead of the stated
    expiry so a request already in flight cannot outlive the credential it was
    sent with. The margin is halved down for tokens shorter than twice the
    margin, so a short-lived token is not born already stale — which would
    otherwise re-fetch on every single request.

    ``None`` means "no deadline": the token endpoint reported no usable
    ``expires_in``, so there is nothing to count down from and expiry is left
    to the 401 retry.
    """
    if expires_in is None or expires_in <= 0:
        return None
    margin = min(_TOKEN_REFRESH_MARGIN_SECONDS, expires_in / 2)
    return time.monotonic() + expires_in - margin


class _TokenRefreshingHTTPClient(HTTPClient):
    """:class:`HTTPClient` that keeps the Admin OAuth Bearer token current.

    The token endpoint issues short-lived tokens and the Bearer value is baked
    into the headers the underlying httpx client was built with. Left alone, an
    :class:`Admin` held past that lifetime would
    return bare 401s forever. So the live token is consulted before every
    request, re-minted a margin ahead of its expiry, and a request that still
    comes back 401 is retried exactly once against a freshly minted token.

    Refresh is serialized on a lock and the deadline re-checked inside it, so N
    threads waking to an expired token cost one token exchange rather than N.
    ``stream`` is deliberately not wrapped: no admin operation streams.
    """

    def __init__(
        self,
        config: PineconeConfig,
        api_version: str,
        *,
        mint: Callable[[], TokenResponse],
        token: str,
        expires_in: int | None,
    ) -> None:
        super().__init__(config, api_version)
        self._mint = mint
        self._token = token
        self._deadline = _refresh_deadline(expires_in)
        self._token_lock = threading.Lock()

    def _apply_token(self, token: str) -> None:
        """Rewrite the Bearer header everywhere :class:`HTTPClient` cached a copy."""
        header = f"Bearer {token}"
        self._headers["Authorization"] = header
        self._post_default_headers["Authorization"] = header
        self._post_default_headers_obj["Authorization"] = header
        self._client.headers["Authorization"] = header

    def _mint_locked(self) -> str:
        response = self._mint()
        self._token = response.access_token
        self._deadline = _refresh_deadline(response.expires_in)
        self._apply_token(self._token)
        return self._token

    def _current_token(self, stale: str | None = None) -> str:
        """Return the token to use, minting a replacement when the old one is spent.

        With *stale* set — the 401 path — a replacement is minted only if the
        token that failed is still the current one; a thread that lost the race
        gets the winner's token instead of triggering a second exchange.
        """
        with self._token_lock:
            if stale is not None:
                if stale != self._token:
                    return self._token
                return self._mint_locked()
            if self._deadline is not None and time.monotonic() >= self._deadline:
                return self._mint_locked()
            return self._token

    def _with_refresh(self, call: Callable[[], httpx.Response]) -> httpx.Response:
        used = self._current_token()
        try:
            return call()
        except UnauthorizedError:
            if self._current_token(stale=used) == used:
                raise
            return call()

    def get(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        bound = super().get
        return self._with_refresh(lambda: bound(path, timeout=timeout, **kwargs))

    def post(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        bound = super().post
        return self._with_refresh(lambda: bound(path, timeout=timeout, **kwargs))

    def put(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        bound = super().put
        return self._with_refresh(lambda: bound(path, timeout=timeout, **kwargs))

    def patch(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        bound = super().patch
        return self._with_refresh(lambda: bound(path, timeout=timeout, **kwargs))

    def delete(
        self, path: str, timeout: float | httpx.Timeout | None = None, **kwargs: Any
    ) -> httpx.Response:
        bound = super().delete
        return self._with_refresh(lambda: bound(path, timeout=timeout, **kwargs))


class Admin:
    """Admin client for Pinecone organization and project management.

    Authenticates via OAuth2 client credentials flow to obtain a Bearer
    token used for all admin API calls.

    **Auth model:** :class:`Admin` uses OAuth2 client credentials (service account), while
    :class:`~pinecone.Pinecone` uses API keys.  These serve different purposes:

    - :class:`Admin` — organization/project/key management (create projects, rotate keys, etc.)
    - :class:`~pinecone.Pinecone` — index and vector operations (upsert, query, etc.)

    A common workflow bridges both: use :class:`Admin` to create a project and API key, then
    pass that key to :class:`~pinecone.Pinecone` for data-plane operations::

        from pinecone import Admin, Pinecone

        admin = Admin(client_id="...", client_secret="...")
        project = admin.projects.create(name="my-project")
        key = admin.api_keys.create(project_id=project.id, name="my-key")
        pc = Pinecone(api_key=key.value)
        pc.indexes.create(
            name="my-index",
            schema={"fields": {"values": {"type": "dense_vector",
                                          "dimension": 1536, "metric": "cosine"}}},
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        )

    Projects are created within the organization associated with your OAuth credentials.

    Operations are grouped into seven namespaces:

    - :attr:`organizations` — the organizations these credentials can reach
    - :attr:`projects` — create, configure, and delete projects
    - :attr:`api_keys` — the project-scoped keys :class:`~pinecone.Pinecone` authenticates with
    - :attr:`users` — the organization's members
    - :attr:`invites` — pending and expired invitations to join the organization
    - :attr:`service_accounts` — the OAuth principals this client itself authenticates as
    - :attr:`role_bindings` — every grant of a role to a principal, at organization or
      project scope; nothing else confers permissions

    .. note::
        **Obtaining OAuth credentials** — a service account's ``client_id`` and
        ``client_secret`` can come from either of two places:

        - :meth:`admin.service_accounts.create()
          <pinecone.admin.service_accounts.ServiceAccounts.create>`, once you already hold
          admin credentials. The ``client_secret`` is returned exactly once, at creation, and
          :meth:`~pinecone.admin.service_accounts.ServiceAccounts.rotate_secret` is the only
          way to obtain another.
        - The `Pinecone console <https://console.pinecone.io>`_, under organization settings.
          This is how the first pair is obtained, since there is nothing to authenticate an
          :class:`Admin` with until one service account exists.

        These differ from the API keys used by :class:`~pinecone.Pinecone`; they are scoped to
        your organization and used exclusively for admin operations.

    .. note::
        **Token lifetime** — the Bearer token is short-lived. The client tracks the
        ``expires_in`` it was issued with and re-fetches transparently shortly before expiry,
        and retries a request once against a freshly minted token if one still comes back 401,
        so a long-lived :class:`Admin` keeps working indefinitely without any caller action.
        Refresh is serialized, so threads sharing one :class:`Admin` cost a single token
        exchange between them rather than one each. Supplying your own ``Authorization`` entry
        in ``additional_headers`` opts out of refresh entirely — the token is then yours to
        manage.

    .. note::
        :class:`Admin` is synchronous only. There is no async form of this client; admin
        operations are infrequent control-plane calls, and
        :class:`~pinecone.AsyncPinecone` is where the async lane lives.

    Args:
        client_id (str | None): OAuth2 client ID. Falls back to ``PINECONE_CLIENT_ID`` env var.
        client_secret (str | None): OAuth2 client secret. Falls back to ``PINECONE_CLIENT_SECRET``
            env var.
        additional_headers (dict[str, str] | None): Extra headers included in every admin API
            request. Merged last, so an entry keyed exactly ``"Authorization"`` or
            ``"X-Pinecone-Api-Version"`` replaces the Bearer token or the API version the SDK
            would otherwise send. Matching is case-sensitive: any other spelling is sent
            alongside the SDK's own header rather than replacing it.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        host (str | None): Admin API host. Falls back to ``PINECONE_CONTROLLER_HOST`` env var,
            then defaults to ``https://api.pinecone.io``. A value with no scheme is prefixed
            with ``https://``, matching :class:`~pinecone.Pinecone`. Intended for pointing the
            client at a local simulator in tests or at a private Pinecone deployment; leave it
            unset against production.
        oauth_url (str | None): Full URL of the OAuth2 token endpoint, including its path.
            Defaults to ``https://login.pinecone.io/oauth/token``. A value with no scheme is
            prefixed with ``https://``. Intended for pointing the token exchange at a local
            simulator in tests or at a private Pinecone deployment; leave it unset against
            production. There is no environment-variable fallback for this parameter.

    Raises:
        :exc:`~pinecone.errors.exceptions.PineconeValueError`:
            If client_id or client_secret cannot be resolved.
        :exc:`ApiError`: If the OAuth token request fails.

    Examples:

        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for org in admin.organizations.list():
        ...     print(org.name)
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        additional_headers: dict[str, str] | None = None,
        proxy_url: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        host: str | None = None,
        oauth_url: str | None = None,
    ) -> None:
        resolved_id = client_id or os.environ.get("PINECONE_CLIENT_ID", "")
        resolved_secret = client_secret or os.environ.get("PINECONE_CLIENT_SECRET", "")

        if not resolved_id or not resolved_id.strip():
            raise ValidationError(
                "No client_id provided. Pass client_id='...' or set the "
                "PINECONE_CLIENT_ID environment variable."
            )
        if not resolved_secret or not resolved_secret.strip():
            raise ValidationError(
                "No client_secret provided. Pass client_secret='...' or set the "
                "PINECONE_CLIENT_SECRET environment variable."
            )

        resolved_source_tag = source_tag or ""

        mint = partial(
            self._fetch_token,
            resolved_id,
            resolved_secret,
            proxy_url=proxy_url,
            ssl_verify=ssl_verify,
            source_tag=resolved_source_tag,
            oauth_url=normalize_host(oauth_url) or _OAUTH_URL,
        )
        token = mint()

        headers: dict[str, str] = {
            "Authorization": f"Bearer {token.access_token}",
            API_VERSION_HEADER: ADMIN_API_VERSION,
        }
        if additional_headers:
            headers.update(additional_headers)

        config = PineconeConfig(
            api_key="",
            host=host or "",
            additional_headers=headers,
            proxy_url=proxy_url or "",
            ssl_verify=ssl_verify,
            source_tag=resolved_source_tag,
        )
        # __post_init__ already applied the PINECONE_CONTROLLER_HOST fallback and
        # scheme normalization; only the api.pinecone.io default is left to fill in.
        if not config.host:
            config = replace(config, host=DEFAULT_BASE_URL)

        # Prevent __post_init__ from falling back to PINECONE_API_KEY env var.
        # The Admin client authenticates via OAuth Bearer token, not Api-Key.
        # Must follow the replace() above, which re-runs __post_init__.
        object.__setattr__(config, "api_key", "")

        caller_pinned_auth = bool(additional_headers and "Authorization" in additional_headers)

        self._http: HTTPClient
        if caller_pinned_auth:
            self._http = HTTPClient(config, ADMIN_API_VERSION)
        else:
            self._http = _TokenRefreshingHTTPClient(
                config,
                ADMIN_API_VERSION,
                mint=mint,
                token=token.access_token,
                expires_in=token.expires_in,
            )

        self._organizations: Organizations | None = None
        self._projects: Projects | None = None
        self._api_keys: ApiKeys | None = None
        self._users: Users | None = None
        self._invites: Invites | None = None
        self._service_accounts: ServiceAccounts | None = None
        self._role_bindings: RoleBindings | None = None

    def _fetch_token(
        self,
        client_id: str,
        client_secret: str,
        *,
        proxy_url: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        oauth_url: str | None = None,
    ) -> TokenResponse:
        """Exchange client credentials for a Bearer token.

        Called once during construction and again by
        :class:`_TokenRefreshingHTTPClient` whenever the live token needs
        replacing, so the refresh path reuses exactly the request the initial
        exchange made.

        Args:
            client_id: OAuth2 client ID.
            client_secret: OAuth2 client secret.
            proxy_url: Optional HTTP proxy URL.
            ssl_verify: Whether to verify SSL certificates.
            source_tag: Optional source tag to append to the User-Agent string.
            oauth_url: Token endpoint URL. Defaults to the production endpoint; overridden
                when testing against a simulator or a private deployment.

        Returns:
            The decoded token response, carrying the access token and the
            ``expires_in`` its lifetime is tracked from.

        Raises:
            ApiError: If the token request fails.
        """
        body = orjson.dumps(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "audience": _OAUTH_AUDIENCE,
            }
        )

        transport = _RetryTransport(
            transport=httpx.HTTPTransport(
                verify=ssl_verify, http2=False, socket_options=_build_socket_options()
            ),
        )
        with httpx.Client(
            transport=transport,
            proxy=proxy_url or None,
            verify=ssl_verify,
        ) as client:
            try:
                response = client.post(
                    oauth_url or _OAUTH_URL,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": build_user_agent(__version__, source_tag),
                        API_VERSION_HEADER: ADMIN_API_VERSION,
                    },
                )
            except httpx.TimeoutException as exc:
                raise PineconeTimeoutError(str(exc)) from exc
            except httpx.TransportError as exc:
                raise PineconeConnectionError(str(exc)) from exc

        if not response.is_success:
            err_body: dict[str, Any] | None = None
            try:
                err_body = response.json()
            except Exception:
                err_body = None

            message = "OAuth token request failed"
            if err_body and isinstance(err_body.get("error_description"), str):
                message = err_body["error_description"]
            elif err_body and isinstance(err_body.get("error"), str):
                message = err_body["error"]

            raise ApiError(
                message=message,
                status_code=response.status_code,
                body=err_body,
            )

        try:
            token = decode_response(response.content, TokenResponse)
        except ResponseParsingError:
            token = None

        if token is None or not token.access_token:
            missing_body: dict[str, Any] | None
            try:
                missing_body = response.json()
            except Exception:
                missing_body = None
            raise ApiError(
                message="OAuth response missing access_token",
                status_code=response.status_code,
                body=missing_body,
            )

        return token

    @property
    def organizations(self) -> Organizations:
        """Access the Organizations namespace for organization operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Organizations` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for org in admin.organizations.list():
            ...     print(org.name)
        """
        if self._organizations is None:
            from pinecone.admin.organizations import Organizations as _Organizations

            self._organizations = _Organizations(http=self._http)
        return self._organizations

    @property
    def projects(self) -> Projects:
        """Access the Projects namespace for project operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Projects` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for project in admin.projects.list():
            ...     print(project.name)
        """
        if self._projects is None:
            from pinecone.admin.projects import Projects as _Projects

            self._projects = _Projects(http=self._http, admin=self)
        return self._projects

    @property
    def api_keys(self) -> ApiKeys:
        """Access the ApiKeys namespace for API key operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`ApiKeys` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> keys = admin.api_keys.list(project_id="proj-abc123")
            >>> for key in keys:
            ...     print(key.key.id)
        """
        if self._api_keys is None:
            from pinecone.admin.api_keys import ApiKeys as _ApiKeys

            self._api_keys = _ApiKeys(http=self._http)
        return self._api_keys

    @property
    def users(self) -> Users:
        """Access the Users namespace for organization-member operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Users` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for user in admin.users.list():
            ...     print(user.email)
        """
        if self._users is None:
            from pinecone.admin.users import Users as _Users

            self._users = _Users(http=self._http)
        return self._users

    @property
    def invites(self) -> Invites:
        """Access the Invites namespace for organization-invite operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Invites` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for invite in admin.invites.list():
            ...     print(invite.email, invite.status)
        """
        if self._invites is None:
            from pinecone.admin.invites import Invites as _Invites

            self._invites = _Invites(http=self._http)
        return self._invites

    @property
    def service_accounts(self) -> ServiceAccounts:
        """Access the ServiceAccounts namespace for service-account operations.

        These are the same credentials this client authenticates with, so
        rotating or deleting the account behind ``client_id``/``client_secret``
        breaks this client. Lazily imported and instantiated on first access.

        Returns:
            :class:`ServiceAccounts` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for account in admin.service_accounts.list():
            ...     print(account.id, account.name)
        """
        if self._service_accounts is None:
            from pinecone.admin.service_accounts import ServiceAccounts as _ServiceAccounts

            self._service_accounts = _ServiceAccounts(http=self._http)
        return self._service_accounts

    @property
    def role_bindings(self) -> RoleBindings:
        """Access the RoleBindings namespace for role-binding operations.

        Role bindings are the only thing that confers permissions in Pinecone, so
        this is where any principal's access — user, service account, API key, or
        pending invite — is read and changed. Lazily imported and instantiated on
        first access.

        Returns:
            :class:`RoleBindings` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for binding in admin.role_bindings.list():
            ...     print(binding.principal_id, binding.role, binding.resource_id)
        """
        if self._role_bindings is None:
            from pinecone.admin.role_bindings import RoleBindings as _RoleBindings

            self._role_bindings = _RoleBindings(http=self._http)
        return self._role_bindings

    def __repr__(self) -> str:
        return (
            "Admin(organizations=<Organizations>, projects=<Projects>, "
            "api_keys=<ApiKeys>, users=<Users>, invites=<Invites>, "
            "service_accounts=<ServiceAccounts>, role_bindings=<RoleBindings>)"
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> Admin:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
