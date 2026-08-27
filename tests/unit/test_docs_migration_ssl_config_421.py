"""Executes ``docs/migration/v10-migration.md`` (#421).

Same discipline as ``test_docs_migration_query_param_enums_371.py``: the guide's
table is read out of the published file and run, never transcribed here, so a
transcription cannot drift from what a reader relies on.

The guide's ``TLS now`` column describes the SSL context of the transport that
opens the socket. Each row is evaluated as constructor keyword arguments against
both the sync and the async client, and the described outcome is checked against
what the context really says. The ``TLS before`` column is not executable — the
old code path is gone — so its citation is the measurement in the PR body.
"""

from __future__ import annotations

import re
import ssl
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone import AsyncPinecone
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.admin.admin import _OAUTH_URL, Admin

GUIDE = Path(__file__).resolve().parents[2] / "docs/migration/v10-migration.md"
SECTION_START = "(ssl-config)="


def _section() -> str:
    return GUIDE.read_text().split(SECTION_START, 1)[1]


def _table_rows() -> list[tuple[str, str, str]]:
    """The (keyword arguments, TLS before, TLS now) rows of the guide's table."""
    rows = []
    for line in _section().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or set(cells[0]) == {"-"} or "TLS before" in cells[1]:
            continue
        if not (cells[0].startswith("`") and cells[0].endswith("`")):
            continue
        rows.append((cells[0].strip("`"), cells[1], cells[2]))
    assert len(rows) == 5, f"expected 5 rows in {GUIDE}, got {len(rows)}"
    return rows


ROWS = _table_rows()


def _socket_ssl_context(client: httpx.Client | httpx.AsyncClient) -> ssl.SSLContext:
    transport = client._transport_for_url(httpx.URL("https://api.example.com"))
    inner = getattr(transport, "_transport", transport)
    return inner._pool._ssl_context  # type: ignore[no-any-return, union-attr]


def _one_ca_pem() -> str:
    context = _socket_ssl_context(HTTPClient(PineconeConfig(api_key="k"), "2026-07")._client)
    return ssl.DER_cert_to_PEM_cert(context.get_ca_certs(binary_form=True)[0])


ONE_CA_PEM = _one_ca_pem()


def _materialize(kwargs: dict[str, object], tmp_path: Path) -> dict[str, object]:
    """Turn the guide's illustrative filenames into real paths under *tmp_path*."""
    ca_certs = kwargs.get("ssl_ca_certs")
    if ca_certs is None:
        return dict(kwargs)
    target = tmp_path / str(ca_certs)
    if str(ca_certs).endswith(".pem"):
        if "missing" not in str(ca_certs):
            target.write_text(ONE_CA_PEM)
    else:
        target.mkdir()
    return {**kwargs, "ssl_ca_certs": str(target)}


def _context(lane: str, kwargs: dict[str, object]) -> ssl.SSLContext:
    config = PineconeConfig(api_key="k", host="https://api.example.com", **kwargs)  # type: ignore[arg-type]
    if lane == "sync":
        return _socket_ssl_context(HTTPClient(config, "2026-07")._client)
    return _socket_ssl_context(AsyncHTTPClient(config, "2026-07")._ensure_client())


@pytest.mark.parametrize("lane", ["sync", "async"])
@pytest.mark.parametrize(("arguments", "before", "now"), ROWS, ids=[r[0] for r in ROWS])
def test_the_tls_now_column_is_what_the_socket_really_gets(
    lane: str, arguments: str, before: str, now: str, tmp_path: Path
) -> None:
    kwargs = _materialize(eval(arguments, {}), tmp_path)  # noqa: S307

    if "FileNotFoundError" in now:
        with pytest.raises(FileNotFoundError):
            _context(lane, kwargs)
        return

    context = _context(lane, kwargs)
    if "verification off" in now:
        assert context.verify_mode is ssl.CERT_NONE
    else:
        assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is ("hostname checked" in now)

    default_ca_count = len(_context(lane, {}).get_ca_certs())
    if "only that bundle" in now:
        assert len(context.get_ca_certs()) == 1
    elif "only that directory" in now:
        assert context.get_ca_certs() == []
    elif "default trust store" in now:
        assert len(context.get_ca_certs()) == default_ca_count


def test_the_guide_says_ssl_ca_certs_wins_and_that_is_true(tmp_path: Path) -> None:
    assert re.search(r"`ssl_ca_certs` continues to win over `ssl_verify`", GUIDE.read_text())
    bundle = tmp_path / "bundle.pem"
    bundle.write_text(ONE_CA_PEM)
    context = _context("sync", {"ssl_ca_certs": str(bundle), "ssl_verify": False})
    assert context.verify_mode is ssl.CERT_REQUIRED


def test_the_guide_says_an_unreadable_bundle_raises_and_that_is_true(tmp_path: Path) -> None:
    assert "raises `ssl.SSLError`" in GUIDE.read_text()
    bundle = tmp_path / "bundle.pem"
    bundle.write_text("not a certificate\n")
    with pytest.raises(ssl.SSLError):
        _context("sync", {"ssl_ca_certs": str(bundle)})


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_guide_says_grpc_secure_false_reaches_the_rest_client_and_that_is_true() -> None:
    text = GUIDE.read_text()
    assert "GrpcIndex` has no `ssl_ca_certs`" in text
    assert "upsert_records" in text and "search" in text

    grpc = pytest.importorskip("pinecone.grpc")
    index = grpc.GrpcIndex(
        api_key="k", host="idx-1234.svc.aped-4627-b74a.pinecone.io", secure=False
    )
    context = _socket_ssl_context(index._http._client)
    assert context.verify_mode is ssl.CERT_NONE
    assert context.check_hostname is False


async def test_the_guide_says_the_async_lane_raises_at_first_request(tmp_path: Path) -> None:
    assert "they raise at the" in _flat(GUIDE.read_text())
    assert "first request instead" in _flat(GUIDE.read_text())
    client = AsyncPinecone(
        api_key="k", host="https://api.example.com", ssl_ca_certs=str(tmp_path / "absent.pem")
    )
    with pytest.raises(FileNotFoundError):
        await client.describe_index("an-index")


@respx.mock
def test_the_guide_says_admin_covers_both_of_its_clients_and_that_is_true() -> None:
    assert "the OAuth token exchange as well as" in GUIDE.read_text()
    respx.post(_OAUTH_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "t", "token_type": "Bearer", "expires_in": 1800}
        )
    )
    created: list[httpx.HTTPTransport] = []
    real = httpx.HTTPTransport

    def spy(**kwargs: object) -> httpx.HTTPTransport:
        transport = real(**kwargs)  # type: ignore[arg-type]
        created.append(transport)
        return transport

    with patch("pinecone.admin.admin.httpx.HTTPTransport", spy):
        admin = Admin(client_id="i", client_secret="s", ssl_verify=False)
    admin.close()

    oauth_context = created[0]._pool._ssl_context  # type: ignore[attr-defined]
    assert oauth_context.verify_mode is ssl.CERT_NONE
    assert _socket_ssl_context(admin._http._client).verify_mode is ssl.CERT_NONE
