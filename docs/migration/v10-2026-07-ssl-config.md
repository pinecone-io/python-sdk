# 2026-07: `ssl_ca_certs` and `ssl_verify` now take effect

`Pinecone`, `AsyncPinecone`, `Index` and `AsyncIndex` accept `ssl_ca_certs` and
`ssl_verify`, and `Admin` accepts `ssl_verify`. Until this release none of them
did anything. The signatures did not change, so there is no code to edit — but
the requests your client makes may now be verified differently than they were,
which is the point of the fix and the reason this note exists.

## Why they did nothing

Both settings were resolved from config and handed to `httpx.Client` /
`httpx.AsyncClient` as `verify=`. The SDK also supplies its own transport, to
add connection retries and TCP keep-alive tuning, and httpx returns a supplied
transport unchanged rather than configuring it — so the object that opened the
socket was built with httpx's defaults. Every connection used the default trust
store no matter what you passed.

## What changes

The rows below are the constructor keyword arguments, and what the connection
does with them:

| keyword arguments | TLS before | TLS now |
| --- | --- | --- |
| `{}` | default trust store, hostname checked | default trust store, hostname checked |
| `{"ssl_verify": False}` | default trust store, hostname checked | verification off, hostname not checked |
| `{"ssl_ca_certs": "bundle.pem"}` | default trust store, hostname checked | only that bundle trusted, hostname checked |
| `{"ssl_ca_certs": "ca-dir"}` | default trust store, hostname checked | only that directory trusted, hostname checked |
| `{"ssl_ca_certs": "missing.pem"}` | default trust store, hostname checked | `FileNotFoundError` when the client is built |

`ssl_ca_certs` continues to win over `ssl_verify` when both are given, as it
always has: supplying a bundle means you want that bundle trusted.

## Was my code affected?

**If you pass neither, nothing changes.** This is the overwhelmingly common
case.

**If you pass `ssl_ca_certs` because you sit behind a TLS-inspecting proxy**,
your bundle is now the one that is trusted. This is what you asked for
originally. If your connections were succeeding before, they were succeeding on
the default trust store, and they will now succeed or fail on your bundle
instead — so a bundle that does not actually contain the proxy's issuing CA will
now surface a certificate error where previously it was ignored.

**If you pass `ssl_verify=False`**, verification and hostname checking are now
genuinely off. Traffic is still encrypted, but the SDK no longer confirms it is
talking to the host it dialled. Only use it against an endpoint you control.

**If you pass a `ssl_ca_certs` path that does not exist**, building the client
now raises `FileNotFoundError` instead of ignoring the setting. A typo in the
path used to leave you verifying against the default store while believing you
had pinned trust; failing early is the only way to tell you that the trust you
configured is not the trust you have. A path that exists but holds no readable
certificate raises `ssl.SSLError` for the same reason.

`Pinecone` and `Index` raise at construction. `AsyncPinecone` and `AsyncIndex`
build their underlying connection pool on first use, so they raise at the first
request instead — the same error, at the point they first need the trust store.

## `Admin` covers both of its clients

`Admin(ssl_verify=False)` now applies to the OAuth token exchange as well as to
the Admin API requests that follow it. The token exchange uses a client of its
own, so before this release the setting was ignored on both.

## `GrpcIndex` inherits the change

`GrpcIndex` has no `ssl_ca_certs` or `ssl_verify` of its own, but its
`secure=False` is forwarded to the REST client that backs `upsert_records` and
`search`, where it means what `ssl_verify=False` means above. Those two
operations are unverified under `secure=False` where before they were verified.
The gRPC channel itself is unaffected: `secure` has always chosen its scheme.
