# Authentication

The SDK has two clients and they authenticate differently.
{class}`~pinecone.Pinecone` — indexes, vectors, documents, inference — uses a
project-scoped **API key**. {class}`~pinecone.admin.Admin` — organizations, projects,
API keys, users, role bindings — uses **OAuth2 client credentials** belonging to a
service account. An API key will not authenticate the Admin client, and client credentials
will not authenticate `Pinecone`.

## Getting an API key

Sign in to the [Pinecone console](https://app.pinecone.io), navigate to **API Keys**,
and create or copy an existing key.

## Environment variable (recommended)

Set `PINECONE_API_KEY` before running your application:

```bash
export PINECONE_API_KEY=your-key-here
```

The client reads it automatically when you call `Pinecone()` with no arguments:

```python
from pinecone import Pinecone

pc = Pinecone()
```

## Explicit argument

Pass the key directly if you manage secrets through your own mechanism:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key-here")
```

## Missing key error

If no API key can be resolved from either the argument or the environment variable,
the client raises `PineconeValueError` on construction, rather than letting the first
request fail with a 401:

```python
from pinecone import Pinecone, PineconeValueError

try:
    pc = Pinecone()
except PineconeValueError as e:
    print(e)  # "No API key provided. Pass api_key='...' or set the PINECONE_API_KEY environment variable."
```

A key that is present but wrong cannot be detected at construction time. That
surfaces on the first call as {exc}`~pinecone.errors.exceptions.UnauthorizedError`;
see [Error handling](../guides/error-handling.md).

## The Admin client

{class}`~pinecone.admin.Admin` takes a service account's `client_id` and
`client_secret`, falling back to `PINECONE_CLIENT_ID` and `PINECONE_CLIENT_SECRET`.
It is synchronous only — there is no async form.

```bash
export PINECONE_CLIENT_ID=your-client-id
export PINECONE_CLIENT_SECRET=your-client-secret
```

```python
from pinecone import Admin

admin = Admin()
```

Constructing it exchanges the credentials for an access token immediately, so bad
credentials fail here rather than on the first call. A missing `client_id` or
`client_secret` raises `PineconeValueError`, the same as a missing API key; a
credential pair the token endpoint rejects raises
{exc}`~pinecone.errors.exceptions.ApiError`.

Admin credentials are the way to mint the API keys `Pinecone` then uses:

```python
key = admin.api_keys.create(project_id="your-project-id", name="ingest-worker")
pc = Pinecone(api_key=key.value)
```

## Environment variables the SDK reads

| Variable | Read by | Effect |
| --- | --- | --- |
| `PINECONE_API_KEY` | `Pinecone`, `AsyncPinecone` | API key, when none is passed |
| `PINECONE_CLIENT_ID` | `Admin` | OAuth2 client ID, when none is passed |
| `PINECONE_CLIENT_SECRET` | `Admin` | OAuth2 client secret, when none is passed |
| `PINECONE_CONTROLLER_HOST` | all three | Control-plane host, when no `host` is passed |
| `PINECONE_ADDITIONAL_HEADERS` | `Pinecone`, `AsyncPinecone` | A JSON object of extra request headers, when no `additional_headers` is passed |

Every one of these is only a fallback: an explicit argument always wins.

## Connecting through a proxy or a private CA

Credentials are not the only thing between you and the API. All three clients accept
`proxy_url` and `ssl_verify`; `Pinecone` and `AsyncPinecone` also accept
`proxy_headers`, for a proxy that authenticates, and `ssl_ca_certs`:

```python
from pinecone import Pinecone

pc = Pinecone(
    proxy_url="http://proxy.corp.internal:3128",
    proxy_headers={"Proxy-Authorization": "Basic base64-of-user-colon-password"},
)
```

`ssl_ca_certs` takes a path to a CA bundle file or a directory of them, for a
corporate root or a self-signed endpoint. It wins over `ssl_verify=False`: pass both
and verification stays on. A path that does not exist raises `FileNotFoundError`
while the client is being constructed, so a typo cannot leave you silently trusting
the default store instead.

## Security best practices

Never hardcode API keys or client secrets in source files. Use environment variables
or a secrets manager. As a lightweight alternative, the `python-dotenv` package loads
a local `.env` file (which you add to `.gitignore`):

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()  # reads PINECONE_API_KEY from .env
pc = Pinecone()
```

Neither client leaks its credential through `repr()`: `Pinecone` shows only the last
four characters of the API key, and `Admin` shows none of its credentials at all. A
client is safe to include in a log line or a traceback.
