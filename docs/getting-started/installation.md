# Installation

## Prerequisites

- Python 3.10 or newer.
- A [Pinecone account](https://app.pinecone.io) and an API key. See
  [Authentication](authentication.md) for where the SDK looks for it.

## Install

```bash
pip install pinecone
```

Or with uv:

```bash
uv add pinecone
```

## What the base install already includes

There are no user-facing extras. `pip install pinecone` gives you the whole SDK:

- The synchronous {class}`~pinecone.Pinecone` client and the `asyncio`
  {class}`~pinecone.async_client.pinecone.AsyncPinecone` client. Both are built on
  `httpx`, so neither needs a separate install. See
  [Sync vs. async](../guides/sync-vs-async.md).
- The gRPC data-plane client, reached with `pc.index(name="product-catalog", grpc=True)`.
  It is a compiled extension module shipped inside the wheel, so there is nothing
  extra to install. See [Using gRPC](../guides/grpc.md).
- The {class}`~pinecone.admin.Admin` client for organization and project
  administration.

The `dev` and `docs` extras exist for working on the SDK itself, not for using it.

Wheels are published for Linux (glibc and musl, x86-64 and arm64), macOS (Intel and
Apple silicon), and Windows (x86-64 and arm64). Installing from the source
distribution instead builds the gRPC extension from Rust and needs a Rust toolchain.

## pandas is not installed for you

One method, {meth}`~pinecone.index.Index.upsert_from_dataframe`, takes a
`pandas.DataFrame`. pandas is not a dependency of the SDK and not offered as an
extra, because that one method is the only thing that needs it. Install it yourself
if you use it:

```bash
pip install pandas
```

Calling the method without pandas installed raises `RuntimeError`. Every other
bulk-upsert path, including {meth}`~pinecone.index.Index.upsert` in a loop and
{meth}`~pinecone.index.Index.start_import`, works without it. See
[Upsert from a DataFrame](../how-to/vectors/upsert-from-dataframe.md).

## Verify the installation

```bash
python -c "import pinecone; print(pinecone.__version__)"
```

That prints the installed version, which should match what you asked pip for.
