# 2026-07: preview namespace removed — surfaces graduated

The `2026-01.alpha` preview surface graduated to the default entry points.
The `pinecone/preview/` package is deleted outright: `import pinecone.preview`
now raises `ModuleNotFoundError`, and the `pc.preview` property is gone from
both `Pinecone` and `AsyncPinecone` (accessing it raises `AttributeError`).
Preview surface was never covered by SemVer, and the graduated replacements
below are the supported paths.

## Entry points

| Removed | Replacement |
| --- | --- |
| `pc.preview.indexes` | `pc.indexes` |
| `pc.preview.index(name=..., host=...)` | `pc.index(name=..., host=...)` — same name/host resolution, with the host cache now on the client |
| `pc.preview.index(...).documents.upsert(...)` | `pc.index(...).upsert_documents(...)` (and `search_documents`, `fetch_documents`, `delete_documents`, `update_documents`, `list_documents`, `batch_upsert_documents`) |
| `from pinecone.preview import SchemaBuilder` | `from pinecone import SchemaBuilder` |
| `pinecone.preview.models.Preview*` models | unprefixed models in `pinecone.models` (e.g. `PreviewIndexModel` → `IndexModel`, `PreviewTextQuery` → `TextQuery`) |

## API version

The preview constant `INDEXES_API_VERSION = "2026-01.alpha"`
(`pinecone/preview/_internal/constants.py`) is deleted with the package.
The graduated surfaces negotiate `X-Pinecone-Api-Version: 2026-07` via
`CONTROL_PLANE_API_VERSION` / `DATA_PLANE_API_VERSION` in
`pinecone/_internal/constants.py`. Nothing in the SDK sends `2026-01.alpha`
any more.

## Failure modes for stale code

- `import pinecone.preview` → `ModuleNotFoundError`
- `pc.preview` / `AsyncPinecone(...).preview` → `AttributeError`
- No `Preview*`-prefixed names are exported from `pinecone` or
  `pinecone.models`.

A fuller narrative migration guide for the graduated surfaces ships with the
2026-07 release notes.
