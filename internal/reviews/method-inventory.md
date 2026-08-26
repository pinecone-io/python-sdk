# Pinecone Python SDK — Public Method Inventory

This is a comprehensive checklist of every public-facing method in the Pinecone
Python SDK, across all three transports (synchronous REST, asynchronous REST,
and gRPC), organized by resource and then by action. It is intended to drive a
systematic, agent-assisted review process — each row is independently
addressable via its unique ID.

## How this is organized

- **Sections** group methods by resource (Index, Collections, Backups, Vectors,
  Assistants, Admin, ...).
- Within a section, methods are grouped by **action** (create, describe, list,
  update, delete, ...), and for each action the transports appear together —
  **REST immediately followed by async**, then gRPC where it applies — so a
  reviewer can compare implementations side by side.
- **Legacy / backcompat shims** (flat methods on `Pinecone`/`AsyncPinecone`
  that delegate to a namespace, deprecated aliases, etc.) are listed
  immediately after the canonical methods they delegate to, tagged
  `*-LEGACY`.

## ID scheme

Each row has a unique ID of the form `<RESOURCE>-<TRANSPORT>-<NN>`, e.g.
`IDX-REST-05`. Transport tags:

| Tag | Meaning |
|---|---|
| `REST` | Synchronous HTTP client (`pinecone.Pinecone` / `pinecone.client.*`) |
| `ASYNC` | Asynchronous HTTP client (`pinecone.AsyncPinecone` / `pinecone.async_client.*`) |
| `GRPC` | gRPC data-plane client (`pinecone.grpc.*`) |
| `*-LEGACY` | Backwards-compatibility shim that delegates to a canonical method above it |
| `*-MODEL` | Method attached to a *model instance* (e.g. `AssistantModel`) rather than a namespace client |

## Notes on line numbers

Per instructions, **no line numbers are given** — only file paths — since line
numbers drift quickly. Use the method/class name to locate the definition
(e.g. `grep -n "def create_for_model" pinecone/client/indexes.py`).

## Legend

- [ ] Unreviewed
- Check the box once the method has been reviewed/verified.

---

## 0. Client Construction & Lifecycle — `PC`

### `Pinecone` (sync) / `AsyncPinecone` (async) / `PineconeGRPC` (gRPC)

- [ ] **PC-REST-01** `Pinecone.__init__()` — construct sync client — `pinecone/_client.py`
- [ ] **PC-ASYNC-01** `AsyncPinecone.__init__()` — construct async client — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-02** `Pinecone.__repr__()` — `pinecone/_client.py`
- [ ] **PC-ASYNC-02** `AsyncPinecone.__repr__()` — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-03** `Pinecone.indexes` (property) — lazy `Indexes` namespace accessor — `pinecone/_client.py`
- [ ] **PC-ASYNC-03** `AsyncPinecone.indexes` (property) — lazy `AsyncIndexes` namespace accessor — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-04** `Pinecone.collections` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-04** `AsyncPinecone.collections` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-05** `Pinecone.backups` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-05** `AsyncPinecone.backups` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-06** `Pinecone.backup_schedules` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-06** `AsyncPinecone.backup_schedules` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-07** `Pinecone.restore_jobs` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-07** `AsyncPinecone.restore_jobs` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-08** `Pinecone.inference` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-08** `AsyncPinecone.inference` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-09** `Pinecone.assistants` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-09** `AsyncPinecone.assistants` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-10** `Pinecone.assistant` (property) — callable proxy alias, not deprecated — `pinecone/_client.py`
- [ ] **PC-ASYNC-10** `AsyncPinecone.assistant` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-11** `Pinecone.index()` — data-plane client factory; `grpc=True` dispatches to `GrpcIndex` — `pinecone/_client.py`
- [ ] **PC-ASYNC-11** `AsyncPinecone.index()` — data-plane client factory (async) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-12** `Pinecone.config` (property) — `pinecone/_client.py`
- [ ] **PC-ASYNC-12** `AsyncPinecone.config` (property) — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-13** `Pinecone.close()` — `pinecone/_client.py`
- [ ] **PC-ASYNC-13** `AsyncPinecone.close()` — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-14** `Pinecone.__enter__()` — `pinecone/_client.py`
- [ ] **PC-ASYNC-14** `AsyncPinecone.__aenter__()` — `pinecone/async_client/pinecone.py`
- [ ] **PC-REST-15** `Pinecone.__exit__()` — `pinecone/_client.py`
- [ ] **PC-ASYNC-15** `AsyncPinecone.__aexit__()` — `pinecone/async_client/pinecone.py`
- [ ] **PC-GRPC-01** `PineconeGRPC(Pinecone).Index()` — subclasses `Pinecone`; overrides `.Index()` to always return a `GrpcIndex`; inherits all `PC-REST-*` control-plane properties/methods above unchanged — `pinecone/grpc/pinecone_grpc.py`

### Legacy factory shims

- [ ] **PC-REST-LEGACY-01** `Pinecone.Index()` — legacy PascalCase shim for `.index()` — `pinecone/_client.py`
- [ ] **PC-REST-LEGACY-02** `Pinecone.IndexAsyncio()` — legacy shim constructing an `AsyncIndex` from a sync `Pinecone` client — `pinecone/_client.py`
- [ ] **PC-ASYNC-LEGACY-01** `AsyncPinecone.IndexAsyncio()` — legacy shim, sync `def` (no I/O) returning an `AsyncIndex` — `pinecone/async_client/pinecone.py`

### Backup-restore entry point (canonical, not a shim)

- [ ] **PC-REST-16** `Pinecone.create_index_from_backup()` — only supported way to restore a backup — `pinecone/_client.py`
- [ ] **PC-ASYNC-16** `AsyncPinecone.create_index_from_backup()` — `pinecone/async_client/pinecone.py`

---

## 0b. Admin Client Construction & Lifecycle — `ADM-CORE`

Admin is **REST-only** — there is no async or gRPC admin client.

- [ ] **ADM-CORE-REST-01** `Admin.__init__()` — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-02** `Admin.organizations` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-03** `Admin.projects` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-04** `Admin.api_keys` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-05** `Admin.users` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-06** `Admin.invites` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-07** `Admin.service_accounts` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-08** `Admin.role_bindings` (property) — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-09** `Admin.__repr__()` — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-10** `Admin.close()` — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-11** `Admin.__enter__()` — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-12** `Admin.__exit__()` — `pinecone/admin/admin.py`
- [ ] **ADM-CORE-REST-13** `_TokenRefreshingHTTPClient.get/post/put/patch/delete()` — internal token-refreshing HTTP wrapper used by all Admin resources (5 methods, one class) — `pinecone/admin/admin.py`

---

## 1. Index (Control Plane) — `IDX`

### `Indexes` (sync) / `AsyncIndexes` (async) namespace

- [ ] **IDX-REST-01** `Indexes.list()` — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-01** `AsyncIndexes.list()` — sync `def` returning an async iterator/paginator (not itself a coroutine) — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-02** `Indexes.describe()` — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-02** `AsyncIndexes.describe()` — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-03** `Indexes.exists()` — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-03** `AsyncIndexes.exists()` — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-04** `Indexes.delete()` — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-04** `AsyncIndexes.delete()` — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-05** `Indexes.create()` — 2026-07 `schema=`/`deployment=`/`read_capacity=`/`cmek_id=` surface — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-05** `AsyncIndexes.create()` — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-06** `Indexes.create_for_model()` — integrated-embedding index creation — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-06** `AsyncIndexes.create_for_model()` — `pinecone/async_client/indexes.py`
- [ ] **IDX-REST-07** `Indexes.configure()` — `pinecone/client/indexes.py`
- [ ] **IDX-ASYNC-07** `AsyncIndexes.configure()` — `pinecone/async_client/indexes.py`

### Legacy flat shims on `Pinecone` / `AsyncPinecone`

- [ ] **IDX-REST-LEGACY-01** `Pinecone.create_index()` — delegates to `Indexes.create()`, rejects 2026-07-only kwargs — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-01** `AsyncPinecone.create_index()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-02** `Pinecone.create_index_for_model()` — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-02** `AsyncPinecone.create_index_for_model()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-03** `Pinecone.describe_index()` — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-03** `AsyncPinecone.describe_index()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-04** `Pinecone.list_indexes()` — returns legacy `IndexList` wrapping `.indexes.list().to_list()` — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-04** `AsyncPinecone.list_indexes()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-05** `Pinecone.has_index()` — delegates to `Indexes.exists()` — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-05** `AsyncPinecone.has_index()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-06** `Pinecone.configure_index()` — rejects 2026-07-only kwargs; returns updated `IndexModel` (9.x returned `None`) — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-06** `AsyncPinecone.configure_index()` — `pinecone/async_client/pinecone.py`
- [ ] **IDX-REST-LEGACY-07** `Pinecone.delete_index()` — `pinecone/_client.py`
- [ ] **IDX-ASYNC-LEGACY-07** `AsyncPinecone.delete_index()` — `pinecone/async_client/pinecone.py`

---

## 2. Collections — `COL`

### `Collections` (sync) / `AsyncCollections` (async) namespace

- [ ] **COL-REST-01** `Collections.create()` — `pinecone/client/collections.py`
- [ ] **COL-ASYNC-01** `AsyncCollections.create()` — `pinecone/async_client/collections.py`
- [ ] **COL-REST-02** `Collections.list()` — `pinecone/client/collections.py`
- [ ] **COL-ASYNC-02** `AsyncCollections.list()` — `pinecone/async_client/collections.py`
- [ ] **COL-REST-03** `Collections.describe()` — `pinecone/client/collections.py`
- [ ] **COL-ASYNC-03** `AsyncCollections.describe()` — `pinecone/async_client/collections.py`
- [ ] **COL-REST-04** `Collections.delete()` — `pinecone/client/collections.py`
- [ ] **COL-ASYNC-04** `AsyncCollections.delete()` — `pinecone/async_client/collections.py`

### Legacy flat shims on `Pinecone` / `AsyncPinecone`

- [ ] **COL-REST-LEGACY-01** `Pinecone.create_collection()` — `pinecone/_client.py`
- [ ] **COL-ASYNC-LEGACY-01** `AsyncPinecone.create_collection()` — `pinecone/async_client/pinecone.py`
- [ ] **COL-REST-LEGACY-02** `Pinecone.list_collections()` — `pinecone/_client.py`
- [ ] **COL-ASYNC-LEGACY-02** `AsyncPinecone.list_collections()` — `pinecone/async_client/pinecone.py`
- [ ] **COL-REST-LEGACY-03** `Pinecone.describe_collection()` — `pinecone/_client.py`
- [ ] **COL-ASYNC-LEGACY-03** `AsyncPinecone.describe_collection()` — `pinecone/async_client/pinecone.py`
- [ ] **COL-REST-LEGACY-04** `Pinecone.delete_collection()` — `pinecone/_client.py`
- [ ] **COL-ASYNC-LEGACY-04** `AsyncPinecone.delete_collection()` — `pinecone/async_client/pinecone.py`

---

## 3. Backups — `BKP`

### `Backups` (sync) / `AsyncBackups` (async) namespace

- [ ] **BKP-REST-01** `Backups.create()` — `pinecone/client/backups.py`
- [ ] **BKP-ASYNC-01** `AsyncBackups.create()` — `pinecone/async_client/backups.py`
- [ ] **BKP-REST-02** `Backups.list()` — `pinecone/client/backups.py`
- [ ] **BKP-ASYNC-02** `AsyncBackups.list()` — `pinecone/async_client/backups.py`
- [ ] **BKP-REST-03** `Backups.describe()` — `pinecone/client/backups.py`
- [ ] **BKP-ASYNC-03** `AsyncBackups.describe()` — `pinecone/async_client/backups.py`
- [ ] **BKP-REST-04** `Backups.get()` — **alias for `.describe()`** (body is `return self.describe(...)`) — `pinecone/client/backups.py`
- [ ] **BKP-ASYNC-04** `AsyncBackups.get()` — alias for `.describe()` — `pinecone/async_client/backups.py`
- [ ] **BKP-REST-05** `Backups.delete()` — `pinecone/client/backups.py`
- [ ] **BKP-ASYNC-05** `AsyncBackups.delete()` — `pinecone/async_client/backups.py`

### Index-scoped convenience methods (on `Indexes` / `AsyncIndexes`, backup actions)

- [ ] **BKP-REST-06** `Indexes.create_backup()` — `pinecone/client/indexes.py`
- [ ] **BKP-ASYNC-06** `AsyncIndexes.create_backup()` — `pinecone/async_client/indexes.py`
- [ ] **BKP-REST-07** `Indexes.list_backups()` — `pinecone/client/indexes.py`
- [ ] **BKP-ASYNC-07** `AsyncIndexes.list_backups()` — sync `def` returning an async iterator/paginator — `pinecone/async_client/indexes.py`
- [ ] **BKP-REST-08** `Indexes.describe_backup()` — `pinecone/client/indexes.py`
- [ ] **BKP-ASYNC-08** `AsyncIndexes.describe_backup()` — `pinecone/async_client/indexes.py`

### Restore entry point (canonical; cross-referenced from section 0)

- [ ] **BKP-REST-09** `Pinecone.create_index_from_backup()` — see **PC-REST-16** — `pinecone/_client.py`
- [ ] **BKP-ASYNC-09** `AsyncPinecone.create_index_from_backup()` — see **PC-ASYNC-16** — `pinecone/async_client/pinecone.py`

### Legacy flat shims on `Pinecone` / `AsyncPinecone`

- [ ] **BKP-REST-LEGACY-01** `Pinecone.create_backup()` — `pinecone/_client.py`
- [ ] **BKP-ASYNC-LEGACY-01** `AsyncPinecone.create_backup()` — `pinecone/async_client/pinecone.py`
- [ ] **BKP-REST-LEGACY-02** `Pinecone.list_backups()` — `pinecone/_client.py`
- [ ] **BKP-ASYNC-LEGACY-02** `AsyncPinecone.list_backups()` — `pinecone/async_client/pinecone.py`
- [ ] **BKP-REST-LEGACY-03** `Pinecone.describe_backup()` — `pinecone/_client.py`
- [ ] **BKP-ASYNC-LEGACY-03** `AsyncPinecone.describe_backup()` — `pinecone/async_client/pinecone.py`
- [ ] **BKP-REST-LEGACY-04** `Pinecone.delete_backup()` — `pinecone/_client.py`
- [ ] **BKP-ASYNC-LEGACY-04** `AsyncPinecone.delete_backup()` — `pinecone/async_client/pinecone.py`

---

## 4. Backup Schedules — `BSC`

No gRPC or legacy-flat-shim surface exists for this resource.

- [ ] **BSC-REST-01** `BackupSchedules.create()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-01** `AsyncBackupSchedules.create()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-02** `BackupSchedules.list()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-02** `AsyncBackupSchedules.list()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-03** `BackupSchedules.iter_schedules()` — sync `def` convenience iterator — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-03** `AsyncBackupSchedules.iter_schedules()` — sync `def` returning an async generator — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-04** `BackupSchedules.describe()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-04** `AsyncBackupSchedules.describe()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-05** `BackupSchedules.get()` — **alias for `.describe()`** — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-05** `AsyncBackupSchedules.get()` — alias for `.describe()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-06** `BackupSchedules.update()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-06** `AsyncBackupSchedules.update()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-07** `BackupSchedules.delete()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-07** `AsyncBackupSchedules.delete()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-08** `BackupSchedules.history()` — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-08** `AsyncBackupSchedules.history()` — `pinecone/async_client/backup_schedules.py`
- [ ] **BSC-REST-09** `BackupSchedules.iter_history()` — sync `def` convenience iterator — `pinecone/client/backup_schedules.py`
- [ ] **BSC-ASYNC-09** `AsyncBackupSchedules.iter_history()` — sync `def` returning an async generator — `pinecone/async_client/backup_schedules.py`

---

## 5. Restore Jobs — `RST`

### `RestoreJobs` (sync) / `AsyncRestoreJobs` (async) namespace

- [ ] **RST-REST-01** `RestoreJobs.list()` — `pinecone/client/restore_jobs.py`
- [ ] **RST-ASYNC-01** `AsyncRestoreJobs.list()` — `pinecone/async_client/restore_jobs.py`
- [ ] **RST-REST-02** `RestoreJobs.describe()` — `pinecone/client/restore_jobs.py`
- [ ] **RST-ASYNC-02** `AsyncRestoreJobs.describe()` — `pinecone/async_client/restore_jobs.py`

### Legacy flat shims on `Pinecone` / `AsyncPinecone`

- [ ] **RST-REST-LEGACY-01** `Pinecone.list_restore_jobs()` — `pinecone/_client.py`
- [ ] **RST-ASYNC-LEGACY-01** `AsyncPinecone.list_restore_jobs()` — `pinecone/async_client/pinecone.py`
- [ ] **RST-REST-LEGACY-02** `Pinecone.describe_restore_job()` — `pinecone/_client.py`
- [ ] **RST-ASYNC-LEGACY-02** `AsyncPinecone.describe_restore_job()` — `pinecone/async_client/pinecone.py`

---

## 6. Inference — `INF`

### `Inference` (sync) / `AsyncInference` (async) namespace

- [ ] **INF-REST-01** `Inference.embed()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-01** `AsyncInference.embed()` — `pinecone/async_client/inference.py`
- [ ] **INF-REST-02** `Inference.rerank()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-02** `AsyncInference.rerank()` — `pinecone/async_client/inference.py`
- [ ] **INF-REST-03** `Inference.list_models()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-03** `AsyncInference.list_models()` — `pinecone/async_client/inference.py`
- [ ] **INF-REST-04** `Inference.get_model()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-04** `AsyncInference.get_model()` — `pinecone/async_client/inference.py`
- [ ] **INF-REST-05** `Inference.model` (property) — `ModelResource` accessor — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-05** `AsyncInference.model` (property) — `AsyncModelResource` accessor — `pinecone/async_client/inference.py`
- [ ] **INF-REST-06** `Inference.close()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-06** `AsyncInference.close()` — `pinecone/async_client/inference.py`

### `ModelResource` (sync) / `AsyncModelResource` (async) — `Inference.model.*`

- [ ] **INF-REST-07** `ModelResource.list()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-07** `AsyncModelResource.list()` — `pinecone/async_client/inference.py`
- [ ] **INF-REST-08** `ModelResource.get()` — `pinecone/client/inference.py`
- [ ] **INF-ASYNC-08** `AsyncModelResource.get()` — `pinecone/async_client/inference.py`

*Note: `INF-REST-03/04` (`Inference.list_models()`/`.get_model()`) and
`INF-REST-07/08` (`Inference.model.list()`/`.get()`) are two independent entry
points to the same underlying operations — worth verifying they stay in sync.*

---

## 7. Assistants — `AST`

gRPC does not apply to Assistants (control/data plane over HTTP only).

### Core CRUD — `Assistants` (sync) / `AsyncAssistants` (async)

- [ ] **AST-REST-01** `Assistants.create()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-01** `AsyncAssistants.create()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-02** `Assistants.describe()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-02** `AsyncAssistants.describe()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-03** `Assistants.list()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-03** `AsyncAssistants.list()` — sync `def` returning an async iterator — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-04** `Assistants.list_page()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-04** `AsyncAssistants.list_page()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-05** `Assistants.update()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-05** `AsyncAssistants.update()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-06** `Assistants.delete()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-06** `AsyncAssistants.delete()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-07** `Assistants.close()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-07** `AsyncAssistants.close()` — `pinecone/async_client/assistants.py`

### Files — `Assistants.*` / `AsyncAssistants.*`

- [ ] **AST-REST-08** `Assistants.upload_file()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-08** `AsyncAssistants.upload_file()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-09** `Assistants.describe_file()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-09** `AsyncAssistants.describe_file()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-10** `Assistants.list_files()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-10** `AsyncAssistants.list_files()` — sync `def` returning an async iterator — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-11** `Assistants.list_files_page()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-11** `AsyncAssistants.list_files_page()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-12** `Assistants.delete_file()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-12** `AsyncAssistants.delete_file()` — `pinecone/async_client/assistants.py`

### Operations — `Assistants.*` / `AsyncAssistants.*`

- [ ] **AST-REST-13** `Assistants.describe_operation()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-13** `AsyncAssistants.describe_operation()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-14** `Assistants.list_operations()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-14** `AsyncAssistants.list_operations()` — sync `def` returning an async iterator — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-15** `Assistants.list_operations_page()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-15** `AsyncAssistants.list_operations_page()` — `pinecone/async_client/assistants.py`

### Chat / Context — `Assistants.*` / `AsyncAssistants.*`

- [ ] **AST-REST-16** `Assistants.context()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-16** `AsyncAssistants.context()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-17** `Assistants.chat()` — supports `stream=True` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-17** `AsyncAssistants.chat()` — `pinecone/async_client/assistants.py`
- [ ] **AST-REST-18** `Assistants.chat_completions()` — OpenAI-compatible endpoint — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-18** `AsyncAssistants.chat_completions()` — `pinecone/async_client/assistants.py`

### Evaluation — `Assistants.*` / `AsyncAssistants.*`

- [ ] **AST-REST-19** `Assistants.evaluate_alignment()` — `pinecone/client/assistants.py`
- [ ] **AST-ASYNC-19** `AsyncAssistants.evaluate_alignment()` — `pinecone/async_client/assistants.py`

### Legacy namespace mixin — `AssistantsLegacyNamespaceMixin` / `AsyncAssistantsLegacyNamespaceMixin`

Mixed into `Assistants`/`AsyncAssistants` (see class declarations: `class
Assistants(AssistantsLegacyNamespaceMixin)`).

- [ ] **AST-REST-LEGACY-01** `Assistants.list_assistants()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-01** `AsyncAssistants.list_assistants()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-02** `Assistants.list_assistants_paginated()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-02** `AsyncAssistants.list_assistants_paginated()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-03** `Assistants.describe_assistant()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-03** `AsyncAssistants.describe_assistant()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-04** `Assistants.update_assistant()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-04** `AsyncAssistants.update_assistant()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-05** `Assistants.create_assistant()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-05** `AsyncAssistants.create_assistant()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-06** `Assistants.delete_assistant()` — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-06** `AsyncAssistants.delete_assistant()` — `pinecone/async_client/_assistants_legacy.py`
- [ ] **AST-REST-LEGACY-07** `Assistants.evaluation.alignment()` — `evaluation` property returns `_AlignmentEvaluationProxy`, whose `.alignment()` delegates — `pinecone/client/_assistants_legacy.py`
- [ ] **AST-ASYNC-LEGACY-07** `AsyncAssistants.evaluation.alignment()` — via `_AsyncAlignmentEvaluationProxy` — `pinecone/async_client/_assistants_legacy.py`

### Legacy model-attached methods — `AssistantModelLegacyMethodsMixin`

Mixed into `AssistantModel` (`pinecone/models/assistant/model.py`). **Sync
only** — every method resolves a sync `Assistants` namespace via
`self._resolve_assistants()` regardless of whether the model instance came
from a sync or async client; deprecated since 9.0.0 in favor of the
`Assistants`/`AsyncAssistants` namespace methods above. **Flag for review:**
confirm what happens when one of these is called on a model instance that
was returned by `AsyncAssistants` (does `_attach_ref` even wire it up, or
does it raise?).

- [ ] **AST-REST-MODEL-01** `AssistantModel.describe_file()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-02** `AssistantModel.upload_bytes_stream()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-03** `AssistantModel.list_files()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-04** `AssistantModel.list_files_paginated()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-05** `AssistantModel.upload_file()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-06** `AssistantModel.delete_file()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-07** `AssistantModel.chat_completions()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-08** `AssistantModel.context()` — `pinecone/models/assistant/_legacy_methods.py`
- [ ] **AST-REST-MODEL-09** `AssistantModel.chat()` — deprecated since 9.0.0, use `Assistants.chat()` — `pinecone/models/assistant/_legacy_methods.py`

### Namespace proxy — `_AssistantNamespaceProxy` / `_AsyncAssistantNamespaceProxy`

Backs `Pinecone.assistant` / `AsyncPinecone.assistant` (see **PC-REST-10** /
**PC-ASYNC-10**).

- [ ] **AST-REST-20** `_AssistantNamespaceProxy.__call__()` — shorthand for `Assistants.describe(name=...)` — `pinecone/client/_assistant_namespace_proxy.py`
- [ ] **AST-ASYNC-20** `_AsyncAssistantNamespaceProxy.__call__()` — `pinecone/client/_assistant_namespace_proxy.py`
- [ ] **AST-REST-21** `_AssistantNamespaceProxy.__getattr__()` — forwards namespace-style calls, e.g. `pc.assistant.create_assistant(...)` — `pinecone/client/_assistant_namespace_proxy.py`
- [ ] **AST-ASYNC-21** `_AsyncAssistantNamespaceProxy.__getattr__()` — `pinecone/client/_assistant_namespace_proxy.py`

---

## 8. Data Plane — Vectors — `VEC`

`Index` (sync REST), `AsyncIndex` (async REST), `GrpcIndex` (gRPC) — obtained
from `pc.index(...)` / `pc.index(..., grpc=True)`.

- [ ] **VEC-REST-01** `Index.upsert()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-01** `AsyncIndex.upsert()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-01** `GrpcIndex.upsert()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-02** `Index.upsert_from_dataframe()` — pandas optional-dependency path — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-02** `AsyncIndex.upsert_from_dataframe()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-02** `GrpcIndex.upsert_from_dataframe()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-03** `Index.upsert_records()` — integrated-embedding upsert — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-03** `AsyncIndex.upsert_records()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-03** `GrpcIndex.upsert_records()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-04** `Index.query()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-04** `AsyncIndex.query()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-04** `GrpcIndex.query()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-05** `Index.query_namespaces()` — fan-out query across namespaces — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-05** `AsyncIndex.query_namespaces()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-05** `GrpcIndex.query_namespaces()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-06** `Index.fetch()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-06** `AsyncIndex.fetch()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-06** `GrpcIndex.fetch()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-07** `Index.fetch_by_metadata()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-07** `AsyncIndex.fetch_by_metadata()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-07** `GrpcIndex.fetch_by_metadata()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-08** `Index.delete()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-08** `AsyncIndex.delete()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-08** `GrpcIndex.delete()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-09** `Index.update()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-09** `AsyncIndex.update()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-09** `GrpcIndex.update()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-10** `Index.describe_index_stats()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-10** `AsyncIndex.describe_index_stats()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-10** `GrpcIndex.describe_index_stats()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-11** `Index.search()` — integrated-embedding search within a namespace — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-11** `AsyncIndex.search()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-11** `GrpcIndex.search()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-12** `Index.search_records()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-12** `AsyncIndex.search_records()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-12** `GrpcIndex.search_records()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-13** `Index.list_paginated()` — vector-ID listing, single page — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-13** `AsyncIndex.list_paginated()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-13** `GrpcIndex.list_paginated()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-14** `Index.list()` — vector-ID listing, auto-paginating generator — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-14** `AsyncIndex.list()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-14** `GrpcIndex.list()` — `pinecone/grpc/__init__.py`

### gRPC-only future/callback-style variants

No REST or async-HTTP equivalent — gRPC's native async-callback pattern
(returns a `PineconeGrpcFuture`, distinct from Python's `async`/`await`).

- [ ] **VEC-GRPC-15** `GrpcIndex.upsert_async()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-GRPC-16** `GrpcIndex.query_async()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-GRPC-17** `GrpcIndex.fetch_async()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-GRPC-18** `GrpcIndex.delete_async()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-GRPC-19** `GrpcIndex.update_async()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-GRPC-20** `GrpcIndex.query_namespaces_async()` — `pinecone/grpc/__init__.py`

### Lifecycle

- [ ] **VEC-REST-15** `Index.close()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-15** `AsyncIndex.close()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-21** `GrpcIndex.close()` — `pinecone/grpc/__init__.py`
- [ ] **VEC-REST-16** `Index.__enter__()` / `__exit__()` — `pinecone/index/__init__.py`
- [ ] **VEC-ASYNC-16** `AsyncIndex.__aenter__()` / `__aexit__()` — `pinecone/async_client/async_index.py`
- [ ] **VEC-GRPC-22** `GrpcIndex.__enter__()` / `__exit__()` — `pinecone/grpc/__init__.py`

### Legacy `async_req=True` execution model (REST only)

Not a distinct method — a thread-pool execution mode layered onto
`VEC-REST-01/04/10/13` (`upsert`/`query`/`describe_index_stats`/
`list_paginated`) when `Pinecone(pool_threads=...)` is set. Implemented by:

- [ ] **VEC-REST-LEGACY-01** `install_async_req_support()` / `_LegacyAsyncPool` — wraps the four methods above to accept `async_req=True` and return a `multiprocessing.pool.ApplyResult`-like handle — `pinecone/_legacy/async_req.py`

---

## 9. Data Plane — Namespaces — `NS`

Methods on `Index` / `AsyncIndex` / `GrpcIndex` (same classes as section 8).

- [ ] **NS-REST-01** `Index.create_namespace()` — `pinecone/index/__init__.py`
- [ ] **NS-ASYNC-01** `AsyncIndex.create_namespace()` — `pinecone/async_client/async_index.py`
- [ ] **NS-GRPC-01** `GrpcIndex.create_namespace()` — `pinecone/grpc/__init__.py`
- [ ] **NS-REST-02** `Index.describe_namespace()` — `pinecone/index/__init__.py`
- [ ] **NS-ASYNC-02** `AsyncIndex.describe_namespace()` — `pinecone/async_client/async_index.py`
- [ ] **NS-GRPC-02** `GrpcIndex.describe_namespace()` — `pinecone/grpc/__init__.py`
- [ ] **NS-REST-03** `Index.delete_namespace()` — `pinecone/index/__init__.py`
- [ ] **NS-ASYNC-03** `AsyncIndex.delete_namespace()` — `pinecone/async_client/async_index.py`
- [ ] **NS-GRPC-03** `GrpcIndex.delete_namespace()` — `pinecone/grpc/__init__.py`
- [ ] **NS-REST-04** `Index.list_namespaces_paginated()` — `pinecone/index/__init__.py`
- [ ] **NS-ASYNC-04** `AsyncIndex.list_namespaces_paginated()` — `pinecone/async_client/async_index.py`
- [ ] **NS-GRPC-04** `GrpcIndex.list_namespaces_paginated()` — `pinecone/grpc/__init__.py`
- [ ] **NS-REST-05** `Index.list_namespaces()` — auto-paginating generator — `pinecone/index/__init__.py`
- [ ] **NS-ASYNC-05** `AsyncIndex.list_namespaces()` — `pinecone/async_client/async_index.py`
- [ ] **NS-GRPC-05** `GrpcIndex.list_namespaces()` — `pinecone/grpc/__init__.py`

---

## 10. Data Plane — Bulk Import — `IMP`

Methods on `Index` / `AsyncIndex` / `GrpcIndex` (same classes as section 8).

- [ ] **IMP-REST-01** `Index.start_import()` — `pinecone/index/__init__.py`
- [ ] **IMP-ASYNC-01** `AsyncIndex.start_import()` — `pinecone/async_client/async_index.py`
- [ ] **IMP-GRPC-01** `GrpcIndex.start_import()` — `pinecone/grpc/__init__.py`
- [ ] **IMP-REST-02** `Index.describe_import()` — `pinecone/index/__init__.py`
- [ ] **IMP-ASYNC-02** `AsyncIndex.describe_import()` — `pinecone/async_client/async_index.py`
- [ ] **IMP-GRPC-02** `GrpcIndex.describe_import()` — `pinecone/grpc/__init__.py`
- [ ] **IMP-REST-03** `Index.cancel_import()` — `pinecone/index/__init__.py`
- [ ] **IMP-ASYNC-03** `AsyncIndex.cancel_import()` — `pinecone/async_client/async_index.py`
- [ ] **IMP-GRPC-03** `GrpcIndex.cancel_import()` — `pinecone/grpc/__init__.py`
- [ ] **IMP-REST-04** `Index.list_imports()` — auto-paginating generator — `pinecone/index/__init__.py`
- [ ] **IMP-ASYNC-04** `AsyncIndex.list_imports()` — `pinecone/async_client/async_index.py`
- [ ] **IMP-GRPC-04** `GrpcIndex.list_imports()` — `pinecone/grpc/__init__.py`
- [ ] **IMP-REST-05** `Index.list_imports_paginated()` — `pinecone/index/__init__.py`
- [ ] **IMP-ASYNC-05** `AsyncIndex.list_imports_paginated()` — `pinecone/async_client/async_index.py`
- [ ] **IMP-GRPC-05** `GrpcIndex.list_imports_paginated()` — `pinecone/grpc/__init__.py`

---

## 11. Data Plane — Documents / Records (batch) — `DOC`

No gRPC surface — `GrpcIndex` has no `.documents` namespace. Accessed via
`Index.documents` / `AsyncIndex.documents` (see **VEC** section's host class).

- [ ] **DOC-REST-00** `Index.documents` (property) — lazy `Documents` namespace accessor — `pinecone/index/__init__.py`
- [ ] **DOC-ASYNC-00** `AsyncIndex.documents` (property) — lazy `AsyncDocuments` namespace accessor — `pinecone/async_client/async_index.py`
- [ ] **DOC-REST-01** `Documents.upsert()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-01** `AsyncDocuments.upsert()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-02** `Documents.batch_upsert()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-02** `AsyncDocuments.batch_upsert()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-03** `Documents.search()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-03** `AsyncDocuments.search()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-04** `Documents.fetch()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-04** `AsyncDocuments.fetch()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-05** `Documents.delete()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-05** `AsyncDocuments.delete()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-06** `Documents.update()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-06** `AsyncDocuments.update()` — `pinecone/async_client/documents.py`
- [ ] **DOC-REST-07** `Documents.list()` — `pinecone/client/documents.py`
- [ ] **DOC-ASYNC-07** `AsyncDocuments.list()` — sync `def` returning an async iterator — `pinecone/async_client/documents.py`

*Note: pinecone-io/python-sdk-internal#494 restored `.documents` as a lazy
namespace and deprecated flat `*_documents` methods that previously lived
directly on `Index`/`AsyncIndex`. Confirm no such flat methods remain (none
were found as of this writing) and that any deprecation warnings fire
correctly if they do.*

---

## 12. Admin — Organizations — `ADM-ORG`

REST only.

- [ ] **ADM-ORG-REST-01** `Organizations.list()` — `pinecone/admin/organizations.py`
- [ ] **ADM-ORG-REST-02** `Organizations.describe()` — `pinecone/admin/organizations.py`
- [ ] **ADM-ORG-REST-03** `Organizations.update()` — `pinecone/admin/organizations.py`
- [ ] **ADM-ORG-REST-04** `Organizations.delete()` — `pinecone/admin/organizations.py`

---

## 13. Admin — Projects — `ADM-PRJ`

REST only.

- [ ] **ADM-PRJ-REST-01** `Projects.list()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-02** `Projects.create()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-03** `Projects.describe()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-04** `Projects.describe_by_name()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-05** `Projects.exists()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-06** `Projects.update()` — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-07** `Projects.delete_with_cleanup()` — deletes all indexes/collections/backups in the project first — `pinecone/admin/projects.py`
- [ ] **ADM-PRJ-REST-08** `Projects.delete()` — `pinecone/admin/projects.py`

---

## 14. Admin — API Keys — `ADM-KEY`

REST only.

- [ ] **ADM-KEY-REST-01** `ApiKeys.list()` — `pinecone/admin/api_keys.py`
- [ ] **ADM-KEY-REST-02** `ApiKeys.create()` — `pinecone/admin/api_keys.py`
- [ ] **ADM-KEY-REST-03** `ApiKeys.describe()` — `pinecone/admin/api_keys.py`
- [ ] **ADM-KEY-REST-04** `ApiKeys.update()` — `pinecone/admin/api_keys.py`
- [ ] **ADM-KEY-REST-05** `ApiKeys.delete()` — `pinecone/admin/api_keys.py`

---

## 15. Admin — Users — `ADM-USR`

REST only.

- [ ] **ADM-USR-REST-01** `Users.list()` — `pinecone/admin/users.py`
- [ ] **ADM-USR-REST-02** `Users.describe()` — `pinecone/admin/users.py`
- [ ] **ADM-USR-REST-03** `Users.delete()` — `pinecone/admin/users.py`

---

## 16. Admin — Invites — `ADM-INV`

REST only.

- [ ] **ADM-INV-REST-01** `Invites.list()` — `pinecone/admin/invites.py`
- [ ] **ADM-INV-REST-02** `Invites.create()` — `pinecone/admin/invites.py`
- [ ] **ADM-INV-REST-03** `Invites.describe()` — `pinecone/admin/invites.py`
- [ ] **ADM-INV-REST-04** `Invites.delete()` — `pinecone/admin/invites.py`
- [ ] **ADM-INV-REST-05** `Invites.resend()` — `pinecone/admin/invites.py`

---

## 17. Admin — Service Accounts — `ADM-SVC`

REST only.

- [ ] **ADM-SVC-REST-01** `ServiceAccounts.list()` — `pinecone/admin/service_accounts.py`
- [ ] **ADM-SVC-REST-02** `ServiceAccounts.create()` — `pinecone/admin/service_accounts.py`
- [ ] **ADM-SVC-REST-03** `ServiceAccounts.describe()` — `pinecone/admin/service_accounts.py`
- [ ] **ADM-SVC-REST-04** `ServiceAccounts.update()` — `pinecone/admin/service_accounts.py`
- [ ] **ADM-SVC-REST-05** `ServiceAccounts.delete()` — `pinecone/admin/service_accounts.py`
- [ ] **ADM-SVC-REST-06** `ServiceAccounts.rotate_secret()` — `pinecone/admin/service_accounts.py`

---

## 18. Admin — Role Bindings — `ADM-RB`

REST only.

- [ ] **ADM-RB-REST-01** `RoleBindings.list()` — `pinecone/admin/role_bindings.py`
- [ ] **ADM-RB-REST-02** `RoleBindings.create()` — `pinecone/admin/role_bindings.py`
- [ ] **ADM-RB-REST-03** `RoleBindings.describe()` — `pinecone/admin/role_bindings.py`
- [ ] **ADM-RB-REST-04** `RoleBindings.delete()` — `pinecone/admin/role_bindings.py`

---

## Appendix: Non-method backcompat surface

These are backwards-compatibility shims that are **not methods** — module- or
class-level aliases. Included for completeness since the review should
confirm they still resolve correctly, even though they don't get a
method-style ID.

- [ ] **APPX-01** `pinecone.ValidationError` — module `__getattr__`-based deprecated alias for `PineconeValueError`, emits `DeprecationWarning` — `pinecone/__init__.py`
- [ ] **APPX-02** `pinecone.PineconeAsyncio` — top-level alias: `from pinecone.async_client.pinecone import AsyncPinecone as PineconeAsyncio` — `pinecone/__init__.py`
- [ ] **APPX-03** `pinecone.admin.resources.ApiKeyResource` — class alias for `pinecone.admin.api_keys.ApiKeys` — `pinecone/admin/resources/api_key.py`
- [ ] **APPX-04** `pinecone.admin.resources.OrganizationResource` — class alias for `pinecone.admin.organizations.Organizations` — `pinecone/admin/resources/organization.py`
- [ ] **APPX-05** `pinecone.admin.resources.ProjectResource` — class alias for `pinecone.admin.projects.Projects` — `pinecone/admin/resources/project.py`

---

## Coverage summary

| Section | Resource | REST | ASYNC | GRPC | Legacy/Model | Total rows |
|---|---|---|---|---|---|---|
| 0 | Client construction & lifecycle | 15 | 15 | 1 | 3 | 34 |
| 0b | Admin construction & lifecycle | 13 | — | — | — | 13 |
| 1 | Index (control plane) | 7 | 7 | — | 14 | 28 |
| 2 | Collections | 4 | 4 | — | 8 | 16 |
| 3 | Backups | 9 | 9 | — | 8 | 26 |
| 4 | Backup Schedules | 9 | 9 | — | — | 18 |
| 5 | Restore Jobs | 2 | 2 | — | 4 | 8 |
| 6 | Inference | 8 | 8 | — | — | 16 |
| 7 | Assistants | 21 | 21 | — | 16 | 58 |
| 8 | Data plane — Vectors | 16 | 16 | 22 | 1 | 55 |
| 9 | Data plane — Namespaces | 5 | 5 | 5 | — | 15 |
| 10 | Data plane — Bulk import | 5 | 5 | 5 | — | 15 |
| 11 | Data plane — Documents | 8 | 8 | — | — | 16 |
| 12 | Admin — Organizations | 4 | — | — | — | 4 |
| 13 | Admin — Projects | 8 | — | — | — | 8 |
| 14 | Admin — API Keys | 5 | — | — | — | 5 |
| 15 | Admin — Users | 3 | — | — | — | 3 |
| 16 | Admin — Invites | 5 | — | — | — | 5 |
| 17 | Admin — Service Accounts | 6 | — | — | — | 6 |
| 18 | Admin — Role Bindings | 4 | — | — | — | 4 |
| Appendix | Non-method aliases | — | — | — | 5 | 5 |

**Total: 368 rows** across every transport, resource, and backcompat shim
identified in this pass (per-section counts above are approximate — some
rows cover a paired `__enter__`/`__exit__` or similar under one ID; run
`grep -c '^\- \[ \]' METHOD-INVENTORY-TEMPLATE.md` for the exact live count).
