# 2026-07: admin and OAuth — four new resources, nothing breaking

Release notes for the **admin** and **OAuth** surfaces at `2026-07`. Both are
additive: **no field on any pre-existing request or response changed, was
renamed, or was removed.** Existing `Admin` code compiles and behaves as it did
at `2025-10`. What is new is four namespaces — users, invites, service accounts,
and role bindings — which together make organization membership and RBAC
manageable from the SDK for the first time.

`Admin` is **synchronous only**. There is no `AsyncioAdmin` or `AsyncAdmin`, and
none is planned for this release; admin calls are infrequent control-plane
operations.

## The version header

`ADMIN_API_VERSION` is now `2026-07`, sent as `X-Pinecone-Api-Version` on every
admin request *and* on the OAuth token exchange. Nothing else about
authentication changed: the client-credentials grant, the audience, and the
token shape are identical to `2025-10`.

The OAuth surface is a **pure version bump**. Once the version string is
normalized away, the `2026-07` OAuth spec is equivalent to both `2026-04` and
`2025-10`: same single operation, same request body, same response shapes, same
error schema. Nothing about it needs migrating.

## Operation classification

Thirty-three operations ship across the two surfaces.

### Version bump only — behaviour unchanged (15)

The request and response schemas of these are unchanged from `2025-10`; only the
header value differs.

| Operation | SDK method |
| --- | --- |
| `get_token` | internal (the OAuth exchange `Admin` performs on construction and refresh) |
| `list_projects` | `admin.projects.list()` |
| `create_project` | `admin.projects.create()` |
| `fetch_project` | `admin.projects.describe()` |
| `update_project` | `admin.projects.update()` |
| `delete_project` | `admin.projects.delete()` |
| `list_organizations` | `admin.organizations.list()` |
| `fetch_organization` | `admin.organizations.describe()` |
| `update_organization` | `admin.organizations.update()` |
| `delete_organization` | `admin.organizations.delete()` |
| `list_project_api_keys` | `admin.api_keys.list()` |
| `create_api_key` | `admin.api_keys.create()` |
| `fetch_api_key` | `admin.api_keys.describe()` |
| `update_api_key` | `admin.api_keys.update()` |
| `delete_api_key` | `admin.api_keys.delete()` |

### Net-new — additive (18)

| Operation | SDK method |
| --- | --- |
| `list_users` | `admin.users.list()` |
| `fetch_user` | `admin.users.describe()` |
| `delete_user` | `admin.users.delete()` |
| `list_invites` | `admin.invites.list()` |
| `create_invite` | `admin.invites.create()` |
| `fetch_invite` | `admin.invites.describe()` |
| `delete_invite` | `admin.invites.delete()` |
| `resend_invite` | `admin.invites.resend()` |
| `list_service_accounts` | `admin.service_accounts.list()` |
| `create_service_account` | `admin.service_accounts.create()` |
| `fetch_service_account` | `admin.service_accounts.describe()` |
| `update_service_account` | `admin.service_accounts.update()` |
| `delete_service_account` | `admin.service_accounts.delete()` |
| `rotate_service_account_secret` | `admin.service_accounts.rotate_secret()` |
| `list_role_bindings` | `admin.role_bindings.list()` |
| `create_role_binding` | `admin.role_bindings.create()` |
| `fetch_role_binding` | `admin.role_bindings.describe()` |
| `delete_role_binding` | `admin.role_bindings.delete()` |

### Deliberately not shipped

`db_metrics` — the Prometheus service-discovery surface — is **not part of the
2026-07 SDK release** and has no client method. Its one operation is a known,
deliberate omission rather than an oversight, deferred to a later release. There
is nothing to migrate; code cannot have depended on it, because it never
shipped.

## New models

All additive. Nothing pre-existing was touched.

| Model | Where it comes from |
| --- | --- |
| `UserModel`, `UserList` | the users operations |
| `InviteModel`, `InviteList`, `InviteStatus` | the invites operations |
| `ServiceAccountModel`, `ServiceAccountList`, `ServiceAccountWithSecret` | the service-account operations |
| `RoleBindingModel`, `RoleBindingList`, `RoleBindingInput`, `PrincipalType`, `ResourceType`, `RoleName` | the role-binding operations |

`RoleBindingInput`, `PrincipalType`, `ResourceType`, and `RoleName` are importable
from `pinecone.models.admin`. Everywhere a role binding is accepted, a plain
`dict` works too, so the enums are a convenience rather than a requirement.

The four listing operations (`users`, `invites`, `service_accounts`,
`role_bindings`) return a lazy `Paginator` rather than an eager list, unlike the
older `projects`/`organizations`/`api_keys` listings which return `*List`
objects. Iterate it, or call `.to_list()`; no request is sent until you do.

## End-to-end: standing up a scoped service account and inviting a teammate

The three new resources compose — this is the workflow they exist for.

```python
from pinecone import Admin
from pinecone.models.admin import PrincipalType, ResourceType, RoleName

admin = Admin(client_id="...", client_secret="...")
project = admin.projects.create(name="search-prod")

# 1. A service account for CI, with no permissions yet.
created = admin.service_accounts.create(name="ci-search-prod")
store_secret(created.client_secret)  # returned exactly once

# 2. Grant it ownership of that one project. Bindings are what confer access;
#    a service account with none can get a token but do nothing with it.
admin.role_bindings.create(
    principal_type=PrincipalType.SERVICE_ACCOUNT,
    principal_id=created.service_account.id,
    resource_type=ResourceType.PROJECT,
    resource_id=project.id,
    role=RoleName.PROJECT_OWNER,
)

# 3. Invite a human, with their initial roles in the same call. At least one
#    organization-scoped membership role is required.
invite = admin.invites.create(
    email="teammate@example.com",
    role_bindings=[
        {"resource_type": "organization", "role": "OrgMember"},
        {"resource_type": "project", "role": "ProjectViewer",
         "resource_id": project.id},
    ],
)

# 4. Read any principal's access back through role_bindings, not through the
#    principal's own model — no other namespace carries bindings.
for binding in admin.role_bindings.list(
    principal_type="service_account",
    principal_id=created.service_account.id,
):
    print(binding.role, binding.resource_type, binding.resource_id)
```

Two things to internalize from that shape:

- **Role bindings are not part of any principal's representation.**
  `UserModel`, `ServiceAccountModel`, and `InviteModel` do not carry them, and
  `create` does not echo the bindings it was given. `admin.role_bindings.list()`
  filtered by `principal_type` and `principal_id` is the only way to read them.
- **Bindings are immutable.** There is no update. Changing a role is
  `create` for the new one then `delete` for the old one, in that order —
  deleting first can strip a principal's last organization-membership binding,
  which the server refuses with a `409`.

## Invite status lifecycle — `list` never shows accepted invites

`InviteStatus` has three values: `pending`, `expired`, and `processed`.
`admin.invites.list()` returns only the first two.

**An accepted invite disappears from the listing.** It has not been deleted, and
`admin.invites.describe(invite_id=...)` still returns it with
`status == InviteStatus.PROCESSED`. Do not treat absence from `list()` as proof
that an invite never existed, or as licence to send a duplicate — a second
`create` for an address that already belongs to a member is a `409`.

```python
ids = {i.id for i in admin.invites.list()}
"9c8e3528-..." in ids                      # False — could mean accepted
admin.invites.describe(invite_id="9c8e3528-...").status  # 'processed'
```

Once an invite is accepted, the invitee is an organization member: manage them
through `admin.users` and `admin.role_bindings`, not through `admin.invites`.
`delete` and `resend` on a processed invite are both a `409`.

## `PINECONE_CONTROLLER_HOST` is now honored, plus `host` / `oauth_url`

Previously `Admin` ignored `PINECONE_CONTROLLER_HOST` and always talked to
`https://api.pinecone.io`. It now applies the same host resolution as
`Pinecone`: the `host` keyword first, then `PINECONE_CONTROLLER_HOST`, then the
default. A value with no scheme is prefixed with `https://`.

**This is a behaviour change for anyone who had that variable set in an
environment where `Admin` also runs**, even though no API surface changed —
admin traffic that used to go to production will now follow the variable.

```python
admin = Admin(client_id="...", client_secret="...", host="http://localhost:5080")
```

`oauth_url` is a second new keyword, pointing the token exchange somewhere other
than the production endpoint. It takes the **full URL including the path**, and
has no environment-variable fallback.

```python
admin = Admin(
    client_id="...", client_secret="...",
    host="http://localhost:5080",
    oauth_url="http://localhost:5080/oauth/token",
)
```

Both are keyword-only and intended for local simulators and private
deployments. Leave both unset against production.

## Token refresh

`Admin` now keeps its own Bearer token current: it re-mints a margin ahead of
the stated expiry, and retries a request once against a fresh token if one still
comes back `401`. A long-lived `Admin` no longer starts returning bare `401`s
after its first token lapses, and threads sharing one `Admin` cost a single
token exchange between them rather than one each.

Passing your own `Authorization` entry in `additional_headers` opts out of
refresh entirely — the token is then yours to manage. Matching is
case-sensitive, so only the exact spelling `"Authorization"` takes over.

## Project deletion now clears assistants too

`admin.projects.delete()` requires an empty project, and indexes, collections,
assistants, and backups each block it with a `412` naming what is left. API keys
are *not* a blocker — they are deleted with the project.

`admin.projects.delete_with_cleanup()` clears **all four** of those, assistants
included. Earlier releases left assistants behind, so a project holding one
still failed the final delete after a nominally successful cleanup. That gap is
closed.
