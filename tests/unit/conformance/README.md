# 2026-07 conformance suite

This package is the numerator of the epoch #87 coverage gate. Every test here
claims one or more operations from the 2026-07 API specs, and
`scripts/api_coverage.py` counts an operation as covered only when a claiming
test passes.

```
uv run python scripts/api_coverage.py --report   # covered/total per surface
uv run python scripts/api_coverage.py --gaps     # uncovered operation ids
uv run python scripts/api_coverage.py --verify   # run this suite; claimed tests must pass
uv run python scripts/api_coverage.py --gate     # the epoch stop condition
```

## Operation ids

`<surface>:<operationId>` for HTTP operations, where `<surface>` is the OAS
filename prefix (`admin`, `assistant_control`, `assistant_data`,
`assistant_evaluation`, `db_control`, `db_data`, `db_metrics`, `inference`,
`oauth`), and `db_data_grpc:<RpcName>` for proto rpcs. The full list lives in
`manifest_2026-07.json`, which is generated — never hand-edited — by
`scripts/api_coverage.py --write-manifest` from the spec checkout, and which
`--verify`/`--gate` re-check against the specs so it cannot go stale.

## The claim contract

A conformance test claims an operation with the decorator and must take the
`claim` fixture:

```python
from tests.unit.conformance import api_op


@api_op("db_control:list_indexes")
@respx.mock
def test_list_indexes_conformance(claim, indexes):
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=payload)
    )
    result = indexes.list()

    request = route.calls.last.request
    claim.assert_request(request)              # 1. method + path
    claim.assert_api_version(request)          # 2. X-Pinecone-Api-Version: 2026-07
    claim.assert_roundtrip(                    # 3. schema round-trip
        IndexList, payload, optional_absent=["pagination"]
    )
```

Every claimed operation must have **all three** categories asserted before the
test ends; the `claim` fixture fails the test at teardown otherwise:

1. **Method + path** — `claim.assert_request(request)` checks the *actual*
   request against the method and path template recorded in the manifest (from
   the OAS itself), so the test cannot assert a different endpoint than it
   claims. The expected path is the manifest's `base_path` (the path component
   of the spec's `servers` URL — `/assistant` for the assistant surfaces, empty
   for the rest) followed by the operation's path, so surfaces mounted under a
   prefix are still compared whole rather than by suffix. One surface's prefix
   is not derivable from its spec and comes from a registered override instead
   (see *Base-path overrides* below). For gRPC rpcs use
   `claim.assert_grpc_request(full_method)` with the invoked full method name
   (e.g. `/VectorService/Upsert`).
2. **API version** — `claim.assert_api_version(request_or_headers_or_metadata)`
   requires `X-Pinecone-Api-Version: 2026-07` on the wire (gRPC: in call
   metadata). The expected value is hardcoded here on purpose: it must not be
   imported from `pinecone._internal.constants`, or a wrong constant would
   certify itself.
3. **Schema round-trip** — `claim.assert_roundtrip(ModelCls, payload,
   optional_absent=[...])` first validates the payload against the operation's
   OAS response schema (see below), then decodes the payload into the msgspec
   model, re-encodes it, and requires nothing to be lost. `optional_absent` must name
   at least one optional field whenever the payload carries any: the reduced
   payload (those fields stripped) must still decode, and the model must not
   invent values for them. A payload that carries no optional field at all —
   the spec declares only required properties, or only ones this model treats
   as required — needs no `optional_absent`, because decoding it has already
   proved every optional field tolerates absence.

   Operations whose spec declares no success response body — 202/204 deletes
   and the like — satisfy this category with
   `claim.assert_no_response_body(returned)` instead, where `returned` is the
   SDK call's return value and must be `None`. Which of the two applies is not
   the test's choice: the manifest records `success_body` per operation from
   the OAS, and each method refuses the operations the other one owns. So an
   operation with a real response schema cannot dodge the round-trip by
   claiming to have no body. A 2xx whose schema is a bare `type: object` with
   no properties counts as no body: there is no field to lose, and modelling
   one would be the inflation `success_body` exists to prevent.

   A few SDK methods answer a bodyless operation with a struct they build
   themselves — `upsert_records` returns a caller-side record count. Those pass
   `client_side=[...]`, naming every field that comes back populated; any other
   populated field could only have come from a body the spec does not declare,
   and fails.

## Fixture validation against the OAS response schema

Round-tripping a fixture only proves the SDK is consistent with the test's own
invention. To make a claim mean *spec conformance*, the manifest also vendors,
per HTTP operation with a success body, the OAS response schema — refs
resolved, OAS 3.0 `nullable` translated to JSON Schema, annotations stripped,
and every object that declares `properties` sealed with
`additionalProperties: false` — and `assert_roundtrip` validates the payload
against it before the round-trip legs. A fixture carrying a key the spec never
declared, a wrong type, or a missing required property fails the test. gRPC
rpcs have no OAS schema and skip this leg.

### Divergence exceptions

Some operations deliberately implement backend behavior over the OAS. Those
are registered in the hand-maintained `divergences_2026-07.json`, which
`--write-manifest` folds into the manifest as a per-op
`divergence: {issue, reason, response_schema}` entry that switches fixture
validation to the documented alternative component schema. The rules, enforced
at generation time, at test time, and by `--gate`:

- Every exception must reference a `SPEC-vs-BACKEND` question issue by number
  and give a reason; the generator and the registry both refuse anything less,
  so silent exceptions cannot exist.
- `--gate` fails if a referenced issue is closed or is not labeled `question`:
  resolving the question means removing the exception (or fixing the spec) —
  not keeping a stale exemption.
- The manifest keeps recording what the OAS actually declares
  (`response_schema`) alongside the alternative, so the divergence stays
  visible instead of overwriting the spec's story.

Current exceptions: `assistant_control:update_assistant` (#170 — the backend
returns the full `Assistant` shape, not `UpdateAssistantResponse`).

### Base-path overrides

The response-side divergence has a request-side twin. A spec's `servers` URL
can omit a path prefix the deployed surface really carries, in which case the
derived `base_path` would make `assert_request` reject the path the SDK
correctly sends. `divergences_2026-07.json` therefore also carries
`base_path_overrides`, keyed by surface, under the same no-silent-exceptions
contract: question issue by number plus a reason. `--write-manifest` folds
each one into every operation of that surface as a `base_path_divergence:
{issue, reason, spec_base_path}` entry — the overridden prefix goes in
`base_path`, and what the spec actually declares stays visible in
`spec_base_path`. `--gate` checks the referenced issue the same way it checks
response-schema divergences, and the registry refuses an override that has
lost its issue number or reason.

Current overrides: `assistant_data` (#173 — the spec's
`https://{assistant_host}` server has no path, but the data plane is mounted
under `/assistant`).

Additional rules:

- Tests must exercise the SDK's real request path (public client classes, or
  internal clients constructed with the SDK's own version constants). Never
  pass a hardcoded `"2026-07"` into the client under test — the version on the
  wire has to come from the SDK, or the test certifies nothing. The gate
  independently checks that the constants are `2026-07`.
- Sync and async variants are separate tests; both may claim the same
  operation.
- A test may claim several operations by stacking `@api_op`; the recorder then
  needs `op=` on each assertion.
- Unknown operation ids raise `UnknownOperationError` at import time.

## Why inflation is detectable

- The denominator comes from parsing the spec files, not a hand-kept list.
- Expected method/path/service/rpc and whether a success response body exists
  come from the manifest, not the test.
- Response fixtures must validate against the OAS response schema vendored in
  the manifest (or an explicitly registered, issue-referenced divergence), so
  a test cannot claim an operation by mocking a shape the spec never promised.
- A claimed test that skips any mandatory assertion fails at teardown.
- A claimed test that fails, errors, or is skipped does not count under
  `--verify` or `--gate`.
- Claims for ids not in the specs fail at import, and `--verify` re-checks
  against the live specs.
