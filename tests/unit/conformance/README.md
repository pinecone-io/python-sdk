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
   claims. For gRPC rpcs use `claim.assert_grpc_request(full_method)` with the
   invoked full method name (e.g. `/VectorService/Upsert`).
2. **API version** — `claim.assert_api_version(request_or_headers_or_metadata)`
   requires `X-Pinecone-Api-Version: 2026-07` on the wire (gRPC: in call
   metadata). The expected value is hardcoded here on purpose: it must not be
   imported from `pinecone._internal.constants`, or a wrong constant would
   certify itself.
3. **Schema round-trip** — `claim.assert_roundtrip(ModelCls, payload,
   optional_absent=[...])` decodes the payload into the msgspec model,
   re-encodes it, and requires nothing to be lost. `optional_absent` must name
   at least one optional field whenever the schema has any: the reduced
   payload (those fields stripped) must still decode, and the model must not
   invent values for them.

   Operations whose spec declares no success response body — 202/204 deletes
   and the like — satisfy this category with
   `claim.assert_no_response_body(returned)` instead, where `returned` is the
   SDK call's return value and must be `None`. Which of the two applies is not
   the test's choice: the manifest records `success_body` per operation from
   the OAS, and each method refuses the operations the other one owns. So an
   operation with a real response schema cannot dodge the round-trip by
   claiming to have no body.

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
- A claimed test that skips any mandatory assertion fails at teardown.
- A claimed test that fails, errors, or is skipped does not count under
  `--verify` or `--gate`.
- Claims for ids not in the specs fail at import, and `--verify` re-checks
  against the live specs.
