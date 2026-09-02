# Coverage measurement

The unit suite runs under `pytest-cov` in CI (`.github/workflows/on-push.yml`,
`unit-tests` job) so "is this code covered?" has a visible answer without
anyone running a command locally. This started from
[#48](https://github.com/pinecone-io/python-sdk-internal/issues/48), which
found that three real gaps (DataFrame tests skipped via `importorskip`,
`cargo test` never linking, dead `tqdm` branches) had gone unnoticed for a
while because nothing measured or preserved a coverage signal.

## Running it locally

```bash
uv run pytest tests/unit --cov --cov-report=term-missing
```

`source = ["pinecone"]` and the report settings live in `pyproject.toml`
under `[tool.coverage.run]` / `[tool.coverage.report]`, so a bare `--cov`
(no target) picks up the right package.

## CI behavior

- Every matrix leg (`py3.10`–`py3.14`) runs with `--cov`, but only the
  `py3.12` leg posts a markdown table to the GitHub Actions job summary —
  posting it five times over would just be duplicate noise on the PR
  checks tab.
- There is **no failure threshold**. A floor set before there's real data
  on where coverage naturally sits gets gamed with tests that execute
  lines without asserting anything — worse than no signal. Revisit this
  after a few weeks of observed numbers across PRs.

## Scope decisions

- `pinecone/preview/**` is included. It's hand-written code on the same
  release train as everything else in `pinecone/`, not a separate or
  generated surface.
- No `omit`/`exclude` list was needed: there's no generated-code directory
  under `pinecone/` (the gRPC/proto layer lives in `rust/` and isn't
  measured by this Python coverage run at all).
- A handful of modules report **0%** and aren't real gaps: they're
  backwards-compatibility re-export shims for pre-`python-sdk2` import
  paths (e.g. `pinecone/exceptions.py`, `pinecone/utils/response_info.py`,
  `pinecone/admin/resources/__init__.py` and its per-resource siblings,
  the `pinecone/db_control/enums/*` and `pinecone/db_control/models/*`
  compat modules). Nothing in the test suite imports the legacy paths, so
  they never execute. Don't chase these to 100%; a test that only imports
  a shim to satisfy coverage asserts nothing real.

## Baseline (2026-08-23)

Measured on `main` via `uv run pytest tests/unit --cov --cov-report=term-missing`
(macOS; `tests/unit/test_socket_options.py`'s two platform-specific failures
are a known local-only quirk — see the CI-only `on-push.yml` matrix, which
runs on `ubuntu-latest` and doesn't hit them):

| | |
|---|---|
| Statements | 11114 |
| Missed | 442 |
| **Total** | **96%** |

Recorded here so future erosion is detectable against a real number instead
of a vibe. Update this table the next time someone does a deliberate pass
over coverage, not on every PR.
