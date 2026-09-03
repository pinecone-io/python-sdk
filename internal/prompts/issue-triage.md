# Pre-release issue triage

Reconcile the open issue backlog of `pinecone-io/python-sdk-internal` with the
actual state of `main`, then produce an ordered work queue for the upcoming
public release.

Background: the `2026-07` API-version branch was merged into `main` (merge
commit `0bb4c728`). Before that merge, `main` and `2026-07` had substantially
different code, and most issues were written against `2026-07`. Separately,
GitHub API failures during that period meant some issues were never closed when
their PR merged. So the backlog now contains three kinds of noise on top of the
real work: issues that are silently done, issues whose text describes a codebase
that no longer exists, and issues that contradict each other.

You are the orchestrator. You will fan investigation out to read-only
subagents, then do every GitHub write yourself, serially, after a human
checkpoint.

---

## Non-negotiables

1. **Subagents are read-only.** They never run `gh issue close`, `gh issue edit`,
   `gh issue comment`, `gh label`, or any other mutating command; they never
   commit, never push, and never edit tracked files. They read, investigate, and
   return a verdict. All writes happen in Phase 4, from you, one at a time.
2. **No close without code-level proof.** A merged PR that says "Closes #N" is a
   lead, not evidence. The evidence is the change being present in the working
   tree at `main` HEAD, cited as `path/to/file.py:123`. PRs get reverted, and
   PR #549 was an explicit conflict-resolution-only merge, so a fix that existed
   on `2026-07` may not have survived onto `main`.
3. **Never destroy issue content.** When you rewrite a body, preserve the
   original verbatim inside a collapsed `<details><summary>Original issue text
   (pre-triage)</summary>` block at the bottom. The original measurements are
   evidence even when they are now stale.
4. **Re-verify, don't relay.** Every factual claim in an issue was measured at
   some commit, often `origin/2026-07` at a named SHA. Treat it as a hypothesis
   and re-measure at `main` HEAD before acting on it.
5. **Do not create new issues** for problems you discover along the way, except
   when splitting the unfinished residue off a partially-done issue. Collect
   everything else in a "found during triage" list for the human.

## Source-of-truth hierarchy

When sources disagree, this order settles it:

1. **pinecone-db** (the Rust monorepo) at the build actually serving the
   `2026-07` API version — authoritative for what the backend does. Use the
   `pinecone-db-lookup` skill; do not grep the tree by hand, the repo is a flat
   workspace of ~150 crates and route X is never where you would guess.
2. **Observed behavior** against the real API. Authoritative for what actually
   happens today, but note the environment and the date — several issues in this
   backlog (#312, #319) exist only because a specific deployment is stale.
3. **This SDK at `main` HEAD** — authoritative for what the SDK does.
4. **minicone** — a simulator. Useful for cheap reproduction, never
   authoritative for backend semantics. If minicone and pinecone-db disagree,
   that is a minicone bug: file it with the `minicone-bug-report` skill and say
   so in the verdict. Use `minicone-testing` to drive it.
5. **The OpenAPI spec** — frequently stale in this backlog. Never authoritative
   on its own; a spec-vs-backend disagreement is a finding, not a tiebreak.
6. **Issue text, PR descriptions, code comments** — hypotheses.

Also treat **DECISION comments on the epoch tracker #87** (77 comments) as
binding on scope and design, unless a later DECISION supersedes them — the
guided-hard-break decision, for example, was superseded via #505/#513. Build a
short ledger of live vs superseded decisions early; several issues are only
stale because they cite a superseded one.

---

## Phase 0 — Snapshot (you, no subagents)

Work under `internal/sessions/<YYYY-MM-DD>-issue-triage/` (gitignored, see
`internal/sessions/README.md`). Everything below lands there.

1. Record `git rev-parse HEAD` for `main`, and the `2026-07` merge commit.
2. Snapshot every open issue to `issues/<number>.json` — number, title, body,
   all comments, labels, milestone, timestamps, timeline cross-references,
   linked PRs. One API pass, so ~20 subagents can read files instead of
   hammering the GitHub API and re-triggering the rate-limit problem that
   created this mess.
3. Snapshot merged PRs since 2026-08-01 with title, body, merge commit, base
   branch, and changed-file list, to `prs.json`.
4. Snapshot issues **closed** since 2026-08-15 to `closed.json` — you need these
   for the reverse sweep in Phase 2.
5. Write `state.json` tracking per-issue status (`pending` /
   `assigned` / `verdict` / `written`) so a crashed run resumes instead of
   restarting.

## Phase 1 — Mechanical pre-pass (you)

Cheap signals, no judgment yet. Write `leads.md`:

- Issues whose number appears in a merged PR title, body, or commit message.
- Issues referenced by `Closes|Fixes|Resolves #N` anywhere in git log since the
  merge base.
- Duplicate candidates: issues whose titles or bodies overlap heavily.
- Cross-reference clusters: issues that cite each other (#333 ↔ #466 ↔ #540 is a
  known contradiction cluster about `read_capacity` on
  `create_index_from_backup`; there will be others).
- Issues whose body quotes a `origin/2026-07` SHA, a file path, or a line number
  — all mechanically stale after the merge.

## Phase 2 — Parallel investigation (subagents)

Batch issues into groups of **3–6 that share a lane or a theme** (`lane:models`,
`lane:rest`, `lane:grpc`, `lane:docs`, `lane:tooling`, the bulk-rewrite
milestone, the SPEC-vs-BACKEND questions). Shared context makes a batch cheaper
than the sum of its issues, and one agent seeing a whole cluster is what catches
duplicates and contradictions. Run 6–8 batches concurrently.

Model: Sonnet is the default. Escalate to Opus for the "2026-07 open questions"
milestone (36 issues of spec-vs-backend reconciliation) and for anything whose
resolution needs pinecone-db source reading.

Timebox each issue. If an investigation exceeds it, return
`UNRESOLVABLE-COSTLY` with what was learned and what remains — do not let one
issue eat a batch.

### Verdict vocabulary

Every issue gets exactly one:

| Verdict | Meaning | Action in Phase 4 |
|---|---|---|
| `DONE-VERIFIED` | Change present at `main` HEAD, cited by file:line, plus the PR that landed it | Close as completed, with evidence comment |
| `DONE-PARTIAL` | Some acceptance criteria met, others not | Keep open; rewrite down to the residue only |
| `OBSOLETE` | Premise no longer holds — code deleted, surface removed, backend changed, or the merge mooted it | Close as not planned, with reason |
| `DUPLICATE` | Covered by another open issue | Merge unique content into the survivor first, then close |
| `NOT-OURS` | Real problem, but the fix belongs to pinecone-db, the OAS, minicone, or a deployment | Do **not** close. Label `blocked`, record where it was relayed and the external ticket, drop from the SDK work queue |
| `STILL-VALID-STALE-TEXT` | Real and wanted, but the text misdescribes the current tree | Rewrite body; keep open |
| `STILL-VALID-CURRENT` | Real, wanted, accurate | Leave the text alone; classify for the queue |
| `NEEDS-HUMAN` | Evidence genuinely conflicts, or the call is a product/API-shape decision | Escalate with both sides stated; never guess |
| `WONTFIX-PROPOSED` | Valid but not worth doing | Propose only; the human decides |

### What each subagent returns, per issue

```
issue: <number>
verdict: <from the table>
confidence: high | medium | low
evidence:
  - <file:line at main HEAD, PR number, pinecone-db citation, or observed response>
current_state: <2-4 sentences: what is actually true today>
stale_claims: <each specific sentence in the issue that is now wrong, and why>
proposed_title: <only if it should change>
proposed_body: <full rewritten body, only for DONE-PARTIAL / STILL-VALID-STALE-TEXT>
release_blocking: yes | no | needs-human, with the reason
one_way_door: yes | no   # does shipping lock in a public API shape?
depends_on: [<issue numbers>]
lane: <label>
files_touched: [<paths an implementer would edit>]
size: S | M | L
found_during_triage: <new problems noticed; do not file them>
```

### Staleness patterns to sweep for specifically

- `origin/2026-07 <sha>` measurements — re-measure at `main` HEAD, restate.
- File paths and line numbers that moved in the merge.
- "on `main`" claims written when `main` was pre-merge — the sentence may now be
  false in either direction.
- Citations of decisions later superseded (check #87's ledger).
- "weakened because of a minicone bug" — #445 says several of these are fixed
  upstream now; verify per-instance.
- Acceptance criteria assuming the pre-merge module layout.
- References to issues that have since been closed.

### Verifying a `DONE-VERIFIED` on a test-adding issue

The house standard here is that a test must fail without the fix. For any issue
whose acceptance criteria are "add a test", the test existing is not enough:
confirm it is non-vacuous (assertions actually exercise the claim). #465 and
#422 are examples of tests that read as though they cover something and do not.
If the test is vacuous, the verdict is `DONE-PARTIAL`, not `DONE-VERIFIED`.

### Reverse sweep (assign this as its own batch)

Over `closed.json`: find issues closed as completed whose change is **not**
present at `main` HEAD. The 2026-07 → main merge, and PR #549 in particular,
could have dropped work during conflict resolution. Verdict for these is
`REOPEN`, with the same evidence requirements.

## Phase 2b — Cross-issue reconciliation (one agent, after Phase 2)

Feed it every verdict. It produces `reconciliation.md`:

- Contradictions between verdicts (two issues asserting opposite backend
  behavior — settle against pinecone-db, or mark both `NEEDS-HUMAN`).
- Duplicate clusters, with a nominated survivor and the content to merge in.
- Dependency ordering, including issues whose verdict depends on another's.
- The state of the #87 epoch tracker's checklists versus reality.
- Any verdict whose evidence does not actually support it — push back.

## Phase 3 — Human checkpoint (stop here)

Present, and write nothing to GitHub until approved:

1. **Closes** — a single table: issue, verdict, one-line evidence. Reviewed as a
   list, not one by one.
2. **Reopens** — same shape.
3. **Body rewrites** — grouped by lane, with a diff or before/after summary.
4. **Escalations** — every `NEEDS-HUMAN` and `WONTFIX-PROPOSED`, each stated as
   a decision with options, not an open question.
5. **The draft work queue** (Phase 5 format), so the human sees the shape of the
   release before any writes land.

Per-category batch approval is fine. Proceeding without approval is not.

## Phase 4 — Execution (you, serially)

The GitHub API failures that caused this backlog drift are the thing to design
against. Therefore:

- One writer. No concurrent mutation.
- Before each write, re-read the issue's current state — someone may have
  touched it since Phase 0.
- Comment first with the evidence, then close. Use the right state reason:
  `gh issue close N -r completed` for work that landed, `-r "not planned"` for
  obsolete, duplicate, and wontfix.
- Log every write and its result to `ledger.jsonl`.
- Watch `gh api rate_limit`; pace writes and back off on failure rather than
  retrying in a tight loop.
- **Reconciliation pass at the end**: re-fetch every issue you touched and diff
  against the intended state. Retry anything that did not take, and report any
  write you could not confirm. Do not report success on an unverified write.

Labels to apply while you are in there: drop `needs-triage` once triaged; add
`agent-ready` when the acceptance criteria are self-contained and verified;
`blocked` for `NOT-OURS`; move post-release work off the release milestones.

Keep internal hostnames, credentials, and customer data out of any text you
write.

## Phase 5 — The work queue

Write `internal/reviews/release-work-queue.md` (tracked, committed — a durable
artifact, not session scratch).

Release-blocking bar, applied explicitly per issue:

- Wrong results, data loss, or silent data corruption.
- A public API shape that cannot be changed after release without a break —
  these are the **one-way doors** and they come first (#468 and #476 are
  examples: unsettled path-parameter and PATCH semantics).
- Documented behavior that does not exist, or existing behavior that is
  undocumented and surprising.
- A crash on a common path (#529 is one).

Everything else is post-release, and say so rather than leaving it ambiguous.

Structure:

1. **Decisions needed from a human** — at the top, blocking everything below.
2. **One-way doors** — ordered, must land before the cut.
3. **Release blockers** — ordered, grouped by lane so parallel implementers do
   not collide in the same files, with dependencies marked.
4. **Post-release** — with the milestone each should move to.
5. **Relayed elsewhere** — `NOT-OURS` items, with where they went.
6. **Found during triage** — not filed, for the human to decide on.

Each entry: issue number and link, one line on what it is, why it does or does
not block, lane, files touched, dependencies, S/M/L.

Finish with a summary: counts by verdict, how many writes were confirmed, what
could not be confirmed, and what remains unresolved.
