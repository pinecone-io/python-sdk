# Agent sessions

This directory holds **ephemeral working copies** used by long-running
agentic workflows (e.g. an orchestrator driving a review across every row of
[`internal/reviews/method-inventory.md`](../reviews/method-inventory.md)).

It is gitignored — nothing under here except this README is tracked.

## Convention

Each run gets its own subdirectory, named for the session:

```
internal/sessions/<session-id>/
    method-inventory.md   # copy of internal/reviews/method-inventory.md,
                           # checkboxes/notes updated in place as the run progresses
    ...                    # any other run-scoped scratch state
```

`<session-id>` should be descriptive and roughly sortable, e.g.
`2026-08-26-method-inventory-review` or a timestamp-prefixed slug.

Rules of thumb:

- **Copy, never edit in place.** The files under `internal/reviews/` are the
  source of truth. A session starts by copying the relevant source file(s)
  in here, then marks up its own copy.
- **Ephemeral.** Nothing here is expected to survive past the run it
  belongs to. If a run produces a durable finding or decision worth keeping,
  promote it back into `internal/reviews/` (or `internal/decisions/`, etc.)
  as its own commit — don't leave it stranded in a session directory.
- **Safe to delete.** Since nothing here is tracked, deleting a session
  directory (or the whole `internal/sessions/` tree) is always safe and
  loses no committed history.
