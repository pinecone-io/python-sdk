"""Gate: pinecone/_internal/ helpers must not go orphaned unnoticed.

`scripts/find_orphaned_helpers.find_orphaned_helpers` re-derives, from the
current tree, which module-level helper functions are exercised only by
their own unit tests (or by nothing at all) with no caller reachable from
outside `pinecone/_internal/`. Ticket #337 found `build_create_body` in
that state; this test is the mechanism it built to keep finding the next
one, rather than a one-time list of today's findings.

`ACKNOWLEDGED_ORPHANS` is the known, already-filed backlog (issue #479) —
not a growing allowlist. Equality (not subset) is asserted both ways: a
name outside this set fails the test as a newly discovered orphan, and a
name in this set that the detector no longer reports (because someone
deleted or wired it up) also fails, so the set can't go stale silently.
"""

from __future__ import annotations

from scripts.find_orphaned_helpers import find_orphaned_helpers

ACKNOWLEDGED_ORPHANS = frozenset(
    {
        "build_byoc_body",
        "build_integrated_body",
        "validate_read_capacity",
        "_normalize_schema",
        "chunked",
        "with_progress",
        "validate_byoc_inputs",
        "validate_integrated_inputs",
        "_validate_deletion_protection",
    }
)


def test_no_new_orphaned_helpers() -> None:
    report = find_orphaned_helpers()
    found = set(report.orphaned) | set(report.unreferenced)

    new = found - ACKNOWLEDGED_ORPHANS
    assert not new, (
        f"new orphaned helper(s) with no production caller: {sorted(new)}. "
        "See ticket #337 and https://github.com/pinecone-io/python-sdk-internal/issues/479 "
        "for the pattern; delete the helper (and its now-dead test), wire it up if it "
        "was meant to be, or add it to ACKNOWLEDGED_ORPHANS with a tracking issue."
    )

    stale = ACKNOWLEDGED_ORPHANS - found
    assert not stale, (
        f"ACKNOWLEDGED_ORPHANS is stale: {sorted(stale)} no longer detected as orphaned "
        "(fixed, deleted, or renamed) -- remove from the set."
    )


def test_detector_flags_a_known_dead_function() -> None:
    """Self-test: the detector must flag a real no-caller function.

    Guards against the detector accidentally treating everything as live
    (e.g. an overbroad "live" seed) by asserting on one of the acknowledged
    orphans directly, independent of the acknowledged-set bookkeeping above.
    """
    report = find_orphaned_helpers()
    assert "chunked" in report.orphaned
