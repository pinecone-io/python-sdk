"""Gate: pinecone/_internal/ helpers must not go orphaned unnoticed.

`scripts/find_orphaned_helpers.find_orphaned_helpers` re-derives, from the
current tree, which module-level helper functions are exercised only by
their own unit tests (or by nothing at all) with no caller reachable from
outside `pinecone/_internal/`. Ticket #337 found `build_create_body` in
that state; this test is the mechanism it built to keep finding the next
one, rather than a one-time list of today's findings.

`ACKNOWLEDGED_ORPHANS` is the known, already-filed backlog (issue #479) —
not a growing allowlist. `PENDING_CONSUMERS` is the opposite direction: a
helper landed deliberately ahead of the ticket that calls it. Equality (not
subset) is asserted both ways over the two sets together: a name outside them
fails the test as a newly discovered orphan, and a name in either of them that
the detector no longer reports (because someone deleted or wired it up) also
fails, so no set can go stale silently.
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

#: Helpers merged before their caller exists, each with the ticket that will
#: consume them. #498 landed the legacy-to-2026-07 translation module as pure
#: functions so #500 (create) and #501 (configure) could share one
#: implementation across the sync and async surfaces; wiring it in is those
#: tickets' work, and this set is what forces these names off it when they do.
PENDING_CONSUMERS = frozenset()

EXPECTED_ORPHANS = ACKNOWLEDGED_ORPHANS | PENDING_CONSUMERS


def test_no_new_orphaned_helpers() -> None:
    report = find_orphaned_helpers()
    found = set(report.orphaned) | set(report.unreferenced)

    new = found - EXPECTED_ORPHANS
    assert not new, (
        f"new orphaned helper(s) with no production caller: {sorted(new)}. "
        "See ticket #337 and https://github.com/pinecone-io/python-sdk-internal/issues/479 "
        "for the pattern; delete the helper (and its now-dead test), wire it up if it "
        "was meant to be, or add it to ACKNOWLEDGED_ORPHANS with a tracking issue."
    )

    stale = EXPECTED_ORPHANS - found
    assert not stale, (
        f"the expected-orphan sets are stale: {sorted(stale)} no longer detected as "
        "orphaned (fixed, deleted, or renamed) -- remove from ACKNOWLEDGED_ORPHANS "
        "or PENDING_CONSUMERS."
    )


def test_detector_flags_a_known_dead_function() -> None:
    """Self-test: the detector must flag a real no-caller function.

    Guards against the detector accidentally treating everything as live
    (e.g. an overbroad "live" seed) by asserting on one of the acknowledged
    orphans directly, independent of the acknowledged-set bookkeeping above.
    """
    report = find_orphaned_helpers()
    assert "chunked" in report.orphaned
