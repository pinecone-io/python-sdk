"""Delete leaked smoke-test resources.

Looks for resources whose names start with ``smoke-`` (the prefix every smoke
test uses) and deletes them best-effort. Useful after a killed pytest run
left indexes, collections, or assistants behind.

Usage, from the repository root::

    uv run python -m tests.smoke.scripts.cleanup_orphans

Run it as a module, not as a path. ``python <path>`` puts the *script's own*
directory on ``sys.path[0]``, so the ``tests.live_suite`` import below cannot
resolve; ``-m`` puts the working directory there instead. The path form appears
to work in a synced checkout only because the editable install adds the project
root via a ``.pth`` file — an environment that installs a built wheel has no
such entry, which is how both CI jobs ran this and cleaned up nothing (#412).

Optionally pass ``--dry-run`` to print what would be deleted without
actually deleting it.

The script reads ``PINECONE_API_KEY`` from the environment / ``.env``, resolved
through ``tests.live_suite.load_env`` so a run from a git worktree finds the
``.env`` in the main checkout, and so the path it used is printed rather than
guessed at. Override with ``PINECONE_SDK_ENV_FILE``.

It does NOT delete any backups — backup smoke tests always clean up their
own backups in-test, so backups are never leaked to this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pinecone import Pinecone
from tests.live_suite import load_env

PREFIX = "smoke"

_HERE = Path(__file__).resolve().parent


def cleanup(*, dry_run: bool = False) -> int:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set (in env or .env)")
        return 1

    pc = Pinecone(api_key=api_key)
    failures: list[str] = []

    # ---- Collections (must come BEFORE indexes; pod indexes that
    #      sourced a collection cannot be deleted while the collection is
    #      still pending) ----
    try:
        collection_names = sorted(n for n in pc.collections.list().names() if n.startswith(PREFIX))
    except Exception as exc:
        print(f"ERROR listing collections: {exc}")
        collection_names = []

    print(f"Found {len(collection_names)} smoke collections")
    for name in collection_names:
        if dry_run:
            print(f"  [dry-run] would delete collection: {name}")
            continue
        try:
            pc.collections.delete(name)
            print(f"  deleted collection: {name}")
        except Exception as exc:
            failures.append(f"collection/{name}: {exc}")
            print(f"  FAILED to delete collection {name}: {exc}")

    # Wait for collection deletes to propagate before touching indexes.
    if collection_names and not dry_run:
        import time

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            remaining = [n for n in pc.collections.list().names() if n.startswith(PREFIX)]
            if not remaining:
                break
            print(f"  waiting for {len(remaining)} collection(s) to fully delete...")
            time.sleep(10)

    # ---- Indexes ----
    try:
        index_names = sorted(i.name for i in pc.indexes.list() if i.name.startswith(PREFIX))
    except Exception as exc:
        print(f"ERROR listing indexes: {exc}")
        index_names = []

    print(f"Found {len(index_names)} smoke indexes")
    for name in index_names:
        if dry_run:
            print(f"  [dry-run] would delete index: {name}")
            continue
        try:
            pc.indexes.delete(name, timeout=-1)
            print(f"  deleted index: {name}")
        except Exception as exc:
            failures.append(f"index/{name}: {exc}")
            print(f"  FAILED to delete index {name}: {exc}")

    # ---- Assistants ----
    try:
        assistant_names = sorted(
            a.name for a in pc.assistants.list().to_list() if a.name.startswith(PREFIX)
        )
    except Exception as exc:
        print(f"ERROR listing assistants: {exc}")
        assistant_names = []

    print(f"Found {len(assistant_names)} smoke assistants")
    for name in assistant_names:
        if dry_run:
            print(f"  [dry-run] would delete assistant: {name}")
            continue
        try:
            pc.assistants.delete(name=name)
            print(f"  deleted assistant: {name}")
        except Exception as exc:
            failures.append(f"assistant/{name}: {exc}")
            print(f"  FAILED to delete assistant {name}: {exc}")

    pc.close()

    if failures:
        print()
        print(f"{len(failures)} cleanup failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 2
    return 0


def main() -> int:
    print(f".env source: {load_env(_HERE)}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resources that would be deleted, but do not delete them.",
    )
    args = parser.parse_args()
    return cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
