#!/usr/bin/env bash
# Mirror this repo's main branch to the public pinecone-io/python-sdk repo.
#
# This repo (python-sdk-internal) is the working copy used for day-to-day
# development, including work with AI agents, away from the prompt-injection
# surface of a public-facing repo (issues/PR comments). This script pushes
# main here to main on the public repo as a straight fast-forward mirror.
#
# Usage:
#   ./scripts/sync_to_public.sh            # dry run: show what would be pushed
#   ./scripts/sync_to_public.sh --push     # actually push after confirmation
#   ./scripts/sync_to_public.sh --push -y  # push without the confirmation prompt
#
# The push is fast-forward only. If the public repo has commits this repo
# doesn't (e.g. someone merged a PR directly against it), the script aborts
# instead of overwriting them -- reconcile that manually first.

set -euo pipefail

PUBLIC_REMOTE_NAME="public"
PUBLIC_REMOTE_URL="git@github.com:pinecone-io/python-sdk.git"
BRANCH="main"

DO_PUSH=false
ASSUME_YES=false
for arg in "$@"; do
    case "$arg" in
        --push) DO_PUSH=true ;;
        -y|--yes) ASSUME_YES=true ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--push] [-y|--yes]" >&2
            exit 1
            ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean. Commit, stash, or discard changes first." >&2
    git status --short >&2
    exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "$BRANCH" ]; then
    echo "ERROR: expected to be on '$BRANCH', currently on '$current_branch'." >&2
    exit 1
fi

echo "==> Fetching origin/$BRANCH"
git fetch origin "$BRANCH"

if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "ERROR: local '$BRANCH' does not match 'origin/$BRANCH'." >&2
    echo "Push/pull with origin first so the internal repo is the source of truth." >&2
    exit 1
fi

if ! git remote get-url "$PUBLIC_REMOTE_NAME" >/dev/null 2>&1; then
    echo "==> Adding remote '$PUBLIC_REMOTE_NAME' -> $PUBLIC_REMOTE_URL"
    git remote add "$PUBLIC_REMOTE_NAME" "$PUBLIC_REMOTE_URL"
fi

echo "==> Fetching $PUBLIC_REMOTE_NAME/$BRANCH"
git fetch "$PUBLIC_REMOTE_NAME" "$BRANCH"

if ! git merge-base --is-ancestor "$PUBLIC_REMOTE_NAME/$BRANCH" "origin/$BRANCH"; then
    echo "ERROR: $PUBLIC_REMOTE_NAME/$BRANCH is not an ancestor of origin/$BRANCH." >&2
    echo "The public repo has commits this repo doesn't -- a fast-forward push" >&2
    echo "would be rejected (or would need --force, which this script won't do)." >&2
    echo "Reconcile the histories manually before re-running." >&2
    exit 1
fi

commits=$(git log --oneline "$PUBLIC_REMOTE_NAME/$BRANCH..origin/$BRANCH")
if [ -z "$commits" ]; then
    echo "Nothing to sync -- $PUBLIC_REMOTE_NAME/$BRANCH is already up to date."
    exit 0
fi

echo ""
echo "Commits that would be pushed to $PUBLIC_REMOTE_NAME/$BRANCH:"
echo "$commits"
echo ""

if [ "$DO_PUSH" = false ]; then
    echo "Dry run only. Re-run with --push to actually sync."
    exit 0
fi

if [ "$ASSUME_YES" = false ]; then
    read -r -p "Push the above commits to the PUBLIC repo's $BRANCH? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

echo "==> Pushing $BRANCH to $PUBLIC_REMOTE_NAME (fast-forward only)"
git push "$PUBLIC_REMOTE_NAME" "origin/$BRANCH:refs/heads/$BRANCH"

echo "Done. Public repo main is now at $(git rev-parse --short origin/$BRANCH)."
