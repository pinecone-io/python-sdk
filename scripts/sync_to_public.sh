#!/usr/bin/env bash
# Mirror a branch from this repo to the public pinecone-io/python-sdk repo.
#
# This repo (python-sdk-internal) is the working copy used for day-to-day
# development, including work with AI agents, away from the prompt-injection
# surface of a public-facing repo (issues/PR comments). This script pushes
# a branch's tip here to the same-named branch on the public repo as a
# straight fast-forward mirror -- it reads from origin/<branch>, not your
# local checkout, so it works for branches you don't have checked out.
#
# Usage:
#   ./scripts/sync_to_public.sh [branch]            # dry run (default branch: main)
#   ./scripts/sync_to_public.sh [branch] --push     # actually push after confirmation
#   ./scripts/sync_to_public.sh [branch] --push -y  # push without the confirmation prompt
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
        -*)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [branch] [--push] [-y|--yes]" >&2
            exit 1
            ;;
        *) BRANCH="$arg" ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

echo "==> Fetching origin/$BRANCH"
git fetch origin "$BRANCH"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    local_head=$(git rev-parse "refs/heads/$BRANCH")
    origin_head=$(git rev-parse "origin/$BRANCH")
    if [ "$local_head" != "$origin_head" ]; then
        echo "NOTE: your local '$BRANCH' differs from 'origin/$BRANCH'."
        echo "      Syncing origin's copy -- push/pull locally first if that's stale."
    fi
fi

if ! git remote get-url "$PUBLIC_REMOTE_NAME" >/dev/null 2>&1; then
    echo "==> Adding remote '$PUBLIC_REMOTE_NAME' -> $PUBLIC_REMOTE_URL"
    git remote add "$PUBLIC_REMOTE_NAME" "$PUBLIC_REMOTE_URL"
fi

baseline_excludes=()
if [ "$BRANCH" != "main" ] && git ls-remote --exit-code --heads "$PUBLIC_REMOTE_NAME" main >/dev/null 2>&1; then
    echo "==> Fetching $PUBLIC_REMOTE_NAME/main (baseline for shared history)"
    git fetch "$PUBLIC_REMOTE_NAME" main
    baseline_excludes+=("$PUBLIC_REMOTE_NAME/main")
fi

public_branch_exists=true
if git ls-remote --exit-code --heads "$PUBLIC_REMOTE_NAME" "$BRANCH" >/dev/null 2>&1; then
    echo "==> Fetching $PUBLIC_REMOTE_NAME/$BRANCH"
    git fetch "$PUBLIC_REMOTE_NAME" "$BRANCH"
else
    echo "==> $PUBLIC_REMOTE_NAME/$BRANCH does not exist yet -- it will be created."
    public_branch_exists=false
fi

if [ "$public_branch_exists" = true ]; then
    if ! git merge-base --is-ancestor "$PUBLIC_REMOTE_NAME/$BRANCH" "origin/$BRANCH"; then
        echo "ERROR: $PUBLIC_REMOTE_NAME/$BRANCH is not an ancestor of origin/$BRANCH." >&2
        echo "The public repo has commits this repo doesn't -- a fast-forward push" >&2
        echo "would be rejected (or would need --force, which this script won't do)." >&2
        echo "Reconcile the histories manually before re-running." >&2
        exit 1
    fi
    baseline_excludes+=("$PUBLIC_REMOTE_NAME/$BRANCH")
fi

if [ ${#baseline_excludes[@]} -gt 0 ]; then
    commits=$(git log --oneline "origin/$BRANCH" --not "${baseline_excludes[@]}")
else
    commits=$(git log --oneline "origin/$BRANCH")
fi

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

echo "Done. Public repo $BRANCH is now at $(git rev-parse --short "origin/$BRANCH")."
