#!/usr/bin/env bash
# Pull a branch (and any new tags) back from the public pinecone-io/python-sdk
# repo into this one.
#
# This is the reverse of sync_to_public.sh, and deliberately the narrower of
# the two. Development happens here; public is a mirror. But a few things are
# still *born* in the public repo and have to come back:
#
#   - release commits + tags. The release workflows run in the public repo
#     (that's where the PyPI Trusted Publisher, the SLSA provenance identity,
#     the GitHub Release page and the docs deploy key live), and release-prod
#     creates a "release: X.Y.Z" version-bump commit, tags it, and
#     fast-forwards public main onto it.
#   - PRs merged directly against the public repo (outside contributions).
#
# Like its counterpart, this is fast-forward only: it moves origin/<branch> to
# public/<branch> and refuses to do anything clever. If the two have diverged
# (both sides have commits the other doesn't) it aborts -- reconcile that with
# a real merge by hand.
#
# Usage:
#   ./scripts/sync_from_public.sh [branch]            # dry run (default branch: main)
#   ./scripts/sync_from_public.sh [branch] --push     # actually push after confirmation
#   ./scripts/sync_from_public.sh [branch] --push -y  # push without the confirmation prompt
#
# ## Review the incoming commits
#
# Everything this script pulls in was authored somewhere with a public
# prompt-injection surface (issue and PR comments, forks). Content coming *in*
# from public has not been through this repo's review, and agents run against
# this repo. The dry run prints every incoming commit with its author and
# changed files precisely so that's a decision and not a default. Read the full
# diff before --push on anything you didn't author:
#
#   git diff origin/main...public/main
#
# Bot-authored release commits touching only the version-stamped files are the
# routine case. Anything else deserves a real read.

set -euo pipefail

PUBLIC_REMOTE_NAME="public"
PUBLIC_REMOTE_URL="git@github.com:pinecone-io/python-sdk.git"
BRANCH="main"

# Incoming tags are staged here so a dry run can describe them without
# creating real local tags. Cleaned up on exit.
SCRATCH_NS="refs/sync-from-public"

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

cleanup() {
    for ref in $(git for-each-ref --format='%(refname)' "$SCRATCH_NS/**" 2>/dev/null); do
        git update-ref -d "$ref" 2>/dev/null || true
    done
}
trap cleanup EXIT

if ! git remote get-url "$PUBLIC_REMOTE_NAME" >/dev/null 2>&1; then
    echo "==> Adding remote '$PUBLIC_REMOTE_NAME' -> $PUBLIC_REMOTE_URL"
    git remote add "$PUBLIC_REMOTE_NAME" "$PUBLIC_REMOTE_URL"
fi

if ! git ls-remote --exit-code --heads "$PUBLIC_REMOTE_NAME" "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: $PUBLIC_REMOTE_NAME has no branch '$BRANCH'." >&2
    exit 1
fi

echo "==> Fetching origin/$BRANCH and $PUBLIC_REMOTE_NAME/$BRANCH"
git fetch origin "$BRANCH"
git fetch "$PUBLIC_REMOTE_NAME" "$BRANCH"

origin_head=$(git rev-parse "origin/$BRANCH")
public_head=$(git rev-parse "$PUBLIC_REMOTE_NAME/$BRANCH")

# ---------------------------------------------------------------------------
# Which tags does public have that origin doesn't? Compared remote-to-remote
# on purpose: git's tag auto-following quietly copies public's tags into the
# local clone on any fetch, so a local-vs-public comparison under-reports and
# would silently skip tags origin is actually missing.
#
# A same-named tag pointing at a different object is not something to paper
# over: surface it and stop.
# ---------------------------------------------------------------------------
remote_tags() {
    git ls-remote --tags "$1" | awk '
        $2 ~ /\^\{\}$/ { next }
        { sub("refs/tags/", "", $2); print $2, $1 }
    ' | sort
}

origin_tags=$(remote_tags origin)

new_tags=""
tag_conflicts=""
while read -r tag sha; do
    [ -n "${tag:-}" ] || continue
    origin_sha=$(printf '%s\n' "$origin_tags" | awk -v t="$tag" '$1 == t { print $2 }')
    if [ -z "$origin_sha" ]; then
        new_tags="$new_tags $tag"
    elif [ "$origin_sha" != "$sha" ]; then
        tag_conflicts="${tag_conflicts}  $tag (origin: ${origin_sha:0:8}, public: ${sha:0:8})
"
    fi
done < <(remote_tags "$PUBLIC_REMOTE_NAME")

if [ -n "$tag_conflicts" ]; then
    echo "" >&2
    echo "ERROR: these tags exist in both repos but point at different objects:" >&2
    printf '%s' "$tag_conflicts" >&2
    echo "That shouldn't happen with a mirror. Investigate before syncing anything." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Branch relationship.
# ---------------------------------------------------------------------------
branch_action="none"
if [ "$origin_head" = "$public_head" ]; then
    echo "==> origin/$BRANCH and $PUBLIC_REMOTE_NAME/$BRANCH are identical (${origin_head:0:8})."
elif git merge-base --is-ancestor "$origin_head" "$public_head"; then
    branch_action="fast-forward"
elif git merge-base --is-ancestor "$public_head" "$origin_head"; then
    echo "==> origin/$BRANCH is ahead of $PUBLIC_REMOTE_NAME/$BRANCH -- nothing to pull back."
    echo "    (Use sync_to_public.sh to push this repo's work out.)"
else
    echo "" >&2
    echo "ERROR: origin/$BRANCH and $PUBLIC_REMOTE_NAME/$BRANCH have diverged." >&2
    printf '  %-30s %s\n' "origin/$BRANCH" "${origin_head:0:8}" >&2
    printf '  %-30s %s\n' "$PUBLIC_REMOTE_NAME/$BRANCH" "${public_head:0:8}" >&2
    printf '  %-30s %s\n' "merge base" "$(git merge-base "$origin_head" "$public_head" | cut -c1-8)" >&2
    echo "" >&2
    echo "A fast-forward can't express this. Merge by hand -- and merge, don't squash:" >&2
    echo "squashing rewrites the public commits' SHAs, after which sync_to_public.sh's" >&2
    echo "fast-forward check can never pass again." >&2
    echo "" >&2
    echo "  git checkout $BRANCH && git merge --no-ff $PUBLIC_REMOTE_NAME/$BRANCH" >&2
    exit 1
fi

if [ "$branch_action" = "none" ] && [ -z "$new_tags" ]; then
    echo "Nothing to sync."
    exit 0
fi

if [ "$branch_action" = "fast-forward" ]; then
    echo ""
    echo "Commits that would be pulled into origin/$BRANCH:"
    git log --format='  %h  %an  %s' "$public_head" --not "$origin_head"
    echo ""
    echo "Files they touch:"
    git diff --stat "$origin_head" "$public_head" | sed 's/^/  /'
    echo ""
    echo "Review the full diff before pushing anything you didn't author:"
    echo "  git diff $origin_head..$public_head"
fi

if [ -n "$new_tags" ]; then
    refspecs=()
    for tag in $new_tags; do
        refspecs+=("+refs/tags/$tag:$SCRATCH_NS/$tag")
    done
    echo ""
    echo "==> Staging new tags for review"
    git fetch "$PUBLIC_REMOTE_NAME" "${refspecs[@]}" >/dev/null 2>&1

    echo ""
    echo "Tags on $PUBLIC_REMOTE_NAME that this repo doesn't have:"
    for tag in $new_tags; do
        target=$(git rev-parse "$SCRATCH_NS/$tag^{commit}")
        if git merge-base --is-ancestor "$target" "$public_head"; then
            reachable="on $PUBLIC_REMOTE_NAME/$BRANCH"
        else
            reachable="off-branch"
        fi
        echo "  $tag -> ${target:0:8}  ($reachable)  $(git log -1 --format='%s' "$target")"
    done
    echo ""
    echo "  'off-branch' means the tag drags in a commit that isn't on $BRANCH."
    echo "  Normal for RC tags: the RC workflow tags a bump commit without"
    echo "  advancing main."
fi

echo ""
if [ "$DO_PUSH" = false ]; then
    echo "Dry run only. Re-run with --push to actually sync."
    exit 0
fi

if [ "$ASSUME_YES" = false ]; then
    read -r -p "Pull the above into origin ($BRANCH and/or tags)? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

if [ "$branch_action" = "fast-forward" ]; then
    echo "==> Fast-forwarding origin/$BRANCH to ${public_head:0:8}"
    git push origin "$public_head:refs/heads/$BRANCH"
fi

for tag in $new_tags; do
    echo "==> Pushing tag $tag to origin"
    git push origin "$SCRATCH_NS/$tag:refs/tags/$tag"
    git update-ref "refs/tags/$tag" "$(git rev-parse "$SCRATCH_NS/$tag")"
done

echo "Done. origin/$BRANCH is now at $(git rev-parse --short "$public_head")."
