#!/usr/bin/env bash
#
# sync-upstream.sh - rebase the current branch onto upstream/master and
# force-push it to origin.
#
# Replaces the removed daily GitHub Actions sync workflow (.github/workflows/
# sync-upstream.yml): same operation, run locally so rebase conflicts can be
# resolved interactively instead of aborting unattended.
#
# usage:
#   ./scripts/sync-upstream.sh
#
# notes:
#   - the current branch is replayed on top of upstream/master (rebase, not
#     merge - keep the history linear) and pushed with --force-with-lease
#   - the working tree must be clean; commit or stash first
#   - on conflict the rebase is left in progress for you to resolve

set -euo pipefail

# must run from the repo root
if [[ ! -f "scripts/sync-upstream.sh" ]]; then
    echo "error: this script must be run from the root of the repository"
    exit 1
fi

branch=$(git branch --show-current)
if [[ -z "$branch" ]]; then
    echo "error: detached HEAD - checkout a branch first"
    exit 1
fi

# refuse to start with uncommitted tracked changes or a rebase already in progress
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "error: working tree is not clean - commit or stash your changes first"
    exit 1
fi
if [[ -d "$(git rev-parse --git-path rebase-merge)" || -d "$(git rev-parse --git-path rebase-apply)" ]]; then
    echo "error: a rebase is already in progress - finish or abort it first"
    exit 1
fi

# make sure the upstream remote exists
if ! git remote get-url upstream >/dev/null 2>&1; then
    echo "adding remote 'upstream' -> https://github.com/ggml-org/llama.cpp"
    git remote add upstream https://github.com/ggml-org/llama.cpp
fi

echo "fetching upstream/master..."
git fetch --no-tags upstream master:refs/remotes/upstream/master

echo "rebasing $branch onto upstream/master..."
if ! git rebase upstream/master; then
    echo
    echo "error: rebase conflicts - resolve them, then:"
    echo "  git rebase --continue"
    echo "  git push --force-with-lease origin $branch"
    exit 1
fi

echo "pushing $branch to origin..."
if ! git push --force-with-lease origin "$branch"; then
    echo
    echo "error: push refused - origin/$branch moved since your last fetch?"
    echo "reconcile manually, e.g.:"
    echo "  git fetch origin && git rebase origin/$branch && git push --force-with-lease origin $branch"
    exit 1
fi

echo "done: $branch rebased onto upstream/master and pushed"
