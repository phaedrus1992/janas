#!/usr/bin/env bash
# The fork sync. Fast-forwards clean from upstream/main, then merges clean
# into main through an auto-merged pull request. One implementation for a
# local session and for .github/workflows/fork-sync.yml.
# Design: docs/superpowers/specs/2026-10-13-arm64-metal-port/e0-fork-workflow.md §4
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sync-upstream.sh

Runs from main. Fast-forwards clean from upstream/main, merges clean into a
fresh sync/YYYY-MM-DD-HHMMSS branch, and merges that branch into main
through a pull request. Exits 0 when main is already in sync.
EOF
}

case "${1:-}" in
-h | --help)
  usage
  exit 0
  ;;
esac

upstream_url="https://github.com/prabanta-dev/janas.git"

die() {
  echo "sync-upstream: $*" >&2
  exit 1
}
note() { echo "sync-upstream: $*"; }
warn() { echo "sync-upstream: $*" >&2; }

cd "$(git rev-parse --show-toplevel)"

status_out="$(git status --porcelain)" || die "cannot read the working tree status"
[[ -z "$status_out" ]] || die "the working tree is dirty; commit or stash first"

head_ref="$(git rev-parse --abbrev-ref HEAD)"
[[ "$head_ref" == "main" ]] || die "run this from main, not from $head_ref"

# The configured URL, not the expanded one: git remote get-url applies
# url.<base>.insteadOf, and a rewrite is the operator's own choice.
if upstream_found="$(git config --get remote.upstream.url)"; then
  case "$upstream_found" in
  "$upstream_url" | git@github.com:prabanta-dev/janas.git) ;;
  *) die "the upstream remote is $upstream_found, not $upstream_url" ;;
  esac
else
  git remote add upstream "$upstream_url"
fi

# A CI runner has no identity. The merge commit needs one.
git config user.name >/dev/null 2>&1 || git config user.name janas-sync
git config user.email >/dev/null 2>&1 || git config user.email janas-sync@users.noreply.github.com

git fetch --quiet origin
git fetch --quiet upstream
git rev-parse --verify --quiet origin/main >/dev/null || die "origin/main is missing"
git rev-parse --verify --quiet upstream/main >/dev/null || die "upstream/main is missing"
[[ "$(git rev-parse main)" == "$(git rev-parse origin/main)" ]] ||
  die "local main differs from origin/main; run git pull --ff-only first"

# clean: the mirror of upstream/main. Create it when this clone has none.
if git rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  git switch --quiet clean
elif git rev-parse --verify --quiet refs/remotes/origin/clean >/dev/null; then
  git switch --quiet --create clean --track origin/clean
else
  git switch --quiet --create clean upstream/main
  note "created clean from upstream/main"
fi

git merge --quiet --ff-only upstream/main
git push --quiet origin clean
git switch --quiet main

# main may already hold every upstream commit: a CI run, or an earlier run of
# this script, may have merged it. Then there is nothing to sync. Testing
# containment, not "clean moved", keeps this true when clean moved in an
# earlier run whose pull request never merged.
if git merge-base --is-ancestor clean origin/main; then
  note "main already contains upstream/main; nothing to do"
  exit 0
fi

# A name no earlier run used, and one clock read so the branch name and the
# date in the messages always agree. Re-creating an existing branch (delete,
# push, then open the pull request a second later) makes GitHub refuse
# createPullRequest until the delete has settled (#31).
stamp="$(date +%F-%H%M%S)"
today="${stamp:0:10}"
sync_branch="sync/$stamp"

# An open sync pull request means an earlier run stopped and needs a hand, or
# another sync is in flight right now. Either way this run stops and names
# the pull request: a silent exit 0 would let main fall behind a green
# workflow. A pull request from another fork does not count: it has
# isCrossRepository true, and merging it would sync someone else's branch.
if ! pr_json="$(gh pr list --state open --limit 1000 --json url,headRefName,isCrossRepository)"; then
  die "cannot list pull requests; is gh installed and authenticated?"
fi
open_pr="$(jq -r '[.[] | select(.isCrossRepository == false) | select(.headRefName | startswith("sync/")) | .url] | first // "none"' <<<"$pr_json")" ||
  die "cannot read the pull request list"
if [[ "$open_pr" != "none" ]]; then
  die "a sync pull request is open: $open_pr. Merge it, or close it, then run again"
fi

git switch --quiet --create "$sync_branch" origin/main

# A plain merge would fast-forward. --no-ff keeps the sync branch's own merge
# commit, under the pull request's merge commit (design §4).
if ! git merge --no-ff --no-commit clean; then
  conflicted="$(git diff --name-only --diff-filter=U)"
  if [[ "$conflicted" == "README.md" ]]; then
    note "README.md conflicts; keeping upstream's copy and the fork header"
    git checkout clean -- README.md
    cat .fork/readme-header.md README.md >README.md.new
    mv README.md.new README.md
    git add README.md
  else
    if ! git merge --abort; then
      die "the merge cannot be aborted; resolve it by hand in $sync_branch"
    fi
    git switch --quiet main
    git branch --delete --force "$sync_branch"
    [[ -n "$conflicted" ]] || die "the merge failed; see the errors above"
    title="sync blocked: $conflicted conflicts with upstream"
    # A failed search is not proof that no such issue exists. Say so, rather
    # than reading the failure as "none open".
    if ! open_issues="$(gh issue list --state open --search "$title in:title" --json number --jq 'length')"; then
      note "cannot list the issues; skipping the duplicate check"
      open_issues=0
    fi
    if [[ "$open_issues" == "0" ]]; then
      gh issue create --title "$title" \
        --body "The sync could not merge upstream/main into main. Conflicting paths: $conflicted. Resolve them by hand, then run .fork/sync-upstream.sh again." ||
        note "could not open the issue; file it by hand"
    fi
    die "merge conflict in $conflicted; resolve it, then run this script again"
  fi
fi

git commit --quiet --message "sync: merge upstream/main into main ($today)"

git push --quiet --set-upstream origin "$sync_branch"
pr_url="$(gh pr create --base main --head "$sync_branch" \
  --title "sync: merge upstream/main into main ($today)" \
  --body "Automated sync of upstream/main into main, by .fork/sync-upstream.sh.")" ||
  die "cannot open the pull request for $sync_branch"
# Merge the pull request this run opened, by its URL. The branch name alone
# could name a pull request from another fork.
gh pr merge "$pr_url" --merge --delete-branch ||
  die "cannot merge $pr_url; merge it by hand and delete $sync_branch"

# Branches left by earlier runs that never merged. A concurrent run is the
# only way another sync branch can be live, and it shows up as an open sync
# pull request; re-check for one just before the sweep, because a local run
# can start between this run's guard and here. Delete after the merge, never
# before the create (#31).
stale_skipped=0
if ! pr_json="$(gh pr list --state open --limit 1000 --json headRefName,isCrossRepository)"; then
  warn "cannot list the open pull requests; skipping the stale cleanup"
  stale_skipped=1
elif [[ "$(jq -r '[.[] | select(.isCrossRepository == false) | select(.headRefName | startswith("sync/"))] | length' <<<"$pr_json")" != "0" ]]; then
  note "another sync pull request is open; skipping the stale cleanup"
  stale_skipped=1
elif ! stale="$(git ls-remote --heads origin 'refs/heads/sync/*')"; then
  warn "cannot list the sync branches; skipping the stale cleanup"
  stale_skipped=1
else
  while IFS=$'\t' read -r _sha ref; do
    [[ -n "$ref" ]] || continue
    name="${ref#refs/heads/}"
    if [[ "$name" == "$sync_branch" ]]; then continue; fi
    git push --quiet origin --delete "$name" || warn "cannot delete the stale branch $name"
  done <<<"$stale"
fi

git switch --quiet main
if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  git pull --quiet --ff-only
fi
if [[ "$stale_skipped" == "0" ]]; then
  note "sync complete"
else
  note "sync complete; the stale cleanup was skipped"
fi
