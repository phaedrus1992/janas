#!/usr/bin/env bash
# Fixture test for .fork/sync-upstream.sh. Two bare repositories stand in for
# upstream and origin, and a stub gh stands in for GitHub. The stub writes
# every call to a log, so the test can assert the PR flow.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
script="$root/.fork/sync-upstream.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sync.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# A fixed identity and no signing: the fixtures must not need a key.
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=user.name GIT_CONFIG_VALUE_1=fixture
export GIT_CONFIG_KEY_2=user.email GIT_CONFIG_VALUE_2=fixture@example.invalid

fails=0
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}
ok() { echo "ok: $*"; }

upstream="$scratch/upstream.git"
origin="$scratch/origin.git"
git init --quiet --bare --initial-branch=main "$upstream"
git init --quiet --bare --initial-branch=main "$origin"

seed="$scratch/seed"
git clone --quiet "$upstream" "$seed"
cd "$seed"
printf '# Upstream\n\none\n' >README.md
mkdir -p src
printf 'int one(void) { return 1; }\n' >src/engine.c
git add -A
git commit --quiet -m "upstream: first"
git push --quiet origin main

# The sync script validates the upstream remote against the real URL, so the
# fixtures keep that URL and rewrite it to the local bare repository.
canonical="https://github.com/prabanta-dev/janas.git"

fork="$scratch/fork"
git clone --quiet "$origin" "$fork"
cd "$fork"
git config "url.$upstream.insteadOf" "$canonical"
git remote add upstream "$canonical"
git fetch --quiet upstream
git switch --quiet --create main upstream/main
{
  cat "$root/.fork/readme-header.md"
  cat README.md
} >README.new
mv README.new README.md
git add README.md
git commit --quiet -m "fork: header"
git push --quiet --set-upstream origin main

# The stub gh. It records every call. GH_STUB_OPEN_PRS is the count that
# `gh pr list` reports.
mkdir -p "$scratch/bin"
cat >"$scratch/bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$GH_STUB_LOG"
case "$1 ${2:-}" in
  "pr list")
    if [[ "${GH_STUB_FAIL_PR_LIST:-0}" == "1" ]]; then
      echo "gh-stub: pr list failed" >&2
      exit 1
    fi
    echo "${GH_STUB_OPEN_PRS:-0}"
    ;;
  "pr create") echo "https://example.invalid/pull/1" ;;
  "pr merge")
    if [[ "${GH_STUB_FAIL_PR_MERGE:-0}" == "1" ]]; then
      echo "gh-stub: pr merge failed" >&2
      exit 1
    fi
    echo "merged"
    ;;
  "issue list")
    if [[ "${GH_STUB_FAIL_ISSUE_LIST:-0}" == "1" ]]; then
      echo "gh-stub: issue list failed" >&2
      exit 1
    fi
    echo "0"
    ;;
  "issue create") echo "https://example.invalid/issues/1" ;;
  *)
    echo "gh-stub: unhandled: $*" >&2
    exit 1
    ;;
esac
STUB
chmod +x "$scratch/bin/gh"
export PATH="$scratch/bin:$PATH"
export GH_STUB_LOG="$scratch/gh.log"
: >"$GH_STUB_LOG"

today="$(date +%F)"
sync_branch="sync/$today"

run_sync() { cd "$1" && "$script"; }

# Scenario 1: the first sync. Upstream touches README.md, so the merge
# conflicts and the script must restore the header.
cd "$seed"
printf 'two\n' >>README.md
git commit --quiet -am "upstream: readme two"
git push --quiet origin main

if run_sync "$fork"; then ok "sync 1 exits 0"; else bad "sync 1 must exit 0"; fi

clean_sha="$(git --git-dir="$upstream" rev-parse main)"
fork_clean="$(git -C "$fork" rev-parse origin/clean)"
if [[ "$clean_sha" == "$fork_clean" ]]; then ok "clean mirrors upstream"; else bad "clean must mirror upstream"; fi

readme="$(git -C "$fork" show "origin/$sync_branch:README.md")"
if [[ "$readme" == '> **This is a fork.**'* ]]; then ok "the header is back"; else bad "the header must be back"; fi
header_count="$(printf '%s' "$readme" | grep -cF '> **This is a fork.**')"
if [[ "$header_count" -eq 1 ]]; then ok "the header appears once"; else bad "the header must appear once, not $header_count times"; fi
if [[ "$readme" == *"two"* ]]; then ok "upstream's README text is kept"; else bad "upstream's text must be kept"; fi

parents="$(git --git-dir="$origin" rev-list --parents --no-walk "$sync_branch" | wc -w)"
if [[ "$parents" -eq 3 ]]; then ok "the sync branch carries a merge commit"; else bad "the sync branch needs a merge commit"; fi

if grep -q -- "pr create" "$GH_STUB_LOG" 2>/dev/null; then ok "the PR was opened"; else bad "gh pr create must run"; fi
if grep -q -- "pr merge" "$GH_STUB_LOG"; then ok "the PR was merged"; else bad "gh pr merge must run"; fi

# Scenario 2: a second sync on the same day. The branch exists, no PR is
# open, so the script deletes it and cuts it again.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'int two(void) { return 2; }\n' >src/engine.c
git commit --quiet -am "upstream: engine two"
git push --quiet origin main
before="$(git --git-dir="$origin" rev-parse "$sync_branch")"
if run_sync "$fork"; then ok "sync 2 exits 0"; else bad "sync 2 must exit 0"; fi
after="$(git --git-dir="$origin" rev-parse "$sync_branch")"
if [[ "$before" != "$after" ]]; then ok "the sync branch was cut again"; else bad "the branch must be cut again"; fi
if git -C "$fork" show "origin/$sync_branch:src/engine.c" | grep -q "two"; then
  ok "upstream's engine change arrived"
else
  bad "the engine change must arrive"
fi

# Scenario 3: an open PR for the same head. The script must do nothing.
rm -f "$GH_STUB_LOG"
cd "$seed"
printf 'three\n' >>README.md
git commit --quiet -am "upstream: readme three"
git push --quiet origin main
cd "$fork"
before="$(git --git-dir="$origin" rev-parse "$sync_branch")"
export GH_STUB_OPEN_PRS=1
if run_sync "$fork"; then ok "sync 3 exits 0"; else bad "sync 3 must exit 0"; fi
unset GH_STUB_OPEN_PRS
after="$(git --git-dir="$origin" rev-parse "$sync_branch")"
if [[ "$before" == "$after" ]]; then ok "nothing was pushed"; else bad "nothing must be pushed"; fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then bad "no PR must be opened"; else ok "no PR was opened"; fi

# Scenario 4: a conflict on a file other than README.md. The script must
# abort, delete the branch, open one issue, and exit 1.
rm -f "$GH_STUB_LOG"
cd "$fork"
git switch --quiet main
printf 'fork edit\n' >src/engine.c
git commit --quiet -am "fork: engine edit"
git push --quiet origin main
cd "$seed"
printf 'upstream edit\n' >src/engine.c
git commit --quiet -am "upstream: engine edit"
git push --quiet origin main
cd "$fork"
if run_sync "$fork" 2>"$scratch/conflict.err"; then
  bad "sync 4 must fail"
else
  ok "sync 4 fails"
fi
if grep -q "src/engine.c" "$scratch/conflict.err"; then ok "the failure names the path"; else bad "the failure must name the path"; fi
if grep -q "issue create" "$GH_STUB_LOG"; then ok "an issue was opened"; else bad "an issue must be opened"; fi
if [[ "$(git -C "$fork" rev-parse --abbrev-ref HEAD)" == "main" ]]; then
  ok "the script returned to main"
else
  bad "the script must return to main"
fi
if git -C "$fork" rev-parse --verify --quiet "refs/heads/$sync_branch" >/dev/null; then
  bad "the local sync branch must be deleted"
else
  ok "the local sync branch is gone"
fi

# Scenario 5: a clone with no clean branch. The script must create it and
# push it, not fail. The engine conflict from scenario 4 goes away first.
cd "$fork"
git fetch --quiet upstream
git checkout --quiet upstream/main -- src/engine.c
git commit --quiet -am "fork: take upstream's engine"
git push --quiet origin main

rm -f "$GH_STUB_LOG"
git --git-dir="$origin" update-ref -d refs/heads/clean
fresh="$scratch/fresh"
git clone --quiet "$origin" "$fresh"
cd "$fresh"
git config "url.$upstream.insteadOf" "$canonical"
git remote add upstream "$canonical"
git fetch --quiet upstream
git switch --quiet main
if run_sync "$fresh"; then ok "sync 5 exits 0"; else bad "sync 5 must exit 0"; fi
if git --git-dir="$origin" rev-parse --verify --quiet refs/heads/clean >/dev/null; then
  ok "clean was created and pushed"
else
  bad "clean must be created and pushed"
fi

# Scenario 6: clean sits at upstream/main but main is behind, because an
# earlier run's pull request never merged. The script must try again, not
# report that main is in sync.
rm -f "$GH_STUB_LOG"
cd "$fresh"
if run_sync "$fresh"; then ok "sync 6 exits 0"; else bad "sync 6 must exit 0"; fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then ok "sync 6 opens a pull request"; else bad "sync 6 must open a pull request"; fi

# Scenario 7: main already contains everything clean has, as it would after a
# CI sync merged. The script must stop before it cuts a branch.
cd "$fork"
git switch --quiet main
git fetch --quiet origin
git fetch --quiet upstream
git merge --no-ff --no-commit clean >/dev/null 2>&1 || true
git checkout clean -- README.md
cat "$root/.fork/readme-header.md" README.md >README.md.new
mv README.md.new README.md
git add README.md
git commit --quiet -m "fixture: main contains clean"
git push --quiet origin main
rm -f "$GH_STUB_LOG"
if run_sync "$fork"; then ok "sync 7 exits 0"; else bad "sync 7 must exit 0"; fi
if grep -q "pr create" "$GH_STUB_LOG" 2>/dev/null; then bad "sync 7 must not open a pull request"; else ok "sync 7 opens nothing"; fi

# Scenario 8: gh cannot list pull requests. That is an error, not "a pull
# request is open".
cd "$seed"
printf 'four\n' >>README.md
git commit --quiet -am "upstream: readme four"
git push --quiet origin main
cd "$fresh"
export GH_STUB_FAIL_PR_LIST=1
if run_sync "$fresh" >/dev/null 2>&1; then
  bad "sync 8 must fail when gh pr list fails"
else
  ok "sync 8 fails"
fi
unset GH_STUB_FAIL_PR_LIST

# Scenario 9: a conflict while gh cannot list issues. The run must still fail
# and must not claim that an issue was filed.
cd "$fork"
git switch --quiet main
printf 'fork edit two\n' >src/engine.c
git commit --quiet -am "fork: engine edit two"
git push --quiet origin main
cd "$seed"
printf 'upstream edit two\n' >src/engine.c
git commit --quiet -am "upstream: engine edit two"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
export GH_STUB_FAIL_ISSUE_LIST=1
if run_sync "$fork" >"$scratch/conflict2.out" 2>"$scratch/conflict2.err"; then
  bad "sync 9 must fail"
else
  ok "sync 9 fails"
fi
unset GH_STUB_FAIL_ISSUE_LIST
if grep -q "a fork issue has the details" "$scratch/conflict2.err"; then
  bad "the failure must not claim an issue when none was filed"
else
  ok "the failure does not claim a filed issue"
fi
if grep -q "cannot list the issues" "$scratch/conflict2.out"; then
  ok "the failed issue search is reported"
else
  bad "a failed issue search must be reported"
fi
if grep -q "issue create" "$GH_STUB_LOG"; then
  ok "the issue is still created after a failed search"
else
  bad "the issue must be created after a failed search"
fi
if [[ "$(git -C "$fork" rev-parse --abbrev-ref HEAD)" == "main" ]]; then
  ok "sync 9 returned to main"
else
  bad "sync 9 must return to main"
fi

# Scenario 10: the merge of the pull request fails. The run must fail and name
# the recovery, because the branch and the pull request stay behind.
cd "$fork"
git fetch --quiet upstream
git checkout --quiet upstream/main -- src/engine.c
git commit --quiet -am "fork: take upstream's engine again"
git push --quiet origin main
cd "$seed"
printf 'five\n' >>README.md
git commit --quiet -am "upstream: readme five"
git push --quiet origin main
cd "$fork"
rm -f "$GH_STUB_LOG"
export GH_STUB_FAIL_PR_MERGE=1
if run_sync "$fork" 2>"$scratch/merge.err"; then
  bad "sync 10 must fail when the merge fails"
else
  ok "sync 10 fails"
fi
unset GH_STUB_FAIL_PR_MERGE
if grep -q "cannot merge" "$scratch/merge.err"; then
  ok "the failure names the recovery"
else
  bad "the failure must name the recovery"
fi

if [[ "$fails" -gt 0 ]]; then
  exit 1
fi
echo "test-sync-upstream: all checks passed"
