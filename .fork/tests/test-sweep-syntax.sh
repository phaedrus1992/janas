#!/usr/bin/env bash
# Fixture test for .fork/sweep-syntax.sh. A stub compiler stands in for cc, so
# the test is fast and needs no toolchain. It writes only to a scratch
# directory, and the trap removes it.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
sweep="$root/.fork/sweep-syntax.sh"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/janas-sweep.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

fails=0
bad() {
  echo "FAIL: $*" >&2
  fails=$((fails + 1))
}

# The stub records one line for each call: the source, a tab, then all the
# arguments. For src/server/http.c it also records whether the generated
# header it includes is on the include path. STUB_EXIT sets its exit status.
stub="$scratch/cc"
cat >"$stub" <<'EOF'
#!/usr/bin/env bash
printf '%s\t%s\n' "${!#}" "$*" >>"$STUB_RECORD"
if [[ "${!#}" == src/server/http.c ]]; then
  for arg in "$@"; do
    if [[ "$arg" == -I* && -f "${arg#-I}/server_test_html.h" ]]; then
      grep -q janas_server_test_html "${arg#-I}/server_test_html.h" &&
        echo "generated header found" >>"$STUB_RECORD.gen"
    fi
  done
fi
exit "${STUB_EXIT:-0}"
EOF
chmod +x "$stub"
export STUB_RECORD="$scratch/record"

run_sweep() { # exit status for the stub, then the sweep arguments
  local status="$1"
  shift
  : >"$STUB_RECORD"
  STUB_EXIT="$status" CC="$stub" bash "$sweep" "$@"
}

# Usage errors.
if ! bash "$sweep" --help 2>"$scratch/help"; then
  bad "--help must exit 0"
fi
grep -q '^usage: sweep-syntax.sh' "$scratch/help" || bad "--help must print the usage"
set +e
bash "$sweep" 2>/dev/null
rc=$?
bash "$sweep" "$scratch/missing/list.txt" 2>"$scratch/err"
rc_dir=$?
set -e
[[ $rc -eq 2 ]] || bad "no argument must exit 2, got $rc"
[[ $rc_dir -eq 2 ]] || bad "a missing output directory must exit 2, got $rc_dir"
grep -q "$scratch/missing" "$scratch/err" || bad "the error must name the missing directory"

# Every file passes.
summary="$(run_sweep 0 "$scratch/pass.txt")"
total="$(wc -l <"$STUB_RECORD" | tr -d ' ')"
[[ $total -gt 0 ]] || bad "the sweep compiled no file"
[[ "$summary" == "0 of $total files fail" ]] ||
  bad "all pass: expected '0 of $total files fail', got '$summary'"
[[ ! -s "$scratch/pass.txt" ]] || bad "all pass: the failing list must be empty"
[[ -f "$scratch/pass.txt.log" ]] || bad "the sweep must write <failing-list>.log"

# Every file fails, and a second run replaces the list instead of adding to it.
summary="$(run_sweep 1 "$scratch/list.txt")"
[[ "$summary" == "$total of $total files fail" ]] ||
  bad "all fail: expected '$total of $total files fail', got '$summary'"
listed="$(wc -l <"$scratch/list.txt" | tr -d ' ')"
[[ $listed -eq $total ]] || bad "all fail: expected $total listed files, got $listed"
run_sweep 0 "$scratch/list.txt" >/dev/null
[[ ! -s "$scratch/list.txt" ]] || bad "a second run must replace the failing list"

# Outside a clone the sweep has no tree to sweep.
mkdir "$scratch/outside"
set +e
(cd "$scratch/outside" && bash "$sweep" list.txt 2>"$scratch/err")
rc=$?
set -e
[[ $rc -eq 2 ]] || bad "outside a clone the sweep must exit 2, got $rc"
grep -q 'is not a Janas clone' "$scratch/err" || bad "outside a clone the error must say so"

# Without git, as in a build container, the current directory is the clone.
nogit="$scratch/nogit"
mkdir "$nogit"
for tool in bash dirname basename uname tr find sort grep cat mktemp rm; do
  ln -s "$(command -v "$tool")" "$nogit/$tool"
done
(cd "$root" && PATH="$nogit" STUB_EXIT=0 CC="$stub" bash "$sweep" "$scratch/nogit.txt" \
  >"$scratch/nogit.out") || bad "without git the sweep must run in the current directory"
grep -q ' files fail$' "$scratch/nogit.out" || bad "without git the sweep must print its summary"

# A relative output path is relative to the caller, not to the clone root.
mkdir -p "$root/.tmp"
inside="$(mktemp -d "$root/.tmp/janas-sweep.XXXXXX")"
trap 'rm -rf "$scratch" "$inside"' EXIT
(cd "$root/src" && STUB_EXIT=1 CC="$stub" bash "$sweep" "../.tmp/${inside##*/}/list.txt" >/dev/null) ||
  bad "a relative output path must work from a subdirectory of the clone"
[[ -s "$inside/list.txt" ]] || bad "a relative output path must land in the caller's directory"

# The flag split: -w for third-party sources, -Werror for Janas's own.
run_sweep 0 "$scratch/pass.txt" >/dev/null
flags_of() { awk -F'\t' -v s="$1" '$1 == s { print $2; exit }' "$STUB_RECORD"; }
check_flags() { # source, a flag it must have, a flag it must not have
  local flags
  flags="$(flags_of "$1")"
  if [[ -z "$flags" ]]; then
    bad "$1 was not compiled"
    return
  fi
  [[ " $flags " == *" $2 "* ]] || bad "$1 must get $2"
  [[ " $flags " != *" $3 "* ]] || bad "$1 must not get $3"
}
check_flags src/third_party/libmicrohttpd/src/microhttpd/daemon.c -w -Werror
check_flags src/llm/api.c -Werror -w
check_flags src/common/mcp_tls.c -Werror -w
[[ " $(flags_of src/common/mcp_tls.c) " == *" -isystemsrc/third_party/mbedtls/include "* ]] ||
  bad "src/common/mcp_tls.c must see the mbedtls headers as system headers"
tests_c=("$root"/tests/*.c)
first_test="${tests_c[0]#"$root"/}"
check_flags "$first_test" -Werror -w

# The generated web-page headers come from the sweep, not from an old build.
: >"$STUB_RECORD.gen"
run_sweep 0 "$scratch/pass.txt" >/dev/null
grep -q 'generated header found' "$STUB_RECORD.gen" ||
  bad "src/server/http.c must see a server_test_html.h that declares janas_server_test_html"
leftover="$(find "${TMPDIR:-/tmp}" -maxdepth 1 -name 'janas-sweep-gen.*' 2>/dev/null)"
[[ -z "$leftover" ]] || bad "the sweep must remove its stub headers: $leftover"

if ((fails > 0)); then
  echo "test-sweep-syntax: $fails failures" >&2
  exit 1
fi
echo "test-sweep-syntax: ok"
