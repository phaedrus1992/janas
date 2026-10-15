#!/usr/bin/env bash
# The portability sweep. Compiles every source build.sh compiles with
# -fsyntax-only, with build.sh's flags, and lists the files that fail. It
# measures how far a port/* branch moves the failure count.
# Design: docs/superpowers/specs/2026-10-13-arm64-metal-port/e1-portability.md §2
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sweep-syntax.sh <failing-list>

Sweeps the Janas clone that holds the current directory, or the current
directory itself when git is not available (as in a build container). Writes
the failing files to <failing-list>, one for each line, and the compiler
output to <failing-list>.log. Prints "<failing> of <total> files fail" on
stdout. Exit 0 when the sweep ran, whatever the count; 2 for a usage error.

CC selects the compiler (default cc). A port/* branch carries no .fork/, so
copy the script from main first:

  git show main:.fork/sweep-syntax.sh > .tmp/sweep-syntax.sh
  bash .tmp/sweep-syntax.sh .tmp/sweep.txt
EOF
}

case "${1:-}" in
-h | --help)
  usage
  exit 0
  ;;
"")
  usage
  exit 2
  ;;
esac

out_dir="$(dirname "$1")"
if [[ ! -d "$out_dir" ]]; then
  echo "sweep-syntax: the directory $out_dir does not exist." >&2
  echo "sweep-syntax: create it (mkdir -p $out_dir), then run again." >&2
  exit 2
fi
out="$(cd "$out_dir" && pwd)/$(basename "$1")"
log="$out.log"

if ! top="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  top="$PWD"
fi
if [[ ! -f "$top/build.sh" ]]; then
  echo "sweep-syntax: $top is not a Janas clone: it has no build.sh." >&2
  echo "sweep-syntax: cd into the Janas clone to sweep, then run again." >&2
  exit 2
fi
cd "$top"
CC="${CC:-cc}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
VERSION="$(git describe --always --dirty 2>/dev/null || echo unknown)"

# build.sh writes each src/<tool>/<name>.html as a C array in a generated
# header. A syntax check needs only the declaration, so the sweep writes
# stubs of its own: a header left in lib/ by an old build can be stale or
# absent, and an absent one adds a failure that is not a portability one.
gen="$(mktemp -d "${TMPDIR:-/tmp}/janas-sweep-gen.XXXXXX")"
trap 'rm -rf "$gen"' EXIT
for html in src/*/*.html; do
  if [[ -f "$html" ]]; then
    tool="$(basename "$(dirname "$html")")"
    page="$(basename "${html%.html}")"
    echo "static const char janas_${tool}_${page}_html[] = {0};" \
      >"$gen/${tool}_${page}_html.h"
  fi
done

# build.sh gives third-party sources -w and its own sources
# -Wall -Wextra -Werror. A sweep that puts -Werror on third-party sources
# reports about 160 failures that the real build does not have.
BASE=(-std=gnu11 -D_GNU_SOURCE= -pthread -Isrc -Iinclude -fPIC -fvisibility=hidden)
BASE+=(-I"$gen" -DJANAS_VERSION="\"$VERSION\"")
JANAS=("${BASE[@]}" -Wall -Wextra -Werror
  -Isrc/third_party/libmicrohttpd/src/include -Isrc/third_party/yyjson)
TP=("${BASE[@]}" -w)

d=src/third_party/mbedtls
c=$d/tf-psa-crypto
v=$c/drivers
MBEDTLS_INC=(
  "-I$d/library" "-I$d/include" "-I$c/include"
  "-I$v/builtin/include" "-I$v/everest/include"
  "-I$v/everest/include/tf-psa-crypto/private/everest"
  "-I$v/everest/include/tf-psa-crypto/private/everest/kremlib"
  "-I$v/p256-m/p256-m/include" "-I$v/p256-m/p256-m/include/p256-m"
  "-I$v/p256-m/p256-m_driver_interface" "-I$v/pqcp/include"
  "-I$c/core" "-I$v/builtin/src" "-I$c/dispatch" "-I$c/extras"
  "-I$c/platform" "-I$c/utilities" "-I$v/pqcp/src"
  "-I$v/pqcp/mldsa-native/mldsa")

# The same list as build.sh's tp_sources mbedtls. build.sh does not compile
# the other files under mbedtls/, and some of them do not compile at all.
mbedtls_sources() {
  local f
  for f in "$d"/library/*.c "$c"/core/*.c "$c"/extras/*.c "$c"/platform/*.c \
    "$c"/utilities/*.c "$v"/builtin/src/*.c "$v"/pqcp/src/*.c \
    "$v"/everest/library/x25519.c \
    "$v"/everest/library/Hacl_Curve25519_joined.c \
    "$v"/p256-m/p256-m_driver_entrypoints.c \
    "$v"/p256-m/p256-m/p256-m.c; do
    if [[ -f "$f" ]]; then
      echo "$f"
    fi
  done | sort
}

total=0
fail=0
: >"$out"
: >"$log"
sweep_one() { # flags... source
  local src="${!#}"
  total=$((total + 1))
  if ! "$CC" -fsyntax-only "$@" >>"$log" 2>&1; then
    fail=$((fail + 1))
    echo "$src" >>"$out"
  fi
}

lmhd_flags=(-DHAVE_CONFIG_H -Isrc/third_party/libmicrohttpd
  -Isrc/third_party/libmicrohttpd/src/include
  -Isrc/third_party/libmicrohttpd/src/microhttpd)
if [[ "$OS" == darwin ]]; then
  lmhd_flags=(-Isrc/third_party/libmicrohttpd/darwin "${lmhd_flags[@]}")
fi
while IFS= read -r src; do
  sweep_one "${TP[@]}" "${lmhd_flags[@]}" "$src"
done < <(find src/third_party/libmicrohttpd -name '*.c' | sort)
while IFS= read -r src; do
  sweep_one "${TP[@]}" "${MBEDTLS_INC[@]}" "$src"
done < <(mbedtls_sources)
if [[ -d src/third_party/yyjson ]]; then
  while IFS= read -r src; do
    sweep_one "${TP[@]}" "$src"
  done < <(find src/third_party/yyjson -name '*.c' | sort)
fi

# A directory with a main.c is a program. Its other files go to the program
# pass below, as build.sh splits them.
tool_dirs="$(for m in src/*/main.c src/services/*/main.c; do dirname "$m"; done)"
while IFS= read -r src; do
  if grep -qx "$(dirname "$src")" <<<"$tool_dirs"; then
    continue
  fi
  extra=()
  if [[ "$src" == src/common/mcp_tls.c ]]; then
    extra=("${MBEDTLS_INC[@]/#-I/-isystem}")
  fi
  sweep_one "${JANAS[@]}" ${extra[@]+"${extra[@]}"} "$src"
done < <(find src -name '*.c' ! -name main.c ! -name win_start.c \
  ! -path 'src/third_party/*' | sort)
for src in tests/*.c tools/*.c src/*/main.c src/services/*/main.c; do
  sweep_one "${JANAS[@]}" "$src"
  if [[ "$(basename "$src")" == main.c ]]; then
    while IFS= read -r part; do
      sweep_one "${JANAS[@]}" "$part"
    done < <(find "$(dirname "$src")" -maxdepth 1 -name '*.c' ! -name main.c | sort)
  fi
done

echo "$fail of $total files fail"
