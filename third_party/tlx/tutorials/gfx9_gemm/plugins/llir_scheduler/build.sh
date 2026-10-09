#!/usr/bin/env bash
# Build the LLIR scheduler pass plugin against the LLVM that Triton is built with.
#
# The plugin does not link LLVM: it resolves LLVM symbols from libtriton at load
# time, so it is ABI-locked to that exact LLVM. Rebuild it whenever the pin in
# cmake/llvm-info.json moves.
#
#   LLVM_SYSPATH=/path/to/llvm ./build.sh     # explicit LLVM install
#   ./build.sh                                # the LLVM Triton downloaded
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../../../.." && pwd)"

llvm="${LLVM_SYSPATH:-}"
if [[ -z "$llvm" ]]; then
  hash="$(sed -n 's/.*"llvm_hash": *"\([0-9a-f]\{8\}\).*/\1/p' "$repo/cmake/llvm-info.json")"
  llvm="$(ls -d "${TRITON_HOME:-$HOME}"/.triton/llvm/llvm-"$hash"-* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$llvm" || ! -x "$llvm/bin/llvm-config" ]]; then
  echo "error: no LLVM with bin/llvm-config found; set LLVM_SYSPATH" >&2
  exit 1
fi

echo "building against $llvm"
# shellcheck disable=SC2046
"${CXX:-g++}" -shared -fPIC -fvisibility=default $("$llvm/bin/llvm-config" --cxxflags) \
  -o "$here/libLlirSched.so" "$here/LlirSchedPlugin.cpp"
echo "wrote $here/libLlirSched.so"
