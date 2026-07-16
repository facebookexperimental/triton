#!/usr/bin/env bash
# One `git bisect run` step for a nightly regression: build Triton at the
# currently-checked-out commit and run the failing test's repro command.
#
# Exit-code contract expected by `git bisect run`:
#   0   -> commit is GOOD (repro passed)
#   1   -> commit is BAD  (repro failed)
#   125 -> SKIP this commit (could not build it; do not blame it)
#
# Required env:
#   BISECT_REPRO   the reproduce command to run (e.g. "python -m pytest a/b.py::t -v")
# Optional env:
#   BISECT_MODE    "gpu" (default) builds via the tritonbench install helper;
#                  "lit" builds triton-opt via cmake/ninja for LIT repros.
#   SETUP_SCRIPT   conda/env activation script for GPU runners
#                  (default: /workspace/setup_instance.sh)
set -uo pipefail

MODE="${BISECT_MODE:-gpu}"
REPRO="${BISECT_REPRO:?BISECT_REPRO (repro command) is required}"
SETUP_SCRIPT="${SETUP_SCRIPT:-/workspace/setup_instance.sh}"

echo "::group::bisect step $(git rev-parse --short HEAD) — build (mode=$MODE)"
# Each bisect checkout can move submodule pointers; resync before building.
git submodule update --init --recursive || true

build_ok=0
if [[ "$MODE" == "lit" ]]; then
  # Assumes an llvm-install/ tree is already present on the runner (as in
  # compiler.yml). Rebuilds only the LIT tooling target.
  python3 -m pip install -q cmake ninja lit || true
  LLVM="${GITHUB_WORKSPACE:-$PWD}/llvm-install"
  if cmake -G Ninja -B build \
        -DTRITON_CACHE_PATH="$HOME/.triton" \
        -DLLVM_INCLUDE_DIRS="$LLVM/include" \
        -DLLVM_LIBRARY_DIR="$LLVM/lib" \
        -DLLVM_SYSPATH="$LLVM" \
        -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
        -DTRITON_BUILD_UT=OFF \
        -DLLVM_EXTERNAL_LIT="$(which lit)" . \
     && ninja -C build check-triton-lit-tests-build 2>/dev/null; then
    build_ok=1
  fi
else
  # GPU path: mirror the nightly "Compile Triton" step.
  # shellcheck disable=SC1090
  { . "$SETUP_SCRIPT" \
      && . /workspace/tritonbench/.ci/triton/triton_install_utils.sh \
      && install_triton "$PWD"; } && build_ok=1
fi
echo "::endgroup::"

if [[ "$build_ok" -ne 1 ]]; then
  echo "build failed at $(git rev-parse --short HEAD) -> SKIP (125)"
  exit 125
fi

echo "::group::bisect step $(git rev-parse --short HEAD) — repro"
if [[ "$MODE" == "gpu" ]]; then
  # shellcheck disable=SC1090
  . "$SETUP_SCRIPT" || true
fi
set -x
eval "$REPRO"
rc=$?
set +x
echo "::endgroup::"

if [[ "$rc" -eq 0 ]]; then
  echo "repro passed -> GOOD (0)"
  exit 0
fi
echo "repro failed (rc=$rc) -> BAD (1)"
exit 1
