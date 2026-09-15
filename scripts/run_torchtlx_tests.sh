#!/usr/bin/env bash
# CI entry point for the torchTLX tests and the torchTLX mm perf bench.
#
#   scripts/run_torchtlx_tests.sh [pytest args]    # template + fusion tests
#   scripts/run_torchtlx_tests.sh --perf [args]    # mm perf bench, via pytest
#
# Local runs do not need this: call pytest, or
# python/test/tlx_benchmark/bench_torchtlx_mm.py, directly. It exists so CI
# preflights the environment instead of reporting a broken env as N test
# failures. Set PYTHON to use an interpreter other than python3.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python3}"

COMMON_TESTS=(
  python/test/unit/language/test_torchtlx_templates.py
  python/test/unit/language/test_torchtlx_fusions.py
)
case "${TORCHTLX_ARCH:-all}" in
  sm100)
    ARCH_TESTS=(python/test/unit/tlx_ops/test_torchtlx_mm_sm100.py)
    PERF_FILTER=mm_torchtlx
    ;;
  gfx950)
    ARCH_TESTS=(
      python/test/unit/tlx_ops/test_torchtlx_addmm_gfx950.py
      python/test/unit/tlx_ops/test_torchtlx_bmm_gfx950.py
    )
    PERF_FILTER="addmm_torchtlx or bmm_torchtlx"
    ;;
  all)
    ARCH_TESTS=(
      python/test/unit/tlx_ops/test_torchtlx_mm_sm100.py
      python/test/unit/tlx_ops/test_torchtlx_addmm_gfx950.py
      python/test/unit/tlx_ops/test_torchtlx_bmm_gfx950.py
    )
    PERF_FILTER=torchtlx
    ;;
  *)
    echo "[torchtlx] unsupported TORCHTLX_ARCH=${TORCHTLX_ARCH}; expected sm100 or gfx950" >&2
    exit 2
    ;;
esac
TESTS=("${COMMON_TESTS[@]}" "${ARCH_TESTS[@]}")
BENCH=python/test/tlx_benchmark/test_ops_perf.py

PERF=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--perf" ]; then PERF=1; else ARGS+=("$arg"); fi
done

"$PY" - <<'PREFLIGHT'
import sys

try:
    import torch
    import triton
    from torch._inductor import config
except ImportError as e:
    sys.exit(f"[torchtlx] {e}")

print(f"[torchtlx] torch  -> {torch.__version__} ({torch.__file__})")
print(f"[torchtlx] triton -> {triton.__file__}")
try:
    mode = config.triton.tlx_mode
except AttributeError:
    sys.exit(f"[torchtlx] torch {torch.__version__} has no config.triton.tlx_mode; "
             "torchTLX needs a PyTorch build with the TorchTLX Inductor changes")
try:
    import triton.language.extra.tlx.inductor.registry  # noqa: F401
except ImportError as e:
    sys.exit(f"[torchtlx] the active Triton is not the fbtriton fork: {e}")
print(f"[torchtlx] tlx_mode -> {mode}")
PREFLIGHT

# Autotune subprocess workers otherwise crash with '0 active drivers'.
export TORCHINDUCTOR_COMPILE_THREADS=1

set -- "${ARGS[@]+"${ARGS[@]}"}"
if [ "$PERF" = 1 ]; then
  exec "$PY" -m pytest -p no:cacheprovider "$BENCH" -k "$PERF_FILTER" "$@" -v
fi
exec "$PY" -m pytest -p no:cacheprovider "${TESTS[@]}" "$@" -v
