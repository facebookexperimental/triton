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

TESTS=(
  python/test/unit/language/test_torchtlx_templates.py
  python/test/unit/language/test_torchtlx_fusions.py
)
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
             "torchTLX needs PyTorch >= 2.14")
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
  exec "$PY" -m pytest -p no:cacheprovider "$BENCH" -k torchtlx "$@" -v
fi
exec "$PY" -m pytest -p no:cacheprovider "${TESTS[@]}" "$@" -v
