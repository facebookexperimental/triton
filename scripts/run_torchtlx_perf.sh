#!/usr/bin/env bash
# One-shot perf runner for torchTLX Inductor templates (git repo; no tritonbench).
#
# This is the unified command external contributors use so everyone measures
# torchTLX perf the same way. It sets up just enough environment and forwards all
# args to the harness (third_party/.../inductor/benchmarks/torchtlx_perf.py).
#
# Prereq: a built env (venv + fork Triton + torch). If you have not built yet,
# run scripts/run_torchtlx_tests.sh once -- it bootstraps everything; this script
# then reuses that .venv.
#
# Usage:
#   # single shape
#   scripts/run_torchtlx_perf.sh --op mm --only aten,torch_tlx_mm,tlx_ws \
#       --precision bf16 --m 4096 --n 4096 --k 4096 --metrics accuracy,tflops
#
#   # batch (named shape set: default | square | ads)
#   scripts/run_torchtlx_perf.sh --op mm --only aten,torch_tlx_mm \
#       --precision bf16 --shape-set default --metrics tflops,speedup --baseline aten
#
#   # ad-hoc shapes
#   scripts/run_torchtlx_perf.sh --shapes 4096x4096x4096,8192x8192x8192
#
#   scripts/run_torchtlx_perf.sh --list      # list torchTLX templates
#
# Pin a GPU with CUDA_VISIBLE_DEVICES=<id> as usual.
#
# Env knobs:
#   TORCHTLX_VENV=<dir>     venv location (default: <repo>/.venv)
#   TORCHTLX_STDCXX_DIR=<d> dir with a modern libstdc++.so.6 (auto-detected)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
VENV="${TORCHTLX_VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
MODULE="triton.language.extra.tlx.inductor.benchmarks.torchtlx_perf"

log() { echo "[run_torchtlx_perf] $*" >&2; }

if [ ! -x "$PY" ]; then
  log "no venv at $VENV -- run scripts/run_torchtlx_tests.sh first to bootstrap the env"
  exit 1
fi

# --- locate a modern libstdc++ (the prebuilt LLVM needs GLIBCXX_3.4.30) -------
detect_stdcxx_dir() {
  if [ -n "${TORCHTLX_STDCXX_DIR:-}" ]; then echo "$TORCHTLX_STDCXX_DIR"; return; fi
  local base_home cand
  base_home="$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg" 2>/dev/null || true)"
  for cand in "${base_home%/bin}/lib" "${CONDA_PREFIX:-}/lib" /opt/conda/lib; do
    [ -n "$cand" ] || continue
    if [ -e "$cand/libstdc++.so.6" ] && grep -qa GLIBCXX_3.4.30 "$cand/libstdc++.so.6" 2>/dev/null; then
      echo "$cand"; return
    fi
  done
  echo ""
}
STDCXX="$(detect_stdcxx_dir)"
[ -n "$STDCXX" ] && export LD_LIBRARY_PATH="$STDCXX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ensure the torch-side stopgap (tlx_mode knob + fork-loading tlx.py) is present
"$PY" scripts/_torchtlx_torch_shim.py >/dev/null

# TORCHINDUCTOR_COMPILE_THREADS=1: autotune subprocess workers otherwise crash
# with '0 active drivers' (no GPU backend in the worker).
export TORCHINDUCTOR_COMPILE_THREADS=1

"$PY" -c "import triton; print('[run_torchtlx_perf] triton ->', triton.__file__)" >&2

exec "$PY" -m "$MODULE" "$@"
