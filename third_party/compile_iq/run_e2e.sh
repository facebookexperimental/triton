#!/usr/bin/env bash
#
# compile_iq end-to-end smoke test: collect -> factory -> consume on the unmodified naive matmul
# (examples/user_kernel.py). Stores are hermetic temp dirs, so it leaves no state behind.
#
# The factory runs in --smoke-test mode: a tiny bounded search that stores an ACF REGARDLESS of
# speedup, purely so the consume hook actually HITs (a normal search on the ~optimal naive matmul
# almost always finds nothing better than baseline -> MISS, which never exercises apply). The stored
# ACF is NOT a validated speedup. The consume step is wrapped in `timeout` so a wedging ACF fails the
# test fast instead of hanging.
#
# Requirements:
#   - a python with torch + this fbtriton + the proprietary `compileiq` engine
#   - ptxas >= 13.3 (auto-discovered from the nvidia-cuda-nvcc wheel, or TRITON_PTXAS_BLACKWELL_PATH)
#   - COMPILE_IQ_SEARCH_SPACE_BIN -> the ptxas13.3 search-space .bin
#   (NOTE: no GPU-driver version requirement -- a 13.3-assembled ACF runs on any same-major driver.)
#
# Usage:
#   COMPILE_IQ_SEARCH_SPACE_BIN=/path/to/ptxas13.3.bin third_party/compile_iq/run_e2e.sh
#
# Optional env: PY (interpreter, default `python`), TRITON_PTXAS_BLACKWELL_PATH, CUDA_VISIBLE_DEVICES,
#   CONSUME_TIMEOUT (seconds, default 180), COMPILE_IQ_TASK_DIR, COMPILE_IQ_STORE.
#
set -euo pipefail

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$HERE/examples/user_kernel.py"
# Default to fresh temp dirs (never rm a user-provided store/task dir).
export COMPILE_IQ_TASK_DIR="${COMPILE_IQ_TASK_DIR:-$(mktemp -d)}"
export COMPILE_IQ_STORE="${COMPILE_IQ_STORE:-$(mktemp -d)}"
CONSUME_TIMEOUT="${CONSUME_TIMEOUT:-180}"

: "${COMPILE_IQ_SEARCH_SPACE_BIN:?set COMPILE_IQ_SEARCH_SPACE_BIN to the ptxas13.3 search-space .bin}"

# ---------------------------------------------------------------------------
# Step 1 -- Event collection (stdlib-only; no CompileIQ / ptxas needed here).
# A gated jit.py hook dumps a self-contained "compileIQ task" per kernel run; the user kernel is
# unmodified (zero surface).
#   => $COMPILE_IQ_TASK_DIR/<ptx_sha16>/{kernel.ptx,task.json,source.py}
# ---------------------------------------------------------------------------
echo "== [1/3] collect =="
FBTRITON_COMPILE_IQ_COLLECT=1 "$PY" "$KERNEL"
TASK="$(ls -d "$COMPILE_IQ_TASK_DIR"/*/ | head -1)"
echo "   task: $TASK"

# ---------------------------------------------------------------------------
# Step 2 -- ACF factory, --smoke-test (offline; needs CompileIQ engine + ptxas>=13.3 + ss-bin).
# Tiny bounded search; stores the best candidate UNCONDITIONALLY so consume has something to apply.
# (A real tuning run drops --smoke-test and stores only a faster-than-baseline ACF; reruns keep the
# best unless --force.)
#   => [factory] SMOKE-TEST wrote ACF -> $COMPILE_IQ_STORE/<arch>/<sha>.acf
# ---------------------------------------------------------------------------
echo "== [2/3] factory (--smoke-test) =="
"$PY" -m triton.compile_iq.factory "$TASK" --smoke-test

# ---------------------------------------------------------------------------
# Step 3 -- Consume ACF (gated make_cubin hook; needs ptxas>=13.3 to apply).
# Re-runs the same unmodified kernel; on a store HIT the tuned ACF is applied via --apply-controls.
# TRITON_ALWAYS_COMPILE=1 forces re-assembly so the hook re-runs ptxas. timeout-guarded so a wedging
# ACF fails fast rather than hanging.
#   => [compile_iq.consume] HIT <sha16> <arch>  then  "matmul ran ..."
# ---------------------------------------------------------------------------
echo "== [3/3] consume (timeout ${CONSUME_TIMEOUT}s) =="
rc=0
timeout -k 10 "$CONSUME_TIMEOUT" \
    env FBTRITON_COMPILE_IQ_APPLY=1 FBTRITON_COMPILE_IQ_DEBUG=1 TRITON_ALWAYS_COMPILE=1 "$PY" "$KERNEL" || rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "!! consume TIMED OUT after ${CONSUME_TIMEOUT}s -- the applied ACF likely wedged the GPU. FAIL."
    exit 1
elif [ "$rc" != 0 ]; then
    echo "!! consume exited $rc. FAIL."
    exit "$rc"
fi

echo "== done; store: =="
find "$COMPILE_IQ_STORE" -type f 2>/dev/null || echo "   (store empty)"
