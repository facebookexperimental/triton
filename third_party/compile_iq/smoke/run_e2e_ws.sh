#!/usr/bin/env bash
#
# compile_iq WS PTX-direct end-to-end: collect -> factory(EVO) -> store -> consume on the
# warp-specialized Blackwell GEMM (examples/blackwell_gemm_ws_sample.py).
#
# PURE PTX-DIRECT: the factory never recompiles via Triton. Each candidate ACF is scored by
# ptxas-assembling the FIXED collected WS PTX (--apply-controls) and launching the cubin via the CUDA
# driver (ptx_bench_one.py / ptx_launch.py) -- including the host-side TMA ABI: 4 TensorDescriptors
# rebuilt with fill_tma_descriptor_tiled and passed as by-value 128B tensormaps. Correctness per
# candidate is self-consistency vs the no-ACF launch; consume then validates vs torch in the sample.
#
# Defaults: WS_GEMM_SIZE=2048 (the 8192^3 config overflows B200 smem; 2048^3 fits), ptxas 13.0 + the
# cuda-13.0 p0 tier (safest). Use TIER=p2 for the perf-win search. Version matching is required: the
# ptxas must match the tier AND the installed driver (PTX ISA).
#
# Usage: third_party/compile_iq/smoke/run_e2e_ws.sh
# Optional env: WS_GEMM_SIZE, TIER (p0/p1/p2, default p0), PTXAS_KNOBS, SS_CONFIG, PTXAS,
#   PTXAS_MIN_VERSION, EVO_PY, PY, PER_CAND_TIMEOUT, CONSUME_TIMEOUT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$HERE/../examples/blackwell_gemm_ws_sample.py"

PY="${PY:-python}"
EVO_PY="${EVO_PY:-$(conda info --base 2>/dev/null)/envs/evo/bin/python}"
PTXAS="${PTXAS:-/usr/local/cuda-13.0/bin/ptxas}"
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"
TIER="${TIER:-p0}"
SS_CONFIG="${SS_CONFIG:-$PTXAS_KNOBS/cuda-13.0-ptxas-${TIER}.config}"
PTXAS_MIN_VERSION="${PTXAS_MIN_VERSION:-13.0}"
PER_CAND_TIMEOUT="${PER_CAND_TIMEOUT:-30}"
CONSUME_TIMEOUT="${CONSUME_TIMEOUT:-360}"

export WS_GEMM_SIZE="${WS_GEMM_SIZE:-2048}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_PTXAS_BLACKWELL_PATH="$PTXAS"
export BASE_PY="$PY"
export COMPILE_IQ_TASK_DIR="${COMPILE_IQ_TASK_DIR:-$(mktemp -d)}"
export COMPILE_IQ_STORE="${COMPILE_IQ_STORE:-$(mktemp -d)}"

[ -x "$EVO_PY" ]    || { echo "FAIL: evo python not found at '$EVO_PY' -- set EVO_PY"; exit 1; }
[ -f "$SS_CONFIG" ] || { echo "FAIL: search space not found at '$SS_CONFIG' -- set SS_CONFIG/PTXAS_KNOBS"; exit 1; }
[ -x "$PTXAS" ]     || { echo "FAIL: ptxas not found at '$PTXAS' -- set PTXAS"; exit 1; }

echo "ptxas=$PTXAS ($("$PTXAS" --version | grep -oE 'V[0-9.]+' | tail -1)) ; space=$(basename "$SS_CONFIG") ; size=${WS_GEMM_SIZE}^3"

# --- [1/3] collect: the gated jit hook dumps source-free tasks (kernel.ptx + spec.json). ---
echo "== [1/3] collect =="
TRITON_COMPILE_IQ_COLLECT=1 "$PY" "$KERNEL" 2>&1 | grep -iE "collector|ws *:" || true
# pick the warp-specialized GEMM task (the TMA kernel), not the split-K reduce helper.
WSTASK=""
for d in "$COMPILE_IQ_TASK_DIR"/*/; do
    if grep -q '"matmul_kernel_tma_ws_blackwell"' "$d/spec.json" 2>/dev/null; then WSTASK="$d"; break; fi
done
[ -n "$WSTASK" ] || { echo "FAIL: no WS (TMA) task collected"; exit 1; }
echo "   ws task: $WSTASK"

# --- [2/3] factory: EVO over the tier, each candidate scored by PTX-direct driver launch. ---
echo "== [2/3] factory: EVO PTX-direct ($(basename "$SS_CONFIG"), per-candidate ${PER_CAND_TIMEOUT}s) =="
"$EVO_PY" -u "$HERE/ptx_evo_search.py" "$WSTASK/kernel.ptx" "$WSTASK/spec.json" "$SS_CONFIG" "$PER_CAND_TIMEOUT" \
    2>&1 | grep -E "ptx-evo|PTX_EVO_SEARCH" || true
if [ -f "$WSTASK/best.acf.hex" ]; then
    "$PY" "$HERE/store_acf.py" "$WSTASK" "$WSTASK/best.acf.hex"
else
    echo "   (no ACF minted -- consume will MISS = baseline)"
fi

# --- [3/3] consume: gated make_cubin hook applies the stored ACF on a HIT; sample checks vs torch. ---
echo "== [3/3] consume (timeout ${CONSUME_TIMEOUT}s) =="
rc=0
timeout -k 10 "$CONSUME_TIMEOUT" env \
    TRITON_COMPILE_IQ_APPLY=1 TRITON_COMPILE_IQ_DEBUG=1 TRITON_ALWAYS_COMPILE=1 \
    COMPILE_IQ_PTXAS_MIN_VERSION="$PTXAS_MIN_VERSION" WS_GEMM_SIZE="$WS_GEMM_SIZE" \
    "$PY" "$KERNEL" 2>&1 | grep -iE "compile_iq|HIT|MISS|ws *:|cuBLAS|speed" || rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "!! consume TIMED OUT -- applied ACF likely wedged the GPU. FAIL."; exit 1
fi

echo "== done; store: =="
find "$COMPILE_IQ_STORE" -type f
