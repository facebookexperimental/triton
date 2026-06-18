#!/usr/bin/env bash
#
# compile_iq WS end-to-end: collect -> factory(EVO) -> consume on the warp-specialized
# blackwell GEMM (examples/blackwell_gemm_ws_sample.py). EVO, ptxas 12.8, p2 search space,
# NON-smoke (a candidate is stored only if it beats baseline). WS kernels can't be replayed
# from reconstructed args, so the factory uses the REAL matmul launch (bench_one_ws.py) with
# the ACF applied via PTXAS_OPTIONS.
#
# Defaults to WS_GEMM_SIZE=2048: the sample's 8192^3 config overflows B200 shared memory with
# the current TLX heuristic; 2048^3 uses a fitting config and runs the real WS kernel.
#
# Usage: third_party/compile_iq/smoke/run_e2e_ws.sh
# Optional env: WS_GEMM_SIZE, PTXAS_KNOBS, SS_CONFIG, PTXAS, PTXAS_MIN_VERSION, EVO_PY, PY,
#   PER_CAND_TIMEOUT, CONSUME_TIMEOUT, EVO_GENERATIONS, EVO_POOL, CUDA_VISIBLE_DEVICES.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$HERE/../examples/blackwell_gemm_ws_sample.py"

PY="${PY:-python}"
EVO_PY="${EVO_PY:-$(conda info --base 2>/dev/null)/envs/evo/bin/python}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"
SS_CONFIG="${SS_CONFIG:-$PTXAS_KNOBS/cuda-12.8-ptxas-p2.config}"
PTXAS_MIN_VERSION="${PTXAS_MIN_VERSION:-12.8}"
PER_CAND_TIMEOUT="${PER_CAND_TIMEOUT:-300}"
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

echo "== [1/3] collect =="
FBTRITON_COMPILE_IQ_COLLECT=1 "$PY" "$KERNEL"
TASK="$(ls -d "$COMPILE_IQ_TASK_DIR"/*/ | head -1)"
echo "   task: $TASK"

echo "== [2/3] factory: EVO ws search ($(basename "$SS_CONFIG"), non-smoke, per-candidate ${PER_CAND_TIMEOUT}s) =="
"$EVO_PY" -u "$HERE/evo_search_ws.py" "$TASK" "$SS_CONFIG" "$PER_CAND_TIMEOUT" 2>&1 | grep -E "evo-ws|EVO_WS" || true
if [ -f "$TASK/best.acf.hex" ]; then
    "$PY" "$HERE/store_best.py" "$TASK" "$TASK/best.acf.hex"
else
    echo "   (no ACF beat baseline -- nothing stored; consume will MISS = baseline)"
fi

echo "== [3/3] consume (timeout ${CONSUME_TIMEOUT}s) =="
rc=0
timeout -k 10 "$CONSUME_TIMEOUT" env \
    FBTRITON_COMPILE_IQ_APPLY=1 FBTRITON_COMPILE_IQ_DEBUG=1 TRITON_ALWAYS_COMPILE=1 \
    COMPILE_IQ_PTXAS_MIN_VERSION="$PTXAS_MIN_VERSION" WS_GEMM_SIZE="$WS_GEMM_SIZE" \
    "$PY" "$KERNEL" 2>&1 | grep -iE "compile_iq|HIT|MISS|ws *:|cuBLAS|speed" || rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "!! consume TIMED OUT -- applied ACF likely wedged the GPU. FAIL."; exit 1
fi

echo "== done; store: =="
find "$COMPILE_IQ_STORE" -type f
