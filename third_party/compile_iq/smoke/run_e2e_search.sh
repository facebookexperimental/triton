#!/usr/bin/env bash
#
# compile_iq end-to-end (search route): collect -> factory -> consume, on the unmodified naive
# matmul (../examples/user_kernel.py). The 3-step flow is engine-neutral; only the factory step
# picks an optimizer. This script currently drives the EVO engine (evo_search.py) over a
# CONSTRAINED ptxas knob space (a manifold `.config` tier) instead of CIQ's full `ptxas13.3.bin`
# -- whose random candidates wedge the GPU so the search never converges. A CIQ-backed factory
# could be swapped in here without touching collect/consume.
#
# Why EVO here: the constrained `.config` tiers are EVO-format (the CIQ core can't parse them), and
# the EVO env (Python 3.10) has no triton/torch, so each candidate is benchmarked in base 3.13 via
# the bench_one.py bridge (subprocess). Stores are hermetic temp dirs; leaves no state behind.
#
# Version matching is REQUIRED: a `cuda-X.Y` search space yields ACFs that only a ptxas X.Y accepts
# (a mismatched ptxas rejects them: "Invalid compiler controls file"). Defaults below pair the
# cuda-13.0 p0 tier with ptxas 13.0, and tell the consume hook to allow 13.0 via
# COMPILE_IQ_PTXAS_MIN_VERSION (the hook defaults to >=13.3 for the production 13.3 search space).
#
# Requirements: base python with torch + this fbtriton; the `evo` conda env (evo_nda); a ptxas
# matching the search-space version; the search-space `.config`.
#
# Usage: third_party/compile_iq/smoke/run_e2e_search.sh
# Optional env: PTXAS_KNOBS (where the Manifold drop is), PY, EVO_PY, PTXAS, SS_CONFIG,
#   PTXAS_MIN_VERSION, PER_CAND_TIMEOUT, CONSUME_TIMEOUT, CUDA_VISIBLE_DEVICES.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$HERE/../examples/user_kernel.py"

PY="${PY:-python}"                                          # base 3.13 (fbtriton + torch)
EVO_PY="${EVO_PY:-$(conda info --base 2>/dev/null)/envs/evo/bin/python}"  # the `evo` env (3.10)
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"  # where the Manifold drop was unpacked
SS_CONFIG="${SS_CONFIG:-$PTXAS_KNOBS/cuda-13.0-ptxas-p0.config}"
PTXAS="${PTXAS:-/usr/local/cuda-13.0/bin/ptxas}"            # must match SS_CONFIG's cuda version
PTXAS_MIN_VERSION="${PTXAS_MIN_VERSION:-13.0}"
PER_CAND_TIMEOUT="${PER_CAND_TIMEOUT:-15}"
CONSUME_TIMEOUT="${CONSUME_TIMEOUT:-90}"

[ -x "$EVO_PY" ]   || { echo "FAIL: evo python not found at '$EVO_PY' -- set EVO_PY=/path/to/envs/evo/bin/python"; exit 1; }
[ -f "$SS_CONFIG" ] || { echo "FAIL: search space not found at '$SS_CONFIG' -- set SS_CONFIG or PTXAS_KNOBS"; exit 1; }
[ -x "$PTXAS" ]    || { echo "FAIL: ptxas not found at '$PTXAS' -- set PTXAS"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_PTXAS_BLACKWELL_PATH="$PTXAS"
export BASE_PY="$PY"                                         # evo_search.py shells back to this for bench_one
export COMPILE_IQ_TASK_DIR="${COMPILE_IQ_TASK_DIR:-$(mktemp -d)}"
export COMPILE_IQ_STORE="${COMPILE_IQ_STORE:-$(mktemp -d)}"

echo "ptxas=$PTXAS ($("$PTXAS" --version | grep -oE 'V[0-9.]+' | tail -1)) ; space=$(basename "$SS_CONFIG")"

# --- [1/3] collect -- gated jit.py hook dumps a task (stdlib only; zero kernel surface). ---
echo "== [1/3] collect =="
FBTRITON_COMPILE_IQ_COLLECT=1 "$PY" "$KERNEL"
TASK="$(ls -d "$COMPILE_IQ_TASK_DIR"/*/ | head -1)"
echo "   task: $TASK"

# --- [2/3] factory (EVO search over the constrained .config) + store the best ACF. ---
echo "== [2/3] factory: EVO search ($(basename "$SS_CONFIG"), per-candidate ${PER_CAND_TIMEOUT}s) =="
"$EVO_PY" -u "$HERE/evo_search.py" "$TASK" "$SS_CONFIG" "$PER_CAND_TIMEOUT" \
    2>&1 | grep -E "evo-search|EVO_SEARCH" || true
"$PY" "$HERE/store_best.py" "$TASK" "$TASK/best.acf.hex"

# --- [3/3] consume -- gated make_cubin hook applies the stored ACF on a HIT. ---
echo "== [3/3] consume (timeout ${CONSUME_TIMEOUT}s) =="
rc=0
timeout -k 10 "$CONSUME_TIMEOUT" env \
    FBTRITON_COMPILE_IQ_APPLY=1 FBTRITON_COMPILE_IQ_DEBUG=1 TRITON_ALWAYS_COMPILE=1 \
    COMPILE_IQ_PTXAS_MIN_VERSION="$PTXAS_MIN_VERSION" \
    "$PY" "$KERNEL" 2>&1 | grep -iE "compile_iq|HIT|matmul ran" || rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "!! consume TIMED OUT -- applied ACF likely wedged the GPU. FAIL."; exit 1
fi

echo "== done; store: =="
find "$COMPILE_IQ_STORE" -type f
