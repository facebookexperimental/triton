#!/usr/bin/env bash
#
# compile_iq end-to-end -- PTX-direct route. Tune and apply a ptxas ACF by assembling a FIXED
# kernel.ptx and launching the cubin via the CUDA DRIVER API -- no Triton recompile from source,
# and NOTHING but PTX + a launch spec is collected (no source.py is ever dumped). Three stages:
#
#   [1/3] emit   -- compile the naive matmul once -> kernel.ptx + spec.json (the source-free launch
#                   description), and prove ptxas(kernel.ptx) -> cubin -> driver-launch == torch + Triton.
#   [2/3] mint   -- EVO search over a CONSTRAINED ptxas .config tier, scoring each candidate through
#                   the PTX-direct driver-launch bridge (ptx_bench_one.py). Writes best.acf.hex.
#   [3/3] apply  -- ptxas(kernel.ptx, --apply-controls=best.acf) -> cubin -> driver-launch, validated
#                   (self-consistent) and timed vs baseline.
#
# VERSION MATCHING IS REQUIRED. A cuda-X.Y ACF is only accepted by ptxas X.Y, AND the *PTX itself*
# must be X.Y-compatible: Triton picks the PTX ISA .version from the ptxas it sees at compile time,
# and a CUDA-X.Y driver cannot run a newer-ISA cubin. So PTXAS must match the search-space cuda
# version AND the installed driver. Use the run_e2e_cuda12{8,30}.sh wrappers to pin a matching set;
# this script defaults to the cuda-13.0 tier + /usr/local/cuda-13.0 ptxas (a CUDA-13.0 driver box).
#
# Requirements: base python (fbtriton + torch + cuda-python); the `evo` conda env (evo_nda); a ptxas
# matching the tier's cuda version AND the driver; the search-space `.config`.
#
# Usage: third_party/compile_iq/smoke/run_ptx_e2e.sh
# Optional env: PTXAS_KNOBS, PY, EVO_PY, PTXAS, SS_CONFIG, PER_CAND_TIMEOUT, WORK, CUDA_VISIBLE_DEVICES.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="$HERE/ptx_direct_smoke.py"
EVO_SEARCH="$HERE/ptx_evo_search.py"

PY="${PY:-python}"                                          # base 3.13 (fbtriton + torch + cuda-python)
EVO_PY="${EVO_PY:-$(conda info --base 2>/dev/null)/envs/evo/bin/python}"  # the `evo` env (3.10)
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}" # where the Manifold drop was unpacked
SS_CONFIG="${SS_CONFIG:-$PTXAS_KNOBS/cuda-13.0-ptxas-p0.config}"
PTXAS="${PTXAS:-/usr/local/cuda-13.0/bin/ptxas}"           # must match SS_CONFIG version AND the driver
PER_CAND_TIMEOUT="${PER_CAND_TIMEOUT:-10}"  # valid PTX-direct candidates run in ~2-3s; wedges die fast
WORK="${WORK:-$HOME/.compile_iq/ptx_smoke}"

[ -x "$EVO_PY" ]    || { echo "FAIL: evo python not found at '$EVO_PY' -- set EVO_PY=/path/to/envs/evo/bin/python"; exit 1; }
[ -f "$SS_CONFIG" ] || { echo "FAIL: search space not found at '$SS_CONFIG' -- set SS_CONFIG or PTXAS_KNOBS"; exit 1; }
[ -x "$PTXAS" ]     || { echo "FAIL: ptxas not found at '$PTXAS' -- set PTXAS"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_PTXAS_BLACKWELL_PATH="$PTXAS"  # Triton emits PTX targeting THIS ptxas's ISA version
export BASE_PY="$PY"                          # ptx_evo_search shells back to this for ptx_bench_one

echo "ptxas=$PTXAS ($("$PTXAS" --version | grep -oE 'V[0-9.]+' | tail -1)) ; space=$(basename "$SS_CONFIG")"

# --- [1/3] emit kernel.ptx + spec.json and prove baseline driver launch ---
echo "== [1/3] emit + baseline (PTX-direct driver launch) =="
"$PY" -u "$SMOKE" --ptxas "$PTXAS" --work-dir "$WORK"

# --- [2/3] mint an ACF: EVO search scored via the PTX-direct bridge ---
echo "== [2/3] mint ACF: EVO over $(basename "$SS_CONFIG") (per-candidate ${PER_CAND_TIMEOUT}s) =="
"$EVO_PY" -u "$EVO_SEARCH" "$WORK/kernel.ptx" "$WORK/spec.json" "$SS_CONFIG" "$PER_CAND_TIMEOUT" \
    2>&1 | grep -E "ptx-evo|PTX_EVO_SEARCH" || true
[ -f "$WORK/best.acf.hex" ] || { echo "FAIL: EVO minted no ACF (no best.acf.hex)"; exit 1; }

# --- [3/3] apply the minted ACF via PTX-direct driver launch ---
echo "== [3/3] apply (ptxas --apply-controls + driver launch) =="
"$PY" -u "$SMOKE" --ptxas "$PTXAS" --work-dir "$WORK" --acf-hex "$WORK/best.acf.hex"

echo "== done =="
