#!/usr/bin/env bash
#
# compile_iq PTX-direct e2e pinned to ptxas 12.8 (the B200 production toolkit). Thin wrapper over
# run_ptx_e2e.sh that pins the version-coupled pair together -- the ptxas binary and the matching
# cuda-12.8 search space. They MUST agree: a cuda-12.8 ACF is rejected ("Invalid compiler controls
# file") by any other ptxas, and the PTX ISA Triton emits must be runnable by the installed driver.
#
# Override as usual: TIER (p0/p1/p2 search-space tier, default p0), PTXAS_KNOBS,
# CUDA_VISIBLE_DEVICES, PER_CAND_TIMEOUT, etc. (passed through to run_ptx_e2e.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"
TIER="${TIER:-p0}"

exec env \
    PTXAS=/usr/local/cuda-12.8/bin/ptxas \
    SS_CONFIG="$PTXAS_KNOBS/cuda-12.8-ptxas-${TIER}.config" \
    bash "$HERE/run_ptx_e2e.sh"
