#!/usr/bin/env bash
#
# compile_iq e2e pinned to ptxas 12.8 (the B200 production toolkit). Thin wrapper over
# run_e2e_search.sh that sets the three version-coupled knobs together -- the ptxas binary,
# the matching cuda-12.8 search space, and the consume min-version. They MUST agree: a
# cuda-12.8 ACF is rejected ("Invalid compiler controls file") by any other ptxas.
#
# Override as usual: TIER (p0/p1/p2 search-space tier, default p0), PTXAS_KNOBS,
# CUDA_VISIBLE_DEVICES, PER_CAND_TIMEOUT, etc. (passed through to run_e2e_search.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"
TIER="${TIER:-p0}"

exec env \
    PTXAS=/usr/local/cuda-12.8/bin/ptxas \
    SS_CONFIG="$PTXAS_KNOBS/cuda-12.8-ptxas-${TIER}.config" \
    PTXAS_MIN_VERSION=12.8 \
    bash "$HERE/run_e2e_search.sh"
