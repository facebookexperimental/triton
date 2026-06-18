#!/usr/bin/env bash
#
# compile_iq e2e pinned to ptxas 13.0. Thin wrapper over run_e2e_search.sh that sets the
# three version-coupled knobs together -- the ptxas binary, the matching cuda-13.0 search
# space, and the consume min-version. They MUST agree: a cuda-13.0 ACF is rejected
# ("Invalid compiler controls file") by any other ptxas.
#
# Override as usual: TIER (p0/p1/p2 search-space tier, default p0), PTXAS_KNOBS,
# CUDA_VISIBLE_DEVICES, PER_CAND_TIMEOUT, etc. (passed through to run_e2e_search.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXAS_KNOBS="${PTXAS_KNOBS:-/data/users/$USER/ptxas_knobs}"
TIER="${TIER:-p0}"

exec env \
    PTXAS=/usr/local/cuda-13.0/bin/ptxas \
    SS_CONFIG="$PTXAS_KNOBS/cuda-13.0-ptxas-${TIER}.config" \
    PTXAS_MIN_VERSION=13.0 \
    bash "$HERE/run_e2e_search.sh"
