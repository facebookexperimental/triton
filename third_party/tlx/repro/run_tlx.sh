#!/usr/bin/env bash
# Local packaged Torch/Triton runtime; choose an idle gfx950 using HIP_VISIBLE_DEVICES.
set -euo pipefail
ulimit -c 0
exec timeout --signal=INT --kill-after=10s 180 \
    env -u LD_PRELOAD -u LD_LIBRARY_PATH -u ROCP_TOOL_LIBRARIES -u ROC_TOOL_LIB \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    /bin/bash /tmp/hoy_onefeeds/prof/att_workload.sh \
    "$(dirname -- "$(realpath -- "$0")")/repro_tlx_cross_wave.py" "$@"
