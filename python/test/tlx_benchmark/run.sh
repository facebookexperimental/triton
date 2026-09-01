#!/bin/bash
# Run the TLX op perf suite with no arguments.
#
# Picks the idlest GPU and applies third_party/tlx/denoise.sh, because a number
# taken on a shared or unmanaged card is not comparable to anything -- measured
# on B200, the same kernel spanned 13.9% run-to-run unmanaged against 1.7%
# denoised. Everything else is defaulted by bench_mm.py itself.
#
#   run.sh                 measure and gate against the committed baseline
#   run.sh --pytest        same, through pytest, emitting junitxml
#   run.sh <flag>...       any bench_mm.py flag, e.g. --replicates 3
#
# CUDA_VISIBLE_DEVICES, if already set, is respected.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Idlest by used memory. Deliberately not just index 0: on a shared box a
    # co-tenant is the failure mode most likely to go unnoticed, and the
    # harness would otherwise only report it after spending the run.
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | sort -t, -k2 -n | head -n1 | cut -d, -f1 | tr -d ' ')
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "run.sh: no GPU reported by nvidia-smi" >&2
        exit 1
    fi
    echo "run.sh: using GPU $CUDA_VISIBLE_DEVICES (idlest by used memory)"
fi
export CUDA_VISIBLE_DEVICES

if [[ "${1:-}" == "--pytest" ]]; then
    shift
    exec "$ROOT/third_party/tlx/denoise.sh" python -m pytest "$HERE/test_ops_perf.py" -s "$@"
fi
exec "$ROOT/third_party/tlx/denoise.sh" python "$HERE/bench_mm.py" "$@"
