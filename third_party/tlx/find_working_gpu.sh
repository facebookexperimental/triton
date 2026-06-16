#!/bin/bash

# Find which physical GPUs actually work by running a tiny PyTorch probe on each.
#
# On a shared dev server GPUs are often busy, OOM, or in a bad state. This script
# scans every physical GPU (ignoring any inherited CUDA_VISIBLE_DEVICES) and runs
# a small real CUDA workload on it. It prints which GPU indices are usable and
# ends with a machine-parseable line:
#
#   WORKING_GPUS=0,2,3
#
# Exit code is 0 if at least one GPU works, non-zero otherwise.

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found; cannot enumerate GPUs." >&2
  exit 1
fi

# Per-probe timeout (seconds) so a hung/stuck GPU doesn't block the whole scan.
PROBE_TIMEOUT="${PROBE_TIMEOUT:-60}"

# Enumerate all physical GPU indices. Deliberately ignore CUDA_VISIBLE_DEVICES:
# the point is to scan all hardware to find a healthy one.
GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

if [ -z "$GPU_INDICES" ]; then
  echo "No GPUs reported by nvidia-smi." >&2
  echo "WORKING_GPUS="
  exit 1
fi

WORKING=()

for i in $GPU_INDICES; do
  # Isolate the probe to a single physical GPU so torch sees it as device 0.
  OUTPUT=$(CUDA_VISIBLE_DEVICES="$i" timeout "$PROBE_TIMEOUT" python -c "
import torch
x = torch.randn(256, 256, device='cuda')
torch.mm(x, x)
torch.cuda.synchronize()
print('ok')
" 2>/dev/null)

  if [ "$OUTPUT" = "ok" ]; then
    echo "GPU $i: OK"
    WORKING+=("$i")
  else
    echo "GPU $i: FAILED (busy/unavailable)"
  fi
done

# Join working indices with commas.
WORKING_CSV=$(
  IFS=,
  echo "${WORKING[*]}"
)

echo "WORKING_GPUS=$WORKING_CSV"

if [ "${#WORKING[@]}" -eq 0 ]; then
  exit 1
fi
