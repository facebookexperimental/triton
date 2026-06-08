#!/usr/bin/env bash
# Waits for the GB200 NVLink fabric / CUDA to come up, then runs the full T3+T4
# GPU validation (constraint-prune tests, the existing autotuner regression
# suite, and the examples). Results are written to:
#     bitequiv/reports/gpu_validation_result.txt
#
# Safe to run unattended (e.g. `nohup bash bitequiv/reports/run_gpu_validation.sh &`)
# or by hand once `python -c "import torch; print(torch.cuda.is_available())"` is True.
set -u
REPO=/home/youngzt/bitwise-equiv/triton
PY=/home/youngzt/bitwise-equiv/triton/.venv/bin/python
OUT="$REPO/bitequiv/reports/gpu_validation_result.txt"
cd "$REPO" || exit 1

{
  echo "=== GPU validation started $(date) ==="
  # Wait up to ~8h (960 * 30s) for the fabric to finish initializing.
  for _ in $(seq 1 960); do
    ok=$("$PY" -c "import torch;print(torch.cuda.is_available())" 2>/dev/null | tail -1)
    [ "$ok" = "True" ] && break
    sleep 30
  done
  ok=$("$PY" -c "import torch;print(torch.cuda.is_available())" 2>/dev/null | tail -1)
  echo "cuda_available=$ok at $(date)"
  if [ "$ok" != "True" ]; then
    echo "RESULT: fabric never came up within the wait window; GPU validation not run."
    echo "=== GPU validation finished $(date) ==="
    exit 0
  fi
  echo
  echo "=== T3/T4 constraint-prune tests (CPU logic + GPU e2e) ==="
  "$PY" -m pytest python/test/unit/runtime/test_autotuner_constraint_prune.py -q 2>&1
  echo
  echo "=== Regression: existing autotuner tests ==="
  "$PY" -m pytest python/test/unit/runtime/test_autotuner.py -q 2>&1 | tail -25
  echo
  echo "=== Examples ==="
  "$PY" bitequiv/examples/constraint_pruning_examples.py 2>&1
  echo
  echo "=== GPU validation finished $(date) ==="
} >"$OUT" 2>&1
