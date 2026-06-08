#!/usr/bin/env bash
# Run all Starter-Task-T2 examples (both groups) on a GPU and report pass/fail.
#
#   bash bitequiv/reports/run_t2_examples.sh
#
# Each example compiles a kernel, prints the IR/PTX evidence for a compiler
# behavior, and asserts its runtime result. Group A (knowledge-base/compilation-
# pipeline) is bit-neutral pipeline mechanics; Group B (examples/layout_numerics)
# is the project's bit-changing numerics. Blackwell-only examples self-skip
# elsewhere. Exits non-zero if any example fails.
set -u

# Repo root = two levels up from this script (bitequiv/reports/ -> repo).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2

# Prefer the repo venv python, else whatever python3 is on PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
echo "python: $PY"
"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || { echo "ERROR: no CUDA GPU available"; exit 2; }

EXAMPLES=(
  bitequiv/knowledge-base/compilation-pipeline/01_read_ttir.py
  bitequiv/knowledge-base/compilation-pipeline/02_layout_assignment.py
  bitequiv/knowledge-base/compilation-pipeline/03_coalesce_vectorization.py
  bitequiv/knowledge-base/compilation-pipeline/04_remove_layout_conversions.py
  bitequiv/knowledge-base/compilation-pipeline/05_software_pipelining.py
  bitequiv/knowledge-base/compilation-pipeline/06_warp_specialization.py
  bitequiv/examples/layout_numerics/b01_reduction_tree_from_layout.py
  bitequiv/examples/layout_numerics/b02_reduction_ordering_inner_tree.py
  bitequiv/examples/layout_numerics/b03_mma_precision.py
  bitequiv/examples/layout_numerics/b04_f32_dot_tc.py
  bitequiv/examples/layout_numerics/b05_fma_contraction.py
  bitequiv/examples/layout_numerics/b06_elementwise_math_precision.py
)

fail=0
for ex in "${EXAMPLES[@]}"; do
  if "$PY" "$ex" >"/tmp/$(basename "$ex").log" 2>&1; then
    if grep -q "skipped" "/tmp/$(basename "$ex").log"; then
      echo "SKIP  $ex"
    else
      echo "PASS  $ex"
    fi
  else
    echo "FAIL  $ex   (see /tmp/$(basename "$ex").log)"
    fail=1
  fi
done

echo
if [[ $fail -eq 0 ]]; then echo "ALL T2 EXAMPLES OK"; else echo "SOME T2 EXAMPLES FAILED"; fi
exit $fail
