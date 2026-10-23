#!/bin/bash
# Usage: ./check.sh <llc> [FileCheck]
# Exit 0 if this llc keeps lgkmcnt(0) before the loop-header s_barrier, 1 if it has the bug.
set -u
LLC=${1:?usage: $0 <llc> [FileCheck]}
FILECHECK=${2:-$(dirname "$LLC")/FileCheck}
cd "$(dirname "$0")"
LLC_FLAGS=(-O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950)
status=0

# Print the s_waitcnt (or "none") right before the first s_barrier of the inner loop.
loop_head_wait() {
  "$LLC" "${LLC_FLAGS[@]}" "$1" -o - | awk '
    /Loop Header: Depth=1/ { inloop = 1; prev = "none"; next }
    inloop && /^\s*s_waitcnt/ { prev = $0; sub(/^\s*/, "", prev) }
    inloop && /^\s*s_barrier/ { print prev; exit }
    inloop && /^\s*(v_|ds_|buffer_)/ { prev = "none" }'
}

echo "llc: $LLC"
if "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -o - waitcnt-loop-soft-wait.ll |
    "$FILECHECK" waitcnt-loop-soft-wait.ll > /dev/null 2>&1; then
  echo "PASS  waitcnt-loop-soft-wait.ll"
else
  echo "FAIL  waitcnt-loop-soft-wait.ll  (soft lgkmcnt(0) erased at the loop header)"
  status=1
fi

for kernel in kernel_nofix.ll kernel_fix.ll; do
  wait=$(loop_head_wait "$kernel")
  if [[ $wait == *"lgkmcnt(0)"* ]]; then
    echo "PASS  $kernel  loop head: $wait; s_barrier"
  else
    echo "FAIL  $kernel  loop head: $wait; s_barrier  (pending ds_read crosses the barrier)"
    status=1
  fi
done
exit $status
