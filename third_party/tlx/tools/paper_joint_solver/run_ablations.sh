#!/bin/bash
# UNSAT ablations (paper sec 6.2.2) on the sub-tiled forward fixture:
#   1. reduced warp count      (--num-warps: groups-1 and half)
#   2. no sub-tiling           (the non-subtiled fwd ddg as input)
#   3. no cross-warp traffic   (--no-cross-warp)
# Run from the package directory.  Each solve gets the same (I, L) budget the
# SAT baseline used; a verdict of "no solution" across the searched window is
# the ablation's UNSAT result.
set -u
VENV=../../../../.venv/bin/python
export LD_LIBRARY_PATH=/projects/kzhou6/hwu27/tools/yices-2.7.0/lib:/projects/kzhou6/hwu27/tools/cudd/lib
SUB=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD=../sched2tlx/examples/case3_FA_fp16/ddg.json
OUT=ablations
mkdir -p $OUT
run() {
  name=$1; shift
  echo "=== ablation: $name ==="
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $VENV -m paper_joint_solver "$@" -o $OUT/$name.json \
    --ilp-seconds 240 --smt-seconds 300 --max-wall-s 3600
}
run warps_minus_one $SUB --num-warps 4
run warps_half      $SUB --num-warps 3
run no_subtiling    $FWD
run no_cross_warp   $SUB --no-cross-warp
grep -H satisfiable $OUT/*.json
