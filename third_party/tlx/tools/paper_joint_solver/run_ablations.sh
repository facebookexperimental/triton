#!/bin/bash
# UNSAT ablations (paper sec 6.2.2) on the sub-tiled forward fixture:
#   1. reduced physical warps  (--num-warps: baseline-1 and half)
#   2. no sub-tiling           (the non-subtiled fwd ddg as input)
#   3. no cross-warp traffic   (--no-cross-warp)
# Run from the package directory. Each solve gets the same search/time limits.
# The JSON status is authoritative. Neither unknown nor resource_limited is
# promoted to an UNSAT proof.
set -u
VENV=../../../../.venv/bin/python
# SOLVER_LIB_PATH: colon-separated lib dirs holding yices/cudd shared objects
export LD_LIBRARY_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to <yices>/lib:<cudd>/lib}"
SUB=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD=../sched2tlx/examples/case3_FA_fp16/ddg.json
SUB_GRAPH=../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json
FWD_GRAPH=../sched2tlx/examples/case3_FA_fp16/schedule_graph.json
OUT=ablations
FIXED_WARPS=4  # sched2tlx emits a four-warp default task outside the loop DDG
mkdir -p $OUT
run() {
  name=$1; shift
  echo "=== ablation: $name ==="
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    $VENV -m paper_joint_solver "$@" -o $OUT/$name.json \
    --warp-fixed-overhead $FIXED_WARPS \
    --ilp-seconds 240 --smt-seconds 300 --max-wall-s 3600
}
# Derive physical-warp ablations from a baseline produced by this exact model.
run baseline $SUB --baseline-graph $SUB_GRAPH
BASE_WARPS=$("$VENV" - "$OUT/baseline.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
if result["status"] != "sat":
    raise SystemExit("baseline did not produce a SAT solution")
print(result["stats"]["num_warps"])
PY
)
run warps_minus_one $SUB --baseline-graph $SUB_GRAPH \
  --num-warps $((BASE_WARPS - 1))
run warps_half $SUB --baseline-graph $SUB_GRAPH \
  --num-warps $((BASE_WARPS / 2))
run no_subtiling $FWD --baseline-graph $FWD_GRAPH
run no_cross_warp $SUB --baseline-graph $SUB_GRAPH --no-cross-warp
"$VENV" - "$OUT"/*.json <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        result = json.load(handle)
    print(f"{path}: status={result['status']} satisfiable={result['satisfiable']}")
PY
