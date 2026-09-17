#!/bin/bash
# Reproduce the four prepared-regime solves and their four UNSAT ablations.
# Run from third_party/tlx/tools/paper_joint_solver.
#
# This is the disclosed parallel regime, not the paper path: every solve goes
# through `paper_joint_solver.prepared`, which applies the eight input
# assumptions listed in DEVIATIONS "The prepared regime" before handing the
# problem to the unmodified Fig 4/5/6 solver.  `run_main_cases.sh` is the
# literal batch and is untouched by anything here.
#
# Outputs are append-never: remove stale targets or point OUT somewhere fresh.
# Time bounds live in this harness, never in the solver; no exit code is ever
# read as a verdict.
set -euo pipefail

PYTHON_REQUESTED=${PYTHON:-../../../../.venv/bin/python}
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to colon-separated dirs holding libyices and every library libyices itself links against}"
WATCHDOG_S=${WATCHDOG_S:-86400}   # execution-protocol parameter, not a solver input
OUT=${OUT:-prepared}
ABLATIONS="$OUT/ablations"

# B5: the per-case normalization budget.  `fwd` is the one case whose formula
# size explodes at 300.
U_SUBTILED=300
U_FWD=150
U_BWD=300
REG_BUDGET_LR=4096
REFIT_WINDOW=6
FA4_TEMPLATE=subtiled_fa4exact_solution.json

# C1: the min-G descent.  `fwd_subtiled` runs the full bisection because it is
# the case the FA4 shape claim is about and no discount is acceptable there.
# The backward pair is seeded at its v6-era group count instead, with an
# asymmetric L window -- a counting constraint is the SMT solver's hardest
# shape, and a refutation costs the whole window every time while a
# satisfiability probe short-circuits.  `fwd` is skipped: its shape is not
# gated, and until B6 normalizes it the question is not meaningful.
MIN_G_BUDGET_S=${MIN_G_BUDGET_S:-14400}
MIN_G_BWD_BUDGET_S=${MIN_G_BWD_BUDGET_S:-7200}
MIN_G_SAT_WINDOW=6
MIN_G_DECISIVE_WINDOW=8
BWD_GROUP_ANCHOR=4
BWD_LR_GROUP_ANCHOR=8
FA4_GROUP_COUNT=5

GOLDEN=tests/golden_curated
EXAMPLES=../sched2tlx/examples
SUB_CURATED="$GOLDEN/fwd_subtiled/fwd_subtiled.curated_ddg.json"
SUB_MANIFEST="$GOLDEN/fwd_subtiled/fwd_subtiled.curation_manifest.json"
SUB_RAW="$EXAMPLES/case3_FA_fp16_subtiled/ddg.json"
FWD_CURATED="$GOLDEN/fwd/fwd.curated_ddg.json"
FWD_MANIFEST="$GOLDEN/fwd/fwd.curation_manifest.json"
FWD_RAW="$EXAMPLES/case3_FA_fp16/ddg.json"
BWD_CURATED="$GOLDEN/bwd/bwd.curated_ddg.json"
BWD_MANIFEST="$GOLDEN/bwd/bwd.curation_manifest.json"
# The backward golden fixture was curated from the head-dim-128 dump, not from
# the sub-tiled backward dump beside it.  `load_ops_table` checks this against
# curation_source.ddg_sha256 rather than trusting the path written here.
BWD_RAW="$EXAMPLES/case4_FA_bwd/ddg_hd128.json"

if [[ ! -f paper_joint_solver/prepared.py ]]; then
  echo "run from third_party/tlx/tools/paper_joint_solver" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_REQUESTED" ]]; then
  echo "PYTHON must name an executable with PySCIPOpt and Yices" >&2
  exit 1
fi
PYTHON=$PYTHON_REQUESTED
if [[ ! "$WATCHDOG_S" =~ ^[0-9]+$ ]]; then
  echo "WATCHDOG_S must be a non-negative integer" >&2
  exit 1
fi
for required_command in hostname lscpu nproc timeout; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command is not available: $required_command" >&2
    exit 1
  fi
done
if [[ "$SOLVER_LIB_PATH" == :* || "$SOLVER_LIB_PATH" == *: || \
      "$SOLVER_LIB_PATH" == *::* ]]; then
  echo "SOLVER_LIB_PATH must contain non-empty colon-separated directories" >&2
  exit 1
fi
IFS=: read -r -a solver_lib_dirs <<<"$SOLVER_LIB_PATH"
for directory in "${solver_lib_dirs[@]}"; do
  if [[ ! -d "$directory" ]]; then
    echo "solver library directory does not exist: $directory" >&2
    exit 1
  fi
done

mkdir -p "$OUT" "$ABLATIONS"
ENVIRONMENT_LOG="$OUT/environment.log"
COMMANDS_LOG="$OUT/commands.log"
OBS_LOG="$OUT/observations.log"
: >"$COMMANDS_LOG"
: >"$OBS_LOG"

solver_env() {
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" "$@"
}

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'lscpu_model_name=%s\n' \
    "$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
  printf 'nproc=%s\n' "$(nproc)"
  printf 'watchdog_s=%s\n' "$WATCHDOG_S"
  printf 'python_requested=%s\n' "$PYTHON_REQUESTED"
  printf 'SOLVER_LIB_PATH=%s\n' "$SOLVER_LIB_PATH"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH-<unset>}"
  solver_env "$PYTHON" - <<'PY'
import importlib
import sys

from paper_joint_solver.prepared import prepared_sources_sha256
from paper_joint_solver.schedule_plan import solver_sources_sha256

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
for name in ("pyscipopt", "yices", "paper_joint_solver"):
    module = importlib.import_module(name)
    print(f"module_{name}_file={getattr(module, '__file__', '<unset>')}")
print(f"solver_sources_sha256={solver_sources_sha256()}")
print(f"prepared_sources_sha256={prepared_sources_sha256()}")
PY
} >"$ENVIRONMENT_LOG"

# Watchdogged run.  A timeout is a recorded observation, not a failure: the
# regime's non-goal is an unbounded solve, and the harness owns the bound.
run_watchdogged() {
  local label=$1 log=$2
  shift 2
  { printf '%q ' "$@"; printf '\n'; } >>"$COMMANDS_LOG"
  local rc=0
  timeout "$WATCHDOG_S" "$@" >"$log" 2>&1 || rc=$?
  if [[ $rc -eq 124 ]]; then
    printf 'observation label=%s outcome=did-not-terminate watchdog_s=%s\n' \
      "$label" "$WATCHDOG_S" >>"$OBS_LOG"
    echo "$label: did not terminate within ${WATCHDOG_S}s"
    return 1
  fi
  if [[ $rc -ne 0 ]]; then
    cat "$log" >&2
    echo "$label exited $rc" >&2
    return 1
  fi
  printf 'observation label=%s outcome=completed\n' "$label" >>"$OBS_LOG"
  return 0
}

run_case() {
  local stem=$1 curated=$2 raw=$3 manifest=$4 u=$5
  shift 5
  echo "running $stem"
  local -a command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -u -m paper_joint_solver.prepared "$curated"
    --raw-ddg "$raw"
    --normalization-u "$u"
    -o "$OUT/$stem.json"
    --pool-out "$OUT/$stem.pool.json"
    --strategy-out "$OUT/$stem.strategy.json"
    --curation-manifest "$manifest"
    "$@"
  )
  run_watchdogged "$stem" "$OUT/$stem.log" "${command[@]}" || true
}

run_probe() {
  local stem=$1 curated=$2 raw=$3 u=$4
  shift 4
  echo "probing $stem"
  local -a command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -u -m paper_joint_solver.prepared "$curated"
    --raw-ddg "$raw"
    --normalization-u "$u"
    --probe
    -o "$ABLATIONS/$stem.json"
    "$@"
  )
  run_watchdogged "ablation.$stem" "$ABLATIONS/$stem.log" "${command[@]}" || true
}

# The four canonical cases.  fwd_subtiled additionally runs the exact-partition
# refit audit, which is the only case the FA4 template covers.
run_case fwd_subtiled_prep1 "$SUB_CURATED" "$SUB_RAW" "$SUB_MANIFEST" \
  "$U_SUBTILED" --fa4-template "$FA4_TEMPLATE" --refit-window "$REFIT_WINDOW" \
  --min-warp-sets --min-warp-sets-window "$MIN_G_SAT_WINDOW" \
  --min-warp-sets-decisive-window "$MIN_G_DECISIVE_WINDOW" \
  --min-warp-sets-budget-s "$MIN_G_BUDGET_S" \
  --shape-probe-groups "$FA4_GROUP_COUNT"
run_case fwd_prep1 "$FWD_CURATED" "$FWD_RAW" "$FWD_MANIFEST" "$U_FWD"
run_case bwd_prep1 "$BWD_CURATED" "$BWD_RAW" "$BWD_MANIFEST" "$U_BWD" \
  --min-warp-sets --min-warp-sets-anchor "$BWD_GROUP_ANCHOR" \
  --min-warp-sets-window "$MIN_G_SAT_WINDOW" \
  --min-warp-sets-decisive-window "$MIN_G_DECISIVE_WINDOW" \
  --min-warp-sets-budget-s "$MIN_G_BWD_BUDGET_S"
run_case bwd_lr4096_prep1 "$BWD_CURATED" "$BWD_RAW" "$BWD_MANIFEST" "$U_BWD" \
  --reg-budget "$REG_BUDGET_LR" \
  --min-warp-sets --min-warp-sets-anchor "$BWD_LR_GROUP_ANCHOR" \
  --min-warp-sets-window "$MIN_G_SAT_WINDOW" \
  --min-warp-sets-decisive-window "$MIN_G_DECISIVE_WINDOW" \
  --min-warp-sets-budget-s "$MIN_G_BWD_BUDGET_S"

# The ablations use the published first-window protocol rather than Algorithm
# 1: Algorithm 1 does not terminate on a structurally unsatisfiable model, so
# bounded evidence needs the probe verb.
run_probe warps_minus_one "$SUB_CURATED" "$SUB_RAW" "$U_SUBTILED" \
  --num-warps-override 31
run_probe warps_half "$SUB_CURATED" "$SUB_RAW" "$U_SUBTILED" \
  --num-warps-override 16
run_probe no_subtiling "$FWD_CURATED" "$FWD_RAW" "$U_FWD"
run_probe no_cross_warp "$SUB_CURATED" "$SUB_RAW" "$U_SUBTILED" --no-cross-warp

echo "wrote $OUT (observations in $OBS_LOG)"
