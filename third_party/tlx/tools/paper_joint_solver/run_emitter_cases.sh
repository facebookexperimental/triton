#!/bin/bash
# Solve the three sched2tlx-compatible benchmark cases. These artifacts add
# non-paper prefix/full-group lane-mask selection constraints and must remain
# separate from the paper-fidelity v8 solutions produced by run_main_cases.sh.
set -euo pipefail

PYTHON_REQUESTED=${PYTHON:-../../../../.venv/bin/python}
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to <yices>/lib:<cudd>/lib}"
SOLVER_CPU=${SOLVER_CPU:-0}
DEFAULT_OUT=solutions
OUT_WAS_EXPLICIT=${OUT+x}
OUT=${OUT:-$DEFAULT_OUT}
UNPINNED_HOST=${UNPINNED_HOST:-0}
NORMALIZATION_U=300

if [[ ! -f paper_joint_solver/__main__.py ]]; then
  echo "run from third_party/tlx/tools/paper_joint_solver" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_REQUESTED" ]]; then
  echo "PYTHON must name an executable with PySCIPOpt and Yices" >&2
  exit 1
fi
PYTHON=$PYTHON_REQUESTED
if [[ ! "$SOLVER_CPU" =~ ^[0-9]+$ ]]; then
  echo "SOLVER_CPU must be a non-negative integer" >&2
  exit 1
fi
for required_command in hostname lscpu nproc taskset; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command is not available: $required_command" >&2
    exit 1
  fi
done
if [[ "$UNPINNED_HOST" != 0 && "$UNPINNED_HOST" != 1 ]]; then
  echo "UNPINNED_HOST must be 0 or 1" >&2
  exit 1
fi
if ! taskset -c "$SOLVER_CPU" true >/dev/null 2>&1; then
  echo "SOLVER_CPU is not an online CPU available to this process: $SOLVER_CPU" >&2
  exit 1
fi
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

FWD_DDG=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD_GRAPH=../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json
BWD_DDG=../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json
BWD_GRAPH=../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json
for input in "$FWD_DDG" "$FWD_GRAPH" "$BWD_DDG" "$BWD_GRAPH"; do
  if [[ ! -f "$input" ]]; then
    echo "required input does not exist: $input" >&2
    exit 1
  fi
done

if [[ -e "$OUT" && ! -d "$OUT" ]]; then
  echo "OUT exists but is not a directory: $OUT" >&2
  exit 1
fi
stems=(
  fwd_subtiled_emitter_v8
  bwd_emitter_v8
  bwd_lr4096_emitter_v8
)
reserved=(
  "$OUT/run_emitter_cases_environment.log"
  "$OUT/run_emitter_cases_validation.log"
)
for stem in "${stems[@]}"; do
  reserved+=(
    "$OUT/$stem.json"
    "$OUT/$stem.solve.command"
    "$OUT/$stem.solve.log"
  )
done
for target in "${reserved[@]}"; do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing target: $target" >&2
    exit 1
  fi
done

MACHINE_HOSTNAME=$(hostname)
LSCPU_OUTPUT=$(LC_ALL=C lscpu)
LSCPU_MODEL=$(sed -n 's/^Model name:[[:space:]]*//p' <<<"$LSCPU_OUTPUT")
LSCPU_CPU_COUNT=$(sed -n 's/^CPU(s):[[:space:]]*//p' <<<"$LSCPU_OUTPUT")
# BEGIN HOST PINNING POLICY
PAPER_COMPARABLE=yes
if [[ "$UNPINNED_HOST" == 1 ]]; then
  if [[ "$OUT_WAS_EXPLICIT" != x ]]; then
    echo "UNPINNED_HOST=1 requires an explicit non-default OUT" >&2
    exit 1
  fi
  OUT_REALPATH=$(
    "$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$OUT"
  )
  DEFAULT_OUT_REALPATH=$(
    "$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' \
      "$DEFAULT_OUT"
  )
  if [[ "$OUT_REALPATH" == "$DEFAULT_OUT_REALPATH" ]]; then
    echo "UNPINNED_HOST=1 cannot write the canonical OUT=$DEFAULT_OUT" >&2
    exit 1
  fi
  PAPER_COMPARABLE=no
else
  if [[ "$MACHINE_HOSTNAME" != dgx003 && "$MACHINE_HOSTNAME" != dgx003.* ]]; then
    echo "controlled emitter solver runs must execute on dgx003" >&2
    exit 1
  fi
  if [[ "$LSCPU_MODEL" != *8570* ]]; then
    echo "solver runs require Xeon Platinum 8570: $LSCPU_MODEL" >&2
    exit 1
  fi
fi
# END HOST PINNING POLICY
CUDA_VERSION='<unavailable>'
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION=$(nvcc --version | tail -n 1)
fi
NVIDIA_SMI_INFO='<unavailable>'
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia_smi_output=$(
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
  ); then
    NVIDIA_SMI_INFO=$(printf '%s\n' "$nvidia_smi_output" | tr '\n' ';')
    NVIDIA_SMI_INFO=${NVIDIA_SMI_INFO%;}
  else
    NVIDIA_SMI_INFO="<failed: $nvidia_smi_output>"
  fi
fi

mkdir -p "$OUT"
# BEGIN SOURCE VCS DETECTION
SOURCE_VCS='' SOURCE_REVISION='' SOURCE_DIRTY='<unavailable>'
if git rev-parse --git-dir >/dev/null 2>&1; then
  SOURCE_VCS=git
  SOURCE_REVISION=$(git rev-parse HEAD)
  if git_status=$(git status --porcelain --untracked-files=no 2>&1); then
    [[ -n "$git_status" ]] && SOURCE_DIRTY=yes || SOURCE_DIRTY=no
  fi
elif command -v sl >/dev/null 2>&1; then
  sapling_root=$(sl root --reason \
    "Detect the Sapling checkout - sl help root" 2>/dev/null) || sapling_root=''
  if [[ -n "$sapling_root" && -d "$sapling_root" ]]; then
    SOURCE_VCS=sapling
    SOURCE_REVISION=$(sl log --reason \
      "Record the paper-solver source revision - sl help log" \
      -r . -T '{node}\n')
    if sl_status=$(sl status --reason \
      "Record source dirtiness - sl help status" 2>&1); then
      [[ -n "$sl_status" ]] && SOURCE_DIRTY=yes || SOURCE_DIRTY=no
    fi
  fi
fi
if [[ -z "$SOURCE_VCS" || -z "$SOURCE_REVISION" ]]; then
  echo "cannot record the source revision: not a git worktree and not a Sapling checkout" >&2
  exit 1
fi
# END SOURCE VCS DETECTION
{
  printf 'classification=%s\n' 'non-paper-emitter-compatibility'
  printf 'selection_constraints=%s\n' 'prefix_lane_masks,full_group_lane_masks'
  printf 'paper_comparable=%s\n' "$PAPER_COMPARABLE"
  printf 'working_directory=%s\n' "$(pwd -P)"
  printf 'hostname=%s\n' "$MACHINE_HOSTNAME"
  printf 'lscpu_model_name=%s\n' "$LSCPU_MODEL"
  printf 'lscpu_cpu_count=%s\n' "$LSCPU_CPU_COUNT"
  printf 'nproc=%s\n' "$(nproc)"
  printf 'solver_cpu=%s\n' "$SOLVER_CPU"
  printf 'cuda_version=%s\n' "$CUDA_VERSION"
  printf 'nvidia_smi=%s\n' "$NVIDIA_SMI_INFO"
  printf 'source_vcs=%s\n' "$SOURCE_VCS"
  printf 'source_revision=%s\n' "$SOURCE_REVISION"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'python_requested=%s\n' "$PYTHON_REQUESTED"
  printf 'SOLVER_LIB_PATH=%s\n' "$SOLVER_LIB_PATH"
  printf 'effective_LD_LIBRARY_PATH=%s\n' "$SOLVER_LIB_PATH"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH-<unset>}"
  printf 'PYTHONUSERBASE=%s\n' "${PYTHONUSERBASE-<unset>}"
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" - <<'PY'
import importlib
import sys

from paper_joint_solver.schedule_plan import solver_sources_sha256

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
for name in ("pyscipopt", "yices", "paper_joint_solver", "skc"):
    module = importlib.import_module(name)
    print(f"module_{name}_file={getattr(module, '__file__', '<unset>')}")
    print(f"module_{name}_version={getattr(module, '__version__', '<unset>')}")
print(f"solver_sources_sha256={solver_sources_sha256()}")
PY
} >"$OUT/run_emitter_cases_environment.log"

write_command() {
  local destination=$1
  shift
  {
    printf '%q ' "$@"
    printf '\n'
  } >"$destination"
}

run_case() {
  local stem=$1
  local ddg=$2
  local graph=$3
  local regs_per_warp=$4
  local warp_fixed_overhead=$5
  local solution="$OUT/$stem.json"
  local -a command=(
    taskset -c "$SOLVER_CPU"
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver "$ddg"
    --baseline-graph "$graph"
    -o "$solution"
    --emitter-compatible-lane-masks
    --normalization-u "$NORMALIZATION_U"
    --reg-budget "$regs_per_warp"
    --warp-fixed-overhead "$warp_fixed_overhead"
    --ilp-seconds 240
    --smt-seconds 300
    --max-wall-s 3600
  )
  write_command "$OUT/$stem.solve.command" "${command[@]}"
  echo "running $stem (non-paper emitter constraints)"
  if ! "${command[@]}" >"$OUT/$stem.solve.log" 2>&1; then
    cat "$OUT/$stem.solve.log" >&2
    return 1
  fi
}

run_case fwd_subtiled_emitter_v8 "$FWD_DDG" "$FWD_GRAPH" 8160 4
run_case bwd_emitter_v8 "$BWD_DDG" "$BWD_GRAPH" 8160 0
run_case bwd_lr4096_emitter_v8 "$BWD_DDG" "$BWD_GRAPH" 4096 0

env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT" >"$OUT/run_emitter_cases_validation.log" <<'PY'
import json
import sys
from pathlib import Path

from paper_joint_solver.schedule_plan import (
    EMITTER_MODEL_PROFILE,
    load_schedule_plan,
)

out = Path(sys.argv[1])
environment = {}
for line in (out / "run_emitter_cases_environment.log").read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        environment[key] = value
if environment.get("paper_comparable") != "yes":
    raise SystemExit("canonical v8 validation requires paper_comparable=yes")
cases = {
    "fwd_subtiled_emitter_v8": (
        Path("../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json"),
        Path("../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json"),
        8160,
        4,
        False,
    ),
    "bwd_emitter_v8": (
        Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        Path("../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"),
        8160,
        0,
        True,
    ),
    "bwd_lr4096_emitter_v8": (
        Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        Path("../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"),
        4096,
        0,
        True,
    ),
}
for stem, (ddg, graph, regs, fixed_warps, require_tmem) in cases.items():
    solution = out / f"{stem}.json"
    plan = load_schedule_plan(
        solution,
        ddg,
        graph,
        model_profile=EMITTER_MODEL_PROFILE,
    )
    payload = json.loads(solution.read_text())
    final = payload["attempts"][-1]
    if final.get("stage") != "joint" or final.get("result") != "sat":
        raise SystemExit(f"{stem}: missing final SAT joint attempt")
    if final.get("ii") != plan.ii or final.get("L") != plan.length:
        raise SystemExit(f"{stem}: final SAT attempt does not match solution")
    if plan.machine["regs_per_warp"] != regs:
        raise SystemExit(f"{stem}: register budget mismatch")
    if plan.machine["warp_fixed_overhead"] != fixed_warps:
        raise SystemExit(f"{stem}: fixed-warp budget mismatch")
    if plan.normalization_u != 300:
        raise SystemExit(f"{stem}: normalization budget mismatch")
    if require_tmem:
        stats = payload["stats"]
        if stats.get("tmem_object_count", 0) <= 0:
            raise SystemExit(f"{stem}: missing TMEM object evidence")
        if stats.get("peak_tmem_cols", 0) <= 0:
            raise SystemExit(f"{stem}: missing peak TMEM evidence")
    print(f"{stem}: validated emitter-constrained SAT II={plan.ii} L={plan.length}")
PY

echo "emitter-constrained v8 solutions validated in $OUT"
