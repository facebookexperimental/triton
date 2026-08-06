#!/bin/bash
# Ablation reruns (paper sec 6.2.2) on the sub-tiled forward fixture:
#   1. reduced physical warps  (--num-warps: baseline-1 and half)
#   2. no sub-tiling           (the non-subtiled fwd ddg as input)
#   3. no cross-warp traffic   (--no-cross-warp)
# Run from the package directory. Each solve gets the same search/time limits.
# The JSON status is authoritative. Neither unknown nor resource_limited is
# promoted to an UNSAT proof.
set -euo pipefail
PYTHON=${PYTHON:-../../../../.venv/bin/python}
SOLVER_CPU=${SOLVER_CPU:-0}
UNPINNED_HOST=${UNPINNED_HOST:-0}
if [[ ! -x "$PYTHON" ]]; then
  echo "PYTHON must name an executable with PySCIPOpt and Yices" >&2
  exit 1
fi
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
# SOLVER_LIB_PATH: colon-separated lib dirs holding yices/cudd shared objects
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to <yices>/lib:<cudd>/lib}"
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
SUB=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD=../sched2tlx/examples/case3_FA_fp16/ddg.json
SUB_GRAPH=../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json
FWD_GRAPH=../sched2tlx/examples/case3_FA_fp16/schedule_graph.json
DEFAULT_OUT=ablations_v8
OUT_WAS_EXPLICIT=${OUT+x}
OUT=${OUT:-$DEFAULT_OUT}
NORMALIZATION_U=300
FIXED_WARPS=4  # sched2tlx emits a four-warp default task outside the loop DDG
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing output directory: $OUT" >&2
  exit 1
fi

for input in "$SUB" "$FWD" "$SUB_GRAPH" "$FWD_GRAPH"; do
  if [[ ! -f "$input" ]]; then
    echo "required input does not exist: $input" >&2
    exit 1
  fi
done

PYTHON_EXECUTABLE=$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.executable))')
PYTHON_VERSION=$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" -c 'import sys; print(sys.version.replace("\n", " "))')

LSCPU_OUTPUT=$(LC_ALL=C lscpu)
LSCPU_MODEL=$(sed -n 's/^Model name:[[:space:]]*//p' <<<"$LSCPU_OUTPUT")
LSCPU_CPU_COUNT=$(sed -n 's/^CPU(s):[[:space:]]*//p' <<<"$LSCPU_OUTPUT")
MACHINE_HOSTNAME=$(hostname)
if [[ -z "$LSCPU_MODEL" || -z "$LSCPU_CPU_COUNT" ]]; then
  echo "lscpu did not report a model name and CPU count" >&2
  exit 1
fi
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
    echo "paper-comparable v8 solves must run on dgx003, not $MACHINE_HOSTNAME" >&2
    exit 1
  fi
  if [[ "$LSCPU_MODEL" != *8570* ]]; then
    echo "paper-comparable v8 solves require Xeon Platinum 8570: $LSCPU_MODEL" >&2
    exit 1
  fi
fi
# END HOST PINNING POLICY
mkdir -p "$OUT"
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
  printf 'working_directory=%s\n' "$(pwd -P)"
  printf 'paper_comparable=%s\n' "$PAPER_COMPARABLE"
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
  printf 'python_requested=%s\n' "$PYTHON"
  printf 'python_executable=%s\n' "$PYTHON_EXECUTABLE"
  printf 'python_version=%s\n' "$PYTHON_VERSION"
  printf 'SOLVER_LIB_PATH=%s\n' "$SOLVER_LIB_PATH"
  printf 'effective_LD_LIBRARY_PATH=%s\n' "$SOLVER_LIB_PATH"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH-<unset>}"
  printf 'PYTHONUSERBASE=%s\n' "${PYTHONUSERBASE-<unset>}"
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" - <<'PY'
import importlib

from paper_joint_solver.schedule_plan import solver_sources_sha256
from yices import Yices

for name in ("pyscipopt", "yices", "paper_joint_solver"):
    module = importlib.import_module(name)
    print(f"module_{name}_file={getattr(module, '__file__', '<unset>')}")
    print(f"module_{name}_version={getattr(module, '__version__', '<unset>')}")
print(f"libyices_version={Yices.version}")
print(f"solver_sources_sha256={solver_sources_sha256()}")
PY
} >"$OUT/environment.log"
: >"$OUT/commands.log"
DIAGNOSTIC_OUT="$OUT/diagnostics"
mkdir -p "$DIAGNOSTIC_OUT"
: >"$DIAGNOSTIC_OUT/commands.log"
printf 'variant\tcli_rc\tsolver_status\tminimum_joint_ii\tminimum_ii_attempts\tpremise_holds\n' \
  >"$DIAGNOSTIC_OUT/statuses.tsv"

run_case() {
  local name=$1
  local expected_rcs=$2
  shift 2
  local output="$OUT/$name.json"
  local log="$OUT/$name.log"
  local rc
  local -a command=(
    taskset -c "$SOLVER_CPU"
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver "$@" -o "$output"
    --normalization-u "$NORMALIZATION_U"
    --warp-fixed-overhead "$FIXED_WARPS"
    --ilp-seconds 240 --smt-seconds 300 --max-wall-s 3600
  )

  {
    printf 'case=%s\n' "$name"
    printf 'expected_rcs=%s\n' "$expected_rcs"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$log"
  {
    printf 'case=%s\n' "$name"
    printf 'expected_rcs=%s\n' "$expected_rcs"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n\n'
  } >>"$OUT/commands.log"

  if "${command[@]}" >>"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '\nactual_rc=%d\n' "$rc" >>"$log"
  printf 'case=%s actual_rc=%d\n' "$name" "$rc" >>"$OUT/commands.log"
  cat "$log"
  if [[ ",$expected_rcs," != *",$rc,"* ]]; then
    echo "$name returned $rc; expected one of $expected_rcs" >&2
    return 1
  fi
}

joint_premise_holds() {
  local result_path=$1
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" - "$result_path" <<'PY'
# BEGIN MINIMUM-II JOINT PREMISE
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        result = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid result JSON: {error}") from error
if not isinstance(result, dict):
    raise SystemExit("result JSON is not an object")
attempts = result.get("attempts")
if not isinstance(attempts, list):
    raise SystemExit("result JSON has no attempts list")
joint_attempts = [
    attempt
    for attempt in attempts
    if isinstance(attempt, dict) and attempt.get("stage") == "joint"
]
if not joint_attempts:
    print("no")
    raise SystemExit(0)
for attempt in joint_attempts:
    ii = attempt.get("ii")
    verdict = attempt.get("result")
    if isinstance(ii, bool) or not isinstance(ii, int):
        raise SystemExit("joint attempt has no integer ii")
    if not isinstance(verdict, str):
        raise SystemExit("joint attempt has no string result")
min_ii = min(attempt["ii"] for attempt in joint_attempts)
premise = any(
    attempt["ii"] == min_ii and attempt["result"] == "sat"
    for attempt in joint_attempts
)
print("yes" if premise else "no")
# END MINIMUM-II JOINT PREMISE
PY
}

run_diagnostic_case() {
  local name=$1
  shift
  local output="$DIAGNOSTIC_OUT/$name.json"
  local log="$DIAGNOSTIC_OUT/$name.log"
  local rc
  local premise
  local summary
  local -a command=(
    taskset -c "$SOLVER_CPU"
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver "$SUB"
    --baseline-graph "$SUB_GRAPH" -o "$output" "$@"
    --normalization-u "$NORMALIZATION_U"
    --warp-fixed-overhead "$FIXED_WARPS"
    --ilp-seconds 240 --smt-seconds 300 --max-wall-s 3600
  )

  {
    printf 'variant=%s\n' "$name"
    printf 'expected_rcs=0,2\n'
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"$log"
  {
    printf 'variant=%s\n' "$name"
    printf 'expected_rcs=0,2\n'
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n\n'
  } >>"$DIAGNOSTIC_OUT/commands.log"

  if "${command[@]}" >>"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  if [[ ",$rc," != *,0,* && ",$rc," != *,2,* ]]; then
    printf '\nactual_rc=%d\n' "$rc" >>"$log"
    cat "$log"
    echo "$name returned $rc; expected one of 0,2" >&2
    return 1
  fi
  premise=$(joint_premise_holds "$output")
  summary=$(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
      "$PYTHON" - "$output" "$premise" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        result = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid diagnostic JSON: {error}") from error
if not isinstance(result, dict):
    raise SystemExit("diagnostic JSON is not an object")
status = result.get("status")
if not isinstance(status, str):
    raise SystemExit("diagnostic JSON has no string status")
joint_attempts = [
    attempt
    for attempt in result.get("attempts", [])
    if isinstance(attempt, dict) and attempt.get("stage") == "joint"
]
if joint_attempts:
    min_ii = min(attempt["ii"] for attempt in joint_attempts)
    minimum_attempts = [
        attempt for attempt in joint_attempts if attempt["ii"] == min_ii
    ]
    attempt_text = "; ".join(
        f"(L={attempt.get('L', '-')}, {attempt.get('result', 'missing')})"
        for attempt in minimum_attempts
    )
else:
    min_ii = "-"
    attempt_text = "missing"
premise = sys.argv[2]
if premise not in {"yes", "no"}:
    raise SystemExit(f"invalid diagnostic premise: {premise}")
fields = (status, min_ii, attempt_text, premise)
print("\t".join(str(field) for field in fields))
PY
  )
  printf '\nactual_rc=%d\n%s\n' "$rc" "$summary" >>"$log"
  printf '%s\t%d\t%s\n' "$name" "$rc" "$summary" \
    >>"$DIAGNOSTIC_OUT/statuses.tsv"
  {
    printf 'variant=%s actual_rc=%d\n' "$name" "$rc"
    printf 'variant=%s %s\n' "$name" "$summary"
  } >>"$DIAGNOSTIC_OUT/commands.log"
  cat "$log"
}

write_diagnostic_report() {
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" - "$OUT/baseline.json" "$DIAGNOSTIC_OUT" \
    "$BASELINE_JOINT_PREMISE" <<'PY'
import csv
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
diagnostic_dir = Path(sys.argv[2])
baseline_premise = sys.argv[3]
if baseline_premise not in {"yes", "no"}:
    raise SystemExit(f"invalid baseline premise: {baseline_premise}")
with baseline_path.open() as handle:
    baseline = json.load(handle)
joint_attempts = [
    attempt
    for attempt in baseline["attempts"]
    if isinstance(attempt, dict) and attempt.get("stage") == "joint"
]
if joint_attempts:
    min_ii = min(attempt["ii"] for attempt in joint_attempts)
    minimum_attempts = [
        attempt
        for attempt in joint_attempts
        if attempt["ii"] == min_ii
    ]
    attempt_window = ", ".join(
        f"(L={attempt.get('L')}, {attempt.get('result')})"
        for attempt in minimum_attempts
    )
else:
    min_ii = "<none>"
    attempt_window = "<no joint-stage attempts>"
variants = {
    "no_spill_smem_footprint": (
        "`--no-spill-smem-footprint`",
        "omit spill staging from SMEM capacity",
    ),
    "no_reg_carried_same_lane": (
        "`--no-reg-carried-same-lane`",
        "omit same-lane constraints on register-carried edges",
    ),
    "ignore_fixed_overheads": (
        "`--ignore-fixed-overheads`",
        "do not subtract function-scope fixed resources",
    ),
    "paper_pure": (
        "`--paper-pure`",
        "apply all three diagnostic relaxations together",
    ),
}

with (diagnostic_dir / "statuses.tsv").open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
by_name = {row["variant"]: row for row in rows}
lines = [
    "# B2 Baseline-Premise Attribution",
    "",
    "This directory is diagnostic-only. Its JSON files are excluded from the "
    "top-level ablation evidence set and cannot satisfy a main evidence gate.",
    "",
    "## Baseline premise",
    "",
    (
        f"The minimum joint-stage II is `{min_ii}`. Its `(L, verdict)` attempts "
        f"are: {attempt_window}. Premise holds: `{baseline_premise}`."
    ),
]
if baseline_premise == "yes":
    lines.extend(
        [
            "The paper premise already holds, so no diagnostic relaxation was run.",
            "",
            "## Attribution",
            "",
            "No extension attribution is needed.",
        ]
    )
else:
    lines.extend(
        [
            f"No SAT attempt exists at minimum joint-stage II `{min_ii}`; "
            f"its `(L, verdict)` window is {attempt_window}.",
            "The paper premise does not hold, so each extension relaxation "
            "was rerun against the same DDG, baseline graph, normalization, "
            "CPU binding, and solver limits.",
            "",
            "## Relaxation matrix",
            "",
            "| Variant | Relaxation | CLI rc | Solver status | Minimum joint II | "
            "`(L, verdict)` attempts | Restores premise |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    restored = []
    for name, (flag, description) in variants.items():
        row = by_name.get(name)
        if row is None:
            raise SystemExit(f"missing diagnostic status for {name}")
        if (
            min_ii != "<none>"
            and row["minimum_joint_ii"] != str(min_ii)
        ):
            raise SystemExit(
                f"{name} minimum joint II {row['minimum_joint_ii']} does not "
                f"match baseline {min_ii}"
            )
        restores = row["premise_holds"] == "yes"
        if restores:
            restored.append(name)
        lines.append(
            f"| `{name}` | {flag}: {description} | {row['cli_rc']} | "
            f"`{row['solver_status']}` | `{row['minimum_joint_ii']}` | "
            f"`{row['minimum_ii_attempts']}` | "
            f"{'yes' if restores else 'no'} |"
        )
    lines.extend(["", "## Attribution", ""])
    individual = [name for name in restored if name != "paper_pure"]
    if individual:
        joined = ", ".join(f"`{name}`" for name in individual)
        lines.append(f"Individual relaxations restoring the premise: {joined}.")
    elif "paper_pure" in restored:
        lines.append(
            "Only the combined `paper_pure` relaxation restores the premise; "
            "the tested extensions interact, so no single extension is solely "
            "responsible."
        )
    else:
        lines.append(
            "None of the tested relaxations restores the premise. The cause is "
            "outside C1 spill SMEM, C2 register-carried same-lane, and fixed "
            "overheads; corrected accumulator TMEM liveness remains enabled."
        )
lines.extend(
    [
        "",
        "## Audit trail",
        "",
        "Full shell commands and exit codes are in `commands.log`; structured "
        "statuses are in `statuses.tsv`; each executed variant has a `.log` and "
        "a solver `.json` in this directory.",
        "",
    ]
)
(diagnostic_dir / "README.md").write_text("\n".join(lines))
PY
}

# Derive physical-warp ablations from a baseline produced by this exact model.
run_case baseline 0,2 "$SUB" --baseline-graph "$SUB_GRAPH"
BASELINE_JOINT_PREMISE=$(joint_premise_holds "$OUT/baseline.json")
if [[ "$BASELINE_JOINT_PREMISE" != yes ]]; then
  run_diagnostic_case no_spill_smem_footprint --no-spill-smem-footprint
  run_diagnostic_case no_reg_carried_same_lane --no-reg-carried-same-lane
  run_diagnostic_case ignore_fixed_overheads --ignore-fixed-overheads
  run_diagnostic_case paper_pure --paper-pure
fi
write_diagnostic_report
read -r WARP_MINUS_ONE WARP_HALF < <(
env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT/baseline.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as handle:
        result = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid baseline JSON: {error}") from error
if not isinstance(result, dict):
    raise SystemExit("baseline JSON is not an object")
if result.get("status") != "sat" or result.get("satisfiable") is not True:
    raise SystemExit("baseline did not produce a SAT solution")
stats = result.get("stats")
if not isinstance(stats, dict):
    raise SystemExit("baseline has no stats object")
values = {
    field: stats.get(field)
    for field in (
        "scheduled_warps",
        "fixed_warps",
        "physical_warp_budget",
    )
}
if any(
    isinstance(value, bool) or not isinstance(value, int)
    for value in values.values()
):
    raise SystemExit("baseline warp statistics must be integers")
used_warps = values["scheduled_warps"] + values["fixed_warps"]
physical_budget = values["physical_warp_budget"]
# Prefer the paper's full-machine 31/16 cuts when the baseline is binding.
# Otherwise cut from actual use so both ablations necessarily remove capacity.
budget_basis = physical_budget if used_warps >= physical_budget - 1 else used_warps
minus_one = budget_basis - 1
half = budget_basis // 2
if half <= values["fixed_warps"] or minus_one <= values["fixed_warps"]:
    raise SystemExit("derived warp budget leaves no scheduled warp")
print(minus_one, half)
PY
)
printf 'warps_minus_one=%s\nwarps_half=%s\n' \
  "$WARP_MINUS_ONE" "$WARP_HALF" >"$OUT/warp_budgets.log"
run_case warps_minus_one 0,2 "$SUB" --baseline-graph "$SUB_GRAPH" \
  --num-warps "$WARP_MINUS_ONE"
run_case warps_half 0,2 "$SUB" --baseline-graph "$SUB_GRAPH" \
  --num-warps "$WARP_HALF"
run_case no_subtiling 0,2 "$FWD" --baseline-graph "$FWD_GRAPH"
run_case no_cross_warp 2 "$SUB" --baseline-graph "$SUB_GRAPH" \
  --no-cross-warp --no-cross-warp-domain reg-data \
  --no-structural-precheck --max-ii-span 16

if ! env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT" "$BASELINE_JOINT_PREMISE" >"$OUT/validation.log" 2>&1 <<'PY'
import json
import hashlib
import os
import re
import sys

from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import MAX_U_MULTIPLIER
from paper_joint_solver.schedule_plan import solver_sources_sha256

out = sys.argv[1]
environment = {}
with open(os.path.join(out, "environment.log")) as handle:
    for line in handle:
        key, separator, value = line.rstrip("\n").partition("=")
        if separator:
            environment[key] = value
if environment.get("paper_comparable") != "yes":
    raise SystemExit("canonical v8 validation requires paper_comparable=yes")
# This value is produced once by joint_premise_holds; do not redefine it here.
baseline_joint_premise = sys.argv[2]
if baseline_joint_premise != "yes":
    raise SystemExit(
        "baseline has no SAT attempt at its minimum joint-stage II; "
        "see diagnostics/README.md for conditional B2 attribution"
    )
expected = {
    "baseline": {"sat"},
    "warps_minus_one": {"sat", "unsat", "resource_limited"},
    "warps_half": {"sat", "unsat", "resource_limited"},
    "no_subtiling": {"sat", "unsat", "resource_limited"},
    "no_cross_warp": {"unsat", "resource_limited"},
}
source_hashes = set()
results = {}
actual_json = {name for name in os.listdir(out) if name.endswith(".json")}
expected_json = {f"{name}.json" for name in expected}
if actual_json != expected_json:
    raise SystemExit(
        "unexpected ablation JSON set: "
        f"missing={sorted(expected_json - actual_json)}, "
        f"extra={sorted(actual_json - expected_json)}"
    )
for name, expected_statuses in expected.items():
    path = os.path.join(out, f"{name}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing required result: {path}")
    try:
        with open(path) as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"malformed result {path}: {error}") from error
    if not isinstance(result, dict):
        raise SystemExit(f"result is not a JSON object: {path}")
    status = result.get("status")
    satisfiable = result.get("satisfiable")
    if status not in expected_statuses or satisfiable is not (status == "sat"):
        raise SystemExit(
            f"unexpected result for {name}: status={status!r}, "
            f"satisfiable={satisfiable!r}"
        )
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"{path} has no provenance object")
    source_hash = provenance.get("solver_sources_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise SystemExit(f"invalid solver_sources_sha256 in {path}")
    source_hashes.add(source_hash)
    results[name] = result
    print(f"{path}: status={status} satisfiable={satisfiable}")

current_source_hash = solver_sources_sha256()
if source_hashes != {current_source_hash}:
    raise SystemExit(
        "ablation solver_sources_sha256 does not match the current toolchain: "
        + ", ".join(sorted(source_hashes))
    )
baseline_stats = results["baseline"].get("stats")
if not isinstance(baseline_stats, dict):
    raise SystemExit("baseline has no stats object")


def integer_stat(name):
    value = baseline_stats.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"baseline stats.{name} is not an integer")
    return value


scheduled_warps = integer_stat("scheduled_warps")
fixed_warps = integer_stat("fixed_warps")
physical_budget = integer_stat("physical_warp_budget")
used_warps = scheduled_warps + fixed_warps
budget_basis = physical_budget if used_warps >= physical_budget - 1 else used_warps
warp_minus_one = budget_basis - 1
warp_half = budget_basis // 2

sub_ddg = "../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json"
sub_graph = "../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json"
fwd_ddg = "../sched2tlx/examples/case3_FA_fp16/ddg.json"
fwd_graph = "../sched2tlx/examples/case3_FA_fp16/schedule_graph.json"
default_warps = MachineModel().num_warps
case_inputs = {
    "baseline": (sub_ddg, sub_graph, default_warps),
    "warps_minus_one": (sub_ddg, sub_graph, warp_minus_one),
    "warps_half": (sub_ddg, sub_graph, warp_half),
    "no_subtiling": (fwd_ddg, fwd_graph, default_warps),
    "no_cross_warp": (sub_ddg, sub_graph, default_warps),
}
default_model_options = {
    "allow_cross_warp": True,
    "no_cross_warp_domain": "reg-data",
    "prefix_lane_masks": False,
    "full_group_lane_masks": False,
    "spill_smem_footprint": True,
    "reg_carried_same_lane": True,
    "ignore_fixed_overheads": False,
    "minimize_groups": False,
    "max_groups": None,
    "num_warps_override": None,
}
for name, (ddg, graph, warp_budget) in case_inputs.items():
    provenance = results[name]["provenance"]
    for field, source in (
        ("ddg_sha256", ddg),
        ("baseline_graph_sha256", graph),
    ):
        with open(source, "rb") as handle:
            expected_hash = hashlib.sha256(handle.read()).hexdigest()
        if provenance.get(field) != expected_hash:
            raise SystemExit(f"{name} {field} does not match {source}")
    if provenance.get("normalization_u") != 300:
        raise SystemExit(f"{name} normalization_u is not 300")
    normalization_u_effective = provenance.get("normalization_u_effective")
    if (
        isinstance(normalization_u_effective, bool)
        or not isinstance(normalization_u_effective, int)
        or normalization_u_effective < 300
    ):
        raise SystemExit(f"{name} has invalid normalization_u_effective")
    if normalization_u_effective >= 300 * MAX_U_MULTIPLIER:
        raise SystemExit(
            f"{name} normalization_u_effective has no retry headroom"
        )
    machine = provenance.get("machine")
    if not isinstance(machine, dict):
        raise SystemExit(f"{name} has no machine provenance")
    if machine.get("num_warps") != warp_budget:
        raise SystemExit(f"{name} physical warp budget mismatch")
    if machine.get("warp_fixed_overhead") != 4:
        raise SystemExit(f"{name} fixed warp overhead mismatch")
    expected_model_options = dict(default_model_options)
    if name == "warps_minus_one":
        expected_model_options["num_warps_override"] = warp_minus_one
    elif name == "warps_half":
        expected_model_options["num_warps_override"] = warp_half
    elif name == "no_cross_warp":
        expected_model_options["allow_cross_warp"] = False
    if provenance.get("model_options") != expected_model_options:
        raise SystemExit(
            f"{name} model_options mismatch: "
            f"{provenance.get('model_options')!r}"
        )

def joint_attempts(name):
    attempts = results[name].get("attempts")
    if not isinstance(attempts, list):
        raise SystemExit(f"{name} has no attempts list")
    if any(attempt.get("stage") == "structural" for attempt in attempts):
        raise SystemExit(f"{name} used a structural UNSAT shortcut")
    return [attempt for attempt in attempts if attempt.get("stage") == "joint"]


baseline_point = (results["baseline"].get("ii"), results["baseline"].get("length"))
if not all(isinstance(value, int) for value in baseline_point):
    raise SystemExit("baseline has no integer SAT point")
baseline_joint = joint_attempts("baseline")
if not baseline_joint or min(attempt.get("ii") for attempt in baseline_joint) != baseline_point[0]:
    raise SystemExit(
        "baseline SAT point is not at the minimum joint-stage II; see "
        "diagnostics/README.md for conditional B2 attribution"
    )
if not any(
    (attempt.get("ii"), attempt.get("L")) == baseline_point
    and attempt.get("result") == "sat"
    for attempt in baseline_joint
):
    raise SystemExit("baseline attempts do not contain its SAT point")

for name in ("warps_minus_one", "warps_half"):
    attempts = joint_attempts(name)
    if not any(
        (attempt.get("ii"), attempt.get("L")) == baseline_point
        and attempt.get("result") == "unsat"
        for attempt in attempts
    ):
        raise SystemExit(f"{name} is not joint-UNSAT at the baseline SAT point")
    if results[name].get("status") == "sat":
        terminal_point = (results[name].get("ii"), results[name].get("length"))
    else:
        terminal_point = (attempts[-1].get("ii"), attempts[-1].get("L"))
    if not all(isinstance(value, int) for value in terminal_point):
        raise SystemExit(f"{name} has no terminal search point")
    if terminal_point <= baseline_point:
        raise SystemExit(f"{name} did not force search beyond the baseline point")

for name in ("no_subtiling", "no_cross_warp"):
    attempts = joint_attempts(name)
    if not attempts or attempts[0].get("result") != "unsat":
        raise SystemExit(f"{name} lacks joint-UNSAT at its first ZLP point")

print(f"b2_attribution_report={out}/diagnostics/README.md")
print(f"solver_sources_sha256={current_source_hash}")
PY
then
  cat "$OUT/validation.log" >&2
  exit 1
fi
cat "$OUT/validation.log"
