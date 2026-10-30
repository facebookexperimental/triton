#!/bin/bash
# Ablation reruns (paper sec 6.2.2) on the sub-tiled forward fixture:
#   1. reduced physical warps  (--num-warps-override: 31 and 16)
#   2. no sub-tiling           (the non-subtiled fwd ddg as input)
#   3. no cross-warp traffic   (--no-cross-warp)
# Run from the package directory.
#
# Ablation execution protocol (disclosed, not solver logic): the baseline runs
# the literal Algorithm 1 search; each modified configuration is probed by
# python -m paper_joint_solver.probe at the smallest ZLP-feasible I over its
# full L window, reporting per-point verdict counts. This realizes the paper's
# sec 6.2.2 claim ('unsatisfiable for the smallest I which makes the ZLP
# succeed, forcing a search over larger values of I and L') without asserting
# any terminal outcome for searches the literal algorithm never finishes.
set -euo pipefail
PYTHON=${PYTHON:-../../../../.venv/bin/python}
SOLVER_CPU=${SOLVER_CPU:-0}
WATCHDOG_S=${WATCHDOG_S:-86400}   # execution-protocol parameter, not a solver input
UNPINNED_HOST=${UNPINNED_HOST:-0}
PAPER_HOST=${PAPER_HOST:-dgx003}
PAPER_CPU_MODEL=${PAPER_CPU_MODEL:-8570}
if [[ ! -x "$PYTHON" ]]; then
  echo "PYTHON must name an executable with PySCIPOpt and Yices" >&2
  exit 1
fi
if [[ ! "$SOLVER_CPU" =~ ^[0-9]+$ ]]; then
  echo "SOLVER_CPU must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$WATCHDOG_S" =~ ^[0-9]+$ ]]; then
  echo "WATCHDOG_S must be a non-negative integer" >&2
  exit 1
fi
for required_command in hostname lscpu nproc taskset timeout; do
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
# SOLVER_LIB_PATH: colon-separated lib dirs holding libyices and every library
# libyices itself links against (the paper host's build needs the BDD one too).
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to colon-separated dirs holding libyices and every library libyices itself links against (paper host: <yices>/lib:<cudd>/lib)}"
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

# Recorded provenance must not pin this host's buck-out artifact paths: the
# hierarchy is not stable, and the open-source export gate rejects it. The
# stable entries are kept; the buck artifact is named rather than located.
SOLVER_LIB_PATH_RECORDED=
for directory in "${solver_lib_dirs[@]}"; do
  case $directory in
    */buck-out/*) directory='<yices2-artifact-dir>' ;;
  esac
  SOLVER_LIB_PATH_RECORDED+="${SOLVER_LIB_PATH_RECORDED:+:}$directory"
done
SUB=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD=../sched2tlx/examples/case3_FA_fp16/ddg.json
SUB_GRAPH=../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json
FWD_GRAPH=../sched2tlx/examples/case3_FA_fp16/schedule_graph.json
DEFAULT_OUT=ablations_lit1
OUT_WAS_EXPLICIT=${OUT+x}
OUT=${OUT:-$DEFAULT_OUT}
NORMALIZATION_U=300
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
# sys.version carries the toolchain's own source URL on fbcode builds, which
# the open-source export gate rejects. Keep the compiler name and version.
PYTHON_VERSION=$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" -c 'import re, sys
version = re.sub(r"\s*\([a-z][a-z0-9+.-]*://.*", "", sys.version.replace("\n", " ")).rstrip()
if version.count("[") > version.count("]"):
    version += "]"
print(version)')

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
  if [[ "$MACHINE_HOSTNAME" != "$PAPER_HOST" && \
        "$MACHINE_HOSTNAME" != "$PAPER_HOST".* ]]; then
    echo "paper-comparable lit1 solves must run on the pinned host" \
      "$PAPER_HOST, not $MACHINE_HOSTNAME" >&2
    exit 1
  fi
  if [[ "$LSCPU_MODEL" != *"$PAPER_CPU_MODEL"* ]]; then
    echo "paper-comparable lit1 solves require the pinned CPU model" \
      "$PAPER_CPU_MODEL: $LSCPU_MODEL" >&2
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
  printf 'pinned_host=%s\n' "$PAPER_HOST"
  printf 'pinned_cpu_model=%s\n' "$PAPER_CPU_MODEL"
  printf 'lscpu_model_name=%s\n' "$LSCPU_MODEL"
  printf 'lscpu_cpu_count=%s\n' "$LSCPU_CPU_COUNT"
  printf 'nproc=%s\n' "$(nproc)"
  printf 'solver_cpu=%s\n' "$SOLVER_CPU"
  printf 'watchdog_s=%s\n' "$WATCHDOG_S"
  printf 'cuda_version=%s\n' "$CUDA_VERSION"
  printf 'nvidia_smi=%s\n' "$NVIDIA_SMI_INFO"
  printf 'source_vcs=%s\n' "$SOURCE_VCS"
  printf 'source_revision=%s\n' "$SOURCE_REVISION"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'python_requested=%s\n' "$PYTHON"
  printf 'python_executable=%s\n' "$PYTHON_EXECUTABLE"
  printf 'python_version=%s\n' "$PYTHON_VERSION"
  printf 'SOLVER_LIB_PATH=%s\n' "$SOLVER_LIB_PATH_RECORDED"
  printf 'effective_LD_LIBRARY_PATH=%s\n' "$SOLVER_LIB_PATH_RECORDED"
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
OBS_LOG="$OUT/observations.log"
: >"$OBS_LOG"

write_command() {
  local destination=$1
  shift
  local rendered
  rendered=$(printf '%q ' "$@")
  # SOLVER_LIB_PATH is an input of this script, not a constant of the host, so
  # the record names the variable instead of pinning its buck-out expansion.
  rendered=${rendered//$(printf '%q' "$SOLVER_LIB_PATH")/\"\$SOLVER_LIB_PATH\"}
  rendered=${rendered//"$SOLVER_LIB_PATH"/\"\$SOLVER_LIB_PATH\"}
  printf '%s\n' "$rendered" >"$destination"
  printf 'command=%s\n' "$rendered" >>"$OUT/commands.log"
}

# BEGIN WATCHDOG RC CONTRACT
# rc=0 is the only normal terminal of the solve and probe verbs. 124/137 is
# a harness observation ("did not terminate"), never a verdict. Any other rc
# (including argparse usage errors, rc 2) is an infrastructure failure and
# aborts the batch. No rc is ever mapped to sat/unsat.
run_solver_watchdogged() {
  # $1 observation label, $2 command log, $3 output log; rest: command
  # (starting at `env ...`; taskset+timeout are prepended here)
  local label=$1 command_log=$2 output_log=$3; shift 3
  local -a command=(
    taskset -c "$SOLVER_CPU"
    timeout --kill-after=60 "$WATCHDOG_S"
    "$@"
  )
  write_command "$command_log" "${command[@]}"
  local rc=0
  "${command[@]}" >"$output_log" 2>&1 || rc=$?
  case "$rc" in
    0)
      printf 'observation label=%s outcome=completed rc=0\n' \
        "$label" >>"$OBS_LOG"
      printf 'completed\n' ;;
    124|137)
      printf 'observation label=%s outcome=did-not-terminate-within-%ss rc=%d\n' \
        "$label" "$WATCHDOG_S" "$rc" >>"$OBS_LOG"
      printf 'did-not-terminate\n' ;;
    *)
      printf 'observation label=%s outcome=infrastructure-error rc=%d\n' \
        "$label" "$rc" >>"$OBS_LOG"
      cat "$output_log" >&2
      return 1 ;;
  esac
}
# END WATCHDOG RC CONTRACT

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

# Curation is the paper path's own input stage: raw ddg -> curated G.
run_curate() { # name raw_ddg graph
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" -m paper_joint_solver.curate_ddg "$2" \
    --baseline-graph "$3" -o "$OUT/$1.curated_ddg.json" \
    --manifest-out "$OUT/$1.curation_manifest.json" \
    >"$OUT/$1.curate.log" 2>&1
}
run_curate sub "$SUB" "$SUB_GRAPH"
run_curate fwd "$FWD" "$FWD_GRAPH"
SUB_CURATED="$OUT/sub.curated_ddg.json"
FWD_CURATED="$OUT/fwd.curated_ddg.json"

run_solve_case() { # name curated [experiment args...]
  local name=$1 curated=$2
  shift 2
  run_solver_watchdogged "$name.solve" \
    "$OUT/$name.solve.command" "$OUT/$name.log" \
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" -m paper_joint_solver "$curated" -o "$OUT/$name.json" \
    --normalization-u "$NORMALIZATION_U" "$@" >/dev/null
}

run_probe_case() { # name curated [experiment args...]
  local name=$1 curated=$2
  shift 2
  run_solver_watchdogged "$name.probe" \
    "$OUT/$name.probe.command" "$OUT/$name.probe.log" \
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" -m paper_joint_solver.probe "$curated" \
    -o "$OUT/$name.probe.json" \
    --normalization-u "$NORMALIZATION_U" "$@" >/dev/null
}

# Literal warp-count cuts, disclosed protocol choice: the paper says only
# "Reducing the number of warps" (sec 6.2.2); we cut from the FULL machine
# budget W=32 (machine.MAX_PHYSICAL_WARPS): W-1 and W/2.
WARP_MINUS_ONE=31
WARP_HALF=16
printf 'warps_minus_one=%s\nwarps_half=%s\nbasis=full-machine-32\n' \
  "$WARP_MINUS_ONE" "$WARP_HALF" >"$OUT/warp_budgets.log"

run_solve_case baseline "$SUB_CURATED"
BASELINE_JOINT_PREMISE=$(joint_premise_holds "$OUT/baseline.json")
printf 'observation label=baseline.joint_premise expected=yes observed=%s\n' \
  "$BASELINE_JOINT_PREMISE" >>"$OBS_LOG"
run_probe_case warps_minus_one "$SUB_CURATED" \
  --num-warps-override "$WARP_MINUS_ONE"
run_probe_case warps_half "$SUB_CURATED" --num-warps-override "$WARP_HALF"
run_probe_case no_subtiling "$FWD_CURATED"
run_probe_case no_cross_warp "$SUB_CURATED" --no-cross-warp

if ! env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT" "$BASELINE_JOINT_PREMISE" >"$OUT/validation.log" 2>&1 <<'PY'
import json
import hashlib
import os
import re
import sys

from paper_joint_solver.probe import probe_sources_sha256
from paper_joint_solver.schedule_plan import solver_sources_sha256

current_probe_hash = probe_sources_sha256()
out = sys.argv[1]
environment = {}
with open(os.path.join(out, "environment.log")) as handle:
    for line in handle:
        key, separator, value = line.rstrip("\n").partition("=")
        if separator:
            environment[key] = value
if environment.get("paper_comparable") != "yes":
    raise SystemExit("canonical lit1 validation requires paper_comparable=yes")
# This value is produced once by joint_premise_holds; do not redefine it here.
baseline_joint_premise = sys.argv[2]
report = [
    "# lit1 ablation observations",
    "",
    f"OBSERVATION baseline.joint_premise expected=yes "
    f"observed={baseline_joint_premise}",
    "",
]
print(report[2])

# The expected column is the paper's claim; observed is read from the probe
# JSON, never from a process exit code.
expected = {
    "baseline": "sat",
    "warps_minus_one": "first-window-unsat",
    "warps_half": "first-window-unsat",
    "no_subtiling": "first-window-unsat",
    "no_cross_warp": "first-window-unsat",
}
probes = [name for name in expected if name != "baseline"]

# The observation log decides which cases have artifacts to check at all. Its
# vocabulary is completed / did-not-terminate; no rc, no verdict. A case the
# watchdog stopped is allowed to be missing its JSON -- an observation line
# takes its place below.
observations = {}
with open(os.path.join(out, "observations.log")) as handle:
    for line in handle:
        fields = dict(
            field.split("=", 1) for field in line.split() if "=" in field
        )
        label = fields.get("label")
        if label is not None:
            observations[label] = fields.get("outcome", "")


def completed(name):
    label = f"{name}.solve" if name == "baseline" else f"{name}.probe"
    return observations.get(label) == "completed"


expected_files = (
    ({"baseline.json"} if completed("baseline") else set())
    | {f"{name}.probe.json" for name in probes if completed(name)}
    | {f"{name}.curated_ddg.json" for name in ("sub", "fwd")}
    | {f"{name}.curation_manifest.json" for name in ("sub", "fwd")}
)
actual_json = {name for name in os.listdir(out) if name.endswith(".json")}
if actual_json != expected_files:
    raise SystemExit(
        "unexpected ablation JSON set: "
        f"missing={sorted(expected_files - actual_json)}, "
        f"extra={sorted(actual_json - expected_files)}"
    )

source_hashes = set()
results = {}


def load(path):
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"malformed result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"result is not a JSON object: {path}")
    return payload


def collect_provenance(name, payload, path):
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"{path} has no provenance object")
    if provenance.get("model") != "paper":
        raise SystemExit(
            f"{name} was not produced by the paper model: "
            f"{provenance.get('model')!r}"
        )
    source_hash = provenance.get("solver_sources_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise SystemExit(f"invalid solver_sources_sha256 in {path}")
    source_hashes.add(source_hash)
    return provenance


baseline_path = os.path.join(out, "baseline.json")
if completed("baseline"):
    results["baseline"] = load(baseline_path)
    # Mirror of the rc contract: the literal search can only terminate sat.
    if (
        results["baseline"].get("status") != "sat"
        or results["baseline"].get("satisfiable") is not True
    ):
        raise SystemExit("baseline is not a SAT solution")
    collect_provenance("baseline", results["baseline"], baseline_path)

for name in probes:
    if not completed(name):
        continue
    path = os.path.join(out, f"{name}.probe.json")
    payload = load(path)
    # Provenance of the execution protocol itself, frozen spelling.
    if payload.get("execution_mode") != "first-window-probe":
        raise SystemExit(
            f"{path} is not a first-window probe: "
            f"{payload.get('execution_mode')!r}"
        )
    for field in ("ii", "window", "window_size", "unsat_count", "sat_found"):
        if field not in payload:
            raise SystemExit(f"{path} has no {field}")
    provenance = collect_provenance(name, payload, path)
    # Two hashes, two scopes: the shared solver core this depended on, and the
    # probe module that chose the points and summarized the verdicts.
    probe_hash = provenance.get("probe_sources_sha256")
    if not isinstance(probe_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", probe_hash
    ):
        raise SystemExit(f"invalid probe_sources_sha256 in {path}")
    if probe_hash != current_probe_hash:
        raise SystemExit(
            f"{name} probe_sources_sha256 does not match the current probe "
            f"module: {probe_hash}"
        )
    results[name] = payload

current_source_hash = solver_sources_sha256()
if source_hashes and source_hashes != {current_source_hash}:
    raise SystemExit(
        "ablation solver_sources_sha256 does not match the current toolchain: "
        + ", ".join(sorted(source_hashes))
    )

warp_minus_one, warp_half = 31, 16
sub_ddg = "../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json"
sub_graph = "../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json"
fwd_ddg = "../sched2tlx/examples/case3_FA_fp16/ddg.json"
fwd_graph = "../sched2tlx/examples/case3_FA_fp16/schedule_graph.json"
default_warps = 32
case_inputs = {
    "baseline": ("sub", sub_ddg, sub_graph, default_warps),
    "warps_minus_one": ("sub", sub_ddg, sub_graph, warp_minus_one),
    "warps_half": ("sub", sub_ddg, sub_graph, warp_half),
    "no_subtiling": ("fwd", fwd_ddg, fwd_graph, default_warps),
    "no_cross_warp": ("sub", sub_ddg, sub_graph, default_warps),
}
for name, (curated_stem, ddg, graph, warp_budget) in case_inputs.items():
    if name not in results:
        continue
    provenance = results[name]["provenance"]
    curated_path = os.path.join(out, f"{curated_stem}.curated_ddg.json")
    with open(curated_path, "rb") as handle:
        curated_hash = hashlib.sha256(handle.read()).hexdigest()
    if provenance.get("ddg_sha256") != curated_hash:
        raise SystemExit(f"{name} ddg_sha256 does not match {curated_path}")
    curation_source = provenance.get("curation_source")
    if not isinstance(curation_source, dict):
        raise SystemExit(f"{name} has no curation_source object")
    for field, source in (
        ("ddg_sha256", ddg),
        ("baseline_graph_sha256", graph),
    ):
        with open(source, "rb") as handle:
            expected_hash = hashlib.sha256(handle.read()).hexdigest()
        if curation_source.get(field) != expected_hash:
            raise SystemExit(
                f"{name} curation_source.{field} does not match {source}"
            )
    manifest_path = os.path.join(
        out, f"{curated_stem}.curation_manifest.json"
    )
    manifest_source = load(manifest_path).get("curation_source")
    if not isinstance(manifest_source, dict):
        raise SystemExit(f"{manifest_path} has no curation_source object")
    for field in (
        "ddg_sha256",
        "baseline_graph_sha256",
        "curator_sources_sha256",
    ):
        if curation_source.get(field) != manifest_source.get(field):
            raise SystemExit(
                f"{name} curation_source.{field} disagrees with {manifest_path}"
            )
    if provenance.get("normalization_u") != 300:
        raise SystemExit(f"{name} normalization_u is not 300")
    machine = provenance.get("machine")
    if not isinstance(machine, dict):
        raise SystemExit(f"{name} has no machine provenance")
    if machine.get("num_warps") != warp_budget:
        raise SystemExit(f"{name} physical warp budget mismatch")
    if machine.get("warp_fixed_overhead") != 0:
        raise SystemExit(
            f"{name} warp_fixed_overhead is not 0: "
            f"{machine.get('warp_fixed_overhead')!r}"
        )
    inputs = provenance.get("experiment_inputs")
    if not isinstance(inputs, dict):
        raise SystemExit(f"{name} has no experiment_inputs object")
    line = (
        f"OBSERVATION {name} experiment_input "
        f"allow_cross_warp={inputs.get('allow_cross_warp')} "
        f"num_warps_override={inputs.get('num_warps_override')} "
        f"reg_budget={inputs.get('reg_budget')}"
    )
    print(line)
    report.append(line)


def joint_attempts(name):
    attempts = results[name].get("attempts")
    if not isinstance(attempts, list):
        raise SystemExit(f"{name} has no attempts list")
    return [attempt for attempt in attempts if attempt.get("stage") == "joint"]


report.extend(
    [
        "",
        "| variant | expected | observed |",
        "| --- | --- | --- |",
    ]
)
if "baseline" in results:
    baseline_point = (
        results["baseline"].get("ii"),
        results["baseline"].get("length"),
    )
    if not all(isinstance(value, int) for value in baseline_point):
        raise SystemExit("baseline has no integer SAT point")
    baseline_joint = joint_attempts("baseline")
    if not baseline_joint:
        raise SystemExit("baseline has no joint-stage attempts")
    minimum_joint_ii = min(attempt.get("ii") for attempt in baseline_joint)
    baseline_observed = (
        f"sat at (I={baseline_point[0]}, L={baseline_point[1]}); "
        f"minimum joint-stage II={minimum_joint_ii}"
    )
else:
    baseline_observed = "did-not-terminate"
print(f"OBSERVATION baseline expected=sat observed={baseline_observed}")
report.append(f"| baseline | {expected['baseline']} | {baseline_observed} |")

for name in probes:
    if name not in results:
        observed = "did-not-terminate"
        print(f"OBSERVATION {name} expected={expected[name]} observed={observed}")
        report.append(f"| {name} | {expected[name]} | {observed} |")
        continue
    payload = results[name]
    unsat_count = payload.get("unsat_count")
    window_size = payload.get("window_size")
    if payload.get("sat_found"):
        observed = (
            f"first-window-sat (unsat {unsat_count}/{window_size}) "
            f"at I={payload.get('ii')}"
        )
    else:
        observed = (
            f"first-window-unsat ({unsat_count}/{window_size}) "
            f"at I={payload.get('ii')}"
        )
    print(f"OBSERVATION {name} expected={expected[name]} observed={observed}")
    report.append(f"| {name} | {expected[name]} | {observed} |")

report.append("")
with open(os.path.join(out, "ablation_report.md"), "w") as handle:
    handle.write("\n".join(report))
print(f"ablation_report={out}/ablation_report.md")
print(f"solver_sources_sha256={current_source_hash}")
PY
then
  cat "$OUT/validation.log" >&2
  exit 1
fi
cat "$OUT/validation.log"
