#!/bin/bash
# Reproduce the four literal Phase-3 solves and their manual-CUDA handoff
# scaffolds. Run this script from third_party/tlx/tools/paper_joint_solver.
# Outputs are append-never: remove stale targets or choose a fresh OUT
# directory.
#
# Each case is curated first (raw ddg -> curated G) and then solved by the
# literal single-model CLI. Time bounds live in the harness watchdog, never in
# the solver; no exit code is ever read as a verdict.
set -euo pipefail

PYTHON_REQUESTED=${PYTHON:-../../../../.venv/bin/python}
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to colon-separated dirs holding libyices and every library libyices itself links against (paper host: <yices>/lib:<cudd>/lib)}"
SOLVER_CPU=${SOLVER_CPU:-0}
WATCHDOG_S=${WATCHDOG_S:-86400}   # execution-protocol parameter, not a solver input
DEFAULT_OUT=solutions
OUT_WAS_EXPLICIT=${OUT+x}
OUT=${OUT:-$DEFAULT_OUT}
UNPINNED_HOST=${UNPINNED_HOST:-0}
PAPER_HOST=${PAPER_HOST:-dgx003}
PAPER_CPU_MODEL=${PAPER_CPU_MODEL:-8570}
NORMALIZATION_U=300
FA4_TEMPLATE=subtiled_fa4exact_solution.json
VER=lit1

if [[ ! -f paper_joint_solver/__main__.py || ! -f skc/__main__.py ]]; then
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

FWD_SUB_DDG=../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
FWD_SUB_GRAPH=../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json
FWD_DDG=../sched2tlx/examples/case3_FA_fp16/ddg.json
FWD_GRAPH=../sched2tlx/examples/case3_FA_fp16/schedule_graph.json
BWD_DDG=../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json
BWD_GRAPH=../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json

for input in \
  "$FA4_TEMPLATE" \
  "$FWD_SUB_DDG" "$FWD_SUB_GRAPH" \
  "$FWD_DDG" "$FWD_GRAPH" \
  "$BWD_DDG" "$BWD_GRAPH"; do
  if [[ ! -f "$input" ]]; then
    echo "required input does not exist: $input" >&2
    exit 1
  fi
done

if [[ -e "$OUT" && ! -d "$OUT" ]]; then
  echo "OUT exists but is not a directory: $OUT" >&2
  exit 1
fi

stems=(fwd_subtiled_${VER} fwd_${VER} bwd_${VER} bwd_lr4096_${VER})
reserved=(
  "$OUT/run_main_cases_environment_${VER}.log"
  "$OUT/run_main_cases_validation_${VER}.log"
  "$OUT/run_main_cases_observations_${VER}.log"
  "$OUT/refit_${VER}.json"
  "$OUT/refit_${VER}.command"
  "$OUT/refit_${VER}.log"
)
for stem in "${stems[@]}"; do
  reserved+=(
    "$OUT/$stem.json"
    "$OUT/${stem}_ir.json"
    "$OUT/${stem}_handoff.json"
    "$OUT/${stem}_manual"
    "$OUT/$stem.curated_ddg.json"
    "$OUT/$stem.curation_manifest.json"
    "$OUT/$stem.curate.command"
    "$OUT/$stem.curate.log"
    "$OUT/$stem.solve.command"
    "$OUT/$stem.solve.log"
    "$OUT/$stem.strategy.json"
    "$OUT/$stem.strategy.command"
    "$OUT/$stem.strategy.log"
    "$OUT/$stem.scaffold.command"
    "$OUT/$stem.scaffold.log"
    "$OUT/$stem.audit.command"
    "$OUT/$stem.audit.log"
  )
done
for target in "${reserved[@]}"; do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing target: $target" >&2
    exit 1
  fi
done

ENVIRONMENT_LOG="$OUT/run_main_cases_environment_${VER}.log"
VALIDATION_LOG="$OUT/run_main_cases_validation_${VER}.log"
OBS_LOG="$OUT/run_main_cases_observations_${VER}.log"

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
: >"$OBS_LOG"
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
  printf 'python_requested=%s\n' "$PYTHON_REQUESTED"
  printf 'SOLVER_LIB_PATH=%s\n' "$SOLVER_LIB_PATH_RECORDED"
  printf 'effective_LD_LIBRARY_PATH=%s\n' "$SOLVER_LIB_PATH_RECORDED"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH-<unset>}"
  printf 'PYTHONUSERBASE=%s\n' "${PYTHONUSERBASE-<unset>}"
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" - <<'PY'
import importlib
import re
import sys

from paper_joint_solver.schedule_plan import solver_sources_sha256

# sys.version carries the toolchain's own source URL on fbcode builds, which
# the open-source export gate rejects. Keep the compiler name and version.
python_version = re.sub(
    r"\s*\([a-z][a-z0-9+.-]*://.*", "", sys.version.replace(chr(10), " ")
).rstrip()
if python_version.count("[") > python_version.count("]"):
    python_version += "]"

print(f"python_executable={sys.executable}")
print(f"python_version={python_version}")
for name in ("pyscipopt", "yices", "paper_joint_solver", "skc"):
    module = importlib.import_module(name)
    print(f"module_{name}_file={getattr(module, '__file__', '<unset>')}")
    print(f"module_{name}_version={getattr(module, '__version__', '<unset>')}")
print(f"solver_sources_sha256={solver_sources_sha256()}")
PY
} >"$ENVIRONMENT_LOG"

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
}

run_logged() {
  local expected_rc=$1
  local command_log=$2
  local output_log=$3
  shift 3
  local rc
  local -a command=("$@")

  write_command "$command_log" "${command[@]}"
  if "${command[@]}" >"$output_log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  if ((rc != expected_rc)); then
    cat "$output_log" >&2
    echo "command returned $rc; expected $expected_rc (see $command_log)" >&2
    return 1
  fi
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

run_case() {
  local stem=$1
  local ddg=$2
  local graph=$3
  shift 3
  local solution="$OUT/$stem.json"
  local ir="$OUT/${stem}_ir.json"
  local handoff="$OUT/${stem}_handoff.json"
  local manual="$OUT/${stem}_manual"
  local strategy="$OUT/$stem.strategy.json"
  local curated="$OUT/$stem.curated_ddg.json"
  local curation_manifest="$OUT/$stem.curation_manifest.json"
  local -a solver_command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver "$curated"
    --baseline-graph "$graph"
    -o "$solution"
    --ir-out "$ir"
    --handoff-manifest-out "$handoff"
    "$@"
  )
  local -a scaffold_command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m skc scaffold
    --ir "$ir"
    --handoff "$handoff"
    --out-dir "$manual"
  )
  local -a strategy_command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver.strategy_report
    "$curated" "$solution"
    --curation-manifest "$curation_manifest"
    --output "$strategy"
  )
  if [[ "$stem" == "fwd_subtiled_${VER}" ]]; then
    strategy_command+=(--fa4-template "$FA4_TEMPLATE")
  fi
  local -a audit_command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m skc audit-bundle
    --ir "$ir"
    --handoff "$handoff"
    --authoring "$manual/manual_cuda_authoring.json"
    --mapping "$manual/mapping_manifest.json"
    --memory "$manual/memory_plan.json"
    --sync "$manual/sync_manifest.json"
  )

  echo "running $stem"
  run_logged 0 "$OUT/$stem.curate.command" "$OUT/$stem.curate.log" \
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
    "$PYTHON" -m paper_joint_solver.curate_ddg "$ddg" \
    --baseline-graph "$graph" -o "$curated" \
    --manifest-out "$curation_manifest"

  local solve_outcome
  solve_outcome=$(run_solver_watchdogged "$stem.solve" \
    "$OUT/$stem.solve.command" "$OUT/$stem.solve.log" "${solver_command[@]}")
  if [[ "$solve_outcome" != completed ]]; then
    printf 'observation label=%s.downstream outcome=skipped reason=%s\n' \
      "$stem" "$solve_outcome" >>"$OBS_LOG"
    return 0
  fi

  run_logged 0 "$OUT/$stem.strategy.command" "$OUT/$stem.strategy.log" \
    "${strategy_command[@]}"
  run_logged 0 "$OUT/$stem.scaffold.command" "$OUT/$stem.scaffold.log" \
    "${scaffold_command[@]}"
  run_logged 2 "$OUT/$stem.audit.command" "$OUT/$stem.audit.log" \
    "${audit_command[@]}"
  if [[ "$(<"$OUT/$stem.audit.log")" != \
        *"skc: authoring status must be approved"* ]]; then
    cat "$OUT/$stem.audit.log" >&2
    echo "$stem draft audit did not fail at the authoring approval gate" >&2
    return 1
  fi
}

# Runs are serial to bound solver memory pressure. Canonical solves use the
# literal single-model CLI; only the paper's own experiment inputs are passed.
run_case "fwd_subtiled_${VER}" "$FWD_SUB_DDG" "$FWD_SUB_GRAPH" \
  --normalization-u "$NORMALIZATION_U"
run_case "fwd_${VER}" "$FWD_DDG" "$FWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U"
run_case "bwd_${VER}" "$BWD_DDG" "$BWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U"
run_case "bwd_lr4096_${VER}" "$BWD_DDG" "$BWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U" --reg-budget 4096

refit_outcome=$(run_solver_watchdogged refit \
  "$OUT/refit_${VER}.command" "$OUT/refit_${VER}.log" \
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" refit_check.py \
  --solution-dir "$OUT" \
  --fa4-template "$FA4_TEMPLATE" \
  --json-output "$OUT/refit_${VER}.json") || true

if ! env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT" "$VER" >"$VALIDATION_LOG" 2>&1 <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

from skc._schema import (
    AUTHORING_SCHEMA,
    MAPPING_SCHEMA,
    MEMORY_PLAN_SCHEMA,
    PIPELINED_IR_SCHEMA,
    SYNC_PLAN_SCHEMA,
    load_handoff,
    load_json_artifact,
    load_pipelined_ir,
    validate_plan_header,
)

out = Path(sys.argv[1])
version = sys.argv[2]
environment = {}
for line in (
    out / f"run_main_cases_environment_{version}.log"
).read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        environment[key] = value
if environment.get("paper_comparable") != "yes":
    raise SystemExit(f"canonical {version} validation requires paper_comparable=yes")

# The observation log decides which cases have artifacts to check at all.
# Its vocabulary is completed / did-not-terminate; no rc, no verdict.
observations = {}
for line in (
    out / f"run_main_cases_observations_{version}.log"
).read_text().splitlines():
    fields = dict(
        field.split("=", 1) for field in line.split() if "=" in field
    )
    label = fields.get("label")
    if label is not None:
        observations[label] = fields.get("outcome", "")

cases = {
    f"fwd_subtiled_{version}": {
        "ddg": Path("../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "strategy_observations": ("fa4_optimal_set_member", "fa4_like"),
    },
    f"fwd_{version}": {
        "ddg": Path("../sched2tlx/examples/case3_FA_fp16/ddg.json"),
        "graph": Path("../sched2tlx/examples/case3_FA_fp16/schedule_graph.json"),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "strategy_observations": (),
    },
    f"bwd_{version}": {
        "ddg": Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "strategy_observations": ("bwd_2wg_pingpong",),
    },
    f"bwd_lr4096_{version}": {
        "ddg": Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 4096,
        "strategy_observations": ("bwd_3wg_fa4",),
    },
}
scaffold_files = {
    "kernel.cu",
    "manual_cuda_authoring.json",
    "mapping_manifest.json",
    "memory_plan.json",
    "sync_manifest.json",
}
schemas = {
    "manual_cuda_authoring.json": AUTHORING_SCHEMA,
    "mapping_manifest.json": MAPPING_SCHEMA,
    "memory_plan.json": MEMORY_PLAN_SCHEMA,
    "sync_manifest.json": SYNC_PLAN_SCHEMA,
}
solver_source_hashes = set()
completed = []

for stem, expected in cases.items():
    outcome = observations.get(f"{stem}.solve")
    if outcome != "completed":
        # Integrity: a case that did not finish must not have left artifacts.
        for orphan in (
            out / f"{stem}.json",
            out / f"{stem}_ir.json",
            out / f"{stem}_handoff.json",
            out / f"{stem}_manual",
        ):
            if orphan.exists():
                raise SystemExit(f"{orphan} exists for an unfinished {stem}")
        print(
            f"OBSERVATION case={stem} expected=completed "
            f"observed=did-not-terminate"
        )
        continue
    completed.append(stem)

    solution_path = out / f"{stem}.json"
    solution_bytes = solution_path.read_bytes()
    solution = json.loads(solution_bytes)
    if not isinstance(solution, dict):
        raise SystemExit(f"{solution_path} root is not an object")
    # Mirror of the rc contract: the literal Algorithm 1 can only terminate sat.
    if solution.get("status") != "sat" or solution.get("satisfiable") is not True:
        raise SystemExit(f"{solution_path} is not a SAT solution")

    provenance = solution.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"{solution_path} has no provenance object")
    # provenance.model is the model-identity assertion: there is one model and
    # every canonical artifact must say so.
    if provenance.get("model") != "paper":
        raise SystemExit(
            f"{solution_path} was not produced by the paper model: "
            f"{provenance.get('model')!r}"
        )
    source_hash = provenance.get("solver_sources_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise SystemExit(f"{solution_path} has an invalid solver source hash")
    solver_source_hashes.add(source_hash)

    # raw ddg -> curated -> solution, closed through curation_source.
    curated_path = out / f"{stem}.curated_ddg.json"
    curated_hash = hashlib.sha256(curated_path.read_bytes()).hexdigest()
    if provenance.get("ddg_sha256") != curated_hash:
        raise SystemExit(f"{solution_path} ddg_sha256 does not match {curated_path}")
    curation_source = provenance.get("curation_source")
    if not isinstance(curation_source, dict):
        raise SystemExit(f"{solution_path} has no curation_source object")
    for field, path in (
        ("ddg_sha256", expected["ddg"]),
        ("baseline_graph_sha256", expected["graph"]),
    ):
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if curation_source.get(field) != actual_hash:
            raise SystemExit(
                f"{solution_path} curation_source.{field} does not match {path}"
            )
    manifest_path = out / f"{stem}.curation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_source = manifest.get("curation_source")
    if not isinstance(manifest_source, dict):
        raise SystemExit(f"{manifest_path} has no curation_source object")
    for field in ("ddg_sha256", "baseline_graph_sha256", "curator_sources_sha256"):
        if curation_source.get(field) != manifest_source.get(field):
            raise SystemExit(
                f"{stem} curation_source.{field} disagrees with {manifest_path}"
            )

    if provenance.get("normalization_u") != expected["normalization_u"]:
        raise SystemExit(f"{solution_path} normalization_u mismatch")
    machine = provenance.get("machine")
    if not isinstance(machine, dict):
        raise SystemExit(f"{solution_path} has no machine provenance")
    if machine.get("regs_per_warp") != expected["regs_per_warp"]:
        raise SystemExit(f"{solution_path} machine.regs_per_warp mismatch")
    if machine.get("warp_fixed_overhead") != 0:
        raise SystemExit(
            f"{solution_path} machine.warp_fixed_overhead is not 0: "
            f"{machine.get('warp_fixed_overhead')!r}"
        )
    inputs = provenance.get("experiment_inputs")
    if not isinstance(inputs, dict):
        raise SystemExit(f"{solution_path} has no experiment_inputs object")
    print(
        f"OBSERVATION case={stem} experiment_input "
        f"allow_cross_warp={inputs.get('allow_cross_warp')} "
        f"num_warps_override={inputs.get('num_warps_override')} "
        f"reg_budget={inputs.get('reg_budget')}"
    )

    # Schema integrity only: the encoding is warp sets, contents are free.
    warp_sets = solution.get("warp_sets")
    if not isinstance(warp_sets, dict) or not warp_sets:
        raise SystemExit(f"{solution_path} has no warp_sets allocation")

    if stem.startswith("bwd"):
        stats = solution.get("stats")
        if not isinstance(stats, dict):
            raise SystemExit(f"{solution_path} has no stats object")
        print(
            f"OBSERVATION case={stem} tmem: expected peak>0 "
            f"observed peak_tmem_cols={stats.get('peak_tmem_cols')!r} "
            f"tmem_value_count={stats.get('tmem_value_count')!r}"
        )

    solution_sha256 = hashlib.sha256(solution_bytes).hexdigest()
    strategy_path = out / f"{stem}.strategy.json"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    if strategy.get("schema_version") != "paper-joint-strategy-v2":
        raise SystemExit(f"{strategy_path} has an unexpected schema")
    if strategy.get("solution_sha256") != solution_sha256:
        raise SystemExit(f"{strategy_path} does not reference its solution")
    for name in expected["strategy_observations"]:
        print(
            f"OBSERVATION case={stem} strategy.{name}: expected=True "
            f"observed={strategy.get(name)!r}"
        )
    refit = strategy.get("fa4_template_refit")
    if isinstance(refit, dict):
        print(
            f"OBSERVATION case={stem} strategy.fa4_template_refit: "
            f"expected=sat at (II={solution.get('ii')}, L={solution.get('length')}) "
            f"observed verdict={refit.get('verdict')!r} "
            f"ii={refit.get('ii')!r} length={refit.get('length')!r}"
        )

    ir = load_pipelined_ir(out / f"{stem}_ir.json")
    if PIPELINED_IR_SCHEMA != "paper-joint-pipelined-ir-v3":
        raise SystemExit(f"unexpected current IR schema: {PIPELINED_IR_SCHEMA}")
    if ir.payload.get("schema_version") != "paper-joint-pipelined-ir-v3":
        raise SystemExit(f"{stem} did not emit v3 pipelined IR")
    if ir.payload.get("solution_sha256") != solution_sha256:
        raise SystemExit(f"{stem} IR does not reference its exact solution bytes")
    handoff = load_handoff(out / f"{stem}_handoff.json", ir)

    manual = out / f"{stem}_manual"
    present = {path.name for path in manual.iterdir()}
    if present != scaffold_files:
        raise SystemExit(
            f"{manual} has the wrong files: "
            f"missing={sorted(scaffold_files - present)}, "
            f"extra={sorted(present - scaffold_files)}"
        )
    kernel = (manual / "kernel.cu").read_text(encoding="utf-8")
    if '#error "Manual CUDA lowering is required' not in kernel:
        raise SystemExit(f"{manual}/kernel.cu is not a fail-closed scaffold")

    plans = {}
    for filename, schema in schemas.items():
        artifact = load_json_artifact(manual / filename, filename)
        validate_plan_header(artifact.payload, schema, ir)
        if artifact.payload.get("status") != "manual_completion_required":
            raise SystemExit(f"{manual}/{filename} is not a draft")
        plans[filename] = artifact
    authoring = plans["manual_cuda_authoring.json"].payload
    if authoring.get("solution_sha256") != solution_sha256:
        raise SystemExit(f"{stem} authoring record solution hash mismatch")
    handoff_ref = authoring.get("handoff")
    if not isinstance(handoff_ref, dict) or handoff_ref.get("sha256") != handoff.sha256:
        raise SystemExit(f"{stem} authoring record handoff hash mismatch")
    manifest_refs = authoring.get("manifests")
    if not isinstance(manifest_refs, dict):
        raise SystemExit(f"{stem} authoring record has no manifest references")
    for kind, filename in (
        ("mapping", "mapping_manifest.json"),
        ("memory", "memory_plan.json"),
        ("synchronization", "sync_manifest.json"),
    ):
        reference = manifest_refs.get(kind)
        if not isinstance(reference, dict):
            raise SystemExit(f"{stem} has no {kind} manifest reference")
        if reference.get("path") != filename:
            raise SystemExit(f"{stem} {kind} manifest path mismatch")
        if reference.get("sha256") != plans[filename].sha256:
            raise SystemExit(f"{stem} {kind} manifest hash mismatch")
    print(f"{stem}: solution, v3 IR, handoff, and draft scaffold validated")

if len(solver_source_hashes) > 1:
    raise SystemExit(
        "solutions have different solver_sources_sha256 values: "
        + ", ".join(sorted(solver_source_hashes))
    )
if solver_source_hashes:
    print(f"solver_sources_sha256={solver_source_hashes.pop()}")
refit_path = out / f"refit_{version}.json"
if refit_path.is_file():
    refit_report = json.loads(refit_path.read_text(encoding="utf-8"))
    observed = (
        refit_report.get("passed")
        if isinstance(refit_report, dict)
        else "<not an object>"
    )
else:
    observed = "<absent>"
print(f"OBSERVATION refit: expected passed=True observed={observed!r}")
print("all draft audit-bundle checks rejected at the authoring approval gate")
PY
then
  cat "$VALIDATION_LOG" >&2
  exit 1
fi

cat "$VALIDATION_LOG"
