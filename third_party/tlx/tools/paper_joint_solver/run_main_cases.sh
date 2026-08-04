#!/bin/bash
# Reproduce the four paper-fidelity Phase-3 solves and their manual-CUDA
# handoff scaffolds. Run this script from third_party/tlx/tools/paper_joint_solver.
# Outputs are append-never: remove stale targets or choose a fresh OUT directory.
set -euo pipefail

PYTHON_REQUESTED=${PYTHON:-../../../../.venv/bin/python}
SOLVER_LIB_PATH="${SOLVER_LIB_PATH:?set SOLVER_LIB_PATH to <yices>/lib:<cudd>/lib}"
SOLVER_CPU=${SOLVER_CPU:-0}
DEFAULT_OUT=solutions
OUT_WAS_EXPLICIT=${OUT+x}
OUT=${OUT:-$DEFAULT_OUT}
UNPINNED_HOST=${UNPINNED_HOST:-0}
NORMALIZATION_U=300
FA4_TEMPLATE=subtiled_fa4exact_solution.json
FA4_REFIT_TIMEOUT_S=1800
DESYM_TIMEOUT_S=1800

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

stems=(fwd_subtiled_v8 fwd_v8 bwd_v8 bwd_lr4096_v8)
reserved=(
  "$OUT/run_main_cases_environment.log"
  "$OUT/run_main_cases_validation.log"
  "$OUT/fig9_v8.dot"
  "$OUT/refit_v8.json"
  "$OUT/refit_v8.command"
  "$OUT/refit_v8.log"
)
for stem in "${stems[@]}"; do
  reserved+=(
    "$OUT/$stem.json"
    "$OUT/${stem}_ir.json"
    "$OUT/${stem}_handoff.json"
    "$OUT/${stem}_manual"
    "$OUT/$stem.solve.command"
    "$OUT/$stem.solve.log"
    "$OUT/$stem.strategy.json"
    "$OUT/$stem.strategy.command"
    "$OUT/$stem.strategy.log"
    "$OUT/$stem.desym.command"
    "$OUT/$stem.desym.log"
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

ENVIRONMENT_LOG="$OUT/run_main_cases_environment.log"
VALIDATION_LOG="$OUT/run_main_cases_validation.log"

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
} >"$ENVIRONMENT_LOG"

write_command() {
  local destination=$1
  shift
  {
    printf '%q ' "$@"
    printf '\n'
  } >"$destination"
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
  local -a solver_command=(
    taskset -c "$SOLVER_CPU"
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" -m paper_joint_solver "$ddg"
    --baseline-graph "$graph"
    -o "$solution"
    --ir-out "$ir"
    --handoff-manifest-out "$handoff"
    --ilp-seconds 240
    --smt-seconds 300
    --max-wall-s 3600
    "$@"
  )
  if [[ "$stem" == fwd_subtiled_v8 ]]; then
    solver_command+=(--viz "$OUT/fig9_v8.dot")
  fi
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
    "$ddg" "$solution"
    --baseline-graph "$graph"
    --output "$strategy"
  )
  if [[ "$stem" == fwd_subtiled_v8 ]]; then
    strategy_command+=(
      --fa4-template "$FA4_TEMPLATE"
      --fa4-refit-timeout-s "$FA4_REFIT_TIMEOUT_S"
    )
  fi
  local -a desym_command=(
    env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH"
    "$PYTHON" desym_check.py
    "$ddg" "$solution" "$DESYM_TIMEOUT_S"
    --baseline-graph "$graph"
  )
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
  run_logged 0 "$OUT/$stem.solve.command" "$OUT/$stem.solve.log" \
    "${solver_command[@]}"
  run_logged 0 "$OUT/$stem.strategy.command" "$OUT/$stem.strategy.log" \
    "${strategy_command[@]}"
  run_logged 0 "$OUT/$stem.desym.command" "$OUT/$stem.desym.log" \
    "${desym_command[@]}"
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

# Runs are serial to bound solver memory pressure. Every model-affecting case
# choice is explicit even where it matches the current CLI default.
run_case fwd_subtiled_v8 "$FWD_SUB_DDG" "$FWD_SUB_GRAPH" \
  --normalization-u "$NORMALIZATION_U" --reg-budget 8160 \
  --warp-fixed-overhead 4
run_case fwd_v8 "$FWD_DDG" "$FWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U" --reg-budget 8160 \
  --warp-fixed-overhead 4
run_case bwd_v8 "$BWD_DDG" "$BWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U" --reg-budget 8160 \
  --warp-fixed-overhead 0
run_case bwd_lr4096_v8 "$BWD_DDG" "$BWD_GRAPH" \
  --normalization-u "$NORMALIZATION_U" --reg-budget 4096 \
  --warp-fixed-overhead 0

run_logged 0 "$OUT/refit_v8.command" "$OUT/refit_v8.log" \
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" refit_check.py \
  --solution-dir "$OUT" \
  --fa4-template "$FA4_TEMPLATE" \
  --timeout-s "$FA4_REFIT_TIMEOUT_S" \
  --json-output "$OUT/refit_v8.json"

if ! env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$SOLVER_LIB_PATH" \
  "$PYTHON" - "$OUT" >"$VALIDATION_LOG" 2>&1 <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

from paper_joint_solver.normalize import MAX_U_MULTIPLIER
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
environment = {}
for line in (out / "run_main_cases_environment.log").read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        environment[key] = value
if environment.get("paper_comparable") != "yes":
    raise SystemExit("canonical v8 validation requires paper_comparable=yes")
cases = {
    "fwd_subtiled_v8": {
        "ddg": Path("../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case3_FA_fp16_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "warp_fixed_overhead": 4,
        "strategy_gate": "fa4_optimal_set_member",
        "strategy_observation": "fa4_like",
    },
    "fwd_v8": {
        "ddg": Path("../sched2tlx/examples/case3_FA_fp16/ddg.json"),
        "graph": Path("../sched2tlx/examples/case3_FA_fp16/schedule_graph.json"),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "warp_fixed_overhead": 4,
        "strategy_gate": None,
        "strategy_observation": None,
    },
    "bwd_v8": {
        "ddg": Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 8160,
        "warp_fixed_overhead": 0,
        "strategy_gate": "bwd_2wg_pingpong",
        "strategy_observation": None,
    },
    "bwd_lr4096_v8": {
        "ddg": Path("../sched2tlx/examples/case4_FA_bwd_subtiled/ddg.json"),
        "graph": Path(
            "../sched2tlx/examples/case4_FA_bwd_subtiled/schedule_graph.json"
        ),
        "normalization_u": 300,
        "regs_per_warp": 4096,
        "warp_fixed_overhead": 0,
        "strategy_gate": "bwd_3wg_fa4",
        "strategy_observation": None,
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

for stem, expected in cases.items():
    solution_path = out / f"{stem}.json"
    solution_bytes = solution_path.read_bytes()
    solution = json.loads(solution_bytes)
    if not isinstance(solution, dict):
        raise SystemExit(f"{solution_path} root is not an object")
    if solution.get("status") != "sat" or solution.get("satisfiable") is not True:
        raise SystemExit(f"{solution_path} is not a SAT solution")

    provenance = solution.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"{solution_path} has no provenance object")
    source_hash = provenance.get("solver_sources_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise SystemExit(f"{solution_path} has an invalid solver source hash")
    solver_source_hashes.add(source_hash)
    for field, path in (
        ("ddg_sha256", expected["ddg"]),
        ("baseline_graph_sha256", expected["graph"]),
    ):
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if provenance.get(field) != actual_hash:
            raise SystemExit(f"{solution_path} {field} does not match {path}")
    if provenance.get("normalization_u") != expected["normalization_u"]:
        raise SystemExit(f"{solution_path} normalization_u mismatch")
    normalization_u_effective = provenance.get("normalization_u_effective")
    if (
        isinstance(normalization_u_effective, bool)
        or not isinstance(normalization_u_effective, int)
        or normalization_u_effective < expected["normalization_u"]
    ):
        raise SystemExit(f"{solution_path} normalization_u_effective mismatch")
    if (
        normalization_u_effective
        >= expected["normalization_u"] * MAX_U_MULTIPLIER
    ):
        raise SystemExit(
            f"{solution_path} normalization_u_effective has no retry headroom"
        )
    machine = provenance.get("machine")
    if not isinstance(machine, dict):
        raise SystemExit(f"{solution_path} has no machine provenance")
    for field in ("regs_per_warp", "warp_fixed_overhead"):
        if machine.get(field) != expected[field]:
            raise SystemExit(f"{solution_path} machine.{field} mismatch")
    if stem.startswith("bwd"):
        stats = solution.get("stats")
        if not isinstance(stats, dict):
            raise SystemExit(f"{solution_path} has no stats object")
        peak_tmem_cols = stats.get("peak_tmem_cols")
        tmem_object_count = stats.get("tmem_object_count")
        if (
            isinstance(peak_tmem_cols, bool)
            or not isinstance(peak_tmem_cols, int)
            or peak_tmem_cols <= 0
        ):
            raise SystemExit(f"{solution_path} did not charge accumulator TMEM")
        if (
            isinstance(tmem_object_count, bool)
            or not isinstance(tmem_object_count, int)
            or tmem_object_count <= 0
        ):
            raise SystemExit(f"{solution_path} has no TMEM storage objects")

    solution_sha256 = hashlib.sha256(solution_bytes).hexdigest()
    strategy_path = out / f"{stem}.strategy.json"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    if strategy.get("schema_version") != "paper-joint-strategy-v1":
        raise SystemExit(f"{strategy_path} has an unexpected schema")
    if strategy.get("solution_sha256") != solution_sha256:
        raise SystemExit(f"{strategy_path} does not reference its solution")
    strategy_gate = expected["strategy_gate"]
    if strategy_gate is not None and strategy.get(strategy_gate) is not True:
        raise SystemExit(
            f"{strategy_path} failed {strategy_gate}: "
            f"groups={strategy.get('group_ops')}"
        )
    strategy_observation = expected["strategy_observation"]
    if strategy_observation is not None:
        observed = strategy.get(strategy_observation)
        if not isinstance(observed, bool):
            raise SystemExit(
                f"{strategy_path} has no boolean {strategy_observation} observation"
            )
        print(f"{stem}: observed {strategy_observation}={observed}")
    if strategy_gate == "fa4_optimal_set_member":
        refit = strategy.get("fa4_template_refit")
        if not isinstance(refit, dict) or refit.get("verdict") != "sat":
            raise SystemExit(f"{strategy_path} has no SAT FA4 template refit")
        if refit.get("ii") != solution.get("ii") or refit.get("length") != solution.get(
            "length"
        ):
            raise SystemExit(f"{strategy_path} refit point does not match free optimum")
    ir = load_pipelined_ir(out / f"{stem}_ir.json")
    if PIPELINED_IR_SCHEMA != "twill-pipelined-warp-ir-v2":
        raise SystemExit(f"unexpected current IR schema: {PIPELINED_IR_SCHEMA}")
    if ir.payload.get("schema_version") != "twill-pipelined-warp-ir-v2":
        raise SystemExit(f"{stem} did not emit v2 pipelined IR")
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
    print(f"{stem}: solution, v2 IR, handoff, and draft scaffold validated")

if len(solver_source_hashes) != 1:
    raise SystemExit(
        "solutions have different solver_sources_sha256 values: "
        + ", ".join(sorted(solver_source_hashes))
    )
print(f"solver_sources_sha256={solver_source_hashes.pop()}")
figure = out / "fig9_v8.dot"
if not figure.is_file() or figure.stat().st_size == 0:
    raise SystemExit(f"{figure} was not generated")
refit_report = json.loads((out / "refit_v8.json").read_text(encoding="utf-8"))
if not isinstance(refit_report, dict) or refit_report.get("passed") is not True:
    raise SystemExit("refit_v8.json is not a passing report")
print("all draft audit-bundle checks rejected at the authoring approval gate")
PY
then
  cat "$VALIDATION_LOG" >&2
  exit 1
fi

cat "$VALIDATION_LOG"
