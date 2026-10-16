"""Shared schemas and validation for the manual-CUDA SKC workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paper_joint_solver.pipelined_ir import (
    MANUAL_CUDA_HANDOFF_SCHEMA as HANDOFF_SCHEMA,
    PIPELINED_IR_SCHEMA,
)
from paper_joint_solver.schedule_plan import (
    PAPER_MODEL_PROFILE,
    solver_sources_sha256,
)


# v2 is one identity switch: renamed off the old codename and reshaped onto
# physical warps in the same migration.  Splitting the two would leave a window
# in which one string name denoted two different shapes.
AUTHORING_SCHEMA = "paper-joint-agent-assisted-manual-cuda-v2"
MAPPING_SCHEMA = "paper-joint-manual-cuda-mapping-v2"
MEMORY_PLAN_SCHEMA = "paper-joint-manual-cuda-memory-v2"
SYNC_PLAN_SCHEMA = "paper-joint-manual-cuda-sync-v2"
AUTHORING_MODE = "agent-assisted-manual-cuda"
TARGET_GPU = "B200"
TARGET_ARCH = "sm_100a"
MAX_REGS_PER_WARP = 8160

B200_CAPACITIES = {
    "register_words": 65536,
    "shared_bytes": 232448,
    "tmem_columns": 512,
    "physical_warps": 32,
}

ASYNC_COMPLETION_KINDS = {
    "tt.descriptor_load": "tma_load_completion",
    "tt.descriptor_store": "tma_store_completion",
    "ttng.async_tma_copy_global_to_local": "tma_load_completion",
    "ttng.async_tma_copy_local_to_global": "tma_store_completion",
    "ttng.async_tma_store": "tma_store_completion",
    "ttng.tc_gen5_mma": "tcgen05_completion",
    "ttng.tmem_store": "tmem_publication",
}


class SKCError(ValueError):
    """Base error for deterministic manual-CUDA tooling."""


class ArtifactError(SKCError):
    """Raised when an input artifact is malformed or inconsistent."""


class AuditError(SKCError):
    """Raised when a manual plan fails an audit."""


@dataclass(frozen=True)
class LoadedArtifact:
    payload: dict[str, Any]
    sha256: str
    path: Path | None


ArtifactSource = str | Path | Mapping[str, Any]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ArtifactError(f"{name} must be an integer >= {minimum}")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    value = _require_string(value, name)
    if len(value) != 64:
        raise ArtifactError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ArtifactError(f"{name} must be a lowercase SHA-256") from error
    if value != value.lower():
        raise ArtifactError(f"{name} must be a lowercase SHA-256")
    return value


def _require_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactError(f"{name} must be a list")
    return value


def strict_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def load_json_artifact(source: ArtifactSource, name: str) -> LoadedArtifact:
    path: Path | None = None
    if isinstance(source, Mapping):
        payload = dict(source)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    else:
        path = Path(source)
        try:
            encoded = path.read_bytes()
            payload = json.loads(encoded)
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"cannot load {name}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"{name} root must be an object")
    return LoadedArtifact(
        payload=payload,
        sha256=hashlib.sha256(encoded).hexdigest(),
        path=path,
    )


def _validate_source_program(payload: dict[str, Any]) -> dict[str, Any]:
    source = _require_object(payload.get("source_program"), "source_program")
    if source.get("format") != "ttgir-derived-operation-table":
        raise ArtifactError("source_program has an unsupported format")
    _require_sha256(source.get("sha256"), "source_program.sha256")
    if source.get("input_schema_version") != "0.1":
        raise ArtifactError("source_program.input_schema_version must be 0.1")
    kernel = _require_object(source.get("kernel"), "source_program.kernel")
    _require_string(kernel.get("name"), "source_program.kernel.name")
    arguments = _require_list(kernel.get("args"), "source_program.kernel.args")
    argument_names: set[str] = set()
    for position, raw in enumerate(arguments):
        argument = _require_object(raw, f"source_program.kernel.args[{position}]")
        argument_name = _require_string(
            argument.get("name"), f"source_program.kernel.args[{position}].name"
        )
        if argument_name in argument_names:
            raise ArtifactError(f"duplicate source kernel argument {argument_name}")
        argument_names.add(argument_name)
        _require_string(
            argument.get("type"), f"source_program.kernel.args[{position}].type"
        )
    ops = _require_object(source.get("ops"), "source_program.ops")
    if not ops:
        raise ArtifactError("source_program.ops must not be empty")
    for op_ref, raw in ops.items():
        _require_string(op_ref, "source operation reference")
        operation = _require_object(raw, f"source operation {op_ref}")
        _require_string(operation.get("kind"), f"source operation {op_ref}.kind")
        _require_string(operation.get("scope"), f"source operation {op_ref}.scope")
        _require_list(operation.get("operands"), f"source operation {op_ref}.operands")
        result_types = _require_list(
            operation.get("result_types"), f"source operation {op_ref}.result_types"
        )
        if any(
            not isinstance(result_type, str) or not result_type
            for result_type in result_types
        ):
            raise ArtifactError(
                f"source operation {op_ref}.result_types must contain strings"
            )
        _require_object(
            operation.get("attributes"), f"source operation {op_ref}.attributes"
        )
    binary_kinds = (
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.max",
        "arith.minimum",
        "arith.maximum",
    )
    for op_ref, operation in ops.items():
        operands = operation["operands"]
        kind = operation["kind"]
        if kind.startswith(binary_kinds) and len(operands) < 2:
            raise ArtifactError(f"source operation {op_ref} is missing binary operands")
        if kind in {"tt.descriptor_load", "tt.descriptor_store"} and not operands:
            raise ArtifactError(
                f"source operation {op_ref} is missing descriptor operands"
            )
        for position, raw_operand in enumerate(operands):
            operand = _require_object(
                raw_operand, f"source operation {op_ref}.operands[{position}]"
            )
            kinds = {
                key
                for key in (
                    "arg",
                    "const",
                    "iter_arg",
                    "iv",
                    "op",
                    "unknown_block_arg",
                )
                if key in operand
            }
            if len(kinds) != 1:
                raise ArtifactError(
                    f"source operation {op_ref}.operands[{position}] has invalid kind"
                )
            operand_kind = next(iter(kinds))
            if operand_kind == "arg":
                if operand["arg"] not in argument_names:
                    raise ArtifactError(
                        f"source operation {op_ref} references unknown argument"
                    )
            elif operand_kind == "op":
                producer = operand["op"]
                if producer not in ops:
                    raise ArtifactError(
                        f"source operation {op_ref} references unknown operation"
                    )
                result = operand.get("result", 0)
                if not _is_int(result) or not 0 <= result < len(
                    ops[producer]["result_types"]
                ):
                    raise ArtifactError(
                        f"source operation {op_ref} references invalid result"
                    )
            elif operand_kind == "const":
                _require_string(
                    operand.get("type"),
                    f"source operation {op_ref}.operands[{position}].type",
                )
            elif operand_kind == "iter_arg":
                reference = _require_object(
                    operand["iter_arg"],
                    f"source operation {op_ref}.operands[{position}].iter_arg",
                )
                _require_int(reference.get("loop"), "iter_arg.loop")
                _require_int(reference.get("idx"), "iter_arg.idx")
            elif operand_kind == "iv":
                _require_int(operand["iv"], "source induction-variable reference")
            elif operand_kind == "unknown_block_arg":
                if operand["unknown_block_arg"] is not True:
                    raise ArtifactError("unknown_block_arg marker must be true")
    _require_list(
        source.get("data_partition_candidates"),
        "source_program.data_partition_candidates",
    )
    return ops


def _validate_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    pipeline = _require_object(payload.get("pipeline"), "pipeline")
    ii = _require_int(pipeline.get("ii"), "pipeline.ii", minimum=1)
    length = _require_int(pipeline.get("length"), "pipeline.length", minimum=1)
    # L < I is a legal degenerate shape: the region formulas are applied
    # verbatim, giving an empty prologue and an inverted (empty) epilogue.
    copies = _require_int(pipeline.get("copies"), "pipeline.copies", minimum=1)
    horizon = _require_int(pipeline.get("horizon"), "pipeline.horizon", minimum=1)
    expected_copies = -(-length // ii)
    expected_horizon = (copies - 1) * ii + length
    if copies != expected_copies:
        raise ArtifactError(
            f"pipeline.copies={copies} != ceil(length/ii)={expected_copies}"
        )
    if horizon != expected_horizon:
        raise ArtifactError(f"pipeline.horizon={horizon} != {expected_horizon}")
    prologue_end = (copies - 1) * ii
    steady_end = prologue_end + ii
    expected_regions = [
        {"kind": "prologue", "cycle_begin": 0, "cycle_end": prologue_end},
        {
            "kind": "steady_state",
            "cycle_begin": prologue_end,
            "cycle_end": steady_end,
        },
        {
            "kind": "epilogue",
            "cycle_begin": steady_end,
            "cycle_end": horizon,
        },
    ]
    if not strict_equal(pipeline.get("regions"), expected_regions):
        raise ArtifactError("pipeline.regions do not match ii/length/copies/horizon")
    return pipeline


def _validate_instructions(
    payload: dict[str, Any], pipeline: dict[str, Any], source_ops: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    instructions = _require_list(payload.get("instructions"), "instructions")
    if not instructions:
        raise ArtifactError("instructions must not be empty")
    by_id: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(instructions):
        instruction = _require_object(raw, f"instructions[{position}]")
        node_id = _require_int(instruction.get("id"), f"instructions[{position}].id")
        if node_id in by_id:
            raise ArtifactError(f"duplicate instruction id {node_id}")
        op_ref = _require_string(
            instruction.get("op_ref"), f"instruction {node_id}.op_ref"
        )
        if op_ref not in source_ops:
            raise ArtifactError(f"instruction {node_id} references unknown source op")
        op_kind = _require_string(
            instruction.get("op_kind"), f"instruction {node_id}.op_kind"
        )
        if source_ops[op_ref].get("kind") != op_kind:
            raise ArtifactError(f"instruction {node_id} source operation kind mismatch")
        _require_string(instruction.get("pipeline"), f"instruction {node_id}.pipeline")
        cycle = _require_int(instruction.get("cycle"), f"instruction {node_id}.cycle")
        stage = _require_int(instruction.get("stage"), f"instruction {node_id}.stage")
        offset = _require_int(
            instruction.get("offset"), f"instruction {node_id}.offset"
        )
        if offset >= pipeline["ii"]:
            raise ArtifactError(f"instruction {node_id} offset exceeds pipeline II")
        if cycle != stage * pipeline["ii"] + offset:
            raise ArtifactError(f"instruction {node_id} cycle/stage/offset mismatch")
        if cycle >= pipeline["length"]:
            raise ArtifactError(f"instruction {node_id} cycle exceeds pipeline length")
        warps = _require_list(instruction.get("warps"), f"instruction {node_id}.warps")
        if not warps or any(not _is_int(warp) or warp < 0 for warp in warps):
            raise ArtifactError(
                f"instruction {node_id} warps must be non-empty non-negative ints"
            )
        if len(set(warps)) != len(warps):
            raise ArtifactError(f"instruction {node_id} has duplicate warps")
        if warps != sorted(warps):
            raise ArtifactError(f"instruction {node_id} warps must be sorted")
        if any(warp >= B200_CAPACITIES["physical_warps"] for warp in warps):
            raise ArtifactError(f"instruction {node_id} warp exceeds the B200 limit")
        by_id[node_id] = instruction
    return by_id


def _validate_warps(
    payload: dict[str, Any], instructions: dict[int, dict[str, Any]]
) -> dict[int, list[int]]:
    """One issue trace per physical warp.

    A multi-warp operation appears in every member warp's trace, so the
    traces are not a partition of the instructions.
    """
    warps = _require_list(payload.get("warps"), "warps")
    if not warps:
        raise ArtifactError("warps must not be empty")
    issue_order: dict[int, list[int]] = {}
    for position, raw in enumerate(warps):
        entry = _require_object(raw, f"warps[{position}]")
        warp_id = _require_int(entry.get("id"), f"warps[{position}].id")
        if not 0 <= warp_id < B200_CAPACITIES["physical_warps"]:
            raise ArtifactError(f"warp {warp_id} is outside the B200 warp range")
        if warp_id in issue_order:
            raise ArtifactError(f"duplicate warp {warp_id}")
        trace = _require_list(
            entry.get("issue_trace"), f"warps[{position}].issue_trace"
        )
        nodes: list[int] = []
        for trace_position, raw_entry in enumerate(trace):
            item = _require_object(
                raw_entry, f"warp {warp_id} issue_trace[{trace_position}]"
            )
            node_id = _require_int(
                item.get("node"), f"warp {warp_id} issue trace node"
            )
            instruction = instructions.get(node_id)
            if instruction is None or warp_id not in instruction["warps"]:
                raise ArtifactError(
                    f"warp {warp_id} issue trace references invalid node {node_id}"
                )
            for field in ("cycle", "stage", "offset"):
                if not strict_equal(item.get(field), instruction.get(field)):
                    raise ArtifactError(
                        f"instruction {node_id} issue-trace {field} mismatch"
                    )
            nodes.append(node_id)
        if len(nodes) != len(set(nodes)):
            raise ArtifactError(f"warp {warp_id} issue trace has duplicate nodes")
        expected = sorted(
            (
                node_id
                for node_id, instruction in instructions.items()
                if warp_id in instruction["warps"]
            ),
            key=lambda node_id: (
                instructions[node_id]["stage"],
                instructions[node_id]["offset"],
                node_id,
            ),
        )
        if nodes != expected:
            raise ArtifactError(
                f"warp {warp_id} issue trace is incomplete or unordered"
            )
        issue_order[warp_id] = nodes
    used_warps = {
        warp for instruction in instructions.values() for warp in instruction["warps"]
    }
    if set(issue_order) != used_warps:
        raise ArtifactError("warps must cover exactly the instruction warps")
    return issue_order


def _is_cross_warp_dependency(edge: dict[str, Any]) -> bool:
    return edge["ws_semantics"] and (
        edge["producer_warps"] != edge["consumer_warps"]
    )


def _validate_dependencies(
    payload: dict[str, Any],
    instructions: dict[int, dict[str, Any]],
    pipeline: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    dependencies = _require_list(payload.get("dependencies"), "dependencies")
    if not dependencies:
        raise ArtifactError("dependencies must not be empty")
    by_index: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(dependencies):
        edge = _require_object(raw, f"dependencies[{position}]")
        index = _require_int(edge.get("index"), f"dependencies[{position}].index")
        if index in by_index:
            raise ArtifactError(f"duplicate dependency index {index}")
        src = _require_int(edge.get("src"), f"dependency {index} src")
        dst = _require_int(edge.get("dst"), f"dependency {index} dst")
        if src not in instructions or dst not in instructions:
            raise ArtifactError(f"dependency {index} references an unknown instruction")
        distance = _require_int(edge.get("distance"), f"dependency {index} distance")
        latency = _require_int(edge.get("latency"), f"dependency {index} latency")
        producer_warps = _require_list(
            edge.get("producer_warps"), f"dependency {index} producer warps"
        )
        consumer_warps = _require_list(
            edge.get("consumer_warps"), f"dependency {index} consumer warps"
        )
        if not strict_equal(producer_warps, instructions[src]["warps"]):
            raise ArtifactError(f"dependency {index} producer warps mismatch")
        if not strict_equal(consumer_warps, instructions[dst]["warps"]):
            raise ArtifactError(f"dependency {index} consumer warps mismatch")
        if not isinstance(edge.get("ws_semantics"), bool):
            raise ArtifactError(f"dependency {index} ws_semantics must be a boolean")
        spill_cost = _require_int(
            edge.get("spill_cost"), f"dependency {index} spill cost"
        )
        if not _is_cross_warp_dependency(edge) and spill_cost != 0:
            raise ArtifactError(f"dependency {index} non-cross spill cost must be zero")
        available = instructions[dst]["cycle"] + distance * pipeline["ii"]
        required = instructions[src]["cycle"] + latency + spill_cost
        if available < required:
            raise ArtifactError(f"dependency {index} violates scheduled latency")
        by_index[index] = edge
    cross = _require_list(
        payload.get("cross_warp_dependencies"), "cross_warp_dependencies"
    )
    cross_indices: list[int] = []
    for position, raw in enumerate(cross):
        edge = _require_object(raw, f"cross_warp_dependencies[{position}]")
        index = _require_int(edge.get("index"), "cross-warp dependency index")
        if index not in by_index or not strict_equal(edge, by_index[index]):
            raise ArtifactError(
                f"cross-warp dependency {index} does not match dependencies"
            )
        cross_indices.append(index)
    if len(cross_indices) != len(set(cross_indices)):
        raise ArtifactError("cross_warp_dependencies contains duplicate edges")
    expected_cross = {
        index
        for index, edge in by_index.items()
        if _is_cross_warp_dependency(edge)
    }
    if set(cross_indices) != expected_cross:
        raise ArtifactError(
            "cross_warp_dependencies must cover exactly cross-warp edges"
        )
    return by_index


def _program_region(cycle: int, regions: list[dict[str, Any]]) -> str:
    for region in regions:
        if region["cycle_begin"] <= cycle < region["cycle_end"]:
            return region["kind"]
    raise ArtifactError(f"expanded cycle {cycle} is outside pipeline regions")


def _expected_instances(
    nodes: list[dict[str, Any]], pipeline: dict[str, Any]
) -> list[dict[str, Any]]:
    ii = pipeline["ii"]
    copies = pipeline["copies"]
    regions = pipeline["regions"]
    result = []
    for node in nodes:
        for copy in range(copies):
            cycle = node["cycle"] + copy * ii
            result.append(
                {
                    "node": node["id"],
                    "copy": copy,
                    "cycle": cycle,
                    "region": _program_region(cycle, regions),
                }
            )
    return result


def _expected_instance_dependencies(
    edges: list[dict[str, Any]], copies: int
) -> list[dict[str, Any]]:
    result = []
    for edge in edges:
        for consumer_copy in range(copies):
            producer_copy = consumer_copy - edge["distance"]
            result.append(
                {
                    "edge_index": edge["index"],
                    "producer": {"node": edge["src"], "copy": producer_copy},
                    "consumer": {"node": edge["dst"], "copy": consumer_copy},
                    "external": producer_copy < 0,
                    "carried_distance": edge["distance"],
                }
            )
    return result


def _expected_steady_state_slots(
    nodes: list[dict[str, Any]], pipeline: dict[str, Any]
) -> list[dict[str, Any]]:
    ii = pipeline["ii"]
    copies = pipeline["copies"]
    regions = pipeline["regions"]
    result = []
    for node in nodes:
        copy = (copies - 1) - node["stage"]
        cycle = node["cycle"] + copy * ii
        iteration_lag = (copies - 1) - copy
        if (
            not 0 <= copy < copies
            or _program_region(cycle, regions) != "steady_state"
            or iteration_lag != node["stage"]
        ):
            raise ArtifactError(
                f"instruction {node['id']} has no canonical steady-state instance"
            )
        result.append(
            {
                "node": node["id"],
                "copy": copy,
                "cycle": cycle,
                "iteration_lag": iteration_lag,
            }
        )
    return result


def _expected_pipelined_program(
    instructions: dict[int, dict[str, Any]],
    dependencies: dict[int, dict[str, Any]],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    nodes = [instructions[node_id] for node_id in sorted(instructions)]
    edges = [dependencies[index] for index in sorted(dependencies)]
    return {
        "instances": _expected_instances(nodes, pipeline),
        "instance_dependencies": _expected_instance_dependencies(
            edges, pipeline["copies"]
        ),
        "steady_state": {
            "slots": _expected_steady_state_slots(nodes, pipeline)
        },
    }


def _validate_pipelined_program(
    payload: dict[str, Any],
    instructions: dict[int, dict[str, Any]],
    dependencies: dict[int, dict[str, Any]],
    pipeline: dict[str, Any],
) -> None:
    program = _require_object(payload.get("pipelined_program"), "pipelined_program")
    expected = _expected_pipelined_program(instructions, dependencies, pipeline)
    if not strict_equal(program, expected):
        raise ArtifactError("pipelined_program does not match the schedule expansion")


def _validate_provenance(payload: dict[str, Any], source_sha256: str) -> None:
    provenance = _require_object(payload.get("provenance"), "provenance")
    if provenance.get("model") != PAPER_MODEL_PROFILE:
        raise ArtifactError(
            f"provenance.model must be {PAPER_MODEL_PROFILE!r}, got "
            f"{provenance.get('model')!r}"
        )
    _require_sha256(provenance.get("ddg_sha256"), "provenance.ddg_sha256")
    curation_source = _require_object(
        provenance.get("curation_source"), "provenance.curation_source"
    )
    for field in ("ddg_sha256", "baseline_graph_sha256", "curator_sources_sha256"):
        _require_sha256(
            curation_source.get(field), f"provenance.curation_source.{field}"
        )
    if curation_source["baseline_graph_sha256"] != source_sha256:
        raise ArtifactError("source program and provenance hashes disagree")
    # Experiment inputs are transcribed observations, never compared.
    _require_object(
        provenance.get("experiment_inputs"), "provenance.experiment_inputs"
    )
    solver_hash = _require_sha256(
        provenance.get("solver_sources_sha256"),
        "provenance.solver_sources_sha256",
    )
    current_solver_hash = solver_sources_sha256()
    if solver_hash != current_solver_hash:
        raise ArtifactError(
            "provenance.solver_sources_sha256 does not match the current toolchain"
        )
    _require_int(
        provenance.get("normalization_u"),
        "provenance.normalization_u",
        minimum=1,
    )
    machine = _require_object(provenance.get("machine"), "provenance.machine")
    expected = {
        "smem_bytes": B200_CAPACITIES["shared_bytes"],
        "tmem_cols": B200_CAPACITIES["tmem_columns"],
        "regs_per_sm": B200_CAPACITIES["register_words"],
        "num_warps": B200_CAPACITIES["physical_warps"],
        "num_warpgroups": B200_CAPACITIES["physical_warps"],
        "group_widths": [1, 2, 4, 8],
    }
    for field, value in expected.items():
        if not strict_equal(machine.get(field), value):
            raise ArtifactError(
                f"provenance.machine.{field} is not the B200 model value {value!r}"
            )
    regs_per_warp = _require_int(
        machine.get("regs_per_warp"),
        "provenance.machine.regs_per_warp",
        minimum=1,
    )
    if regs_per_warp > MAX_REGS_PER_WARP:
        raise ArtifactError(
            "provenance.machine.regs_per_warp exceeds the B200 limit "
            f"{MAX_REGS_PER_WARP}"
        )
    capacities = machine.get("capacities")
    if not strict_equal(
        capacities,
        {name: 1 for name in ("TC", "TMA", "TMEM", "SFU", "CUDA")},
    ):
        raise ArtifactError("provenance.machine.capacities is not the B200 model")
    for field in (
        "regs_fixed_overhead",
        "smem_fixed_overhead",
        "tmem_fixed_cols",
        "warp_fixed_overhead",
    ):
        _require_int(machine.get(field), f"provenance.machine.{field}")


def validate_pipelined_ir(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != PIPELINED_IR_SCHEMA:
        raise ArtifactError(f"input is not {PIPELINED_IR_SCHEMA}")
    source_ops = _validate_source_program(payload)
    pipeline = _validate_pipeline(payload)
    instructions = _validate_instructions(payload, pipeline, source_ops)
    _validate_warps(payload, instructions)
    dependencies = _validate_dependencies(payload, instructions, pipeline)
    _validate_pipelined_program(payload, instructions, dependencies, pipeline)
    source = payload["source_program"]
    _validate_provenance(payload, source["sha256"])
    _require_sha256(payload.get("solution_sha256"), "solution_sha256")


def load_pipelined_ir(source: ArtifactSource) -> LoadedArtifact:
    artifact = load_json_artifact(source, "pipelined IR")
    validate_pipelined_ir(artifact.payload)
    return artifact


def load_handoff(source: ArtifactSource, ir: LoadedArtifact) -> LoadedArtifact:
    artifact = load_json_artifact(source, "manual CUDA handoff")
    payload = artifact.payload
    if payload.get("schema_version") != HANDOFF_SCHEMA:
        raise ArtifactError(f"handoff is not {HANDOFF_SCHEMA}")
    reference = _require_object(payload.get("pipelined_ir"), "handoff.pipelined_ir")
    if reference.get("schema_version") != PIPELINED_IR_SCHEMA:
        raise ArtifactError("handoff references the wrong pipelined IR schema")
    if reference.get("sha256") != ir.sha256:
        raise ArtifactError("handoff pipelined IR sha256 mismatch")
    if payload.get("solution_sha256") != ir.payload["solution_sha256"]:
        raise ArtifactError("handoff solution sha256 mismatch")
    lowering = _require_object(payload.get("lowering"), "handoff.lowering")
    if lowering.get("paper_endpoint") != "expert-manual-cuda-cpp":
        raise ArtifactError("handoff does not target expert manual CUDA C++")
    if lowering.get("status") != "not_performed":
        raise ArtifactError("handoff lowering status must be not_performed")
    if lowering.get("executable_generated") is not False:
        raise ArtifactError("handoff must not claim an executable was generated")
    expected_steps = {
        "memory_allocation",
        "data_layout_conversion",
        "synchronization_placement",
        "instruction_selection",
    }
    steps = _require_list(
        lowering.get("remaining_manual_steps"), "handoff remaining_manual_steps"
    )
    if set(steps) != expected_steps:
        raise ArtifactError("handoff remaining_manual_steps are incomplete")
    return artifact


def source_reference(ir: LoadedArtifact) -> dict[str, str]:
    return {"schema_version": PIPELINED_IR_SCHEMA, "sha256": ir.sha256}


def target_reference() -> dict[str, str]:
    return {"gpu": TARGET_GPU, "cuda_arch": TARGET_ARCH}


def validate_plan_header(
    payload: dict[str, Any], schema: str, ir: LoadedArtifact
) -> None:
    if payload.get("schema_version") != schema:
        raise AuditError(f"plan is not {schema}")
    if payload.get("authoring_mode") != AUTHORING_MODE:
        raise AuditError(f"plan authoring_mode must be {AUTHORING_MODE}")
    if payload.get("target") != target_reference():
        raise AuditError("plan target must be B200/sm_100a")
    reference = payload.get("pipelined_ir")
    if not isinstance(reference, dict):
        raise AuditError("plan pipelined_ir reference must be an object")
    if reference.get("schema_version") != PIPELINED_IR_SCHEMA:
        raise AuditError("plan references the wrong pipelined IR schema")
    if reference.get("sha256") != ir.sha256:
        raise AuditError("plan pipelined IR sha256 mismatch")


def instruction_map(ir: LoadedArtifact) -> dict[int, dict[str, Any]]:
    return {
        instruction["id"]: instruction for instruction in ir.payload["instructions"]
    }


def issue_ordinals(ir: LoadedArtifact) -> dict[int, dict[str, int]]:
    """Issue position of every node in each of its physical warps."""
    result: dict[int, dict[str, int]] = {}
    for warp in ir.payload["warps"]:
        for ordinal, entry in enumerate(warp["issue_trace"]):
            result.setdefault(entry["node"], {})[str(warp["id"])] = ordinal
    return result


def dependency_map(ir: LoadedArtifact) -> dict[int, dict[str, Any]]:
    return {edge["index"]: edge for edge in ir.payload["dependencies"]}


def cross_dependency_map(ir: LoadedArtifact) -> dict[int, dict[str, Any]]:
    return {edge["index"]: edge for edge in ir.payload["cross_warp_dependencies"]}


def async_obligation_kind(op_kind: str) -> str | None:
    return ASYNC_COMPLETION_KINDS.get(op_kind)
