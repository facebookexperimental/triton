"""Paper-faithful handoff from a joint solution to downstream code generation.

The paper emits software-pipelined IR annotated with the warp or warps
assigned to each instruction.  Its evaluated kernels were then hand-compiled
from that IR to CUDA C++; the remaining CUDA lowering is deliberately not
performed by this module.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .schedule_plan import ScheduleArtifactError, SchedulePlan, load_schedule_plan


PIPELINED_IR_SCHEMA = "paper-joint-pipelined-ir-v3"
MANUAL_CUDA_HANDOFF_SCHEMA = "paper-joint-handoff-v2"


@dataclass(frozen=True)
class HandoffResult:
    ir_path: Path
    manifest_path: Path
    ir_sha256: str


def _pipeline_regions(plan: SchedulePlan) -> list[dict]:
    # The paper's region formulas (p.4) carry no L >= I precondition, so they
    # are applied verbatim.  When L < I the prologue is empty, the steady
    # state is [0, I) and the epilogue is an inverted -- hence empty --
    # interval that no cycle can fall into (I-4).
    prologue_end = (plan.copies - 1) * plan.ii
    steady_end = prologue_end + plan.ii
    return [
        {"kind": "prologue", "cycle_begin": 0, "cycle_end": prologue_end},
        {
            "kind": "steady_state",
            "cycle_begin": prologue_end,
            "cycle_end": steady_end,
        },
        {
            "kind": "epilogue",
            "cycle_begin": steady_end,
            "cycle_end": plan.horizon,
        },
    ]


def _region_for_cycle(cycle: int, regions: list[dict]) -> str:
    for region in regions:
        if region["cycle_begin"] <= cycle < region["cycle_end"]:
            return region["kind"]
    raise ScheduleArtifactError(f"expanded cycle {cycle} is outside pipeline regions")


def _expanded_instances(plan: SchedulePlan, regions: list[dict]) -> list[dict]:
    result = []
    for node in sorted(plan.nodes, key=lambda item: item.node_id):
        for copy in range(plan.copies):
            cycle = node.cycle + copy * plan.ii
            result.append(
                {
                    "node": node.node_id,
                    "copy": copy,
                    "cycle": cycle,
                    "region": _region_for_cycle(cycle, regions),
                }
            )
    return result


def _expanded_dependencies(plan: SchedulePlan) -> list[dict]:
    result = []
    for edge in sorted(plan.edges, key=lambda item: item.index):
        for consumer_copy in range(plan.copies):
            producer_copy = consumer_copy - edge.distance
            result.append(
                {
                    "edge_index": edge.index,
                    "producer": {"node": edge.src, "copy": producer_copy},
                    "consumer": {"node": edge.dst, "copy": consumer_copy},
                    "external": producer_copy < 0,
                    "carried_distance": edge.distance,
                }
            )
    return result


def _steady_state_slots(plan: SchedulePlan, regions: list[dict]) -> list[dict]:
    result = []
    for node in sorted(plan.nodes, key=lambda item: item.node_id):
        copy = (plan.copies - 1) - node.stage
        cycle = node.cycle + copy * plan.ii
        iteration_lag = (plan.copies - 1) - copy
        if (
            not 0 <= copy < plan.copies
            or _region_for_cycle(cycle, regions) != "steady_state"
            or iteration_lag != node.stage
        ):
            raise ScheduleArtifactError(
                f"node {node.node_id} has no canonical steady-state instance"
            )
        result.append(
            {
                "node": node.node_id,
                "copy": copy,
                "cycle": cycle,
                "iteration_lag": iteration_lag,
            }
        )
    return result


def _build_pipelined_program(plan: SchedulePlan, regions: list[dict]) -> dict:
    return {
        "instances": _expanded_instances(plan, regions),
        "instance_dependencies": _expanded_dependencies(plan),
        "steady_state": {"slots": _steady_state_slots(plan, regions)},
    }


def _is_cross_warp_dependency(edge: dict) -> bool:
    # Overlapping but unequal warp sets still cross a warp boundary, which is
    # the same reading the spill cost uses (I-1).
    return edge["ws_semantics"] and (
        edge["producer_warps"] != edge["consumer_warps"]
    )


def _build_ir(plan: SchedulePlan, baseline_graph_path: str | Path) -> dict:
    source_graph = json.loads(Path(baseline_graph_path).read_text())
    schedule = plan.manifest()
    regions = _pipeline_regions(plan)
    return {
        "schema_version": PIPELINED_IR_SCHEMA,
        "source_program": {
            "format": "ttgir-derived-operation-table",
            "sha256": plan.baseline_graph_sha256,
            "input_schema_version": source_graph.get("schema_version"),
            "kernel": source_graph.get("kernel"),
            "ops": source_graph.get("ops", {}),
            "data_partition_candidates": source_graph.get(
                "data_partition_candidates", []
            ),
        },
        "pipeline": {
            "ii": plan.ii,
            "length": plan.length,
            "copies": plan.copies,
            "horizon": plan.horizon,
            "regions": regions,
        },
        "pipelined_program": _build_pipelined_program(plan, regions),
        "warps": schedule["warps"],
        "instructions": schedule["nodes"],
        "dependencies": schedule["edges"],
        "cross_warp_dependencies": [
            edge
            for edge in schedule["edges"]
            if _is_cross_warp_dependency(edge)
        ],
        "provenance": schedule["provenance"],
        "solution_sha256": plan.solution_sha256,
    }


def prepare_manual_cuda_handoff(
    *,
    solution_path: str | Path,
    ddg_path: str | Path,
    baseline_graph_path: str | Path,
    ir_out: str | Path,
    manifest_out: str | Path,
) -> HandoffResult:
    """Emit the paper's IR and an explicit contract for manual CUDA lowering."""
    plan = load_schedule_plan(solution_path, ddg_path, baseline_graph_path)
    ir_out = Path(ir_out)
    manifest_out = Path(manifest_out)
    ir_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    ir_text = json.dumps(_build_ir(plan, baseline_graph_path), indent=2) + "\n"
    ir_sha256 = hashlib.sha256(ir_text.encode()).hexdigest()
    manifest = {
        "schema_version": MANUAL_CUDA_HANDOFF_SCHEMA,
        "pipelined_ir": {
            "schema_version": PIPELINED_IR_SCHEMA,
            "sha256": ir_sha256,
        },
        "solution_sha256": plan.solution_sha256,
        "lowering": {
            "paper_endpoint": "expert-manual-cuda-cpp",
            "status": "not_performed",
            "executable_generated": False,
            "remaining_manual_steps": [
                "memory_allocation",
                "data_layout_conversion",
                "synchronization_placement",
                "instruction_selection",
            ],
        },
    }

    with tempfile.TemporaryDirectory(dir=ir_out.parent) as tmpdir:
        tmp = Path(tmpdir)
        tmp_ir = tmp / "pipelined_ir.json"
        tmp_manifest = tmp / "handoff_manifest.json"
        tmp_ir.write_text(ir_text)
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp_ir.replace(ir_out)
        tmp_manifest.replace(manifest_out)

    return HandoffResult(
        ir_path=ir_out,
        manifest_path=manifest_out,
        ir_sha256=ir_sha256,
    )
