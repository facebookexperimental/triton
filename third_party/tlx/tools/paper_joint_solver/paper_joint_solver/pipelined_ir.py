"""Paper-faithful handoff from a Twill solution to downstream code generation.

Twill emits software-pipelined IR annotated with the warp or warps assigned to
each instruction.  The paper's evaluated kernels were then hand-compiled from
that IR to CUDA C++; the remaining CUDA lowering is deliberately not performed
by this module.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .schedule_plan import ScheduleArtifactError, SchedulePlan, load_schedule_plan


PIPELINED_IR_SCHEMA = "twill-pipelined-warp-ir-v1"
MANUAL_CUDA_HANDOFF_SCHEMA = "twill-manual-cuda-handoff-v1"


@dataclass(frozen=True)
class HandoffResult:
    ir_path: Path
    manifest_path: Path
    ir_sha256: str


def _pipeline_regions(plan: SchedulePlan) -> list[dict]:
    if plan.length < plan.ii:
        raise ScheduleArtifactError(
            f"schedule length {plan.length} is smaller than II {plan.ii}"
        )
    prologue_end = (plan.copies - 1) * plan.ii
    steady_end = prologue_end + plan.ii
    if steady_end > plan.horizon:
        raise ScheduleArtifactError(
            "pipeline steady state extends beyond the validated horizon"
        )
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


def _build_ir(plan: SchedulePlan, baseline_graph_path: str | Path) -> dict:
    source_graph = json.loads(Path(baseline_graph_path).read_text())
    schedule = plan.manifest()
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
            "regions": _pipeline_regions(plan),
        },
        "warp_groups": schedule["groups"],
        "instructions": schedule["nodes"],
        "dependencies": schedule["edges"],
        "cross_warp_dependencies": [
            edge
            for edge in schedule["edges"]
            if edge["producer_group"] is not None
            and edge["consumer_group"] is not None
            and edge["producer_group"] != edge["consumer_group"]
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
    """Emit Twill IR and an explicit handoff contract for manual CUDA lowering."""
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
