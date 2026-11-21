"""Deterministic, deliberately non-executable manual-CUDA scaffolding."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ._schema import (
    ArtifactError,
    ArtifactSource,
    async_obligation_kind,
    AUTHORING_MODE,
    AUTHORING_SCHEMA,
    B200_CAPACITIES,
    HANDOFF_SCHEMA,
    instruction_map,
    issue_ordinals,
    load_handoff,
    load_pipelined_ir,
    MAPPING_SCHEMA,
    MEMORY_PLAN_SCHEMA,
    source_reference,
    SYNC_PLAN_SCHEMA,
    target_reference,
)


@dataclass(frozen=True)
class ScaffoldResult:
    cuda_path: Path
    authoring_path: Path
    mapping_path: Path
    memory_plan_path: Path
    sync_plan_path: Path
    ir_sha256: str


def _header(schema: str, ir) -> dict:
    return {
        "schema_version": schema,
        "authoring_mode": AUTHORING_MODE,
        "target": target_reference(),
        "pipelined_ir": source_reference(ir),
        "status": "manual_completion_required",
    }


def _mapping_template(ir) -> dict:
    instructions = instruction_map(ir)
    ordinals = issue_ordinals(ir)
    program = ir.payload["pipelined_program"]
    instances_by_node = {
        node_id: [
            instance
            for instance in program["instances"]
            if instance["node"] == node_id
        ]
        for node_id in instructions
    }
    slots_by_node = {
        slot["node"]: slot for slot in program["steady_state"]["slots"]
    }
    relations_by_edge = {
        edge["index"]: [
            relation
            for relation in program["instance_dependencies"]
            if relation["edge_index"] == edge["index"]
        ]
        for edge in ir.payload["dependencies"]
    }
    payload = _header(MAPPING_SCHEMA, ir)
    payload.update(
        {
            "cuda_source": {"path": "kernel.cu", "sha256": None},
            "build_evidence": {
                "status": "pending",
                "command": None,
                "toolchain": None,
                "cuda_arch": "sm_100a",
                "source_sha256": None,
                "binary": {"path": None, "sha256": None},
                "disassembly": {"path": None, "sha256": None},
                "reviewer": None,
            },
            # The paper's IR names physical warps directly, so there is no
            # logical group left for the author to bind.
            "physical_warps": [
                {
                    "warp": warp["id"],
                    "thread_mapping": None,
                    "dispatch_anchor": None,
                    "entry_predicate": None,
                    "regions": None,
                    "reviewer": None,
                }
                for warp in ir.payload["warps"]
            ],
            "regions": [
                {
                    **region,
                    "source_anchor": None,
                    "realization": None,
                    "reviewer": None,
                }
                for region in ir.payload["pipeline"]["regions"]
            ],
            "nodes": [
                {
                    "instruction_id": node_id,
                    "op_ref": instruction["op_ref"],
                    "op_kind": instruction["op_kind"],
                    "pipeline": instruction["pipeline"],
                    "cycle": instruction["cycle"],
                    "stage": instruction["stage"],
                    "offset": instruction["offset"],
                    "warps": instruction["warps"],
                    "issue_order": ordinals[node_id],
                    "instances": instances_by_node[node_id],
                    "steady_state_slot": slots_by_node[node_id],
                    "manual_cuda": {
                        "source_anchor": None,
                        "disassembly_anchor": None,
                        "machine_realization": None,
                        "equivalence_proof": None,
                        "physical_warps": None,
                        "lane_predicate": None,
                        "iteration_expression": None,
                        "region_realization": None,
                        "stage_slot_expression": None,
                        "semantic_lowering": None,
                        "primitive": None,
                        "operand_allocations": [],
                        "result_allocations": [],
                        "sync_channels": [],
                        "predicate": None,
                        "semantic_completion": None,
                        "source_status": "pending",
                        "disassembly_status": "pending",
                        "reviewer": None,
                    },
                }
                for node_id, instruction in sorted(instructions.items())
            ],
            "dependencies": [
                {
                    **edge,
                    "instance_dependencies": relations_by_edge[edge["index"]],
                    "dynamic_iteration_relation": None,
                    "transport": None,
                    "program_order_proof": None,
                    "sync_channel": None,
                    "completion_witness": None,
                    "prologue_availability": None,
                    "epilogue_drain": None,
                    "reviewer": None,
                }
                for edge in ir.payload["dependencies"]
            ],
            "coverage": {
                "ir_node_count": len(instructions),
                "mapped_node_count": 0,
                "ir_edge_count": len(ir.payload["dependencies"]),
                "mapped_edge_count": 0,
                "ir_instance_count": len(program["instances"]),
                "mapped_instance_count": 0,
                "ir_instance_dependency_count": len(
                    program["instance_dependencies"]
                ),
                "mapped_instance_dependency_count": 0,
                "ir_steady_state_slot_count": len(
                    program["steady_state"]["slots"]
                ),
                "mapped_steady_state_slot_count": 0,
            },
            "deviations": [],
            "review": {"status": "pending", "reviewer": None},
        }
    )
    return payload


def _memory_template(ir) -> dict:
    payload = _header(MEMORY_PLAN_SCHEMA, ir)
    payload.update(
        {
            "capacities": B200_CAPACITIES,
            "allocations": [],
            "alias_sets": [],
            "descriptors": [],
            "resource_budget": {
                "planned": {
                    "register_words": None,
                    "per_warp_register_words": None,
                    "dynamic_smem_bytes": None,
                    "barrier_smem_bytes": None,
                    "tmem_columns": None,
                    "physical_warps": len(ir.payload["warps"]),
                },
                "compiler_reported": {
                    "register_words": None,
                    "dynamic_smem_bytes": None,
                    "spill_bytes": None,
                },
            },
            "lifetime_audit": [],
            "review": {"status": "pending", "reviewer": None},
        }
    )
    return payload


def _async_obligations(ir) -> list[dict]:
    result = []
    for instruction in ir.payload["instructions"]:
        kind = async_obligation_kind(instruction["op_kind"])
        if kind is not None:
            result.append(
                {
                    "id": f"node-{instruction['id']}-{kind}",
                    "kind": kind,
                    "instruction_id": instruction["id"],
                    "wait_kind": None,
                    "wait_node": None,
                    "distance": None,
                    "kernel_exit_wait": None,
                    "channel": None,
                    "completion_witness": None,
                    "predicate": None,
                    "epilogue_drain": None,
                    "reviewer": None,
                }
            )
    return result


def _sync_template(ir) -> dict:
    async_obligations = _async_obligations(ir)
    instance_dependencies = ir.payload["pipelined_program"][
        "instance_dependencies"
    ]
    payload = _header(SYNC_PLAN_SCHEMA, ir)
    payload.update(
        {
            "channels": [],
            "dependencies": [
                {
                    "edge_index": edge["index"],
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "distance": edge["distance"],
                    "latency": edge["latency"],
                    "producer_warps": edge["producer_warps"],
                    "consumer_warps": edge["consumer_warps"],
                    "ws_semantics": edge["ws_semantics"],
                    "spill_cost": edge["spill_cost"],
                    "instance_dependencies": [
                        relation
                        for relation in instance_dependencies
                        if relation["edge_index"] == edge["index"]
                    ],
                    "mechanism": None,
                    "channel": None,
                    "program_order_proof": None,
                    "completion_witness": None,
                    "consumer_wait": None,
                    "predicate": None,
                    "prologue": None,
                    "epilogue": None,
                    "reviewer": None,
                }
                for edge in ir.payload["dependencies"]
            ],
            "additional_obligations": async_obligations,
            "ordering_edges": [],
            "phase_traces": [],
            "coverage": {
                "ir_edges": len(ir.payload["dependencies"]),
                "mapped_ir_edges": 0,
                "cross_warp_edges": len(ir.payload["cross_warp_dependencies"]),
                "mapped_cross_warp_edges": 0,
                "ir_instance_dependencies": len(instance_dependencies),
                "mapped_instance_dependencies": 0,
                "async_completion_obligations": len(async_obligations),
                "mapped_async_completion_obligations": 0,
                "reuse_obligations": None,
                "mapped_reuse_obligations": 0,
            },
            "review": {"status": "pending", "reviewer": None},
        }
    )
    return payload


def _cuda_skeleton(ir) -> str:
    lines = [
        "/*",
        " * Deterministic SKC scaffold for expert manual CUDA lowering.",
        " * This file intentionally contains no executable kernel.",
        " * Complete and review the manifests before authoring CUDA.",
        " */",
        '#error "Manual CUDA lowering is required; this scaffold is not executable"',
        "",
        f"// pipelined IR sha256: {ir.sha256}",
        "// Scheduled instructions (schedule facts only):",
    ]
    for instruction in sorted(ir.payload["instructions"], key=lambda item: item["id"]):
        facts = {
            "id": instruction["id"],
            "op_ref": instruction["op_ref"],
            "op_kind": instruction["op_kind"],
            "warps": instruction["warps"],
            "stage": instruction["stage"],
            "offset": instruction["offset"],
        }
        lines.append(f"// {json.dumps(facts, sort_keys=True)}")
    lines.extend(
        [
            "",
            "// MANUAL: select CUDA instructions and data layouts.",
            "// MANUAL: implement allocations and synchronization from reviewed plans.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _authoring_template(ir, handoff, manifest_text: dict[str, str]) -> dict:
    return {
        "schema_version": AUTHORING_SCHEMA,
        "authoring_mode": AUTHORING_MODE,
        "target": target_reference(),
        "status": "manual_completion_required",
        "pipelined_ir": source_reference(ir),
        "handoff": {"schema_version": HANDOFF_SCHEMA, "sha256": handoff.sha256},
        "solution_sha256": ir.payload["solution_sha256"],
        "source_program_sha256": ir.payload["source_program"]["sha256"],
        "solver_provenance": ir.payload["provenance"],
        "clean_room": {
            "fa4_fa3_use": "black-box-baselines-only",
            "prohibited_reuse": ["warp_roles", "mma_order", "barrier_protocol"],
            "reviewed_leaf_wrappers": [],
            "declaration_reviewed_by": None,
        },
        "manifests": {
            "mapping": {
                "path": "mapping_manifest.json",
                "sha256": _sha256_text(manifest_text["mapping_manifest.json"]),
            },
            "memory": {
                "path": "memory_plan.json",
                "sha256": _sha256_text(manifest_text["memory_plan.json"]),
            },
            "synchronization": {
                "path": "sync_manifest.json",
                "sha256": _sha256_text(manifest_text["sync_manifest.json"]),
            },
        },
        "human_reviews": [],
    }


def _write_scaffold(output_dir: Path, contents: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(name for name in contents if (output_dir / name).exists())
    if existing:
        raise ArtifactError(
            "refusing to overwrite existing manual-CUDA artifacts: "
            + ", ".join(existing)
        )
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        for name, content in contents.items():
            (temporary_dir / name).write_text(content, encoding="utf-8")
        for name in contents:
            (temporary_dir / name).replace(output_dir / name)


def scaffold_manual_cuda(
    ir_path: ArtifactSource,
    handoff_path: ArtifactSource,
    output_dir: str | Path,
) -> ScaffoldResult:
    """Emit deterministic templates after validating the complete handoff chain."""
    ir = load_pipelined_ir(ir_path)
    handoff = load_handoff(handoff_path, ir)
    output_dir = Path(output_dir)
    manifests = {
        "mapping_manifest.json": _json_text(_mapping_template(ir)),
        "memory_plan.json": _json_text(_memory_template(ir)),
        "sync_manifest.json": _json_text(_sync_template(ir)),
    }
    authoring = _json_text(_authoring_template(ir, handoff, manifests))
    contents = {
        "kernel.cu": _cuda_skeleton(ir),
        "manual_cuda_authoring.json": authoring,
        **manifests,
    }
    _write_scaffold(output_dir, contents)
    return ScaffoldResult(
        cuda_path=output_dir / "kernel.cu",
        authoring_path=output_dir / "manual_cuda_authoring.json",
        mapping_path=output_dir / "mapping_manifest.json",
        memory_plan_path=output_dir / "memory_plan.json",
        sync_plan_path=output_dir / "sync_manifest.json",
        ir_sha256=ir.sha256,
    )
