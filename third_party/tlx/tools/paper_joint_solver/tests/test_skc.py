import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.pipelined_ir import (
    MANUAL_CUDA_HANDOFF_SCHEMA as HANDOFF_SCHEMA,
    PIPELINED_IR_SCHEMA,
    _build_pipelined_program,
)
from paper_joint_solver.schedule_plan import solver_sources_sha256
from skc import (
    ArtifactError,
    audit_authoring,
    audit_bundle,
    audit_mapping,
    audit_memory,
    audit_sync,
    AuditError,
    scaffold_manual_cuda,
)
from skc._schema import (
    MEMORY_PLAN_SCHEMA,
    PIPELINED_IR_SCHEMA as VALIDATED_PIPELINED_IR_SCHEMA,
    SYNC_PLAN_SCHEMA,
    validate_pipelined_ir,
)
from skc.__main__ import main

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = solver_sources_sha256()
_SHA_D = "d" * 64


def _instruction(node_id, op_kind, warps, cycle):
    return {
        "id": node_id,
        "op_ref": f"op{node_id}",
        "op_kind": op_kind,
        "pipeline": "TMA" if op_kind == "tt.descriptor_load" else "CUDA",
        "cycle": cycle,
        "stage": cycle // 4,
        "offset": cycle % 4,
        "warps": warps,
    }


def _edge(
    index,
    producer,
    consumer,
    distance=0,
    latency=1,
    spill_cost=0,
    ws_semantics=True,
):
    return {
        "index": index,
        "src": producer["id"],
        "dst": consumer["id"],
        "distance": distance,
        "latency": latency,
        "producer_warps": producer["warps"],
        "consumer_warps": consumer["warps"],
        "ws_semantics": ws_semantics,
        "spill_cost": spill_cost,
    }


def _region_for_cycle(cycle, regions):
    return next(
        region["kind"]
        for region in regions
        if region["cycle_begin"] <= cycle < region["cycle_end"]
    )


def _pipelined_program(instructions, dependencies, pipeline):
    ii = pipeline["ii"]
    copies = pipeline["copies"]
    regions = pipeline["regions"]
    nodes = sorted(instructions, key=lambda node: node["id"])
    edges = sorted(dependencies, key=lambda edge: edge["index"])
    instances = [
        {
            "node": node["id"],
            "copy": copy,
            "cycle": cycle,
            "region": _region_for_cycle(cycle, regions),
        }
        for node in nodes
        for copy in range(copies)
        for cycle in (node["cycle"] + copy * ii,)
    ]
    instance_dependencies = []
    for edge in edges:
        for consumer_copy in range(copies):
            producer_copy = consumer_copy - edge["distance"]
            instance_dependencies.append(
                {
                    "edge_index": edge["index"],
                    "producer": {"node": edge["src"], "copy": producer_copy},
                    "consumer": {"node": edge["dst"], "copy": consumer_copy},
                    "external": producer_copy < 0,
                    "carried_distance": edge["distance"],
                }
            )
    slots = []
    for node in nodes:
        copy = (copies - 1) - node["stage"]
        slots.append(
            {
                "node": node["id"],
                "copy": copy,
                "cycle": node["cycle"] + copy * ii,
                "iteration_lag": node["stage"],
            }
        )
    return {
        "instances": instances,
        "instance_dependencies": instance_dependencies,
        "steady_state": {"slots": slots},
    }


def _is_cross_warp_dependency(edge):
    return edge["ws_semantics"] and (
        edge["producer_warps"] != edge["consumer_warps"]
    )


def _machine():
    return {
        "capacities": {name: 1 for name in ("TC", "TMA", "TMEM", "SFU", "CUDA")},
        "smem_bytes": 232448,
        "tmem_cols": 512,
        "regs_per_sm": 65536,
        "regs_per_warp": 8160,
        "num_warps": 32,
        "num_warpgroups": 32,
        "group_widths": [1, 2, 4, 8],
        "regs_fixed_overhead": 0,
        "smem_fixed_overhead": 0,
        "tmem_fixed_cols": 0,
        "warp_fixed_overhead": 0,
    }


def _ir():
    instructions = [
        _instruction(0, "tt.descriptor_load", [0], 0),
        _instruction(1, "arith.addf", [1, 2], 1),
        _instruction(2, "arith.mulf", [1, 2], 4),
    ]
    dependencies = [
        _edge(10, instructions[0], instructions[1]),
        _edge(11, instructions[1], instructions[2], distance=1),
    ]
    pipeline = {
        "ii": 4,
        "length": 5,
        "copies": 2,
        "horizon": 9,
        "regions": [
            {"kind": "prologue", "cycle_begin": 0, "cycle_end": 4},
            {"kind": "steady_state", "cycle_begin": 4, "cycle_end": 8},
            {"kind": "epilogue", "cycle_begin": 8, "cycle_end": 9},
        ],
    }
    return {
        "schema_version": PIPELINED_IR_SCHEMA,
        "pipeline": pipeline,
        "pipelined_program": _pipelined_program(
            instructions, dependencies, pipeline
        ),
        "warps": [
            {
                "id": 0,
                "issue_trace": [
                    {"node": 0, "cycle": 0, "stage": 0, "offset": 0}
                ],
            },
            {
                "id": 1,
                "issue_trace": [
                    {"node": 1, "cycle": 1, "stage": 0, "offset": 1},
                    {"node": 2, "cycle": 4, "stage": 1, "offset": 0},
                ],
            },
            {
                "id": 2,
                "issue_trace": [
                    {"node": 1, "cycle": 1, "stage": 0, "offset": 1},
                    {"node": 2, "cycle": 4, "stage": 1, "offset": 0},
                ],
            },
        ],
        "instructions": instructions,
        "dependencies": dependencies,
        "cross_warp_dependencies": [dependencies[0]],
        "source_program": {
            "format": "ttgir-derived-operation-table",
            "sha256": _SHA_B,
            "input_schema_version": "0.1",
            "kernel": {
                "name": "test_kernel",
                "args": [{"name": "Q", "type": "*f16"}],
            },
            "ops": {
                "op0": {
                    "kind": "tt.descriptor_load",
                    "scope": "loop",
                    "operands": [{"arg": "Q"}],
                    "result_types": ["tensor<32xf16>"],
                    "attributes": {},
                },
                "op1": {
                    "kind": "arith.addf",
                    "scope": "loop",
                    "operands": [{"op": "op0"}, {"op": "op0"}],
                    "result_types": ["tensor<32xf16>"],
                    "attributes": {},
                },
                "op2": {
                    "kind": "arith.mulf",
                    "scope": "loop",
                    "operands": [{"op": "op1"}, {"op": "op1"}],
                    "result_types": ["tensor<32xf16>"],
                    "attributes": {},
                },
            },
            "data_partition_candidates": [],
        },
        "provenance": {
            "model": "paper",
            "ddg_sha256": _SHA_A,
            "curation_source": {
                "ddg_sha256": _SHA_D,
                "baseline_graph_sha256": _SHA_B,
                "curator_sources_sha256": _SHA_A,
            },
            "solver_sources_sha256": _SHA_C,
            "normalization_u": 300,
            "experiment_inputs": {
                "allow_cross_warp": True,
                "num_warps_override": None,
                "reg_budget": None,
            },
            "machine": _machine(),
        },
        "solution_sha256": _SHA_A,
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load(path):
    return json.loads(path.read_text())


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path, payload=None):
    ir_path = tmp_path / "pipelined_ir.json"
    _write_json(ir_path, _ir() if payload is None else payload)
    handoff_path = tmp_path / "handoff.json"
    _write_json(
        handoff_path,
        {
            "schema_version": HANDOFF_SCHEMA,
            "pipelined_ir": {
                "schema_version": PIPELINED_IR_SCHEMA,
                "sha256": _sha256(ir_path),
            },
            "solution_sha256": _SHA_A,
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
        },
    )
    return ir_path, handoff_path


def _scaffold(tmp_path, payload=None):
    ir_path, handoff_path = _write_inputs(tmp_path, payload)
    result = scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "manual_cuda")
    return ir_path, handoff_path, result


def _source_text():
    anchors = [
        "PAPER_WARP_0",
        "PAPER_WARP_1",
        "PAPER_WARP_2",
        "PAPER_REGION_PROLOGUE",
        "PAPER_REGION_STEADY",
        "PAPER_REGION_EPILOGUE",
        "PAPER_NODE_0",
        "PAPER_NODE_1",
        "PAPER_NODE_2",
    ]
    body = "\n".join(f"  // {anchor}" for anchor in anchors)
    return f'extern "C" __global__ void test_kernel() {{\n{body}\n}}\n'


def _complete_mapping(result):
    mapping = _load(result.mapping_path)
    mapping["status"] = "approved"
    result.cuda_path.write_text(_source_text())
    mapping["cuda_source"]["sha256"] = _sha256(result.cuda_path)
    binary_path = result.cuda_path.parent / "kernel.cubin"
    cubin_header = bytearray(64)
    cubin_header[:4] = b"\x7fELF"
    cubin_header[4] = 2
    cubin_header[5] = 1
    cubin_header[18:20] = (190).to_bytes(2, "little")
    binary_path.write_bytes(cubin_header)
    disassembly_path = result.cuda_path.parent / "kernel.sass"
    disassembly_path.write_text(
        "\n".join(f"SASS_NODE_{node_id} REVIEWED_OP" for node_id in range(3)) + "\n"
    )
    mapping["build_evidence"] = {
        "status": "verified",
        "command": "reviewed B200 build command",
        "toolchain": "reviewed CUDA toolchain",
        "cuda_arch": "sm_100a",
        "source_sha256": _sha256(result.cuda_path),
        "binary": {"path": "kernel.cubin", "sha256": _sha256(binary_path)},
        "disassembly": {
            "path": "kernel.sass",
            "sha256": _sha256(disassembly_path),
        },
        "reviewer": "mapping-reviewer",
    }
    for entry in mapping["physical_warps"]:
        warp_id = entry["warp"]
        entry.update(
            {
                "thread_mapping": "local warp index maps directly to physical warp",
                "dispatch_anchor": f"PAPER_WARP_{warp_id}",
                "entry_predicate": f"physical_warp == {warp_id}",
                "regions": "prologue, steady state, and epilogue",
                "reviewer": "mapping-reviewer",
            }
        )
    region_anchors = (
        "PAPER_REGION_PROLOGUE",
        "PAPER_REGION_STEADY",
        "PAPER_REGION_EPILOGUE",
    )
    for region, anchor in zip(mapping["regions"], region_anchors):
        region.update(
            {
                "source_anchor": anchor,
                "realization": "symbolically expanded pipeline region",
                "reviewer": "mapping-reviewer",
            }
        )
    for node in mapping["nodes"]:
        node_id = node["instruction_id"]
        manual = node["manual_cuda"]
        manual.update(
            {
                "source_anchor": f"PAPER_NODE_{node_id}",
                "disassembly_anchor": f"SASS_NODE_{node_id}",
                "machine_realization": "instruction",
                "equivalence_proof": None,
                "physical_warps": list(node["warps"]),
                "lane_predicate": "all mapped lanes participate",
                "iteration_expression": "logical_iteration",
                "region_realization": "expanded across all pipeline regions",
                "stage_slot_expression": "logical_iteration modulo two",
                "semantic_lowering": "reviewed source operation lowering",
                "primitive": "REVIEWED_OP",
                "operand_allocations": (
                    ["input-tensor-map", "tile"] if node_id == 0 else ["tile"]
                ),
                "result_allocations": ["tile"],
                "sync_channels": ["ready"] if node_id in (0, 1) else [],
                "predicate": "uniform supported-shape predicate",
                "semantic_completion": "completion at the mapped source anchor",
                "source_status": "verified",
                "disassembly_status": "verified",
                "reviewer": "mapping-reviewer",
            }
        )
    for edge in mapping["dependencies"]:
        edge.update(
            {
                "dynamic_iteration_relation": "consumer iteration equals producer plus distance",
                "transport": "reviewed payload transport",
                "completion_witness": "reviewed completion witness",
                "prologue_availability": "producer exists before first consumer",
                "epilogue_drain": "last consumer completes before exit",
                "reviewer": "mapping-reviewer",
            }
        )
        if _is_cross_warp_dependency(edge):
            edge["sync_channel"] = "ready"
        else:
            edge["program_order_proof"] = "same physical warps and issue order"
    mapping["coverage"] = {
        "ir_node_count": 3,
        "mapped_node_count": 3,
        "ir_edge_count": 2,
        "mapped_edge_count": 2,
        "ir_instance_count": 6,
        "mapped_instance_count": 6,
        "ir_instance_dependency_count": 4,
        "mapped_instance_dependency_count": 4,
        "ir_steady_state_slot_count": 3,
        "mapped_steady_state_slot_count": 3,
    }
    mapping["review"] = {"status": "approved", "reviewer": "mapping-reviewer"}
    _write_json(result.mapping_path, mapping)


def _complete_memory(result):
    memory = _load(result.memory_plan_path)
    memory["status"] = "approved"
    memory["allocations"] = [
        {
            "id": "tile",
            "semantic_value": "test tile and descriptor storage",
            "storage_kind": "shared",
            "unit": "shared_bytes",
            "offset": 0,
            "size": 64,
            "slot_size": 32,
            "alignment": 16,
            "element_type": "f16",
            "logical_shape": "[32]",
            "physical_shape": "64 bytes",
            "layout": "linear test layout",
            "owner": "CTA groups 0 and 1",
            "slots": 2,
            "slot_expression": "logical_iteration modulo two",
            "phase_expression": "logical_iteration divided by two modulo two",
            "producer_nodes": [0, 1, 2],
            "consumer_nodes": [0, 1, 2],
            "lifetime": {
                "start": 0,
                "end": 9,
                "wraps_checked": 2,
                "first_write": "TMA launch at node 0",
                "publication": "ready channel completion",
                "last_completion": "node 2 completion",
                "release": "ready release event",
                "release_channel": "ready",
            },
            "alias_set": None,
            "source_anchor": "PAPER_NODE_0",
            "reviewer": "memory-reviewer",
        },
        {
            "id": "ready-barrier-storage",
            "semantic_value": "ready channel mbarrier",
            "storage_kind": "barrier",
            "unit": "shared_bytes",
            "offset": 64,
            "size": 32,
            "slot_size": 16,
            "alignment": 8,
            "element_type": "mbarrier",
            "logical_shape": "[1]",
            "physical_shape": "two 16-byte barrier slots",
            "layout": "full and empty barrier pair",
            "owner": "CTA",
            "slots": 2,
            "slot_expression": "logical_iteration modulo two",
            "phase_expression": "logical_iteration divided by two modulo two",
            "producer_nodes": [0],
            "consumer_nodes": [1, 2],
            "lifetime": {
                "start": 0,
                "end": 9,
                "wraps_checked": 2,
                "first_write": "uniform barrier initialization",
                "publication": "barrier initialization completion",
                "last_completion": "final consumer release",
                "release": "kernel exit after quiescence",
                "release_channel": "ready",
            },
            "alias_set": None,
            "source_anchor": "PAPER_WARP_0",
            "reviewer": "memory-reviewer",
        },
        {
            "id": "input-tensor-map",
            "semantic_value": "input TMA tensor map",
            "storage_kind": "descriptor",
            "unit": "shared_bytes",
            "offset": 96,
            "size": 64,
            "slot_size": 64,
            "alignment": 16,
            "element_type": "CUtensorMap",
            "logical_shape": "rank-two descriptor",
            "physical_shape": "64 bytes",
            "layout": "CUDA tensor-map encoding",
            "owner": "CTA",
            "slots": 1,
            "slot_expression": "zero",
            "phase_expression": "immutable",
            "producer_nodes": [],
            "consumer_nodes": [0],
            "lifetime": {
                "start": 0,
                "end": 9,
                "wraps_checked": 2,
                "first_write": "launcher descriptor initialization",
                "publication": "kernel argument visibility",
                "last_completion": "final TMA issue",
                "release": "kernel exit",
                "release_channel": None,
            },
            "alias_set": None,
            "source_anchor": "PAPER_NODE_0",
            "reviewer": "memory-reviewer",
        },
    ]
    memory["resource_budget"] = {
        "planned": {
            "register_words": 96,
            "per_warp_register_words": {"0": 32, "1": 32, "2": 32},
            "dynamic_smem_bytes": 160,
            "barrier_smem_bytes": 32,
            "tmem_columns": 0,
            "physical_warps": 3,
        },
        "compiler_reported": {
            "register_words": 96,
            "dynamic_smem_bytes": 160,
            "spill_bytes": 0,
        },
    }
    memory["descriptors"] = [
        {
            "id": "input-desc",
            "allocation_id": "input-tensor-map",
            "rank": 2,
            "element_type": "f16",
            "global_shape": "[1, 32]",
            "global_strides": "[32, 1]",
            "box_shape": "[1, 32]",
            "swizzle": "none",
            "interleave": "none",
            "source_anchor": "PAPER_NODE_0",
            "reviewer": "memory-reviewer",
        }
    ]
    memory["lifetime_audit"] = [
        {
            "allocation_id": allocation_id,
            "wraps_checked": 2,
            "prologue": "first slot initialized",
            "steady_state": "two complete wraps audited",
            "epilogue": "last use and release drained",
            "reviewer": "memory-reviewer",
        }
        for allocation_id in ("tile", "ready-barrier-storage", "input-tensor-map")
    ]
    memory["review"] = {"status": "approved", "reviewer": "memory-reviewer"}
    _write_json(result.memory_plan_path, memory)


def _complete_sync(result):
    sync = _load(result.sync_plan_path)
    sync["status"] = "approved"
    sync["channels"] = [
        {
            "id": "ready",
            "barrier_id": "ready-barrier",
            "allocation_id": "ready-barrier-storage",
            "payload_allocation": "tile",
            "scope": "cta",
            "slot_count": 2,
            "slot_expression": "logical_iteration modulo two",
            "producer_groups": [0, 1],
            "consumer_groups": [1],
            "producer_warps": [0, 1, 2],
            "consumer_warps": [1, 2],
            "initialization": {
                "actor": "warp 0",
                "arrival_count": 1,
                "arrival_actors": ["elected lane of warp 0"],
                "transaction_bytes": 64,
            },
            "producer_witness": "TMA transaction completion",
            "consumer_wait": {
                "primitive": "mbarrier try-wait parity",
                "phase_expression": "logical_iteration modulo two",
                "predicate": "uniform supported-shape predicate",
            },
            "release": {
                "primitive": "mbarrier arrive",
                "phase_expression": "logical_iteration modulo two",
                "predicate": "uniform supported-shape predicate",
            },
            "initial_phase": 0,
            "phase_recurrence": "phase toggles after every completed iteration",
            "prologue": "barrier initialized before first TMA",
            "steady_state": "producer and consumers alternate phases",
            "epilogue": "final wait and release complete before exit",
            "predicate_behavior": "producer and waiters use the same predicate",
            "specification": "reviewed CUDA mbarrier and TMA specifications",
            "reviewer": "sync-reviewer",
        }
    ]
    for dependency in sync["dependencies"]:
        dependency.update(
            {
                "completion_witness": "reviewed completion witness",
                "consumer_wait": "wait or same-warp issue order",
                "predicate": "uniform supported-shape predicate",
                "prologue": "producer availability checked",
                "epilogue": "consumer drain checked",
                "reviewer": "sync-reviewer",
            }
        )
        if _is_cross_warp_dependency(dependency):
            dependency.update({"mechanism": "channel", "channel": "ready"})
        else:
            dependency.update(
                {
                    "mechanism": "program_order",
                    "program_order_proof": "same physical warps and issue order",
                }
            )
    sync["additional_obligations"][0].update(
        {
            "wait_kind": "instruction",
            "wait_node": 1,
            "distance": 0,
            "kernel_exit_wait": None,
            "channel": "ready",
            "completion_witness": "TMA transaction completion",
            "predicate": "uniform supported-shape predicate",
            "epilogue_drain": "final TMA completion is waited",
            "reviewer": "sync-reviewer",
        }
    )
    sync["ordering_edges"] = [
        {
            "id": "tile-reuse",
            "allocation_id": "tile",
            "src": 2,
            "dst": 0,
            "distance": 2,
            "channel": "ready",
            "reason": "last tile consumer releases the next producer slot",
            "reviewer": "sync-reviewer",
        },
        {
            "id": "ready-barrier-reuse",
            "allocation_id": "ready-barrier-storage",
            "src": 2,
            "dst": 0,
            "distance": 2,
            "channel": "ready",
            "reason": "last waiter releases the next barrier phase",
            "reviewer": "sync-reviewer",
        },
    ]
    sync["phase_traces"] = [
        {
            "channel": "ready",
            "initial_phase": 0,
            "wraps_checked": 2,
            "regions": ["prologue", "steady_state", "epilogue"],
            "events": [
                {
                    "iteration": iteration,
                    "slot": iteration % 2,
                    "phase": (iteration // 2) % 2,
                    "producer": "TMA launch",
                    "completion": "TMA transaction completion",
                    "wait": "consumer parity wait",
                    "release": "consumer release",
                }
                for iteration in range(4)
            ],
            "zero_trip": "no producer and no waiter",
            "partial_iteration": "producer and waiters share the same predicate",
            "epilogue_drain": "final completion and release are waited",
            "reviewer": "sync-reviewer",
        }
    ]
    cross_edges = sum(
        _is_cross_warp_dependency(dependency)
        for dependency in sync["dependencies"]
    )
    sync["coverage"] = {
        "ir_edges": 2,
        "mapped_ir_edges": 2,
        "cross_warp_edges": cross_edges,
        "mapped_cross_warp_edges": cross_edges,
        "ir_instance_dependencies": 4,
        "mapped_instance_dependencies": 4,
        "async_completion_obligations": 1,
        "mapped_async_completion_obligations": 1,
        "reuse_obligations": 2,
        "mapped_reuse_obligations": 2,
    }
    sync["review"] = {"status": "approved", "reviewer": "sync-reviewer"}
    _write_json(result.sync_plan_path, sync)


def _complete_authoring(result):
    authoring = _load(result.authoring_path)
    authoring["status"] = "approved"
    authoring["clean_room"]["declaration_reviewed_by"] = "clean-room-reviewer"
    digests = {
        "mapping": _sha256(result.mapping_path),
        "memory": _sha256(result.memory_plan_path),
        "synchronization": _sha256(result.sync_plan_path),
    }
    for kind, digest in digests.items():
        authoring["manifests"][kind]["sha256"] = digest
    clean_room_sha256 = hashlib.sha256(
        json.dumps(
            authoring["clean_room"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    authoring["human_reviews"] = [
        {
            "kind": kind,
            "reviewer": f"{kind}-reviewer",
            "status": "approved",
            "artifact_sha256": digest,
        }
        for kind, digest in digests.items()
    ] + [
        {
            "kind": "clean_room",
            "reviewer": "clean-room-reviewer",
            "status": "approved",
            "artifact_sha256": clean_room_sha256,
        }
    ]
    _write_json(result.authoring_path, authoring)


def _complete_bundle(tmp_path, payload=None):
    ir_path, handoff_path, result = _scaffold(tmp_path, payload)
    _complete_mapping(result)
    _complete_memory(result)
    _complete_sync(result)
    _complete_authoring(result)
    return ir_path, handoff_path, result


def _overlapping_warp_ir():
    """Producer on warps {1, 2}, consumer on {1}: overlapping but unequal, so
    the edge still crosses a warp boundary."""
    payload = _ir()
    payload["instructions"][2]["warps"] = [1]
    payload["warps"][2]["issue_trace"] = payload["warps"][2]["issue_trace"][:1]
    edge = payload["dependencies"][1]
    edge["consumer_warps"] = [1]
    payload["cross_warp_dependencies"].append(edge)
    return payload


def test_v2_pipelined_program_materializes_all_copies_and_boundaries():
    payload = _ir()
    program = payload["pipelined_program"]
    instances = program["instances"]
    slots = program["steady_state"]["slots"]
    node_count = len(payload["instructions"])
    copies = payload["pipeline"]["copies"]

    assert (
        payload["schema_version"]
        == PIPELINED_IR_SCHEMA
        == VALIDATED_PIPELINED_IR_SCHEMA
        == PIPELINED_IR_SCHEMA
    )
    plan = SimpleNamespace(
        ii=payload["pipeline"]["ii"],
        copies=copies,
        nodes=tuple(
            SimpleNamespace(
                node_id=node["id"], cycle=node["cycle"], stage=node["stage"]
            )
            for node in payload["instructions"]
        ),
        edges=tuple(SimpleNamespace(**edge) for edge in payload["dependencies"]),
    )
    assert (
        _build_pipelined_program(plan, payload["pipeline"]["regions"])
        == program
    )
    assert len(instances) == node_count * copies
    assert len(slots) == node_count
    assert len([item for item in instances if item["region"] != "steady_state"]) == (
        node_count * (copies - 1)
    )
    instructions = {node["id"]: node for node in payload["instructions"]}
    instance_keys = {
        (item["node"], item["copy"], item["cycle"], item["region"])
        for item in instances
    }
    for slot in slots:
        instruction = instructions[slot["node"]]
        assert slot["iteration_lag"] == instruction["stage"]
        assert (
            slot["node"],
            slot["copy"],
            slot["cycle"],
            "steady_state",
        ) in instance_keys


def test_v2_pipelined_program_materializes_carried_dependencies():
    program = _ir()["pipelined_program"]
    edge_instances = [
        item
        for item in program["instance_dependencies"]
        if item["edge_index"] == 11
    ]
    assert edge_instances == [
        {
            "edge_index": 11,
            "producer": {"node": 1, "copy": -1},
            "consumer": {"node": 2, "copy": 0},
            "external": True,
            "carried_distance": 1,
        },
        {
            "edge_index": 11,
            "producer": {"node": 1, "copy": 0},
            "consumer": {"node": 2, "copy": 1},
            "external": False,
            "carried_distance": 1,
        },
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda program: program["instances"][0].update(cycle=1),
        lambda program: program["instances"][0].update(copy=False),
        lambda program: program["instances"][0].update(cycle=0.0),
        lambda program: program["instances"].pop(),
        lambda program: program["instance_dependencies"][0].update(external=True),
        lambda program: program["steady_state"]["slots"][0].update(
            iteration_lag=1
        ),
    ],
)
def test_schema_rejects_noncanonical_pipelined_program(tmp_path, mutation):
    payload = _ir()
    mutation(payload["pipelined_program"])
    ir_path, handoff_path = _write_inputs(tmp_path, payload)

    with pytest.raises(ArtifactError, match="schedule expansion"):
        scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "output")


def test_schema_accepts_reduced_register_budget():
    payload = _ir()
    payload["provenance"]["machine"]["regs_per_warp"] = 4096

    validate_pipelined_ir(payload)


def test_schema_rejects_unsorted_instruction_warps(tmp_path):
    payload = _ir()
    payload["instructions"][1]["warps"] = [2, 1]
    payload["dependencies"][0]["consumer_warps"] = [2, 1]
    payload["dependencies"][1]["producer_warps"] = [2, 1]
    ir_path, handoff_path = _write_inputs(tmp_path, payload)

    with pytest.raises(ArtifactError, match="warps must be sorted"):
        scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda ir: ir["dependencies"][0].update(consumer_warps=[1]),
            "consumer warps mismatch",
        ),
        (
            lambda ir: ir["dependencies"][0].update(spill_cost=-1),
            "spill cost must be an integer >= 0",
        ),
        (
            lambda ir: ir["dependencies"][0].update(ws_semantics=None),
            "ws_semantics must be a boolean",
        ),
        (
            lambda ir: ir["instructions"][0].update(warps=[True]),
            "warps must be non-empty non-negative ints",
        ),
        (
            lambda ir: ir["cross_warp_dependencies"][0].update(
                producer_warps=[False]
            ),
            "producer warps mismatch",
        ),
        (
            lambda ir: ir["dependencies"][1].update(spill_cost=1),
            "non-cross spill cost must be zero",
        ),
        (
            lambda ir: ir["dependencies"][0].update(spill_cost=1),
            "violates scheduled latency",
        ),
    ],
)
def test_schema_validates_dependency_warp_and_spill_facts(tmp_path, mutation, message):
    payload = _ir()
    mutation(payload)
    ir_path, handoff_path = _write_inputs(tmp_path, payload)

    with pytest.raises(ArtifactError, match=message):
        scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "output")


def test_overlapping_warp_dependency_requires_channel(tmp_path):
    ir_path, _, result = _scaffold(tmp_path, _overlapping_warp_ir())
    _complete_sync(result)
    sync = _load(result.sync_plan_path)
    binding = next(
        dependency
        for dependency in sync["dependencies"]
        if dependency["edge_index"] == 11
    )

    assert set(binding["consumer_warps"]) < set(binding["producer_warps"])
    assert binding["mechanism"] == "channel"
    assert audit_sync(ir_path, result.sync_plan_path).ok

    sync["channels"][0]["producer_warps"] = [0]
    _write_json(result.sync_plan_path, sync)
    with pytest.raises(AuditError, match="channel lacks producer warps"):
        audit_sync(ir_path, result.sync_plan_path)
    sync["channels"][0]["producer_warps"] = [0, 1, 2]

    binding.update(
        mechanism="program_order",
        channel=None,
        program_order_proof="same physical warps",
    )
    _write_json(result.sync_plan_path, sync)
    with pytest.raises(AuditError, match="needs a channel"):
        audit_sync(ir_path, result.sync_plan_path)


def test_overlapping_warp_dependency_passes_complete_bundle(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path, _overlapping_warp_ir())

    assert audit_mapping(ir_path, result.mapping_path).ok
    assert audit_bundle(
        ir_path,
        handoff_path,
        result.authoring_path,
        result.mapping_path,
        result.memory_plan_path,
        result.sync_plan_path,
    ).ok


def test_mapping_audit_rejects_mutated_materialized_instance(tmp_path):
    ir_path, _, result = _scaffold(tmp_path)
    _complete_mapping(result)
    mapping = _load(result.mapping_path)
    mapping["nodes"][0]["instances"][0]["copy"] = False
    _write_json(result.mapping_path, mapping)

    with pytest.raises(AuditError, match="instances mismatch"):
        audit_mapping(ir_path, result.mapping_path)


def test_sync_audit_rejects_mutated_instance_dependency(tmp_path):
    ir_path, _, result = _scaffold(tmp_path)
    _complete_sync(result)
    sync = _load(result.sync_plan_path)
    sync["dependencies"][0]["instance_dependencies"][0]["external"] = True
    _write_json(result.sync_plan_path, sync)

    with pytest.raises(AuditError, match="instance dependencies mismatch"):
        audit_sync(ir_path, result.sync_plan_path)


def test_non_ws_dependency_is_not_classified_as_cross_warp():
    payload = _ir()
    payload["dependencies"][0]["ws_semantics"] = False
    payload["cross_warp_dependencies"] = []

    validate_pipelined_ir(payload)


def test_scaffold_is_deterministic_non_executable_and_contract_aligned(tmp_path):
    ir_path, handoff_path = _write_inputs(tmp_path)
    first = scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "first")
    second = scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "second")

    for name in (
        "kernel.cu",
        "manual_cuda_authoring.json",
        "mapping_manifest.json",
        "memory_plan.json",
        "sync_manifest.json",
    ):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()
    assert '#error "Manual CUDA lowering is required' in first.cuda_path.read_text()
    assert "__global__" not in first.cuda_path.read_text()
    assert _load(first.authoring_path)["authoring_mode"] == "agent-assisted-manual-cuda"
    assert (
        _load(first.memory_plan_path)["schema_version"] == MEMORY_PLAN_SCHEMA
    )
    assert _load(first.sync_plan_path)["schema_version"] == SYNC_PLAN_SCHEMA


def test_scaffold_rejects_missing_or_stale_handoff(tmp_path):
    ir_path, handoff_path = _write_inputs(tmp_path)
    handoff = _load(handoff_path)
    handoff["pipelined_ir"]["sha256"] = "0" * 64
    _write_json(handoff_path, handoff)

    with pytest.raises(ArtifactError, match="handoff pipelined IR sha256 mismatch"):
        scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ir: ir.update(provenance={}), "provenance.model must be"),
        (
            lambda ir: ir["provenance"].update(solver_sources_sha256="0" * 64),
            "does not match the current toolchain",
        ),
        (lambda ir: ir["instructions"][0].update(cycle=1), "cycle/stage/offset"),
        (
            lambda ir: ir["warps"][0]["issue_trace"][0].update(
                cycle=False
            ),
            "issue-trace cycle mismatch",
        ),
        (lambda ir: ir["pipeline"].update(regions=[]), "pipeline.regions"),
        (
            lambda ir: ir["pipeline"]["regions"][0].update(cycle_begin=False),
            "pipeline.regions",
        ),
        (lambda ir: ir["source_program"].update(ops={}), "ops must not be empty"),
        (
            lambda ir: ir["source_program"]["ops"]["op0"].pop("operands"),
            "source operation op0.operands",
        ),
        (lambda ir: ir["provenance"]["machine"].update(tmem_cols=1024), "B200 model"),
        (
            lambda ir: ir["provenance"]["machine"].update(
                group_widths=[True, 2, 4, 8]
            ),
            "B200 model",
        ),
        (
            lambda ir: ir["provenance"]["machine"]["capacities"].update(
                TC=True
            ),
            "capacities is not the B200 model",
        ),
        (
            lambda ir: ir["provenance"]["machine"].update(regs_per_warp=8161),
            "regs_per_warp exceeds",
        ),
    ],
)
def test_scaffold_rejects_partial_or_inconsistent_ir(tmp_path, mutation, message):
    payload = _ir()
    mutation(payload)
    ir_path, handoff_path = _write_inputs(tmp_path, payload)

    with pytest.raises(ArtifactError, match=message):
        scaffold_manual_cuda(ir_path, handoff_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("audit", "path_name", "message"),
    [
        (audit_mapping, "mapping_path", "status must be approved"),
        (audit_memory, "memory_plan_path", "status must be approved"),
        (audit_sync, "sync_plan_path", "status must be approved"),
    ],
)
def test_draft_manifests_never_pass_an_audit(tmp_path, audit, path_name, message):
    ir_path, _, result = _scaffold(tmp_path)

    with pytest.raises(AuditError, match=message):
        audit(ir_path, getattr(result, path_name))


def test_complete_reviewed_bundle_passes(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path)
    sync = _load(result.sync_plan_path)
    same_warps = next(
        dependency
        for dependency in sync["dependencies"]
        if dependency["edge_index"] == 11
    )

    assert same_warps["producer_warps"] == same_warps["consumer_warps"] == [1, 2]
    assert same_warps["mechanism"] == "program_order"
    assert audit_mapping(ir_path, result.mapping_path).ok
    assert audit_memory(ir_path, result.memory_plan_path).ok
    assert audit_sync(ir_path, result.sync_plan_path).ok
    assert audit_authoring(ir_path, handoff_path, result.authoring_path).ok
    audit = audit_bundle(
        ir_path,
        handoff_path,
        result.authoring_path,
        result.mapping_path,
        result.memory_plan_path,
        result.sync_plan_path,
    )
    assert audit.ok
    assert audit.kind == "bundle"


def test_mapping_audit_checks_schedule_and_real_source(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    mapping = _load(result.mapping_path)
    mapping["nodes"][0]["cycle"] = 1
    _write_json(result.mapping_path, mapping)
    with pytest.raises(AuditError, match="cycle mismatch"):
        audit_mapping(ir_path, result.mapping_path)

    mapping["nodes"][0]["cycle"] = 0
    mapping["cuda_source"]["sha256"] = "0" * 64
    _write_json(result.mapping_path, mapping)
    with pytest.raises(AuditError, match="CUDA source sha256 mismatch"):
        audit_mapping(ir_path, result.mapping_path)


def test_shallow_sync_channel_and_bad_phase_do_not_pass(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    sync = _load(result.sync_plan_path)
    sync["channels"] = [{"id": "ready"}]
    _write_json(result.sync_plan_path, sync)
    with pytest.raises(AuditError, match="barrier_id"):
        audit_sync(ir_path, result.sync_plan_path)

    _complete_sync(result)
    sync = _load(result.sync_plan_path)
    sync["phase_traces"][0]["events"][1]["phase"] = 1
    _write_json(result.sync_plan_path, sync)
    with pytest.raises(AuditError, match="slot/phase recurrence"):
        audit_sync(ir_path, result.sync_plan_path)


@pytest.mark.parametrize(
    ("manifest", "mutation", "audit", "message"),
    [
        (
            "mapping",
            lambda plan: plan["nodes"][0].update(issue_order=False),
            audit_mapping,
            "issue_order mismatch",
        ),
        (
            "mapping",
            lambda plan: plan["nodes"][0].update(warps=[False]),
            audit_mapping,
            "warps mismatch",
        ),
        (
            "mapping",
            lambda plan: plan["nodes"][0]["manual_cuda"].update(
                physical_warps=[False]
            ),
            audit_mapping,
            "physical warp mapping mismatch",
        ),
        (
            "sync",
            lambda plan: plan["additional_obligations"][0].update(
                instruction_id=False
            ),
            audit_sync,
            "does not match the IR",
        ),
        (
            "sync",
            lambda plan: plan["phase_traces"][0].update(initial_phase=False),
            audit_sync,
            "initial phase mismatch",
        ),
        (
            "sync",
            lambda plan: plan["phase_traces"][0]["events"][0].update(slot=False),
            audit_sync,
            "slot/phase recurrence",
        ),
    ],
)
def test_manual_audits_reject_boolean_integer_aliases(
    tmp_path, manifest, mutation, audit, message
):
    ir_path, _, result = _complete_bundle(tmp_path)
    path = result.mapping_path if manifest == "mapping" else result.sync_plan_path
    plan = _load(path)
    mutation(plan)
    _write_json(path, plan)

    with pytest.raises(AuditError, match=message):
        audit(ir_path, path)


def test_ordering_edge_metadata_cannot_bypass_cycle_check(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    sync = _load(result.sync_plan_path)
    sync["ordering_edges"][0].update(
        src=0,
        dst=0,
        distance=0,
        edge_index=10,
        materialized_edge_index=10,
    )
    _write_json(result.sync_plan_path, sync)

    with pytest.raises(AuditError, match="has a cycle"):
        audit_sync(ir_path, result.sync_plan_path)


def test_memory_audit_rejects_empty_and_non_b200_plans(tmp_path):
    ir_path, _, result = _scaffold(tmp_path)
    memory = _load(result.memory_plan_path)
    memory["status"] = "approved"
    memory["review"] = {"status": "approved", "reviewer": "reviewer"}
    _write_json(result.memory_plan_path, memory)
    with pytest.raises(AuditError, match="at least one allocation"):
        audit_memory(ir_path, result.memory_plan_path)

    memory["capacities"]["tmem_columns"] = 1024
    _write_json(result.memory_plan_path, memory)
    with pytest.raises(AuditError, match="fixed B200 capacities"):
        audit_memory(ir_path, result.memory_plan_path)


def test_authoring_audit_rejects_stale_manifest_hash(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path)
    mapping = _load(result.mapping_path)
    mapping["review"]["reviewer"] = "changed-reviewer"
    _write_json(result.mapping_path, mapping)

    with pytest.raises(AuditError, match="mapping manifest sha256 mismatch"):
        audit_authoring(ir_path, handoff_path, result.authoring_path)


def test_bundle_rejects_unsigned_alternate_manifest(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path)
    alternate = result.mapping_path.parent / "alternate_mapping.json"
    alternate.write_bytes(result.mapping_path.read_bytes())

    with pytest.raises(AuditError, match="not the authoring-signed file"):
        audit_bundle(
            ir_path,
            handoff_path,
            result.authoring_path,
            alternate,
            result.memory_plan_path,
            result.sync_plan_path,
        )


def test_mapping_requires_executable_source_and_unique_node_anchors(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    mapping = _load(result.mapping_path)
    result.cuda_path.write_text("// PAPER_NODE_0\n// PAPER_NODE_1\n// PAPER_NODE_2\n")
    mapping["cuda_source"]["sha256"] = _sha256(result.cuda_path)
    _write_json(result.mapping_path, mapping)
    with pytest.raises(AuditError, match="executable kernel"):
        audit_mapping(ir_path, result.mapping_path)

    result.cuda_path.write_text(_source_text())
    mapping["cuda_source"]["sha256"] = _sha256(result.cuda_path)
    mapping["nodes"][1]["manual_cuda"]["source_anchor"] = "PAPER_NODE_0"
    _write_json(result.mapping_path, mapping)
    with pytest.raises(AuditError, match="unique source anchors"):
        audit_mapping(ir_path, result.mapping_path)


def test_mapping_allows_reviewed_zero_cost_and_shared_realizations(tmp_path):
    payload = _ir()
    payload["instructions"][2]["op_kind"] = "tt.broadcast"
    payload["instructions"][2]["pipeline"] = "NONE"
    payload["source_program"]["ops"]["op2"]["kind"] = "tt.broadcast"
    ir_path, _, result = _scaffold(tmp_path, payload)
    _complete_mapping(result)
    mapping = _load(result.mapping_path)
    zero_cost = mapping["nodes"][2]["manual_cuda"]
    zero_cost.update(
        {
            "machine_realization": "zero_cost_alias",
            "disassembly_anchor": None,
            "disassembly_status": "not_applicable",
            "equivalence_proof": "broadcast is a reviewed view of the same fragment",
        }
    )
    _write_json(result.mapping_path, mapping)

    assert audit_mapping(ir_path, result.mapping_path).ok

    shared_proof = "two scalar IR nodes are fused into one reviewed machine op"
    for index in (0, 1):
        manual = mapping["nodes"][index]["manual_cuda"]
        manual.update(
            {
                "machine_realization": "shared_instruction",
                "disassembly_anchor": "SASS_NODE_0",
                "equivalence_proof": shared_proof,
            }
        )
    _write_json(result.mapping_path, mapping)

    assert audit_mapping(ir_path, result.mapping_path).ok


def test_bundle_derives_reuse_obligations_from_memory(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path)
    sync = _load(result.sync_plan_path)
    sync["ordering_edges"] = []
    sync["coverage"]["reuse_obligations"] = 0
    sync["coverage"]["mapped_reuse_obligations"] = 0
    _write_json(result.sync_plan_path, sync)
    _complete_authoring(result)

    assert audit_sync(ir_path, result.sync_plan_path).ok
    with pytest.raises(AuditError, match="lacks a derived reuse edge"):
        audit_bundle(
            ir_path,
            handoff_path,
            result.authoring_path,
            result.mapping_path,
            result.memory_plan_path,
            result.sync_plan_path,
        )


def test_memory_audit_reconciles_slots_and_compiler_resources(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    memory = _load(result.memory_plan_path)
    memory["resource_budget"]["compiler_reported"]["register_words"] = 95
    _write_json(result.memory_plan_path, memory)
    with pytest.raises(AuditError, match="register use disagrees"):
        audit_memory(ir_path, result.memory_plan_path)

    memory["resource_budget"]["compiler_reported"]["register_words"] = 96
    memory["allocations"][0]["slot_size"] = 64
    _write_json(result.memory_plan_path, memory)
    with pytest.raises(AuditError, match="size must include every slot"):
        audit_memory(ir_path, result.memory_plan_path)


def test_sync_audit_preserves_dependency_latency(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    sync = _load(result.sync_plan_path)
    sync["dependencies"][0]["latency"] += 1
    _write_json(result.sync_plan_path, sync)

    with pytest.raises(AuditError, match="latency mismatch"):
        audit_sync(ir_path, result.sync_plan_path)


def test_audits_preserve_dependency_warp_and_spill_facts(tmp_path):
    ir_path, _, result = _complete_bundle(tmp_path)
    mapping = _load(result.mapping_path)
    mapping["dependencies"][0]["producer_warps"] = [1]
    _write_json(result.mapping_path, mapping)

    with pytest.raises(AuditError, match="producer_warps mismatch"):
        audit_mapping(ir_path, result.mapping_path)

    sync = _load(result.sync_plan_path)
    sync["dependencies"][0]["spill_cost"] = 1
    _write_json(result.sync_plan_path, sync)
    with pytest.raises(AuditError, match="spill_cost mismatch"):
        audit_sync(ir_path, result.sync_plan_path)


def test_scaffold_includes_tma_store_completion_obligation(tmp_path):
    payload = _ir()
    payload["instructions"][0]["op_kind"] = "tt.descriptor_store"
    payload["instructions"][0]["pipeline"] = "TMA"
    payload["source_program"]["ops"]["op0"]["kind"] = "tt.descriptor_store"
    _, _, result = _scaffold(tmp_path, payload)

    obligations = _load(result.sync_plan_path)["additional_obligations"]
    assert obligations[0]["kind"] == "tma_store_completion"


def test_scaffold_refuses_to_overwrite_manual_work(tmp_path):
    ir_path, handoff_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "manual_cuda"
    result = scaffold_manual_cuda(ir_path, handoff_path, output_dir)
    result.cuda_path.write_text("manual work\n")

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        scaffold_manual_cuda(ir_path, handoff_path, output_dir)
    assert result.cuda_path.read_text() == "manual work\n"


def test_cli_scaffold_requires_handoff_and_emits_all_artifacts(tmp_path):
    ir_path, handoff_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "manual_cuda"
    assert (
        main(
            [
                "scaffold",
                "--ir",
                str(ir_path),
                "--handoff",
                str(handoff_path),
                "--out-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert {path.name for path in output_dir.iterdir()} == {
        "kernel.cu",
        "manual_cuda_authoring.json",
        "mapping_manifest.json",
        "memory_plan.json",
        "sync_manifest.json",
    }


def test_cli_audit_bundle_uses_the_reviewed_artifact_set(tmp_path):
    ir_path, handoff_path, result = _complete_bundle(tmp_path)

    assert (
        main(
            [
                "audit-bundle",
                "--ir",
                str(ir_path),
                "--handoff",
                str(handoff_path),
                "--authoring",
                str(result.authoring_path),
                "--mapping",
                str(result.mapping_path),
                "--memory",
                str(result.memory_plan_path),
                "--sync",
                str(result.sync_plan_path),
            ]
        )
        == 0
    )


def test_cli_preserves_legacy_direct_handoff(monkeypatch, tmp_path):
    captured = {}

    def fake_handoff(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ir_path=tmp_path / "ir.json",
            manifest_path=tmp_path / "manifest.json",
            ir_sha256="abc",
        )

    monkeypatch.setattr("skc.__main__.prepare_manual_cuda_handoff", fake_handoff)

    assert (
        main(
            [
                "--solution",
                "solution.json",
                "--ddg",
                "ddg.json",
                "--baseline-graph",
                "graph.json",
                "--ir-out",
                "ir.json",
                "--manifest-out",
                "manifest.json",
            ]
        )
        == 0
    )
    assert captured == {
        "solution_path": "solution.json",
        "ddg_path": "ddg.json",
        "baseline_graph_path": "graph.json",
        "ir_out": "ir.json",
        "manifest_out": "manifest.json",
    }
