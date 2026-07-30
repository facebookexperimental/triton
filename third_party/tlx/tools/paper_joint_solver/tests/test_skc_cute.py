"""Manual CUDA handoff and retired CuTe-shim regression tests."""

import ast
import json
from pathlib import Path

import pytest

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel, required_group_width
from paper_joint_solver.schedule_plan import (
    SOLUTION_SCHEMA,
    LegacySolutionError,
    ScheduleArtifactError,
    _maximum_useful_group_labels,
    load_schedule_plan,
    solution_provenance,
)
from skc import ArtifactError
from skc.compiler import prepare_manual_cuda_handoff, scaffold_manual_cuda
from skc_cute import binder_cute, driver

PKG = Path(__file__).resolve().parent.parent
EXAMPLES = PKG.parent / "sched2tlx" / "examples"


def _solver_stats(machine, problem, widths, horizon):
    scheduled_warps = sum(widths.values())
    return {
        "T": horizon,
        "num_groups": len(widths),
        "scheduled_warps": scheduled_warps,
        "fixed_warps": machine.warp_fixed_overhead,
        "num_warps": scheduled_warps + machine.warp_fixed_overhead,
        "physical_warp_budget": machine.num_warps,
        "max_groups": _maximum_useful_group_labels(problem, machine),
        "regs_per_warp": machine.regs_per_warp,
        "regs_per_sm": machine.regs_per_sm,
        "regs_fixed_overhead": machine.regs_fixed_overhead,
        "smem_capacity": machine.smem_bytes,
        "smem_fixed_overhead": machine.smem_fixed_overhead,
        "barrier_pair_bytes": machine.barrier_pair_bytes,
        "tmem_capacity": machine.tmem_cols,
        "tmem_fixed_cols": machine.tmem_fixed_cols,
        "emitter_infra_nodes": len(problem.emitter_infra),
        "lane_mask_model": "paper_physical_masks",
    }


@pytest.mark.parametrize(
    ("solution", "ddg", "graph"),
    [
        (
            PKG / "subtiled_fa4exact_solution.json",
            EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json",
            EXAMPLES / "case3_FA_fp16_subtiled" / "schedule_graph.json",
        ),
        (
            PKG / "bwd_skc_solution.json",
            EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json",
            EXAMPLES / "case4_FA_bwd" / "schedule_graph_hd128.json",
        ),
    ],
)
def test_legacy_phase_b_solutions_are_rejected(solution, ddg, graph):
    with pytest.raises(LegacySolutionError, match="refusing to infer"):
        load_schedule_plan(solution, ddg, graph)


def test_inherited_fa4_backend_is_removed():
    with pytest.raises(binder_cute.InheritedFA4BackendRemoved):
        binder_cute.bind_fwd()
    with pytest.raises(binder_cute.InheritedFA4BackendRemoved):
        binder_cute.bind_bwd()
    with pytest.raises(binder_cute.InheritedFA4BackendRemoved):
        driver.install("fwd", None)


def test_benchmark_registry_does_not_expose_tlx_as_twill():
    tree = ast.parse((PKG / "bench" / "bench_bars.py").read_text())
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "make_registry":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys.update(
                    key.value
                    for key in child.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
    assert keys.isdisjoint(
        {
            "jos",
            "jos_bwd",
            "skc_default",
            "skc_bwd_default",
            "skc_bwd_q3",
            "skc_bwd_q4",
        }
    )


def _write_case1_artifact(tmp_path):
    case = EXAMPLES / "case1_simple_gemm"
    ddg = case / "ddg.json"
    graph = case / "schedule_graph.json"
    graph_data = json.loads(graph.read_text())
    loop = next(loop for loop in graph_data["loops"] if not loop.get("is_outer", False))
    schedule = loop["schedule_loop"]

    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    cycles = {
        str(node["id"]): node["schedule"]["cycle"]
        for node in schedule["graph"]["nodes"]
    }
    # The fixture has two TMA nodes, two zero-cost alloc wrappers, and one
    # MMA.  Separate those three exact roles so every emitted task owns all
    # lanes selected by each operation.
    warp = {}
    for node in schedule["graph"]["nodes"]:
        node_id = node["id"]
        if node_id in problem.variable_latency:
            group = 0
        elif node["op_kind"] == "ttng.tc_gen5_mma":
            group = 2
        else:
            group = 1
        warp[str(node_id)] = group

    ii = schedule["II"]
    length = max(
        cycles[str(node_id)] + problem.lat[node_id] for node_id in problem.nodes
    )
    copies = -(-length // ii)
    widths = {0: 1, 1: 1, 2: 1}
    horizon = (copies - 1) * ii + length
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": solution_provenance(ddg, graph, machine, 300),
        "ii": ii,
        "length": length,
        "copies": copies,
        "horizon": horizon,
        "cycles": cycles,
        "warp": warp,
        "group_widths": {str(group): width for group, width in widths.items()},
        "lane_masks": {node_id: [0] for node_id in warp},
        "stats": _solver_stats(machine, problem, widths, horizon),
    }
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def _handoff(tmp_path, solution, ddg, graph, stem="out"):
    return prepare_manual_cuda_handoff(
        solution_path=solution,
        ddg_path=ddg,
        baseline_graph_path=graph,
        ir_out=tmp_path / f"{stem}_ir.json",
        manifest_out=tmp_path / f"{stem}_manifest.json",
    )


def test_handoff_emits_pipelined_warp_annotated_ir(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    result = _handoff(tmp_path, solution, ddg, graph)

    ir_text = result.ir_path.read_text()
    assert "tlx.async_tasks" not in ir_text
    assert "@triton.jit" not in ir_text
    assert "sched2tlx-standalone" not in ir_text
    ir = json.loads(ir_text)
    assert ir["schema_version"] == "twill-pipelined-warp-ir-v1"
    assert [region["kind"] for region in ir["pipeline"]["regions"]] == [
        "prologue",
        "steady_state",
        "epilogue",
    ]
    assert ir["pipeline"]["regions"][-1]["cycle_end"] == ir["pipeline"][
        "horizon"
    ]
    assert ir["source_program"]["format"] == "ttgir-derived-operation-table"
    assert ir["source_program"]["kernel"]
    assert ir["source_program"]["ops"]

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schema_version"] == "twill-manual-cuda-handoff-v1"
    assert manifest["pipelined_ir"]["sha256"] == result.ir_sha256
    assert manifest["lowering"]["paper_endpoint"] == "expert-manual-cuda-cpp"
    assert manifest["lowering"]["status"] == "not_performed"
    assert manifest["lowering"]["executable_generated"] is False
    assert set(manifest["lowering"]["remaining_manual_steps"]) == {
        "memory_allocation",
        "data_layout_conversion",
        "synchronization_placement",
        "instruction_selection",
    }
    assert len(ir["dependencies"]) > 0
    assert {
        (edge["src"], edge["dst"], edge["distance"])
        for edge in ir["cross_warp_dependencies"]
    } == {(0, 1, 0), (2, 3, 0), (1, 4, 0), (3, 4, 0)}


def test_handoff_with_stale_source_op_refs_fails_scaffold_input_gate(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    handoff = _handoff(tmp_path, solution, ddg, graph)

    with pytest.raises(ArtifactError, match="references unknown source op"):
        scaffold_manual_cuda(
            handoff.ir_path,
            handoff.manifest_path,
            tmp_path / "manual_cuda",
        )


def test_cycle_mutation_changes_structure_or_is_rejected(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    original = _handoff(tmp_path, solution, ddg, graph, "original")
    original_ir = json.loads(original.ir_path.read_text())

    payload = json.loads(solution.read_text())
    payload["cycles"]["2"] += 1
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload))
    try:
        changed = _handoff(tmp_path, mutated, ddg, graph, "mutated")
    except ScheduleArtifactError:
        return
    changed_ir = json.loads(changed.ir_path.read_text())
    assert changed_ir["instructions"] != original_ir["instructions"]


def test_zero_latency_node_at_length_is_rejected(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["cycles"]["1"] = payload["length"]
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="COMPLETION"):
        load_schedule_plan(solution, ddg, graph)


def _write_schema_only_bwd_artifact(tmp_path):
    ddg = EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json"
    graph = EXAMPLES / "case4_FA_bwd" / "schedule_graph_hd128.json"
    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    payload = json.loads((PKG / "bwd_skc_solution.json").read_text())
    widths = {}
    for node_id, group in payload["warp"].items():
        group = int(group)
        width = required_group_width(problem.nodes[int(node_id)].min_warps)
        widths[group] = max(widths.get(group, 0), width)
    payload["length"] = max(
        int(payload["cycles"][str(node_id)]) + max(1, problem.lat[node_id])
        for node_id in problem.nodes
    )
    payload["copies"] = -(-payload["length"] // payload["ii"])
    payload.update(
        {
            "schema_version": SOLUTION_SCHEMA,
            "status": "sat",
            "satisfiable": True,
            "provenance": solution_provenance(ddg, graph, machine, 300),
            "horizon": ((payload["copies"] - 1) * payload["ii"] + payload["length"]),
            "group_widths": {str(group): width for group, width in widths.items()},
            "lane_masks": {
                node_id: list(
                    range(required_group_width(problem.nodes[int(node_id)].min_warps))
                )
                for node_id in payload["warp"]
            },
        }
    )
    payload["stats"].update(_solver_stats(machine, problem, widths, payload["horizon"]))
    solution = tmp_path / "schema_only_bwd.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def test_handoff_preserves_paper_valid_partial_lane_roles(tmp_path):
    solution, ddg, graph = _write_schema_only_bwd_artifact(tmp_path)
    result = _handoff(tmp_path, solution, ddg, graph, "bwd")
    instructions = json.loads(result.ir_path.read_text())["instructions"]
    assert any(
        len(node["lanes"]) < node["group_width"]
        for node in instructions
        if node["group_width"] is not None
    )


def test_provenance_mismatch_is_rejected(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["ddg_sha256"] = "0" * 64
    solution.write_text(json.dumps(payload))
    with pytest.raises(ScheduleArtifactError, match="ddg_sha256 mismatch"):
        _handoff(tmp_path, solution, ddg, graph)


def test_manual_handoff_rejects_emitter_lane_mask_mode(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["stats"]["lane_mask_model"] = "emitter_full_group"
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="paper_physical_masks"):
        _handoff(tmp_path, solution, ddg, graph)


def test_manual_handoff_rejects_restrictive_group_probe(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["stats"]["max_groups"] -= 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="stats.max_groups"):
        _handoff(tmp_path, solution, ddg, graph)
