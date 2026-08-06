"""Manual CUDA handoff and retired CuTe-shim regression tests."""

import ast
import json
from pathlib import Path

import pytest

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.schedule_plan import (
    EMITTER_MODEL_PROFILE,
    EMITTER_MODEL_OPTIONS,
    SOLUTION_SCHEMA,
    LegacySolutionError,
    ScheduleArtifactError,
    _maximum_useful_group_labels,
    load_schedule_context,
    load_schedule_plan,
    solution_provenance,
)
from skc import ArtifactError
from skc.compiler import prepare_manual_cuda_handoff, scaffold_manual_cuda
from skc_cute import binder_cute, driver

PKG = Path(__file__).resolve().parent.parent
EXAMPLES = PKG.parent / "sched2tlx" / "examples"


def _solver_stats(
    machine,
    problem,
    widths,
    horizon,
    lane_mask_model="paper_physical_masks",
):
    scheduled_warps = sum(widths.values())
    return {
        **problem.normalization_stats(),
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
        "spill_smem_footprint": True,
        "tmem_capacity": machine.tmem_cols,
        "tmem_fixed_cols": machine.tmem_fixed_cols,
        "reg_carried_same_lane": True,
        "no_cross_warp_domain": "reg-data",
        "emitter_infra_nodes": len(problem.emitter_infra),
        "lane_mask_model": lane_mask_model,
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


def _write_case1_artifact(tmp_path, *, stale_source_ops=False):
    case = EXAMPLES / "case1_simple_gemm"
    ddg = case / "ddg.json"
    ddg_data = json.loads(ddg.read_text())
    graph_data = json.loads((case / "schedule_graph.json").read_text())
    loop = next(loop for loop in graph_data["loops"] if not loop.get("is_outer", False))
    schedule = loop["schedule_loop"]
    ddg_nodes = {
        node["id"]: node for node in ddg_data["loops"][0]["ddg"]["nodes"]
    }
    for node in schedule["graph"]["nodes"]:
        node["op_ref"] = ddg_nodes[node["id"]]["op_ref"]
        node["op_kind"] = ddg_nodes[node["id"]]["op_kind"]
    if not stale_source_ops:
        graph_data["ops"] = ddg_data["ops"]
    graph = tmp_path / "case1_schedule_graph.json"
    graph.write_text(json.dumps(graph_data))

    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    cycles = {
        str(node["id"]): node["schedule"]["cycle"]
        for node in schedule["graph"]["nodes"]
    }
    # Keep the blocking wrappers and MMA in one two-lane group while assigning
    # them disjoint physical lanes. This exercises same-group cross-warp edges
    # without violating CONCURRENCY.
    warp = {}
    lane_masks = {}
    for node in schedule["graph"]["nodes"]:
        node_id = node["id"]
        if node_id in problem.variable_latency:
            group = 0
            lanes = [0]
        elif node["op_kind"] == "ttng.tc_gen5_mma":
            group = 1
            lanes = [1]
        else:
            group = 1
            lanes = [0]
        warp[str(node_id)] = group
        lane_masks[str(node_id)] = lanes

    ii = schedule["II"]
    length = max(
        cycles[str(node_id)] + problem.lat[node_id] for node_id in problem.nodes
    )
    copies = -(-length // ii)
    widths = {0: 1, 1: 2}
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
        "lane_masks": lane_masks,
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


def _write_case1_emitter_artifact(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    graph_data = json.loads(graph.read_text())
    nodes = graph_data["loops"][0]["schedule_loop"]["graph"]["nodes"]
    warp = {}
    for node in nodes:
        node_id = node["id"]
        if node_id in problem.variable_latency:
            group = 0
        elif node["op_kind"] == "ttng.tc_gen5_mma":
            group = 2
        else:
            group = 1
        warp[str(node_id)] = group
    widths = {group: 1 for group in set(warp.values())}
    payload["warp"] = warp
    payload["lane_masks"] = {node_id: [0] for node_id in warp}
    payload["group_widths"] = {
        str(group): width for group, width in widths.items()
    }
    payload["provenance"] = solution_provenance(
        ddg,
        graph,
        machine,
        300,
        model_options=EMITTER_MODEL_OPTIONS,
    )
    payload["stats"] = _solver_stats(
        machine,
        problem,
        widths,
        payload["horizon"],
        lane_mask_model="emitter_full_group",
    )
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def test_emitter_solution_requires_explicit_profile(tmp_path):
    solution, ddg, graph = _write_case1_emitter_artifact(tmp_path)

    with pytest.raises(ScheduleArtifactError, match="unknown production model"):
        load_schedule_plan(
            solution,
            ddg,
            graph,
            model_profile="caller-supplied-diagnostic",
        )
    with pytest.raises(ScheduleArtifactError, match="emitter selection constraints"):
        load_schedule_plan(solution, ddg, graph)

    plan = load_schedule_plan(
        solution,
        ddg,
        graph,
        model_profile=EMITTER_MODEL_PROFILE,
    )
    assert plan.model_options == EMITTER_MODEL_OPTIONS
    assert all(node.lanes == (0,) for node in plan.nodes)


def _write_ws_edge_artifact(tmp_path):
    ddg = tmp_path / "ws_edges.json"
    graph = tmp_path / "ws_edges_graph.json"
    specs = [
        ("infra", "arith.constant", ["f32"]),
        ("infra_user", "arith.addf", []),
        ("data", "arith.mulf", ["tensor<4xf32>"]),
        ("data_user", "arith.addf", []),
        ("signal", "ttng.async_commit_group", ["!ttg.async.token"]),
        ("signal_user", "ttng.async_wait", []),
    ]
    nodes = [
        {
            "id": node_id,
            "op_ref": op_ref,
            "op_kind": op_kind,
            "pipeline": "NONE",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
        }
        for node_id, (op_ref, op_kind, _) in enumerate(specs)
    ]
    nodes[2]["spill_cost"] = 5
    ddg.write_text(
        json.dumps(
            {
                "schema_version": "ddg-0.1",
                "ops": {
                    op_ref: {"result_types": result_types}
                    for op_ref, _, result_types in specs
                },
                "loops": [
                    {
                        "trip_count": 4,
                        "min_ii": 1,
                        "ddg": {
                            "nodes": nodes,
                            "edges": [
                                {
                                    "src": 0,
                                    "dst": 1,
                                    "latency": 0,
                                    "src_result_idx": 0,
                                },
                                {
                                    "src": 2,
                                    "dst": 3,
                                    "latency": 2,
                                    "src_result_idx": 0,
                                },
                                {
                                    "src": 4,
                                    "dst": 5,
                                    "latency": 0,
                                    "src_result_idx": 0,
                                },
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph.write_text(
        json.dumps(
            {
                "loops": [
                    {
                        "schedule_loop": {
                            "graph": {
                                "nodes": [
                                    {
                                        "id": node_id,
                                        "op_ref": nodes[node_id]["op_ref"],
                                        "op_kind": nodes[node_id]["op_kind"],
                                        "warp_group": -1 if node_id == 0 else 0,
                                    }
                                    for node_id in range(len(nodes))
                                ]
                            }
                        }
                    }
                ]
            }
        )
    )

    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    data_edge = problem.edges[1]
    data_delay = problem.edge_latency(data_edge) + problem.spill[data_edge.src]
    cycles = {node_id: 0 for node_id in problem.nodes}
    cycles[data_edge.dst] = data_delay
    length = data_delay + 1
    widths = {0: 1, 1: 2, 2: 1}
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": solution_provenance(ddg, graph, machine, 300),
        "ii": length,
        "length": length,
        "copies": 1,
        "horizon": length,
        "cycles": {str(node_id): cycle for node_id, cycle in cycles.items()},
        "warp": {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 2},
        "group_widths": {str(group): width for group, width in widths.items()},
        "lane_masks": {
            "0": [0],
            "1": [0],
            "2": [0],
            "3": [1],
            "4": [0],
            "5": [0],
        },
        "stats": _solver_stats(machine, problem, widths, length),
    }
    solution = tmp_path / "ws_edges_solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph, data_delay


def _write_tmem_overlap_artifact(tmp_path):
    ddg = tmp_path / "tmem_overlap.json"
    graph = tmp_path / "tmem_overlap_graph.json"
    nodes = [
        {
            "id": node_id,
            "op_ref": op_ref,
            "op_kind": op_kind,
            "pipeline": "NONE",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
        }
        for node_id, (op_ref, op_kind) in enumerate(
            (
                ("alloc_a", "ttng.tmem_alloc"),
                ("use_a", "arith.addi"),
                ("alloc_b", "ttng.tmem_alloc"),
                ("use_b", "arith.addi"),
            )
        )
    ]
    tmem_type = (
        "!ttg.memdesc<64x128xf32, "
        "#ttng.tensor_memory_encoding<blockM = 64, blockN = 128, "
        "colStride = 1>, #ttng.tensor_memory, mutable>"
    )
    ddg.write_text(
        json.dumps(
            {
                "schema_version": "ddg-0.1",
                "ops": {
                    "alloc_a": {
                        "result_types": [tmem_type, "!ttg.async.token"]
                    },
                    "use_a": {"result_types": []},
                    "alloc_b": {
                        "result_types": [tmem_type, "!ttg.async.token"]
                    },
                    "use_b": {"result_types": []},
                },
                "loops": [
                    {
                        "trip_count": 4,
                        "min_ii": 1,
                        "ddg": {
                            "nodes": nodes,
                            "edges": [
                                {
                                    "src": 0,
                                    "dst": 1,
                                    "latency": 1,
                                    "src_result_idx": 1,
                                },
                                {
                                    "src": 2,
                                    "dst": 3,
                                    "latency": 1,
                                    "src_result_idx": 1,
                                },
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph.write_text(
        json.dumps(
            {
                "loops": [
                    {
                        "schedule_loop": {
                            "graph": {
                                "nodes": [
                                    {
                                        "id": node["id"],
                                        "op_ref": node["op_ref"],
                                        "op_kind": node["op_kind"],
                                        "warp_group": 0,
                                    }
                                    for node in nodes
                                ]
                            }
                        }
                    }
                ]
            }
        )
    )

    machine = MachineModel(tmem_cols=64)
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    delay = problem.edge_latency(problem.edges[0])
    cycles = {0: 0, 1: delay, 2: delay, 3: delay * 2}
    length = delay * 2 + 1
    widths = {0: 1}
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": solution_provenance(ddg, graph, machine, 300),
        "ii": length,
        "length": length,
        "copies": 1,
        "horizon": length,
        "cycles": {str(node_id): cycle for node_id, cycle in cycles.items()},
        "warp": {str(node_id): 0 for node_id in problem.nodes},
        "group_widths": {"0": 1},
        "lane_masks": {str(node_id): [0] for node_id in problem.nodes},
        "stats": _solver_stats(machine, problem, widths, length),
    }
    solution = tmp_path / "tmem_overlap_solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph, delay


def _write_carried_register_artifact(tmp_path):
    ddg = tmp_path / "carried_register.json"
    graph = tmp_path / "carried_register_graph.json"
    nodes = [
        {
            "id": 0,
            "op_ref": "producer",
            "op_kind": "arith.addf",
            "pipeline": "CUDA",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
        },
        {
            "id": 1,
            "op_ref": "consumer",
            "op_kind": "arith.addf",
            "pipeline": "CUDA",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
        },
    ]
    ddg.write_text(
        json.dumps(
            {
                "schema_version": "ddg-0.1",
                "ops": {
                    "producer": {"result_types": ["f32"]},
                    "consumer": {"result_types": []},
                },
                "loops": [
                    {
                        "trip_count": 4,
                        "min_ii": 1,
                        "ddg": {
                            "nodes": nodes,
                            "edges": [
                                {
                                    "src": 0,
                                    "dst": 1,
                                    "distance": 1,
                                    "latency": 0,
                                    "src_result_idx": 0,
                                }
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph.write_text(
        json.dumps(
            {
                "loops": [
                    {
                        "schedule_loop": {
                            "graph": {
                                "nodes": [
                                    {
                                        "id": node["id"],
                                        "op_ref": node["op_ref"],
                                        "op_kind": node["op_kind"],
                                        "warp_group": 0,
                                    }
                                    for node in nodes
                                ]
                            }
                        }
                    }
                ]
            }
        )
    )

    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    ii = problem.spill[0]
    widths = {0: 1}
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": solution_provenance(ddg, graph, machine, 300),
        "ii": ii,
        "length": 1,
        "copies": 1,
        "horizon": 1,
        "cycles": {"0": 0, "1": 0},
        "warp": {"0": 0, "1": 0},
        "group_widths": {"0": 1},
        "lane_masks": {"0": [0], "1": [0]},
        "stats": _solver_stats(machine, problem, widths, 1),
    }
    solution = tmp_path / "carried_register_solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def test_handoff_emits_pipelined_warp_annotated_ir(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    result = _handoff(tmp_path, solution, ddg, graph)

    ir_text = result.ir_path.read_text()
    assert "tlx.async_tasks" not in ir_text
    assert "@triton.jit" not in ir_text
    assert "sched2tlx-standalone" not in ir_text
    ir = json.loads(ir_text)
    assert ir["schema_version"] == "twill-pipelined-warp-ir-v2"
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

    program = ir["pipelined_program"]
    node_count = len(ir["instructions"])
    assert len(program["instances"]) == node_count * ir["pipeline"]["copies"]
    assert len(program["steady_state"]["slots"]) == node_count
    assert {slot["node"] for slot in program["steady_state"]["slots"]} == {
        instruction["id"] for instruction in ir["instructions"]
    }
    stages = {
        instruction["id"]: instruction["stage"]
        for instruction in ir["instructions"]
    }
    assert all(
        slot["iteration_lag"] == stages[slot["node"]]
        for slot in program["steady_state"]["slots"]
    )
    assert sum(
        instance["region"] != "steady_state"
        for instance in program["instances"]
    ) == node_count * (ir["pipeline"]["copies"] - 1)
    carried = [
        dependency
        for dependency in program["instance_dependencies"]
        if dependency["carried_distance"] == 1
    ]
    assert carried
    assert carried[0]["external"] is True
    assert carried[0]["producer"]["copy"] == -1

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
    mixed_lane_edges = {
        (edge["src"], edge["dst"]): edge
        for edge in ir["cross_warp_dependencies"]
        if edge["producer_group"] == edge["consumer_group"]
    }
    assert set(mixed_lane_edges) == {(1, 4), (3, 4)}
    assert all(
        edge["producer_lanes"] != edge["consumer_lanes"]
        for edge in mixed_lane_edges.values()
    )


def test_handoff_excludes_infra_edge_but_keeps_ws_cross_edges(tmp_path):
    solution, ddg, graph, data_delay = _write_ws_edge_artifact(tmp_path)
    plan = load_schedule_plan(solution, ddg, graph)
    edges = {edge.index: edge for edge in plan.edges}

    assert edges[0].ws_semantics is False
    assert edges[0].producer_group != edges[0].consumer_group
    assert edges[0].spill_cost == 0
    assert edges[1].ws_semantics is True
    assert edges[1].producer_group == edges[1].consumer_group
    assert edges[1].producer_lanes != edges[1].consumer_lanes
    assert edges[1].spill_cost > 0
    assert edges[1].latency + edges[1].spill_cost == data_delay
    assert edges[2].ws_semantics is True
    assert edges[2].spill_cost == 0

    result = _handoff(tmp_path, solution, ddg, graph, "ws_edges")
    ir = json.loads(result.ir_path.read_text())
    dependencies = {edge["index"]: edge for edge in ir["dependencies"]}
    cross_edges = {edge["index"] for edge in ir["cross_warp_dependencies"]}
    assert dependencies[0]["ws_semantics"] is False
    assert dependencies[1]["ws_semantics"] is True
    assert dependencies[1]["spill_cost"] == edges[1].spill_cost
    assert cross_edges == {1, 2}


def test_emit_gate_rejects_missing_cross_lane_spill_slack(tmp_path):
    solution, ddg, graph, data_delay = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["cycles"]["3"] = data_delay - 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="dependence violation"):
        load_schedule_plan(solution, ddg, graph)


@pytest.mark.parametrize(
    ("machine_field", "stats_field", "message"),
    (
        ("regs_per_warp", "regs_per_warp", "REGISTERLIMIT"),
        ("regs_per_sm", "regs_per_sm", "register-file capacity"),
    ),
)
def test_emit_gate_revalidates_register_capacity(
    tmp_path, machine_field, stats_field, message
):
    solution, ddg, graph, _ = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"][machine_field] = 3
    payload["stats"][stats_field] = 3
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match=message):
        load_schedule_plan(solution, ddg, graph)


def test_emit_gate_charges_cross_lane_spill_staging_to_smem(tmp_path):
    solution, ddg, graph, _ = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["smem_bytes"] = 31
    payload["stats"]["smem_capacity"] = 31
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="SMEM capacity"):
        load_schedule_plan(solution, ddg, graph)


def test_emit_gate_rejects_tampered_tmem_overlap(tmp_path):
    solution, ddg, graph, delay = _write_tmem_overlap_artifact(tmp_path)
    load_schedule_plan(solution, ddg, graph)

    payload = json.loads(solution.read_text())
    payload["cycles"]["2"] = delay - 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="TMEM capacity"):
        load_schedule_plan(solution, ddg, graph)


def test_emit_gate_rejects_carried_register_lane_tamper(tmp_path):
    solution, ddg, graph = _write_carried_register_artifact(tmp_path)
    load_schedule_plan(solution, ddg, graph)

    payload = json.loads(solution.read_text())
    payload["lane_masks"]["1"] = [1]
    payload["group_widths"]["0"] = 2
    payload["stats"]["scheduled_warps"] = 2
    payload["stats"]["num_warps"] = 2
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="reg-carried same-lane"):
        load_schedule_plan(solution, ddg, graph)


def test_handoff_with_stale_source_op_refs_fails_scaffold_input_gate(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path, stale_source_ops=True)
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


def _write_concurrency_artifact(tmp_path, other_latency):
    ddg = tmp_path / f"concurrency_{other_latency}.json"
    graph = tmp_path / f"concurrency_{other_latency}_graph.json"
    nodes = [
        {
            "id": 0,
            "op_ref": "producer",
            "op_kind": "ttng.tc_gen5_mma",
            "pipeline": "TC",
            "latency": 4,
            "occupancy": 1,
            "min_warps": 1,
        },
        {
            "id": 1,
            "op_ref": "waiter",
            "op_kind": "arith.addi",
            "pipeline": "NONE",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
        },
        {
            "id": 2,
            "op_ref": "other",
            "op_kind": "arith.muli",
            "pipeline": "NONE",
            "latency": other_latency,
            "occupancy": 0,
            "min_warps": 1,
        },
    ]
    ddg.write_text(
        json.dumps(
            {
                "schema_version": "ddg-0.1",
                "ops": {
                    node["op_ref"]: {"result_types": ["f32"]}
                    for node in nodes
                },
                "loops": [
                    {
                        "trip_count": 4,
                        "min_ii": 1,
                        "ddg": {
                            "nodes": nodes,
                            "edges": [
                                {
                                    "src": 0,
                                    "dst": 1,
                                    "distance": 0,
                                    "latency": 4,
                                    "src_result_idx": 0,
                                }
                            ],
                        },
                    }
                ],
            }
        )
    )
    graph.write_text(
        json.dumps(
            {
                "loops": [
                    {
                        "schedule_loop": {
                            "graph": {
                                "nodes": [
                                    {
                                        "id": node["id"],
                                        "op_ref": node["op_ref"],
                                        "op_kind": node["op_kind"],
                                        "warp_group": 0,
                                    }
                                    for node in nodes
                                ]
                            }
                        }
                    }
                ]
            }
        )
    )
    machine = MachineModel()
    problem = load_problem(ddg, machine=machine, baseline_graph=graph)
    wait_cycle = problem.edge_latency(problem.edges[0])
    other_cycle = wait_cycle if other_latency == 0 else wait_cycle - 1
    cycles = {"0": 0, "1": wait_cycle, "2": other_cycle}
    length = max(
        cycles[str(node_id)] + max(1, problem.lat[node_id])
        for node_id in problem.nodes
    )
    ii = length
    copies = 1
    widths = {0: 1}
    payload = {
        "schema_version": SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": solution_provenance(ddg, graph, machine, 300),
        "ii": ii,
        "length": length,
        "copies": copies,
        "horizon": length,
        "cycles": cycles,
        "warp": {str(node_id): 0 for node_id in problem.nodes},
        "group_widths": {"0": 1},
        "lane_masks": {str(node_id): [0] for node_id in problem.nodes},
        "stats": _solver_stats(machine, problem, widths, length),
    }
    solution = tmp_path / f"concurrency_{other_latency}_solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def test_handoff_preserves_paper_valid_partial_lane_roles(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    result = _handoff(tmp_path, solution, ddg, graph, "partial")
    instructions = json.loads(result.ir_path.read_text())["instructions"]
    assert any(
        len(node["lanes"]) < node["group_width"]
        for node in instructions
        if node["group_width"] is not None
    )


def test_emit_gate_allows_zero_latency_coissue_with_blocking_wait(tmp_path):
    solution, ddg, graph = _write_concurrency_artifact(tmp_path, 0)
    plan = load_schedule_plan(solution, ddg, graph)
    assert plan.cycles[1] == plan.cycles[2]


def test_emit_gate_rejects_full_concurrency_window(tmp_path):
    solution, ddg, graph = _write_concurrency_artifact(tmp_path, 2)
    with pytest.raises(ScheduleArtifactError, match="CONCURRENCY violation"):
        load_schedule_plan(solution, ddg, graph)


def test_provenance_mismatch_is_rejected(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["ddg_sha256"] = "0" * 64
    solution.write_text(json.dumps(payload))
    with pytest.raises(ScheduleArtifactError, match="ddg_sha256 mismatch"):
        _handoff(tmp_path, solution, ddg, graph)


@pytest.mark.parametrize(
    "field", ("normalization_u_effective", "model_options")
)
def test_manual_handoff_requires_explicit_model_provenance(tmp_path, field):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    del payload["provenance"][field]
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match=field):
        _handoff(tmp_path, solution, ddg, graph)


def test_manual_handoff_rejects_bool_int_model_option_alias(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["model_options"]["ignore_fixed_overheads"] = 0
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="diagnostic model options"):
        _handoff(tmp_path, solution, ddg, graph)


def test_schedule_context_reconstructs_provenance_machine(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["regs_per_warp"] = 4096
    payload["stats"]["regs_per_warp"] = 4096
    solution.write_text(json.dumps(payload))

    problem, plan = load_schedule_context(solution, ddg, graph)

    assert problem.machine.regs_per_warp == plan.machine["regs_per_warp"] == 4096


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
