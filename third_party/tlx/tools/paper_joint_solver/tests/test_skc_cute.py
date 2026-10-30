"""Manual CUDA handoff and retired CuTe-shim regression tests."""

import ast
import json
from pathlib import Path

import pytest

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.schedule_plan import (
    EMITTER_MODEL_PROFILE,
    EMITTER_SOLUTION_SCHEMA,
    PAPER_MODEL_PROFILE,
    PAPER_SOLUTION_SCHEMA,
    LegacySolutionError,
    ScheduleArtifactError,
    experiment_inputs,
    load_schedule_context,
    load_schedule_plan,
    solution_provenance,
)
from paper_joint_solver.pipelined_ir import (
    MANUAL_CUDA_HANDOFF_SCHEMA,
    PIPELINED_IR_SCHEMA,
)
from skc import ArtifactError
from skc.compiler import prepare_manual_cuda_handoff, scaffold_manual_cuda
from skc_cute import binder_cute, driver

PKG = Path(__file__).resolve().parent.parent
EXAMPLES = PKG.parent / "sched2tlx" / "examples"


def _solver_stats(machine, problem, widths, horizon):
    """Solver statistics are pure observations; nothing revalidates them."""
    return {
        **problem.normalization_stats(),
        "T": horizon,
        "profile": "paper",
        "physical_warp_budget": machine.num_warps,
        "regs_per_warp": machine.regs_per_warp,
        "smem_capacity": machine.smem_bytes,
        "tmem_capacity": machine.tmem_cols,
    }


def _warp_sets(warp_map, lane_map, widths, *, first_warp=1):
    """Lay the logical groups of a fixture out on consecutive physical warps
    and read each node's warp set through its lane mask.  Physical warp 0
    belongs to the variable-latency operation, so a fixture without one
    starts at warp 1."""
    base, cursor = {}, first_warp
    for group in sorted(widths):
        base[group] = cursor
        cursor += widths[group]
    return {
        node: sorted(base[int(group)] + lane for lane in lane_map[node])
        for node, group in warp_map.items()
    }


def _paper_provenance(ddg, machine, inputs=None):
    return solution_provenance(
        ddg,
        machine,
        300,
        model=PAPER_MODEL_PROFILE,
        inputs=inputs or experiment_inputs(True, None, None),
    )


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


def test_benchmark_registry_does_not_expose_tlx_under_paper_names():
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


def _curate_to_tmp(raw_ddg, graph):
    """Curate a raw fixture pair and return the curated_ddg.json path.

    The paper path consumes curated G only, so every fixture that used to hand
    `load_problem` a raw dump plus a baseline graph now curates first.
    """
    import tempfile

    from paper_joint_solver.curate_ddg import curate, dumps

    curated, _manifest = curate(raw_ddg, graph)
    out = Path(tempfile.mkdtemp()) / "curated_ddg.json"
    out.write_text(dumps(curated))
    return out


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
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
    cycles = {
        str(node["id"]): node["schedule"]["cycle"]
        for node in schedule["graph"]["nodes"]
        if node["id"] in problem.nodes
    }
    # Keep the blocking wrappers and MMA in one two-lane group while assigning
    # them disjoint physical lanes. This exercises same-group cross-warp edges
    # without violating CONCURRENCY.
    warp = {}
    lane_masks = {}
    for node in schedule["graph"]["nodes"]:
        node_id = node["id"]
        if node_id not in problem.nodes:
            continue
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
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": ii,
        "length": length,
        "copies": copies,
        "horizon": horizon,
        "cycles": cycles,
        # Group 0 is the variable-latency group and owns physical warp 0.
        "warp_sets": _warp_sets(warp, lane_masks, widths, first_warp=0),
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
    payload["schema_version"] = EMITTER_SOLUTION_SCHEMA
    payload.pop("warp_sets")
    machine = MachineModel()
    problem = load_problem(ddg, machine=machine)
    graph_data = json.loads(graph.read_text())
    nodes = graph_data["loops"][0]["schedule_loop"]["graph"]["nodes"]
    warp = {}
    for node in nodes:
        node_id = node["id"]
        if node_id not in problem.nodes:
            continue
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
        machine,
        300,
        model=EMITTER_MODEL_PROFILE,
        baseline_graph_path=graph,
    )
    payload["stats"] = _solver_stats(machine, problem, widths, payload["horizon"])
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
    # An emitter artifact is not a paper artifact and cannot be coerced.
    with pytest.raises(LegacySolutionError, match="is not paper-joint-solution"):
        load_schedule_plan(solution, ddg, graph)
    mislabelled = json.loads(solution.read_text())
    mislabelled["schema_version"] = PAPER_SOLUTION_SCHEMA
    mislabelled["warp_sets"] = {"0": [0]}
    for key in ("warp", "group_widths", "lane_masks"):
        mislabelled.pop(key)
    mislabelled_path = solution.parent / "mislabelled.json"
    mislabelled_path.write_text(json.dumps(mislabelled))
    # The branch is decided by provenance.model alone; there are no options.
    with pytest.raises(ScheduleArtifactError, match="is not the requested"):
        load_schedule_plan(mislabelled_path, ddg, graph)

    plan = load_schedule_plan(
        solution,
        ddg,
        graph,
        model_profile=EMITTER_MODEL_PROFILE,
    )
    assert plan.model == EMITTER_MODEL_PROFILE
    assert all(node.lanes == (0,) for node in plan.nodes)
    assert all(node.warps == () for node in plan.nodes)


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
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
    # Curation drops the infra node and re-sorts E, so pick the data edge
    # by identity rather than by position.
    data_edge = next(
        edge for edge in problem.edges if (edge.src, edge.dst) == (2, 3)
    )
    data_delay = problem.edge_latency(data_edge) + problem.spill[data_edge.src]
    cycles = {node_id: 0 for node_id in problem.nodes}
    cycles[data_edge.dst] = data_delay
    length = data_delay + 1
    # Node 0 is emitter infra; curation dropped it, so it carries no
    # assignment any more.
    warp_map = {
        key: group
        for key, group in {"0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 2}.items()
        if int(key) in problem.nodes
    }
    lane_map = {
        key: lanes
        for key, lanes in {
            "0": [0], "1": [0], "2": [0], "3": [1], "4": [0], "5": [0],
        }.items()
        if int(key) in problem.nodes
    }
    widths = {
        group: width
        for group, width in {0: 1, 1: 2, 2: 1}.items()
        if group in set(warp_map.values())
    }
    payload = {
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": length,
        "length": length,
        "copies": 1,
        "horizon": length,
        "cycles": {str(node_id): cycle for node_id, cycle in cycles.items()},
        "warp_sets": _warp_sets(warp_map, lane_map, widths),
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
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
    delay = problem.edge_latency(problem.edges[0])
    cycles = {0: 0, 1: delay, 2: delay, 3: delay * 2}
    length = delay * 2 + 1
    widths = {0: 1}
    payload = {
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": length,
        "length": length,
        "copies": 1,
        "horizon": length,
        "cycles": {str(node_id): cycle for node_id, cycle in cycles.items()},
        "warp_sets": {str(node_id): [1] for node_id in problem.nodes},
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
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
    ii = problem.spill[0]
    widths = {0: 1}
    payload = {
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": ii,
        "length": 1,
        "copies": 1,
        "horizon": 1,
        "cycles": {"0": 0, "1": 0},
        "warp_sets": {"0": [1], "1": [1]},
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
    assert ir["schema_version"] == PIPELINED_IR_SCHEMA
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
    assert manifest["schema_version"] == MANUAL_CUDA_HANDOFF_SCHEMA
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
    # Every cross edge is one whose producer and consumer warp sets differ;
    # there is no group/lane layer left to distinguish.
    assert all(
        edge["producer_warps"] != edge["consumer_warps"]
        for edge in ir["cross_warp_dependencies"]
    )
    assert not any(
        key in ir["instructions"][0]
        for key in ("group", "group_width", "lanes")
    )


def test_handoff_drops_the_infra_edge_and_keeps_ws_cross_edges(tmp_path):
    """The infra edge is gone because curation removed its node (C1), not
    because the solver stripped its WS semantics."""
    solution, ddg, graph, data_delay = _write_ws_edge_artifact(tmp_path)
    plan = load_schedule_plan(solution, ddg, graph)
    by_pair = {(edge.src, edge.dst): edge for edge in plan.edges}

    assert (0, 1) not in by_pair
    data = by_pair[(2, 3)]
    assert data.ws_semantics is True
    assert data.producer_warps != data.consumer_warps
    assert data.spill_cost > 0
    assert data.latency + data.spill_cost == data_delay
    completion = by_pair[(4, 5)]
    assert completion.ws_semantics is True
    assert completion.spill_cost == 0

    result = _handoff(tmp_path, solution, ddg, graph, "ws_edges")
    ir = json.loads(result.ir_path.read_text())
    pairs = {(edge["src"], edge["dst"]) for edge in ir["dependencies"]}
    assert (0, 1) not in pairs
    dependencies = {
        (edge["src"], edge["dst"]): edge for edge in ir["dependencies"]
    }
    assert dependencies[(2, 3)]["spill_cost"] == data.spill_cost
    cross_pairs = {
        (edge["src"], edge["dst"]) for edge in ir["cross_warp_dependencies"]
    }
    assert cross_pairs == {(2, 3), (4, 5)}


def test_emit_gate_rejects_missing_cross_lane_spill_slack(tmp_path):
    solution, ddg, graph, data_delay = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["cycles"]["3"] = data_delay - 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="dependence violation"):
        load_schedule_plan(solution, ddg, graph)


def test_emit_gate_revalidates_register_capacity(tmp_path):
    """Fig 6 REGISTERLIMIT is per physical warp -- the only register bound
    the paper states."""
    solution, ddg, graph, _ = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["regs_per_warp"] = 3
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="REGISTERLIMIT"):
        load_schedule_plan(solution, ddg, graph)


def test_paper_branch_has_no_sm_wide_register_bound(tmp_path):
    """Fig 6 has no SM-level register-file constraint, so shrinking the SM
    budget alone cannot reject a paper artifact."""
    solution, ddg, graph, _ = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["regs_per_sm"] = 3
    solution.write_text(json.dumps(payload))

    assert load_schedule_plan(solution, ddg, graph).ii > 0


def test_paper_branch_has_no_spill_smem_charge(tmp_path):
    """The paper charges a cross-warp spill only as schedule delay; it never
    reserves shared memory or barrier bytes for it."""
    solution, ddg, graph, _ = _write_ws_edge_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["smem_bytes"] = 31
    solution.write_text(json.dumps(payload))

    plan = load_schedule_plan(solution, ddg, graph)
    assert any(edge.spill_cost > 0 for edge in plan.edges)


def test_emit_gate_rejects_tampered_tmem_overlap(tmp_path):
    solution, ddg, graph, delay = _write_tmem_overlap_artifact(tmp_path)
    load_schedule_plan(solution, ddg, graph)

    payload = json.loads(solution.read_text())
    payload["cycles"]["2"] = delay - 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="TMEM capacity"):
        load_schedule_plan(solution, ddg, graph)


def test_paper_branch_accepts_cross_warp_carried_register_edge(tmp_path):
    """CROSS-WARPSPILLS charges a distance>=1 register edge its spill delay
    like any other; the paper has no same-lane placement rule."""
    solution, ddg, graph = _write_carried_register_artifact(tmp_path)
    load_schedule_plan(solution, ddg, graph)

    payload = json.loads(solution.read_text())
    payload["warp_sets"]["1"] = [2]
    solution.write_text(json.dumps(payload))

    plan = load_schedule_plan(solution, ddg, graph)
    carried = next(edge for edge in plan.edges if edge.distance == 1)
    assert carried.spill_cost > 0


def test_emitter_branch_still_rejects_reg_carried_cross_lane(tmp_path):
    """The frozen emitter model keeps the same-lane placement requirement."""
    solution, ddg, graph = _write_carried_register_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["schema_version"] = EMITTER_SOLUTION_SCHEMA
    payload.pop("warp_sets")
    payload["warp"] = {"0": 0, "1": 1}
    payload["group_widths"] = {"0": 1, "1": 1}
    payload["lane_masks"] = {"0": [0], "1": [0]}
    machine = MachineModel()
    payload["provenance"] = solution_provenance(
        ddg,
        machine,
        300,
        model=EMITTER_MODEL_PROFILE,
        baseline_graph_path=graph,
    )
    emitter = solution.parent / "carried_register_emitter.json"
    emitter.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="reg-carried same-lane"):
        load_schedule_plan(
            emitter, ddg, graph, model_profile=EMITTER_MODEL_PROFILE
        )


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
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
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
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": ii,
        "length": length,
        "copies": copies,
        "horizon": length,
        "cycles": cycles,
        "warp_sets": {str(node_id): [1] for node_id in problem.nodes},
        "stats": _solver_stats(machine, problem, widths, length),
    }
    solution = tmp_path / f"concurrency_{other_latency}_solution.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph


def test_handoff_records_one_issue_trace_per_physical_warp(tmp_path):
    """The paper IR is expanded per physical warp; a node in several warps
    appears in every member's trace, in the same relative order."""
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    result = _handoff(tmp_path, solution, ddg, graph, "partial")
    ir = json.loads(result.ir_path.read_text())
    warps = {entry["id"]: entry for entry in ir["warps"]}
    instructions = {node["id"]: node for node in ir["instructions"]}

    assert set(warps) == {
        warp for node in instructions.values() for warp in node["warps"]
    }
    for warp_id, entry in warps.items():
        traced = [item["node"] for item in entry["issue_trace"]]
        assert traced == sorted(
            (
                node_id
                for node_id, node in instructions.items()
                if warp_id in node["warps"]
            ),
            key=lambda node_id: (
                instructions[node_id]["stage"],
                instructions[node_id]["offset"],
                node_id,
            ),
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


def test_manual_handoff_requires_the_paper_model_constant(tmp_path):
    """The only model judgement left is a single provenance constant."""
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    del payload["provenance"]["model"]
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="is not the requested"):
        _handoff(tmp_path, solution, ddg, graph)


def test_manual_handoff_requires_the_curation_hash_chain(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["curation_source"]["ddg_sha256"] = "0" * 64
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="curation_source mismatch"):
        _handoff(tmp_path, solution, ddg, graph)


def test_manual_handoff_rejects_a_nonzero_fixed_overhead(tmp_path):
    """A paper artifact carries no fixed overhead; a non-zero one is a
    hand-edited artifact, not a model choice."""
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["warp_fixed_overhead"] = 1
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="no fixed overhead"):
        _handoff(tmp_path, solution, ddg, graph)


def test_stats_drift_no_longer_rejects_a_handoff(tmp_path):
    """Solver statistics are observations; only their object shape matters."""
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["stats"] = {"anything": "at all"}
    solution.write_text(json.dumps(payload))

    assert _handoff(tmp_path, solution, ddg, graph, "stats").ir_path.exists()


def test_schedule_context_reconstructs_provenance_machine(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["provenance"]["machine"]["regs_per_warp"] = 4096
    solution.write_text(json.dumps(payload))

    problem, plan = load_schedule_context(solution, ddg)

    assert problem.machine.regs_per_warp == plan.machine["regs_per_warp"] == 4096


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda sets: sets.__setitem__("1", [1, 1]), "repeats a physical warp"),
        (lambda sets: sets.__setitem__("1", []), "empty warp set"),
        (lambda sets: sets.__setitem__("1", [64]), "leaves the physical budget"),
        (lambda sets: sets.__setitem__("1", [1, 2]), "requires 1"),
        (lambda sets: sets.pop("1"), "must cover exactly the DDG nodes"),
    ),
)
def test_paper_branch_rejects_malformed_warp_sets(tmp_path, mutation, message):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    mutation(payload["warp_sets"])
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match=message):
        load_schedule_plan(solution, ddg, graph)


def test_paper_branch_rejects_group_and_lane_fields(tmp_path):
    """No compatibility layer: an old-shaped artifact is refused outright."""
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    payload["warp"] = {"0": 0}
    solution.write_text(json.dumps(payload))

    with pytest.raises(ScheduleArtifactError, match="group/lane fields"):
        load_schedule_plan(solution, ddg, graph)


def _write_multi_warp_artifact(tmp_path, *, regs_per_warp):
    """A two-warp producer feeding a one-warp consumer that shares one of its
    warps: exercises the per-member ceil(regs/warps) charge on overlapping
    sets."""
    ddg = tmp_path / "multi_warp.json"
    graph = tmp_path / "multi_warp_graph.json"
    nodes = [
        {
            "id": 0,
            "op_ref": "producer",
            "op_kind": "arith.addf",
            "pipeline": "CUDA",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 2,
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
                    "producer": {"result_types": ["tensor<8xf32>"]},
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
    machine = MachineModel(regs_per_warp=regs_per_warp)
    ddg = _curate_to_tmp(ddg, graph)
    problem = load_problem(ddg, machine=machine)
    payload = {
        "schema_version": PAPER_SOLUTION_SCHEMA,
        "status": "sat",
        "satisfiable": True,
        "provenance": _paper_provenance(ddg, machine),
        "ii": max(1, problem.spill[0]),
        "length": 1,
        "copies": 1,
        "horizon": 1,
        "cycles": {"0": 0, "1": 0},
        "warp_sets": {"0": [1, 2], "1": [1]},
        "stats": {},
    }
    solution = tmp_path / f"multi_warp_{regs_per_warp}.json"
    solution.write_text(json.dumps(payload))
    return solution, ddg, graph, problem


def test_paper_registerlimit_charges_ceil_regs_over_min_warps(tmp_path):
    solution, ddg, graph, problem = _write_multi_warp_artifact(
        tmp_path, regs_per_warp=4
    )
    # 8 register words spread over the node's two warps: 4 each.
    assert problem.regs[0] == 8
    plan = load_schedule_plan(solution, ddg, graph)
    assert plan.warp_sets == {0: (1, 2), 1: (1,)}

    solution, ddg, graph, _ = _write_multi_warp_artifact(
        tmp_path, regs_per_warp=3
    )
    with pytest.raises(ScheduleArtifactError, match="REGISTERLIMIT"):
        load_schedule_plan(solution, ddg, graph)


def test_paper_manifest_carries_warp_sets_only(tmp_path):
    solution, ddg, graph = _write_case1_artifact(tmp_path)
    manifest = load_schedule_plan(solution, ddg, graph).manifest()

    assert manifest["schema_version"] == PAPER_SOLUTION_SCHEMA
    assert "groups" not in manifest
    assert {entry["id"] for entry in manifest["warps"]} == {
        warp for node in manifest["nodes"] for warp in node["warps"]
    }
    for node in manifest["nodes"]:
        assert node["warps"]
        assert not {"group", "group_width", "lanes"} & node.keys()
    for edge in manifest["edges"]:
        assert not {
            "producer_group",
            "consumer_group",
            "producer_lanes",
            "consumer_lanes",
        } & edge.keys()


def test_cross_warp_dependency_by_set_inequality(tmp_path):
    """Overlapping-but-unequal warp sets still cross a warp boundary."""
    from paper_joint_solver.pipelined_ir import _is_cross_warp_dependency

    same = {"ws_semantics": True, "producer_warps": [1, 2], "consumer_warps": [1, 2]}
    overlapping = {
        "ws_semantics": True,
        "producer_warps": [1, 2],
        "consumer_warps": [1],
    }
    assert _is_cross_warp_dependency(same) is False
    assert _is_cross_warp_dependency(overlapping) is True

    solution, ddg, graph, _ = _write_multi_warp_artifact(
        tmp_path, regs_per_warp=4
    )
    ir = json.loads(
        _handoff(tmp_path, solution, ddg, graph, "multi").ir_path.read_text()
    )
    assert [
        (edge["src"], edge["dst"]) for edge in ir["cross_warp_dependencies"]
    ] == [(0, 1)]


def test_pipeline_regions_degenerate_when_length_lt_ii(tmp_path):
    """The paper's region formulas have no L >= I precondition (I-4): an
    empty prologue and an inverted epilogue are the literal result."""
    solution, ddg, graph = _write_carried_register_artifact(tmp_path)
    payload = json.loads(solution.read_text())
    # A loop-carried edge lets II exceed the schedule length outright.
    payload["ii"] = 3
    solution.write_text(json.dumps(payload))
    plan = load_schedule_plan(solution, ddg, graph)
    assert plan.length < plan.ii

    ir = json.loads(
        _handoff(tmp_path, solution, ddg, graph, "degenerate").ir_path.read_text()
    )
    regions = {region["kind"]: region for region in ir["pipeline"]["regions"]}
    assert regions["prologue"]["cycle_begin"] == regions["prologue"]["cycle_end"]
    assert regions["epilogue"]["cycle_begin"] > regions["epilogue"]["cycle_end"]
    assert all(
        instance["region"] == "steady_state"
        for instance in ir["pipelined_program"]["instances"]
    )
