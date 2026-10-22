import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import NormalizationResult
from paper_joint_solver.resource_model import register_words

GOLDEN = Path(__file__).resolve().parent / "golden_curated"
FIXTURES = {
    "fwd": GOLDEN / "fwd" / "fwd.curated_ddg.json",
    "subtiled": GOLDEN / "fwd_subtiled" / "fwd_subtiled.curated_ddg.json",
    "bwd": GOLDEN / "bwd" / "bwd.curated_ddg.json",
}


@pytest.fixture(scope="module", params=list(FIXTURES))
def prob(request):
    return request.param, load_problem(FIXTURES[request.param])


@pytest.fixture(scope="module", params=list(FIXTURES))
def low_resolution_prob(request):
    """Small normalization used only to exercise the modulo ILP mechanics."""
    return request.param, load_problem(FIXTURES[request.param])


def _write_ddg(tmp_path, nodes, edges, name="ddg"):
    """Write a curated-ddg-0.2 fixture.

    Node/edge dicts are given in the same shorthand the raw-dump helper used;
    the curation rules that would have filled the remaining fields (C4 rrt,
    C6 blocking, C8 variable_latency/streaming) are applied here so fixtures
    stay readable. Anything passed explicitly wins.
    """
    prepared = []
    incoming = {edge["dst"] for edge in edges}
    for values in nodes:
        values = values.copy()
        result_types = values.pop("result_types", ["i32"])
        node = {
            "op_ref": f"op_{values['id']}",
            "op_kind": "arith.addi",
            "pipeline": "NONE",
            "latency": 0,
            "occupancy": 0,
            "min_warps": 1,
            "regs": 0,
            "spill_cost": 0,
            "smem_footprint": 0,
            "tmem_footprint": 0,
            **values,
        }
        if "regs" not in values:
            node["regs"] = register_words(result_types)
        if "rrt" not in node:
            unit, latency, occupancy = (
                node["pipeline"],
                node["latency"],
                node["occupancy"],
            )
            node["rrt"] = (
                {unit: [1] * occupancy + [0] * (latency - occupancy)}
                if unit in MachineModel().capacities and latency > 0
                else {}
            )
        if "variable_latency" not in node:
            node["variable_latency"] = (
                node["pipeline"] == "TMA" and "load" in node["op_kind"]
            )
        if "streaming" not in node:
            node["streaming"] = (
                node["variable_latency"] and node["id"] not in incoming
            )
        prepared.append(node)
    by_id = {node["id"]: node for node in prepared}
    prepared_edges = []
    for edge in edges:
        edge = {"distance": 0, "latency": 0, **edge}
        if "blocking" not in edge:
            edge["blocking"] = (
                by_id[edge["src"]]["pipeline"] in ("TC", "TMA")
            )
        prepared_edges.append(edge)
    payload = {
        "schema_version": "curated-ddg-0.2",
        "curation_source": {
            "ddg_sha256": "0" * 64,
            "baseline_graph_sha256": "0" * 64,
            "curator_sources_sha256": "0" * 64,
        },
        "loops": [
            {
                "loop_id": 0,
                "trip_count": 4,
                "ddg": {"nodes": prepared, "edges": prepared_edges},
            }
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _write_baseline(tmp_path, warp_groups, name="baseline"):
    payload = {
        "loops": [
            {
                "schedule_loop": {
                    "graph": {
                        "nodes": [
                            {"id": node_id, "warp_group": warp_group}
                            for node_id, warp_group in warp_groups.items()
                        ]
                    }
                }
            }
        ]
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_and_derive(prob):
    name, p = prob
    assert p.nodes and p.edges
    mmas = [v for v in p.nodes.values() if "mma" in v.op_kind]
    assert len(mmas) == {"fwd": 2, "subtiled": 4, "bwd": 5}[name]
    # Section 5.3 operates on G, which excludes emitter-infrastructure inputs.
    incoming = {edge.dst for edge in p.edges}
    expected_streaming = {
        node.id
        for node in p.nodes.values()
        if node.pipeline == "TMA"
        and "load" in node.op_kind
        and node.id not in incoming
    }
    assert p.streaming == expected_streaming
    assert p.streaming == {
        "fwd": {2, 3},
        "subtiled": {2, 3},
        "bwd": {0, 2},
    }[name]
    assert p.streaming <= p.variable_latency
    # Streaming outgoing latency is zeroed (sec 5.3).
    for e in p.edges:
        if e.src in p.streaming:
            assert p.edge_latency(e) == 0
    for node_id, rows in p.rrt.items():
        assert all(len(row) == p.lat[node_id] for row in rows.values())
    # Blocking edges exist (TC/TMA producers).
    assert p.blocking
    assert max(p.lat.values()) <= 300


def test_normalization_matches_paper_zlp(prob):
    name, p = prob
    costs = []
    scaled = []
    for node in p.nodes.values():
        active = p.occ[node.id]
        if node.pipeline in p.machine.capacities:
            if node.occupancy > 0:
                costs.append(node.occupancy)
                scaled.append(active)
            tail = node.latency - node.occupancy
            if tail > 0:
                costs.append(tail)
                scaled.append(p.lat[node.id] - active)
        elif node.latency > 0:
            costs.append(node.latency)
            scaled.append(p.lat[node.id])
    for edge in p.edges:
        latency = 0 if edge.src in p.streaming else edge.latency
        if latency > 0:
            costs.append(latency)
            scaled.append(p.edge_latency(edge))
    for producer, spill in p.spill.items():
        raw_spill = p.nodes[producer].spill_cost
        has_cross_edge = any(
            edge.src == producer and edge.dst != producer
            for edge in p.edges
        )
        if raw_spill > 0 and has_cross_edge:
            costs.append(raw_spill)
            scaled.append(spill)

    assert 1 <= sum(scaled) <= 300, name
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            error = abs(costs[i] * scaled[j] - costs[j] * scaled[i])
            assert error <= p.normalization_f, (name, i, j, error)


def test_modulo_ilp_finds_schedule_ascending_from_one(low_resolution_prob):
    pytest.importorskip("pyscipopt")
    from paper_joint_solver.modulo_ilp import solve_modulo

    name, p = low_resolution_prob
    # Algorithm 1 line 3-7: ascend I from 1 and let the ILP reject each
    # infeasible one; there is no lower bound to start from.
    sched = None
    for ii in range(1, 40):
        sched = solve_modulo(p, ii)
        if sched is not None:
            break
    assert sched is not None, f"{name}: no schedule in [1, 39]"
    ii = sched.ii
    # Validate dependences and modulo resource usage by hand.
    for e in p.edges:
        d = p.edge_latency(e)
        assert (sched.cycles[e.dst] >= sched.cycles[e.src] + d
                - e.distance * ii), (name, e)
    for pipe, cap in p.machine.capacities.items():
        for r in range(ii):
            use = sum(
                amount
                for node_id in p.nodes
                for functional_unit, cycle, amount in p.reservations(node_id)
                if functional_unit == pipe
                and (sched.cycles[node_id] + cycle) % ii == r
            )
            assert use <= cap, (name, pipe, r, use)


def test_zero_latency_emitter_node_is_assigned_before_horizon(tmp_path):
    pytest.importorskip("pyscipopt")
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint
    from paper_joint_solver.modulo_ilp import solve_modulo

    ddg = _write_ddg(tmp_path, [{"id": 0}], [])
    problem = load_problem(ddg)

    modulo = solve_modulo(problem, 1)
    with pytest.raises(ValueError, match="schedule length"):
        solve_joint(problem, 1, 0)
    solution, status = solve_joint(problem, 1, modulo.length)

    assert problem.regs[0] == 1
    assert modulo.length == 1
    assert status == "sat"
    assert solution.warp_sets.keys() == {0}
    # WARPUNIQUENESS gives it exactly one warp, and VARIABLELATENCY is an
    # iff, so a regular op never sits on W_vl.
    assert len(solution.warp_sets[0]) == 1
    assert 0 not in solution.warp_sets[0]
    assert 0 <= solution.cycles[0] < solution.horizon


def test_exact_warp_sets_probe_counts_distinct_sets(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint_audit

    ddg = _write_ddg(tmp_path, [{"id": 0}, {"id": 1}, {"id": 2}], [])
    problem = load_problem(ddg)
    separate = [(0, 1), (0, 2), (1, 2)]

    solution, status = solve_joint_audit(
        problem,
        1,
        1,
        num_warps_override=4,
        exact_warp_sets=3,
        separate=separate,
    )
    impossible, impossible_status = solve_joint_audit(
        problem,
        1,
        1,
        num_warps_override=4,
        exact_warp_sets=2,
        separate=separate,
    )

    assert status == "sat"
    assert solution is not None
    assert len(set(solution.warp_sets.values())) == 3
    assert impossible_status == "unsat"
    assert impossible is None
    with pytest.raises(ValueError, match="exact warp-set count"):
        solve_joint_audit(problem, 1, 1, exact_warp_sets=100)


def test_solve_joint_signature_is_paper_literal():
    import inspect

    from paper_joint_solver.joint_smt import solve_joint

    assert list(inspect.signature(solve_joint).parameters) == [
        "prob",
        "ii",
        "length",
        "num_warps_override",
        "allow_cross_warp",
    ]


def test_solve_joint_has_no_probe_parameters():
    import inspect

    from paper_joint_solver.joint_smt import solve_joint

    parameters = set(inspect.signature(solve_joint).parameters)
    assert parameters.isdisjoint(
        {"colocate", "separate", "exact_warp_sets", "exact_groups"}
    )


def test_legacy_rrt_has_zero_tail_and_rejects_overoccupancy(
    tmp_path, monkeypatch
):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    valid = _write_ddg(
        tmp_path,
        [{"id": 0, "pipeline": "CUDA", "latency": 4, "occupancy": 2}],
        [],
        name="valid_legacy",
    )
    invalid = _write_ddg(
        tmp_path,
        [{"id": 0, "pipeline": "CUDA", "latency": 2, "occupancy": 3}],
        [],
        name="invalid_legacy",
    )

    problem = load_problem(valid)

    assert problem.lat[0] == 4
    assert problem.rrt[0] == {"CUDA": (1, 1, 0, 0)}
    with pytest.raises(ValueError, match="occupancy 3 exceeds latency 2"):
        load_problem(invalid)


def test_explicit_rrt_span_must_equal_raw_latency(tmp_path):
    invalid = _write_ddg(
        tmp_path,
        [{"id": 0, "latency": 3, "rrt": {"CUDA": [1, 1]}}],
        [],
    )

    with pytest.raises(ValueError, match="span 2, expected latency 3"):
        load_problem(invalid)


def test_sparse_rrt_implicitly_zero_fills_to_raw_latency(
    tmp_path, monkeypatch
):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "latency": 4,
                "rrt": {"CUDA": {"0": 2, "2": 1}},
            }
        ],
        [],
    )

    problem = load_problem(ddg)

    assert problem.lat[0] == 4
    assert problem.rrt[0] == {"CUDA": (2, 0, 1, 0)}


def test_streaming_keeps_rrt_and_zeroes_only_outgoing_delay(
    tmp_path, monkeypatch
):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "op_kind": "tt.descriptor_load",
                "pipeline": "TMA",
                "latency": 4,
                "occupancy": 2,
            },
            {"id": 1},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 4}],
    )

    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    assert problem.streaming == {0}
    assert problem.lat[0] == 4
    assert problem.rrt[0] == {"TMA": (1, 1, 0, 0)}
    assert problem.edge_latency(problem.edges[0]) == 0


@pytest.mark.parametrize("distance", [0, 1])
def test_any_incoming_dependence_disqualifies_streaming(
    tmp_path, monkeypatch, distance
):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "pipeline": "CUDA",
                "latency": 1,
                "occupancy": 1,
            },
            {
                "id": 1,
                "op_kind": "tt.descriptor_load",
                "pipeline": "TMA",
                "latency": 4,
                "occupancy": 2,
            },
            {"id": 2},
        ],
        [
            {"src": 0, "dst": 1, "distance": distance, "latency": 1},
            {"src": 1, "dst": 2, "distance": 0, "latency": 4},
        ],
        name=f"incoming_{distance}",
    )

    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    assert problem.variable_latency == {1}
    assert problem.streaming == set()
    assert problem.edge_latency(problem.edges[1]) == 4


def _load_weighted_rrt_problem(tmp_path):
    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "latency": 1, "rrt": {"CUDA": [2], "SFU": [1]}},
            {"id": 1, "latency": 1, "rrt": {"CUDA": [2]}},
        ],
        [],
    )
    machine = MachineModel(capacities={"CUDA": 2, "SFU": 1})
    return load_problem(ddg, machine=machine, u=2)


def test_arbitrary_rrt_drives_ilp_and_joint_capacity(tmp_path):
    pytest.importorskip("pyscipopt")
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint
    from paper_joint_solver.modulo_ilp import solve_modulo

    problem = _load_weighted_rrt_problem(tmp_path)

    assert problem.rrt[0] == {"CUDA": (2,), "SFU": (1,)}
    assert solve_modulo(problem, 1) is None
    assert solve_modulo(problem, 2) is not None
    assert solve_joint(problem, 1, 1)[1] == "unsat"
    assert solve_joint(problem, 2, 2)[1] == "sat"


def test_solve_modulo_is_cold_and_unbounded(tmp_path, monkeypatch):
    """WI-2: no warm start, no time limit — one cold SCIP solve per I."""
    pytest.importorskip("pyscipopt")
    import pyscipopt

    from paper_joint_solver import modulo_ilp

    params: list[str] = []
    added: list[tuple] = []

    class RecordingModel(pyscipopt.Model):
        def setParam(self, name, value):
            params.append(name)
            return super().setParam(name, value)

        def addSol(self, *args, **kwargs):
            added.append(args)
            return super().addSol(*args, **kwargs)

    monkeypatch.setattr(modulo_ilp, "Model", RecordingModel)
    problem = _load_weighted_rrt_problem(tmp_path)

    assert modulo_ilp.solve_modulo(problem, 2) is not None
    assert "limits/time" not in params
    assert added == []


def test_sparse_rrt_normalizes_maximal_segment_durations(tmp_path, monkeypatch):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "latency": 8,
                "rrt": {"CUDA": [1, 1, 0, 0, 0, 0, 3, 3]},
            }
        ],
        [],
    )

    problem = load_problem(ddg)

    assert problem.rrt[0] == {"CUDA": (1, 1, 0, 0, 0, 0, 3, 3)}
    assert problem.lat[0] == 8
    assert problem.occ[0] == 4


def _load_spill_problem(tmp_path):
    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "spill_cost": 10},
            {"id": 1, "spill_cost": 20},
            {"id": 2, "spill_cost": 0},
        ],
        [
            {"src": 0, "dst": 2, "distance": 0, "latency": 0},
            {"src": 1, "dst": 2, "distance": 0, "latency": 0},
        ],
    )
    return load_problem(ddg, u=3)


def test_spill_cost_is_normalized_per_producer(tmp_path):
    problem = _load_spill_problem(tmp_path)

    assert problem.spill == {0: 1, 1: 2, 2: 0}


def test_joint_uses_producer_specific_spill_cost(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint_audit

    problem = _load_spill_problem(tmp_path)

    low_solution, low_status = solve_joint_audit(
        problem,
        1,
        2,
        num_warps_override=3,
        colocate=[[1, 2]],
        separate=[(0, 2)],
    )
    high_solution, high_status = solve_joint_audit(
        problem,
        1,
        2,
        num_warps_override=3,
        colocate=[[0, 2]],
        separate=[(1, 2)],
    )

    assert low_status == "sat"
    assert low_solution is not None
    assert high_status == "unsat"
    assert high_solution is None


def test_joint_zero_cycle_op_can_coissue_with_blocking_wait(
    tmp_path, monkeypatch
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0},
            {"id": 1, "pipeline": "TC", "spill_cost": 0},
            {"id": 2},
        ],
        [{"src": 1, "dst": 2, "distance": 0, "latency": 0}],
        name="zero_cycle_concurrency",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    # A two-warp budget is the paper's "everything on one warp": W_vl takes
    # warp 0 and every regular op is forced onto warp 1.
    solution, status = solve_joint(problem, 1, 1, num_warps_override=2)

    assert status == "sat"
    assert solution is not None
    assert solution.cycles == {0: 0, 1: 0, 2: 0}
    assert {warps for warps in solution.warp_sets.values()} == {(1,)}


def test_joint_rejects_full_cycle_window_before_blocking_wait(
    tmp_path, monkeypatch
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "op_kind": "ttng.tc_gen5_mma",
                "pipeline": "TC",
                "latency": 2,
                "spill_cost": 0,
            },
            {"id": 1},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 1}],
        name="full_concurrency_window",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    solution, status = solve_joint(problem, 2, 2, num_warps_override=2)

    assert problem.lat[0] == 2
    assert status == "unsat"
    assert solution is None


def test_parallel_edges_keep_distinct_normalized_delays(tmp_path):
    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "latency": 10}, {"id": 1, "latency": 10}],
        [
            {"src": 0, "dst": 1, "distance": 0, "latency": 20},
            {"src": 0, "dst": 1, "distance": 0, "latency": 10},
            {"src": 1, "dst": 0, "distance": 1, "latency": 0},
        ],
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0), u=5)

    assert [problem.edge_latency(edge) for edge in problem.edges] == [2, 1, 0]


@pytest.mark.parametrize("distance", [0, 1])
def test_plain_completion_edge_reduces_to_dependence(tmp_path, distance):
    """A desugared completion edge is an ordinary dependence: with a zero
    spill cost on its producer, CROSS-WARPSPILLS adds nothing over DEPENDENCE,
    so separating the endpoints stays satisfiable either way."""
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint_audit

    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "result_types": ["!ttg.async.token"],
                "spill_cost": 0,
            },
            # A positive cost anchors the ZLP so the zero spill stays zero
            # rather than being forced positive by the sum>=1 bound.
            {"id": 1, "pipeline": "CUDA", "latency": 4, "occupancy": 4},
        ],
        [
            {
                "src": 0,
                "dst": 1,
                "distance": distance,
                "latency": 0,
                "blocking": True,
            }
        ],
        name=f"completion_{distance}",
    )
    problem = load_problem(ddg, machine=MachineModel(smem_bytes=232448))

    solution, status = solve_joint_audit(
        problem,
        5,
        5,
        num_warps_override=3,
        separate=[(0, 1)],
    )
    no_cross_solution, no_cross_status = solve_joint_audit(
        problem,
        5,
        5,
        num_warps_override=3,
        allow_cross_warp=False,
        separate=[(0, 1)],
    )

    # G3: spill is defined for every node, zeros included.
    assert problem.spill == {0: 0, 1: 0}
    assert problem.regs[0] == 0
    # The edge carries no register transport (spill == 0), so disabling
    # cross-warp traffic cannot change the verdict: CROSS-WARPSPILLS degenerates
    # to plain DEPENDENCE.
    assert status == no_cross_status
    assert (solution is None) == (no_cross_solution is None)


@pytest.mark.parametrize(
    "accumulator_cols,expected_status",
    [(511, "sat"), (512, "unsat")],
)
def test_joint_signal_use_extends_tmem_liveness_to_last_consumer(
    tmp_path, monkeypatch, accumulator_cols, expected_status
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "op_kind": "ttng.tc_gen5_mma",
                "pipeline": "TC",
                "result_types": ["i32", "!ttg.async.token"],
            },
            {"id": 2, "op_kind": "ttng.tmem_load", "pipeline": "TMEM"},
            {"id": 3},
            {"id": 4},
        ],
        [
            {"src": 0, "dst": 3, "distance": 0, "latency": 1},
            {
                "src": 0,
                "dst": 2,
                "distance": 0,
                "latency": 2,
                "blocking": True,
            },
            {"src": 3, "dst": 4, "distance": 0, "latency": 1},
            {"src": 4, "dst": 2, "distance": 0, "latency": 0},
        ],
        name=f"tmem_signal_liveness_{accumulator_cols}",
    )
    problem = load_problem(
        ddg, machine=MachineModel(tmem_cols=512, spill_cost=0)
    )
    problem.tmem_footprint.update({0: accumulator_cols, 3: 1})
    problem.regs.update({0: 0, 3: 0})

    solution, status = solve_joint(problem, 3, 3, num_warps_override=2)

    assert problem.edges[1].blocking
    assert status == expected_status
    assert (solution is not None) == (expected_status == "sat")
    if solution is not None:
        assert solution.cycles == {0: 0, 2: 2, 3: 1, 4: 2}
        assert solution.stats["tmem_value_count"] == 2
        assert solution.stats["peak_tmem_cols"] == 512


@pytest.mark.parametrize(
    "accumulator_cols,expected_status",
    [(512, "sat"), (513, "unsat")],
)
def test_joint_carried_signal_initializes_tmem_liveness(
    tmp_path, accumulator_cols, expected_status
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "result_types": ["!ttg.async.token"]},
            {"id": 1},
        ],
        [
            {
                "src": 0,
                "dst": 1,
                "distance": 1,
                "latency": 0,
                # C6: a desugared completion token is a blocking dependence.
                "blocking": True,
            }
        ],
        name=f"tmem_carried_signal_{accumulator_cols}",
    )
    problem = load_problem(ddg, machine=MachineModel(tmem_cols=512))
    problem.tmem_footprint.update({0: accumulator_cols})

    solution, status = solve_joint(problem, 1, 1, num_warps_override=2)

    assert problem.edges[0].blocking
    assert status == expected_status
    assert (solution is not None) == (expected_status == "sat")
    if solution is not None:
        assert solution.stats["tmem_value_count"] == 1
        assert solution.stats["peak_tmem_cols"] == 512


def test_no_cross_warp_reg_data_domain_allows_buffer_edge(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "op_kind": "ttg.local_alloc",
                "result_types": [
                    "!ttg.memdesc<1xi32, #ttg.shared_memory>"
                ],
            },
            {
                "id": 1,
                "op_kind": "tt.descriptor_load",
                "pipeline": "TMA",
                "result_types": ["!ttg.async.token"],
            },
            # An isolated positive spill anchors the ZLP so the zero entries
            # keep a zero image (otherwise sum(C')>=1 forces one positive).
            {"id": 2, "spill_cost": 10},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="buffer_cross_variable_latency",
    )
    problem = load_problem(ddg)

    weak_solution, weak_status = solve_joint(
        problem, 1, 1, num_warps_override=2, allow_cross_warp=False
    )
    # R1 removed the 'all-ws' reading; the no-cross-warp domain is the paper's
    # register-data one, so a zero-register buffer edge stays satisfiable.
    assert problem.regs[0] == 0
    assert weak_status == "sat"
    assert weak_solution is not None


def test_no_cross_warp_reg_data_domain_rejects_register_edge(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0},
            {
                "id": 1,
                "op_kind": "tt.descriptor_load",
                "pipeline": "TMA",
                "result_types": ["!ttg.async.token"],
            },
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="register_cross_variable_latency",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(
        problem, 1, 1, num_warps_override=2, allow_cross_warp=False
    )

    assert problem.regs[0] == 1
    assert status == "unsat"
    assert solution is None


def _identity_normalization(monkeypatch):
    """Keep the fixture's raw costs so capacity boundaries stay readable."""

    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)


def test_paper_opw_cardinality_equals_min_warps(tmp_path):
    """WARPUNIQUENESS, extended: an op occupies exactly min_warps warps."""
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path, [{"id": 0, "min_warps": 4}], [], name="opw_cardinality"
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(problem, 1, 1)

    assert status == "sat"
    assert len(solution.warp_sets[0]) == 4
    assert 0 not in solution.warp_sets[0]
    assert solution.stats["warps_used"] == 4


def test_paper_vl_iff_on_warp_zero(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 1},
        ],
        [],
        name="vl_warp_zero",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(problem, 1, 1)

    assert status == "sat"
    assert problem.variable_latency == {0}
    assert solution.warp_sets[0] == (0,)
    assert 0 not in solution.warp_sets[1]


def test_paper_charges_actual_warps(tmp_path):
    """The paper's budget charges the warps an op actually occupies: a
    three-warp op takes exactly three, with no rounding up to a width."""
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 1, "min_warps": 3},
        ],
        [],
        name="actual_warps",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(problem, 1, 1, num_warps_override=4)

    assert status == "sat"
    assert len(solution.warp_sets[1]) == 3
    assert solution.stats["warps_used"] == 4


def test_paper_allows_overlapping_warp_sets(tmp_path):
    """Two wide ops share warps rather than being clustered or separated.

    Eight and seven warps do not fit disjointly in the twelve available to
    regular ops, so a satisfying assignment must overlap them partially.
    """
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "min_warps": 8}, {"id": 1, "min_warps": 7}],
        [],
        name="overlapping_warp_sets",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(problem, 1, 1, num_warps_override=13)

    assert status == "sat"
    first, second = solution.warp_sets[0], solution.warp_sets[1]
    assert (len(first), len(second)) == (8, 7)
    assert set(first) & set(second)
    assert first != second
    assert set(first) | set(second) <= set(range(1, 13))


def test_paper_registerlimit_divides_by_min_warps(tmp_path, monkeypatch):
    """A multi-warp value is distributed over its member warps, so each one
    holds ceil(regs/min_warps) words -- not ceil(regs/required_width)."""
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    _identity_normalization(monkeypatch)
    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "min_warps": 3, "regs": 300}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 1}],
        name="registerlimit_min_warps",
    )
    problem = load_problem(ddg, machine=MachineModel(regs_per_warp=99))

    tight, tight_status = solve_joint(problem, 2, 2)
    problem.machine.regs_per_warp = 100
    loose, loose_status = solve_joint(problem, 2, 2)

    # 300 words over three warps is 100 per warp: over the 99-word budget,
    # and exactly at the 100-word one.  Dividing by a rounded-up width of
    # four would give 75 and fit both.
    assert tight_status == "unsat"
    assert tight is None
    assert loose_status == "sat"
    assert loose is not None


def test_paper_model_has_no_spill_smem_charge(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint_audit

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "spill_cost": 0},
            {"id": 1},
            # An isolated positive spill anchors the ZLP so the zero entries
            # keep a zero image (otherwise sum(C')>=1 forces one positive).
            {"id": 2, "spill_cost": 10},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="spill_smem_extension",
    )
    problem = load_problem(
        ddg, machine=MachineModel(smem_bytes=0, spill_cost=0)
    )

    solution, status = solve_joint_audit(
        problem, 1, 1, num_warps_override=3, separate=[(0, 1)]
    )

    # A cross-warp spill costs latency, never modelled SMEM or barrier bytes.
    assert problem.regs[0] > 0
    assert status == "sat"
    assert solution is not None


def test_paper_model_allows_cross_warp_carried_register_edge(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint_audit

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "spill_cost": 0},
            {"id": 1},
            {"id": 2, "spill_cost": 10},
        ],
        [{"src": 0, "dst": 1, "distance": 1, "latency": 0}],
        name="carried_register_extension",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    solution, status = solve_joint_audit(
        problem, 1, 2, num_warps_override=3, separate=[(0, 1)]
    )

    # The paper prices a cross-warp carried register edge; it never forbids it.
    assert problem.spill[0] == 0
    assert status == "sat"
    assert solution is not None


def _write_shared_sink_ddg(tmp_path, producers, sink_id, name, **node_fields):
    nodes = [{"id": node_id, **node_fields} for node_id in producers]
    nodes.append({"id": sink_id, "regs": 0})
    edges = [
        {"src": node_id, "dst": sink_id, "distance": 0, "latency": 1}
        for node_id in producers
    ]
    return _write_ddg(tmp_path, nodes, edges, name=name)


def test_paper_model_has_no_sm_level_register_cap(tmp_path, monkeypatch):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    _identity_normalization(monkeypatch)
    ddg = _write_shared_sink_ddg(
        tmp_path, range(9), 9, "sm_register_cap", regs=8160
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(problem, 2, 2)

    # Nine warps at the full per-warp budget is 73440 words, over the SM's
    # 65536 -- a bound the paper never states, so the solve still succeeds.
    assert problem.machine.regs_per_warp * 9 > problem.machine.regs_per_sm
    assert status == "sat"
    assert solution.stats["warps_used"] == 9


def test_paper_model_ignores_fixed_overheads(tmp_path, monkeypatch):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    _identity_normalization(monkeypatch)
    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "smem_footprint": 1000}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 1}],
        name="fixed_overheads",
    )
    machine = MachineModel(smem_bytes=1000)
    machine.smem_fixed_overhead = 1000
    machine.warp_fixed_overhead = 4
    problem = load_problem(ddg, machine=machine)

    solution, status = solve_joint(problem, 2, 2)

    assert status == "sat"
    assert solution is not None
    # Capacities come from D; work absent from G does not shrink them, even
    # when the machine description carries non-zero fixed overheads.
    assert solution.stats["physical_warp_budget"] == machine.num_warps
    assert solution.stats["smem_capacity"] == 1000


@pytest.mark.parametrize("tmem_cols,expected_status", [(200, "sat"), (199, "unsat")])
def test_paper_model_memorycapacity_is_per_value_sum(
    tmp_path, monkeypatch, tmem_cols, expected_status
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    _identity_normalization(monkeypatch)
    ddg = _write_shared_sink_ddg(
        tmp_path,
        (0, 1),
        2,
        f"per_value_sum_{tmem_cols}",
        tmem_footprint=100,
    )
    problem = load_problem(ddg, machine=MachineModel(tmem_cols=tmem_cols))

    solution, status = solve_joint(problem, 2, 2)

    assert status == expected_status
    if solution is not None:
        assert solution.stats["tmem_value_count"] == 2
        assert solution.stats["peak_tmem_cols"] == 200


def test_paper_model_liveness_quantifies_all_curated_edges(tmp_path, monkeypatch):
    """Every curated edge is a use: dropping one shortens its source's
    liveness and frees the capacity the pair needed together."""
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    _identity_normalization(monkeypatch)
    nodes = [
        {"id": 0, "tmem_footprint": 100},
        {"id": 1},
        {"id": 2, "tmem_footprint": 100},
        {"id": 3},
    ]
    edges = [
        {"src": 0, "dst": 1, "distance": 0, "latency": 2},
        {"src": 2, "dst": 3, "distance": 0, "latency": 2},
    ]
    machine = MachineModel(tmem_cols=199)
    both = load_problem(
        _write_ddg(tmp_path, nodes, edges, name="liveness_both"),
        machine=machine,
    )
    dropped = load_problem(
        _write_ddg(tmp_path, nodes, edges[1:], name="liveness_dropped"),
        machine=MachineModel(tmem_cols=199),
    )

    both_solution, both_status = solve_joint(both, 3, 3)
    dropped_solution, dropped_status = solve_joint(dropped, 3, 3)

    assert both_status == "unsat"
    assert both_solution is None
    assert dropped_status == "sat"
    assert dropped_solution is not None


