import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.greedy_ims import greedy_modulo
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import NormalizationResult
from paper_joint_solver.resource_model import StorageObject

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"
FIXTURES = {
    "fwd": (
        EXAMPLES / "case3_FA_fp16" / "ddg.json",
        EXAMPLES / "case3_FA_fp16" / "schedule_graph.json",
    ),
    "subtiled": (
        EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json",
        EXAMPLES / "case3_FA_fp16_subtiled" / "schedule_graph.json",
    ),
    "bwd": (
        EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json",
        EXAMPLES / "case4_FA_bwd" / "schedule_graph_hd128.json",
    ),
}


@pytest.fixture(scope="module", params=list(FIXTURES))
def prob(request):
    ddg, baseline = FIXTURES[request.param]
    return request.param, load_problem(
        ddg,
        baseline_graph=baseline,
    )


@pytest.fixture(scope="module", params=list(FIXTURES))
def low_resolution_prob(request):
    """Small normalization used only to exercise the modulo ILP mechanics."""
    ddg, baseline = FIXTURES[request.param]
    return request.param, load_problem(
        ddg,
        baseline_graph=baseline,
        allow_zero_normalized_costs=True,
    )


def _write_ddg(tmp_path, nodes, edges, name="ddg"):
    prepared = []
    ops = {}
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
            **values,
        }
        prepared.append(node)
        ops[node["op_ref"]] = {"result_types": result_types}
    payload = {
        "schema_version": "ddg-0.1",
        "ops": ops,
        "loops": [
            {
                "trip_count": 4,
                "min_ii": 1,
                "ddg": {"nodes": prepared, "edges": edges},
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


@pytest.mark.parametrize("warp_groups", [{}, {0: 0}])
def test_strict_baseline_requires_complete_ddg_coverage(tmp_path, warp_groups):
    ddg = _write_ddg(tmp_path, [{"id": 0}, {"id": 1}], [])
    baseline = _write_baseline(tmp_path, warp_groups)

    with pytest.raises(ValueError, match="baseline graph is missing DDG nodes"):
        load_problem(ddg, baseline_graph=baseline)


def test_strict_baseline_rejects_mismatched_node_identity(tmp_path):
    ddg = _write_ddg(tmp_path, [{"id": 0}], [])
    baseline = _write_baseline(tmp_path, {0: 0})
    payload = json.loads(baseline.read_text())
    payload["loops"][0]["schedule_loop"]["graph"]["nodes"][0][
        "op_ref"
    ] = "wrong_op"
    baseline.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="op_ref does not match the DDG"):
        load_problem(ddg, baseline_graph=baseline)


def test_strict_baseline_requires_node_identity(tmp_path):
    ddg = _write_ddg(tmp_path, [{"id": 0}], [])
    baseline = _write_baseline(tmp_path, {0: 0})

    with pytest.raises(ValueError, match="baseline graph node 0 is missing op_ref"):
        load_problem(ddg, baseline_graph=baseline)


def test_load_and_derive(prob):
    name, p = prob
    assert p.nodes and p.edges
    mmas = [v for v in p.nodes.values() if "mma" in v.op_kind]
    assert len(mmas) == {"fwd": 2, "subtiled": 4, "bwd": 5}[name]
    # Section 5.3 operates on G, which excludes emitter-infrastructure inputs.
    incoming = {
        edge.dst
        for edge in p.edges
        if not edge.signal_only and p.edge_has_ws_semantics(edge)
    }
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
    assert p.emitter_infra == {
        "fwd": {0, 1},
        "subtiled": {0, 1},
        "bwd": {35},
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
    assert max(p.lat.values()) <= p.normalization_u_effective
    assert p.normalization_zero_collapses == 0


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
            edge.src == producer
            and edge.dst != producer
            and not edge.signal_only
            and p.edge_has_ws_semantics(edge)
            for edge in p.edges
        )
        if raw_spill > 0 and has_cross_edge:
            costs.append(raw_spill)
            scaled.append(spill)

    assert 1 <= sum(scaled) <= p.normalization_u_effective, name
    assert all(value > 0 for value in scaled), name
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            error = abs(costs[i] * scaled[j] - costs[j] * scaled[i])
            assert error <= p.normalization_f, (name, i, j, error)


def test_modulo_ilp_finds_schedule_near_min_ii(low_resolution_prob):
    pytest.importorskip("pyscipopt")
    from paper_joint_solver.modulo_ilp import solve_modulo

    name, p = low_resolution_prob
    lo = p.min_ii()
    # Algorithm-1 style: ascend from MinII (a fully saturated unit at MinII
    # can make the packing genuinely infeasible there).
    sched = None
    for ii in range(lo, lo + 5):
        sched = solve_modulo(p, ii, max_seconds=120)
        if sched is not None:
            break
    assert sched is not None, f"{name}: no schedule in [{lo}, {lo + 4}]"
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
    baseline = _write_baseline(tmp_path, {0: -1})
    problem = load_problem(ddg, baseline_graph=baseline, strict_baseline=False)

    modulo = solve_modulo(problem, 1)
    with pytest.raises(ValueError, match="schedule length"):
        solve_joint(problem, 1, 0)
    solution, status = solve_joint(
        problem, 1, modulo.length, num_warps=1, max_groups=2
    )

    assert problem.regs[0] == 1
    assert modulo.length == 1
    assert status == "sat"
    assert solution.warp.keys() == {0}
    assert 0 <= solution.cycles[0] < solution.horizon


def test_joint_group_symmetry_uses_restricted_growth_labels(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 9}, {"id": 2}, {"id": 5}],
        [],
        name="group_label_symmetry",
    )
    problem = load_problem(ddg)
    separate = [(2, 5), (2, 9), (5, 9)]

    canonical, canonical_status = solve_joint(
        problem,
        1,
        1,
        num_warps=3,
        max_groups=4,
        separate=separate,
    )
    unrestricted, unrestricted_status = solve_joint(
        problem,
        1,
        1,
        num_warps=3,
        max_groups=4,
        symmetry_breaking=False,
        separate=separate,
    )

    assert canonical_status == unrestricted_status == "sat"
    assert canonical is not None
    assert unrestricted is not None
    assert canonical.warp == {2: 1, 5: 2, 9: 3}


def test_joint_group_symmetry_keeps_variable_latency_group_zero(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 7, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 1, "op_kind": "tt.descriptor_load", "pipeline": "TMA"},
            {"id": 9},
            {"id": 3},
        ],
        [],
        name="variable_latency_group_symmetry",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(
        problem,
        1,
        1,
        num_warps=3,
        max_groups=4,
        separate=[(3, 9)],
    )

    assert status == "sat"
    assert solution is not None
    assert solution.warp[1] == solution.warp[7] == 0
    assert solution.warp[3] == 1
    assert solution.warp[9] == 2


def test_joint_lane_symmetry_orders_membership_columns(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "min_warps": 4},
            {"id": 1, "min_warps": 1},
            {"id": 2, "min_warps": 1},
        ],
        [],
        name="lane_label_symmetry",
    )
    problem = load_problem(ddg)

    solution, status = solve_joint(
        problem,
        1,
        1,
        num_warps=4,
        max_groups=2,
        colocate=[[0, 1, 2]],
    )
    unrestricted, unrestricted_status = solve_joint(
        problem,
        1,
        1,
        num_warps=4,
        max_groups=2,
        symmetry_breaking=False,
        colocate=[[0, 1, 2]],
    )

    assert status == unrestricted_status == "sat"
    assert solution is not None
    assert unrestricted is not None
    columns = [
        tuple(lane in solution.lane_masks[node_id] for node_id in (0, 1, 2))
        for lane in range(4)
    ]
    assert columns == sorted(columns, reverse=True)


def test_joint_group_label_bound_follows_width_and_warp_budget():
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import _maximum_useful_group_labels

    required_width = {
        **{node_id: 1 for node_id in range(13)},
        **{node_id: 4 for node_id in range(13, 55)},
    }

    assert _maximum_useful_group_labels(required_width, {0, 1}, 32) == 17


def test_joint_exact_group_partition_is_disjoint(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(tmp_path, [{"id": 0}, {"id": 1}, {"id": 2}], [])
    problem = load_problem(ddg)
    separate = [(0, 1), (0, 2), (1, 2)]

    solution, status = solve_joint(
        problem,
        1,
        1,
        num_warps=3,
        exact_groups=3,
        separate=separate,
    )
    impossible, impossible_status = solve_joint(
        problem,
        1,
        1,
        num_warps=3,
        exact_groups=2,
        separate=separate,
    )

    assert status == "sat"
    assert solution is not None
    assert len(set(solution.warp.values())) == 3
    assert impossible_status == "unsat"
    assert impossible is None
    with pytest.raises(ValueError, match="exact group count"):
        solve_joint(problem, 1, 1, exact_groups=100)


@pytest.mark.parametrize(
    "ii,length,distance,num_warps,separate,expected_status",
    [
        (1, 1, 0, 2, None, "sat"),
        (2, 3, 0, 2, None, "sat"),
        (2, 5, 1, 2, None, "sat"),
        (2, 3, 3, 2, None, "sat"),
        (2, 3, 0, 1, [(0, 1)], "unsat"),
    ],
)
def test_joint_compact_encoding_matches_explicit_grid(
    tmp_path,
    ii,
    length,
    distance,
    num_warps,
    separate,
    expected_status,
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 0}, {"id": 1}],
        [
            {
                "src": 0,
                "dst": 1,
                "distance": distance,
                "latency": 0,
            }
        ],
        name=f"compact_{ii}_{length}_{distance}_{expected_status}",
    )
    problem = load_problem(ddg)
    common = {
        "num_warps": num_warps,
        "max_groups": 3,
        "separate": separate,
    }

    compact, compact_status = solve_joint(
        problem, ii, length, compact_encoding=True, **common
    )
    explicit, explicit_status = solve_joint(
        problem, ii, length, compact_encoding=False, **common
    )

    assert compact_status == explicit_status == expected_status
    assert (compact is None) == (explicit is None)
    if compact is not None:
        assert explicit is not None
        assert compact.stats["num_asserts"] < explicit.stats["num_asserts"]


@pytest.mark.parametrize(
    "prefix_lane_masks,full_group_lane_masks,expected_status",
    [
        (False, False, "sat"),
        (True, False, "sat"),
        (True, True, "unsat"),
    ],
)
def test_joint_compact_lane_factorization_matches_explicit_rows(
    tmp_path,
    prefix_lane_masks,
    full_group_lane_masks,
    expected_status,
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "min_warps": 4}, {"id": 1, "min_warps": 1}],
        [],
        name=f"lane_factor_{prefix_lane_masks}_{full_group_lane_masks}",
    )
    problem = load_problem(ddg)
    common = {
        "num_warps": 4,
        "max_groups": 3,
        "colocate": [[0, 1]],
        "prefix_lane_masks": prefix_lane_masks,
        "full_group_lane_masks": full_group_lane_masks,
    }

    compact, compact_status = solve_joint(
        problem, 1, 1, compact_encoding=True, **common
    )
    explicit, explicit_status = solve_joint(
        problem, 1, 1, compact_encoding=False, **common
    )

    assert compact_status == explicit_status == expected_status
    assert (compact is None) == (explicit is None)
    if compact is not None:
        assert explicit is not None
        assert all(
            len(compact.lane_masks[node_id]) == problem.nodes[node_id].min_warps
            for node_id in problem.nodes
        )


def test_joint_compact_lane_factorization_supports_eight_warps(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(tmp_path, [{"id": 0, "min_warps": 8}], [])
    problem = load_problem(ddg)

    compact, compact_status = solve_joint(
        problem, 1, 1, num_warps=8, compact_encoding=True
    )
    explicit, explicit_status = solve_joint(
        problem, 1, 1, num_warps=8, compact_encoding=False
    )

    assert compact_status == explicit_status == "sat"
    assert compact is not None
    assert explicit is not None
    assert compact.group_widths[compact.warp[0]] == 8
    assert compact.lane_masks[0] == tuple(range(8))


def test_joint_wide_group_can_combine_narrow_lane_masks(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "min_warps": 4},
            {"id": 1, "min_warps": 4},
            {"id": 2},
            {"id": 3},
        ],
        [
            {"src": 0, "dst": 2, "distance": 0, "latency": 1},
            {"src": 1, "dst": 3, "distance": 0, "latency": 1},
        ],
        name="cooperative_wide_group",
    )
    problem = load_problem(ddg)
    problem.machine.regs_per_warp = 1
    problem.regs.update({0: 4, 1: 4, 2: 0, 3: 0})
    problem.spill.update({0: 0, 1: 0})
    common = {
        "num_warps": 8,
        "max_groups": 2,
        "colocate": [[0, 1, 2, 3]],
    }

    compact, compact_status = solve_joint(
        problem, 2, 2, compact_encoding=True, **common
    )
    explicit, explicit_status = solve_joint(
        problem, 2, 2, compact_encoding=False, **common
    )

    assert compact_status == explicit_status == "sat"
    assert compact is not None
    assert explicit is not None
    assert compact.group_widths[compact.warp[0]] == 8
    assert set(compact.lane_masks[0]).isdisjoint(compact.lane_masks[1])


@pytest.mark.parametrize("compact_encoding", [True, False])
def test_joint_lane_symmetry_preserves_partially_used_width(
    tmp_path, compact_encoding
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0},
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
            {"id": 6, "min_warps": 4},
        ],
        [
            {"src": 0, "dst": 3, "distance": 0, "latency": 1},
            {"src": 1, "dst": 4, "distance": 0, "latency": 1},
            {"src": 2, "dst": 5, "distance": 0, "latency": 1},
        ],
        name=f"partially_used_width_{compact_encoding}",
    )
    problem = load_problem(ddg)
    problem.machine.regs_per_warp = 1
    problem.regs.update({0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0})
    problem.spill.update({0: 0, 1: 0, 2: 0})
    common = {
        "num_warps": 8,
        "max_groups": 3,
        "colocate": [[0, 1, 2, 3, 4, 5]],
        "separate": [(0, 6)],
        "compact_encoding": compact_encoding,
    }

    symmetric, symmetric_status = solve_joint(
        problem, 2, 2, symmetry_breaking=True, **common
    )
    unrestricted, unrestricted_status = solve_joint(
        problem, 2, 2, symmetry_breaking=False, **common
    )

    assert symmetric_status == unrestricted_status == "sat"
    assert symmetric is not None
    assert unrestricted is not None
    source_masks = [set(symmetric.lane_masks[node_id]) for node_id in range(3)]
    assert all(
        source_masks[left].isdisjoint(source_masks[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    source_group = symmetric.warp[0]
    assert symmetric.group_widths[source_group] == 4
    assert len(set().union(*source_masks)) == 3


@pytest.mark.parametrize(
    "group_widths,match",
    [
        ((), "unique positive"),
        ((1, 1, 4, 8), "unique positive"),
        ((1, 2, 8), "do not cover"),
    ],
)
def test_joint_rejects_invalid_machine_group_widths(
    tmp_path, group_widths, match
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(tmp_path, [{"id": 0, "min_warps": 4}], [])
    problem = load_problem(ddg)
    problem.machine.group_widths = group_widths

    with pytest.raises(ValueError, match=match):
        solve_joint(problem, 1, 1)


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


def test_infra_input_retains_dependence_but_not_ws_semantics(
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
                "pipeline": "TC",
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
            {"src": 0, "dst": 1, "distance": 0, "latency": 1},
            {"src": 1, "dst": 2, "distance": 0, "latency": 4},
        ],
        name="infra_input",
    )
    baseline = _write_baseline(tmp_path, {0: -1}, name="infra_baseline")

    problem = load_problem(
        ddg,
        baseline_graph=baseline,
        strict_baseline=False,
        machine=MachineModel(spill_cost=7),
    )

    assert problem.emitter_infra == {0}
    assert problem.streaming == {1}
    assert problem.edge_latency(problem.edges[0]) == 1
    assert (0, 1) not in problem.blocking
    assert 0 not in problem.spill


def test_signal_only_input_does_not_disqualify_streaming(
    tmp_path, monkeypatch
):
    def identity(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", identity)
    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "result_types": ["!ttg.async.token"]},
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
            {
                "src": 0,
                "dst": 1,
                "distance": 0,
                "latency": 0,
                "src_result_idx": 0,
            },
            {"src": 1, "dst": 2, "distance": 0, "latency": 4},
        ],
        name="signal_input",
    )

    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    assert problem.edges[0].signal_only
    assert problem.streaming == {1}
    assert problem.edge_latency(problem.edges[1]) == 0


def test_infra_data_edge_rejects_non_scalar_result(tmp_path):
    ddg = _write_ddg(
        tmp_path,
        [
            {"id": 0, "result_types": ["tensor<4xi32>"]},
            {"id": 1},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="infra_tensor",
    )
    baseline = _write_baseline(tmp_path, {0: -1}, name="infra_tensor_baseline")

    with pytest.raises(
        ValueError,
        match=r"infra data edge 0->1 result 0 must be scalar",
    ):
        load_problem(ddg, baseline_graph=baseline, strict_baseline=False)


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


def test_arbitrary_rrt_drives_min_ii_and_greedy(tmp_path):
    problem = _load_weighted_rrt_problem(tmp_path)

    assert problem.rrt[0] == {"CUDA": (2,), "SFU": (1,)}
    assert problem.min_ii() == 2
    assert greedy_modulo(problem, 2) is not None


def test_arbitrary_rrt_drives_ilp_and_joint_capacity(tmp_path):
    pytest.importorskip("pyscipopt")
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint
    from paper_joint_solver.modulo_ilp import solve_modulo

    problem = _load_weighted_rrt_problem(tmp_path)

    assert solve_modulo(problem, 1) is None
    assert solve_modulo(problem, 2) is not None
    assert solve_joint(problem, 1, 1, max_groups=2)[1] == "unsat"
    assert solve_joint(problem, 2, 2, max_groups=2)[1] == "sat"


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
        [{"id": 0, "spillcost": 10}, {"id": 1}, {"id": 2}],
        [
            {"src": 0, "dst": 2, "distance": 0, "latency": 0},
            {"src": 1, "dst": 2, "distance": 0, "latency": 0},
        ],
    )
    return load_problem(ddg, machine=MachineModel(spill_cost=20), u=3)


def test_spill_cost_is_normalized_per_producer(tmp_path):
    problem = _load_spill_problem(tmp_path)

    assert problem.spill == {0: 1, 1: 2}


def test_joint_uses_producer_specific_spill_cost(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    problem = _load_spill_problem(tmp_path)

    low_solution, low_status = solve_joint(
        problem,
        1,
        2,
        num_warps=2,
        max_groups=3,
        prefix_lane_masks=True,
        colocate=[[1, 2]],
        separate=[(0, 2)],
    )
    high_solution, high_status = solve_joint(
        problem,
        1,
        2,
        num_warps=2,
        max_groups=3,
        prefix_lane_masks=True,
        colocate=[[0, 2]],
        separate=[(1, 2)],
    )

    assert low_status == "sat"
    assert low_solution is not None
    assert high_status == "unsat"
    assert high_solution is None


@pytest.mark.parametrize("compact_encoding", [True, False])
def test_joint_zero_cycle_op_can_coissue_with_blocking_wait(
    tmp_path, monkeypatch, compact_encoding
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
            {"id": 1, "pipeline": "TC", "spillcost": 0},
            {"id": 2},
        ],
        [{"src": 1, "dst": 2, "distance": 0, "latency": 0}],
        name="zero_cycle_concurrency",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    solution, status = solve_joint(
        problem,
        1,
        1,
        num_warps=1,
        max_groups=2,
        prefix_lane_masks=True,
        compact_encoding=compact_encoding,
    )

    assert status == "sat"
    assert solution is not None
    assert solution.cycles == {0: 0, 1: 0, 2: 0}


@pytest.mark.parametrize("compact_encoding", [True, False])
def test_joint_rejects_full_cycle_window_before_blocking_wait(
    tmp_path, monkeypatch, compact_encoding
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
                "spillcost": 0,
            },
            {"id": 1},
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 1}],
        name="full_concurrency_window",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))

    solution, status = solve_joint(
        problem,
        2,
        2,
        num_warps=1,
        max_groups=2,
        prefix_lane_masks=True,
        compact_encoding=compact_encoding,
    )

    assert problem.lat[0] == 2
    assert status == "unsat"
    assert solution is None


def test_joint_infra_edge_ignores_spill_buffer_and_no_cross_gate(
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
        [{"id": 0}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="infra_cross_edge",
    )
    baseline = _write_baseline(tmp_path, {0: -1}, name="infra_cross_baseline")
    problem = load_problem(
        ddg,
        baseline_graph=baseline,
        strict_baseline=False,
        machine=MachineModel(spill_cost=1, smem_bytes=0),
    )
    common = {
        "num_warps": 2,
        "max_groups": 3,
        "prefix_lane_masks": True,
        "separate": [(0, 1)],
    }

    solution, status = solve_joint(problem, 1, 1, **common)
    no_cross_solution, no_cross_status = solve_joint(
        problem,
        1,
        1,
        allow_cross_warp=False,
        **common,
    )

    assert problem.spill == {}
    assert status == no_cross_status == "sat"
    assert solution is not None
    assert no_cross_solution is not None


def test_joint_infra_carried_edge_ignores_lane_realizability(
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
        [{"id": 0}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 1, "latency": 0}],
        name="infra_carried_edge",
    )
    baseline = _write_baseline(tmp_path, {0: -1}, name="infra_carried_baseline")
    problem = load_problem(
        ddg,
        baseline_graph=baseline,
        strict_baseline=False,
        machine=MachineModel(spill_cost=0),
    )

    solution, status = solve_joint(
        problem,
        1,
        2,
        num_warps=2,
        max_groups=3,
        prefix_lane_masks=True,
        separate=[(0, 1)],
    )

    assert status == "sat"
    assert solution is not None


def test_joint_infra_predecessor_does_not_gate_concurrency(
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
            {"id": 0, "spillcost": 0},
            {
                "id": 1,
                "op_kind": "ttng.tc_gen5_mma",
                "pipeline": "TC",
                "latency": 2,
                "spillcost": 0,
            },
            {"id": 2},
        ],
        [{"src": 0, "dst": 2, "distance": 0, "latency": 1}],
        name="infra_concurrency_predecessor",
    )
    baseline = _write_baseline(
        tmp_path,
        {0: -1},
        name="infra_concurrency_baseline",
    )
    problem = load_problem(
        ddg,
        baseline_graph=baseline,
        strict_baseline=False,
        machine=MachineModel(spill_cost=0),
    )

    solution, status = solve_joint(
        problem,
        2,
        2,
        num_warps=2,
        max_groups=3,
        prefix_lane_masks=True,
        colocate=[[1, 2]],
        separate=[(0, 2)],
    )

    assert status == "sat"
    assert solution is not None
    assert solution.cycles == {0: 0, 1: 0, 2: 1}


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
    assert problem.rec_mii() == 2


def test_parallel_async_token_edge_uses_source_result_index(tmp_path):
    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "pipeline": "TMEM",
                "result_types": ["i32", "!ttg.async.token"],
            },
            {"id": 1},
            {
                "id": 2,
                "result_types": ["i32", "!ttg.async.token"],
            },
            {"id": 3},
        ],
        [
            {
                "src": 0,
                "dst": 1,
                "distance": 0,
                "latency": 0,
                "src_result_idx": 1,
            },
            {"src": 0, "dst": 1, "distance": 0, "latency": 0},
            {
                "src": 2,
                "dst": 3,
                "distance": 0,
                "latency": 0,
                "src_result_idx": 1,
            },
        ],
    )

    problem = load_problem(ddg)
    signal, data, pure_signal = problem.edges

    assert signal.src_result_idx == 1
    assert signal.signal_only
    assert data.src_result_idx == 0
    assert not data.signal_only
    assert pure_signal.src_result_idx == 1
    assert pure_signal.signal_only
    assert (0, 1) in problem.blocking
    assert (2, 3) in problem.blocking
    assert 0 in problem.spill
    assert 2 not in problem.spill


@pytest.mark.parametrize(
    "distance, spill_cost, smem_bytes",
    [
        (0, 1, 232448),
        (0, 0, 0),
        (1, 0, 232448),
    ],
)
def test_joint_signal_edge_has_no_register_transport_or_no_cross_gate(
    tmp_path, distance, spill_cost, smem_bytes
):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [
            {
                "id": 0,
                "result_types": ["tensor<70000xi32>", "!ttg.async.token"],
            },
            {"id": 1},
        ],
        [
            {
                "src": 0,
                "dst": 1,
                "distance": distance,
                "latency": 0,
                "src_result_idx": 1,
            }
        ],
        name=f"signal_{distance}_{spill_cost}_{smem_bytes}",
    )
    machine = MachineModel(spill_cost=spill_cost, smem_bytes=smem_bytes)
    problem = load_problem(ddg, machine=machine)

    solution, status = solve_joint(
        problem,
        1,
        1,
        num_warps=2,
        max_groups=3,
        prefix_lane_masks=True,
        separate=[(0, 1)],
    )
    no_cross_solution, no_cross_status = solve_joint(
        problem,
        1,
        1,
        num_warps=2,
        max_groups=3,
        allow_cross_warp=False,
        prefix_lane_masks=True,
        separate=[(0, 1)],
    )

    assert problem.spill == {}
    assert status == no_cross_status == "sat"
    assert solution is not None
    assert no_cross_solution is not None


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
                "src_result_idx": 1,
            },
            {"src": 3, "dst": 4, "distance": 0, "latency": 1},
            {"src": 4, "dst": 2, "distance": 0, "latency": 0},
        ],
        name=f"tmem_signal_liveness_{accumulator_cols}",
    )
    problem = load_problem(
        ddg, machine=MachineModel(tmem_cols=512, spill_cost=0)
    )
    problem.memory_objects = {
        "tmem:acc": StorageObject(
            "tmem:acc", "tmem", accumulator_cols, frozenset({0})
        ),
        "tmem:scratch": StorageObject(
            "tmem:scratch", "tmem", 1, frozenset({3})
        ),
    }
    problem.regs.update({0: 0, 3: 0})

    solution, status = solve_joint(
        problem, 3, 3, num_warps=1, max_groups=2
    )

    assert problem.edges[1].signal_only
    assert status == expected_status
    assert (solution is not None) == (expected_status == "sat")
    if solution is not None:
        assert solution.cycles == {0: 0, 2: 2, 3: 1, 4: 2}
        assert solution.stats["tmem_object_count"] == 2
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
                "src_result_idx": 0,
            }
        ],
        name=f"tmem_carried_signal_{accumulator_cols}",
    )
    problem = load_problem(ddg, machine=MachineModel(tmem_cols=512))
    problem.memory_objects = {
        "tmem:acc": StorageObject(
            "tmem:acc", "tmem", accumulator_cols, frozenset({0})
        )
    }

    solution, status = solve_joint(
        problem, 1, 1, num_warps=1, max_groups=2
    )

    assert problem.edges[0].signal_only
    assert status == expected_status
    assert (solution is not None) == (expected_status == "sat")
    if solution is not None:
        assert solution.stats["tmem_object_count"] == 1
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
        ],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="buffer_cross_variable_latency",
    )
    problem = load_problem(ddg)

    weak_solution, weak_status = solve_joint(
        problem,
        1,
        1,
        num_warps=2,
        max_groups=2,
        allow_cross_warp=False,
    )
    strong_solution, strong_status = solve_joint(
        problem,
        1,
        1,
        num_warps=2,
        max_groups=2,
        allow_cross_warp=False,
        no_cross_warp_domain="all-ws",
    )

    assert problem.regs[0] == 0
    assert weak_status == "sat"
    assert weak_solution is not None
    assert strong_status == "unsat"
    assert strong_solution is None


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
        problem,
        1,
        1,
        num_warps=2,
        max_groups=2,
        allow_cross_warp=False,
    )

    assert problem.regs[0] == 1
    assert status == "unsat"
    assert solution is None


def test_spill_smem_footprint_can_be_disabled_for_paper_pure(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "spillcost": 0}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 0, "latency": 0}],
        name="spill_smem_extension",
    )
    problem = load_problem(
        ddg, machine=MachineModel(smem_bytes=0, spill_cost=0)
    )
    common = {
        "num_warps": 2,
        "max_groups": 3,
        "separate": [(0, 1)],
    }

    extended_solution, extended_status = solve_joint(
        problem, 1, 1, **common
    )
    pure_solution, pure_status = solve_joint(
        problem, 1, 1, spill_smem_footprint=False, **common
    )

    assert extended_status == "unsat"
    assert extended_solution is None
    assert pure_status == "sat"
    assert pure_solution is not None


def test_carried_register_same_lane_can_be_disabled_for_paper_pure(tmp_path):
    pytest.importorskip("yices")
    from paper_joint_solver.joint_smt import solve_joint

    ddg = _write_ddg(
        tmp_path,
        [{"id": 0, "spillcost": 0}, {"id": 1}],
        [{"src": 0, "dst": 1, "distance": 1, "latency": 0}],
        name="carried_register_extension",
    )
    problem = load_problem(ddg, machine=MachineModel(spill_cost=0))
    common = {
        "num_warps": 2,
        "max_groups": 3,
        "separate": [(0, 1)],
    }

    extended_solution, extended_status = solve_joint(
        problem, 1, 2, **common
    )
    pure_solution, pure_status = solve_joint(
        problem, 1, 2, reg_carried_same_lane=False, **common
    )

    assert extended_status == "unsat"
    assert extended_solution is None
    assert pure_status == "sat"
    assert pure_solution is not None
