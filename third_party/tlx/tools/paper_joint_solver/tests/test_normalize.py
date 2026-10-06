import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import NormalizationResult
from paper_joint_solver.normalize import normalize_costs

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"


def assert_zlp_invariants(costs, res, u):
    """The paper only claims F is optimal, so any point on the F-optimal face
    is a legal answer. Assert the model invariants, never the chosen point."""
    assert all(s >= 0 for s in res.scaled)
    assert 1 <= sum(res.scaled) <= u
    for i in range(len(costs)):
        for j in range(len(costs)):
            cross = costs[i] * res.scaled[j] - costs[j] * res.scaled[i]
            assert abs(cross) <= res.objective


def test_b200_latency_mix():
    # Representative B200 op latencies: tcgen05 MMA, SFU exp2, TMEM load,
    # TMEM store, TMA-load occupancy.
    costs = [900, 570, 179, 64, 96]
    res = normalize_costs(costs)
    assert res.optimal
    assert res.objective == 150
    assert_zlp_invariants(costs, res, 300)
    # More resolution (larger U) must not worsen the optimum.
    assert normalize_costs(costs, u=1000).objective <= res.objective


def test_equal_costs_stay_equal():
    res = normalize_costs([1000, 1000, 1000])
    assert res.optimal and len(set(res.scaled)) == 1 and res.objective == 0


def test_scaling_is_isomorphic():
    # Multiplying all costs by k preserves the feasible C' set and scales F.
    a = normalize_costs([100, 233, 401])
    b = normalize_costs([300, 699, 1203])
    assert b.objective == 3 * a.objective


def test_exact_small_ratios():
    costs = [100, 200, 400]
    res = normalize_costs(costs)
    assert res.objective == 0
    assert_zlp_invariants(costs, res, 300)
    # F == 0 pins the ratios exactly; that is a consequence of the pairwise
    # constraints, not a point assertion.
    assert res.scaled[1] == 2 * res.scaled[0]
    assert res.scaled[2] == 4 * res.scaled[0]


def test_duplicate_costs_retain_their_u_budget_weight():
    unique = normalize_costs([1, 2], u=5)
    repeated = normalize_costs([1, 1, 2, 2], u=5)

    assert unique.objective == 0
    assert repeated.objective > 0
    assert 1 <= sum(repeated.scaled) <= 5


def test_equal_cost_entries_remain_independent_variables():
    result = normalize_costs([1, 1], u=1)

    assert result.objective == 1
    assert sorted(result.scaled) == [0, 1]


def test_degenerate_f_tie_returns_solver_pick():
    costs = [1, 2]
    result = normalize_costs(costs, u=2)

    assert result.objective == 1
    assert_zlp_invariants(costs, result, 2)


def test_normalize_costs_solves_exactly_once(monkeypatch):
    calls = []
    real_model = normalize_costs.__globals__["Model"]

    class CountingModel(real_model):
        def optimize(self, *args, **kwargs):
            calls.append(1)
            return super().optimize(*args, **kwargs)

    monkeypatch.setattr("paper_joint_solver.normalize.Model", CountingModel)
    normalize_costs([900, 570, 179, 64, 96])

    assert len(calls) == 1


def test_zero_entries_participate_in_c():
    costs = [0, 300]
    res = normalize_costs(costs)

    assert len(res.scaled) == 2
    assert res.objective == 0
    # 300*C'[0] <= F == 0 forces the zero entry to zero; C'[1] is free on the
    # F-optimal face, so it is deliberately not asserted.
    assert res.scaled[0] == 0
    assert sum(res.scaled) >= 1


def test_all_zero_costs_force_a_positive_entry():
    res = normalize_costs([0, 0])

    assert res.objective == 0
    assert 1 <= sum(res.scaled) <= 300


def test_empty_costs_infeasible():
    with pytest.raises(RuntimeError, match="infeasible"):
        normalize_costs([])


def test_scip_suite_version_pin():
    from pyscipopt import Model

    assert Model().version() == 10.0


def test_ddg_passes_the_complete_unmodified_cost_multiset(
    tmp_path, monkeypatch
):
    captured = []

    def capture(costs, u=300):
        del u
        captured.extend(costs)
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", capture)
    costs = (1, 100, 105, 7, 1)
    data = {
        "schema_version": "curated-ddg-0.2",
        "curation_source": {
            "ddg_sha256": "0" * 64,
            "baseline_graph_sha256": "0" * 64,
            "curator_sources_sha256": "0" * 64,
        },
        "loops": [
            {
                "loop_id": 0,
                "trip_count": None,
                "ddg": {
                    "nodes": [
                        {
                            "id": i,
                            "op_ref": f"op{i}",
                            "op_kind": (
                                "arith.addi"
                                if i == 3
                                else "arith.addf" if i == 4 else "math.exp2"
                            ),
                            "pipeline": "CUDA" if i >= 3 else "SFU",
                            "latency": cost,
                            "occupancy": cost,
                            "rrt": {
                                ("CUDA" if i >= 3 else "SFU"): [1] * cost
                            },
                            "min_warps": 1,
                            "regs": 0,
                            "spill_cost": 0,
                            "smem_footprint": 0,
                            "tmem_footprint": 0,
                            "variable_latency": False,
                            "streaming": False,
                        }
                        for i, cost in enumerate(costs)
                    ],
                    "edges": [
                        {
                            "src": 0,
                            "dst": 1,
                            "distance": 0,
                            "latency": 17,
                            "blocking": False,
                        }
                    ],
                },
            }
        ],
    }
    path = tmp_path / "curated_ddg.json"
    path.write_text(json.dumps(data))

    problem = load_problem(path, machine=MachineModel(spill_cost=0))

    # 5 segment durations + 1 edge latency + one spill entry per node, zeros
    # included (R7: every enumerated entry joins C).
    assert captured == [*costs, 17, 0, 0, 0, 0, 0]
    assert problem.lat == dict(enumerate(costs))
    assert problem.occ == dict(enumerate(costs))
    assert problem.edge_latency(problem.edges[0]) == 17
    assert problem.spill == {i: 0 for i in range(5)}


GOLDEN = Path(__file__).resolve().parent / "golden_curated"


def _curated_case(case, tmp_path_factory):
    """The three canonical cases load their committed golden curated file;
    bwd_subtiled is a measurement fixture only, so it is curated on the fly."""
    if case != "bwd_subtiled":
        return GOLDEN / case / f"{case}.curated_ddg.json"
    from paper_joint_solver.curate_ddg import curate, dumps

    out = tmp_path_factory.mktemp("curated") / "bwd_subtiled.curated_ddg.json"
    curated, _ = curate(
        EXAMPLES / "case4_FA_bwd_subtiled" / "ddg.json",
        EXAMPLES / "case4_FA_bwd_subtiled" / "schedule_graph.json",
    )
    out.write_text(dumps(curated))
    return out


@pytest.mark.parametrize(
    ("case", "expected_f"),
    [
        ("fwd", 411),
        ("fwd_subtiled", 394),
        ("bwd", 388),
        ("bwd_subtiled", 388),
    ],
)
def test_main_case_single_shot_at_u300(case, expected_f, tmp_path_factory):
    problem = load_problem(_curated_case(case, tmp_path_factory))
    pairs = problem.normalization_cost_pairs

    assert problem.normalization_u == 300
    assert problem.normalization_f == expected_f
    assert 1 <= sum(scaled for _, scaled in pairs) <= 300
    for raw_i, scaled_i in pairs:
        for raw_j, scaled_j in pairs:
            cross = raw_i * scaled_j - raw_j * scaled_i
            assert abs(cross) <= problem.normalization_f
    # The literal single shot collapses positive costs; that is the accepted
    # baseline, not a defect (R9/D1).
    assert any(raw > 0 and scaled == 0 for raw, scaled in pairs)
