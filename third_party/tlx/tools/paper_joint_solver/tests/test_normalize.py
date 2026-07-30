import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.machine import MachineModel
from paper_joint_solver.normalize import NormalizationResult
from paper_joint_solver.normalize import normalize_costs


def max_relative_ratio_drift(costs, scaled):
    worst = 0.0
    for i in range(len(costs)):
        for j in range(len(costs)):
            if i == j or scaled[j] == 0 or costs[j] == 0:
                continue
            want = costs[i] / costs[j]
            got = scaled[i] / scaled[j]
            worst = max(worst, abs(want - got) / want)
    return worst


def test_b200_latency_mix():
    # Representative B200 op latencies: tcgen05 MMA, SFU exp2, TMEM load,
    # TMEM store, TMA-load occupancy.
    costs = [900, 570, 179, 64, 96]
    res = normalize_costs(costs)
    assert res.optimal
    assert 1 <= sum(res.scaled) <= 300
    assert all(s >= 0 for s in res.scaled)
    assert max_relative_ratio_drift(costs, res.scaled) < 0.1
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
    res = normalize_costs([100, 200, 400])
    assert res.objective == 0
    s = res.scaled
    assert s[1] == 2 * s[0] and s[2] == 4 * s[0] and s[0] >= 1


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


def test_ddg_passes_the_complete_unmodified_cost_multiset(
    tmp_path, monkeypatch
):
    captured = []

    def capture(costs, u=300, time_limit_s=None):
        del u, time_limit_s
        captured.extend(costs)
        return NormalizationResult(costs.copy(), 0, 0.0, True)

    monkeypatch.setattr("paper_joint_solver.ddg.normalize_costs", capture)
    costs = (1, 100, 105, 7, 1)
    data = {
        "ops": {
            f"op{i}": {
                "kind": (
                    "arith.addi"
                    if i == 3
                    else "arith.addf" if i == 4 else "math.exp2"
                ),
                "operands": [],
                "result_types": [
                    "!ttg.async.token"
                    if i == 3
                    else "tensor<256xf32>" if i == 4 else "f32"
                ],
            }
            for i in range(len(costs))
        },
        "loops": [
            {
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
                            "min_warps": 1,
                        }
                        for i, cost in enumerate(costs)
                    ],
                    "edges": [
                        {"src": 0, "dst": 1, "distance": 0, "latency": 17}
                    ],
                }
            }
        ],
    }
    path = tmp_path / "ddg.json"
    path.write_text(json.dumps(data))

    problem = load_problem(path, machine=MachineModel(spill_cost=0))

    assert captured == [*costs, 17]
    assert problem.lat == dict(enumerate(costs))
    assert problem.occ == dict(enumerate(costs))
    assert problem.edge_latency(problem.edges[0]) == 17
    assert problem.spill == {0: 0}
