import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    assert sum(res.scaled) <= 300
    assert all(s >= 1 for s in res.scaled)
    assert res.solve_time_s < 0.5, f"paper reports <500 ms, got {res.solve_time_s}"
    assert max_relative_ratio_drift(costs, res.scaled) < 0.1
    order = sorted(range(len(costs)), key=lambda i: costs[i])
    assert all(res.scaled[a] <= res.scaled[b] for a, b in zip(order, order[1:]))
    # More resolution (larger U) must not worsen the optimum.
    assert normalize_costs(costs, u=1000).objective <= res.objective


def test_equal_costs_stay_equal():
    res = normalize_costs([1000, 1000, 1000])
    assert res.optimal and len(set(res.scaled)) == 1 and res.objective == 0


def test_scaling_is_isomorphic():
    # Multiplying all costs by k must not change C', and scales F by k.
    a = normalize_costs([100, 233, 401])
    b = normalize_costs([300, 699, 1203])
    assert a.scaled == b.scaled
    assert b.objective == 3 * a.objective

def test_exact_small_ratios():
    res = normalize_costs([100, 200, 400])
    assert res.objective == 0
    s = res.scaled
    assert s[1] == 2 * s[0] and s[2] == 4 * s[0] and s[0] >= 1
