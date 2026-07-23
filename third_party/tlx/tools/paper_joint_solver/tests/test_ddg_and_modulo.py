import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.modulo_ilp import solve_modulo

EXAMPLES = Path(__file__).resolve().parents[2] / "sched2tlx" / "examples"
FIXTURES = {
    "fwd": EXAMPLES / "case3_FA_fp16" / "ddg.json",
    "subtiled": EXAMPLES / "case3_FA_fp16_subtiled" / "ddg.json",
    "bwd": EXAMPLES / "case4_FA_bwd" / "ddg_hd128.json",
}


@pytest.fixture(scope="module", params=list(FIXTURES))
def prob(request):
    return request.param, load_problem(FIXTURES[request.param])


def test_load_and_derive(prob):
    name, p = prob
    assert p.nodes and p.edges
    mmas = [v for v in p.nodes.values() if "mma" in v.op_kind]
    assert len(mmas) == {"fwd": 2, "subtiled": 4, "bwd": 5}[name]
    # K/V loads stream (no in-loop producer, TMA pipeline).
    assert len(p.streaming) == 2
    assert p.streaming <= p.variable_latency
    # Streaming outgoing latency is zeroed (sec 5.3).
    for e in p.edges:
        if e.src in p.streaming:
            assert p.edge_lat[(e.src, e.dst, e.distance)] == 0
    # Blocking edges exist (TC/TMA producers).
    assert p.blocking
    # Normalized costs are small.
    assert max(p.lat.values()) <= 300


def test_normalization_keeps_structure(prob):
    # The failure mode that matters: the normalization ILP collapsing the
    # pool (every big cost -> 0/1), which erases all scheduling structure.
    name, p = prob
    mma_lat = [p.lat[v.id] for v in p.nodes.values() if "mma" in v.op_kind]
    assert min(mma_lat) >= 10, (name, mma_lat)
    assert p.res_mii() >= 20 and p.rec_mii() >= 20, name
    # TC must carry real occupancy (the MMAs), TMEM its port traffic.
    for pipe in ("TC", "TMEM"):
        assert sum(p.occ[v.id] for v in p.nodes.values()
                   if v.pipeline == pipe) > 0, (name, pipe)


def test_modulo_ilp_finds_schedule_near_min_ii(prob):
    name, p = prob
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
        d = p.edge_lat[(e.src, e.dst, e.distance)]
        assert (sched.cycles[e.dst] >= sched.cycles[e.src] + d
                - e.distance * ii), (name, e)
    for pipe, cap in p.machine.capacities.items():
        for r in range(ii):
            use = sum(1 for v in p.nodes.values() if v.pipeline == pipe
                      for c in range(p.occ[v.id])
                      if (sched.cycles[v.id] + c) % ii == r)
            assert use <= cap, (name, pipe, r, use)
