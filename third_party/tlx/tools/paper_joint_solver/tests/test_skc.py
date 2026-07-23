"""SKC classifier/binder tests against the real FA4-exact solution (CPU-only)."""

import json
from pathlib import Path

import pytest

from paper_joint_solver.ddg import load_problem
from skc.binder import bind, bind_bwd
from skc.roles import RoleClassificationError, classify, classify_bwd

PKG = Path(__file__).resolve().parent.parent
DDG = PKG.parent / "sched2tlx" / "examples" / "case3_FA_fp16_subtiled" / "ddg.json"
SOL = PKG / "subtiled_fa4exact_solution.json"
DDG_BWD = PKG.parent / "sched2tlx" / "examples" / "case4_FA_bwd" / "ddg_hd128.json"
SOL_BWD_FREE = PKG / "bwd_joint_solution_v6.json"
SOL_BWD_SKC = PKG / "bwd_skc_solution.json"


@pytest.fixture(scope="module")
def prob():
    return load_problem(DDG)


@pytest.fixture(scope="module")
def sol():
    return json.loads(SOL.read_text())


def test_classify_fa4_exact(prob, sol):
    roles = classify(prob, sol)
    assert roles.load == 0
    assert roles.mma == 3
    assert roles.softmax == [1, 2]
    assert roles.correction == 4
    # solver parked qk tmem loads / p stores on the mma warp — the skeleton
    # protocol overrides those placements and the audit must say so
    assert roles.protocol_overrides
    assert all(o["solver_warp"] == 3 for o in roles.protocol_overrides)


def test_classify_rejects_headless_partition(prob, sol):
    mutated = json.loads(json.dumps(sol))
    # move every tc_gen5 node off warp 3 onto warp 1 -> two mma-ish warps? no:
    # warp 3 loses TC, warp 1 gains it while keeping softmax -> both classify
    # softmax/mma inconsistently and validation must fail
    for idx, n in prob.nodes.items():
        if n.op_kind == "ttng.tc_gen5_mma":
            mutated["warp"][str(idx)] = 1
    with pytest.raises(RoleClassificationError):
        classify(prob, mutated)


def test_bind_fa4_exact(prob, sol):
    roles = classify(prob, sol)
    ddg_raw = json.loads(DDG.read_text())
    b = bind(prob, sol, roles, ddg_raw=ddg_raw)
    p = b.params
    assert p["NUM_MMA_GROUPS"] == 2
    assert p["NUM_BUFFERS_KV"] == 3  # K liveness 2 copies + V 1
    assert p["NUM_BUFFERS_QK"] == 2  # qk span 66 > II
    assert p["MMA_PV_SKEW"] == 1  # PV(i) issues one II frame after QK(i)
    assert p["BLOCK_N"] == 64  # solved geometry
    assert p["BLOCK_M"] == 256  # SUB_M 64 -> 128-row build minimum, 2 chains
    assert b.audit["geometry"]["sub_m"] == 64
    assert b.audit["geometry"]["realized_split"] == 128
    assert b.audit["chain_order"] == [1, 2]


def test_bind_without_ddg_keeps_default_geometry(prob, sol):
    roles = classify(prob, sol)
    b = bind(prob, sol, roles)
    assert "BLOCK_M" not in b.params
    assert any("geometry" in d or "qk tile shape" in d
               for d in b.audit.get("dropped", []))


def test_classify_bwd_rejects_free_optimum():
    # The free v6 bwd optimum spreads the 5 MMAs over three warps — the
    # dedicated-issuer skeleton must reject it explicitly, not degrade.
    prob_bwd = load_problem(DDG_BWD)
    sol_free = json.loads(SOL_BWD_FREE.read_text())
    with pytest.raises(RoleClassificationError):
        classify_bwd(prob_bwd, sol_free)


@pytest.mark.skipif(not SOL_BWD_SKC.exists(),
                    reason="skeleton-family probe solution not produced yet")
def test_classify_and_bind_bwd_skeleton_family():
    prob_bwd = load_problem(DDG_BWD)
    sol = json.loads(SOL_BWD_SKC.read_text())
    roles = classify_bwd(prob_bwd, sol)
    ddg_raw = json.loads(DDG_BWD.read_text())
    b = bind_bwd(prob_bwd, sol, roles, ddg_raw=ddg_raw)
    assert b.params["NUM_BUFFERS_Q"] >= 2
    assert b.params["BLOCK_M"] == 128 and b.params["BLOCK_N"] == 128
    # M/D loads sit on the VL warp in the model; skeleton relocates them
    assert any(o["op_kind"] == "tt.load" for o in
               b.audit["protocol_overrides"])
