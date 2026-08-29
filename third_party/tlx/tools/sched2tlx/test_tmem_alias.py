"""Unit tests for TMEM accumulator aliasing (interval coloring).

Covers the case the shipping FA-bwd configs do NOT exercise (aliasing only
triggers above the 512-column budget, i.e. large HEAD_DIM): a MULTI-STAGE /
skewed schedule where a consumer reads an accumulator produced in an EARLIER
iteration. In that regime a within-II-only lifetime inverts (hi < lo) and the
cyclic-overlap test silently misses real conflicts, mis-coloring two live
accumulators onto one TMEM slot. Absolute issue time (stage*II + cycle) fixes
it — these tests pin that behavior.
"""

from __future__ import annotations

import sys

from sched2tlx.schedule_graph import (
    ConstRef,
    Loop,
    Node,
    Op,
    OpRef,
    ScheduleGraph,
    ScheduleLoop,
)
from sched2tlx.emitter import (
    RenderCtx,
    _tmem_alias_groups,
    _tmem_cyclic_occupies,
    _tmem_lifetimes_conflict,
)

ok = True


def check(cond, msg):
    global ok
    print(("  PASS" if cond else "  FAIL") + f": {msg}")
    ok = ok and cond


def test_cyclic_occupies():
    print("== _tmem_cyclic_occupies ==")
    II = 3136
    check(_tmem_cyclic_occupies(100, 400, II) == [(100, 400)], "non-wrapping window")
    # absolute lifetime that crosses the II boundary → splits into two pieces
    check(
        _tmem_cyclic_occupies(2000, 3636, II) == [(2000, 3136), (0, 500)],
        "wraps past II into two pieces",
    )
    check(
        _tmem_cyclic_occupies(0, 5000, II) == [(0, II)], "span >= II covers whole loop"
    )


def test_lifetimes_conflict():
    print("== _tmem_lifetimes_conflict (the multi-stage inversion bug) ==")
    II = 3136
    # accA: produced iter i @cyc2000, consumed iter i+1 @cyc500  -> absolute
    # (2000, 3136+500) = (2000, 3636); cyclic footprint [(2000,3136),(0,500)].
    accA = (2000, 3636)
    # accB: produced+consumed same iter @[300,400]; overlaps accA's (0,500) piece.
    accB = (300, 400)
    check(
        _tmem_lifetimes_conflict(accA, accB, II),
        "prior-iteration consumer overlap IS detected (bug would miss it)",
    )
    # accC lives entirely in a gap accA does not touch -> no conflict.
    accC = (600, 1500)
    check(
        not _tmem_lifetimes_conflict(accA, accC, II),
        "genuinely disjoint lifetimes do NOT conflict",
    )
    check(_tmem_lifetimes_conflict(None, accB, II), "whole-loop (None) conflicts all")


# ---- end-to-end _tmem_alias_groups on a synthetic multi-stage graph ----


def _op(oid, kind, scope, operands, rtypes):
    return Op(
        op_id=oid,
        kind=kind,
        scope=scope,
        operands=operands,
        result_types=rtypes,
        attributes={},
    )


def _node(nid, op_ref, kind, stage, cycle):
    return Node(
        id=nid,
        op_ref=op_ref,
        op_kind=kind,
        pipeline="TC",
        warp_group=0,
        latency=0,
        self_latency=0,
        frequency_multiplier=1,
        schedule_cycle=cycle,
        schedule_stage=stage,
        schedule_cluster=0,
        produces_buffer=None,
        consumes_buffers=[],
    )


def _build(II, accs):
    """accs: list of (acc_id, cols, prod_stage, prod_cyc, cons_stage, cons_cyc)."""
    ops = {}
    nodes = []
    nid = 0
    for aid, cols, ps, pc, cs, cc in accs:
        ops[aid] = _op(
            aid,
            "ttng.tmem_alloc",
            "function",
            [],
            [f"!ttg.memdesc<128x{cols}xf32, #ttng.tensor_memory, mutable>"],
        )
        mma = aid + "_mma"
        ops[mma] = _op(
            mma,
            "ttng.tc_gen5_mma",
            "loop:0",
            [OpRef("dA"), OpRef("dB"), OpRef(aid)],
            [],
        )
        ld = aid + "_ld"
        ops[ld] = _op(
            ld, "ttng.tmem_load", "loop:0", [OpRef(aid)], ["tensor<128x128xf32>"]
        )
        nodes.append(_node(nid, mma, "ttng.tc_gen5_mma", ps, pc))
        nid += 1
        nodes.append(_node(nid, ld, "ttng.tmem_load", cs, cc))
        nid += 1
    sch = ScheduleLoop(
        id=0,
        II=II,
        max_stage=1,
        prologue_latency=0,
        trip_count=8,
        trip_count_estimated=False,
        induction_var_name="i",
        induction_var_type="i32",
        lower_bound=ConstRef(0, "i32"),
        upper_bound=ConstRef(8, "i32"),
        step=ConstRef(1, "i32"),
        buffers=[],
        nodes=nodes,
        edges=[],
    )
    loop = Loop(loop_id=0, is_outer=False, warp_groups=[], schedule=sch)
    return ScheduleGraph(schema_version="1", kernel=None, ops=ops, loops=[loop])


def test_alias_groups_end_to_end():
    print("== _tmem_alias_groups end-to-end (multi-stage) ==")
    II = 1000
    # accA prod@stage0 cyc700, cons@stage1 cyc200 -> absolute (700, 1200) ->
    # cyclic [(700,1000),(0,200)]. accB @[100,150] overlaps accA's (0,200) piece.
    # 512+512 cols > 512 budget so aliasing is active. Correct coloring keeps
    # them in different colors -> nothing aliased -> {}. The stage-blind bug
    # would mis-see them as disjoint and alias them (return both at one color).
    g = _build(
        II,
        [
            ("accA", 512, 0, 700, 1, 200),
            ("accB", 512, 0, 100, 1, 150),
        ],
    )
    res = _tmem_alias_groups(g)
    check(res == {}, "overlapping (cross-iteration) accumulators are NOT aliased")

    # Genuinely disjoint lifetimes SHOULD still alias (fix isn't over-conservative):
    # accA @[100,200], accB @[500,600], same stage, no overlap -> one shared color.
    g2 = _build(
        II,
        [
            ("accA", 512, 0, 100, 0, 200),
            ("accB", 512, 0, 500, 0, 600),
        ],
    )
    res2 = _tmem_alias_groups(g2)
    check(
        res2.get("accA") is not None and res2.get("accA") == res2.get("accB"),
        "disjoint accumulators DO share a color (aliasing still works)",
    )


def _rctx(g):
    return RenderCtx(graph=g, op_var={}, buffer_var={}, alloc_op_var={}, loop_iv={})


def test_alias_groups_serial_program_order():
    """FA-bwd HEAD_DIM=128 regime: the skewed schedule cycles say the qkT and
    dQ accumulators overlap ACROSS iterations (cyclic model → no alias, 544
    TMEM cols → OutOfResources), but the emitter dropped the skew plan for
    that WG (serial fallback) — so the emitted wg body runs iterations
    back-to-back and program order [MMA issue .. last tmem_load] is the true
    lifetime. Passing rctx (which carries the — here empty — skew plan) must
    recover the alias; an ACCEPTED skew plan for the WG must suppress it."""
    print("== _tmem_alias_groups serial-WG program order ==")
    II = 1280
    # Mirrors sg_hd128_depth2 (absolute-cycle convention): accA = qkT tile,
    # prod st0@556 / cons st1@1456; accB = dQ tile, prod st1@2547 / cons
    # st2@3447. Cyclically accB wraps into accA's window → schedule-time model
    # says conflict.
    accs = [
        ("accA", 512, 0, 556, 1, 1456),
        ("accB", 512, 1, 2547, 2, 3447),
    ]
    g = _build(II, accs)
    check(
        _tmem_alias_groups(g) == {},
        "without rctx (schedule-time model) the pair is NOT aliased",
    )
    res = _tmem_alias_groups(g, _rctx(g))
    check(
        res.get("accA") is not None and res.get("accA") == res.get("accB"),
        "serial WG: program-order-disjoint accumulators DO alias",
    )
    # An accepted skew plan for (loop0, wg0) invalidates program order.
    rctx = _rctx(g)
    rctx.skew_plan[(0, 0)] = {"group_of": {}, "n_groups": 2}
    check(
        _tmem_alias_groups(g, rctx) == {},
        "skewed WG falls back to the schedule-time model (no alias)",
    )
    # Program order still detects real overlap: interleave the emitted order
    # to (MMA_C, MMA_D, ld_C, ld_D) — overlapping windows → never aliased.
    g2 = _build(
        II,
        [
            ("accC", 512, 0, 100, 0, 700),
            ("accD", 512, 0, 200, 0, 800),
        ],
    )
    n_c_mma, n_c_ld, n_d_mma, n_d_ld = g2.loops[0].schedule.nodes
    n_c_mma.id, n_d_mma.id, n_c_ld.id, n_d_ld.id = 0, 1, 2, 3
    check(
        _tmem_alias_groups(g2, _rctx(g2)) == {},
        "program-order-overlapping accumulators are NOT aliased",
    )


def main():
    test_cyclic_occupies()
    test_lifetimes_conflict()
    test_alias_groups_end_to_end()
    test_alias_groups_serial_program_order()
    print("\n=== ALL PASS ===" if ok else "\n=== FAILURES ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
