"""Joint SWP + WS constraint system (paper sec 4, Figures 4/5/6), QF_LIA,
discharged to Yices2.

Two entries share this module.  `solve_joint` builds the paper's system and
nothing else: it takes no mode parameters, only the experiment inputs the paper
itself varies (the reduced-warp budget and the cross-warp prohibition).
`solve_joint_audit` is the same paper model plus structural probes, used only
by the FA4 refit instrument.

Encoding note (fidelity): the paper presents a boolean grid op[v,i,t].  Every
instance is scheduled at exactly one cycle (UNIQUENESS), so the grid is in
bijection with integer start times t[v,i]; we make the integers primary and
use arithmetic equality atoms (t[v,i] == c) where a grid cell is needed.
Under this bijection each Figure-4/5/6 constraint maps one-to-one:

  UNIQUENESS   — implicit (an integer variable has exactly one value);
  CONSISTENCY  — t[v,i] == t[v,0] + i*I;
  COMPLETION   — t[v,i] + cycles(v) <= T;
  DEPENDENCE   — t[v,i+delta] >= t[u,i] + d           (== forall t' < t+d: not op);
  CAPACITY     — sum RRT[v][f,c]*ind(t[v,i] == tau-c) <= cap(f);
  MEMORYCAPACITY / INIT / LIVEPROP / DEADPROP — boolean live grid, charging
                   each curated value's own footprint;
  WARPUNIQUENESS / VARIABLELATENCY — boolean opw row per physical warp;
  REGISTERLIMIT — per tau and physical warp;
  CROSS-WARPSPILLS — not samewarp(u,v) =>
                     t[v,i+delta] >= t[u,i] + d + spillcost(u);
  CONCURRENCY  — blocking consumer v, overlapping warp o: t[o,i'] outside
                 [t[v,i]-(cycles(o)-1), t[v,i]].

`opw[v,w]` ranges over the machine's physical warps, so a warp assignment is a
set of warps per operation and sets may overlap.

The system is satisfiability-only; Algorithm 1 (search.py) owns optimality.
"""

from dataclasses import dataclass, field

from yices import Config, Context, Model, Status, Terms, Types

from .ddg import Problem

INT = None  # lazily initialized yices int type


def _int_type():
    global INT
    if INT is None:
        INT = Types.int_type()
    return INT


@dataclass
class JointSolution:
    ii: int
    length: int
    copies: int
    horizon: int
    cycles: dict[int, int]  # t[v, 0] — steady-state modulo schedule M*(v)
    warp_sets: dict[int, tuple[int, ...]]  # A*(v) — physical warps, may overlap
    stats: dict = field(default_factory=dict)


def solve_joint(prob: Problem, ii: int, length: int,
                num_warps_override: int | None = None,
                allow_cross_warp: bool = True):
    """SWP-AND-WS(G, M, I, L): satisfiability of the paper's Fig 4/5/6
    system at fixed (I, L). Returns (JointSolution, "sat") or (None, "unsat")."""
    return _solve_joint_paper(
        prob, ii, length,
        num_warps_override=num_warps_override,
        allow_cross_warp=allow_cross_warp,
        profile="paper",
    )


def solve_joint_audit(prob: Problem, ii: int, length: int,
                      num_warps_override: int | None = None,
                      allow_cross_warp: bool = True,
                      colocate: list[list[int]] | None = None,
                      separate: list[tuple[int, int]] | None = None,
                      exact_warp_sets: int | None = None):
    """Audit-only structural probes over the paper model; no paper
    counterpart. FA4-refit instrument (refit_check / strategy_report)."""
    return _solve_joint_paper(
        prob, ii, length,
        num_warps_override=num_warps_override,
        allow_cross_warp=allow_cross_warp,
        colocate=colocate,
        separate=separate,
        exact_warp_sets=exact_warp_sets,
        profile="audit",
    )


def _solve_joint_paper(prob: Problem, ii: int, length: int,
                       num_warps_override: int | None = None,
                       allow_cross_warp: bool = True,
                       colocate: list[list[int]] | None = None,
                       separate: list[tuple[int, int]] | None = None,
                       exact_warp_sets: int | None = None,
                       profile: str = "paper"):
    machine = prob.machine
    if ii < 1 or length < 1:
        raise ValueError("initiation interval and schedule length must be positive")
    # Figure 6 quantifies opw over the machine's physical warps, so the warp
    # budget is the variable domain rather than a separate constraint.
    B = (
        num_warps_override
        if num_warps_override is not None
        else machine.num_warps
    )
    if B < 1:
        raise ValueError("physical warp budget must be positive")
    copies = -(-length // ii)
    T = (copies - 1) * ii + length
    ids = list(prob.nodes)
    assignable_ids = list(ids)
    assignable = set(assignable_ids)
    if not assignable_ids:
        raise ValueError("DDG has no operations to partition")
    if exact_warp_sets is not None and not 1 <= exact_warp_sets <= len(
        assignable_ids
    ):
        raise ValueError(
            f"exact warp-set count must be in [1, {len(assignable_ids)}]"
        )
    it = _int_type()

    cfg = Config()
    cfg.default_config_for_logic("QF_LIA")
    ctx = Context(cfg)
    asserts = []

    # ---- primary variables -------------------------------------------------
    base_t = {v: Terms.new_uninterpreted_term(it, f"t_{v}_0") for v in ids}
    t = {
        (v, i): (
            base_t[v]
            if i == 0
            else Terms.add(base_t[v], Terms.integer(i * ii))
        )
        for v in ids
        for i in range(copies)
    }
    opw = {(v, w): Terms.new_uninterpreted_term(Types.bool_type(),
                                                f"w_{v}_{w}")
           for v in assignable_ids for w in range(B)}

    def same_warpset(u, v):
        if u not in assignable or v not in assignable:
            return Terms.true()
        return Terms.yand(
            [Terms.iff(opw[(u, w)], opw[(v, w)]) for w in range(B)]
        )

    def overlaps_warp(u, v):
        if u not in assignable or v not in assignable:
            return Terms.true()
        return Terms.yor(
            [Terms.yand([opw[(u, w)], opw[(v, w)]]) for w in range(B)]
        )

    live = {}
    ind_cache = {}

    def ind(v, i, tau):  # op[v,i,tau] as an arithmetic atom
        cycle = tau - i * ii
        if not 0 <= cycle <= length - max(1, prob.lat[v]):
            return Terms.false()
        key = (v, cycle)
        if key not in ind_cache:
            ind_cache[key] = Terms.arith_eq_atom(
                t[(v, 0)], Terms.integer(cycle)
            )
        return ind_cache[key]

    zero, one = Terms.integer(0), Terms.integer(1)

    # ---- Figure 4: modulo schedule ----------------------------------------
    for v in ids:
        # CONSISTENCY substitutes every copy with t[v,0] + i*I. Since I is
        # positive, the first copy supplies the strongest lower bound and the
        # last copy reduces the strongest completion bound to L.
        asserts.append(Terms.arith_geq_atom(t[(v, 0)], zero))
        asserts.append(
            Terms.arith_leq_atom(
                t[(v, 0)],
                Terms.integer(length - max(1, prob.lat[v])),
            )
        )
    for e in prob.edges:  # DEPENDENCE
        d = prob.edge_latency(e)
        if e.distance < copies:
            asserts.append(
                Terms.arith_geq_atom(
                    Terms.sub(t[(e.dst, e.distance)], t[(e.src, 0)]),
                    Terms.integer(d),
                )
            )

    # CAPACITY
    for pipe, cap in machine.capacities.items():
        reservations = [
            (v, cycle, amount)
            for v in ids
            for functional_unit, cycle, amount in prob.reservations(v)
            if functional_unit == pipe
        ]
        if not reservations:
            continue
        for tau in range(T):
            terms = []
            for v, cycle, amount in reservations:
                for i in range(copies):
                    start = tau - cycle
                    if 0 <= start < T:
                        terms.append(
                            Terms.ite(
                                ind(v, i, start),
                                Terms.integer(amount),
                                zero,
                            )
                        )
            if terms:
                asserts.append(Terms.arith_leq_atom(
                    Terms.sum(terms), Terms.integer(cap)))

    # ---- Figure 5: liveness + memory capacity -----------------------------
    consumers: dict[int, list] = {}
    for e in prob.edges:
        consumers.setdefault(e.src, []).append(e)
    carried = {e.src for e in prob.edges if e.distance > 0}
    for v in ids:
        # INIT: only loop-carried results of the last copy live at time T.
        for i in range(copies):
            live[(v, i, T)] = (
                Terms.true()
                if i == copies - 1 and v in carried
                else Terms.false()
            )
        for i in range(copies):
            for tau in range(T, 0, -1):
                lv = live[(v, i, tau)]
                o = ind(v, i, tau) if tau < T else Terms.false()
                uses = []
                for e in consumers.get(v, ()):  # noqa: B023
                    j = i + e.distance
                    if j < copies and tau < T:
                        uses.append(ind(e.dst, j, tau))
                any_use = Terms.yor(uses) if uses else Terms.false()
                # LIVEPROP/DEADPROP uniquely determine the preceding cell, so
                # use their equivalent Boolean recurrence.
                live[(v, i, tau - 1)] = Terms.ite(lv, Terms.ynot(o), any_use)

    # MEMORYCAPACITY, per curated value. Aliasing is resolved by the curation
    # stage, so the paper's literal per-value sum is also the physical one.
    for kind, capacity in (("smem", machine.smem_bytes),
                           ("tmem", machine.tmem_cols)):
        holders = [v for v in ids if prob.footprint(v, kind) > 0]
        if not holders:
            continue
        for tau in range(T + 1):
            terms = [
                Terms.ite(live[(v, i, tau)],
                          Terms.integer(prob.footprint(v, kind)), zero)
                for v in holders
                for i in range(copies)
            ]
            asserts.append(Terms.arith_leq_atom(
                Terms.sum(terms), Terms.integer(capacity)))

    # ---- Figure 6: warp assignment ----------------------------------------
    W_VL = 0  # designated variable-latency warp
    for v in assignable_ids:
        # WARPUNIQUENESS, extended to operations spanning several warps.
        asserts.append(
            Terms.arith_eq_atom(
                Terms.sum(
                    [Terms.ite(opw[(v, w)], one, zero) for w in range(B)]
                ),
                Terms.integer(prob.nodes[v].min_warps),
            )
        )
        # VARIABLELATENCY (iff: only variable-latency ops sit on W_vl)
        is_vl = v in prob.variable_latency
        asserts.append(Terms.iff(opw[(v, W_VL)],
                                 Terms.true() if is_vl else Terms.false()))

    # REGISTERLIMIT, asserted per physical warp. A multi-warp operation
    # distributes its value across its member warps (INTERPRETATIONS-2).
    reg_users = [v for v in assignable_ids if prob.regs[v] > 0]
    if reg_users:
        for w in range(B):
            for tau in range(T + 1):
                terms = [
                    Terms.ite(
                        Terms.yand([live[(v, i, tau)], opw[(v, w)]]),
                        Terms.integer(
                            (prob.regs[v] + prob.nodes[v].min_warps - 1)
                            // prob.nodes[v].min_warps
                        ),
                        zero,
                    )
                    for v in reg_users
                    for i in range(copies)
                ]
                asserts.append(
                    Terms.arith_leq_atom(
                        Terms.sum(terms),
                        Terms.integer(machine.regs_per_warp),
                    )
                )

    # Audit-only structural probes: force listed node sets onto one warp set,
    # pairs onto different warp sets, or fix the number of distinct sets.
    for group in (colocate or []):
        for a, b in zip(group, group[1:]):
            asserts.append(same_warpset(a, b))
    for a, b in (separate or []):
        asserts.append(Terms.ynot(same_warpset(a, b)))
    if exact_warp_sets is not None:
        ordered = sorted(assignable_ids)
        leaders = []
        for index, v in enumerate(ordered):
            is_leader = (
                Terms.yand(
                    [Terms.ynot(same_warpset(u, v)) for u in ordered[:index]]
                )
                if index
                else Terms.true()
            )
            leaders.append(Terms.ite(is_leader, one, zero))
        asserts.append(
            Terms.arith_eq_atom(
                Terms.sum(leaders), Terms.integer(exact_warp_sets)
            )
        )

    # CROSS-WARPSPILLS (+ prohibition when cross-warp traffic is disabled —
    # the paper's third UNSAT ablation, whose edge domain is register data).
    for e in prob.edges:
        sw = same_warpset(e.src, e.dst)
        if not allow_cross_warp:
            if prob.regs.get(e.src, 0) > 0:
                asserts.append(sw)
            continue
        d = prob.edge_latency(e)
        if e.distance < copies:
            asserts.append(
                Terms.yor(
                    [
                        sw,
                        Terms.arith_geq_atom(
                            Terms.sub(
                                t[(e.dst, e.distance)], t[(e.src, 0)]
                            ),
                            Terms.integer(d + prob.spill[e.src]),
                        ),
                    ]
                )
            )

    # CONCURRENCY: a blocking wait stalls in-order issue on the waiter's
    # warps.  Waits come from (a) consumers of asynchronous TC/TMA results
    # (the curated blocking edges), and (b) consumers of cross-warp spills:
    # "While elided for brevity, the implementation includes a similar
    # constraint for cross-warp spills, as transferring register data through
    # shared memory requires blocking synchronization" (paper sec 4.3).  (b)
    # is conditional on the warp assignment, so it is gated on "some in-edge
    # is cross-warp".
    blocking_consumers = {v for (_, v) in prob.blocking}
    in_edges: dict[int, list] = {}
    for e in prob.edges:
        if e.src != e.dst:
            in_edges.setdefault(e.dst, []).append(e)
    for v in assignable_ids:
        if v in blocking_consumers:
            gate = None  # unconditional
        else:
            preds = [
                e for e in in_edges.get(v, ()) if prob.regs.get(e.src, 0) > 0
            ]
            if not preds:
                continue
            gate = Terms.yor(
                [Terms.ynot(same_warpset(e.src, v)) for e in preds]
            )
        for o in assignable_ids:
            if o == v:
                continue
            sw = overlaps_warp(o, v)
            cond = sw if gate is None else Terms.yand([gate, sw])
            # Paper Figure 6 excludes ISSUE points in
            # [t[v, i] - (cycles(o) - 1), t[v, i]]. A zero-cycle operation
            # therefore has an empty exclusion window.
            win = prob.lat[o]
            if win == 0:
                continue
            copy_pairs = [
                (max(0, -offset), max(0, -offset) + offset)
                for offset in range(-(copies - 1), copies)
            ]
            for i, ip in copy_pairs:
                lo_gap = Terms.arith_leq_atom(
                    Terms.sub(t[(o, ip)], t[(v, i)]),
                    Terms.integer(-win),
                )
                hi_gap = Terms.arith_gt_atom(
                    Terms.sub(t[(o, ip)], t[(v, i)]), zero
                )
                asserts.append(
                    Terms.implies(cond, Terms.yor([lo_gap, hi_gap]))
                )

    ctx.assert_formulas(asserts)
    status = ctx.check_context()
    if status != Status.SAT:
        ctx.dispose()
        cfg.dispose()
        if status != Status.UNSAT:
            raise RuntimeError(f"yices returned {status}")
        return None, "unsat"
    model = Model.from_context(ctx, 1)
    cycles = {v: model.get_value(t[(v, 0)]) for v in ids}
    warp_sets = {
        v: tuple(w for w in range(B) if model.get_value(opw[(v, w)]))
        for v in assignable_ids
    }
    warps_used = len({w for warps in warp_sets.values() for w in warps})

    def peak_storage(kind: str) -> int:
        holders = [v for v in ids if prob.footprint(v, kind) > 0]
        return max(
            (
                sum(
                    prob.footprint(v, kind)
                    for v in holders
                    for i in range(copies)
                    if model.get_value(live[(v, i, tau)])
                )
                for tau in range(T + 1)
            ),
            default=0,
        )

    tmem_values = sum(1 for v in ids if prob.footprint(v, "tmem") > 0)
    sol = JointSolution(ii=ii, length=length, copies=copies, horizon=T,
                        cycles=cycles, warp_sets=warp_sets,
                        stats={**prob.normalization_stats(),
                               "profile": profile,
                               "num_asserts": len(asserts), "T": T,
                               "num_warp_sets": len(set(warp_sets.values())),
                               "warps_used": warps_used,
                               "num_warps": warps_used,
                               "physical_warp_budget": B,
                               "regs_per_warp": machine.regs_per_warp,
                               "smem_capacity": machine.smem_bytes,
                               "tmem_capacity": machine.tmem_cols,
                               "tmem_value_count": tmem_values,
                               "peak_tmem_cols": peak_storage("tmem"),
                               "time_variables": len(ids),
                               "live_variables": 0})
    ctx.dispose()
    cfg.dispose()
    return sol, "sat"
