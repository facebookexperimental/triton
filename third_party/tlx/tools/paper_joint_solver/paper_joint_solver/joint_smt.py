"""Joint SWP + WS constraint system (paper sec 4, Figures 4/5/6), QF_LIA,
discharged to Yices2.

Encoding note (fidelity): the paper presents a boolean grid op[v,i,t].  Every
instance is scheduled at exactly one cycle (UNIQUENESS), so the grid is in
bijection with integer start times t[v,i]; we make the integers primary and
use arithmetic equality atoms (t[v,i] == c) where a grid cell is needed.
Under this bijection each Figure-4/5/6 constraint maps one-to-one:

  UNIQUENESS   — implicit (an integer variable has exactly one value);
  CONSISTENCY  — t[v,i] == t[v,0] + i*I;
  COMPLETION   — t[v,i] + cycles(v) <= T;
  DEPENDENCE   — t[v,i+delta] >= t[u,i] + d           (== forall t' < t+d: not op);
  CAPACITY     — sum_{v,i,c} ind(t[v,i] == tau - c) <= cap(f)  per tau, f;
  MEMORYCAPACITY / INIT / LIVEPROP / DEADPROP — boolean live grid, verbatim;
  WARPUNIQUENESS / VARIABLELATENCY — boolean opw row per op;
  REGISTERLIMIT — per tau, w: sum ind(live and opw)*regs <= limit;
  CROSS-WARPSPILLS — not samewarp(u,v) => t[v,i+delta] >= t[u,i] + d + spill;
  CONCURRENCY  — blocking consumer v, same-warp o: t[o,i'] outside
                 [t[v,i]-(cycles(o)-1), t[v,i]].

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
    warp: dict[int, int]  # A*(v)
    stats: dict = field(default_factory=dict)


def _footprints(prob: Problem) -> dict[int, tuple[str, int]]:
    """footprint(v, m): (memory kind, amount) for v's result."""
    out = {}
    for v in prob.nodes.values():
        if v.pipeline == "TMA" and "load" in v.op_kind:
            out[v.id] = ("smem", prob.regs[v.id] * 4)  # staged tile bytes
        elif "mma" in v.op_kind:
            out[v.id] = ("tmem", 64)  # accumulator columns per instance
        elif prob.regs[v.id] > 0:
            out[v.id] = ("reg", prob.regs[v.id])
    return out


def solve_joint(prob: Problem, ii: int, length: int,
                num_warps: int | None = None,
                allow_cross_warp: bool = True,
                timeout_s: float | None = None,
                symmetry_breaking: bool = True,
                colocate: list[list[int]] | None = None,
                separate: list[tuple[int, int]] | None = None):
    """SWP-AND-WS(G, M, I, L): satisfiability of the joint system at fixed
    (I, L).  Returns the discovered (M*, A*) or None if UNSAT."""
    machine = prob.machine
    W = num_warps if num_warps is not None else machine.num_warpgroups
    copies = -(-length // ii)
    T = (copies - 1) * ii + length
    ids = list(prob.nodes)
    it = _int_type()

    cfg = Config()
    cfg.default_config_for_logic("QF_LIA")
    ctx = Context(cfg)
    asserts = []

    # ---- primary variables -------------------------------------------------
    t = {(v, i): Terms.new_uninterpreted_term(it, f"t_{v}_{i}")
         for v in ids for i in range(copies)}
    opw = {(v, w): Terms.new_uninterpreted_term(Types.bool_type(),
                                                f"w_{v}_{w}")
           for v in ids for w in range(W)}
    live = {(v, i, tau): Terms.new_uninterpreted_term(Types.bool_type(),
                                                      f"l_{v}_{i}_{tau}")
            for v in ids for i in range(copies) for tau in range(T + 1)}

    def ind(v, i, tau):  # op[v,i,tau] as an arithmetic atom
        return Terms.arith_eq_atom(t[(v, i)], Terms.integer(tau))

    zero, one = Terms.integer(0), Terms.integer(1)

    # ---- Figure 4: modulo schedule ----------------------------------------
    for v in ids:
        for i in range(copies):
            asserts.append(Terms.arith_geq_atom(t[(v, i)], zero))
            # COMPLETION
            asserts.append(Terms.arith_leq_atom(
                t[(v, i)], Terms.integer(T - prob.lat[v])))
            if i:  # CONSISTENCY
                asserts.append(Terms.arith_eq_atom(
                    t[(v, i)],
                    Terms.add(t[(v, 0)], Terms.integer(i * ii))))
    for e in prob.edges:  # DEPENDENCE
        d = prob.edge_lat[(e.src, e.dst, e.distance)]
        for i in range(copies):
            j = i + e.distance
            if j >= copies:
                continue
            asserts.append(Terms.arith_geq_atom(
                Terms.sub(t[(e.dst, j)], t[(e.src, i)]), Terms.integer(d)))

    # CAPACITY
    for pipe, cap in machine.capacities.items():
        users = [v for v in ids if prob.nodes[v].pipeline == pipe
                 and prob.occ[v] > 0]
        if not users:
            continue
        for tau in range(T):
            terms = []
            for v in users:
                for i in range(copies):
                    for c in range(prob.occ[v]):
                        if 0 <= tau - c <= T - prob.lat[v]:
                            terms.append(Terms.ite(ind(v, i, tau - c),
                                                   one, zero))
            if len(terms) > cap:
                asserts.append(Terms.arith_leq_atom(
                    Terms.sum(terms), Terms.integer(cap)))

    # ---- Figure 5: liveness + memory capacity -----------------------------
    consumers: dict[int, list] = {}
    for e in prob.edges:
        consumers.setdefault(e.src, []).append(e)
    carried = {e.src for e in prob.edges if e.distance > 0}
    for v in ids:
        # INIT: only loop-carried results of the last copy live at time T.
        want = Terms.true() if v in carried else Terms.false()
        asserts.append(Terms.iff(live[(v, copies - 1, T)], want))
        for i in range(copies):
            if i != copies - 1:
                asserts.append(Terms.ynot(live[(v, i, T)]))
            for tau in range(T, 0, -1):
                lv, lp = live[(v, i, tau)], live[(v, i, tau - 1)]
                o = ind(v, i, tau) if tau < T else Terms.false()
                # LIVEPROP-1 / LIVEPROP-2
                asserts.append(Terms.implies(Terms.yand([lv, o]),
                                             Terms.ynot(lp)))
                asserts.append(Terms.implies(
                    Terms.yand([lv, Terms.ynot(o)]), lp))
                # DEADPROP-1 / DEADPROP-2
                uses = []
                for e in consumers.get(v, ()):  # noqa: B023
                    j = i + e.distance
                    if j < copies and tau < T:
                        uses.append(ind(e.dst, j, tau))
                any_use = Terms.yor(uses) if uses else Terms.false()
                asserts.append(Terms.implies(
                    Terms.yand([Terms.ynot(lv), any_use]), lp))
                asserts.append(Terms.implies(
                    Terms.yand([Terms.ynot(lv), Terms.ynot(any_use)]),
                    Terms.ynot(lp)))

    fp = _footprints(prob)
    for kind, capacity in (("smem", machine.smem_bytes
                            - machine.smem_fixed_overhead),
                           ("tmem", machine.tmem_cols
                            - machine.tmem_fixed_cols)):
        users = [v for v in ids if fp.get(v, ("", 0))[0] == kind]
        if not users:
            continue
        for tau in range(T + 1):
            terms = [Terms.ite(live[(v, i, tau)],
                               Terms.integer(fp[v][1]), zero)
                     for v in users for i in range(copies)]
            asserts.append(Terms.arith_leq_atom(Terms.sum(terms),
                                                Terms.integer(capacity)))

    # ---- Figure 6: warp assignment ----------------------------------------
    W_VL = 0  # designated variable-latency warp
    for v in ids:
        row = [opw[(v, w)] for w in range(W)]
        asserts.append(Terms.yor(row))
        for a in range(W):
            for b in range(a + 1, W):
                asserts.append(Terms.ynot(Terms.yand([row[a], row[b]])))
        # VARIABLELATENCY (iff: only variable-latency ops sit on W_vl)
        is_vl = v in prob.variable_latency
        asserts.append(Terms.iff(opw[(v, W_VL)],
                                 Terms.true() if is_vl else Terms.false()))

    # Symmetry breaking: warp ids above W_vl are interchangeable labels, and
    # symmetric op chains (the two sub-tile softmax chains) make the label
    # permutations explode the search.  Require non-gapped usage: warp w can
    # host ops only if warp w-1 does.  Solution-preserving up to relabeling.
    if symmetry_breaking:
        def used(w):
            return Terms.yor([opw[(v, w)] for v in ids])
        for w in range(2, W):
            asserts.append(Terms.implies(used(w), used(w - 1)))

    def samewarp(u, v):
        return Terms.yor([Terms.yand([opw[(u, w)], opw[(v, w)]])
                          for w in range(W)])

    # Code-generator realizability: loop-carried values that live in
    # registers (the m_i / l_i online-softmax recurrences) cannot cross warp
    # groups — the emitter routes cross-group data through SMEM/TMEM
    # channels, which register iter_args don't get.  Buffer-mediated carried
    # values (TC accumulators, TMA tiles) are exempt.
    for e in prob.edges:
        if (e.distance >= 1 and e.src != e.dst
                and prob.regs.get(e.src, 0) > 0
                and prob.nodes[e.src].pipeline in ("CUDA", "SFU", "NONE")):
            asserts.append(samewarp(e.src, e.dst))

    # Optional structural probes: force listed node sets onto one warp, or
    # pairs onto different warps.
    for group in (colocate or []):
        for a, b in zip(group, group[1:]):
            asserts.append(samewarp(a, b))
    for a, b in (separate or []):
        asserts.append(Terms.ynot(samewarp(a, b)))

    # REGISTERLIMIT
    reg_users = [v for v in ids if fp.get(v, ("", 0))[0] == "reg"]
    for w in range(W):
        for tau in range(T + 1):
            terms = [Terms.ite(Terms.yand([live[(v, i, tau)], opw[(v, w)]]),
                               Terms.integer(fp[v][1]), zero)
                     for v in reg_users for i in range(copies)]
            if terms:
                asserts.append(Terms.arith_leq_atom(
                    Terms.sum(terms),
                    Terms.integer(machine.regs_per_warpgroup)))

    # CROSS-WARPSPILLS (+ prohibition when cross-warp traffic is disabled —
    # the paper's third UNSAT ablation)
    for e in prob.edges:
        d = prob.edge_lat[(e.src, e.dst, e.distance)]
        sw = samewarp(e.src, e.dst)
        for i in range(copies):
            j = i + e.distance
            if j >= copies:
                continue
            if allow_cross_warp:
                asserts.append(Terms.yor([sw, Terms.arith_geq_atom(
                    Terms.sub(t[(e.dst, j)], t[(e.src, i)]),
                    Terms.integer(d + prob.spill))]))
            else:
                asserts.append(sw)

    # CONCURRENCY: a blocking wait stalls in-order issue on the waiter's
    # warp.  Waits come from (a) consumers of asynchronous TC/TMA results
    # (static), and (b) consumers of cross-warp spills — the paper notes the
    # spill case gets the same treatment, since receiving register data
    # through shared memory also blocks.  (b) is conditional on the warp
    # assignment, so it is gated on "some in-edge is cross-warp".
    blocking_consumers = {v for (_, v) in prob.blocking}
    in_edges: dict[int, list] = {}
    for e in prob.edges:
        if e.src != e.dst:
            in_edges.setdefault(e.dst, []).append(e)
    for v in ids:
        if v in blocking_consumers:
            gate = None  # unconditional
        else:
            preds = in_edges.get(v, ())
            if not preds:
                continue
            gate = Terms.yor([Terms.ynot(samewarp(e.src, v)) for e in preds])
        for o in ids:
            if o == v:
                continue
            sw = samewarp(o, v)
            cond = sw if gate is None else Terms.yand([gate, sw])
            # The wait excludes other ops from the warp's ISSUE slots.  An
            # asynchronous op (TC/TMA) holds its warp only for the issue
            # cycle — its execution runs on the fixed-function unit (paper
            # Fig 2); a synchronous op occupies the warp for its full
            # duration cycles(o) — the paper's window bound, the same
            # quantity COMPLETION uses, i.e. normalized lat.
            async_issue = prob.nodes[o].pipeline in ("TC", "TMA")
            win = 1 if async_issue else max(1, prob.lat[o])
            for i in range(copies):
                for ip in range(copies):
                    lo_gap = Terms.arith_leq_atom(
                        Terms.sub(t[(o, ip)], t[(v, i)]),
                        Terms.integer(-win))
                    hi_gap = Terms.arith_gt_atom(
                        Terms.sub(t[(o, ip)], t[(v, i)]), zero)
                    asserts.append(Terms.implies(
                        cond, Terms.yor([lo_gap, hi_gap])))

    ctx.assert_formulas(asserts)
    status = ctx.check_context(timeout=int(timeout_s) if timeout_s else None)
    if status != Status.SAT:
        ctx.dispose()
        cfg.dispose()
        # UNSAT is a verdict; anything else (timeout/interrupt) is not — the
        # distinction matters for the ablation claims.
        return None, ("unsat" if status == Status.UNSAT else "unknown")
    model = Model.from_context(ctx, 1)
    cycles = {v: model.get_value(t[(v, 0)]) for v in ids}
    warp = {}
    for v in ids:
        for w in range(W):
            if model.get_value(opw[(v, w)]):
                warp[v] = w
                break
    sol = JointSolution(ii=ii, length=length, copies=copies, horizon=T,
                        cycles=cycles, warp=warp,
                        stats={"num_asserts": len(asserts), "T": T,
                               "num_warps": W})
    ctx.dispose()
    cfg.dispose()
    return sol, "sat"
