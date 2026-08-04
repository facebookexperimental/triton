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
  CAPACITY     — sum RRT[v][f,c]*ind(t[v,i] == tau-c) <= cap(f);
  MEMORYCAPACITY / INIT / LIVEPROP / DEADPROP — boolean live grid, with
                   alias-equivalent SMEM/TMEM values charged once;
  WARPUNIQUENESS / VARIABLELATENCY — boolean opw row per op;
  REGISTERLIMIT — per tau and physical warp, with group widths explicit;
  CROSS-WARPSPILLS — not samewarp(u,v) =>
                     t[v,i+delta] >= t[u,i] + d + spillcost(u);
  CONCURRENCY  — blocking consumer v, same-warp o: t[o,i'] outside
                 [t[v,i]-(cycles(o)-1), t[v,i]].

Mutable memdesc accumulators are the one place where the compiler DDG is not a
literal SSA-value graph: an MMA mutates a TMEM storage object and exposes
completion through an async-token edge.  For Figure-5 liveness, token edges
whose source is a member of a memory object therefore act as synthesized uses
of that object.  Pure synchronization tokens remain excluded.

The emitter represents physical warp sets as group labels.  Width variables make
that representation explicit in the solver: each used group has width 1/2/4/8,
the sum is bounded by the physical warp budget. Each operation selects an
explicit physical-lane mask of the width required by its value layout, and
REGISTERLIMIT is asserted for every lane. Emitter compatibility modes can add
prefix-mask and full-group-width restrictions; the paper model itself does not.
The system is satisfiability-only; Algorithm 1 (search.py) owns optimality.
"""

from dataclasses import dataclass, field

from yices import Config, Context, Model, Status, Terms, Types

from .ddg import Problem
from .machine import (
    physical_group_width,
    register_allocation_words,
    required_group_width,
)

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
    warp: dict[int, int]  # emitter-compatible group label for A*(v)
    group_widths: dict[int, int]  # physical warps in each used group
    lane_masks: dict[int, tuple[int, ...]]  # local physical lanes used by op
    stats: dict = field(default_factory=dict)


def _maximum_useful_group_labels(required_width, variable_latency,
                                 physical_warp_budget):
    """Count group labels that can possibly be active under the warp budget."""
    variable_width = max(
        (required_width[node_id] for node_id in variable_latency),
        default=0,
    )
    remaining_budget = physical_warp_budget - variable_width
    regular_widths = [
        width
        for node_id, width in required_width.items()
        if node_id not in variable_latency
    ]
    if not regular_widths:
        return 1

    # At least one widest regular group is needed to host the widest ops.
    widest = max(regular_widths)
    regular_widths.remove(widest)
    used = widest
    regular_groups = 1
    for width in sorted(regular_widths):
        if used + width > remaining_budget:
            break
        used += width
        regular_groups += 1
    # Label zero is reserved for the variable-latency group, even if unused.
    return 1 + regular_groups


def solve_joint(prob: Problem, ii: int, length: int,
                num_warps: int | None = None,
                max_groups: int | None = None,
                allow_cross_warp: bool = True,
                no_cross_warp_domain: str = "reg-data",
                timeout_s: float | None = None,
                symmetry_breaking: bool = True,
                prefix_lane_masks: bool = False,
                full_group_lane_masks: bool = False,
                spill_smem_footprint: bool = True,
                reg_carried_same_lane: bool = True,
                colocate: list[list[int]] | None = None,
                separate: list[tuple[int, int]] | None = None,
                compact_encoding: bool = True,
                exact_groups: int | None = None):
    """SWP-AND-WS(G, M, I, L): satisfiability of the joint system at fixed
    (I, L). `num_warps` is the physical warp budget; `max_groups` only
    bounds the emitter-facing partition labels. `full_group_lane_masks` is a
    backend realizability constraint: every member of a group must require
    and occupy that group's full physical width. Returns (M*, A*) or None."""
    machine = prob.machine
    if ii < 1 or length < 1:
        raise ValueError("initiation interval and schedule length must be positive")
    if no_cross_warp_domain not in ("reg-data", "all-ws"):
        raise ValueError(
            "no-cross-warp domain must be 'reg-data' or 'all-ws'"
        )
    if full_group_lane_masks and not prefix_lane_masks:
        raise ValueError(
            "full-group lane masks require prefix-lane-mask lowering"
        )
    requested_groups = (
        max_groups if max_groups is not None else machine.num_warpgroups
    )
    physical_warp_budget = (
        num_warps if num_warps is not None else machine.num_warps
    )
    scheduled_warp_budget = machine.scheduled_warp_budget(num_warps)
    if not 0 <= machine.regs_fixed_overhead < machine.regs_per_sm:
        raise ValueError("fixed register overhead must leave register capacity")
    if not 0 <= machine.smem_fixed_overhead <= machine.smem_bytes:
        raise ValueError("fixed SMEM overhead exceeds machine capacity")
    if not 0 <= machine.tmem_fixed_cols <= machine.tmem_cols:
        raise ValueError("fixed TMEM overhead exceeds machine capacity")
    if requested_groups < 1:
        raise ValueError("group budget must be positive")
    if requested_groups > machine.num_warpgroups:
        raise ValueError(
            f"group budget {requested_groups} exceeds "
            f"machine limit {machine.num_warpgroups}"
        )
    copies = -(-length // ii)
    T = (copies - 1) * ii + length
    ids = list(prob.nodes)
    data_edges = [edge for edge in prob.edges if not edge.signal_only]
    memory_members = {
        member
        for storage in prob.memory_objects.values()
        for member in storage.members
    }
    liveness_edges = [
        edge
        for edge in prob.edges
        if not edge.signal_only or edge.src in memory_members
    ]
    ws_data_edges = [
        edge for edge in data_edges if prob.edge_has_ws_semantics(edge)
    ]
    assignable_ids = list(ids)
    assignable = set(assignable_ids)
    if not assignable_ids:
        raise ValueError("DDG has no operations to partition")
    required_width = {
        v: required_group_width(prob.nodes[v].min_warps)
        for v in assignable_ids
    }
    if (
        not machine.group_widths
        or any(width <= 0 for width in machine.group_widths)
        or len(set(machine.group_widths)) != len(machine.group_widths)
    ):
        raise ValueError("group widths must be unique positive integers")
    missing_widths = set(required_width.values()) - set(machine.group_widths)
    if missing_widths:
        raise ValueError(
            f"machine group widths do not cover required widths "
            f"{sorted(missing_widths)}"
        )
    W = min(
        requested_groups,
        _maximum_useful_group_labels(
            required_width,
            prob.variable_latency,
            scheduled_warp_budget,
        ),
    )
    if exact_groups is not None and not 1 <= exact_groups <= W:
        raise ValueError(f"exact group count must be in [1, {W}]")
    encoded_group_widths = tuple(sorted(machine.group_widths))
    max_group_width = max(encoded_group_widths)
    it = _int_type()

    cfg = Config()
    cfg.default_config_for_logic("QF_LIA")
    ctx = Context(cfg)
    asserts = []

    # ---- primary variables -------------------------------------------------
    if compact_encoding:
        base_t = {
            v: Terms.new_uninterpreted_term(it, f"t_{v}_0") for v in ids
        }
        t = {
            (v, i): (
                base_t[v]
                if i == 0
                else Terms.add(base_t[v], Terms.integer(i * ii))
            )
            for v in ids
            for i in range(copies)
        }
    else:
        t = {
            (v, i): Terms.new_uninterpreted_term(it, f"t_{v}_{i}")
            for v in ids
            for i in range(copies)
        }
    opw = {(v, w): Terms.new_uninterpreted_term(Types.bool_type(),
                                                f"w_{v}_{w}")
           for v in assignable_ids for w in range(W)}
    if compact_encoding:
        lane_member = {
            (v, lane): Terms.new_uninterpreted_term(
                Types.bool_type(), f"lane_{v}_{lane}"
            )
            for v in assignable_ids
            for lane in range(max_group_width)
        }
        oplane = {}
    else:
        lane_member = {}
        oplane = {
            (v, w, lane): Terms.new_uninterpreted_term(
                Types.bool_type(), f"lane_{v}_{w}_{lane}"
            )
            for v in assignable_ids
            for w in range(W)
            for lane in range(max_group_width)
        }

    def same_group(u, v):
        if u not in assignable or v not in assignable:
            return Terms.true()
        return Terms.yor(
            [Terms.yand([opw[(u, w)], opw[(v, w)]]) for w in range(W)]
        )

    def same_lane_mask(u, v):
        if u not in assignable or v not in assignable:
            return Terms.true()
        if compact_encoding:
            return Terms.yand(
                [
                    same_group(u, v),
                    *[
                        Terms.iff(
                            lane_member[(u, lane)],
                            lane_member[(v, lane)],
                        )
                        for lane in range(max_group_width)
                    ],
                ]
            )
        return Terms.yor(
            [
                Terms.yand(
                    [
                        opw[(u, w)],
                        opw[(v, w)],
                        *[
                            Terms.iff(
                                oplane[(u, w, lane)],
                                oplane[(v, w, lane)],
                            )
                            for lane in range(max_group_width)
                        ],
                    ]
                )
                for w in range(W)
            ]
        )

    def overlaps_lane(u, v):
        if u not in assignable or v not in assignable:
            return Terms.true()
        if compact_encoding:
            return Terms.yand(
                [
                    same_group(u, v),
                    Terms.yor(
                        [
                            Terms.yand(
                                [
                                    lane_member[(u, lane)],
                                    lane_member[(v, lane)],
                                ]
                            )
                            for lane in range(max_group_width)
                        ]
                    ),
                ]
            )
        return Terms.yor(
            [
                Terms.yand(
                    [
                        opw[(u, w)],
                        opw[(v, w)],
                        oplane[(u, w, lane)],
                        oplane[(v, w, lane)],
                    ]
                )
                for w in range(W)
                for lane in range(max_group_width)
            ]
        )

    live = {}
    if not compact_encoding:
        live = {
            (v, i, tau): Terms.new_uninterpreted_term(
                Types.bool_type(), f"l_{v}_{i}_{tau}"
            )
            for v in ids
            for i in range(copies)
            for tau in range(T + 1)
        }

    ind_cache = {}

    def ind(v, i, tau):  # op[v,i,tau] as an arithmetic atom
        if not compact_encoding:
            return Terms.arith_eq_atom(t[(v, i)], Terms.integer(tau))
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

    def physical_group_width_term(active_lane_count):
        width_term = Terms.integer(encoded_group_widths[-1])
        for width in reversed(encoded_group_widths[:-1]):
            width_term = Terms.ite(
                Terms.arith_leq_atom(
                    active_lane_count, Terms.integer(width)
                ),
                Terms.integer(width),
                width_term,
            )
        return Terms.ite(
            Terms.arith_eq_atom(active_lane_count, zero),
            zero,
            width_term,
        )

    # ---- Figure 4: modulo schedule ----------------------------------------
    for v in ids:
        if compact_encoding:
            # CONSISTENCY substitutes every copy with t[v,0] + i*I. Since I
            # is positive, the first copy supplies the strongest lower bound
            # and the last copy reduces the strongest completion bound to L.
            asserts.append(Terms.arith_geq_atom(t[(v, 0)], zero))
            asserts.append(
                Terms.arith_leq_atom(
                    t[(v, 0)],
                    Terms.integer(length - max(1, prob.lat[v])),
                )
            )
            continue
        for i in range(copies):
            asserts.append(Terms.arith_geq_atom(t[(v, i)], zero))
            # COMPLETION plus the grid domain t in [0, T). A zero-latency
            # operation still owns one issue point and cannot start at T.
            asserts.append(
                Terms.arith_leq_atom(
                    t[(v, i)], Terms.integer(T - max(1, prob.lat[v]))
                )
            )
            if i:  # CONSISTENCY
                asserts.append(
                    Terms.arith_eq_atom(
                        t[(v, i)],
                        Terms.add(t[(v, 0)], Terms.integer(i * ii)),
                    )
                )
    for e in prob.edges:  # DEPENDENCE
        d = prob.edge_latency(e)
        if compact_encoding:
            if e.distance < copies:
                asserts.append(
                    Terms.arith_geq_atom(
                        Terms.sub(
                            t[(e.dst, e.distance)], t[(e.src, 0)]
                        ),
                        Terms.integer(d),
                    )
                )
            continue
        for i in range(copies):
            j = i + e.distance
            if j >= copies:
                continue
            asserts.append(Terms.arith_geq_atom(
                Terms.sub(t[(e.dst, j)], t[(e.src, i)]), Terms.integer(d)))

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
    for e in liveness_edges:
        consumers.setdefault(e.src, []).append(e)
    carried = {e.src for e in liveness_edges if e.distance > 0}
    for v in ids:
        # INIT: only loop-carried results of the last copy live at time T.
        if compact_encoding:
            for i in range(copies):
                live[(v, i, T)] = (
                    Terms.true()
                    if i == copies - 1 and v in carried
                    else Terms.false()
                )
        else:
            want = Terms.true() if v in carried else Terms.false()
            asserts.append(Terms.iff(live[(v, copies - 1, T)], want))
            for i in range(copies - 1):
                asserts.append(Terms.ynot(live[(v, i, T)]))
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
                if compact_encoding:
                    # LIVEPROP/DEADPROP uniquely determine the preceding
                    # cell, so use their equivalent Boolean recurrence.
                    live[(v, i, tau - 1)] = Terms.ite(
                        lv, Terms.ynot(o), any_use
                    )
                    continue
                lp = live[(v, i, tau - 1)]
                # LIVEPROP-1 / LIVEPROP-2
                asserts.append(
                    Terms.implies(Terms.yand([lv, o]), Terms.ynot(lp))
                )
                asserts.append(
                    Terms.implies(Terms.yand([lv, Terms.ynot(o)]), lp)
                )
                # DEADPROP-1 / DEADPROP-2
                asserts.append(
                    Terms.implies(
                        Terms.yand([Terms.ynot(lv), any_use]), lp
                    )
                )
                asserts.append(
                    Terms.implies(
                        Terms.yand([Terms.ynot(lv), Terms.ynot(any_use)]),
                        Terms.ynot(lp),
                    )
                )
    spill_edges: dict[int, list] = {}
    if spill_smem_footprint:
        for edge in ws_data_edges:
            if prob.regs.get(edge.src, 0) > 0:
                spill_edges.setdefault(edge.src, []).append(edge)

    for kind, capacity in (("smem", machine.smem_bytes
                            - machine.smem_fixed_overhead),
                           ("tmem", machine.tmem_cols
                            - machine.tmem_fixed_cols)):
        objects = [obj for obj in prob.memory_objects.values()
                   if obj.kind == kind]
        if not objects and (kind != "smem" or not spill_edges):
            continue
        for tau in range(T + 1):
            terms = []
            for obj in objects:
                for i in range(copies):
                    object_live = Terms.yor(
                        [live[(v, i, tau)] for v in obj.members]
                    )
                    terms.append(
                            Terms.ite(object_live, Terms.integer(obj.amount), zero)
                    )
            if kind == "smem":
                for producer, edges in spill_edges.items():
                    for i in range(copies):
                        pending_cross_uses = []
                        for edge in edges:
                            j = i + edge.distance
                            if j >= copies:
                                continue
                            pending_cross_uses.append(
                                Terms.yand(
                                    [
                                        Terms.ynot(
                                            same_lane_mask(
                                                producer, edge.dst
                                            )
                                        ),
                                        Terms.arith_geq_atom(
                                            t[(edge.dst, j)],
                                            Terms.integer(tau),
                                        ),
                                    ]
                                )
                            )
                        if not pending_cross_uses:
                            continue
                        terms.append(
                            Terms.ite(
                                Terms.yand(
                                    [
                                        Terms.arith_leq_atom(
                                            t[(producer, i)],
                                            Terms.integer(tau),
                                        ),
                                        Terms.yor(pending_cross_uses),
                                    ]
                                ),
                                Terms.integer(
                                    prob.regs[producer] * 4
                                    + machine.barrier_pair_bytes
                                ),
                                zero,
                            )
                        )
            if terms:
                asserts.append(Terms.arith_leq_atom(
                    Terms.sum(terms), Terms.integer(capacity)))

    # ---- Figure 6: warp assignment ----------------------------------------
    W_VL = 0  # designated variable-latency warp
    for v in assignable_ids:
        row = [opw[(v, w)] for w in range(W)]
        asserts.append(Terms.yor(row))
        for a in range(W):
            for b in range(a + 1, W):
                asserts.append(Terms.ynot(Terms.yand([row[a], row[b]])))
        # VARIABLELATENCY (iff: only variable-latency ops sit on W_vl)
        is_vl = v in prob.variable_latency
        asserts.append(Terms.iff(opw[(v, W_VL)],
                                 Terms.true() if is_vl else Terms.false()))

    def used(w):
        return Terms.yor([opw[(v, w)] for v in assignable_ids])

    if exact_groups is not None:
        asserts.append(
            Terms.arith_eq_atom(
                Terms.sum(
                    [Terms.ite(used(w), one, zero) for w in range(W)]
                ),
                Terms.integer(exact_groups),
            )
        )

    physical_warps = []
    for w in range(W):
        active_lanes = []
        for lane in range(max_group_width):
            if compact_encoding:
                members = [
                    Terms.yand(
                        [opw[(v, w)], lane_member[(v, lane)]]
                    )
                    for v in assignable_ids
                ]
            else:
                members = [
                    oplane[(v, w, lane)] for v in assignable_ids
                ]
            active_lanes.append(Terms.yor(members))
        active_lane_count = Terms.sum(
            [Terms.ite(active, one, zero) for active in active_lanes]
        )
        group_width = physical_group_width_term(active_lane_count)
        physical_warps.append(group_width)
        for v in assignable_ids:
            if full_group_lane_masks:
                asserts.append(
                    Terms.implies(
                        opw[(v, w)],
                        Terms.arith_eq_atom(
                            group_width,
                            Terms.integer(required_width[v]),
                        ),
                    )
                )
            if compact_encoding:
                continue
            lane_terms = []
            for lane in range(max_group_width):
                member = oplane[(v, w, lane)]
                lane_terms.append(Terms.ite(member, one, zero))
                asserts.append(Terms.implies(member, opw[(v, w)]))
                if prefix_lane_masks and lane < required_width[v]:
                    asserts.append(Terms.implies(opw[(v, w)], member))
            asserts.append(
                Terms.arith_eq_atom(
                    Terms.sum(lane_terms),
                    Terms.ite(
                        opw[(v, w)],
                        Terms.integer(required_width[v]),
                        zero,
                    ),
                )
            )

    if compact_encoding:
        for v in assignable_ids:
            lane_terms = [
                Terms.ite(lane_member[(v, lane)], one, zero)
                for lane in range(max_group_width)
            ]
            asserts.append(
                Terms.arith_eq_atom(
                    Terms.sum(lane_terms), Terms.integer(required_width[v])
                )
            )
            if prefix_lane_masks:
                for lane in range(required_width[v]):
                    asserts.append(lane_member[(v, lane)])

    asserts.append(
        Terms.arith_leq_atom(
            Terms.sum(physical_warps),
            Terms.integer(scheduled_warp_budget),
        )
    )

    # Group labels above W_vl are interchangeable.  Use the first member in
    # node-id order to give every partition its unique restricted-growth
    # labeling: the first regular group is 1, and the first member of group w
    # follows a member of group w-1.  This is solution-preserving because all
    # group-local variables and constraints are invariant under a common
    # permutation of nonzero group labels.
    if symmetry_breaking:
        regular_ids = sorted(
            v for v in assignable_ids if v not in prob.variable_latency
        )
        for w in range(2, W):
            previous_group_seen = Terms.false()
            for v in regular_ids:
                asserts.append(
                    Terms.implies(opw[(v, w)], previous_group_seen)
                )
                previous_group_seen = Terms.yor(
                    [previous_group_seen, opw[(v, w - 1)]]
                )
    # Physical lane names are likewise interchangeable within a group. Sort
    # their operation-membership columns lexicographically; this preserves
    # all overlap/equality relations while choosing one lane permutation.
    if symmetry_breaking and not prefix_lane_masks:
        canonical_ids = sorted(assignable_ids)
        active_lanes = max_group_width
        for w in range(W):
            for lane in range(active_lanes - 1):
                lexicographically_geq = Terms.true()
                for v in reversed(canonical_ids):
                    if compact_encoding:
                        left = Terms.yand(
                            [opw[(v, w)], lane_member[(v, lane)]]
                        )
                        right = Terms.yand(
                            [opw[(v, w)], lane_member[(v, lane + 1)]]
                        )
                    else:
                        left = oplane[(v, w, lane)]
                        right = oplane[(v, w, lane + 1)]
                    lexicographically_geq = Terms.yor(
                        [
                            Terms.yand([left, Terms.ynot(right)]),
                            Terms.yand(
                                [
                                    Terms.iff(left, right),
                                    lexicographically_geq,
                                ]
                            ),
                        ]
                    )
                asserts.append(lexicographically_geq)

    # Code-generator realizability: loop-carried values that live in
    # registers (the m_i / l_i online-softmax recurrences) cannot cross warp
    # groups — the emitter routes cross-group data through SMEM/TMEM
    # channels, which register iter_args don't get.  Buffer-mediated carried
    # values (TC accumulators, TMA tiles) are exempt.
    for e in ws_data_edges:
        if (reg_carried_same_lane and e.distance >= 1 and e.src != e.dst
                and prob.regs.get(e.src, 0) > 0
                and prob.nodes[e.src].pipeline in ("CUDA", "SFU", "NONE")):
            asserts.append(same_lane_mask(e.src, e.dst))
        if prefix_lane_masks and prob.regs.get(e.src, 0) > 0:
            # sched2tlx channels communicate between tasks. Within one task,
            # a register edge must use the same physical lanes.
            asserts.append(
                Terms.implies(
                    same_group(e.src, e.dst),
                    same_lane_mask(e.src, e.dst),
                )
            )

    # Optional structural probes: force listed node sets onto one warp, or
    # pairs onto different warps.
    for group in (colocate or []):
        for a, b in zip(group, group[1:]):
            asserts.append(same_group(a, b))
    for a, b in (separate or []):
        asserts.append(Terms.ynot(same_group(a, b)))

    # REGISTERLIMIT. Assert the bound independently for every physical lane
    # selected by each operation, rather than averaging narrow values over
    # the enclosing group width.
    reg_users = [v for v in assignable_ids if prob.regs[v] > 0]
    if reg_users:
        for w in range(W):
            for tau in range(T + 1):
                for lane in range(max_group_width):
                    terms = [
                        Terms.ite(
                            Terms.yand(
                                [
                                    live[(v, i, tau)],
                                    opw[(v, w)],
                                    (
                                        lane_member[(v, lane)]
                                        if compact_encoding
                                        else oplane[(v, w, lane)]
                                    ),
                                ]
                            ),
                            Terms.integer(
                                (prob.regs[v] + required_width[v] - 1)
                                // required_width[v]
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

    # Figure 5 also bounds the global register-file memory kind.
    for tau in range(T + 1):
        terms = [
            Terms.ite(
                live[(v, i, tau)],
                Terms.integer(
                    register_allocation_words(prob.regs[v], required_width[v])
                ),
                zero,
            )
            for v in reg_users
            for i in range(copies)
        ]
        if terms:
            asserts.append(
                Terms.arith_leq_atom(
                    Terms.sum(terms),
                    Terms.integer(
                        machine.regs_per_sm - machine.regs_fixed_overhead
                    ),
                )
            )

    # CROSS-WARPSPILLS (+ prohibition when cross-warp traffic is disabled —
    # the paper's third UNSAT ablation)
    for e in prob.edges:
        if not prob.edge_has_ws_semantics(e):
            continue
        if e.signal_only:
            continue
        sw = same_lane_mask(e.src, e.dst)
        if not allow_cross_warp:
            if (
                no_cross_warp_domain == "all-ws"
                or prob.regs.get(e.src, 0) > 0
            ):
                asserts.append(sw)
            continue
        d = prob.edge_latency(e)
        if compact_encoding:
            if e.distance < copies:
                asserts.append(
                    Terms.yor(
                        [
                            sw,
                            Terms.arith_geq_atom(
                                Terms.sub(
                                    t[(e.dst, e.distance)],
                                    t[(e.src, 0)],
                                ),
                                Terms.integer(d + prob.spill[e.src]),
                            ),
                        ]
                    )
                )
            continue
        for i in range(copies):
            j = i + e.distance
            if j >= copies:
                continue
            asserts.append(Terms.yor([sw, Terms.arith_geq_atom(
                Terms.sub(t[(e.dst, j)], t[(e.src, i)]),
                Terms.integer(d + prob.spill[e.src]))]))

    # CONCURRENCY: a blocking wait stalls in-order issue on the waiter's
    # warp.  Waits come from (a) consumers of asynchronous TC/TMA results
    # (static), and (b) consumers of cross-warp spills — the paper notes the
    # spill case gets the same treatment, since receiving register data
    # through shared memory also blocks.  (b) is conditional on the warp
    # assignment, so it is gated on "some in-edge is cross-warp".
    blocking_consumers = {v for (_, v) in prob.blocking}
    in_edges: dict[int, list] = {}
    for e in ws_data_edges:
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
                [Terms.ynot(same_lane_mask(e.src, v)) for e in preds]
            )
        for o in assignable_ids:
            if o == v:
                continue
            sw = overlaps_lane(o, v)
            cond = sw if gate is None else Terms.yand([gate, sw])
            # Paper Figure 6 excludes ISSUE points in
            # [t[v, i] - (cycles(o) - 1), t[v, i]]. A zero-cycle operation
            # therefore has an empty exclusion window.
            win = prob.lat[o]
            if win == 0:
                continue
            if compact_encoding:
                copy_pairs = [
                    (max(0, -offset), max(0, -offset) + offset)
                    for offset in range(-(copies - 1), copies)
                ]
            else:
                copy_pairs = [
                    (i, ip) for i in range(copies) for ip in range(copies)
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
    for v in assignable_ids:
        for w in range(W):
            if model.get_value(opw[(v, w)]):
                warp[v] = w
                break
    raw_lane_masks = {}
    for v, w in warp.items():
        raw_lane_masks[v] = tuple(
            lane
            for lane in range(max_group_width)
            if model.get_value(
                lane_member[(v, lane)]
                if compact_encoding
                else oplane[(v, w, lane)]
            )
        )
    lane_masks = {}
    for w in set(warp.values()):
        members = [v for v, group in warp.items() if group == w]
        active = sorted(
            {lane for v in members for lane in raw_lane_masks[v]}
        )
        remap = {lane: index for index, lane in enumerate(active)}
        for v in members:
            lane_masks[v] = tuple(
                sorted(remap[lane] for lane in raw_lane_masks[v])
            )
    group_widths = {
        w: physical_group_width(
            len(
                {
                    lane
                    for v, group in warp.items()
                    if group == w
                    for lane in lane_masks[v]
                }
            ),
            encoded_group_widths,
        )
        for w in set(warp.values())
    }
    scheduled_warps = sum(group_widths.values())

    def peak_storage(kind: str) -> int:
        objects = [
            storage
            for storage in prob.memory_objects.values()
            if storage.kind == kind
        ]
        return max(
            (
                sum(
                    storage.amount
                    for storage in objects
                    for i in range(copies)
                    if any(
                        model.get_value(live[(member, i, tau)])
                        for member in storage.members
                    )
                )
                for tau in range(T + 1)
            ),
            default=0,
        )

    tmem_objects = sum(
        storage.kind == "tmem" for storage in prob.memory_objects.values()
    )
    peak_tmem_cols = peak_storage("tmem")
    sol = JointSolution(ii=ii, length=length, copies=copies, horizon=T,
                        cycles=cycles, warp=warp, group_widths=group_widths,
                        lane_masks=lane_masks,
                        stats={**prob.normalization_stats(),
                               "num_asserts": len(asserts), "T": T,
                               "num_groups": len(group_widths),
                               "scheduled_warps": scheduled_warps,
                               "fixed_warps": machine.warp_fixed_overhead,
                               "num_warps": scheduled_warps
                                   + machine.warp_fixed_overhead,
                               "physical_warp_budget": physical_warp_budget,
                               "max_groups": W,
                               "regs_per_warp": machine.regs_per_warp,
                               "regs_per_sm": machine.regs_per_sm,
                               "regs_fixed_overhead":
                                   machine.regs_fixed_overhead,
                               "smem_capacity": machine.smem_bytes,
                               "smem_fixed_overhead":
                                   machine.smem_fixed_overhead,
                               "barrier_pair_bytes":
                                   machine.barrier_pair_bytes,
                               "spill_smem_footprint":
                                   spill_smem_footprint,
                               "tmem_capacity": machine.tmem_cols,
                               "tmem_fixed_cols": machine.tmem_fixed_cols,
                               "tmem_object_count": tmem_objects,
                               "peak_tmem_cols": peak_tmem_cols,
                               "reg_carried_same_lane":
                                   reg_carried_same_lane,
                               "no_cross_warp_domain":
                                   no_cross_warp_domain,
                               "emitter_infra_nodes":
                                   len(prob.emitter_infra),
                               "compact_encoding": compact_encoding,
                               "time_variables":
                                   len(ids)
                                   if compact_encoding
                                   else len(ids) * copies,
                               "live_variables":
                                   0
                                   if compact_encoding
                                   else len(ids) * copies * (T + 1),
                               "lane_variables":
                                   len(assignable_ids) * max_group_width
                                   if compact_encoding
                                   else len(assignable_ids)
                                   * W
                                   * max_group_width,
                               "lane_mask_model":
                                   ("emitter_full_group"
                                    if full_group_lane_masks
                                    else "emitter_prefix"
                                    if prefix_lane_masks
                                    else "paper_physical_masks")})
    ctx.dispose()
    cfg.dispose()
    return sol, "sat"
