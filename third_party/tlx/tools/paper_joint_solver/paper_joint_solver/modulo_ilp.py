"""Optimal modulo scheduling as an ILP (paper sec 4 / Algorithm 1 line 5).

Classical Stoutchinin-style structure: each op's cycle decomposes as
M(v) = II*stage(v) + residue(v), with the residue one-hot encoded.  The
modulo resource constraints then live purely on the residue variables (no
time horizon), which keeps the model small and CBC-friendly even when a
functional unit is fully saturated at MinII.

For a fixed initiation interval I the ILP decides feasibility and, when
feasible, returns the schedule minimizing the schedule length
L = max_v (M(v) + cycles(v)) so Algorithm 1 starts its L-search from LEN(M).

Constraints:
  * assignment      — every v gets exactly one residue and one stage;
  * dependence      — for (u,v,d,delta): M(v) >= M(u) + d - delta*I;
  * modulo resource — for each functional unit f and residue r in [0,I):
                      sum over v on f, over its occupancy rows c, of
                      rho[v,(r-c) mod I]  <=  cap(f).
"""

from dataclasses import dataclass

from mip import BINARY, INTEGER, Model, xsum

from .ddg import Problem
from .greedy_ims import greedy_modulo


@dataclass
class ModuloSchedule:
    ii: int
    cycles: dict[int, int]  # M(v)
    length: int  # LEN(M) = max(M(v) + cycles(v))
    optimal: bool


def solve_modulo(prob: Problem, ii: int, max_stages: int | None = None,
                 max_seconds: float = 300.0) -> ModuloSchedule | None:
    """OPTIMAL-MODULO-SCHEDULE(G, I): min-L schedule at fixed I, or None."""
    ids = list(prob.nodes)
    warm = greedy_modulo(prob, ii)
    # Critical path (intra-iteration longest chain) bounds any min-length
    # schedule: cap L at crit + 2*II.  A time-limited CBC incumbent beyond
    # that cannot be the min-L oracle's answer, and an inflated L poisons
    # Algorithm 1's L-window (the joint problem's size is (copies-1)*II + L).
    crit = 0
    dist = {v: 0 for v in ids}
    for _ in range(len(ids)):
        changed = False
        for e in prob.edges:
            if e.distance:
                continue
            d = dist[e.src] + prob.edge_lat[(e.src, e.dst, e.distance)]
            if d > dist[e.dst]:
                dist[e.dst] = d
                changed = True
        if not changed:
            break
    crit = max(dist[v] + prob.lat[v] for v in ids)
    length_ub = crit + 2 * ii
    if warm:
        length_ub = max(length_ub, max(warm[v] + prob.lat[v] for v in ids))
    if max_stages is None:
        max_stages = -(-length_ub // ii)
    m = Model(solver_name="CBC")
    m.verbose = 0
    rho = {(v, r): m.add_var(var_type=BINARY) for v in ids for r in range(ii)}
    stage = {v: m.add_var(var_type=INTEGER, lb=0, ub=max_stages)
             for v in ids}
    length = m.add_var(var_type=INTEGER, lb=0, ub=length_ub)

    def m_of(v):
        return ii * stage[v] + xsum(r * rho[v, r] for r in range(ii))

    for v in ids:
        m += xsum(rho[v, r] for r in range(ii)) == 1
        m += length >= m_of(v) + prob.lat[v]
        m += m_of(v) + prob.lat[v] <= length_ub

    for e in prob.edges:
        d = prob.edge_lat[(e.src, e.dst, e.distance)]
        m += m_of(e.dst) >= m_of(e.src) + d - e.distance * ii

    for p, cap in prob.machine.capacities.items():
        users = [v for v in ids if prob.nodes[v].pipeline == p]
        if not users:
            continue
        for r in range(ii):
            terms = [rho[v, (r - c) % ii] for v in users
                     for c in range(prob.occ[v])]
            if terms:
                m += xsum(terms) <= cap

    if warm:
        hint = [(rho[v, warm[v] % ii], 1.0) for v in ids]
        hint += [(stage[v], float(warm[v] // ii)) for v in ids]
        m.start = hint

    m.objective = length
    status = m.optimize(max_seconds=max_seconds)
    if not m.num_solutions:
        return None
    cycles = {}
    for v in ids:
        res = next(r for r in range(ii) if rho[v, r].x >= 0.5)
        cycles[v] = ii * round(stage[v].x) + res
    length_val = max(cycles[v] + prob.lat[v] for v in ids)
    # A time-limited CBC incumbent can be far from min-L, and Algorithm 1's
    # L-window (hence the joint problem's size) is seeded by it.  If the
    # greedy schedule is shorter, prefer it — both are valid modulo schedules.
    if warm:
        warm_len = max(warm[v] + prob.lat[v] for v in ids)
        if warm_len < length_val:
            cycles, length_val = dict(warm), warm_len
    return ModuloSchedule(ii=ii, cycles=cycles, length=length_val,
                          optimal=str(status) == "OptimizationStatus.OPTIMAL")
