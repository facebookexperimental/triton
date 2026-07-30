"""Optimal modulo scheduling as an ILP (paper sec 4 / Algorithm 1 line 5).

Classical Stoutchinin-style structure: each op's cycle decomposes as
M(v) = II*stage(v) + residue(v), with the residue one-hot encoded.  The
modulo resource constraints then live purely on the residue variables (no
time horizon), which keeps the model small and SCIP-friendly even when a
functional unit is fully saturated at MinII.

For a fixed initiation interval I the ILP decides feasibility and, when
feasible, returns the schedule minimizing the schedule length
L = max_v (M(v) + cycles(v)) so Algorithm 1 starts its L-search from LEN(M).

Constraints:
  * assignment      — every v gets exactly one residue and one stage;
  * dependence      — for (u,v,d,delta): M(v) >= M(u) + d - delta*I;
  * modulo resource — for each functional unit f and residue r in [0,I):
                      sum RRT[v][f,c] * rho[v,(r-c) mod I] <= cap(f).
"""

from dataclasses import dataclass

from pyscipopt import Model, quicksum

from .ddg import Problem
from .errors import SolverUnknownError
from .greedy_ims import greedy_modulo


@dataclass
class ModuloSchedule:
    ii: int
    cycles: dict[int, int]  # M(v)
    length: int  # LEN(M) = max(M(v) + cycles(v))
    optimal: bool


def _add_warm_start(model, rho, stage, length, prob, ii, warm):
    solution = model.createSol()
    for v in prob.nodes:
        residue = warm[v] % ii
        for r in range(ii):
            model.setSolVal(solution, rho[v, r], float(r == residue))
        model.setSolVal(solution, stage[v], warm[v] // ii)
    warm_length = max(
        warm[v] + max(1, prob.lat[v]) for v in prob.nodes
    )
    model.setSolVal(solution, length, warm_length)
    model.addSol(solution)


def solve_modulo(prob: Problem, ii: int, max_stages: int | None = None,
                 max_seconds: float = 300.0) -> ModuloSchedule | None:
    """OPTIMAL-MODULO-SCHEDULE(G, I): min-L schedule at fixed I, or None."""
    ids = list(prob.nodes)
    warm = greedy_modulo(prob, ii)
    m = Model("optimal_modulo_schedule")
    m.hideOutput()
    m.setParam("limits/time", max_seconds)
    rho = {
        (v, r): m.addVar(f"rho_{v}_{r}", vtype="B")
        for v in ids
        for r in range(ii)
    }
    stage = {
        v: m.addVar(f"stage_{v}", vtype="I", lb=0, ub=max_stages)
        if max_stages is not None
        else m.addVar(f"stage_{v}", vtype="I", lb=0)
        for v in ids
    }
    length = m.addVar("length", vtype="I", lb=0)

    def m_of(v):
        return ii * stage[v] + quicksum(
            r * rho[v, r] for r in range(ii)
        )

    for v in ids:
        m.addCons(quicksum(rho[v, r] for r in range(ii)) == 1)
        # M(v) belongs to [0, L), including when cycles(v) is zero.
        m.addCons(length >= m_of(v) + max(1, prob.lat[v]))

    for e in prob.edges:
        d = prob.edge_latency(e)
        m.addCons(m_of(e.dst) >= m_of(e.src) + d - e.distance * ii)

    for p, cap in prob.machine.capacities.items():
        reservations = [
            (v, cycle, amount)
            for v in ids
            for functional_unit, cycle, amount in prob.reservations(v)
            if functional_unit == p
        ]
        if not reservations:
            continue
        for r in range(ii):
            terms = [
                amount * rho[v, (r - cycle) % ii]
                for v, cycle, amount in reservations
            ]
            if terms:
                m.addCons(quicksum(terms) <= cap)

    m.setObjective(length, "minimize")
    if warm:
        _add_warm_start(m, rho, stage, length, prob, ii, warm)

    m.optimize()
    status = m.getStatus()
    if status == "infeasible":
        return None
    if status != "optimal" or m.getNSols() == 0:
        raise SolverUnknownError(f"SCIP ended with {status} at II={ii}")
    cycles = {}
    for v in ids:
        res = next(r for r in range(ii) if m.getVal(rho[v, r]) >= 0.5)
        cycles[v] = ii * round(m.getVal(stage[v])) + res
    length_val = max(cycles[v] + max(1, prob.lat[v]) for v in ids)
    return ModuloSchedule(ii=ii, cycles=cycles, length=length_val,
                          optimal=True)
