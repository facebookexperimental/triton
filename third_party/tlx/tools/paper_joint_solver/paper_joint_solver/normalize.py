"""Cost normalization (paper sec 5.2).

Replace a list of cycle counts C with smaller integers C' whose pairwise
ratios approximate the original ones, so the downstream modulo-scheduling
ILP and joint SMT problems stay tractable.  Formulated as an ILP solved
with SCIP:

    minimize F
    s.t.  -F <= C[i]*C'[j] - C[j]*C'[i] <= F   for all i < j
          1 <= sum(C') <= U
          C'[i] >= 0, integer

U trades cost resolution against solve time; the paper picks U = 300 and
reports SCIP finds global minima in under 500 ms.
"""

from dataclasses import dataclass

from pyscipopt import Model, quicksum

DEFAULT_U = 300
DEFAULT_CLUSTER_TOL = 0.10


def cluster_costs(costs: list[int], tol: float = DEFAULT_CLUSTER_TOL) -> dict[int, int]:
    """Merge costs whose ratio is within `tol` to one representative.

    Cycle counts are microbenchmark estimates; treating 532 and 559 as
    distinct exact integers plants coprime traps that pin the normalization
    ILP's F at uselessly coarse resolutions.  Greedy ascending clustering:
    a value joins the current cluster while it stays within `tol` of the
    cluster's first member; the representative is the cluster mean.
    """
    mapping: dict[int, int] = {}
    cluster: list[int] = []
    for c in sorted(set(costs)):
        if cluster and c <= cluster[0] * (1 + tol):
            cluster.append(c)
        else:
            if cluster:
                rep = round(sum(cluster) / len(cluster))
                mapping.update({v: rep for v in cluster})
            cluster = [c]
    if cluster:
        rep = round(sum(cluster) / len(cluster))
        mapping.update({v: rep for v in cluster})
    return mapping


@dataclass
class NormalizationResult:
    scaled: list[int]  # C', same order as the input costs
    objective: int  # optimal F
    solve_time_s: float
    optimal: bool


def normalize_costs(costs: list[int], u: int = DEFAULT_U,
                    time_limit_s: float | None = None) -> NormalizationResult:
    """`costs` is the per-instruction cost list C (duplicates included — the
    paper's sum bound 1 <= sum(C') <= U counts every instruction, which is
    what pins its normalized world at the coarse Fig-1 granularity)."""
    if any(c < 0 for c in costs):
        raise ValueError(f"negative cost in {costs}")
    # One ILP variable per distinct value (duplicates must not diverge); the
    # sum constraints weight each value by its multiplicity.
    distinct = sorted(set(costs))
    mult = {c: costs.count(c) for c in distinct}
    n = len(distinct)
    if n == 0:
        return NormalizationResult([], 0, 0.0, True)

    model = Model("cost_normalization")
    model.hideOutput()
    if time_limit_s is not None:
        model.setParam("limits/time", time_limit_s)

    cp = [model.addVar(f"cp_{i}", vtype="I", lb=0, ub=u) for i in range(n)]
    fmax = max(a * u for a in distinct) if distinct else 0
    f = model.addVar("F", vtype="I", lb=0, ub=fmax)

    for i in range(n):
        for j in range(i + 1, n):
            expr = distinct[i] * cp[j] - distinct[j] * cp[i]
            model.addCons(expr <= f)
            model.addCons(-f <= expr)
    weighted = quicksum(mult[distinct[i]] * cp[i] for i in range(n))
    model.addCons(weighted >= 1)
    model.addCons(weighted <= u)
    model.setObjective(f, "minimize")

    model.optimize()
    status = model.getStatus()
    if status not in ("optimal", "timelimit"):
        raise RuntimeError(f"cost normalization solve failed: {status}")
    f_star = round(model.getVal(f))
    t1 = model.getSolvingTime()

    # min F is heavily tied: the all-ones collapse and a full-resolution
    # scaling can share the same F.  Among min-F solutions, maximize sum(C')
    # — larger C' means smaller relative ratio error at equal F.  Fresh model
    # with F fixed (reusing the phase-1 model via freeTransform silently kept
    # the old objective sense in pyscipopt).
    m2 = Model("cost_normalization_p2")
    m2.hideOutput()
    if time_limit_s is not None:
        m2.setParam("limits/time", time_limit_s)
    cp2 = [m2.addVar(f"cp_{i}", vtype="I", lb=0, ub=u) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            expr = distinct[i] * cp2[j] - distinct[j] * cp2[i]
            m2.addCons(expr <= f_star)
            m2.addCons(-f_star <= expr)
    weighted2 = quicksum(mult[distinct[i]] * cp2[i] for i in range(n))
    m2.addCons(weighted2 >= 1)
    m2.addCons(weighted2 <= u)
    m2.setObjective(weighted2, "maximize")
    m2.optimize()
    status = m2.getStatus()
    if status not in ("optimal", "timelimit"):
        raise RuntimeError(f"cost normalization phase 2 failed: {status}")

    value = {c: round(m2.getVal(v)) for c, v in zip(distinct, cp2)}
    return NormalizationResult(
        scaled=[value[c] for c in costs],
        objective=f_star,
        solve_time_s=t1 + m2.getSolvingTime(),
        optimal=status == "optimal",
    )
