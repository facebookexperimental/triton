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


@dataclass
class NormalizationResult:
    scaled: list[int]  # C', same order as the input costs
    objective: int  # optimal F
    solve_time_s: float
    optimal: bool


def normalize_costs(
    costs: list[int],
    u: int = DEFAULT_U,
) -> NormalizationResult:
    """`costs` is the per-instruction cost list C, assembled as a positional
    multiset: every enumerated entry joins C, zeros included, and duplicates
    are retained so the sum bound 1 <= sum(C') <= U counts each occurrence."""
    if any(c < 0 for c in costs):
        raise ValueError(f"negative cost in {costs}")
    if isinstance(u, bool) or not isinstance(u, int) or u < 1:
        raise ValueError("normalization budget U must be a positive integer")
    n = len(costs)

    model = Model("cost_normalization")
    model.hideOutput()

    cp = [model.addVar(f"cp_{i}", vtype="I", lb=0, ub=u) for i in range(n)]
    fmax = max(a * u for a in costs) if costs else 0
    f = model.addVar("F", vtype="I", lb=0, ub=fmax)

    for i in range(n):
        for j in range(i + 1, n):
            expr = costs[i] * cp[j] - costs[j] * cp[i]
            model.addCons(expr <= f)
            model.addCons(-f <= expr)
    cp_sum = quicksum(cp)
    model.addCons(cp_sum >= 1)
    model.addCons(cp_sum <= u)
    model.setObjective(f, "minimize")

    model.optimize()
    status = model.getStatus()
    if status != "optimal":
        raise RuntimeError(f"cost normalization solve failed: {status}")
    objective = round(model.getVal(f))
    scaled = [round(model.getVal(value)) for value in cp]
    return NormalizationResult(
        scaled=scaled,
        objective=objective,
        solve_time_s=model.getSolvingTime(),
        optimal=True,
    )
