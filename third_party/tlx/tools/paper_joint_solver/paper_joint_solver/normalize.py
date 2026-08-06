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
# The sub-tiled backward fixture already needs 128x at the paper's U=300.
# Keep one more doubling for measurement/fixture drift while retaining a
# finite, fail-closed bound on normalization work.
MAX_U_MULTIPLIER = 256


class NormalizationResolutionError(RuntimeError):
    """The bounded normalization budget cannot preserve positive costs."""

    def __init__(
        self,
        *,
        initial_u: int,
        max_u: int,
        zero_collapses: int,
        positive_costs: int,
        collapsed_cost_values: tuple[int, ...],
    ) -> None:
        self.initial_u = initial_u
        self.max_u = max_u
        self.zero_collapses = zero_collapses
        self.positive_costs = positive_costs
        self.collapsed_cost_values = collapsed_cost_values
        multiplier = max_u // initial_u
        super().__init__(
            f"cost normalization resolution exhausted at U={max_u} "
            f"({multiplier}x initial U={initial_u}): "
            f"{zero_collapses}/{positive_costs} positive entries normalized "
            f"to zero; collapsed raw costs={list(collapsed_cost_values)}"
        )


@dataclass
class NormalizationResult:
    scaled: list[int]  # C', same order as the input costs
    objective: int  # optimal F
    solve_time_s: float
    optimal: bool
    u_eff: int = DEFAULT_U
    zero_collapses: int = 0
    retries: int = 0


def normalize_costs(
    costs: list[int],
    u: int = DEFAULT_U,
    time_limit_s: float | None = None,
) -> NormalizationResult:
    """`costs` is the per-instruction cost list C (duplicates included — the
    paper's sum bound 1 <= sum(C') <= U counts every instruction, which is
    what pins its normalized world at the coarse Fig-1 granularity)."""
    if any(c < 0 for c in costs):
        raise ValueError(f"negative cost in {costs}")
    if isinstance(u, bool) or not isinstance(u, int) or u < 1:
        raise ValueError("normalization budget U must be a positive integer")
    n = len(costs)
    if n == 0:
        return NormalizationResult([], 0, 0.0, True, u_eff=u)

    model = Model("cost_normalization")
    model.hideOutput()
    if time_limit_s is not None:
        model.setParam("limits/time", time_limit_s)

    cp = [model.addVar(f"cp_{i}", vtype="I", lb=0, ub=u) for i in range(n)]
    fmax = max(a * u for a in costs) if costs else 0
    f = model.addVar("F", vtype="I", lb=0, ub=fmax)

    for i in range(n):
        for j in range(i + 1, n):
            expr = costs[i] * cp[j] - costs[j] * cp[i]
            model.addCons(expr <= f)
            model.addCons(-f <= expr)
    weighted = quicksum(cp)
    model.addCons(weighted >= 1)
    model.addCons(weighted <= u)
    model.setObjective(f, "minimize")

    model.optimize()
    status = model.getStatus()
    if status != "optimal":
        raise RuntimeError(f"cost normalization solve failed: {status}")
    objective = round(model.getVal(f))
    scaled = [round(model.getVal(value)) for value in cp]

    # SCIP may return any solution from a degenerate F-optimal face.  Before
    # declaring U too coarse, prove that no equally F-optimal non-collapsing
    # solution exists.  Minimize sum(C') only within that primary-optimal face
    # so equivalent ratio scales do not inflate downstream schedule horizons.
    # This preserves the paper's objective and makes U_eff independent of
    # solver tie selection.
    positive_indices = [index for index, raw in enumerate(costs) if raw > 0]
    if positive_indices:
        model.freeTransform()
        model.addCons(f == objective)
        for index in positive_indices:
            model.addCons(cp[index] >= 1)
        model.setObjective(weighted, "minimize")
        model.optimize()
        no_collapse_status = model.getStatus()
        if no_collapse_status == "optimal":
            scaled = [round(model.getVal(value)) for value in cp]
        elif no_collapse_status != "infeasible":
            raise RuntimeError(
                "cost normalization no-collapse check failed: "
                f"{no_collapse_status}"
            )
    return NormalizationResult(
        scaled=scaled,
        objective=objective,
        solve_time_s=model.getSolvingTime(),
        optimal=True,
        u_eff=u,
        zero_collapses=sum(
            raw > 0 and normalized == 0
            for raw, normalized in zip(costs, scaled)
        ),
    )
