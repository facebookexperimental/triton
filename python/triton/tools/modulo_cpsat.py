"""CP-SAT backend for the NV modulo scheduler (nvgpu-modulo-schedule).

Complete (solver-based) replacement for the heuristic II search: joint
schedule + buffer-depth feasibility, the successor of ExhaustiveScheduler's
branch-and-bound (see docs/SolverMigrationNotes.md, "Suggested sequencing"
step 2). Invoked as a subprocess by CPSATScheduler.cpp:

    python3 -m triton.tools.modulo_cpsat <problem.json> <solution.json>

The problem JSON is produced by the C++ side from the DDG; the solution JSON
carries (II, per-node cycles) back. The C++ side re-verifies the solution
against its own reservation table before trusting it, so this process is
not part of the correctness TCB.

Model (per candidate II, swept from min_ii upward — NO slack window; a
complete search has no reservation-fragmentation failure mode, which is
what deletes guard 2 of SolverMigrationNotes.md):
  - cycle[v] integer, stage[v] = cycle // II, phase[v] = cycle mod II
  - dependence:  cycle[dst] >= cycle[src] + latency - distance*II
  - def-before-use: stage[src] <= stage[dst] for distance-0 edges
    (mirrors ExhaustiveScheduler's dataflow check)
  - resources: per hardware pipeline, NoOverlap of the ops' modular
    occupancy [phase, phase+duration) on the circular [0, II) timeline
    (wrap-around split into an optional second segment). Durations are the
    warp-issue `selfLatency` slots, exactly what ModuloReservationTable
    reserves — NOT the engine `occupancy` that feeds ResMII; the two-level
    model is inherited from the C++ side and min_ii already encodes the
    occupancy floor.
  - SMEM: sum(size_bytes * depth) <= smem_budget with
    depth = stage(last consumer) - stage(alloc) + 1  (buffer depths are
    decided JOINTLY with the schedule, replacing schedule-then-reduce)
  - TMEM: cumulative(cols) over [alloc cycle, last consumer cycle]
    capacity tmem_col_limit. (ExhaustiveScheduler used greedy interval
    coloring — conservative; cumulative is the exact constraint.)
  - objective = ExhaustiveScheduler's score, scaled to integers:
    minimize 10240000*maxStage - 102400*sum(depth) + 1024*regPressure
             + smemTotal

Cost normalization (Twill sec. 5.2): with --normalize U (or "normalize_u" in
the problem JSON, default 300), the distinct duration/latency values are
mapped to small integers by a side CP-SAT ILP minimizing pairwise ratio
distortion, and the normalized model's solution seeds the real-cycle model
as a warm-start hint. Unlike time-indexed ILP/SMT formulations (where
normalization is a tractability REQUIREMENT — solve cost is exponential in
the sum of edge delays), CP-SAT's interval encoding has no time-indexed
variables, so normalization here only accelerates search; the final solve
is always at real cycle granularity and exact.
"""

import json
import sys

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    print(json.dumps({"status": "error", "message": "ortools not installed (pip install ortools)"}))
    sys.exit(2)


def _critical_path(nodes, edges):
    """Longest path over distance-0 edges using edge latencies."""
    n = len(nodes)
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for e in edges:
        if e["distance"] == 0:
            adj[e["src"]].append((e["dst"], e["latency"]))
            indeg[e["dst"]] += 1
    dist = [0] * n
    ready = [i for i in range(n) if indeg[i] == 0]
    order = []
    while ready:
        cur = ready.pop()
        order.append(cur)
        for dst, lat in adj[cur]:
            dist[dst] = max(dist[dst], dist[cur] + lat)
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
    if len(order) != n:
        return None  # cycle in distance-0 edges: malformed DDG
    return max(dist) if dist else 0


def normalize_values(values, u):
    """Twill-style cost normalization: map the distinct positive `values` to
    small integers C' minimizing pairwise ratio distortion, sum(C') <= u.
    Returns {value: normalized} or None if the side-ILP fails."""
    vals = sorted(set(v for v in values if v > 0))
    if not vals or len(vals) > 40:
        return None
    model = cp_model.CpModel()
    vmax = max(vals)
    cprime = {v: model.NewIntVar(1, u, f"c_{v}") for v in vals}
    f = model.NewIntVar(0, vmax * u, "F")
    for i, vi in enumerate(vals):
        for vj in vals[i + 1:]:
            # |vi * c'[vj] - vj * c'[vi]| <= F
            model.Add(vi * cprime[vj] - vj * cprime[vi] <= f)
            model.Add(vj * cprime[vi] - vi * cprime[vj] <= f)
    model.Add(sum(cprime.values()) <= u)
    model.Minimize(f)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {v: solver.Value(cprime[v]) for v in vals}


def solve_at_ii(prob, ii, time_limit_s, hint=None):
    """Build and solve the joint schedule+depth model at a fixed II.
    Returns (status, cycles, objective) where cycles is {node_id: cycle}."""
    nodes = prob["nodes"]
    edges = prob["edges"]
    n = len(nodes)

    # An op whose reservation is longer than II can never be placed.
    for nd in nodes:
        if nd["pipeline"] != "NONE" and nd["duration"] > ii:
            return "infeasible", None, None

    cp = _critical_path(nodes, edges)
    if cp is None:
        return "error", None, None
    max_dur = max((nd["duration"] for nd in nodes), default=1)
    # Generous stage bound: the maxStage objective term keeps real solutions
    # shallow; the domain just has to admit the dependence-forced depth.
    max_stages = min(32, max(4, (cp + max_dur) // ii + 2))
    horizon = max_stages * ii

    model = cp_model.CpModel()
    cycle = [model.NewIntVar(0, horizon - 1, f"cyc_{i}") for i in range(n)]
    stage = [model.NewIntVar(0, max_stages - 1, f"stg_{i}") for i in range(n)]
    phase = [model.NewIntVar(0, ii - 1, f"phs_{i}") for i in range(n)]
    for i in range(n):
        model.AddDivisionEquality(stage[i], cycle[i], ii)
        model.AddModuloEquality(phase[i], cycle[i], ii)

    # Dependences.
    for e in edges:
        model.Add(cycle[e["dst"]] >= cycle[e["src"]] + e["latency"] - e["distance"] * ii)
        if e["distance"] == 0:
            model.Add(stage[e["src"]] <= stage[e["dst"]])

    # Modular resource exclusivity per pipeline (wrap-around split).
    by_pipe = {}
    for i, nd in enumerate(nodes):
        if nd["pipeline"] != "NONE":
            by_pipe.setdefault(nd["pipeline"], []).append(i)
    for pipe, members in by_pipe.items():
        if len(members) < 2 and all(nodes[i]["duration"] <= ii for i in members):
            continue
        segments = []
        for i in members:
            dur = max(nodes[i]["duration"], 1)
            end1 = model.NewIntVar(1, ii, f"end1_{pipe}_{i}")
            model.AddMinEquality(end1, [phase[i] + dur, ii])
            size1 = model.NewIntVar(1, dur, f"size1_{pipe}_{i}")
            model.Add(size1 == end1 - phase[i])
            segments.append(model.NewIntervalVar(phase[i], size1, end1, f"seg1_{pipe}_{i}"))
            if dur > 1:
                over = model.NewIntVar(0, dur - 1, f"over_{pipe}_{i}")
                model.AddMaxEquality(over, [phase[i] + dur - ii, 0])
                wraps = model.NewBoolVar(f"wrap_{pipe}_{i}")
                model.Add(over >= 1).OnlyEnforceIf(wraps)
                model.Add(over == 0).OnlyEnforceIf(wraps.Not())
                segments.append(model.NewOptionalIntervalVar(0, over, over, wraps, f"seg2_{pipe}_{i}"))
        model.AddNoOverlap(segments)

    # SMEM: joint buffer-depth budget. TMEM: cumulative column capacity.
    depth_vars = []
    smem_terms = []
    tmem_intervals = []
    tmem_demands = []
    for b in prob["buffers"]:
        alloc = b["alloc_node"]
        consumers = b["consumers"] or [alloc]
        if b["kind"] == "smem":
            last_stage = model.NewIntVar(0, max_stages - 1, f"lstg_b{alloc}")
            model.AddMaxEquality(last_stage, [stage[c] for c in consumers] + [stage[alloc]])
            depth = model.NewIntVar(1, max_stages, f"depth_b{alloc}")
            model.Add(depth == last_stage - stage[alloc] + 1)
            depth_vars.append(depth)
            smem_terms.append((b["size_bytes"], depth))
        elif b["kind"] == "tmem":
            last_cyc = model.NewIntVar(0, horizon - 1, f"lcyc_b{alloc}")
            model.AddMaxEquality(last_cyc, [cycle[c] for c in consumers] + [cycle[alloc]])
            size = model.NewIntVar(1, horizon, f"tsz_b{alloc}")
            model.Add(size == last_cyc - cycle[alloc] + 1)
            end = model.NewIntVar(1, horizon, f"tend_b{alloc}")
            model.Add(end == last_cyc + 1)
            tmem_intervals.append(model.NewIntervalVar(cycle[alloc], size, end, f"tmem_b{alloc}"))
            tmem_demands.append(b["tmem_cols"])
    smem_total = model.NewIntVar(0, prob["smem_budget"], "smem_total")
    model.Add(smem_total == sum(sz * d for sz, d in smem_terms))
    if tmem_intervals:
        model.AddCumulative(tmem_intervals, tmem_demands, prob["tmem_col_limit"])

    # Objective: ExhaustiveScheduler's composite score, integer-scaled by
    # 1024 so the SMEM-headroom byte term needs no division.
    #
    # Deliberately NO earliest-placement (sum-of-cycles) term: on ops that
    # serialize on one pipeline row it systematically prefers
    # short-duration-first (moving a 639-cycle reduce after a 105-cycle
    # trunc always lowers sum-of-cycles), which anti-correlates with
    # criticality — on case3 FA that order measured 8% slower at
    # (1,32,8192). Tie discipline among model-equivalent optima comes from
    # the incumbent hint instead (the C++ side passes Rau's schedule when
    # available; CP-SAT polishes from it and only departs for a strictly
    # better objective).
    max_stage = model.NewIntVar(0, max_stages - 1, "max_stage")
    model.AddMaxEquality(max_stage, stage)
    reg_pressure = sum(cycle[e["dst"]] - cycle[e["src"]] for e in edges if e["distance"] == 0)
    model.Minimize(10240000 * max_stage - 102400 * sum(depth_vars) + 1024 * reg_pressure + smem_total)

    if hint:
        for i in range(n):
            h = hint.get(i)
            if h is not None:
                model.AddHint(cycle[i], max(0, min(horizon - 1, h)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible", {i: solver.Value(cycle[i]) for i in range(n)}, solver.ObjectiveValue()
    if status == cp_model.INFEASIBLE:
        return "infeasible", None, None
    return "unknown", None, None


def _normalized_hint(prob, ii, time_limit_s):
    """Solve a cost-normalized copy of the problem and map its solution back
    to real cycles as a warm-start hint. Failures are harmless (no hint)."""
    u = prob.get("normalize_u", 300)
    if u <= 0:
        return None
    consts = [nd["duration"] for nd in prob["nodes"]]
    consts += [e["latency"] for e in prob["edges"]]
    mapping = normalize_values(consts, u)
    if not mapping:
        return None
    real_sum = sum(v for v in set(consts) if v > 0)
    norm_sum = sum(mapping.values())
    if real_sum <= 0 or norm_sum <= 0:
        return None
    scale = norm_sum / real_sum

    def norm(v):
        return mapping.get(v, max(1, round(v * scale)) if v > 0 else v)

    nprob = {
        "nodes": [dict(nd, duration=norm(nd["duration"])) for nd in prob["nodes"]],
        "edges": [dict(e, latency=norm(e["latency"])) for e in prob["edges"]],
        "buffers": prob["buffers"],
        "smem_budget": prob["smem_budget"],
        "tmem_col_limit": prob["tmem_col_limit"],
    }
    nii = max(1, round(ii * scale))
    for candidate in range(nii, nii + 3):
        status, cycles, _ = solve_at_ii(nprob, candidate, min(2.0, time_limit_s))
        if status == "feasible":
            return {i: round(c / scale) for i, c in cycles.items()}
    return None


def main():
    if len(sys.argv) != 3:
        print("usage: python -m triton.tools.modulo_cpsat <problem.json> <solution.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        prob = json.load(f)

    time_limit_s = float(prob.get("time_limit_s", 20.0))
    min_ii = prob["min_ii"]
    max_ii = prob["max_ii"]

    incumbent = prob.get("incumbent")  # Rau's schedule, when it succeeded
    stats = {"iis_tried": [], "unknown_iis": [], "hint_used": False}
    result = None
    for ii in range(min_ii, max_ii + 1):
        hint = None
        if incumbent and incumbent.get("ii") == ii:
            hint = {int(k): v for k, v in incumbent["cycles"].items()}
        elif ii == min_ii:
            hint = _normalized_hint(prob, ii, time_limit_s)
        if hint:
            stats["hint_used"] = True
        status, cycles, obj = solve_at_ii(prob, ii, time_limit_s, hint)
        stats["iis_tried"].append(ii)
        if status == "feasible":
            result = {
                "status": "ok", "ii": ii, "cycles": {str(k): v
                                                     for k, v in cycles.items()}, "objective": obj, "stats": stats
            }
            break
        if status == "unknown":
            # Timeout: completeness holds only within the time limit. Record
            # and keep sweeping — the serial upper bound is always feasible,
            # so the sweep terminates with a schedule.
            stats["unknown_iis"].append(ii)
        if status == "error":
            result = {"status": "error", "message": "malformed DDG (cycle in distance-0 edges)"}
            break

    if result is None:
        result = {"status": "error", "message": f"no feasible II in [{min_ii}, {max_ii}]", "stats": stats}
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
