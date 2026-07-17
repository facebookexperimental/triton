"""Joint-solver (OR-Tools CP-SAT) backend for the NV modulo scheduler.

Complete (solver-based) replacement for the heuristic II search: joint
schedule + buffer-depth feasibility, the successor of ExhaustiveScheduler's
branch-and-bound (see docs/SolverMigrationNotes.md, "Suggested sequencing"
step 2). Invoked as a subprocess by JointSolverScheduler.cpp:

    python3 -m triton.tools.modulo_joint_solver <problem.json> <solution.json>

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
import os
import sys

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    print(json.dumps({"status": "error", "message": "ortools not installed (pip install ortools)"}))
    sys.exit(2)


def _async_flight_blocking():
    """Pre-skew emitter model: same-WG sync work is priced as blocked for an
    async op's whole flight (v1 split-cost window) and excluded from it
    outright (v2 hard constraint). This over-prices merging by ~2000 cycle
    units on case4 (the hardware-best 6-WG layout scored 2930 vs the 8-WG
    pick's 892 while running 2.2x faster — 2026-07-11 calibration). The
    correctness hazard that first surfaced with blocking disabled (obj-252
    merged layout emitted WRONG dQ/dK) was an emitter bug — the intra-WG
    SMEM->MMA bridge wrote slot [0] while consumers read the WG ring slot —
    fixed 2026-07-11 by clamping bridge buffers to the protocol's depth-1
    (sched2tlx emitter, _derive_smem_bridge_channels).
    Default OFF since 2026-07-11: with skew-aware windows plus the
    recurrence re-ranker (canonical-shape candidates scored by combined =
    objective + 2*max(0, RecMII_aug - II)), the witness suite flipped —
    case4's auto-pick is the hardware-best 6-WG layout (268.6 TF measured
    end-to-end vs 132 under the legacy window, which demotes it to rank 2)
    and case3's canary holds. Set TRITON_MODULO_ASYNC_FLIGHT_BLOCKING=1 to
    restore the legacy pre-skew pricing for A/B."""
    return os.environ.get("TRITON_MODULO_ASYNC_FLIGHT_BLOCKING", "0") == "1"


def _recmii_rerank_enabled():
    """Recurrence re-ranking of the partition solve's K-best assignments.
    On by default; TRITON_MODULO_RECMII_RERANK=0 restores the raw
    single-solution objective for A/B."""
    return os.environ.get("TRITON_MODULO_RECMII_RERANK", "1") == "1"


def _intrawg_async_reader_pairs(nodes_by_id, edges, cluster_of):
    """(src_cluster, dst_cluster) pairs that must NOT share a warp group.

    LEGALITY (2026-07-16, case9b joint-partition wedge): an ASYNC producer's
    software reader (CUDA/SFU consumer of an MMA/TMA result) handshakes
    through a depth-1 mbarrier parity wait that EVERY warp of the reader's
    group executes. PTX defines mbarrier parity tests only for the current
    and the immediately preceding phase, so a straggler warp that falls a
    full generation behind its group aliases the parity and the kernel
    wedges (measured ~1/300 launches at 4096 after a small-shape bench).
    Single-warp groups execute waits warp-uniformly and are safe — every
    handwritten TLX Blackwell kernel keeps MMAs in 1-warp tasks — but the
    emitter's default task can launch wider than scheduled, so the reader
    must simply live in a different warp group than the async producer.
    MMA→MMA consumers are exempt: TC-only groups are sized single-warp.

    Covers ALL distances: the emitter derives the intra-WG handshake from
    operand references, so a loop-carried MMA→tmem_load edge (case3's
    acc-correction read) lands in the same wedge-prone protocol when the
    pair shares a group.
    """
    out = set()
    for e in edges:
        src = nodes_by_id.get(e["src"])
        dst = nodes_by_id.get(e["dst"])
        if src is None or dst is None:
            continue
        if src["pipeline"] not in ("TMA", "TC", "MFMA"):
            continue
        if dst["pipeline"] not in ("CUDA", "SFU"):
            continue
        ci = cluster_of.get(e["src"])
        cj = cluster_of.get(e["dst"])
        if ci is None or cj is None or ci == cj:
            continue
        out.add((ci, cj))
    return out


def _mem_resident_srcs(prob):
    """Nodes whose result is MEMORY-resident: they produce an SMEM/TMEM
    buffer, so consumers read the buffer, not a register copy. A long
    producer→consumer span costs no register lifetime, and pricing it in
    the reg-pressure term (1024/cycle) dwarfs the 102400 ring-depth reward
    — on case4 FA-bwd the v2 re-solve compressed the K-tile ring from the
    schedule's depth 3 to depth 2 for exactly this reason (0.92x of the
    heuristic draw on hardware). Excluding these sources confines the
    reg-pressure term to values that actually live in the register file.

    MEASURED VERDICT (2026-07-16, case4 B200 hot-L2 @N=16384): the ring
    depth is NOT the 8% — a hand-patched depth-3 variant of the compressed
    kernel is cycle-identical to depth-2 (268.6 TF both), and enabling
    this exclusion made placement WORSE (254.5 TF vs modulo's 292.3): with
    the compression pressure gone the solver drifts to longer spans that
    serialize the softmax WG's waits. The case4 gap lives in stream-order
    placement geometry the objective cannot see (same finding as the
    RecMII saga's layout-vs-draw residual). Kept as an A/B probe.
    Experimental, default OFF: TRITON_MODULO_MEM_RESIDENT_SPANS=1 enables.
    """
    if os.environ.get("TRITON_MODULO_MEM_RESIDENT_SPANS", "0") != "1":
        return set()
    out = set()
    for b in prob.get("buffers", []) or []:
        p = b.get("producer", b.get("alloc_node"))
        if p is not None:
            out.add(p)
    return out


def _has_full_graph(prob):
    """The re-ranker needs the full dependence graph: plain (non-cross)
    edges and buffers with ring counts — shipped by the C++ side since the
    2026-07-11 serialization extension."""
    return bool(prob.get("buffers")) and any("src_cluster" not in e for e in prob.get("edges", []))


_RECMII_HS = 30  # per-handshake overhead (wait/arrive pair), cycles


def _set_partitions(items):
    """All set-partitions of a small list (Bell number growth — callers cap
    the input size)."""
    if not items:
        yield []
        return
    head, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[head] + part[i]] + part[i + 1:]
        yield [[head]] + part


def _canonical_candidates(prob):
    """Domain-shape candidate assignments for the recurrence re-ranker.

    K-best enumeration of the local-term objective climbs a deep plateau of
    merged variants and never reaches pipeline-parallel layouts (case4: the
    hardware-best 6-WG shape sits ~750 objective units above the plateau,
    beyond any practical K). Generate the canonical warp-specialization
    shapes directly instead:
      - every TC cluster gets its own WG (MMA issue streams are cheap and
        isolating them keeps loop-carried MMA recurrences off the CUDA
        bodies);
      - buffer-producing TMA clusters keep their own WGs (their rings
        already decouple them);
      - the remaining heavy (min_warps >= 4, non-trivial duration) CUDA
        clusters are grouped by every set-partition (capped at 5 clusters);
      - light clusters (register-channel TMA loads, scalars) are tried both
        folded into the first heavy group's WG (killing their depth-1
        channel recurrence) and separate.
    """
    clusters = prob["clusters"]
    nodes = {nd["id"]: nd for nd in prob["nodes"]}
    buf_alloc_producers = set()
    preds = {}
    for e in prob["edges"]:
        if int(e["distance"]) == 0:
            preds.setdefault(e["dst"], []).append(e["src"])
    for b in prob.get("buffers", []):
        # buffer "producer" is the staging alloc; its value producer is the
        # dist-0 predecessor — walk one hop.
        for p in preds.get(b["producer"], [b["producer"]]):
            buf_alloc_producers.add(p)
        buf_alloc_producers.add(b["producer"])

    def cdur(c):
        return sum(max(int(nodes[n].get("duration", 0)), 0) for n in c["nodes"])

    tc, ring_tma, heavy, light = [], [], [], []
    for c in clusters:
        pipes = {nodes[n]["pipeline"] for n in c["nodes"]}
        if "TC" in pipes:
            tc.append(c)
        elif c["min_warps"] >= 4 and cdur(c) >= 32:
            # heavy CUDA compute — mergeable even if it stages a buffer
            # (splitting compute chains cuts register handoffs, which the
            # rt/xissue terms price correctly; grouping is what the local
            # objective cannot rank).
            heavy.append(c)
        elif pipes == {"TMA"} and any(n in buf_alloc_producers for n in c["nodes"]):
            ring_tma.append(c)
        else:
            light.append(c)
    if len(heavy) > 5:
        return []  # Bell growth — fall back to plain K-best
    out = []
    for part in _set_partitions(heavy):
        for fold_light in (True, False):
            amap, w = {}, 0
            for c in tc:
                amap[c["id"]] = w
                w += 1
            for c in ring_tma:
                amap[c["id"]] = w
                w += 1
            first_heavy_wg = None
            for grp in part:
                for c in grp:
                    amap[c["id"]] = w
                if first_heavy_wg is None:
                    first_heavy_wg = w
                w += 1
            for c in light:
                if fold_light and first_heavy_wg is not None:
                    amap[c["id"]] = first_heavy_wg
                else:
                    amap[c["id"]] = w
                    w += 1
            out.append(amap)
    # dedup
    seen, uniq = set(), []
    for amap in out:
        key = tuple(sorted(amap.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(amap)
    return uniq


def _recmii_augmented(prob, amap):
    """Predicted steady-state cycles/iteration of the emitted kernel for a
    cluster->WG assignment: max cycle ratio over the dependence graph
    augmented with (a) per-WG in-order body edges + a distance-1 wrap edge
    (a warp re-enters its body once per iteration), (b) buffer-completion
    back-edges at the emitted ring depths (intra-WG bridges are clamped to
    depth-1 by the emitter; cross-WG channels use the schedule's count),
    and (c) depth-1 back-edges for register-channel handoffs. Validated
    2026-07-11 on case4's six hardware-measured layouts (combined-score
    winner == hardware winner) and case3's five probes (v1 pick stays
    first; the softmax-merge trap the raw objective preferred is demoted).
    Returns predicted cycles, or None if the problem lacks the full graph."""
    if not _has_full_graph(prob):
        return None
    nodes = {nd["id"]: nd for nd in prob["nodes"]}
    wg_of = {}
    for c in prob["clusters"]:
        for nid in c["nodes"]:
            wg_of[nid] = amap[c["id"]]
    preds = {}
    succs0 = {}
    for e in prob["edges"]:
        if int(e["distance"]) == 0:
            preds.setdefault(e["dst"], []).append(e["src"])
            succs0.setdefault(e["src"], []).append(e["dst"])
    for _ in range(3):  # glue nodes follow their value producer
        for nid in nodes:
            if nid in wg_of:
                continue
            for p in preds.get(nid, []):
                if p in wg_of:
                    wg_of[nid] = wg_of[p]
                    break

    E = []
    dur = {nid: max(int(n.get("duration", 0)), 0) for nid, n in nodes.items()}
    for e in prob["edges"]:
        E.append((e["src"], e["dst"], int(e["latency"]), int(e["distance"])))

    # Topological tie-break for equal-cycle body nodes: a body edge opposing
    # a distance-0 dependence would create a zero-distance positive cycle
    # (unbounded ratio).
    indeg = {nid: 0 for nid in nodes}
    for e in prob["edges"]:
        if int(e["distance"]) == 0:
            indeg[e["dst"]] = indeg.get(e["dst"], 0) + 1
    topo = {}
    ready = sorted(nid for nid, d in indeg.items() if d == 0)
    order = 0
    while ready:
        cur = ready.pop(0)
        topo[cur] = order
        order += 1
        for s in succs0.get(cur, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    bodies = {}
    for nid, w in wg_of.items():
        if w is not None and nid in nodes:
            bodies.setdefault(w, []).append(nid)
    for w, nids in bodies.items():
        nids.sort(key=lambda i: (nodes[i]["cycle"], topo.get(i, 0)))
        for a, b in zip(nids, nids[1:]):
            E.append((a, b, dur[a], 0))
        E.append((nids[-1], nids[0], dur[nids[-1]], 1))

    covered = set()
    for b in prob["buffers"]:
        p = b["producer"]
        cnt = max(1, int(b.get("count", 1)))
        for cons in b.get("consumers", []):
            m = cons["node"]
            if m not in nodes or nodes[m].get("pipeline") != "TC":
                continue
            d = 1 if wg_of.get(p) == wg_of.get(m) else cnt
            E.append((m, p, int(nodes[m].get("latency", 0)) + _RECMII_HS, d))
            covered.add((p, m))

    for e in prob["edges"]:
        u, v = e["src"], e["dst"]
        if int(e["distance"]) != 0:
            continue
        wu, wv = wg_of.get(u), wg_of.get(v)
        if wu is None or wv is None or wu == wv:
            continue
        if (u, v) in covered or (v, u) in covered:
            continue
        E.append((v, u, _RECMII_HS, 1))

    # Max cycle ratio: binary search on lambda; a positive cycle under
    # weights (w - lambda*dist) means the ratio exceeds lambda.
    ids = list(nodes)
    idx = {v: i for i, v in enumerate(ids)}
    m = len(ids)

    def has_pos_cycle(lam):
        dist = [0.0] * m
        for _ in range(m):
            changed = False
            for (u, v, w, d) in E:
                iu, iv = idx.get(u), idx.get(v)
                if iu is None or iv is None:
                    continue
                nd = dist[iu] + w - lam * d
                if nd > dist[iv] + 1e-9:
                    dist[iv] = nd
                    changed = True
            if not changed:
                return False
        return True

    lo, hi = 0.0, 8.0 * max(1, prob["ii"])
    while hi - lo > 0.5:
        mid = (lo + hi) / 2
        if has_pos_cycle(mid):
            lo = mid
        else:
            hi = mid
    return lo


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

    # Dependences. Streaming producers (Twill §5.3: variable-latency ops
    # with no incoming deps, e.g. TMA input loads) run ahead of the pipeline
    # behind their ring buffer, so in steady state consumers do not wait
    # their latency — model their outgoing edges as latency 0. Ring depth
    # stays a solver decision: the objective already REWARDS depth
    # (-102400·Σdepth) against the SMEM budget, so removing the latency
    # pressure does not collapse the ring. C++-side verifySolution applies
    # the same effective-latency rule.
    streaming = ({nd["id"] for nd in nodes if nd.get("streaming")} if prob.get("streaming_vl") else set())
    for e in edges:
        lat = 0 if e["src"] in streaming else e["latency"]
        model.Add(cycle[e["dst"]] >= cycle[e["src"]] + lat - e["distance"] * ii)
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
    mem_srcs = _mem_resident_srcs(prob)
    reg_pressure = sum(cycle[e["dst"]] - cycle[e["src"]]
                       for e in edges
                       if e["distance"] == 0 and e["src"] not in mem_srcs)
    # Recurrence-chain criticality (SolverMigrationNotes, step 3): for each
    # loop-carried back edge (u -> v, distance > 0), cycle[u] - cycle[v] is
    # the scheduled span of the recurrence circuit's forward chain.
    # Compressing it pushes all slack onto the back edge, so when the
    # model's latencies UNDERESTIMATE the chain (FA's softmax measures
    # ~1745 cyc/iter against a modeled II of 1459) the realized iteration
    # time degrades as little as possible. Weighted far above regPressure:
    # this is exactly the term whose absence let the solver reorder case3's
    # rowsum after the P-tile store (regPressure strictly preferred it; 8%
    # slower on hardware). Self-edges (u == v) contribute a constant 0.
    rec_span = sum(cycle[e["src"]] - cycle[e["dst"]] for e in edges if e["distance"] > 0 and e["src"] != e["dst"])
    model.Minimize(10240000 * max_stage - 102400 * sum(depth_vars) + 8192 * rec_span + 1024 * reg_pressure + smem_total)

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


def _mod_overlap(a_start, a_dur, b_start, b_dur, ii):
    """Overlap (cycles) of two cyclic intervals on [0, ii)."""

    def segs(s, d):
        s %= ii
        d = min(d, ii)
        if s + d <= ii:
            return [(s, s + d)]
        return [(s, ii), (0, s + d - ii)]

    total = 0
    for s1, e1 in segs(a_start, a_dur):
        for s2, e2 in segs(b_start, b_dur):
            total += max(0, min(e1, e2) - max(s1, s2))
    return total


def solve_partition(prob):
    """Joint-formulation v1 (SolverMigrationNotes step 3): warp-group
    assignment as a constraint problem AGAINST the committed schedule
    (cycles held fixed — the Twill-style re-solve at the schedule's II).
    The partition-relevant costs the heuristic partitioner scores post-hoc
    become model terms:

      - split cost: two clusters sharing a WG serialize on the warp's issue
        stream, so their ops' scheduled modular-interval OVERLAP is exactly
        the parallelism the schedule assumed and the merge destroys
        (replaces computeMultiPipelineMakespan's list-scheduling proxy);
      - merge pressure: a register-compute edge cut across WGs pays the
        measured reg->SMEM->reg hand-off, but only the part that does not
        fit in the schedule's slack (refines accumulateCrossWGRoundTrip,
        which charges the full round-trip regardless of slack);
      - cross-WG barrier issue cost (measured mbarrier/named constants);
      - register budget with the calibrated footprint table + default-WG
        slack model; synthesized channel SMEM as a HARD capacity constraint.

    Schedule-side blind spot this v1 cannot fix (documented in
    SolverMigrationNotes): op ORDER inside a WG is frozen by the scheduler,
    which ran before any partition existed — e.g. case3's alpha hand-off
    urgency. That needs joint cycles+wg (v2).
    """
    ii = prob["ii"]
    clusters = prob["clusters"]
    nodes = {nd["id"]: nd for nd in prob["nodes"]}
    ncl = len(clusters)
    model = cp_model.CpModel()
    wg = [model.NewIntVar(0, ncl - 1, f"wg_{c['id']}") for c in clusters]
    cindex = {c["id"]: i for i, c in enumerate(clusters)}

    # Symmetry breaking: WG ids appear in first-use order.
    prefix_max = wg[0]
    model.Add(wg[0] == 0)
    for i in range(1, ncl):
        model.Add(wg[i] <= prefix_max + 1)
        nxt = model.NewIntVar(0, ncl - 1, f"pmax_{i}")
        model.AddMaxEquality(nxt, [prefix_max, wg[i]])
        prefix_max = nxt
    used_wgs = model.NewIntVar(1, ncl, "used_wgs")
    model.Add(used_wgs == prefix_max + 1)
    # Emitter-capability legality (versioned; see EmitterCaps in
    # ModuloSchedulePass.cpp): max_wgs bounds the WG count (caps v2
    # retired the outer-loop single-WG constraint).
    model.Add(used_wgs <= prob.get("max_wgs", ncl))

    same_cache = {}

    def same_wg(i, j):
        key = (min(i, j), max(i, j))
        if key not in same_cache:
            s = model.NewBoolVar(f"same_{key[0]}_{key[1]}")
            model.Add(wg[key[0]] == wg[key[1]]).OnlyEnforceIf(s)
            model.Add(wg[key[0]] != wg[key[1]]).OnlyEnforceIf(s.Not())
            same_cache[key] = s
        return same_cache[key]

    # LEGALITY: async results read by software (see _intrawg_async_reader_pairs).
    cluster_of_v1 = {}
    for c in clusters:
        for nid in c["nodes"]:
            cluster_of_v1[nid] = cindex[c["id"]]
    for ci, cj in _intrawg_async_reader_pairs(nodes, prob["edges"], cluster_of_v1):
        model.Add(same_wg(ci, cj) == 0)

    terms = []

    # Split cost: scheduled parallelism destroyed by merging two clusters
    # into one in-order warp. The occupied window is the op's ISSUE duration
    # for every class: the warp is busy only while issuing; async flights
    # (TMA/TC) overlap same-WG work because the emitter sinks the waits
    # below independent ops and stage-skews intra-WG async results. The
    # legacy pre-skew model additionally charged async-vs-sync pairs the
    # async op's full LATENCY window ("sync work placed during the flight
    # is blocked by the same-warp wait if merged") — restorable with
    # TRITON_MODULO_ASYNC_FLIGHT_BLOCKING=1.
    ASYNC = ("TMA", "TC", "MFMA")

    def window(nd, other):
        # Skew-aware window (2026-07-11 calibration): the emitter stage-skews
        # intra-WG async results (sinks waits below independent work — the
        # guard-1 retirement emitter capability), so a merged warp is occupied
        # only for the async op's ISSUE duration, same as async-vs-async. The
        # legacy full-latency window priced same-WG sync work as blocked for
        # the whole flight and inverted case4's ranking: the hardware-best
        # 6-WG layout (292 TF) scored 2930 vs the 8-WG pick's 892 (132 TF),
        # with the entire 2036-unit gap in this term.
        # TRITON_MODULO_ASYNC_FLIGHT_BLOCKING=1 restores the legacy window
        # (pre-skew emitter model) for A/B.
        if _async_flight_blocking() and nd["pipeline"] in ASYNC and other["pipeline"] not in ASYNC:
            return max(nd["duration"], nd["latency"])
        return nd["duration"]

    for i in range(ncl):
        for j in range(i + 1, ncl):
            ov = 0
            for u in clusters[i]["nodes"]:
                for v in clusters[j]["nodes"]:
                    nu, nv = nodes[u], nodes[v]
                    if nu["pipeline"] == nv["pipeline"]:
                        continue  # same row: already serialized chip-wide
                    ov += _mod_overlap(nu["cycle"], window(nu, nv), nv["cycle"], window(nv, nu), ii) * max(
                        nu["freq"], nv["freq"])
            if ov > 0:
                terms.append(2 * ov * same_wg(i, j))

    # Merge pressure + barrier issue on cross-WG edges; channel SMEM (hard).
    # The problem now ships EVERY dependence (the recurrence re-ranker needs
    # the full graph); only cross-cluster edges carry the partition metadata
    # (src_cluster/dst_cluster/rt/xissue) — plain edges are skipped here.
    chan_bytes_total = []
    seen_pairs = set()
    for e in prob["edges"]:
        ci = cindex.get(e.get("src_cluster"))
        cj = cindex.get(e.get("dst_cluster"))
        if ci is None or cj is None or ci == cj:
            continue
        if e["distance"] > 0 and e.get("rt", 0) > 0:
            # LEGALITY (same as the joint mode): register loop-carried
            # values live in iter_args with no cross-WG channel semantics.
            model.Add(same_wg(ci, cj) == 1)
            continue
        cross = same_wg(ci, cj).Not()
        slack = (nodes[e["dst"]]["cycle"] - nodes[e["src"]]["cycle"] - e["latency"] + e["distance"] * ii)
        if e.get("rt", 0) > 0:
            shortfall = max(0, e["rt"] - max(0, slack)) * e["freq"]
            if shortfall > 0:
                terms.append(2 * shortfall * cross)
        terms.append(2 * e["xissue"] * cross)
        pair = (e["src"], e["dst"])
        if e.get("chan_bytes", 0) > 0 and pair not in seen_pairs:
            seen_pairs.add(pair)
            cb = model.NewIntVar(0, e["chan_bytes"], f"cb_{pair[0]}_{pair[1]}")
            model.Add(cb == e["chan_bytes"]).OnlyEnforceIf(cross)
            model.Add(cb == 0).OnlyEnforceIf(cross.Not())
            chan_bytes_total.append(cb)
    model.Add(prob["committed_smem"] + sum(chan_bytes_total) <= prob["smem_budget"])

    # Cluster→WG assignment indicators, shared by the register-budget and
    # issue-stream capacity terms below.
    assign = {}
    for w in range(ncl):
        for i in range(ncl):
            a = model.NewBoolVar(f"a_{i}_{w}")
            model.Add(wg[i] == w).OnlyEnforceIf(a)
            model.Add(wg[i] != w).OnlyEnforceIf(a.Not())
            assign[(i, w)] = a

    # Issue-stream capacity: a warp group's per-iteration occupancy is the
    # SUM of its ops' issue/execute durations — ops scheduled at different
    # cycles still serialize on the same in-order instruction stream, so
    # windows that never OVERLAP still ADD. (The split cost above prices
    # only pairwise modular overlap; without this term an all-merged WG
    # whose duration sum approaches II scores as nearly free — case4's
    # obj-252 layout packs ~896 of II=1033 cycles into one 4-warp WG and
    # runs at ~0.5x of the scheduled pipeline.) Excess over II inflates the
    # achieved II one-for-one; charge it at the cycle-term weight (x2).
    dur_c = [int(sum(max(nodes[u]["duration"], 0) * nodes[u].get("freq", 1) for u in c["nodes"])) for c in clusters]
    total_dur = max(1, sum(dur_c))
    if os.environ.get("TRITON_MODULO_CAPACITY_TERM", "1") == "1":
        for w in range(ncl):
            occ_w = model.NewIntVar(0, total_dur, f"occ_{w}")
            model.Add(occ_w == sum(dur_c[i] * assign[(i, w)] for i in range(ncl)))
            exc_w = model.NewIntVar(0, total_dur, f"exc_{w}")
            model.AddMaxEquality(exc_w, [occ_w - ii, 0])
            terms.append(2 * exc_w)

    # Register budget: per-WG warp counts from the assigned clusters'
    # minWarps, footprints from the calibrated table, default-WG slack model.
    fp_table = prob["warp_footprint"]  # indexed by warp count 0..8
    total_regs_terms = []
    for w in range(ncl):
        warps_w = model.NewIntVar(0, 8, f"warps_{w}")
        model.AddAllowedAssignments([warps_w], [[0], [1], [2], [4], [8]])
        for i, c in enumerate(clusters):
            model.Add(warps_w >= c["min_warps"]).OnlyEnforceIf(assign[(i, w)])
        fp_w = model.NewIntVar(0, max(fp_table), f"fp_{w}")
        model.AddElement(warps_w, fp_table, fp_w)
        total_regs_terms.append(fp_w)
    total_regs = model.NewIntVar(0, 1 << 22, "total_regs")
    model.Add(total_regs == prob["default_wg_footprint"] + sum(total_regs_terms))
    # Twill REGISTERLIMIT-style HARD cap (opt-in): with reg_budget set the
    # solver may not oversubscribe at all — re-solving at a REDUCED budget is
    # Twill's documented answer to model-fits-but-ptxas-spills (their
    # Blackwell FA-bwd found the faster 3-WG strategy only under a lowered
    # budget; our case4 v2 oversubscribed 91136 > 65536 and got downscaled
    # 152→96 regs by the emitter). Unset keeps the soft-deficit model.
    if prob.get("reg_budget"):
        model.Add(total_regs <= prob["reg_budget"])
    deficit = model.NewIntVar(0, 1 << 22, "deficit")
    model.AddMaxEquality(deficit, [total_regs - prob["sm_regs"], 0])
    residual = model.NewIntVar(0, 1 << 22, "residual")
    model.AddMaxEquality(residual, [deficit - prob["default_slack"], 0])
    terms.append(residual)  # x1 while cycle terms are x2 == kDeficitPenalty 0.5

    # Tie-break: prefer MORE warp groups on ties (kPerWGTieBreak analog).
    model.Minimize(sum(terms) - used_wgs)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(prob.get("time_limit_s", 20.0))
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "error", "message": f"partition solve status {status}"}

    def _sol(sv):
        return (
            [sv.Value(wg[i]) for i in range(ncl)],
            sv.ObjectiveValue(),
            sv.Value(used_wgs),
        )

    candidates = [_sol(solver)]
    # Recurrence re-ranking (2026-07-11 calibration): the local-term
    # objective cannot see steady-state recurrence cycles that span WG
    # bodies and channel depths — case4's hardware-best 6-WG layout (292
    # TF) and the 132-TF 8-WG pick score within 15% of each other while
    # running 2.2x apart. Enumerate the K best assignments and re-rank by
    # objective + 2 * max(0, RecMII_aug - II), where RecMII_aug predicts
    # the emitted kernel's cycles/iteration (program-order-augmented
    # recurrence MII). Needs the full graph in the problem (plain edges +
    # buffers with counts); silently skipped for old-format problems.
    if _recmii_rerank_enabled() and _has_full_graph(prob):
        k = int(os.environ.get("TRITON_MODULO_RECMII_K", "8"))
        for _ in range(max(0, k - 1)):
            prev = candidates[-1][0]
            model.Add(sum(assign[(i, prev[i])] for i in range(ncl)) <= ncl - 1)
            s2 = cp_model.CpSolver()
            s2.parameters.max_time_in_seconds = 3.0
            s2.parameters.num_workers = 8
            st2 = s2.Solve(model)
            if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            candidates.append(_sol(s2))
        # Canonical-shape candidates: K-best enumeration climbs the local
        # objective's plateau of merged variants and never reaches pipeline-
        # parallel layouts (case4: the hardware-best shape sits ~750 units
        # above the plateau). Score the domain shapes directly with a fixed-
        # assignment solve (assumptions) — infeasible shapes (SMEM cap,
        # loop-carry legality) drop out for free.
        for amap0 in _canonical_candidates(prob):
            vec0 = [amap0[c["id"]] for c in clusters]
            relab, vec = {}, []  # symmetry breaking: first-use order
            for w0 in vec0:
                if w0 not in relab:
                    relab[w0] = len(relab)
                vec.append(relab[w0])
            if any(vec == c[0] for c in candidates):
                continue
            model.ClearAssumptions()
            model.AddAssumptions([assign[(i, vec[i])] for i in range(ncl)])
            s3 = cp_model.CpSolver()
            s3.parameters.max_time_in_seconds = 3.0
            s3.parameters.num_workers = 8
            if s3.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                candidates.append(_sol(s3))
        model.ClearAssumptions()
        scored = []
        for vec, obj, uw in candidates:
            amap = {clusters[i]["id"]: vec[i] for i in range(ncl)}
            rec = _recmii_augmented(prob, amap)
            excess = max(0.0, rec - ii) if rec is not None else 0.0
            scored.append((obj + 2.0 * excess, rec, vec, obj, uw))
        scored.sort(key=lambda t: t[0])
        combined, rec, vec, obj, uw = scored[0]
        return {
            "status":
            "ok",
            "wg": {str(clusters[i]["id"]): vec[i]
                   for i in range(ncl)},
            "used_wgs":
            uw,
            "objective":
            obj,
            "rec_cycles":
            rec,
            "combined":
            combined,
            "pool":
            len(candidates),
            "pool_scored": [{
                "wg": list(map(int, s[2])),
                "objective": s[3],
                "rec_cycles": s[1],
                "combined": s[0],
            } for s in scored],
        }

    vec, obj, uw = candidates[0]
    return {
        "status": "ok",
        "wg": {str(clusters[i]["id"]): vec[i]
               for i in range(ncl)},
        "used_wgs": uw,
        "objective": obj,
    }


def solve_joint(prob):
    """Joint-formulation v2 (SolverMigrationNotes, step 4 item 1): cycles AND
    warp assignment in ONE solve at the committed II. Where v1 held the
    schedule fixed and charged partition costs against its precomputed
    slack/overlap constants, v2 makes those couplings functions of the cycle
    VARIABLES, so the solver can REORDER instead of paying:

      - cross-WG register hand-off is a conditional HARD latency on the
        dependence edge (Twill's CROSS-WARPSPILLS): cutting an edge means
        the consumer really waits rt cycles — which is what lets the
        recurrence-span term pull an urgent hand-off (case3's alpha) early
        instead of merely penalizing the cut;
      - same-WG issue serialization is a conditional modular NoOverlap over
        the WG's issue durations (an in-order warp cannot co-issue);
      - the async-flight exclusion becomes a circular-difference constraint:
        for async u and sync v in one WG, delta = (phase_v - phase_u) mod II
        must satisfy latency_u <= delta <= II - duration_v (v's issue fits
        in the gap after u's flight); if latency_u + duration_v >= II the
        pair simply cannot share a WG (this is what forces the
        load/compute/store specialization of TC-free loops from first
        principles);
      - SMEM buffer depths follow the re-solved stages (lifetime/II + 1
        over the real-consumer walk serialized by the C++ side), sharing
        one budget with the conditional channel bytes.

    TMEM stage structure is NOT re-constrained here (fixed-II re-solves
    rarely move it and the upstream budget already held); revisit if a
    case trips.
    """
    ii = prob["ii"]
    clusters = prob["clusters"]
    nodes = prob["nodes"]
    node_by_id = {nd["id"]: nd for nd in nodes}
    edges = prob["edges"]
    ncl = len(clusters)
    cindex = {c["id"]: i for i, c in enumerate(clusters)}
    cluster_of = {}
    for c in clusters:
        for nid in c["nodes"]:
            cluster_of[nid] = cindex[c["id"]]

    cp = _critical_path([node_by_id[i] for i in sorted(node_by_id)],
                        [dict(e, src=e["src"], dst=e["dst"]) for e in edges])
    if cp is None:
        return {"status": "error", "message": "cycle in distance-0 edges"}
    max_dur = max((nd["duration"] for nd in nodes), default=1)
    max_stages = min(32, max(4, (cp + max_dur) // ii + 2))
    horizon = max_stages * ii

    model = cp_model.CpModel()
    cycle = {nd["id"]: model.NewIntVar(0, horizon - 1, f"cyc_{nd['id']}") for nd in nodes}
    stage = {nd["id"]: model.NewIntVar(0, max_stages - 1, f"stg_{nd['id']}") for nd in nodes}
    phase = {nd["id"]: model.NewIntVar(0, ii - 1, f"phs_{nd['id']}") for nd in nodes}
    for nd in nodes:
        i = nd["id"]
        model.AddDivisionEquality(stage[i], cycle[i], ii)
        model.AddModuloEquality(phase[i], cycle[i], ii)
        model.AddHint(cycle[i], max(0, min(horizon - 1, nd["cycle"])))

    wg = [model.NewIntVar(0, ncl - 1, f"wg_{c['id']}") for c in clusters]
    prefix_max = wg[0]
    model.Add(wg[0] == 0)
    for i in range(1, ncl):
        model.Add(wg[i] <= prefix_max + 1)
        nxt = model.NewIntVar(0, ncl - 1, f"pmax_{i}")
        model.AddMaxEquality(nxt, [prefix_max, wg[i]])
        prefix_max = nxt
    used_wgs = model.NewIntVar(1, ncl, "used_wgs")
    model.Add(used_wgs == prefix_max + 1)
    # Emitter-capability legality (versioned; see EmitterCaps in
    # ModuloSchedulePass.cpp): max_wgs bounds the WG count (caps v2
    # retired the outer-loop single-WG constraint).
    model.Add(used_wgs <= prob.get("max_wgs", ncl))

    same_cache = {}

    def same_wg(i, j):
        key = (min(i, j), max(i, j))
        if key not in same_cache:
            s = model.NewBoolVar(f"same_{key[0]}_{key[1]}")
            model.Add(wg[key[0]] == wg[key[1]]).OnlyEnforceIf(s)
            model.Add(wg[key[0]] != wg[key[1]]).OnlyEnforceIf(s.Not())
            same_cache[key] = s
        return same_cache[key]

    # LEGALITY: async results read by software (see _intrawg_async_reader_pairs).
    for ci, cj in _intrawg_async_reader_pairs(node_by_id, edges, cluster_of):
        model.Add(same_wg(ci, cj) == 0)

    # Dependences, with the cross-WG hand-off as a conditional hard latency.
    # Distance-0 edges get a STRICT +1 floor: half the ScheduleLoop-level
    # edges carry latency 0/1 (issue-order semantics, not the DDG's
    # data-ready latencies), and letting connected ops tie on one cycle
    # makes the emitter's expression rendering inline/duplicate operands
    # with iter-arg version skew (measured: case3 wrong results).
    terms = []
    chan_terms = []
    seen_pairs = set()
    for e in edges:
        lat = max(e["latency"], 1) if e["distance"] == 0 else e["latency"]
        base = cycle[e["src"]] + lat - e["distance"] * ii
        model.Add(cycle[e["dst"]] >= base)
        if e["distance"] == 0:
            model.Add(stage[e["src"]] <= stage[e["dst"]])
        ci = cluster_of.get(e["src"])
        cj = cluster_of.get(e["dst"])
        if ci is None or cj is None or ci == cj:
            continue
        if e["distance"] > 0 and e.get("rt", 0) > 0:
            # LEGALITY: a register loop-carried value cannot cross WGs — it
            # lives in iter_args with no channel semantics (the semaphore
            # layer strips cycle-inverted edges to signal-only and the
            # emitter falls back to cross-WG recomputation, which is wrong).
            model.Add(same_wg(ci, cj) == 1)
            continue
        cross = same_wg(ci, cj).Not()
        if e.get("rt", 0) > 0:
            model.Add(cycle[e["dst"]] >= base + e["rt"]).OnlyEnforceIf(cross)
        terms.append(1024 * e["xissue"] * cross)
        pair = (e["src"], e["dst"])
        if e.get("chan_bytes", 0) > 0 and pair not in seen_pairs:
            seen_pairs.add(pair)
            cb = model.NewIntVar(0, e["chan_bytes"], f"cb_{pair[0]}_{pair[1]}")
            model.Add(cb == e["chan_bytes"]).OnlyEnforceIf(cross)
            model.Add(cb == 0).OnlyEnforceIf(cross.Not())
            chan_terms.append(cb)

    # Chip-wide per-pipeline modular NoOverlap (same as the schedule solve).
    def modular_segments(i, dur, tag, presence=None):
        segs = []
        end1 = model.NewIntVar(1, ii, f"end1_{tag}")
        model.AddMinEquality(end1, [phase[i] + dur, ii])
        size1 = model.NewIntVar(1, dur, f"size1_{tag}")
        model.Add(size1 == end1 - phase[i])
        if presence is None:
            segs.append(model.NewIntervalVar(phase[i], size1, end1, f"seg1_{tag}"))
        else:
            segs.append(model.NewOptionalIntervalVar(phase[i], size1, end1, presence, f"seg1_{tag}"))
        if dur > 1:
            over = model.NewIntVar(0, dur - 1, f"over_{tag}")
            model.AddMaxEquality(over, [phase[i] + dur - ii, 0])
            wraps = model.NewBoolVar(f"wrap_{tag}")
            model.Add(over >= 1).OnlyEnforceIf(wraps)
            model.Add(over == 0).OnlyEnforceIf(wraps.Not())
            pres2 = wraps
            if presence is not None:
                pres2 = model.NewBoolVar(f"wrappres_{tag}")
                model.AddBoolAnd([wraps, presence]).OnlyEnforceIf(pres2)
                model.AddBoolOr([wraps.Not(), presence.Not()]).OnlyEnforceIf(pres2.Not())
            segs.append(model.NewOptionalIntervalVar(0, over, over, pres2, f"seg2_{tag}"))
        return segs

    by_pipe = {}
    for nd in nodes:
        if nd["pipeline"] != "NONE" and nd["duration"] > ii:
            return {"status": "error", "message": "duration exceeds II"}
        if nd["pipeline"] != "NONE":
            by_pipe.setdefault(nd["pipeline"], []).append(nd["id"])
    for pipe, members in by_pipe.items():
        if len(members) < 2:
            continue
        segs = []
        for i in members:
            segs += modular_segments(i, max(node_by_id[i]["duration"], 1), f"p_{pipe}_{i}")
        model.AddNoOverlap(segs)

    # Same-WG issue serialization: an in-order warp cannot co-issue two ops.
    assign = {}
    for i, c in enumerate(clusters):
        for w in range(ncl):
            a = model.NewBoolVar(f"a_{i}_{w}")
            model.Add(wg[i] == w).OnlyEnforceIf(a)
            model.Add(wg[i] != w).OnlyEnforceIf(a.Not())
            assign[(i, w)] = a
    for w in range(ncl):
        segs = []
        for c in clusters:
            for nid in c["nodes"]:
                segs += modular_segments(nid, max(node_by_id[nid]["duration"], 1), f"w{w}_{nid}",
                                         presence=assign[(cindex[c["id"]], w)])
        if segs:
            model.AddNoOverlap(segs)

    # Async-flight exclusion within a WG (circular difference encoding).
    # Legacy-only (TRITON_MODULO_ASYNC_FLIGHT_BLOCKING=1): this HARD
    # constraint embodies the pre-skew emitter model — sync work may not sit
    # anywhere in a same-WG async op's flight, and when flight + duration
    # >= II the pair may not share a WG at all. With case4's MMA flights
    # (559/900 cyc) against II=1033 that forbade nearly every merge and
    # forced the 8-WG layout (95-107 TF vs the 6-WG 292 TF). The emitter's
    # stage-skew rings make issue-duration NoOverlap (above) the correct
    # occupancy model; see solve_partition's window() for the soft-cost
    # analog and the 2026-07-11 calibration numbers.
    ASYNC = ("TMA", "TC", "MFMA")
    clustered = [nid for c in clusters for nid in c["nodes"]]
    for u in clustered if _async_flight_blocking() else []:
        nu = node_by_id[u]
        if nu["pipeline"] not in ASYNC or nu["latency"] <= nu["duration"]:
            continue
        flight = min(nu["latency"], ii)
        for v in clustered:
            nv = node_by_id[v]
            if nv["pipeline"] in ASYNC or nv["pipeline"] == "NONE" or u == v:
                continue
            ci, cj = cluster_of[u], cluster_of[v]
            if ci == cj:
                continue  # same cluster is same WG by construction: tolerated
            s = same_wg(ci, cj)
            dv = max(nv["duration"], 1)
            if flight + dv >= ii:
                model.Add(s == 0)  # cannot share a WG at this II at all
                continue
            tmp = model.NewIntVar(1, 2 * ii - 1, f"cd_{u}_{v}")
            model.Add(tmp == phase[v] - phase[u] + ii)
            delta = model.NewIntVar(0, ii - 1, f"delta_{u}_{v}")
            model.AddModuloEquality(delta, tmp, ii)
            model.Add(delta >= flight).OnlyEnforceIf(s)
            model.Add(delta <= ii - dv).OnlyEnforceIf(s)

    # SMEM: buffer depths follow the re-solved stages; one budget with the
    # conditional channel bytes.
    depth_vars = []
    smem_terms = []
    for b in prob["buffers"]:
        p = b["producer"]
        ends = [cycle[p]]
        for cons in b["consumers"]:
            ends.append(cycle[cons["node"]] + cons["latency"] + cons["distance"] * ii)
        last_end = model.NewIntVar(0, horizon + 2 * ii, f"lend_b{p}")
        model.AddMaxEquality(last_end, ends)
        lifetime = model.NewIntVar(0, horizon + 2 * ii, f"life_b{p}")
        model.Add(lifetime == last_end - cycle[p])
        dm1 = model.NewIntVar(0, max_stages + 2, f"dm1_b{p}")
        model.AddDivisionEquality(dm1, lifetime, ii)
        depth = model.NewIntVar(1, max_stages + 3, f"depth_b{p}")
        model.Add(depth == dm1 + 1)
        depth_vars.append(depth)
        smem_terms.append(b["size_bytes"] * depth)
    smem_total = model.NewIntVar(0, prob["smem_budget"], "smem_total")
    model.Add(smem_total == prob["fixed_smem"] + sum(smem_terms) + sum(chan_terms))

    # Register budget (same as v1 partition mode).
    fp_table = prob["warp_footprint"]
    fp_terms = []
    for w in range(ncl):
        warps_w = model.NewIntVar(0, 8, f"warps_{w}")
        model.AddAllowedAssignments([warps_w], [[0], [1], [2], [4], [8]])
        for i, c in enumerate(clusters):
            model.Add(warps_w >= c["min_warps"]).OnlyEnforceIf(assign[(i, w)])
        fp_w = model.NewIntVar(0, max(fp_table), f"fp_{w}")
        model.AddElement(warps_w, fp_table, fp_w)
        fp_terms.append(fp_w)
    total_regs = model.NewIntVar(0, 1 << 22, "total_regs")
    model.Add(total_regs == prob["default_wg_footprint"] + sum(fp_terms))
    # Hard cap when reg_budget is set — see solve_partition for rationale.
    if prob.get("reg_budget"):
        model.Add(total_regs <= prob["reg_budget"])
    deficit = model.NewIntVar(0, 1 << 22, "deficit")
    model.AddMaxEquality(deficit, [total_regs - prob["sm_regs"], 0])
    residual = model.NewIntVar(0, 1 << 22, "residual")
    model.AddMaxEquality(residual, [deficit - prob["default_slack"], 0])

    # Objective: the schedule solve's terms + the partition costs that stay
    # soft (barrier issue, register residual, WG-count tie-break). The rt
    # and flight-window couplings are HARD constraints now — no double
    # charge.
    max_stage = model.NewIntVar(0, max_stages - 1, "max_stage")
    model.AddMaxEquality(max_stage, list(stage.values()))
    mem_srcs = _mem_resident_srcs(prob)
    reg_pressure = sum(cycle[e["dst"]] - cycle[e["src"]]
                       for e in edges
                       if e["distance"] == 0 and e["src"] not in mem_srcs)
    rec_span = sum(cycle[e["src"]] - cycle[e["dst"]] for e in edges if e["distance"] > 0 and e["src"] != e["dst"])
    model.Minimize(10240000 * max_stage - 102400 * sum(depth_vars) + 8192 * rec_span + 1024 * reg_pressure +
                   smem_total + sum(terms) + 512 * residual - used_wgs)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(prob.get("time_limit_s", 20.0))
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "error", "message": f"joint solve status {status}"}
    return {
        "status": "ok",
        "wg": {str(c["id"]): solver.Value(wg[i])
               for i, c in enumerate(clusters)},
        "cycles": {str(nd["id"]): solver.Value(cycle[nd["id"]])
                   for nd in nodes},
        "used_wgs": solver.Value(used_wgs),
        "objective": solver.ObjectiveValue(),
    }


def main():
    if len(sys.argv) != 3:
        print("usage: python -m triton.tools.modulo_joint_solver <problem.json> <solution.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        prob = json.load(f)

    if prob.get("mode") in ("partition", "joint"):
        result = solve_partition(prob) if prob["mode"] == "partition" else solve_joint(prob)
        with open(sys.argv[2], "w") as f:
            json.dump(result, f)
        return 0 if result["status"] == "ok" else 1

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
