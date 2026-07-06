"""OR-Tools CP-SAT joint modulo-schedule + warp-assignment + memory solver.

Milestone-staged:
  * reproduce  -> emit the committed schedule verbatim (plumbing round-trip).
  * buffering  -> keep the baseline schedule+partition, maximize buffer depth
                  subject to the SMEM/TMEM budgets (Twill memory constraint).
  * joint      -> [M2] full joint schedule + warp assignment (not yet enabled).

Every mode returns a Solution that is cost-<= the baseline, so the pipeline can
always fall back to the baseline for a structural no-regression guarantee.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import baseline as _baseline
from .config import MachineModel
from .ddg_model import LoopModel, Model
from .solution import LoopSolution, Solution

try:
    from ortools.sat.python import cp_model  # noqa: F401
    _HAVE_ORTOOLS = True
except ImportError:  # pragma: no cover
    _HAVE_ORTOOLS = False


@dataclass
class SolveOptions:
    mode: str = "reproduce"  # reproduce | buffering | joint
    time_limit_s: float = 10.0


# ---------------------------------------------------------------------------
# TMEM / SMEM accounting shared by the memory constraints
# ---------------------------------------------------------------------------


def _tmem_cols(buf) -> int:
    """TMEM columns one copy of ``buf`` occupies. TMEM is 128 rows x N cols of
    f32; cols = ceil(bytes_per_copy / (128*4))."""
    per_copy = buf.size_bytes
    return max(1, (per_copy + 511) // 512)


def _smem_bytes_used(loop: LoopModel, counts: dict[int, int]) -> int:
    total = 0
    for b in loop.buffers:
        if b.kind in ("smem", "barrier"):
            total += b.size_bytes * counts.get(b.id, b.count)
    return total


def _tmem_cols_used(loop: LoopModel, counts: dict[int, int]) -> int:
    total = 0
    for b in loop.buffers:
        if b.kind == "tmem":
            total += _tmem_cols(b) * counts.get(b.id, b.count)
    return total


# ---------------------------------------------------------------------------
# buffering optimization (M1): partition + schedule fixed to baseline, deepen
# under-buffered SMEM tiles to their latency-hiding depth (Twill: enough
# in-flight copies of a streaming producer to cover its latency at II), bounded
# by the SMEM/TMEM budget. This is NOT "max out the cap" — over-buffering wastes
# SMEM and can cut CTA-per-SM occupancy. A buffer already at its latency-hiding
# depth (e.g. a GEMM tile with count=2 and TMA latency ~= II) is left untouched.
# ---------------------------------------------------------------------------


def _fill_latency(loop: LoopModel, buf_id: int) -> int:
    """Cycles to fill this buffer's contents: the latency of the async producer
    that feeds the buffer's def op (e.g. the TMA load feeding a local_alloc)."""
    prod = None
    for n in loop.nodes:
        if n.produces_buffer == buf_id:
            prod = n
            break
    if prod is None:
        return 0
    lat = prod.latency
    for e in loop.edges:
        if e.dst == prod.id and e.distance == 0:
            lat = max(lat, e.latency)
    return lat


def _latency_hiding_depth(loop: LoopModel, buf_id: int, II: int,
                          baseline: int, cap: int) -> int:
    """ceil(fill_latency / II) + 1, clamped to [baseline, cap]. +1 = the copy
    being consumed while ceil(L/II) copies are in flight."""
    if II <= 0:
        return baseline
    L = _fill_latency(loop, buf_id)
    target = (L + II - 1) // II + 1
    return max(baseline, min(cap, target))


def _solve_buffering(model: Model, base: Solution,
                     machine: MachineModel) -> Solution:
    from ortools.sat.python import cp_model

    out = Solution()
    for loop in model.loops:
        bl = base.loops[loop.loop_id]
        # A buffer is SAFELY deepenable only if (a) it is SMEM, (b) it is the
        # paired data buffer of a cross-WG barrier (so a producer/consumer
        # mbarrier actually manages its ring slots), and (c) its producer is
        # scheduled at an EARLIER stage than its consumer (so extra copies
        # overlap something). Deepening an unmanaged buffer (e.g. an output
        # staging buffer) or a same-stage buffer desyncs the ring index ->
        # illegal memory access. This coupling is why Twill solves depth and
        # schedule jointly.
        stage = bl.node_stage
        deepenable: set[int] = set()
        for cb in loop.cross_wg_barriers:
            pb = cb.get("paired_buffer_id")
            if pb is None:
                continue
            buf = loop.buffer_by_id(pb)
            if buf is None or buf.kind != "smem":
                continue
            ps = stage.get(cb["producer_node"])
            cs = stage.get(cb["consumer_node"])
            if ps is not None and cs is not None and ps < cs:
                deepenable.add(pb)
        data_bufs = [b for b in loop.buffers if b.id in deepenable]
        if not data_bufs:
            out.loops[loop.loop_id] = _clone_loop(bl)
            continue

        m = cp_model.CpModel()
        cnt = {}
        target = {}
        for b in data_bufs:
            lo = bl.buffer_count.get(b.id, b.count)  # never below baseline
            hi = _latency_hiding_depth(loop, b.id, bl.II, lo,
                                       machine.max_buffer_count)
            target[b.id] = hi
            cnt[b.id] = m.NewIntVar(lo, hi, f"cnt_b{b.id}")
            m.AddHint(cnt[b.id], hi)

        # paired barrier buffers track their data buffer's count.
        paired: dict[int, int] = {}
        for b in loop.buffers:
            if b.kind == "barrier" and b.paired_buffer_id in cnt:
                paired[b.id] = b.paired_buffer_id

        # SMEM budget: sum over smem+barrier of size*count <= budget.
        smem_terms = []
        for b in loop.buffers:
            if b.kind == "smem" and b.id in cnt:
                smem_terms.append(b.size_bytes * cnt[b.id])
            elif b.kind == "barrier" and b.id in paired:
                smem_terms.append(b.size_bytes * cnt[paired[b.id]])
            elif b.kind in ("smem", "barrier"):
                smem_terms.append(b.size_bytes * b.count)
        if smem_terms:
            m.Add(sum(smem_terms) <= machine.smem_budget_bytes)

        # Objective: reach each buffer's latency-hiding target depth (its var's
        # upper bound), backing off only if the SMEM budget binds.
        m.Maximize(sum(cnt.values()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(m)

        new_counts = dict(bl.buffer_count)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for b in data_bufs:
                new_counts[b.id] = int(solver.Value(cnt[b.id]))
            for bar_id, data_id in paired.items():
                new_counts[bar_id] = new_counts[data_id]

        sol = _clone_loop(bl)
        sol.buffer_count = new_counts
        out.loops[loop.loop_id] = sol
    return out


def _clone_loop(bl: LoopSolution) -> LoopSolution:
    return LoopSolution(
        loop_id=bl.loop_id, II=bl.II,
        node_cycle=dict(bl.node_cycle), node_stage=dict(bl.node_stage),
        node_cluster=dict(bl.node_cluster), node_wg=dict(bl.node_wg),
        warp_groups=[dict(wg) for wg in bl.warp_groups],
        buffer_count=dict(bl.buffer_count),
        cross_wg_barriers=[dict(b) for b in bl.cross_wg_barriers],
    )


# ---------------------------------------------------------------------------
# eager MMA operand-release safety (Twill memory-aware codegen enablement)
# ---------------------------------------------------------------------------

_MMA_KINDS = ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled")


def _mma_smem_operands(loop, mma_node) -> list[int]:
    """SMEM buffer ids the MMA reads. Prefer the node's consumes_buffers; fall
    back to the SMEM buffers on cross-WG barriers where this MMA is consumer."""
    smem_ids = {b.id for b in loop.buffers if b.kind == "smem"}
    ids = [b for b in mma_node.consumes_buffers if b in smem_ids]
    if ids:
        return ids
    for cb in loop.cross_wg_barriers:
        pb = cb.get("paired_buffer_id")
        if cb.get("consumer_node") == mma_node.id and pb in smem_ids:
            ids.append(pb)
    return ids


def _eager_release_safe(model: Model, sol: Solution) -> bool:
    """True iff every MMA operand SMEM ring is double-buffered (count>=2) in the
    candidate. Only then is eager operand release race-free: the ring depth
    guarantees the MMA finishes reading a slot before the producer refills it.
    Returns False when there is no MMA (flag would be a no-op).
    """
    saw_mma = False
    for loop in model.loops:
        ls = sol.loops.get(loop.loop_id)
        counts = ls.buffer_count if ls else {b.id: b.count for b in loop.buffers}
        for n in loop.nodes:
            if n.op_kind not in _MMA_KINDS:
                continue
            saw_mma = True
            for bid in _mma_smem_operands(loop, n):
                if counts.get(bid, 1) < 2:
                    return False
    return saw_mma


# ---------------------------------------------------------------------------
# Data-partition factor selection (dp_enum --joint)
# ---------------------------------------------------------------------------

# Lexicographic weights for the cross-variant score: II dominates, then
# prologue depth, then (negatively) buffering. Terms must stay below the next
# weight for strict lexicographic order; dp_score warns if they don't.
_DP_W_II = 1_000_000
_DP_W_STAGE = 1_000


def dp_score(sol: Solution) -> int:
    """Scalarized cross-variant cost of one data-partition variant.

    Aggregates over loops — unlike ``Solution.cost()``, whose per-loop
    lexicographic tuple would let an inner loop's stage count outrank the
    outer loop's II when comparing structurally different variants.
    Comparable across variants only AFTER the A.5 occupancy de-bias
    (applyDataPartition scales the bundle by the issue floor, not xN), which
    keeps M-split MAC area — and hence II — honest.
    """
    ii = sum(ls.II for ls in sol.loops.values())
    stages = sum(sum(ls.node_stage.values()) for ls in sol.loops.values())
    bufs = sum(sum(ls.buffer_count.values()) for ls in sol.loops.values())
    if stages >= _DP_W_STAGE or bufs >= _DP_W_STAGE:
        print(f"[twill] dp_score: stage/buffer term ({stages}/{bufs}) "
              f"overflows its lexicographic slot", file=sys.stderr)
    return ii * _DP_W_II + stages * _DP_W_STAGE - bufs


def select_dp_factor(scored: list[tuple[int, int]],
                     time_limit_s: float = 1.0) -> int:
    """Pick the data-partition factor from ``[(factor, dp_score), ...]``.

    One boolean per emit-viable variant, exactly-one, minimize the score.
    Today the variants share no variables, so this equals argmin; the
    explicit CP-SAT encoding is the seam where per-MMA factor booleans with
    SHARED SMEM/TMEM budget terms slot in once multi-MMA kernels arrive —
    selection and buffering then genuinely co-solve in one model.
    """
    if not scored:
        raise ValueError("select_dp_factor: no viable variants")
    if not _HAVE_ORTOOLS:
        return min(scored, key=lambda t: t[1])[0]
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    sel = {f: m.NewBoolVar(f"sel_n{f}") for f, _ in scored}
    m.AddExactlyOne(sel.values())
    m.Minimize(sum(s * sel[f] for f, s in scored))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(m)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for f, _ in scored:
            if solver.Value(sel[f]):
                return f
    return min(scored, key=lambda t: t[1])[0]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def solve(model: Model, opts: SolveOptions) -> Solution:
    base = _baseline.extract(model)

    if opts.mode == "reproduce":
        return base

    if not _HAVE_ORTOOLS:
        raise RuntimeError(
            "ortools not installed; `pip install ortools` or use --reproduce")

    if opts.mode == "buffering":
        cand = _solve_buffering(model, base, model.machine)
    elif opts.mode == "joint":
        raise NotImplementedError("joint (M2) solver not yet enabled")
    else:
        raise ValueError(f"unknown mode {opts.mode!r}")

    # Twill memory-aware codegen: enable eager MMA operand release when the
    # (candidate) buffering proves every operand ring is double-buffered.
    cand.eager_smem_release = _eager_release_safe(model, cand)

    # Cost-model pre-filter (schedule/buffering). NOTE: the AUTHORITATIVE
    # no-regression guarantee is the GPU A/B gate in testing/ab.py, which
    # rejects a candidate that is incorrect or slower and ships the baseline.
    # This filter only avoids obviously-worse schedules before GPU time.
    if cand.is_better_than(base) or cand.cost() == base.cost():
        return cand
    return base
