# pyre-strict
"""Multi-phase (sibling-loop) template lowering — the buffer-reuse pilot.

SUPERSEDED by the general multi-phase driver (emitter._emit_multiphase,
auto-dispatched on sibling-loop graphs -- arbitrary N phases, heterogeneous
geometry/dtype). Kept as the hand-derived reference oracle the general path
was validated against (case8 A/B within a few % of this template); the
--phases CLI flag still routes here.

Consumes a schedule graph with TWO SIBLING single-loop GEMM nests (case8's
shape: two sequential K-loops, loop_id 0/1, both is_outer=False) and emits ONE
TLX kernel in which the phases run back-to-back inside a shared set of warp
groups. When the pass assigned `smem_phase_group`s (TRITON_MODULO_PHASE_POOL=
pool), the two phases' operand rings share backing storage through
`tlx.storage_alias_spec` — footprint = max(phase0, phase1) instead of the sum.

This is a TEMPLATE lowering in the pingpong.py tradition: a strict validate()
gate, then direct emission specialized to the supported pattern (per-phase
bodies mirror case1's proven single-GEMM emission), NOT a general emit()
restructure. What generalization would still need is documented at the bottom
of this docstring.

Structure of the emitted kernel (4 tasks, both phases in each):
  default      — phase-0 acc handoff + C1 epilogue (direct register store),
                 join arrive, phase-1 acc handoff + C2 epilogue.
  loader A/B   — 1-warp TMA tasks: phase-0 ring fill, join wait, phase-1 fill.
  MMA          — 1-warp TC task: phase-0 dots + commit, phase-1 dots + commit.

Inter-phase join: one mbarrier (arrive_count=1). The default task arrives it
right after `barrier_wait(acc0_full)` — tcgen05_commit semantics guarantee
every phase-0 MMA has finished READING the operand rings by then — and both
loader tasks wait it before their first phase-1 TMA write. The join is emitted
in BOTH pool and non-pool kernels so an A/B between them isolates ring depths
/ II, not synchronization structure.

Epilogues store C directly from registers (row-major, N stride): keeping the
TMA-store staging buffer out of SMEM makes both kernels' footprints equal to
what the pass accounted (the c_smem staging the normal emitter adds is not in
the pass budget — a known accounting gap this template sidesteps).

NOT generalized here (the emit() restructure a production version needs):
per-segment preamble/epilogue splitting of arbitrary function-scope ops,
cross-loop WG unification beyond the symmetric 3-WG GEMM case, phases with
cross-loop data dependencies, non-TMA epilogues, >2 phases.
"""

from __future__ import annotations

from .schedule_graph import ScheduleGraph


class PhasesValidationError(ValueError):
    pass


_ARGS = ["A1", "B1", "C1", "A2", "B2", "C2", "M", "N", "K"]


def _validate_loop(loop, idx: int) -> dict:
    sl = loop.schedule
    if loop.is_outer:
        raise PhasesValidationError(f"loop{idx} is an outer loop")
    kinds = sorted(n.op_kind for n in sl.nodes)
    if kinds != sorted(
        ["tt.descriptor_load", "tt.descriptor_load", "ttg.local_alloc",
         "ttg.local_alloc", "ttng.tc_gen5_mma"]
    ):
        raise PhasesValidationError(f"loop{idx} is not a plain descriptor GEMM: {kinds}")
    if any(n.child_pipeline_id is not None for n in sl.nodes):
        raise PhasesValidationError(f"loop{idx} contains a super-node (nested loop)")
    smem = [b for b in sl.buffers if b.kind == "smem"]
    if len(smem) != 2 or any(b.shape != [128, 128] or b.element_bits != 16 for b in smem):
        raise PhasesValidationError(
            f"loop{idx} rings not the supported (128,128)xf16 pair: "
            f"{[(b.shape, b.element_bits) for b in smem]}"
        )
    step = sl.step
    if not hasattr(step, "value"):
        raise PhasesValidationError(f"loop{idx} step is not a constant")
    ub = sl.upper_bound
    if not hasattr(ub, "name") or ub.name != "K":
        raise PhasesValidationError(f"loop{idx} upper bound is not the K argument")
    return {
        "depth_a": smem[0].count,
        "depth_b": smem[1].count,
        "step": int(step.value),
        "II": sl.II,
    }


def validate(graph: ScheduleGraph) -> tuple[dict, dict, bool]:
    """Gate: exactly the two-sibling-GEMM shape this template supports.

    Returns (phase0_info, phase1_info, pooled).
    """
    if [a.name for a in graph.kernel.args] != _ARGS:
        raise PhasesValidationError(
            f"kernel args {[a.name for a in graph.kernel.args]} != {_ARGS}"
        )
    if len(graph.loops) != 2:
        raise PhasesValidationError(f"need exactly 2 loops, got {len(graph.loops)}")
    p0 = _validate_loop(graph.loops[0], 0)
    p1 = _validate_loop(graph.loops[1], 1)
    groups = [getattr(L, "smem_phase_group", None) for L in graph.loops]
    pooled = groups == [0, 1]
    if not pooled and groups != [None, None]:
        raise PhasesValidationError(f"unsupported smem_phase_group pattern {groups}")
    return p0, p1, pooled


def _phase_allocs(pooled: bool, p0: dict, p1: dict) -> str:
    if pooled:
        return f"""    # ── Pooled operand rings: phase-0 and phase-1 pairs share bytes ──
    # (footprint = max(phase sums), realized by the TLX storage-alias pass)
    smem_pool = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    P0_a = tlx.local_alloc((128, 128), tl.float16, {p0['depth_a']}, reuse=smem_pool)
    P0_b = tlx.local_alloc((128, 128), tl.float16, {p0['depth_b']}, reuse=smem_pool)
    P1_a = tlx.local_alloc((128, 128), tl.float16, {p1['depth_a']}, reuse=smem_pool)
    P1_b = tlx.local_alloc((128, 128), tl.float16, {p1['depth_b']}, reuse=smem_pool)
    smem_pool.set_buffer_overlap(
        tlx.reuse_group(
            tlx.reuse_group(P0_a, P0_b, group_type=tlx.reuse_group_type.distinct),
            tlx.reuse_group(P1_a, P1_b, group_type=tlx.reuse_group_type.distinct),
            group_type=tlx.reuse_group_type.shared,
        )
    )"""
    return f"""    # ── Separate operand rings (no pooling; footprint = sum of phases) ──
    P0_a = tlx.local_alloc((128, 128), tl.float16, {p0['depth_a']})
    P0_b = tlx.local_alloc((128, 128), tl.float16, {p0['depth_b']})
    P1_a = tlx.local_alloc((128, 128), tl.float16, {p1['depth_a']})
    P1_b = tlx.local_alloc((128, 128), tl.float16, {p1['depth_b']})"""


def _loader(ring: str, desc: str, coords: str, depth: int, step: int,
            first: bool) -> str:
    body = f"""            for tile_id in range(0, K, {step}):
                _it = tile_id // {step}
                buf = _it % {depth}
                ph = (_it // {depth}) & 1
                tlx.barrier_wait({ring}_empty[buf], ph ^ 1)
                tlx.barrier_expect_bytes({ring}_full[buf], 32768)
                tlx.async_descriptor_load({desc}, {ring}[buf], {coords}, {ring}_full[buf])"""
    if first:
        return body
    return f"""            tlx.barrier_wait(join_bar[0], 0)
{body}"""


def _mma_phase(pa: str, pb: str, da: int, db: int, acc: str, step: int) -> str:
    return f"""            for tile_id in range(0, K, {step}):
                _it = tile_id // {step}
                bufa = _it % {da}
                pha = (_it // {da}) & 1
                bufb = _it % {db}
                phb = (_it // {db}) & 1
                tlx.barrier_wait({pa}_full[bufa], pha)
                tlx.barrier_wait({pb}_full[bufb], phb)
                use_acc = (tile_id > 0)
                tlx.async_dot({pa}[bufa], {pb}[bufb], {acc}_tmem[0], use_acc=use_acc, mBarriers=[{pa}_empty[bufa], {pb}_empty[bufb]])
            tlx.tcgen05_commit({acc}_full[0])"""


def emit(graph: ScheduleGraph) -> str:
    p0, p1, pooled = validate(graph)
    step0, step1 = p0["step"], p1["step"]
    mode = "pooled (storage_alias_spec)" if pooled else "separate (no pooling)"
    bar_lines = []
    for ring, depth in (
        ("P0_a", p0["depth_a"]), ("P0_b", p0["depth_b"]),
        ("P1_a", p1["depth_a"]), ("P1_b", p1["depth_b"]),
    ):
        bar_lines.append(
            f"    {ring}_full = tlx.alloc_barriers(num_barriers={depth}, arrive_count=1)\n"
            f"    {ring}_empty = tlx.alloc_barriers(num_barriers={depth}, arrive_count=1)"
        )
    bars = "\n".join(bar_lines)

    return f'''# @generated by sched2tlx --phases — do not edit by hand.
# Source: schedule_graph for kernel `dual_gemm_nows` (two sibling GEMM phases)
# SMEM rings: {mode}; depths P0(A/B)={p0["depth_a"]}/{p0["depth_b"]} P1(A/B)={p1["depth_a"]}/{p1["depth_b"]}; IIs {p0["II"]}/{p1["II"]}
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

RECOMMENDED_NUM_WARPS = 4

@triton.jit
def dual_gemm_nows(
    A1,
    B1,
    C1,
    A2,
    B2,
    C2,
    M,
    N,
    K
):
    # ── Preamble ──
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    off_m = pid_0 * 128
    off_n = pid_1 * 128
    a1_desc = tl.make_tensor_descriptor(A1, [M, K], [K, 1], [128, {step0}])
    b1_desc = tl.make_tensor_descriptor(B1, [K, N], [N, 1], [{step0}, 128])
    a2_desc = tl.make_tensor_descriptor(A2, [M, K], [K, 1], [128, {step1}])
    b2_desc = tl.make_tensor_descriptor(B2, [K, N], [N, 1], [{step1}, 128])

{_phase_allocs(pooled, p0, p1)}
    acc0_tmem = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
    acc1_tmem = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)

    # ── Mbarriers (full/empty per ring slot; one-shot handoffs + join) ──
{bars}
    acc0_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    acc1_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    # Inter-phase join: phase-1 loaders must not touch (pooled) ring bytes
    # until every phase-0 MMA read has completed. acc0_full's tcgen05_commit
    # is exactly that condition; the default task relays it here.
    join_bar = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # ── default: epilogues + join relay ──
        with tlx.async_task("default"):
            rows = off_m + tl.arange(0, 128)
            cols = off_n + tl.arange(0, 128)
            tlx.barrier_wait(acc0_full[0], 0)
            tlx.barrier_arrive(join_bar[0], 1)
            acc0 = tlx.local_load(acc0_tmem[0])
            tl.store(C1 + rows[:, None] * N + cols[None, :], acc0.to(tl.float16))
            tlx.barrier_wait(acc1_full[0], 0)
            acc1 = tlx.local_load(acc1_tmem[0])
            tl.store(C2 + rows[:, None] * N + cols[None, :], acc1.to(tl.float16))
        # ── loader A: phase-0 A ring, join, phase-1 A ring ──
        with tlx.async_task(num_warps=1, num_regs=24):
{_loader("P0_a", "a1_desc", "[off_m, tile_id]", p0["depth_a"], step0, first=True)}
{_loader("P1_a", "a2_desc", "[off_m, tile_id]", p1["depth_a"], step1, first=False)}
        # ── loader B: phase-0 B ring, join, phase-1 B ring ──
        with tlx.async_task(num_warps=1, num_regs=24):
{_loader("P0_b", "b1_desc", "[tile_id, off_n]", p0["depth_b"], step0, first=True)}
{_loader("P1_b", "b2_desc", "[tile_id, off_n]", p1["depth_b"], step1, first=False)}
        # ── MMA: phase-0 dots, phase-1 dots (phase-1 gated via its rings) ──
        with tlx.async_task(num_warps=1, num_regs=24):
{_mma_phase("P0_a", "P0_b", p0["depth_a"], p0["depth_b"], "acc0", step0)}
{_mma_phase("P1_a", "P1_b", p1["depth_a"], p1["depth_b"], "acc1", step1)}
'''
