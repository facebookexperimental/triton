"""Symbolic semaphore IR for cross-WG synchronization.

Refactor of the emitter's barrier insertion (per `notes/semaphore_ir_refactor.md`).
Mirrors the NVWS semaphore abstraction:
https://patch-diff.githubusercontent.com/raw/triton-lang/triton/pull/10121.diff

Pipeline:
  derive_semaphores  →  combine_semaphores  →  assign_stage_phase  →  lower_semaphore
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class AsyncKind(enum.Enum):
    """How a release happens at the hardware level. Determines lowering."""

    NONE = "none"  # Generic synchronous release: explicit barrier_arrive.
    TMA_LOAD = "tma_load"  # async_descriptor_load: HW arrives via barrier_expect.
    TC5_MMA = "tc5_mma"  # tcgen05.mma: HW arrives via mBarriers=[…].
    WGMMA = "wgmma"  # Hopper warp-group MMA: similar to TC5_MMA.
    TMEM_COPY = "tmem_copy"


class AccessKind(enum.Enum):
    """First-access classification on the consumer side. Determines stage advance."""

    READ = "read"  # Reads existing data — stage stays.
    STORE = "store"  # Writes fresh data — stage advances.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BufferRef:
    """Stable reference to a buffer by id (within a loop)."""

    loop_id: int
    buffer_id: int


@dataclass(frozen=True)
class NodeRef:
    """Stable reference to a node by id (within a loop)."""

    loop_id: int
    node_id: int


@dataclass
class ReleaseSite:
    """One release of a semaphore (one producer's contribution)."""

    node: NodeRef
    async_kind: AsyncKind = AsyncKind.NONE
    # Filled by AssignStagePhase:
    stage: int | None = None


@dataclass
class AcquireSite:
    """One acquire of a semaphore (one consumer's site)."""

    node: NodeRef
    access_kind: AccessKind = AccessKind.UNKNOWN
    # Filled by AssignStagePhase:
    stage: int | None = None
    phase_expr: str | None = None  # Python expression like "_it & 1"


@dataclass
class Semaphore:
    """One semaphore = one mbarrier with `depth` slots.

    A semaphore guards a buffer. Producers `release` it; consumers `acquire`
    it. Combining merges all (producer→same-consumer→same-buffer) edges into
    a single semaphore with pending_count = number of distinct producer
    arrival kinds.
    """

    sem_id: int
    buffer: BufferRef | None  # None = signal-only (no data buffer)
    depth: int
    is_released: bool  # True ⟹ initial state is "empty" (producer-ready)
    producers: list[ReleaseSite] = field(default_factory=list)
    consumers: list[AcquireSite] = field(default_factory=list)
    # Filled by AssignStagePhase:
    pending_count: int = 1
    # Pass A.5 data partitioning: when the consumer is a partitioned op
    # (e.g., the MMA was split into N parallel async_dot calls), each issued
    # async_dot signals EMPTY once via its mBarriers list. The empty barrier
    # therefore needs `arrive_count = N`. Defaults to 1 (no partition).
    empty_arrive_count: int = 1
    # Provenance (for debug/log):
    note: str = ""

    def name(self) -> str:
        """Stable Python identifier used in emitted code."""
        if self.buffer is not None:
            return f"sem{self.sem_id}_b{self.buffer.buffer_id}"
        return f"sem{self.sem_id}_signal"


# ──────────────────────────────────────────────────────────────────────────
# Pipeline entry points (implementations in follow-up tasks)
# ──────────────────────────────────────────────────────────────────────────


def _classify_async_kind(op_kind: str) -> AsyncKind:
    """Map an MLIR op kind to its async hardware mechanism."""
    if op_kind in ("tt.descriptor_load", "ttng.async_tma_copy_global_to_local"):
        return AsyncKind.TMA_LOAD
    if op_kind in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled"):
        return AsyncKind.TC5_MMA
    if op_kind == "ttng.warp_group_dot":
        return AsyncKind.WGMMA
    if op_kind == "ttng.tmem_copy":
        return AsyncKind.TMEM_COPY
    return AsyncKind.NONE


def _classify_consumer_access(op_kind: str) -> AccessKind:
    """Map a consumer's op kind to whether it reads or writes the buffer.

    Read-style (consumes existing data): tmem_load, local_load, MMA operands
    A/B, reduce/arith on the loaded value.

    Store-style (writes fresh data): tmem_store, local_store, descriptor_load,
    MMA accumulator with useD=false. `local_alloc` is a no-op SSA wrapper but
    typically signals "consumes the freshly-arrived bytes" so we mark Store.
    """
    if op_kind in ("ttng.tmem_load", "ttg.local_load"):
        return AccessKind.READ
    if op_kind in (
        "ttng.tmem_store",
        "ttg.local_store",
        "tt.descriptor_load",
        "ttng.async_tma_copy_global_to_local",
        "ttg.local_alloc",
    ):
        return AccessKind.STORE
    # MMA: assume Read on accumulator (useD=true is typical for FA inner-loop
    # P@V; useD=false only on first iter — proven dynamically). For the
    # initial pass, default to UNKNOWN; AssignStagePhase refines.
    if op_kind in (
        "ttng.tc_gen5_mma",
        "ttng.tc_gen5_mma_scaled",
        "ttng.warp_group_dot",
    ):
        return AccessKind.UNKNOWN
    # Pure ops (arith, math, reduce, broadcast) on a loaded value just read.
    return AccessKind.READ


def derive_semaphores(loop: Any, graph: Any = None) -> list[Semaphore]:
    """Derive symbolic semaphores from a loop's schedule.

    Source of truth: `loop.schedule.cross_wg_barriers` — the schedule pass
    has already identified every cross-WG synchronization point and assigned
    a staging buffer (or marked as signal-only). We convert each entry into
    a typed Semaphore, enriching with op-kind info to determine async_kind
    (producer hardware mechanism) and access_kind (consumer first-use).

    NOTE: we use cross_wg_barriers as INPUT IR, not the raw DAG. The schedule
    pass already did the work of identifying cross-WG edges and synthesizing
    staging buffers. Re-deriving from edges would lose that buffer info.
    The downstream `combine_semaphores` step deduplicates redundant entries.

    For each cross_wg_barrier:
      - producer/consumer nodes → ReleaseSite/AcquireSite
      - paired_buffer_id → buffer reference (or None for signal-only)
      - kind ∈ {mbarrier, named}: BOTH become Semaphore — uniform IR
      - producer.op_kind → AsyncKind (TMA/MMA/etc)
      - consumer.op_kind → AccessKind initial guess (refined in AssignStagePhase)
      - producer cycle > consumer cycle → loop-carry → is_released=True
    """
    sched = loop.schedule
    nodes_by_id = {n.id: n for n in sched.nodes}
    bufs_by_id = {b.id: b for b in sched.buffers}

    sems: list[Semaphore] = []
    for cb in sched.cross_wg_barriers:
        src = nodes_by_id.get(cb.producer_node)
        dst = nodes_by_id.get(cb.consumer_node)
        if src is None or dst is None:
            continue

        # Skip cross_wg_barriers whose consumer is `ttng.tmem_alloc(value)`
        # paired with a synthesized SMEM buffer (def_op=None). The
        # tmem_alloc consumes a register value, not SMEM — the schedule
        # synthesized a SMEM staging buffer as a fallback, but the
        # emitter routes the data through the TMEM bridge channel
        # directly. Without this skip, both paths emit barriers and we
        # duplicate work every iteration.
        if dst.op_kind == "ttng.tmem_alloc" and cb.paired_buffer_id is not None:
            paired = bufs_by_id.get(cb.paired_buffer_id)
            if paired is not None and paired.def_op is None:
                continue

        # Loop-carry edges (producer cycle > consumer cycle within an iter)
        # carry a SIGNAL, not data — their "buffer" in the schedule is a
        # ghost that exists only for tracking. Strip the buffer reference
        # so the lowerer treats it as signal-only and the emitter doesn't
        # fabricate a spurious local_load/local_store.
        is_loop_carry = src.schedule_cycle > dst.schedule_cycle
        has_buffer = (
            cb.paired_buffer_id is not None
            and cb.paired_buffer_id in bufs_by_id
            and not is_loop_carry
        )
        buf_ref = BufferRef(loop.loop_id, cb.paired_buffer_id) if has_buffer else None
        # When the cross_wg_barriers entry references a synthesized routing
        # buffer (def_op=None), the actual data buffer is the one defined by
        # the consumer's ttg.local_alloc op. The semaphore depth must match
        # that data buffer's ring count, otherwise the producer can't
        # pipeline across slots and small-input correctness suffers.
        depth = cb.depth
        if has_buffer:
            paired = bufs_by_id[cb.paired_buffer_id]
            data_buf = paired
            if paired.def_op is None and dst.op_ref:
                real = next(
                    (b for b in loop.schedule.buffers if b.def_op == dst.op_ref), None
                )
                if real is not None:
                    data_buf = real
            depth = data_buf.count
        async_kind = _classify_async_kind(src.op_kind)
        access_kind = _classify_consumer_access(dst.op_kind)

        # Pass A.5: if the consumer node was annotated with partition_count>1,
        # the emitter will issue N parallel ops (e.g., N async_dots), and
        # each signals the buffer's EMPTY barrier via mBarriers — so the
        # barrier needs arrive_count=N on the consumer side.
        empty_ac = max(1, getattr(dst, "partition_count", 1) or 1)
        # When the immediate consumer is an SSA wrapper (e.g., local_alloc
        # that wraps the loaded value), look one hop further: the real HW
        # consumer is whichever node has this barrier's buffer in its
        # `consumes_buffers`. The max partition_count across those true
        # consumers determines how many HW arrives EMPTY will receive.
        if cb.paired_buffer_id is not None:
            for cand in sched.nodes:
                if cb.paired_buffer_id in (cand.consumes_buffers or []):
                    empty_ac = max(empty_ac, cand.partition_count or 1)
        # Also walk SSA: find ops whose operands point to this buffer's
        # def_op directly OR via a wrapper (memdesc_trans / local_load), and
        # if any of those ops is a partitioned schedule node, count it.
        # This catches B-operand SMEM bufs that reach a partitioned MMA
        # through ttg.memdesc_trans without showing up in consumes_buffers.
        if graph is not None and cb.paired_buffer_id is not None:
            buf = bufs_by_id.get(cb.paired_buffer_id)
            if buf is not None and buf.def_op:
                # Collect every op_id whose op consumes buf.def_op (with
                # one wrapper hop allowed).
                def _refs_to(target_op_id: str) -> set[str]:
                    refs: set[str] = set()
                    for oid, op in graph.ops.items():
                        for opnd in op.operands:
                            if hasattr(opnd, "op_id") and opnd.op_id == target_op_id:
                                refs.add(oid)
                    return refs

                touched = _refs_to(buf.def_op)
                # One wrapper hop (e.g., memdesc_trans).
                expanded = set(touched)
                for oid in touched:
                    expanded |= _refs_to(oid)
                # Bump if any touching node has partition_count>1.
                for cand in sched.nodes:
                    if cand.op_ref in expanded:
                        empty_ac = max(empty_ac, cand.partition_count or 1)

        sems.append(
            Semaphore(
                sem_id=len(sems),
                buffer=buf_ref,
                depth=depth,
                is_released=is_loop_carry,
                producers=[
                    ReleaseSite(
                        node=NodeRef(loop.loop_id, src.id),
                        async_kind=async_kind,
                    )
                ],
                consumers=[
                    AcquireSite(
                        node=NodeRef(loop.loop_id, dst.id),
                        access_kind=access_kind,
                    )
                ],
                empty_arrive_count=empty_ac,
                note=(
                    f"N{src.id}→N{dst.id}  {src.op_kind}→{dst.op_kind}  "
                    f"cyc{src.schedule_cycle}→cyc{dst.schedule_cycle}  "
                    f"{'LOOP-CARRY' if is_loop_carry else 'forward'}  "
                    f"buf={cb.paired_buffer_id}  kind={cb.kind}"
                ),
            )
        )

    # ── Intra-WG async-load consumers (general inner-warp async TMA) ─────────
    # `cross_wg_barriers` capture only CROSS-WG edges. When an async TMA load
    # fills a buffer that is ALSO read by a consumer in the SAME warp group
    # (e.g. a partition where the load shares a WG with the MMA reading its
    # operand), that consumer must still wait on the load's completion (full)
    # barrier and arrive its empty barrier — the load is async. Register those
    # intra-WG consumers so `combine_semaphores` fans them into the load's
    # existing semaphore: one shared full barrier (the load already signals it)
    # plus a summed empty arrive_count. Without this the consumer would
    # reference an unallocated `<buf>_full` and the empty recycle would under-
    # count, racing the producer's next refill.
    if graph is not None:
        sems.extend(_derive_intrawg_async_consumers(loop, sched, graph, sems))
    return sems


def _derive_intrawg_async_consumers(
    loop: Any, sched: Any, graph: Any, existing_sems: list[Semaphore]
) -> list[Semaphore]:
    """Semaphores for async-TMA-load buffers consumed within the load's own WG.
    See caller for rationale. One AcquireSite per intra-WG buffer (the buffer's
    `local_alloc` node — the same node the MMA-operand wait path looks up)."""
    nodes_by_op = {n.op_ref: n for n in sched.nodes if n.op_ref}
    existing_keys = {
        (
            s.producers[0].node.node_id,
            s.buffer.buffer_id if s.buffer else None,
            s.consumers[0].node.node_id,
        )
        for s in existing_sems
    }

    def _refs_to(op_id: str) -> set[str]:
        out: set[str] = set()
        for oid, op in graph.ops.items():
            for opnd in getattr(op, "operands", []):
                if getattr(opnd, "op_id", None) == op_id:
                    out.add(oid)
        return out

    extra: list[Semaphore] = []
    for buf in sched.buffers:
        if buf.kind != "smem" or not buf.def_op:
            continue
        alloc_op = graph.ops.get(buf.def_op)
        if (
            alloc_op is None
            or alloc_op.kind != "ttg.local_alloc"
            or not getattr(alloc_op, "operands", None)
        ):
            continue
        ld = alloc_op.operands[0]
        prod = nodes_by_op.get(getattr(ld, "op_id", None))
        if prod is None or _classify_async_kind(prod.op_kind) != AsyncKind.TMA_LOAD:
            continue
        alloc_node = nodes_by_op.get(buf.def_op)
        if alloc_node is None or alloc_node.warp_group != prod.warp_group:
            continue  # cross-WG load→alloc already covered by cross_wg_barriers
        key = (prod.id, buf.id, alloc_node.id)
        if key in existing_keys:
            continue  # already a cross_wg consumer (e.g. cross-WG load→MMA)
        # Require a real HW reader of this buffer in the producer's WG (directly
        # or via one memdesc_trans hop) — otherwise registering a consumer that
        # never arrives empty would deadlock the producer's refill.
        touched = _refs_to(buf.def_op)
        reachable = set(touched)
        for oid in touched:
            reachable |= _refs_to(oid)
        has_reader = any(
            c.warp_group == prod.warp_group
            and c.id not in (prod.id, alloc_node.id)
            and c.op_ref in reachable
            for c in sched.nodes
        )
        if not has_reader:
            continue
        existing_keys.add(key)
        extra.append(
            Semaphore(
                sem_id=len(existing_sems) + len(extra),
                buffer=BufferRef(loop.loop_id, buf.id),
                depth=buf.count,
                is_released=prod.schedule_cycle > alloc_node.schedule_cycle,
                producers=[
                    ReleaseSite(
                        node=NodeRef(loop.loop_id, prod.id),
                        async_kind=AsyncKind.TMA_LOAD,
                    )
                ],
                consumers=[
                    AcquireSite(
                        node=NodeRef(loop.loop_id, alloc_node.id),
                        access_kind=_classify_consumer_access(alloc_node.op_kind),
                    )
                ],
                empty_arrive_count=1,
                note=(
                    f"intra-WG async TMA: N{prod.id}→N{alloc_node.id} "
                    f"buf={buf.id} (load+consumer share wg{prod.warp_group})"
                ),
            )
        )
    return extra


def combine_semaphores(sems: list[Semaphore]) -> list[Semaphore]:
    """Combine semaphores so each unique (consumer, buffer) is one semaphore.

    NVWS rule: multiple producers releasing into the same consumer's slot
    become ONE semaphore with pending_count = #distinct arrival kinds.
    Single-producer single-consumer semaphores pass through unchanged.

    Also coalesces by producer: if two semaphores share the same producer
    node and the same buffer (rare but legal), their consumer lists merge.
    Multi-consumer is supported in NVWS — the consumer barrier_arrives are
    combined into the same arrive_count.

    Loop-carry property (is_released) is OR'd across the merged group.
    """
    # First key: (consumer_node_id, buffer_id-or-None). Same key = mergeable.
    by_target: dict[tuple, list[Semaphore]] = {}
    for s in sems:
        # A semaphore from derive_semaphores has exactly one consumer and one
        # producer. (Multi-consumer/producer come only via combining.)
        c = s.consumers[0]
        bkey = s.buffer.buffer_id if s.buffer else None
        key = (c.node.node_id, bkey)
        by_target.setdefault(key, []).append(s)

    out: list[Semaphore] = []
    next_id = 0
    for (consumer_id, bkey), group in by_target.items():
        if len(group) == 1:
            s = group[0]
            s.sem_id = next_id
            s.pending_count = 1
            out.append(s)
            next_id += 1
            continue
        # Merge: one semaphore guarding the same (consumer, buffer) slot,
        # with multiple producers contributing one arrival each.
        merged_producers = []
        for s in group:
            merged_producers.extend(s.producers)
        # Distinct arrival kinds determine pending count (a producer that
        # arrives via TMA hardware counts the same as one via MMA hardware,
        # but two NONE producers contribute two arrivals).
        pending = sum(1 for _ in merged_producers)  # one slot per producer
        merged = Semaphore(
            sem_id=next_id,
            buffer=group[0].buffer,
            depth=group[0].depth,
            is_released=any(s.is_released for s in group),
            producers=merged_producers,
            consumers=[group[0].consumers[0]],
            pending_count=pending,
            empty_arrive_count=max(s.empty_arrive_count for s in group),
            note=(
                f"combined {len(group)} producers → consumer N{consumer_id}; "
                + "; ".join(s.note for s in group)
            ),
        )
        out.append(merged)
        next_id += 1

    # Second pass: coalesce by (producer_node_id, buffer_id) — same producer
    # broadcasting to multiple consumers can share one semaphore.
    by_source: dict[tuple, list[Semaphore]] = {}
    for s in out:
        if len(s.producers) != 1:
            continue
        p = s.producers[0]
        bkey = s.buffer.buffer_id if s.buffer else None
        key = (p.node.node_id, p.async_kind, bkey)
        by_source.setdefault(key, []).append(s)

    final: list[Semaphore] = []
    used_ids = set()
    for (pid, ak, bkey), group in by_source.items():
        if len(group) == 1:
            s = group[0]
            if s.sem_id not in used_ids:
                final.append(s)
                used_ids.add(s.sem_id)
            continue
        # Same producer, same buffer, multiple consumers → fan-out semaphore.
        merged_consumers = []
        for s in group:
            merged_consumers.extend(s.consumers)
        merged = Semaphore(
            sem_id=group[0].sem_id,
            buffer=group[0].buffer,
            depth=group[0].depth,
            is_released=any(s.is_released for s in group),
            producers=[group[0].producers[0]],
            consumers=merged_consumers,
            pending_count=1,  # single producer → one arrival
            # The empty barrier receives one release per distinct consumer
            # semaphore, so SUM their arrival counts (e.g. case7 dout → MMA +
            # bias-reduce both release the shared buffer → arrive_count=2). Each
            # member already carries its own count (1, or N for an A.5 partition
            # fan-out), and the len(group)==1 branch above leaves the
            # single-consumer-per-buffer case untouched.
            empty_arrive_count=sum(s.empty_arrive_count for s in group),
            note=(
                f"fan-out: producer N{pid} → "
                + ", ".join(f"N{c.consumers[0].node.node_id}" for c in group)
            ),
        )
        final.append(merged)
        for s in group:
            used_ids.add(s.sem_id)

    # Reassign sem_ids consecutively for cleanliness.
    for i, s in enumerate(final):
        s.sem_id = i
    return final


def assign_stage_phase(sems: list[Semaphore], loop: Any) -> None:
    """Assign concrete stage index and phase expression to every acquire/release.

    Single-phase mode (default, fast path):
      - depth == 1 ring: stage = 0; phase toggles every iter → "_it & 1".
      - depth == N: stage = "_it % N"; phase toggles every N iters →
        "(_it // N) & 1".

    Multi-phase mode (NOT NEEDED for our current cases — all visit each
    (sem, slot) at most once per iter). Defer until a kernel demands it.

    Stage advancement (NVWS rule):
      - On Store first-access: stage = (stage + 1) % depth
      - On Read first-access:  stage unchanged
    For depth=1, this is moot (only one slot exists).

    Loop-carry (is_released=True): the FIRST iter's wait must pass without
    a producer arrival. We rely on the lower_semaphore step to emit a
    pre-arrival before the loop. Phase expr is unchanged here.
    """
    for sem in sems:
        depth = sem.depth
        if depth == 1:
            stage_expr = "0"
            phase_expr = "(_it & 1)"
        else:
            stage_expr = f"(_it % {depth})"
            phase_expr = f"((_it // {depth}) & 1)"
        # Acquires
        for acq in sem.consumers:
            acq.stage = 0  # actual stage is dynamic; kept as constant 0 for
            # depth=1 and `_it % depth` lowered at emit time
            acq.phase_expr = phase_expr
        # Releases
        for rel in sem.producers:
            rel.stage = 0
        # Stash the dynamic stage expression on the semaphore for the lowerer.
        sem._stage_expr = stage_expr  # type: ignore[attr-defined]
        sem._phase_expr = phase_expr  # type: ignore[attr-defined]


@dataclass
class LoweredSemaphore:
    """A Semaphore lowered to a FULL+EMPTY mbarrier pair (NVWS protocol).

    Per the NVWS lowering diff each cross-WG Semaphore allocates TWO mbarriers:
      <name>_full   — producer arrives, consumer waits.
      <name>_empty  — consumer arrives, producer waits.

    Each barrier gets exactly one arrive per iter (per side), so the phase
    formula is the canonical `_it & 1` (or `(_it // depth) & 1` for ring
    depth N) — no XOR ^1 needed because each barrier tracks one direction.

    Per-iter protocol:
      producer:  wait(EMPTY, phase) → [write|expect+load|mma(...)] → arrive(FULL,1)
      consumer:  wait(FULL,  phase) → [load|mma(...)] → arrive(EMPTY,1)
    HW-issued sides (TMA, MMA) move arrive to mBarriers/expect/load args.
    """

    sem_id: int
    name: str  # logical identifier; full/empty pair derived from it
    full_name: str  # `<name>_full` python identifier
    empty_name: str  # `<name>_empty` python identifier
    alloc_full_stmt: str  # `<full_name> = tlx.alloc_barriers(...)`
    alloc_empty_stmt: (
        str | None
    )  # `<empty_name> = tlx.alloc_barriers(...)`; None = signal-only
    pre_arrive_stmt: str | None  # for is_released=True; pre-arrives FULL barrier
    arrive_count: int  # pending_count
    depth: int

    slot_expr: str  # e.g., "0" or "(_it % 3)"
    phase_expr: str  # consumer-side phase, e.g., "(_it & 1)"

    # Per-acquire/release the rendered statements. Keyed by node_id.
    consumer_wait_at: dict[int, str]  # node_id -> wait(FULL, phase)
    consumer_arrive_at: dict[int, str]  # node_id -> arrive(EMPTY, 1) SW recycle
    producer_wait_at: dict[int, str]  # node_id -> wait(EMPTY, phase) SW
    producer_arrive_at: dict[int, str]  # node_id -> arrive(FULL, 1) SW
    producer_mbarriers_at: dict[int, list[str]]  # HW: FULL slots for mBarriers=[…]
    consumer_mbarriers_at: dict[int, list[str]]  # HW: EMPTY slots for mBarriers=[…]
    producer_expect_at: dict[int, str]  # TMA: barrier_expect_bytes(FULL,N)

    sem: Semaphore  # back-reference for debug

    @property
    def alloc_stmt(self) -> str:  # legacy alias used by some emitters
        return self.alloc_full_stmt


def _bar_wait(name: str, slot_expr: str, phase_expr: str) -> str:
    return f"tlx.barrier_wait({name}[{slot_expr}], {phase_expr})"


def _bar_arrive(name: str, slot_expr: str) -> str:
    return f"tlx.barrier_arrive({name}[{slot_expr}], 1)"


def _bar_expect(name: str, slot_expr: str, n_bytes: int) -> str:
    return f"tlx.barrier_expect_bytes({name}[{slot_expr}], {n_bytes})"


def lower_semaphore(
    sem: Semaphore, *, bytes_for_buffer=lambda b: 0
) -> LoweredSemaphore:
    """Mechanically lower a Semaphore into emittable TLX strings.

    Each cross-WG buffer Semaphore allocates two mbarriers: <name>_full
    (producer arrives, consumer waits) and <name>_empty (consumer arrives,
    producer waits). Signal-only Semaphores (no buffer) only allocate _full.

    `bytes_for_buffer(BufferRef)` returns the buffer byte count used by TMA's
    `barrier_expect_bytes` against the FULL barrier.
    """
    base = f"sem{sem.sem_id}"
    if sem.buffer is not None:
        base = f"sem{sem.sem_id}_b{sem.buffer.buffer_id}"
    full_name = f"{base}_full"
    empty_name = f"{base}_empty"
    depth = max(sem.depth, 1)

    alloc_full_stmt = (
        f"{full_name} = tlx.alloc_barriers"
        f"(num_barriers={depth}, arrive_count={sem.pending_count})"
    )
    # Signal-only semaphores (no buffer) don't need an empty side.
    # Pass A.5: empty barrier needs arrive_count = N when the consumer is
    # partitioned (each of the N parallel ops signals empty once).
    empty_ac = max(1, getattr(sem, "empty_arrive_count", 1) or 1)
    alloc_empty_stmt = (
        None
        if sem.buffer is None
        else (
            f"{empty_name} = tlx.alloc_barriers("
            f"num_barriers={depth}, arrive_count={empty_ac})"
        )
    )

    pre_arrive_stmt = None
    if sem.is_released:
        # Pre-arrive on the FULL barrier so iter-0 consumer wait passes
        # immediately (loop-carry init).
        pre_arrive_stmt = (
            f"{_bar_arrive(full_name, '0')}  # is_released=True (loop-carry init)"
        )

    stage_expr = getattr(sem, "_stage_expr", "0")
    consumer_phase = getattr(sem, "_phase_expr", "(_it & 1)")
    # Producer wait on EMPTY uses opposite parity from consumer wait on FULL,
    # since each barrier sees one arrive per iter from the opposite side and
    # mbarrier wait(P) returns when current parity flips away from P. The
    # standard hand-written convention: consumer = `phase`, producer = `phase ^ 1`.
    producer_phase = f"({consumer_phase} ^ 1)"

    consumer_wait_at: dict[int, str] = {}
    consumer_arrive_at: dict[int, str] = {}
    consumer_mbarriers_at: dict[int, list[str]] = {}
    for acq in sem.consumers:
        nid = acq.node.node_id
        consumer_wait_at[nid] = _bar_wait(full_name, stage_expr, consumer_phase)
        if alloc_empty_stmt is None:
            continue  # signal-only: no recycle to emit
        if acq.access_kind == AccessKind.UNKNOWN:
            # MMA-like consumer recycles via HW mBarriers (the EMPTY
            # barrier slot lands on the MMA's mBarriers list).
            consumer_mbarriers_at.setdefault(nid, []).append(
                f"{empty_name}[{stage_expr}]"
            )
        else:
            # SW consumer: explicit arrive after read.
            consumer_arrive_at[nid] = _bar_arrive(empty_name, stage_expr)

    producer_wait_at: dict[int, str] = {}
    producer_arrive_at: dict[int, str] = {}
    producer_mbarriers_at: dict[int, list[str]] = {}
    producer_expect_at: dict[int, str] = {}
    for rel in sem.producers:
        nid = rel.node.node_id
        # Producer waits EMPTY before writing. Signal-only semaphores have
        # no empty side — producer just arrives on FULL with no wait. For
        # is_released semaphores the initial "FULL is pre-arrived" + the
        # consumer's later EMPTY arrive keep the protocol consistent
        # without a producer wait on the first iter.
        if alloc_empty_stmt is not None and not sem.is_released:
            producer_wait_at[nid] = _bar_wait(empty_name, stage_expr, producer_phase)
        if rel.async_kind in (AsyncKind.TC5_MMA, AsyncKind.WGMMA, AsyncKind.TMEM_COPY):
            producer_mbarriers_at.setdefault(nid, []).append(
                f"{full_name}[{stage_expr}]"
            )
        elif rel.async_kind == AsyncKind.TMA_LOAD:
            n_bytes = bytes_for_buffer(sem.buffer) if sem.buffer else 0
            if n_bytes > 0:
                producer_expect_at[nid] = _bar_expect(full_name, stage_expr, n_bytes)
            producer_mbarriers_at.setdefault(nid, []).append(
                f"{full_name}[{stage_expr}]"
            )
        else:
            producer_arrive_at[nid] = _bar_arrive(full_name, stage_expr)

    return LoweredSemaphore(
        sem_id=sem.sem_id,
        name=base,
        full_name=full_name,
        empty_name=empty_name,
        alloc_full_stmt=alloc_full_stmt,
        alloc_empty_stmt=alloc_empty_stmt,
        pre_arrive_stmt=pre_arrive_stmt,
        arrive_count=sem.pending_count,
        depth=depth,
        slot_expr=stage_expr,
        phase_expr=consumer_phase,
        consumer_wait_at=consumer_wait_at,
        consumer_arrive_at=consumer_arrive_at,
        consumer_mbarriers_at=consumer_mbarriers_at,
        producer_wait_at=producer_wait_at,
        producer_arrive_at=producer_arrive_at,
        producer_mbarriers_at=producer_mbarriers_at,
        producer_expect_at=producer_expect_at,
        sem=sem,
    )


def lower_all(
    sems: list[Semaphore], buffers_by_id: dict | None = None
) -> list[LoweredSemaphore]:
    """Convenience: lower every semaphore. Resolves byte counts from buffer ids."""

    def bytes_for(buf_ref):
        if buf_ref is None or buffers_by_id is None:
            return 0
        b = buffers_by_id.get(buf_ref.buffer_id)
        return b.size_bytes if b else 0

    return [lower_semaphore(s, bytes_for_buffer=bytes_for) for s in sems]


# ──────────────────────────────────────────────────────────────────────────
# SemSet — emitter-facing lookup over the lowered semaphores
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SemSet:
    """All lowered semaphores for a kernel, with indices for emit-time lookup.

    Built once per kernel (after `derive → combine → assign_stage_phase →
    lower_all`). The emitter consumes this single object instead of the
    fragmented Channel/cross_wg_barrier paths.
    """

    lowered: list[LoweredSemaphore]
    # Index: (loop_id, consumer_node_id) → list of LoweredSemaphores.
    by_consumer: dict[tuple[int, int], list[LoweredSemaphore]] = field(
        default_factory=dict
    )
    # Index: (loop_id, producer_node_id) → list of LoweredSemaphores.
    by_producer: dict[tuple[int, int], list[LoweredSemaphore]] = field(
        default_factory=dict
    )
    # Index: (loop_id, consumer_wg) → list of LoweredSemaphores requiring
    # a pre-arrive in that WG's preamble (is_released=True on the consumer side).
    by_pre_arrive_wg: dict[tuple[int, int], list[LoweredSemaphore]] = field(
        default_factory=dict
    )

    @classmethod
    def build(
        cls,
        lowered: list[LoweredSemaphore],
        wg_of_node: dict[tuple[int, int], int] | None = None,
    ) -> "SemSet":
        out = cls(lowered=lowered)
        for ls in lowered:
            sem = ls.sem
            for cons in sem.consumers:
                key = (cons.node.loop_id, cons.node.node_id)
                out.by_consumer.setdefault(key, []).append(ls)
            for prod in sem.producers:
                key = (prod.node.loop_id, prod.node.node_id)
                out.by_producer.setdefault(key, []).append(ls)
            if sem.is_released and wg_of_node is not None and sem.consumers:
                # Pre-arrive lives in the CONSUMER WG's preamble. Use the
                # canonical first consumer's WG as the host.
                cons = sem.consumers[0]
                wg = wg_of_node.get((cons.node.loop_id, cons.node.node_id))
                if wg is not None:
                    out.by_pre_arrive_wg.setdefault((cons.node.loop_id, wg), []).append(
                        ls
                    )
        return out

    def alloc_lines(self) -> list[str]:
        """All `<name>_full/_empty = tlx.alloc_barriers(...)` statements."""
        out: list[str] = []
        for ls in self.lowered:
            out.append(ls.alloc_full_stmt)
            if ls.alloc_empty_stmt is not None:
                out.append(ls.alloc_empty_stmt)
        return out

    def consumer_waits(self, loop_id: int, node_id: int) -> list[str]:
        """`tlx.barrier_wait(...)` lines a consumer node emits before reading."""
        out: list[str] = []
        for ls in self.by_consumer.get((loop_id, node_id), []):
            stmt = ls.consumer_wait_at.get(node_id)
            if stmt:
                out.append(stmt)
        return out

    def consumer_arrives(self, loop_id: int, node_id: int) -> list[str]:
        """`tlx.barrier_arrive(...)` SW-recycle lines a consumer node emits
        after reading. Empty when the consumer is HW-issued (e.g., MMA reading
        an SMEM operand) — the recycle happens via mBarriers instead."""
        out: list[str] = []
        for ls in self.by_consumer.get((loop_id, node_id), []):
            stmt = ls.consumer_arrive_at.get(node_id)
            if stmt:
                out.append(stmt)
        return out

    def consumer_mbarriers(self, loop_id: int, node_id: int) -> list[str]:
        """EMPTY-slot expressions for an MMA consumer's mBarriers list (HW recycle)."""
        out: list[str] = []
        for ls in self.by_consumer.get((loop_id, node_id), []):
            for slot in ls.consumer_mbarriers_at.get(node_id, []):
                out.append(slot)
        return out

    def producer_waits(self, loop_id: int, node_id: int) -> list[str]:
        """`tlx.barrier_wait(...)` for a SW producer waiting empty before write."""
        out: list[str] = []
        for ls in self.by_producer.get((loop_id, node_id), []):
            stmt = ls.producer_wait_at.get(node_id)
            if stmt:
                out.append(stmt)
        return out

    def producer_arrives(self, loop_id: int, node_id: int) -> list[str]:
        """SW `tlx.barrier_arrive(...)` after a NONE-async producer's write."""
        out: list[str] = []
        for ls in self.by_producer.get((loop_id, node_id), []):
            stmt = ls.producer_arrive_at.get(node_id)
            if stmt:
                out.append(stmt)
        return out

    def expect_bytes_for(self, loop_id: int, node_id: int) -> list[str]:
        """`tlx.barrier_expect_bytes(...)` lines for TMA producers, before the load."""
        out: list[str] = []
        for ls in self.by_producer.get((loop_id, node_id), []):
            stmt = ls.producer_expect_at.get(node_id)
            if stmt:
                out.append(stmt)
        return out

    def mbarriers_for(self, loop_id: int, node_id: int) -> list[str]:
        """Slot expressions to attach to an MMA's `mBarriers=[…]` (HW-issued)."""
        out: list[str] = []
        for ls in self.by_producer.get((loop_id, node_id), []):
            for slot in ls.producer_mbarriers_at.get(node_id, []):
                out.append(slot)
        return out

    def pre_arrives_for_wg(self, loop_id: int, wg: int) -> list[str]:
        """Pre-arrive `tlx.barrier_arrive(...)` lines for is_released=True
        semaphores whose consumer lives in this WG."""
        out: list[str] = []
        for ls in self.by_pre_arrive_wg.get((loop_id, wg), []):
            if ls.pre_arrive_stmt:
                out.append(ls.pre_arrive_stmt)
        return out


def build_sem_set_for_loop(
    loop: Any, *, wg_of_node: dict[tuple[int, int], int] | None = None
) -> SemSet:
    """End-to-end pipeline for a single loop: derive → combine → stage/phase → lower."""
    sems = derive_semaphores(loop)
    sems = combine_semaphores(sems)
    assign_stage_phase(sems, loop)
    bufs_by_id = {b.id: b for b in loop.schedule.buffers}
    lowered = lower_all(sems, buffers_by_id=bufs_by_id)
    return SemSet.build(lowered, wg_of_node=wg_of_node)


def build_sem_set_for_graph(
    graph: Any, *, wg_of_node: dict[tuple[int, int], int] | None = None
) -> SemSet:
    """Build one SemSet covering every loop's cross-WG barriers."""
    all_lowered: list[LoweredSemaphore] = []
    next_id = 0
    for loop in graph.loops:
        sems = derive_semaphores(loop, graph=graph)
        sems = combine_semaphores(sems)
        assign_stage_phase(sems, loop)
        bufs_by_id = {b.id: b for b in loop.schedule.buffers}
        lowered = lower_all(sems, buffers_by_id=bufs_by_id)
        # Renumber so sem_ids are unique across the whole kernel.
        for ls in lowered:
            ls.sem_id = next_id
            ls.sem.sem_id = next_id
            old_base = ls.name
            new_base = (
                f"sem{next_id}_b{ls.sem.buffer.buffer_id}"
                if ls.sem.buffer is not None
                else f"sem{next_id}"
            )
            if new_base != old_base:
                ls.name = new_base
                ls.full_name = f"{new_base}_full"
                ls.empty_name = f"{new_base}_empty"

                def _ren(s: str | None) -> str | None:
                    return s.replace(old_base, new_base) if s else s

                ls.alloc_full_stmt = _ren(ls.alloc_full_stmt)
                ls.alloc_empty_stmt = _ren(ls.alloc_empty_stmt)
                ls.pre_arrive_stmt = _ren(ls.pre_arrive_stmt)
                ls.consumer_wait_at = {
                    k: _ren(v) for k, v in ls.consumer_wait_at.items()
                }
                ls.consumer_arrive_at = {
                    k: _ren(v) for k, v in ls.consumer_arrive_at.items()
                }
                ls.consumer_mbarriers_at = {
                    k: [_ren(s) for s in v] for k, v in ls.consumer_mbarriers_at.items()
                }
                ls.producer_wait_at = {
                    k: _ren(v) for k, v in ls.producer_wait_at.items()
                }
                ls.producer_arrive_at = {
                    k: _ren(v) for k, v in ls.producer_arrive_at.items()
                }
                ls.producer_expect_at = {
                    k: _ren(v) for k, v in ls.producer_expect_at.items()
                }
                ls.producer_mbarriers_at = {
                    k: [_ren(s) for s in v] for k, v in ls.producer_mbarriers_at.items()
                }
            next_id += 1
        all_lowered.extend(lowered)
    return SemSet.build(all_lowered, wg_of_node=wg_of_node)
