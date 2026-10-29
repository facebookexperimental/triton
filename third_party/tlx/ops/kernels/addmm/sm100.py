"""Blackwell (sm100) ``tlx.ops.addmm`` implementation.

Fused ``input + a @ b`` via TMA descriptors and TLX warp specialization.
The kernel is persistent over the M/N tile grid with four tasks -- a
multi-warp epilogue task plus 1-warp MMA, TMA-producer, and TMA-store
tasks (MMA shared with ``tlx.ops.mm``; the producer inlines mm's tile
shell so it can prefetch the bias) -- and drains TMEM through a sliced
epilogue so accumulator, bias, and output staging stay small. 1-D
``(N,)`` and 2-D ``(M, N)`` biases share the kernel through the
``BROADCAST_Y`` specialization.

Inputs that cannot use TMA (misaligned bases, exotic strides, K/N not
multiples of 16, or broadcastable-but-neither-``(N,)``-nor-``(M, N)`` biases)
fall back to the direct-load baseline kernel ported from the
generative-recommenders ``triton_addmm_fwd``, so the op keeps its full
input contract on every path.
"""

import functools

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tlx.ops.kernels.mm import _sm100_core as _mm_core
from triton.tools.tensor_descriptor import TensorDescriptor

from ..._holdout import num_sms as _get_num_sms

# slice_empty recycle handshake on per-slot static named barriers (one
# unique ID per slot: base 3 + slot; paired handoff halves the slots,
# so epi <= 8 needs IDs 3..6, inside the user range 3..15).
# slice_empty carries no data (pure pacing: the
# store task recycles each epilogue smem buffer once its TMA store is
# certified done), and per-slot ops strictly alternate (pre-arrive, wait,
# arrive, wait, ...) since the store only recycles a slot after consuming
# it and the epilogue only reuses a slot after the recycle, so every
# generation pairs exactly one arrive with one wait and no arrival is ever
# lost. The store task pre-arrives each slot once before its loop for the
# initially-empty buffers (replacing the skipped first ``s_cnt <
# NUM_EPILOGUE_SMEM_BUFFERS - 1`` mbarrier arrives). Count = epilogue
# (default task, full num_warps=4) + store (1 warp) threads; recount if
# the task warp layout changes. slice_full stays an mbarrier: it hands
# smem data to the store's TMA (async proxy ordering).
_SLICE_EMPTY_BAR_BASE = tl.constexpr(3)
_SLICE_EMPTY_BAR_COUNT = tl.constexpr((4 + 1) * 32)


@triton.jit
def _process_tile_mma_inner_commit(
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    buffers_A,
    buffers_B,
    tmem_buffers,
    smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    TWO_CTAS: tl.constexpr = False,
):
    """Single-CTA MMA pipeline for one tile that hands each group's accumulator
    to the epilogue with a ``tcgen05_commit`` issued right after that group's
    last MMA, so the epilogue drains group 0 while group 1's final MMA is
    still in flight.

    Each stage has one full barrier per MMA group: group 0's covers B and
    A(group 0), later groups' cover only their own A, so every MMA waits on
    exactly one barrier and group 0 never waits for a later group's A. Only
    the last group's MMA releases the stage (tcgen05 completes in order, so its
    arrival implies every earlier group's MMA has finished reading B and A);
    earlier groups issue with ``force_async`` and no barrier.

    With ``TWO_CTAS`` only the pair leader runs it: each ``async_dot`` is a
    ``cta_group::2`` MMA over both CTAs' A rows and B halves, and its
    completion (``empty`` arrivals and the ``tmem_full`` commit) is multicast
    to both CTAs. The collaborative producer lands both CTAs' bytes on the
    leader's full barriers.
    """
    LAST_GROUP: tl.constexpr = NUM_MMA_GROUPS - 1
    local_k_tiles = k_tile_end - k_tile_start
    buf, phase = _mm_core.get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(smem_full_bars[a_buf], phase)
        tlx.barrier_wait(tmem_empty_bars[acc_buf], tmem_write_phase ^ 1)
        if group_id == LAST_GROUP:
            tlx.async_dot(
                buffers_A[a_buf],
                buffers_B[buf],
                tmem_buffers[acc_buf],
                use_acc=False,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=TWO_CTAS,
                out_dtype=tl.float32,
            )
        else:
            tlx.async_dot(
                buffers_A[a_buf],
                buffers_B[buf],
                tmem_buffers[acc_buf],
                use_acc=False,
                two_ctas=TWO_CTAS,
                force_async=True,
                out_dtype=tl.float32,
            )
        if local_k_tiles == 1:
            tlx.tcgen05_commit(tmem_full_bars[acc_buf], two_ctas=TWO_CTAS)
    smem_accum_cnt += 1
    for _ in range(1, local_k_tiles - 1):
        buf, phase = _mm_core.get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(smem_full_bars[a_buf], phase)
            if group_id == LAST_GROUP:
                tlx.async_dot(
                    buffers_A[a_buf],
                    buffers_B[buf],
                    tmem_buffers[acc_buf],
                    use_acc=True,
                    mBarriers=[A_smem_empty_bars[a_buf]],
                    two_ctas=TWO_CTAS,
                    out_dtype=tl.float32,
                )
            else:
                tlx.async_dot(
                    buffers_A[a_buf],
                    buffers_B[buf],
                    tmem_buffers[acc_buf],
                    use_acc=True,
                    two_ctas=TWO_CTAS,
                    force_async=True,
                    out_dtype=tl.float32,
                )
        smem_accum_cnt += 1
    if local_k_tiles > 1:
        buf, phase = _mm_core.get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(smem_full_bars[a_buf], phase)
            if group_id == LAST_GROUP:
                tlx.async_dot(
                    buffers_A[a_buf],
                    buffers_B[buf],
                    tmem_buffers[acc_buf],
                    use_acc=True,
                    mBarriers=[A_smem_empty_bars[a_buf]],
                    two_ctas=TWO_CTAS,
                    out_dtype=tl.float32,
                )
            else:
                tlx.async_dot(
                    buffers_A[a_buf],
                    buffers_B[buf],
                    tmem_buffers[acc_buf],
                    use_acc=True,
                    two_ctas=TWO_CTAS,
                    force_async=True,
                    out_dtype=tl.float32,
                )
            tlx.tcgen05_commit(tmem_full_bars[acc_buf], two_ctas=TWO_CTAS)
        smem_accum_cnt += 1
    return smem_accum_cnt


@triton.jit
def _process_tile_producer_shared_full(
    pid_m,
    pid_n,
    k_tile_start,
    k_tile_end,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    x_desc,
    w_desc,
    buffers_A,
    buffers_B,
    smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    cluster_cta_rank,
    TWO_CTAS: tl.constexpr,
):
    """A/B producer for one tile that shares the full barriers with
    ``_process_tile_mma_inner_commit``: per stage it waits once for the last
    group's MMA to release the stage, then lands B and A(group 0) on a single
    barrier with one ``expect_bytes`` (larger tile first), and each later
    group's A on its own barrier. With ``TWO_CTAS`` both CTAs issue
    ``two_ctas`` loads that complete on the leader's barriers, so only the
    leader expects the pair's combined bytes.
    """
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    BN_PER_CTA: tl.constexpr = BLOCK_N // NUM_CTAS
    A_BYTES: tl.constexpr = (tlx.size_of(tlx.dtype_of(x_desc)) * BLOCK_M_SPLIT * BLOCK_K * NUM_CTAS)
    B_BYTES: tl.constexpr = (tlx.size_of(tlx.dtype_of(w_desc)) * BLOCK_K * BN_PER_CTA * NUM_CTAS)
    B_FIRST: tl.constexpr = B_BYTES >= A_BYTES
    LAST_A: tl.constexpr = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N + cluster_cta_rank * BN_PER_CTA
    buf, phase = _mm_core.get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
    for k in range(k_tile_start, k_tile_end):
        offs_k = k * BLOCK_K
        tlx.barrier_wait(A_smem_empty_bars[LAST_A + buf], phase ^ 1)
        if TWO_CTAS:
            tlx.barrier_expect_bytes(smem_full_bars[buf], A_BYTES + B_BYTES, pred=cluster_cta_rank == 0)
        else:
            tlx.barrier_expect_bytes(smem_full_bars[buf], A_BYTES + B_BYTES)
        if B_FIRST:
            tlx.async_descriptor_load(
                w_desc,
                buffers_B[buf],
                [offs_k, offs_bn],
                smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=TWO_CTAS,
            )
            tlx.async_descriptor_load(
                x_desc,
                buffers_A[buf],
                [offs_am, offs_k],
                smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=TWO_CTAS,
            )
        else:
            tlx.async_descriptor_load(
                x_desc,
                buffers_A[buf],
                [offs_am, offs_k],
                smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=TWO_CTAS,
            )
            tlx.async_descriptor_load(
                w_desc,
                buffers_B[buf],
                [offs_k, offs_bn],
                smem_full_bars[buf],
                eviction_policy="evict_last",
                two_ctas=TWO_CTAS,
            )
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            if TWO_CTAS:
                tlx.barrier_expect_bytes(smem_full_bars[a_buf], A_BYTES, pred=cluster_cta_rank == 0)
            else:
                tlx.barrier_expect_bytes(smem_full_bars[a_buf], A_BYTES)
            tlx.async_descriptor_load(
                x_desc,
                buffers_A[a_buf],
                [offs_am + group_id * BLOCK_M_SPLIT, offs_k],
                smem_full_bars[a_buf],
                eviction_policy="evict_last",
                two_ctas=TWO_CTAS,
            )
        smem_accum_cnt += 1
        buf += 1
        if buf == NUM_SMEM_BUFFERS:
            buf = 0
            phase ^= 1
    return smem_accum_cnt


@triton.jit
def addmm_kernel_tma_ws(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    CLUSTER_N: tl.constexpr,
    MULTICAST_A: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    Y_EVICT_POLICY: tl.constexpr,
    Z_EVICT_POLICY: tl.constexpr,
    USE_CLC: tl.constexpr = False,
):
    """Persistent warp-specialized addmm: epilogue + store + MMA + A/B
    producer, plus a dedicated bias loader task for a 2-D bias.

    Each CTA walks the M/N tile grid with stride ``NUM_SMS``, or, with
    ``USE_CLC``, the grid launches one CTA per tile and each running CTA
    claims further tiles through Cluster Launch Control: the epilogue task
    issues ``clc_producer`` at the top of every tile and every task
    ``clc_consumer``s the same response at the end, so every task walks the
    identical dynamic tile stream (``-1`` ends the loop). The tile mapping
    matches ``tlx.ops.mm``. For (1,1) and (2,1) clusters the A/B smem ring
    uses shared full barriers (see ``_process_tile_producer_shared_full``):
    B and A(group 0) complete on one barrier, each later group's A on its
    own, and the last group's MMA releases the stage. A 1-D bias row is
    loaded by the A/B producer after the operands each tile; a 2-D bias
    (``EPILOGUE_SUBTILE`` subtiles per group through a 2-deep ring that only
    the epilogue drains) is loaded by its own 1-warp task with its own
    ``y_full``/``y_empty`` barriers, so the A/B producer never waits on the
    epilogue and can prefetch the next tile's operands while the epilogue
    drains the current one. The epilogue task drains TMEM in
    ``EPILOGUE_SUBTILE`` slices of ``slice_size`` columns, adding the bias
    slice into rotating smem buffers, while a dedicated 1-warp store task
    TMA-stores each slice. Epilogue and store walk the same
    tile/group/slice schedule with matching global slice counters, so buffer
    ``s_cnt % NUM_EPILOGUE_SMEM_BUFFERS`` aliases the same slice on both
    sides; ``slice_full`` barriers (cycled with ``get_bufidx_phase`` like the
    TMEM barriers) hand full buffers to the store task, and per-slot named
    barriers recycle them ``NUM_EPILOGUE_SMEM_BUFFERS - 1`` slices later,
    once ``async_descriptor_store_wait(NUM_EPILOGUE_SMEM_BUFFERS - 2)``
    certifies the overwritten store completed. With even epi-buf/slice
    counts the handshake batches per slice pair (``PAIRED_HANDOFF``):
    pair slot ``(e_cnt // 2) % NUM_PAIR_SLOTS`` covers two smem bufs
    with one wait/arrive, recycled ``NUM_PAIR_SLOTS - 1`` pairs later
    after ``async_descriptor_store_wait(2 * NUM_PAIR_SLOTS - 2)``.
    ``NUM_TMEM_BUFFERS=2`` lets MMA run one tile ahead of the epilogue.

    With ``NUM_CTAS=2``, cluster pairs sit along M (adjacent M-tiles, same N),
    split B's N-range between them, and issue collaborative 2-CTA MMAs; each
    CTA still owns its M-rows' full-N epilogue. For a (2,1) cluster
    (``COLLAB_2CTA``) the producers issue ``two_ctas`` TMA loads whose
    completion lands on the leader's full barriers (so no per-K pair
    rendezvous), only the leader runs the MMA task (the peer just advances
    its ring counter), and both epilogues release TMEM by arriving on the
    leader's ``tmem_empty``. With ``CLUSTER_N=2`` each 4-CTA cluster
    covers a logical 2x2 tile block (two N-paired MMA pairs, kept on the
    explicit ``cta_bars`` rendezvous protocol and separate A/B full barriers
    through ``tlx.ops.mm``'s shared core), and ``MULTICAST_A`` issues each A
    tile once from the y==0 leader to its M-row pair ({0,2}/{1,3}), halving
    A traffic (B stays CTA-split).
    TODO(njriasan): try again later — KNOWN-BROKEN on B200 (Oct 2026):
    cross-SM targets get completion without data; forced False in the
    config space until validated.
    """
    SPLIT_K: tl.constexpr = 1
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    BN_PER_CTA: tl.constexpr = BLOCK_N // NUM_CTAS
    SLICE: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
    assert SLICE % 2 == 0, "half-split drain needs an even SLICE"
    HALF_SLICE: tl.constexpr = SLICE // 2
    Y_M: tl.constexpr = 1 if BROADCAST_Y else BLOCK_M_SPLIT
    Y_DSIZE: tl.constexpr = tlx.size_of(tlx.dtype_of(y_desc))
    # Pair-batched slice->store handoff: with an even epilogue-buffer
    # count and an even slice count, the store handshake (slice_empty
    # wait + slice_full arrive/wait) runs once per slice PAIR instead of
    # per slice. Pair slot (e_cnt // 2) % NUM_PAIR_SLOTS covers smem
    # bufs 2*slot, 2*slot+1. Odd epi/subtile keeps the legacy per-slice
    # handshake; both branches are constexpr-folded per config.
    PAIRED_HANDOFF: tl.constexpr = (NUM_EPILOGUE_SMEM_BUFFERS % 2 == 0 and EPILOGUE_SUBTILE % 2 == 0)
    NUM_PAIR_SLOTS: tl.constexpr = NUM_EPILOGUE_SMEM_BUFFERS // 2
    CLC_MULTI_CTAS: tl.constexpr = NUM_CTAS * CLUSTER_N > 1
    Y_LOADER_TASK: tl.constexpr = not BROADCAST_Y
    # Every CTA of the cluster runs 4 tasks (5 with the 2-D bias loader) that
    # each consume every CLC response.
    CLC_NUM_CONSUMERS: tl.constexpr = ((5 if Y_LOADER_TASK else 4) * NUM_CTAS * CLUSTER_N)
    COLLAB_2CTA: tl.constexpr = NUM_CTAS == 2 and CLUSTER_N == 1
    TMEM_EMPTY_ARRIVALS: tl.constexpr = (EPILOGUE_SUBTILE * 2 if COLLAB_2CTA else EPILOGUE_SUBTILE)

    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        if CLUSTER_N == 2:
            pair_cta_base = cluster_cta_rank & ~1
        else:
            pair_cta_base = 0
        pred_cta0 = cluster_cta_rank == pair_cta_base
        if COLLAB_2CTA:
            cta_bars = None
        else:
            cta_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=2)
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        pair_cta_base = 0
        cta_bars = None

    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_K),
        tlx.dtype_of(x_desc),
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    buffers_B = tlx.local_alloc((BLOCK_K, BN_PER_CTA), tlx.dtype_of(w_desc), NUM_SMEM_BUFFERS)
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, SLICE),
        tlx.dtype_of(z_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )
    # Bias staging: whole (1, BN) rows for 1-D (one row per tile, single
    # buffer, no prefetch needed), one (split, SLICE) subtile per (group,
    # slice) for 2-D with double buffering (one subtile of TMA prefetch;
    # depth 1 exposes full per-slice TMA latency on the 2-D path).
    NUM_Y_BUFFERS: tl.constexpr = 1 if BROADCAST_Y else 2
    if BROADCAST_Y:
        y_buffer = tlx.local_alloc((1, BLOCK_N), tlx.dtype_of(y_desc), NUM_Y_BUFFERS)
    else:
        y_buffer = tlx.local_alloc((BLOCK_M_SPLIT, SLICE), tlx.dtype_of(y_desc), NUM_Y_BUFFERS)

    A_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    A_smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    if CLUSTER_N == 1:
        B_smem_full_bars = A_smem_full_bars
    else:
        B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=TMEM_EMPTY_ARRIVALS,
    )
    y_full_bars = tlx.alloc_barriers(num_barriers=NUM_Y_BUFFERS, arrive_count=1)
    y_empty_bars = tlx.alloc_barriers(num_barriers=NUM_Y_BUFFERS, arrive_count=1)
    if PAIRED_HANDOFF:
        slice_full_bars = tlx.alloc_barriers(num_barriers=NUM_PAIR_SLOTS, arrive_count=1)
    else:
        slice_full_bars = tlx.alloc_barriers(num_barriers=NUM_EPILOGUE_SMEM_BUFFERS, arrive_count=1)
    if USE_CLC:
        clc_context = tlx.clc_create_context(num_consumers=CLC_NUM_CONSUMERS)
    else:
        clc_context = None
    # slice_empty uses named barriers _SLICE_EMPTY_BAR_BASE + slot (no
    # mbarrier alloc). IDs stay static (constexpr base + static_range).

    with tlx.async_tasks(exclusive=True):
        # Epilogue task: TMEM drain + bias add into smem, persistent. Bias
        # tiles arrive from the producer task; the store task below issues
        # the TMA stores from the same schedule.
        with tlx.async_task("default"):
            (
                start_pid,
                num_pid_m,
                _num_pid_n,
                num_pid_in_group,
                _num_mn_tiles,
                num_tiles,
                _k_tiles_total,
            ) = _mm_core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                SPLIT_K,
                NUM_CTAS,
                CLUSTER_N=CLUSTER_N,
            )
            tmem_accum_cnt = 0
            y_cnt = 0
            e_cnt = 0
            clc_phase_producer = 1
            clc_phase_consumer = 0
            tile_id = start_pid
            while tile_id != -1:
                if USE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=CLC_MULTI_CTAS)
                    clc_phase_producer ^= 1
                cur_tmem_buf, tmem_read_phase = _mm_core.get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)

                if BROADCAST_Y:
                    # One (1, BN) row per tile, shared by all MMA groups.
                    y_buf, y_phase = _mm_core.get_bufidx_phase(y_cnt, NUM_Y_BUFFERS)
                    tlx.barrier_wait(y_full_bars[y_buf], y_phase)
                    y_tile = y_buffer[y_buf]
                    y_cnt += 1

                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                    tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
                    acc_tmem = tmem_buffers[buf_idx]
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        # Half-split interleave: two TMEM loads back-to-back
                        # give the scheduler freedom to hide the second
                        # load's latency under the first half's FMAs, with
                        # no extra live registers vs the whole-slice form.
                        acc_lo = tlx.local_slice(
                            acc_tmem,
                            [0, slice_id * SLICE],
                            [BLOCK_M_SPLIT, HALF_SLICE],
                        )
                        acc_hi = tlx.local_slice(
                            acc_tmem,
                            [0, slice_id * SLICE + HALF_SLICE],
                            [BLOCK_M_SPLIT, HALF_SLICE],
                        )
                        acc0 = tlx.local_load(acc_lo)
                        acc1 = tlx.local_load(acc_hi)
                        if COLLAB_2CTA:
                            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1, remote_cta_rank=0)
                        else:
                            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                        if not BROADCAST_Y:
                            # 2-D bias arrives one subtile at a time.
                            y_buf, y_phase = _mm_core.get_bufidx_phase(y_cnt, NUM_Y_BUFFERS)
                            tlx.barrier_wait(y_full_bars[y_buf], y_phase)
                            y_lo = tlx.local_slice(y_buffer[y_buf], [0, 0], [BLOCK_M_SPLIT, HALF_SLICE])
                            y_hi = tlx.local_slice(y_buffer[y_buf], [0, HALF_SLICE], [BLOCK_M_SPLIT, HALF_SLICE])
                            y0 = tlx.local_load(y_lo)
                            y1 = tlx.local_load(y_hi)
                            tlx.barrier_arrive(y_empty_bars[y_buf], 1)
                            y_cnt += 1
                        else:
                            y_lo = tlx.local_slice(y_tile, [0, slice_id * SLICE], [Y_M, HALF_SLICE])
                            y_hi = tlx.local_slice(
                                y_tile,
                                [0, slice_id * SLICE + HALF_SLICE],
                                [Y_M, HALF_SLICE],
                            )
                            y0 = tlx.local_load(y_lo)
                            y1 = tlx.local_load(y_hi)
                        out_dtype = tlx.dtype_of(z_desc)
                        fused0 = (acc0 + y0.to(tl.float32)).to(out_dtype)
                        fused1 = (acc1 + y1.to(tl.float32)).to(out_dtype)
                        if PAIRED_HANDOFF:
                            # Pair slot from the pair counter; the pair's
                            # two slices fill bufs 2*slot, 2*slot+1. The
                            # within-pair index is static (slice_id): every
                            # tile contributes an even slice count, so
                            # e_cnt % 2 == slice_id % 2 across groups/tiles.
                            p_slot, _p_phase = _mm_core.get_bufidx_phase(e_cnt // 2, NUM_PAIR_SLOTS)
                            c_buf = p_slot * 2 + (slice_id % 2)
                            if slice_id % 2 == 0:
                                for _ebi in tl.static_range(NUM_PAIR_SLOTS):
                                    if p_slot == _ebi:
                                        tlx.named_barrier_wait(
                                            _SLICE_EMPTY_BAR_BASE + _ebi,
                                            _SLICE_EMPTY_BAR_COUNT,
                                        )
                        else:
                            c_buf, c_phase = _mm_core.get_bufidx_phase(e_cnt, NUM_EPILOGUE_SMEM_BUFFERS)
                            for _ebi in tl.static_range(NUM_EPILOGUE_SMEM_BUFFERS):
                                if c_buf == _ebi:
                                    tlx.named_barrier_wait(
                                        _SLICE_EMPTY_BAR_BASE + _ebi,
                                        _SLICE_EMPTY_BAR_COUNT,
                                    )
                        c_smem = c_smem_buffers[c_buf]
                        c_lo = tlx.local_slice(c_smem, [0, 0], [BLOCK_M_SPLIT, HALF_SLICE])
                        c_hi = tlx.local_slice(c_smem, [0, HALF_SLICE], [BLOCK_M_SPLIT, HALF_SLICE])
                        tlx.local_store(c_lo, fused0)
                        tlx.local_store(c_hi, fused1)
                        if PAIRED_HANDOFF:
                            if slice_id % 2 == 1:
                                tlx.barrier_arrive(slice_full_bars[p_slot], 1)
                        else:
                            tlx.barrier_arrive(slice_full_bars[c_buf], 1)
                        e_cnt += 1
                if BROADCAST_Y:
                    tlx.barrier_arrive(y_empty_bars[y_buf], 1)
                tmem_accum_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=CLC_MULTI_CTAS)
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # MMA task: tcgen05 issue. Single-CTA and (2,1)-collaborative tiles
        # run the shared-full/commit pipeline above (only the pair leader
        # issues MMAs); (2,2) clusters stay on ``..mm._sm100_core``'s
        # explicit-rendezvous inner loop.
        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                _num_pid_m,
                _num_pid_n,
                _num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _mm_core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                SPLIT_K,
                NUM_CTAS,
                CAST_TO_INT32=True,
                CLUSTER_N=CLUSTER_N,
            )
            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            clc_phase_consumer = 0
            tile_id = start_pid
            while tile_id != -1:
                k_tile_start, k_tile_end = _mm_core._compute_k_tile_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_write_phase = _mm_core.get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
                    if NUM_CTAS == 1:
                        smem_accum_cnt = _process_tile_mma_inner_commit(
                            k_tile_start,
                            k_tile_end,
                            NUM_SMEM_BUFFERS,
                            NUM_MMA_GROUPS,
                            NUM_TMEM_BUFFERS,
                            buffers_A,
                            buffers_B,
                            tmem_buffers,
                            A_smem_full_bars,
                            A_smem_empty_bars,
                            tmem_full_bars,
                            cur_tmem_buf,
                            tmem_empty_bars,
                            tmem_write_phase,
                            smem_accum_cnt,
                        )
                    elif COLLAB_2CTA:
                        if cluster_cta_rank == 0:
                            smem_accum_cnt = _process_tile_mma_inner_commit(
                                k_tile_start,
                                k_tile_end,
                                NUM_SMEM_BUFFERS,
                                NUM_MMA_GROUPS,
                                NUM_TMEM_BUFFERS,
                                buffers_A,
                                buffers_B,
                                tmem_buffers,
                                A_smem_full_bars,
                                A_smem_empty_bars,
                                tmem_full_bars,
                                cur_tmem_buf,
                                tmem_empty_bars,
                                tmem_write_phase,
                                smem_accum_cnt,
                                TWO_CTAS=True,
                            )
                        else:
                            smem_accum_cnt += k_tile_end - k_tile_start
                    else:
                        smem_accum_cnt = _mm_core._process_tile_mma_inner(
                            k_tile_start,
                            k_tile_end,
                            NUM_SMEM_BUFFERS,
                            NUM_MMA_GROUPS,
                            NUM_TMEM_BUFFERS,
                            buffers_A,
                            buffers_B,
                            tmem_buffers,
                            A_smem_full_bars,
                            B_smem_full_bars,
                            A_smem_empty_bars,
                            tmem_full_bars,
                            cur_tmem_buf,
                            tmem_empty_bars,
                            tmem_write_phase,
                            smem_accum_cnt,
                            NUM_CTAS,
                            cta_bars,
                            pred_cta0,
                            True,
                            True,
                            EXPLICIT_CTA_SYNC=True,
                            pair_cta_base=pair_cta_base,
                        )
                    tmem_accum_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=CLC_MULTI_CTAS)
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # Producer task: TMA loads of A/B tiles plus the 1-D bias row. The
        # tile shell is inlined from ``..mm._sm100_core``; 2-D bias tiles
        # move to the dedicated loader task below so operand prefetch never
        # waits on the epilogue.
        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                num_pid_m,
                num_pid_n_producer,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _mm_core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                SPLIT_K,
                NUM_CTAS,
                CAST_TO_INT32=True,
                CLUSTER_N=CLUSTER_N,
            )
            smem_accum_cnt = 0
            y_cnt = 0
            clc_phase_consumer = 0
            tile_id = start_pid
            while tile_id != -1:
                k_tile_start, k_tile_end = _mm_core._compute_k_tile_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
                if SPLIT_K == 1 or k_tile_end > k_tile_start:
                    # SPLIT_K is 1, so mn_tile_id is tile_id (as in mm).
                    pid_m, pid_n = _mm_core._compute_pid(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        GROUP_M,
                        CLUSTER_N,
                        num_pid_n_producer,
                    )
                    offs_bn = pid_n * BLOCK_N
                    # Operands first: the epilogue needs MMA done before it
                    # consumes bias, so bias waits must not gate A/B issue.
                    if CLUSTER_N == 1:
                        smem_accum_cnt = _process_tile_producer_shared_full(
                            pid_m,
                            pid_n,
                            k_tile_start,
                            k_tile_end,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            NUM_MMA_GROUPS,
                            NUM_SMEM_BUFFERS,
                            NUM_CTAS,
                            x_desc,
                            w_desc,
                            buffers_A,
                            buffers_B,
                            A_smem_full_bars,
                            A_smem_empty_bars,
                            smem_accum_cnt,
                            cluster_cta_rank,
                            COLLAB_2CTA,
                        )
                    else:
                        smem_accum_cnt = _mm_core._process_tile_producer_inner(
                            tile_id,
                            num_pid_in_group,
                            num_pid_m,
                            num_mn_tiles,
                            GROUP_M,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            NUM_MMA_GROUPS,
                            k_tile_start,
                            k_tile_end,
                            NUM_SMEM_BUFFERS,
                            x_desc,
                            w_desc,
                            buffers_A,
                            buffers_B,
                            A_smem_full_bars,
                            B_smem_full_bars,
                            A_smem_empty_bars,
                            smem_accum_cnt,
                            NUM_CTAS,
                            cluster_cta_rank,
                            SPLIT_K,
                            True,
                            True,
                            COLLAB_2CTA,
                            MULTICAST_A=MULTICAST_A,
                            CLUSTER_N=CLUSTER_N,
                            num_pid_n=num_pid_n_producer,
                        )
                    # The 1-D bias row loads after the operands each tile;
                    # 2-D bias subtiles move to the loader task below.
                    if BROADCAST_Y:
                        y_buf, y_phase = _mm_core.get_bufidx_phase(y_cnt, NUM_Y_BUFFERS)
                        tlx.barrier_wait(y_empty_bars[y_buf], y_phase ^ 1)
                        tlx.barrier_expect_bytes(y_full_bars[y_buf], Y_DSIZE * BLOCK_N)
                        tlx.async_descriptor_load(
                            y_desc,
                            y_buffer[y_buf],
                            [0, offs_bn],
                            y_full_bars[y_buf],
                            eviction_policy=Y_EVICT_POLICY,
                        )
                        y_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=CLC_MULTI_CTAS)
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # Bias loader task (2-D only): TMA loads of (group, slice) bias
        # subtiles on their own barriers, so the A/B producer never waits
        # on the epilogue and prefetches ahead of the drain.
        if Y_LOADER_TASK:
            with tlx.async_task(num_warps=1, num_regs=24):
                (
                    start_pid,
                    num_pid_m,
                    num_pid_n,
                    num_pid_in_group,
                    _num_mn_tiles,
                    num_tiles,
                    _k_tiles_total,
                ) = _mm_core._compute_grid_info(
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    GROUP_M,
                    SPLIT_K,
                    NUM_CTAS,
                    CLUSTER_N=CLUSTER_N,
                )
                y_cnt = 0
                clc_phase_consumer = 0
                tile_id = start_pid
                while tile_id != -1:
                    pid_m, pid_n = _mm_core._compute_pid(
                        tile_id,
                        num_pid_in_group,
                        num_pid_m,
                        GROUP_M,
                        CLUSTER_N,
                        num_pid_n,
                    )
                    offs_bn = pid_n * BLOCK_N
                    for group_id in tl.static_range(NUM_MMA_GROUPS):
                        offs_am_g = pid_m * BLOCK_M + group_id * BLOCK_M_SPLIT
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            y_buf, y_phase = _mm_core.get_bufidx_phase(y_cnt, NUM_Y_BUFFERS)
                            tlx.barrier_wait(y_empty_bars[y_buf], y_phase ^ 1)
                            tlx.barrier_expect_bytes(
                                y_full_bars[y_buf],
                                Y_DSIZE * BLOCK_M_SPLIT * SLICE,
                            )
                            tlx.async_descriptor_load(
                                y_desc,
                                y_buffer[y_buf],
                                [offs_am_g, offs_bn + slice_id * SLICE],
                                y_full_bars[y_buf],
                                eviction_policy=Y_EVICT_POLICY,
                            )
                            y_cnt += 1
                    if USE_CLC:
                        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=CLC_MULTI_CTAS)
                        clc_phase_consumer ^= 1
                    else:
                        tile_id += NUM_SMS
                        if tile_id >= num_tiles:
                            tile_id = -1

        # Store task: TMA stores of epilogue slices, persistent. Walks the
        # same tile/group/slice schedule as the epilogue task so the global
        # slice counter aliases the same smem buffer on both sides.
        with tlx.async_task(num_warps=1, num_regs=24):
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                _num_mn_tiles,
                num_tiles,
                _k_tiles_total,
            ) = _mm_core._compute_grid_info(
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                SPLIT_K,
                NUM_CTAS,
                CLUSTER_N=CLUSTER_N,
            )
            s_cnt = 0
            clc_phase_consumer = 0
            tile_id = start_pid
            # Initial empty credit for each slice buffer (named barriers
            # have no phase bit, so the first wait per slot needs a
            # matching arrive up front). Paired handoff credits pair slots.
            if PAIRED_HANDOFF:
                for _ebi in tl.static_range(NUM_PAIR_SLOTS):
                    tlx.named_barrier_arrive(_SLICE_EMPTY_BAR_BASE + _ebi, _SLICE_EMPTY_BAR_COUNT)
            else:
                for _ebi in tl.static_range(NUM_EPILOGUE_SMEM_BUFFERS):
                    tlx.named_barrier_arrive(_SLICE_EMPTY_BAR_BASE + _ebi, _SLICE_EMPTY_BAR_COUNT)
            while tile_id != -1:
                pid_m, pid_n = _mm_core._compute_pid(
                    tile_id,
                    num_pid_in_group,
                    num_pid_m,
                    GROUP_M,
                    CLUSTER_N,
                    num_pid_n,
                )
                offs_bn = pid_n * BLOCK_N
                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    offs_am = pid_m * BLOCK_M + group_id * BLOCK_M_SPLIT
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        if PAIRED_HANDOFF:
                            # Pair handoff: one full-wait per pair (even
                            # slice), both slices' stores issued back to
                            # back, then one store_wait certifies the
                            # P - 1 pair recycle lag before arriving empty.
                            # s_cnt parity is static (slice_id): the schedule
                            # matches the epilogue's pair-for-pair.
                            s_pair = s_cnt // 2
                            s_slot, s_phase = _mm_core.get_bufidx_phase(s_pair, NUM_PAIR_SLOTS)
                            s_buf = s_slot * 2 + (slice_id % 2)
                            if slice_id % 2 == 0:
                                tlx.barrier_wait(slice_full_bars[s_slot], s_phase)
                            tlx.async_descriptor_store(
                                z_desc,
                                c_smem_buffers[s_buf],
                                [offs_am, offs_bn + slice_id * SLICE],
                                eviction_policy=Z_EVICT_POLICY,
                            )
                            if slice_id % 2 == 1:
                                # In-flight bound 2*P-2 certifies the pair
                                # from P-1 pairs ago done (both its stores:
                                # 2*(s+1) issued, 2*P-2 outstanding max =>
                                # the recycled pair's 2 stores completed;
                                # 2*P-1 leaves the 2nd store in flight).
                                # P=1 drains fully (wait(0)), same
                                # serialization as legacy epi2.
                                tlx.async_descriptor_store_wait(2 * NUM_PAIR_SLOTS - 2)
                                if s_pair >= NUM_PAIR_SLOTS - 1:
                                    _slot = (s_pair - (NUM_PAIR_SLOTS - 1)) % NUM_PAIR_SLOTS
                                    for _ebi in tl.static_range(NUM_PAIR_SLOTS):
                                        if _slot == _ebi:
                                            tlx.named_barrier_arrive(
                                                _SLICE_EMPTY_BAR_BASE + _ebi,
                                                _SLICE_EMPTY_BAR_COUNT,
                                            )
                        else:
                            s_buf, s_phase = _mm_core.get_bufidx_phase(s_cnt, NUM_EPILOGUE_SMEM_BUFFERS)
                            tlx.barrier_wait(slice_full_bars[s_buf], s_phase)
                            tlx.async_descriptor_store_wait(NUM_EPILOGUE_SMEM_BUFFERS - 2)
                            tlx.async_descriptor_store(
                                z_desc,
                                c_smem_buffers[s_buf],
                                [offs_am, offs_bn + slice_id * SLICE],
                                eviction_policy=Z_EVICT_POLICY,
                            )
                            # Recycle the buffer written NUM_EPILOGUE_SMEM_BUFFERS
                            # - 1 slices ago: with at most
                            # NUM_EPILOGUE_SMEM_BUFFERS - 2 stores still in flight
                            # and in-order TMA completion, that store is done.
                            if s_cnt >= NUM_EPILOGUE_SMEM_BUFFERS - 1:
                                _slot = (s_cnt - (NUM_EPILOGUE_SMEM_BUFFERS - 1)) % NUM_EPILOGUE_SMEM_BUFFERS
                                for _ebi in tl.static_range(NUM_EPILOGUE_SMEM_BUFFERS):
                                    if _slot == _ebi:
                                        tlx.named_barrier_arrive(
                                            _SLICE_EMPTY_BAR_BASE + _ebi,
                                            _SLICE_EMPTY_BAR_COUNT,
                                        )
                        s_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=CLC_MULTI_CTAS)
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1
            tlx.async_descriptor_store_wait(0)


@triton.jit
def addmm_kernel_direct(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


def _tma_block_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    split = BLOCK_M // nargs["NUM_MMA_GROUPS"]
    sl = BLOCK_N // nargs["EPILOGUE_SUBTILE"]
    nargs["x_desc"].block_shape = [split, BLOCK_K]
    nargs["w_desc"].block_shape = [BLOCK_K, BLOCK_N // nargs["NUM_CTAS"]]
    nargs["z_desc"].block_shape = [split, sl]
    if nargs["BROADCAST_Y"]:
        nargs["y_desc"].block_shape = [1, BLOCK_N]
    else:
        nargs["y_desc"].block_shape = [split, sl]


def _config(block_m, block_n, block_k, group_m, num_warps, num_stages, pre_hook=None, num_smem_buffers=None):
    kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    if num_smem_buffers is not None:
        kwargs["NUM_SMEM_BUFFERS"] = num_smem_buffers
    return triton.Config(
        kwargs,
        num_warps=num_warps,
        num_stages=num_stages,
        pre_hook=pre_hook,
    )


def _ws_config(block_m, block_n, block_k, group_m, smem_bufs, tmem_bufs, mma_groups, subtile, epi_smem_bufs=3,
               num_ctas=1, cluster_n=1, y_evict=None, z_evict="evict_first", use_clc=True):
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_SMEM_BUFFERS": smem_bufs,
            "NUM_TMEM_BUFFERS": tmem_bufs,
            "NUM_MMA_GROUPS": mma_groups,
            "EPILOGUE_SUBTILE": subtile,
            "NUM_EPILOGUE_SMEM_BUFFERS": epi_smem_bufs,
            "NUM_CTAS": num_ctas,
            "CLUSTER_N": cluster_n,
            # TODO(njriasan): try again later — MULTICAST_A is KNOWN-BROKEN
            # on B200 (cross-SM multicast targets get completion without
            # data; issuerless pairs fault). Forced False until validated
            # on a fixed stack; (2,2) twins below run plain A.
            "MULTICAST_A": False,
            "Y_EVICT_POLICY": y_evict,
            "Z_EVICT_POLICY": z_evict,
            "USE_CLC": use_clc,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_tma_block_hook,
        # Physical cluster dims stay 1-D: (4,1,1) implements the logical 2x2
        # tile block (ranks 0..3 consecutive) and divides the 1-D grid; a
        # (2,2,1) cluster would need a 2-D grid (cuLaunchKernelEx 912).
        ctas_per_cga=(num_ctas * cluster_n, 1, 1) if (num_ctas * cluster_n > 1) else None,
    )


def _ws_prune_configs(configs, named_args, **kwargs):
    """Drop WS configs that cannot fit the B200 resource budget or the shape.

    Safety net over the hand-picked space below: smem (A/B operands +
    epilogue staging + bias tile + barriers) under 232KB, TMEM accumulators
    under 256KB, tcgen05 M-split in {64, 128}, and tiles within 2x of the
    problem. Mirrors the ``tlx.ops.mm`` estimator.
    """
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    # Constexpr kernel params are absent from named_args; the y descriptor's
    # base is (1, N) exactly when the launcher specialized BROADCAST_Y.
    broadcast_y = named_args["y_desc"].base.shape[0] == 1
    out = []
    for conf in configs:
        kw = conf.kwargs
        BM = kw["BLOCK_M"]
        BN = kw["BLOCK_N"]
        BK = kw["BLOCK_K"]
        s = kw["NUM_SMEM_BUFFERS"]
        t = kw["NUM_TMEM_BUFFERS"]
        g = kw["NUM_MMA_GROUPS"]
        subtile = kw["EPILOGUE_SUBTILE"]
        epi_bufs = kw["NUM_EPILOGUE_SMEM_BUFFERS"]
        ctas = kw["NUM_CTAS"]
        c_n = kw["CLUSTER_N"]
        if BM % g != 0:
            continue
        split = BM // g
        if split > 128 or split < 64:
            continue
        # Pair-CTA MMA needs M=128 per MMA group; pairs must stay in one
        # N-group, so GROUP_M must be a multiple of the cluster width.
        if ctas == 2 and split != 128:
            continue
        # (2,2) needs MMA pairs plus an even GROUP_M for M-pair grouping.
        if c_n == 2 and (ctas != 2 or kw["GROUP_M"] % 2 != 0):
            continue
        if kw["GROUP_M"] % ctas != 0:
            continue
        if BN % subtile != 0:
            continue
        if BK > K:
            continue
        if BM > M * 2 or BN > N * 2:
            continue
        smem_a = BM * BK * 2 * s
        smem_b = BK * (BN // ctas) * 2 * s
        smem_epilog = split * (BN // subtile) * 2 * epi_bufs
        y_bufs = 1 if broadcast_y else 2
        smem_y = BN * 2 if broadcast_y else split * (BN // subtile) * 2 * y_bufs
        # slice_empty rides named barriers (zero smem); only slice_full counts here.
        # Paired handoff (even epi/subtile) halves the slice_full barriers.
        # B's full barriers alias A's except on (2,2) clusters, and the pair
        # rendezvous barriers exist only there too (collaborative (2,1)
        # pairs multicast completion onto the leader instead).
        paired = (epi_bufs % 2 == 0) and (subtile % 2 == 0)
        slice_full_n = epi_bufs // 2 if paired else epi_bufs
        b_full_n = s if c_n == 2 else 0
        cta_n = s * g if ctas == 2 and c_n == 2 else 0
        smem_barriers = (s * g * 2 + b_full_n + t * g * 2 + 2 * y_bufs + slice_full_n + cta_n) * 8
        if smem_a + smem_b + smem_epilog + smem_y + smem_barriers > 232 * 1024:
            continue
        if BM * BN * 4 * t > 256 * 1024:
            continue
        out.append(conf)
    return out


def _ws_configs(broadcast_y):
    """Persistent WS space: (BM, BN, BK, smem_bufs, tmem_bufs, mma_groups, subtile, ctas, epi_bufs).

    ``NUM_TMEM_BUFFERS=2`` lets MMA run one tile ahead of the epilogue; the
    sliced epilogue keeps staging small so the wide tiles fit. The 1-D bias
    stages almost nothing, so it additionally gets a deeper wide-tile entry.
    2-CTA entries halve B's smem per CTA, unlocking deeper buffers on the wide
    tiles. The (smem_bufs, epi_bufs) grid on the wide winner-family tiles lets
    autotune size the buffering: TMEM is the binding constraint there, so smem
    has headroom for deeper pipes and the pruner trims what does not fit. The
    (y_evict, z_evict) anchors sweep the TMA eviction hints with and without
    ("") on the top 1- and 2-CTA tiles. The TMA block hook rides on each
    config: the autotuner only replays per-config hooks on cache-hit launches,
    and the descriptors are rebuilt fresh on every call. The KPerfAgent-tuned
    narrow and K-aware rows extend the same families; every entry runs the
    CLC launch except the (2,2) twins below.
    """
    cfgs = [
        (128, 256, 64, 2, 2, 1, 4, 1, 3),
        (128, 256, 64, 2, 1, 1, 4, 1, 3),
        (128, 128, 64, 3, 2, 1, 2, 1, 3),
        (128, 128, 64, 4, 2, 1, 2, 1, 3),
        (256, 128, 64, 2, 2, 2, 2, 1, 3),
        (256, 128, 64, 3, 1, 2, 2, 1, 3),
        (64, 128, 64, 3, 2, 1, 2, 1, 3),
        (64, 256, 64, 2, 2, 1, 4, 1, 3),
        (128, 64, 128, 3, 2, 1, 1, 1, 3),
        (64, 64, 64, 4, 2, 1, 1, 1, 3),
        (256, 256, 64, 2, 1, 2, 4, 1, 3),
        (128, 256, 64, 3, 2, 1, 4, 2, 3),
        (256, 256, 64, 2, 1, 2, 4, 2, 3),
        (256, 128, 64, 3, 2, 2, 2, 2, 3),
        (128, 128, 64, 4, 2, 1, 2, 2, 3),
    ]
    if broadcast_y:
        cfgs.append((128, 256, 64, 3, 2, 1, 4, 1, 3))
        # BK=128 on the wide winner-family tile (1-D only): halves the
        # K-step syncs (B_full/A_full waits + cta syncs, 12 -> 6 steps
        # at K=768) at 2x A/B smem; the pruner keeps s<=3 (2-CTA) and
        # s=2 (1-CTA). 1-D's tiny whole-row y staging leaves the headroom
        # 2-D lacks.
        for s in (2, 3):
            for ctas in (1, 2):
                for epi in (2, 4):
                    cfgs.append((128, 256, 128, s, 2, 1, 8, ctas, epi))
    # KPerfAgent-tuned narrow rows: deep rings on small tiles for skinny or
    # few-tile problems (both bias ranks).
    cfgs.extend([
        (64, 128, 64, 8, 2, 1, 2, 1, 2),
        (128, 64, 64, 8, 2, 1, 2, 1, 2),
        (64, 64, 64, 12, 2, 1, 1, 1, 2),
        (64, 32, 128, 8, 2, 1, 1, 1, 2),
    ])
    if broadcast_y:
        # K-aware 1- and 2-CTA wide/tall/mid rows (1-D y staging leaves room
        # for the deeper pipes).
        cfgs.extend([
            (256, 256, 64, 3, 1, 2, 8, 2, 4),
            (128, 256, 64, 6, 2, 1, 8, 2, 4),
            (256, 256, 64, 3, 1, 2, 8, 1, 4),
            (256, 256, 64, 2, 1, 2, 4, 1, 3),
            (128, 256, 64, 4, 2, 1, 8, 1, 4),
            (128, 128, 64, 6, 2, 1, 2, 1, 2),
        ])
    else:
        cfgs.extend([
            (256, 256, 64, 4, 1, 2, 8, 2, 2),
            (256, 256, 64, 2, 1, 2, 4, 2, 4),
            (128, 256, 64, 4, 2, 1, 4, 2, 4),
            (256, 256, 64, 3, 1, 2, 8, 1, 2),
            (256, 256, 64, 2, 1, 2, 4, 1, 4),
            (128, 256, 64, 4, 2, 1, 8, 1, 2),
            (128, 128, 64, 6, 2, 1, 4, 1, 2),
        ])
    for bm, bn, bk, t, g, sub, ctas in [
        (256, 256, 64, 1, 2, 4, 2),
        (256, 256, 64, 1, 2, 4, 1),
        (128, 256, 64, 2, 1, 4, 2),
        (128, 256, 64, 2, 1, 4, 1),
        (256, 256, 64, 1, 2, 8, 2),
        (256, 256, 64, 1, 2, 8, 1),
        (128, 256, 64, 2, 1, 8, 2),
        (128, 256, 64, 2, 1, 8, 1),
    ]:
        for s in (2, 3, 4, 5):
            for epi in (2, 3, 4, 5, 6):
                cfgs.append((bm, bn, bk, s, t, g, sub, ctas, epi))
    # t=2 TMEM-overlap probe family (2-CTA, g=1): narrower N tiles leave
    # TMEM/smem headroom for deeper pipes, and epi 7-8 plus sub2 extend
    # the epilogue count/size range past the main grid. The pruner trims
    # what does not fit; paired handoff uses epi//2 named slice_empty IDs
    # (epi <= 8 -> 3..6), legacy odd-epi rows use up to 3..10.
    for bm, bn, bk, t, g, sub, ctas in [
        (128, 128, 64, 2, 1, 2, 2),
        (128, 128, 64, 2, 1, 4, 2),
        (128, 256, 64, 2, 1, 2, 2),
    ]:
        for s in (2, 3, 4, 5):
            for epi in (2, 3, 4, 5, 6, 7, 8):
                cfgs.append((bm, bn, bk, s, t, g, sub, ctas, epi))
    cfgs = list(dict.fromkeys(cfgs))
    y_default = "evict_last" if broadcast_y else "evict_first"
    out = [(bm, bn, bk, s, t, g, sub, ctas, epi, y_default, "evict_first")
           for bm, bn, bk, s, t, g, sub, ctas, epi in cfgs]
    for bm, bn, bk, s, t, g, sub, ctas in [
        (256, 256, 64, 2, 1, 2, 4, 2),
        (128, 256, 64, 3, 2, 1, 4, 2),
        (256, 256, 64, 2, 1, 2, 4, 1),
        (128, 256, 64, 2, 2, 1, 4, 1),
    ]:
        for y_evict, z_evict in [("", "evict_first"), (y_default, ""), ("", "")]:
            out.append((bm, bn, bk, s, t, g, sub, ctas, 3, y_evict, z_evict))
    out = list(dict.fromkeys(out))
    out1 = [
        _ws_config(bm, bn, bk, 8, s, t, g, sub, epi_smem_bufs=epi, num_ctas=ctas, y_evict=y_evict, z_evict=z_evict)
        for bm, bn, bk, s, t, g, sub, ctas, epi, y_evict, z_evict in out
    ]
    # (2,2) cluster + A-multicast twins of the winner-family tile: same
    # (s, epi) grid on sub8 plus sub4 anchors. They stay on static
    # scheduling (no CLC): the 4-CTA rendezvous predates CLC validation.
    out2 = []
    for s in (2, 3, 4, 5):
        for epi in (2, 3, 4, 5, 6):
            out2.append(
                _ws_config(256, 256, 64, 8, s, 1, 2, 8, epi_smem_bufs=epi, num_ctas=2, cluster_n=2, y_evict=y_default,
                           z_evict="evict_first", use_clc=False))
            if s in (2, 3) and epi in (3, 5):
                out2.append(
                    _ws_config(256, 256, 64, 8, s, 1, 2, 4, epi_smem_bufs=epi, num_ctas=2, cluster_n=2,
                               y_evict=y_default, z_evict="evict_first", use_clc=False))
    return out1 + out2


def _ws_smoke_configs(broadcast_y):
    y_evict = "evict_last" if broadcast_y else "evict_first"
    # All rows keep even subtile/epi so space="smoke" always exercises the
    # paired slice->store handoff (odd epi/subtile keeps the legacy path,
    # covered by the heuristic-space tests). The static twin and the (2,2)
    # row keep the non-CLC schedules compiling under test.
    cfgs = [
        (128, 128, 64, 2, 1, 1, 2, 2, 1, 1, True),
        (64, 128, 64, 2, 1, 1, 2, 2, 1, 1, True),
        (128, 128, 64, 2, 1, 1, 2, 2, 1, 1, False),
        (256, 256, 64, 2, 1, 2, 4, 2, 2, 2, False),
    ]
    return [
        _ws_config(
            bm,
            bn,
            bk,
            8,
            s,
            t,
            g,
            sub,
            epi,
            num_ctas=ctas,
            cluster_n=c_n,
            y_evict=y_evict,
            use_clc=clc,
        ) for bm, bn, bk, s, t, g, sub, epi, ctas, c_n, clc in cfgs
    ]


def _ws_2cta_tile_config(M, N, K, broadcast_y, num_sms):
    # 2-CTA pairs sit along M and split B's N-range, so each CTA stages only
    # BLOCK_N // 2 columns of B and the pair issues one cta_group::2 MMA.
    if broadcast_y:
        wide = (256, 256, 64, 3, 1, 2, 8, 4, 2)
        tall = (128, 256, 64, 6, 2, 1, 8, 4, 2)
    else:
        wide = (256, 256, 64, 4, 1, 2, 8, 2, 2) if K >= 4096 else (256, 256, 64, 2, 1, 2, 4, 4, 2)
        tall = (128, 256, 64, 4, 2, 1, 4, 4, 2)
    # The wide tile's single TMEM buffer exposes the epilogue between tiles,
    # which short K cannot amortize; the tall tile double-buffers TMEM.
    families = (
        (256, 1.0 if K >= 2048 else 0.94, wide),
        (128, 0.97, tall),
    )
    best_cost, best_cfg = None, None
    for bm, eff, cfg in families:
        tiles = triton.cdiv(triton.cdiv(M, bm), 2) * 2 * triton.cdiv(N, 256)
        cost = triton.cdiv(tiles, num_sms) * bm * 256 / eff
        if best_cost is None or cost < best_cost:
            best_cost, best_cfg = cost, cfg
    return best_cfg


# Narrow tiles for problems that cannot fill the machine with 128x128 tiles
# (skinny M or N, or few output tiles): each stage is only 16-24KB, so the
# operand ring is deepened to back-fill the smem budget, and the extra CTAs
# from the smaller tile spread the B (or A) stream over more SMs.
# Efficiencies are measured B200 throughputs on the same 256x256 = 1.0 scale.
# Rows are (bm, bn, efficiency, config) with config in
# (bm, bn, bk, s, t, g, sub, epi, ctas) order.
_WS_NARROW_FAMILIES = (
    (64, 128, 0.56, (64, 128, 64, 8, 2, 1, 2, 2, 1)),
    (128, 64, 0.56, (128, 64, 64, 8, 2, 1, 2, 2, 1)),
    (64, 64, 0.49, (64, 64, 64, 12, 2, 1, 1, 2, 1)),
    (64, 32, 0.40, (64, 32, 128, 8, 2, 1, 1, 2, 1)),
)


def _ws_pick_family(families, M, N, num_sms):
    # Trade per-tile intensity against wave quantization; ties keep the
    # earlier (larger) tile.
    best_cost, best_cfg = None, None
    for bm, bn, eff, cfg in families:
        waves = triton.cdiv(triton.cdiv(M, bm) * triton.cdiv(N, bn), num_sms)
        cost = waves * bm * bn / eff
        if best_cost is None or cost < best_cost:
            best_cost, best_cfg = cost, cfg
    return best_cfg


def _ws_tile_config(M, N, K, broadcast_y, num_sms):
    # (BLOCK_M, BLOCK_N, BLOCK_K, NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS,
    #  NUM_MMA_GROUPS, EPILOGUE_SUBTILE, NUM_EPILOGUE_SMEM_BUFFERS, NUM_CTAS)
    # for 16-bit operands; every entry keeps A/B stages + epilogue staging +
    # bias tile under B200's 232KB smem and the accumulators within 512 TMEM
    # columns.
    if M <= 64 or N <= 64:
        # A 128-row (column) tile would be mostly padding along the skinny dim.
        families = tuple(f for f in _WS_NARROW_FAMILIES if (f[0] == 64 or M > 64) and (f[1] <= 64 or N > 64))
        return _ws_pick_family(families, M, N, num_sms)
    # With at least two waves of 128x256 tiles the cluster launch and pair
    # handshake amortize; smaller problems stay on single-CTA tiles.
    if triton.cdiv(M, 128) * triton.cdiv(N, 256) >= 2 * num_sms:
        return _ws_2cta_tile_config(M, N, K, broadcast_y, num_sms)
    # Long K amortizes the finer epilogue in exchange for a deeper operand pipeline.
    if K >= 4096:
        wide = (256, 256, 64, 3, 1, 2, 8, 4, 1) if broadcast_y else (256, 256, 64, 3, 1, 2, 8, 2, 1)
    else:
        wide = (256, 256, 64, 2, 1, 2, 4, 3, 1) if broadcast_y else (256, 256, 64, 2, 1, 2, 4, 4, 1)
    families = (
        (256, 256, 1.0, wide),
        (128, 256, 0.86, (128, 256, 64, 4, 2, 1, 8, 4, 1) if broadcast_y else (128, 256, 64, 4, 2, 1, 8, 2, 1)),
        (128, 128, 0.72, (128, 128, 64, 6, 2, 1, 2, 2, 1) if broadcast_y else (128, 128, 64, 6, 2, 1, 4, 2, 1)),
    ) + _WS_NARROW_FAMILIES[1:3]
    return _ws_pick_family(families, M, N, num_sms)


def _ws_heuristic_config(M, N, K, broadcast_y):
    """Choose one warp-specialized configuration without runtime autotuning.

    The KPerfAgent-tuned cost-model pick (always a CLC launch). The op only
    serves 16-bit dtypes (catalog-gated), matching the tuned rows' smem.
    """
    y_evict = "evict_last" if broadcast_y else "evict_first"
    bm, bn, bk, s, t, g, sub, epi, ctas = _ws_tile_config(M, N, K, broadcast_y, _get_num_sms())
    return [_ws_config(
        bm,
        bn,
        bk,
        8,
        s,
        t,
        g,
        sub,
        epi_smem_bufs=epi,
        num_ctas=ctas,
        y_evict=y_evict,
    )]


def _direct_configs():
    """Port of the sm100-filtered CUDA branch of the baseline's ``get_mm_configs``.

    The baseline drops its two ``num_warps=2`` configs on sm100+, so the
    space below is that filtered set plus the sm100-only 4-warp config.
    """
    return [
        _config(128, 256, 64, 8, 8, 3),
        _config(64, 256, 32, 8, 4, 4),
        _config(128, 128, 32, 8, 4, 4),
        _config(128, 64, 32, 8, 4, 4),
        _config(64, 128, 32, 8, 4, 4),
        _config(128, 32, 32, 8, 4, 4),
        _config(128, 256, 64, 8, 4, 3),
    ]


CONFIGS = _direct_configs


def _direct_smoke_configs():
    return [_config(128, 128, 32, 8, 4, 4), _config(64, 128, 32, 8, 4, 4)]


SMOKE_CONFIGS = _direct_smoke_configs


def heuristic_config(M, N, K):
    """Choose one direct-load configuration without runtime autotuning."""
    del K
    # Skinny-M: short row tiles keep masked rows low while reusing B.
    if M <= 64:
        return [_config(64, 128, 32, 8, 4, 4)]
    # Skinny-N: narrow column tiles keep masked columns low.
    if N <= 64:
        return [_config(128, 64, 32, 8, 4, 4)]
    # Large squares have enough tiles for the wide high-reuse shape.
    if M >= 1024 and N >= 1024:
        return [_config(128, 256, 64, 8, 8, 3)]
    # Tall-M: the wide tile's reuse beats square tiles (+6-26% at
    # (16384, 512, 1024) and (41000, 768, 768)).
    if M >= 4096:
        return [_config(128, 256, 64, 8, 8, 3)]
    return [_config(128, 128, 32, 8, 4, 4)]


def _space_configs(space, shape, configs_fn, smoke_fn, heuristic_fn):
    if space == "heuristic":
        return heuristic_fn(*shape) or smoke_fn()
    if space == "full":
        return configs_fn()
    if space == "smoke":
        return smoke_fn()
    raise ValueError(f"Unknown sm100 addmm search space: {space!r}")


@functools.lru_cache(maxsize=None)
def _tuned_ws(space, shape, broadcast_y):
    """Autotuned warp-specialized kernel per search space and bias rank."""
    configs = _space_configs(
        space,
        shape,
        lambda: _ws_configs(broadcast_y),
        lambda: _ws_smoke_configs(broadcast_y),
        lambda M, N, K: _ws_heuristic_config(M, N, K, broadcast_y),
    )
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K", "BROADCAST_Y"],
        prune_configs_by={"early_config_prune": _ws_prune_configs},
    )(addmm_kernel_tma_ws)


@functools.lru_cache(maxsize=None)
def _tuned_direct(space, shape=None):
    """Autotuned direct-load kernel per search space (non-TMA fallback)."""
    configs = _space_configs(space, shape, _direct_configs, _direct_smoke_configs, heuristic_config)
    return triton.autotune(configs=configs, key=["M", "N", "K", "BROADCAST_Y"])(addmm_kernel_direct)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    M, K = a.shape
    N = b.shape[1]
    if out is not None:
        if out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({M}, {N}) on A's device")
    return M, N, K


def _bias_strides(bias, M, N, a):
    if bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if bias.ndim == 1:
        if bias.shape[0] != N:
            raise ValueError(f"1-D addmm input must have shape ({N},), got {tuple(bias.shape)}")
        return 0, bias.stride(0)
    if bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N):
        return (0 if bias.shape[0] == 1 else bias.stride(0), 0 if bias.shape[1] == 1 else bias.stride(1))
    raise ValueError(f"addmm input with shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})")


# Small-output envelope: below 16M outputs with K <= 4096 the persistent
# WS grid cannot fill the GPU, so small TMA outputs run on the
# lower-overhead direct-load kernel instead (2.8x at 2048^3 vs WS).
_TMA_SMALL_MAX = 16 * 1024 * 1024
_TMA_SMALL_MAX_K = 4096


def _tma_eligible(a, b, bias_2d, out, N, K):
    """Whether the warp-specialized TMA path can serve these operands.

    TMA needs 16-element-aligned K/N plus 16-byte-aligned bases and unit inner
    strides; anything else runs on the direct-load kernel instead.
    """
    if K % 16 != 0 or N % 16 != 0:
        return False
    if a.data_ptr() % 16 != 0 or a.stride(1) != 1:
        return False
    # TensorDescriptor requires a contiguous inner dim plus 16-byte-aligned
    # outer strides; anything else (e.g. column-major B) runs on direct loads.
    if b.data_ptr() % 16 != 0 or b.stride(1) != 1:
        return False
    if bias_2d.data_ptr() % 16 != 0 or bias_2d.stride(1) != 1:
        return False
    return out.data_ptr() % 16 == 0


def _launch_ws(bias_2d, a, b, c, M, N, K, broadcast_y, space):
    dummy_block = [1, 1]
    x_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    w_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    y_desc = TensorDescriptor(bias_2d, bias_2d.shape, bias_2d.stride(), dummy_block)
    z_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    num_sms = _get_num_sms()

    def grid(meta):
        tiles_m = triton.cdiv(M, meta["BLOCK_M"])
        if meta["NUM_CTAS"] == 2:
            # Pad M-tiles to whole clusters so no CTA loses its pair (an
            # unpaired CTA would deadlock in the pair rendezvous). Padding
            # tiles fall fully OOB and their TMA stores clip to nothing.
            tiles_m = (tiles_m + 1) // 2 * 2
        tiles_n = triton.cdiv(N, meta["BLOCK_N"])
        if meta["CLUSTER_N"] == 2:
            # Pad N-tiles so (2,2) clusters cover whole 2x2 blocks.
            tiles_n = (tiles_n + 1) // 2 * 2
        tiles = tiles_m * tiles_n
        if meta["USE_CLC"]:
            # CLC launches one CTA per tile; running CTAs cancel unlaunched
            # ones to claim their tiles.
            return (tiles, )
        return (min(tiles, num_sms), )

    kernel = _tuned_ws(space, (M, N, K) if space == "heuristic" else None, broadcast_y)
    kernel[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        NUM_SMS=num_sms,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=broadcast_y,
    )
    return c


def _launch_direct(bias, a, b, c, M, N, K, stride_ym, stride_yn, space):

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    kernel = _tuned_direct(space, (M, N, K) if space == "heuristic" else None)
    kernel[grid](
        a,
        b,
        bias,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_ym,
        stride_yn,
        c.stride(0),
        c.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=bias.ndim == 1,
    )
    return c


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Fused ``input + a @ b`` on Blackwell.

    ``input`` may be ``(N,)`` or two-dimensional and broadcastable to the
    ``(M, N)`` result. Matrix and input scale factors are both one. This op is
    currently forward-only.

    TMA-capable inputs run on the warp-specialized kernel, except small
    outputs, which run faster on the direct-load kernel. Anything else
    (misaligned bases, exotic strides, K/N not multiples of 16, or
    broadcastable-but-neither-``(N,)``-nor-``(M, N)`` biases) also falls
    back to the direct-load kernel.

    `space` selects the search space -- "heuristic" (one config) for
    correctness, "full" for perf, "smoke" for path coverage.
    """
    M, N, K = _validate_operands(a, b, out)
    stride_ym, stride_yn = _bias_strides(input, M, N, a)
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c

    if input.ndim == 1 or input.shape[0] == 1:
        bias_2d = input.reshape(1, N)
        broadcast_y = True
    elif tuple(input.shape) == (M, N):
        bias_2d = input
        broadcast_y = False
    else:
        # (M, 1) and friends: TMA boxes cannot express them; use direct loads.
        return _launch_direct(input, a, b, c, M, N, K, stride_ym, stride_yn, space)

    tma_eligible = _tma_eligible(a, b, bias_2d, c, N, K)
    if tma_eligible and K != 0 and K <= _TMA_SMALL_MAX_K and M * N < _TMA_SMALL_MAX:
        return _launch_direct(input, a, b, c, M, N, K, stride_ym, stride_yn, space)
    if tma_eligible:
        return _launch_ws(bias_2d, a, b, c, M, N, K, broadcast_y, space)
    return _launch_direct(input, a, b, c, M, N, K, stride_ym, stride_yn, space)
