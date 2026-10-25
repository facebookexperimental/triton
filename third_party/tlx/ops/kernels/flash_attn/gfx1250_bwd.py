"""Delayed-dQ BF16 D128 attention backward for MI450 (gfx1250).

Port of PR #2709's two-slot dS bridge to wave32 WMMA. A workgroup owns
one KV tile and loops over all Q tiles of its query-head group. Current dK
and previous dQ have independent accumulator chains. The final dQ tile is
drained explicitly. dK/dV stay resident across both Q tiles and GQA heads.
"""

import dataclasses

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from .gfx950_bwd import _run_bwd_preprocess, _run_bwd_zero_dq


@triton.constexpr_function
def _ds_shared_layout(bm, bn):
    # Match the score WMMA's per-lane Q registers so publishing dS uses
    # vector LDS stores. Its inverse view supplies the delayed-dQ transpose.
    q_bits = [1, 2, 4] + [1 << bit for bit in range(4, bm.bit_length() - 1)]
    q_bases = [[bit, 0] for bit in q_bits]
    lane_bases = [[0, bit] for bit in (1, 2, 4, 8)] + [[8, 0]]
    kv_bases = [[0, 1 << bit] for bit in range(4, bn.bit_length() - 1)]
    bases = q_bases + lane_bases + kv_bases
    return tlx.padded_shared_layout_encoding.with_bases([(512, 16)], bases, [bm, bn])


@dataclasses.dataclass(frozen=True)
class BackwardConfig:
    """Measured policy and independently selectable scheduling experiments.

    resident_kv: 0 streams K/V from LDS; 1 keeps score K/V; 2 also keeps
    the leading 3/4 of dQ's K; 3 keeps V and 7/8 of dQ's K (3/4 for BN128);
    4 keeps V alone. Physical VGPR allocation remains compiler-managed.
    outer_m stages several block_m compute phases in one Q/dO load tile.
    regalloc enables the existing three per-kernel allocation controls.
    """
    block_m: int = 64
    block_n: int = 256
    num_warps: int = 8
    use_tdm: bool = True
    stages: int = 2
    delayed: bool = True
    unroll: int = 1
    fragment_fences: bool = False
    fragment_fence_mask: int = 0
    fragmented: bool = False
    resident_kv: int = 0
    ordered: bool = False
    packed_dq: bool = False
    rematerialize: bool = False
    outer_m: int = 0
    async_stats: bool = False
    queue_wmma: bool = False
    regalloc: bool = False
    q_splits: int = 1
    waves_per_eu: int = 0
    sink_insts_to_avoid_spills: bool | None = None
    reverse_local_assignment: bool | None = None
    regclass_priority_trumps_globalness: bool | None = None
    preprocess_block: int = 64
    convert_block: int = 1024
    reuse_q: bool = False
    pingpong: bool = False
    spread_dq: bool = False
    log2_lse: bool = False
    cache_score_k: bool = False
    # KV tiles on grid X instead of KV heads.
    tile_major: bool = False
    # Groups of up to 16 KV tiles share one shader engine and a private dQ slice.
    cluster_dq: bool = False
    # Split K/V by rows and dS by columns into two LDS partitions. Opposite
    # SIMD pairs walk opposite pieces in the delayed-dQ product.
    partition_lds: bool = False
    # Score/dP front as fenced K32 regions: 1 loads each band in its region,
    # 2 also prefetches the next band's operands.
    front_regions: int = 0
    # Fenced dK/dQ bridge regions: 1 keeps the original reads, 2 prefetches
    # the next K, dS and Q blocks under the current region. Mode 3 seeds
    # the first Q panel after dV, before the existing front rendezvous.
    # Mode 4 also looks ahead one reused Q panel in the dK/dQ bridge.
    bridge_regions: int = 0
    # Separate four-wave schedule: resident V/score K, BM32 phases and a
    # lagged dK/dQ bridge. Never selected without paired performance gates.
    occ1: bool = False
    occ1_k_bands: int = 0
    occ1_prefetch: bool = True
    occ1_sched: bool = False
    # Experimental WMMA order and prefetch placement; gated independently.
    occ1_dq_first: bool = False
    occ1_load_after_dk: bool = False
    phase_iglp: int = -1
    # Split only the final dispatch-round KV owners; one preserves legacy code.
    tail_splits: int = 1


def _select_config_base(n, b=None, hq=None, hk=None, *, num_cus=None, packed_dq=True):
    """Select the measured MI450 policy from the sequence length and grid shape.

    The n-only call preserves the one-owner policy. Shape-aware callers may
    supply num_cus for an offline/deterministic selection calculation.
    """
    config = BackwardConfig(fragmented=True, ordered=True, packed_dq=packed_dq, async_stats=True, rematerialize=True,
                            regalloc=True)
    if n % 256:
        config = dataclasses.replace(config, block_m=32, block_n=128)
    if b is None or hq is None or hk is None:
        return config
    if min(b, hq, hk) < 1 or hq % hk:
        raise ValueError("invalid batch/head counts")
    if num_cus is None:
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if num_cus < 1:
        raise ValueError("num_cus must be positive")
    owners = b * hk * (n // config.block_n)
    pingpong = dict(pingpong=True, regalloc=False, convert_block=4096)
    if hq == hk and config.block_n == 256 and 1024 <= n <= 8192 and owners >= num_cus:
        # MHA has one query head per KV tile: split epilogues dominate short
        # grids. BM32 leaves room for the full score-K cache, saving 16 LDS
        # reads per step. The 1K-8K sweep favored one split; 16K favors BM64.
        # LDS partitions remove cross-SIMD-pair dS/K conflicts in delayed dQ.
        # The cached score K already replaces the fenced front.
        # FP32 short MHA uses the early first-Q read and Q-only lookahead.
        # Paired checks cover multiple batch and head counts.
        bridge = 0
        if not packed_dq and hq == hk and n <= 4096:
            bridge = 4
        # Short sequences use one SE-local cluster per head; 8K uses two
        # clusters with private dQ slices. Both precisions clear the gates.
        cluster_dq = n in (1024, 2048, 4096, 8192)
        return dataclasses.replace(config, block_m=32, q_splits=1, reuse_q=True, cache_score_k=True, partition_lds=True,
                                   bridge_regions=bridge, tile_major=cluster_dq, cluster_dq=cluster_dq, **pingpong)
    best, best_waves = 1, triton.cdiv(owners, num_cus)
    steps = (hq // hk) * n // config.block_m
    # Ping-pong needs at least two steps in every split (see fa_backward).
    max_splits = min(4, steps // 2) if config.block_n == 256 else min(4, steps)
    for splits in range(2, max_splits + 1):
        waves = triton.cdiv(splits * owners, num_cus)
        if waves * best < best_waves * splits:
            best, best_waves = splits, waves
    if config.block_n == 256:
        # Ping-pong was faster on every measured BN256 shape (GQA 1/4/8,
        # N 768-16384). Its one-split epilogue spills, so grids that are
        # already full still take three splits.
        # FP32 benefits from constant-offset atomics spread through the dK
        # bridge. The split/tile sweep retained the packed grid policy.
        # LDS partitions plus the fenced front won on every measured BN256
        # shape. Long packed GQA also gains from the prefetching front and
        # bridge and from KV tiles on grid X; FP32 did not clear the margin.
        long_packed_gqa = packed_dq and hq > hk and n >= 16384
        regions = 2 if long_packed_gqa else 1
        # Q splits own disjoint rows. Long sequences give each cluster
        # group its own dQ slice; dK/dV partials remain device scoped.
        # MHA8K's cached BM32 policy is selected above; MHA16K uses BM64.
        cluster_dq = (hq > hk and n in (1024, 2048, 4096, 8192, 16384)) or (hq == hk and n == 16384)
        return dataclasses.replace(config, q_splits=best if best > 1 else min(3, max_splits), reuse_q=True,
                                   spread_dq=not packed_dq, partition_lds=True, front_regions=regions,
                                   bridge_regions=2 if long_packed_gqa else 0, tile_major=long_packed_gqa or cluster_dq,
                                   cluster_dq=cluster_dq, **pingpong)
    # BN128 ping-pong keeps native row-scattered atomics for split partials;
    # extra splits cost more than the grid tail they remove. Ping-pong was
    # slower at N=640 and faster from N=896.
    config = dataclasses.replace(config, reuse_q=True)
    return dataclasses.replace(config, **pingpong) if n >= 896 else config


def select_config(n, b=None, hq=None, hk=None, *, num_cus=None, packed_dq=True):
    """Select a measured eight-wave or occupancy-one backward policy.

    OCC1 coverage is keyed by attention class, sequence length and precision.
    A class is enabled only where every sampled batch clears the paired gate.
    The n-only API and unmeasured branch conditions retain the base policy.
    """
    config = _select_config_base(n, b, hq, hk, num_cus=num_cus, packed_dq=packed_dq)
    if b is None or hq is None or hk is None:
        return config
    if hq == hk:
        # Short MHA must reach the base selector's saturated BM32 branch.
        # The 16K MHA branch uses clustered BM64 in the committed policy.
        if not (config.cache_score_k or n == 16384 and config.cluster_dq):
            return config
        lengths = (1024, 2048, 4096, 8192, 16384)
    else:
        if not config.cluster_dq or config.block_n != 256:
            return config
        lengths = (2048, 4096, 8192, 16384)
        if n == 1024 and hq == 8 * hk and config.q_splits == 3:
            lengths = (1024, ) + lengths
    if n not in lengths:
        return config
    # Single-split OCC1 requires at least 8/3 rounds where the base path splits Q.
    if num_cus is None:
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    q_splits = 1
    if n == 1024 and hq > hk:
        # GROUP=8 at B=8/16: nine paired rounds favored three-split OCC1 by
        # 1.5-3.0% over the three-split eight-wave path, in both precisions.
        # Keep at least four rounds and the measured base split policy; the
        # smaller grids and other attention groups retain their base path.
        if 3 * b * hk * (n // 256) < 4 * num_cus:
            return config
        q_splits = 3
    elif config.q_splits > 1 and 3 * b * hk * (n // 256) < 8 * num_cus:
        return config
    # Paired measurements favored splitting only the incomplete final
    # round for these GROUP=8 grids. Prefix CTAs retain direct BF16 stores.
    # Keep the policy on the validated 192-CU shapes and precisions.
    tail_splits = 3 if num_cus == 192 and (b, hq, hk, n, packed_dq) in ((8, 64, 8, 2048, False), (8, 64, 8, 2048, True),
                                                                        (16, 64, 8, 2048, False),
                                                                        (16, 64, 8, 2048, True),
                                                                        (8, 64, 8, 4096, False),
                                                                        (8, 64, 8, 4096, True)) else 1
    # Nine paired rounds favored these additional dispatch-tail policies.
    # Keep unmeasured shapes and CU counts on their committed configurations.
    if num_cus == 192 and (b, hq, hk, n, packed_dq) in ((16, 16, 16, 4096, True), ):
        tail_splits = 2
    if num_cus == 192 and (b, hq, hk, n, packed_dq) in ((16, 16, 16, 4096, False), (16, 64, 8, 4096, False),
                                                        (16, 64, 8, 4096, True), (8, 64, 8, 8192, False),
                                                        (8, 64, 8, 8192, True), (16, 64, 8, 8192, False),
                                                        (16, 64, 8, 8192, True), (8, 64, 8, 16384, False),
                                                        (8, 64, 8, 16384, True)):
        tail_splits = 3
    # Split only the final GQA 1K round so prefix owners keep direct BF16
    # stores and convert sums compact tail partials. Nine paired rounds favored
    # these 192-CU grids; other GROUP=8 shapes retain uniform three-way splits.
    if num_cus == 192 and (b, hq, hk, n) in ((8, 64, 8, 1024), (16, 64, 8, 1024)):
        q_splits, tail_splits = 1, 3
    # The measured two-way MHA 1K tail is FP32-only; packed keeps one split.
    if num_cus == 192 and (b, hq, hk, n, packed_dq) == (16, 16, 16, 1024, False):
        tail_splits = 2
    # WMMA queuing (HW_REG_WAVE_SCHED_MODE bit 2) won 0.4-2.0% on 24 of the 26
    # OCC1 points and lost on none.
    return BackwardConfig(block_m=32, num_warps=4, stages=3, ordered=True, packed_dq=packed_dq, outer_m=64,
                          async_stats=True, queue_wmma=True, regalloc=True, q_splits=q_splits, waves_per_eu=1,
                          convert_block=8192 if q_splits > 1 and packed_dq else 4096, tile_major=True, cluster_dq=True,
                          occ1=True, occ1_k_bands=2, tail_splits=tail_splits, occ1_load_after_dk=True)


def _compiler_options(config, n=None):
    options = dict(num_warps=config.num_warps, num_stages=1, enable_fp_fusion=True, waves_per_eu=config.waves_per_eu)
    for name in ("sink_insts_to_avoid_spills", "reverse_local_assignment", "regclass_priority_trumps_globalness"):
        value = getattr(config, name)
        options[name] = config.regalloc if value is None else value
    if config.queue_wmma:
        options["enable_wmma_queuing"] = True
    if config.cluster_dq:
        # Grid X holds one split's KV tiles; each cluster group owns a dQ slice.
        nt = n // 256 if n is not None and n % 256 == 0 else 0
        if (not config.tile_major or config.block_n != 256 or not (nt in (2, 4, 8, 16) or nt > 0 and nt % 16 == 0)):
            raise ValueError("cluster_dq requires tile_major, BN256 and NT in {2, 4, 8, 16} or a multiple of 16")
        options["ctas_per_cga"] = (min(nt, 16), 1, 1)
    return options


def _dq_slices(config, n):
    return max(1, n // config.block_n // 16) if config.cluster_dq else 1


@triton.jit
def _issue_tile(Q, DO, q_desc, do_desc, qb, dob, step, N: tl.constexpr, BM: tl.constexpr, STAGES: tl.constexpr,
                TDM: tl.constexpr, lse_desc=None, delta_desc=None, lseb=None, db=None,
                ASYNC_STATS: tl.constexpr = False, PINGPONG: tl.constexpr = False):
    slot = step % STAGES
    if PINGPONG:
        # Waves 0..3 own every TDM request. With the half-step phase shift,
        # the trailing group has not reached its next issue when the leading
        # group starts consuming that tile. Disjoint hints split the two
        # requests within the leading group, keeping their completion local.
        q_next = tlx.update_tensor_descriptor(q_desc, add_offsets=[step * BM, 0], clamp_bounds=False)
        do_next = tlx.update_tensor_descriptor(do_desc, add_offsets=[step * BM, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load_fused([(q_next, tlx.local_view(qb, slot), 0b0011),
                                             (do_next, tlx.local_view(dob, slot), 0b1100)])
    elif TDM:
        tlx.async_amd_descriptor_load(q_desc, tlx.local_view(qb, slot), [step * BM, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load(do_desc, tlx.local_view(dob, slot), [step * BM, 0], clamp_bounds=False)
    else:
        offsets = (step * BM + tl.arange(0, BM))[:, None] * 128 + tl.arange(0, 128)[None, :]
        tlx.async_load(Q + offsets, tlx.local_view(qb, slot))
        tlx.async_load(DO + offsets, tlx.local_view(dob, slot))
        tlx.async_load_commit_group()
    if ASYNC_STATS:
        if PINGPONG:
            lse_next = tlx.update_tensor_descriptor(lse_desc, add_offsets=[0, step * BM], clamp_bounds=False)
            delta_next = tlx.update_tensor_descriptor(delta_desc, add_offsets=[0, step * BM], clamp_bounds=False)
            tlx.async_amd_descriptor_load_fused([(lse_next, tlx.local_view(lseb, slot), 0b0001),
                                                 (delta_next, tlx.local_view(db, slot), 0b0010)])
        else:
            tlx.async_amd_descriptor_load(lse_desc, tlx.local_view(lseb, slot), [0, step * BM], clamp_bounds=False)
            tlx.async_amd_descriptor_load(delta_desc, tlx.local_view(db, slot), [0, step * BM], clamp_bounds=False)


@triton.jit
def _wmma(a, b, acc, ORDERED: tl.constexpr):
    # Matrix-only compiler scheduling boundaries preserve native WMMA
    # alternation while allowing LDS/global loads to overlap the fragments.
    # No workgroup barrier or hardware completion wait is emitted here.
    if ORDERED:
        tlx.amd_sched_barrier(0xFF6)
    result = tl.dot(a, b, acc)
    if ORDERED:
        tlx.amd_sched_barrier(0xFF6)
    return result


@triton.jit
def _front_regions(kb, qb, ACC: tl.constexpr, LHS: tl.constexpr, RHS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   PREFETCH: tl.constexpr):
    # One fenced region per K32 band of the head dimension. With PREFETCH,
    # the next band's reads are issued under the current band's WMMAs.
    a = tlx.local_load(tlx.local_slice(kb, [0, 0], [BN, 32]), layout=LHS, relaxed=True)
    q = tlx.local_load(tlx.local_slice(qb, [0, 0], [32, BM]), layout=RHS, relaxed=True)
    result = tlx.zeros((BN, BM), tl.float32, layout=ACC)
    for band in tl.static_range(4):
        tlx.amd_sched_barrier(0)
        if PREFETCH and band < 3:
            next_a = tlx.local_load(tlx.local_slice(kb, [0, (band + 1) * 32], [BN, 32]), layout=LHS, relaxed=True)
            next_q = tlx.local_load(tlx.local_slice(qb, [(band + 1) * 32, 0], [32, BM]), layout=RHS, relaxed=True)
        result = tl.dot(a, q, result)
        if band < 3:
            if PREFETCH:
                a = next_a
                q = next_q
            else:
                a = tlx.local_load(tlx.local_slice(kb, [0, (band + 1) * 32], [BN, 32]), layout=LHS, relaxed=True)
                q = tlx.local_load(tlx.local_slice(qb, [(band + 1) * 32, 0], [32, BM]), layout=RHS, relaxed=True)
    tlx.amd_sched_barrier(0)
    return result


@triton.constexpr_function
def _dq_register_bits(bm, mma):
    row_warps = sum(16 * basis[0] for basis in mma.warp_bases)
    col_warps = sum(16 * basis[1] for basis in mma.warp_bases)
    # The per-thread base swaps column 16 with lane-owned row 8. A register-
    # owned column 16 would need the swap in its delta, so reject it.
    assert col_warps & 16, "FP32 dQ addressing requires warp-owned column 16"
    return ((bm - 1) & ~(8 | row_warps)) * 128 + (127 & ~(15 | col_warps))


@triton.jit
def _dq_offsets(step, BM: tl.constexpr, MMA: tl.constexpr, PACKED: tl.constexpr = False, REMAT: tl.constexpr = False,
                PACKED_BASE_IMM: tl.constexpr = False):
    if REMAT:
        local_rows = tlx.rematerialized_range(0, BM, 27090, placement=step)
        cols = tlx.rematerialized_range(0, 128, 27091, placement=step)[None, :]
    else:
        local_rows = tl.arange(0, BM)
        cols = tl.arange(0, 128)[None, :]
    rows = (step * BM + local_rows)[:, None]
    # Swap the row-8 and column-16 bits inside each 16x32 patch. The
    # WMMA's lane-16 bit then addresses the adjacent 16 columns, giving
    # each wave a contiguous 128-byte atomic segment (aiter's shuffle).
    if PACKED:
        # Native WMMA registers advance in rows. Make row bit 0 the
        # contiguous BF16 pair, followed by the five lane bits. Merely
        # declaring contiguity=2 on the FP32 shuffle produces misaligned
        # packed atomics because those registers address different rows.
        logical = rows * 128 + cols
        if PACKED_BASE_IMM:
            # Keep the native packed-pair permutation, but separate its
            # lane/warp base from register-owned constant displacements.
            # Otherwise LLVM scales each permuted i32 offset separately,
            # reusing one address VGPR and serializing every packed atomic.
            thread = tlx.thread_id(0)
            row_lane = (thread & 16) >> 1
            col_lane = thread & 15
            for warp_bit in tl.static_range(len(MMA.warp_bases)):
                warp = (thread >> (5 + warp_bit)) & 1
                row_lane += warp * (16 * MMA.warp_bases[warp_bit][0])
                col_lane += warp * (16 * MMA.warp_bases[warp_bit][1])
            lane_logical = (step * BM + (row_lane & (BM - 1))) * 128 + col_lane
            base = ((lane_logical & ~(15 | 16 | 32 | 128 | 1024)) | ((lane_logical & 15) << 1)
                    | ((lane_logical & 16) << 6) | ((lane_logical & 32) << 2) | ((lane_logical & 128) >> 7)
                    | ((lane_logical & 1024) >> 5))
            base = tl.inline_asm_elementwise("", "=v,0", [base], dtype=tl.int32, is_pure=True, pack=1)
            base = base & ((1 << 29) - 1)
            register_bits: tl.constexpr = _dq_register_bits(BM, MMA)
            logical = logical & register_bits
        offsets = ((logical & ~(15 | 16 | 32 | 128 | 1024)) | ((logical & 15) << 1) | ((logical & 16) << 6) |
                   ((logical & 32) << 2) | ((logical & 128) >> 7) | ((logical & 1024) >> 5))
        if PACKED_BASE_IMM:
            offsets += base
    else:
        # Swap bits 4 (column 16) and 10 (row 8), separating the register
        # coordinates from the lane/warp base before the permutation.
        register_bits: tl.constexpr = _dq_register_bits(BM, MMA)
        logical = rows * 128 + cols
        thread = tlx.thread_id(0)
        row_lane = (thread & 16) >> 1
        col_lane = thread & 15
        for warp_bit in tl.static_range(len(MMA.warp_bases)):
            warp = (thread >> (5 + warp_bit)) & 1
            row_lane += warp * (16 * MMA.warp_bases[warp_bit][0])
            col_lane += warp * (16 * MMA.warp_bases[warp_bit][1])
        lane_rows = step * BM + (row_lane & (BM - 1))
        base = ((lane_rows & ~8) * 128 + ((col_lane & 16) << 6) + (col_lane & ~16) + ((lane_rows & 8) << 1))
        # Keep LLVM from reassociating the dynamic terms through each
        # register's constant delta. One scalar identity emits no instruction
        # and supplies the same base to every register owned by this thread.
        base = tl.inline_asm_elementwise("", "=v,0", [base], dtype=tl.int32, is_pure=True, pack=1)
        # fa_backward bounds each query-head group's FP32 workspace below
        # 2 GiB. Retain that range fact across the opaque identity so adding
        # the register delta cannot wrap the 32-bit byte offset.
        base = base & ((1 << 29) - 1)
        offsets = base + (logical & register_bits)
    offsets = tlx.require_layout(offsets, MMA)
    return offsets


@triton.jit
def _dq_atomic(dq, DQA, step, SM_SCALE: tl.constexpr, BM: tl.constexpr, MMA: tl.constexpr, PACKED: tl.constexpr = False,
               REMAT: tl.constexpr = False, CLUSTER_DQ: tl.constexpr = False, PACKED_BASE_IMM: tl.constexpr = False,
               CONVERT_FIRST: tl.constexpr = False):
    scope: tl.constexpr = "cluster" if CLUSTER_DQ else "gpu"
    offsets = _dq_offsets(step, BM, MMA, PACKED, REMAT, PACKED_BASE_IMM)
    dq = tlx.require_layout(dq * SM_SCALE, MMA)
    if PACKED:
        dq = dq.to(tl.bfloat16)
        if CONVERT_FIRST:
            # Finish every conversion before the atomics so they issue as one
            # clause; interleaving reuses data VGPRs and waits on s_wait_xcnt.
            tlx.amd_sched_barrier(0)
        tlx.buffer_atomic_add(DQA, offsets, dq, sem="relaxed", scope=scope, contiguity=2)
    else:
        tlx.buffer_atomic_add(DQA, offsets, dq, sem="relaxed", scope=scope)


@triton.jit
def _dq_atomic_group(dq_groups, offset_groups, DQA, GROUP: tl.constexpr, BM: tl.constexpr,
                     CLUSTER_DQ: tl.constexpr = False):
    scope: tl.constexpr = "cluster" if CLUSTER_DQ else "gpu"
    # The first dimension is the three low row bits, all register-owned.
    # Slicing it leaves every lane/warp active and emits four FP32 atomics
    # per thread at BM64 (two at BM32), without an LDS layout conversion.
    group_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2, 2), (8, BM // 32, 2)),
                                            stride=((BM // 8, 1, 2 * BM, 4 * BM, 2), (16 * BM, 4, 8 * BM)))
    dq_groups = tlx.require_layout(dq_groups, group_layout)
    offset_groups = tlx.require_layout(offset_groups, group_layout)
    dq = tlx.extract_slice(dq_groups, [1, BM // 8 * 128], [GROUP, 0])
    offsets = tlx.extract_slice(offset_groups, [1, BM // 8 * 128], [GROUP, 0])
    tlx.buffer_atomic_add(DQA, offsets, dq, sem="relaxed", scope=scope)


@triton.jit
def _occ1_k_band(kb, resident, BAND: tl.constexpr, K_BANDS: tl.constexpr, RHS: tl.constexpr):
    if BAND < K_BANDS:
        k = resident[BAND]
    else:
        k = tlx.local_load(tlx.local_slice(kb, [BAND * 32, 0], [32, 128]), layout=RHS, relaxed=True)
    k = tlx.require_layout(k, RHS, pin=False)
    return tlx.extract_slice(k, [32, 64], [0, 0]), tlx.extract_slice(k, [32, 64], [0, 64])


@triton.jit
def _occ1_dv_half(p, do_out, dvs, MMA: tl.constexpr, LHS: tl.constexpr, RHS: tl.constexpr, FIRST: tl.constexpr,
                  ORDERED: tl.constexpr):
    # Each call receives the 64 output columns consumed by its four bands.
    # Like gfx950's fragmented dV: finish independent column fragments while
    # the lagged dS operands can start arriving for the bridge.
    p = tlx.require_layout(p, LHS, pin=False)
    do_out = tlx.require_layout(do_out, RHS, pin=False)
    for band in tl.static_range(0 if FIRST else 4, 4 if FIRST else 8):
        a = tlx.extract_slice(p, [128, 32], [(band % 2) * 128, 0])
        b = tlx.extract_slice(do_out, [32, 32], [0, ((band // 2) % 2) * 32])
        c = _wmma(a, b, tlx.require_layout(dvs[band], MMA, pin=False), ORDERED)
        dvs = dvs[:band] + (c, ) + dvs[band + 1:]
    return dvs


@triton.jit
def _delayed_bwd_kernel(
        Q, K, V, DO, LSE, Delta, DQA, DK, DV, HQ: tl.constexpr, HK: tl.constexpr, N: tl.constexpr,
        SM_SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr, TDM: tl.constexpr,
        STAGES: tl.constexpr, DELAYED: tl.constexpr, UNROLL: tl.constexpr, FRAGMENT_FENCES: tl.constexpr = False,
        FRAGMENT_FENCE_MASK: tl.constexpr = 0, FRAGMENTED: tl.constexpr = False, RESIDENT_KV: tl.constexpr = 0,
        ORDERED: tl.constexpr = False, PACKED_DQ: tl.constexpr = False, REMATERIALIZE: tl.constexpr = False,
        OUTER_M: tl.constexpr = 0, ASYNC_STATS: tl.constexpr = False, Q_SPLITS: tl.constexpr = 1,
        REUSE_Q: tl.constexpr = False, PINGPONG: tl.constexpr = False, SPREAD_DQ: tl.constexpr = False,
        LOG2_LSE: tl.constexpr = False, CACHE_SCORE_K: tl.constexpr = False, TILE_MAJOR: tl.constexpr = False,
        PARTITION_LDS: tl.constexpr = False, FRONT_REGIONS: tl.constexpr = 0, BRIDGE_REGIONS: tl.constexpr = 0,
        CLUSTER_DQ: tl.constexpr = False, DQ_SLICES: tl.constexpr = 1, OCC1: tl.constexpr = False,
        OCC1_K_BANDS: tl.constexpr = 0, OCC1_PREFETCH: tl.constexpr = True, OCC1_SCHED: tl.constexpr = False,
        PHASE_IGLP: tl.constexpr = -1, SPLIT_KV_STORES: tl.constexpr = False, TAIL_DK=None, TAIL_DV=None,
        TAIL_SPLITS: tl.constexpr = 1, TAIL_BASE: tl.constexpr = 0, TAIL_OWNERS: tl.constexpr = 0,
        OCC1_DQ_FIRST: tl.constexpr = False, OCC1_LOAD_AFTER_DK: tl.constexpr = False):
    if OCC1:
        tl.static_assert(NW == 4 and BN == 256 and BM == 32 and (OUTER_M == 64 or OUTER_M == 128))
        tl.static_assert(TDM and DELAYED and CLUSTER_DQ and TILE_MAJOR and ASYNC_STATS and not PINGPONG)
        K_BANDS: tl.constexpr = OCC1_K_BANDS
        PREFETCH: tl.constexpr = OCC1_PREFETCH
        SCHED: tl.constexpr = OCC1_SCHED
        if TAIL_SPLITS > 1:
            tl.static_assert(Q_SPLITS == 1 and TAIL_OWNERS > 0 and TAIL_BASE % (N // BN) == 0)
            flat_owner = tl.program_id(0).to(tl.uint32)
            is_tail = flat_owner >= TAIL_BASE
            tail_position = flat_owner - TAIL_BASE
            owner = tl.where(is_tail, TAIL_BASE + tail_position % TAIL_OWNERS, flat_owner).to(tl.int32)
            split = tl.where(is_tail, tail_position // TAIL_OWNERS, 0).to(tl.int32)
            kv_tile = owner % (N // BN)
            kv_head = owner // (N // BN) % HK
            batch = owner // ((N // BN) * HK)
        else:
            kv_tile, kv_head = tl.program_id(0), tl.program_id(1)
            batch_split = tl.program_id(2)
            batch, split = batch_split // Q_SPLITS, batch_split % Q_SPLITS
        GROUP: tl.constexpr = HQ // HK
        PHASES: tl.constexpr = OUTER_M // 32
        # Split at outer boundaries. Each split owns disjoint complete BM32 rows,
        # including when three splits do not evenly divide the phase count.
        outer_count: tl.constexpr = GROUP * N // OUTER_M
        if TAIL_SPLITS > 1:
            # Both prefix and tail CTAs unconditionally seed STAGES-1 tiles.
            tl.static_assert(outer_count // TAIL_SPLITS >= STAGES - 1)
            outer_start = tl.where(is_tail, split * outer_count // TAIL_SPLITS, 0)
            outer_stop = tl.where(is_tail, (split + 1) * outer_count // TAIL_SPLITS, outer_count)
        else:
            outer_start, outer_stop = split * outer_count // Q_SPLITS, (split + 1) * outer_count // Q_SPLITS
        step_start, step_stop = outer_start * PHASES, outer_stop * PHASES
        q_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N * 128
        kv_base = (batch.to(tl.int64) * HK + kv_head) * N * 128
        stats_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N
        Q += q_base
        DO += q_base
        DQA += q_base
        if DQ_SLICES > 1:
            if TAIL_SPLITS > 1:
                batch_count = (TAIL_BASE + TAIL_OWNERS) // (HK * (N // BN))
            else:
                batch_count = (tl.num_programs(2).to(tl.uint32) // Q_SPLITS).to(tl.int64)
            DQA += (kv_tile.to(tl.uint32) // 16).to(tl.int64) * batch_count * HQ * N * 128
        K += kv_base
        V += kv_base
        if TAIL_SPLITS > 1:
            TAIL_DK += kv_base
            TAIL_DV += kv_base
        else:
            DK += kv_base
            DV += kv_base
        if SPLIT_KV_STORES:
            batch_count = (tl.num_programs(2).to(tl.uint32) // Q_SPLITS).to(tl.int64)
            partial_base = split.to(tl.int64) * batch_count * HK * N * 128
            DK += partial_base
            DV += partial_base
        LSE += stats_base
        Delta += stats_base
        mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0)))
        dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2)), transposed=False)
        lhs: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
        rhs: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
        dq_lhs: tl.constexpr = tlx.dot_operand_layout(0, dq_mma, k_width=8)
        dq_rhs: tl.constexpr = tlx.dot_operand_layout(1, dq_mma, k_width=8)
        kb = tlx.local_alloc((256, 128), tl.bfloat16, 1)
        # V is loaded directly into its lifetime-long operand, avoiding a second
        # persistent LDS image. K retains an LDS image for the nonresident bands.
        kv_offsets = (kv_tile * 256 + tl.arange(0, 256))[:, None] * 128 + tl.arange(0, 128)[None, :]
        v = tlx.buffer_load(V, tlx.require_layout(kv_offsets, lhs))
        k_desc = tl.make_tensor_descriptor(K, [N, 128], [128, 1], [256, 128])
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(kb, 0), [kv_tile * 256, 0], clamp_bounds=False)
        qb = tlx.local_alloc((OUTER_M, 128), tl.bfloat16, STAGES)
        dob = tlx.local_alloc((OUTER_M, 128), tl.bfloat16, STAGES)
        lseb = tlx.local_alloc((1, OUTER_M), tl.float32, STAGES)
        db = tlx.local_alloc((1, OUTER_M), tl.float32, STAGES)
        dsb = tlx.local_alloc((32, 256), tl.bfloat16, 2, layout=_ds_shared_layout(32, 256))
        q_desc = tl.make_tensor_descriptor(Q, [GROUP * N, 128], [128, 1], [OUTER_M, 128])
        do_desc = tl.make_tensor_descriptor(DO, [GROUP * N, 128], [128, 1], [OUTER_M, 128])
        lse_desc = tl.make_tensor_descriptor(LSE, [1, GROUP * N], [GROUP * N, 1], [1, OUTER_M])
        delta_desc = tl.make_tensor_descriptor(Delta, [1, GROUP * N], [GROUP * N, 1], [1, OUTER_M])
        tlx.async_amd_descriptor_wait(0)
        k = tlx.local_load(tlx.local_view(kb, 0), layout=lhs)
        resident = ()
        for band in tl.static_range(K_BANDS):
            resident += (tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 0], [32, 128]),
                                        layout=dq_rhs), )
        dks = (tlx.zeros((128, 32), tl.float32, layout=mma), tlx.zeros(
            (128, 32), tl.float32, layout=mma), tlx.zeros(
                (128, 32), tl.float32, layout=mma), tlx.zeros(
                    (128, 32), tl.float32, layout=mma), tlx.zeros(
                        (128, 32), tl.float32, layout=mma), tlx.zeros(
                            (128, 32), tl.float32, layout=mma), tlx.zeros(
                                (128, 32), tl.float32, layout=mma), tlx.zeros((128, 32), tl.float32, layout=mma))
        dvs = (tlx.zeros((128, 32), tl.float32, layout=mma), tlx.zeros(
            (128, 32), tl.float32, layout=mma), tlx.zeros(
                (128, 32), tl.float32, layout=mma), tlx.zeros(
                    (128, 32), tl.float32, layout=mma), tlx.zeros(
                        (128, 32), tl.float32, layout=mma), tlx.zeros(
                            (128, 32), tl.float32, layout=mma), tlx.zeros(
                                (128, 32), tl.float32, layout=mma), tlx.zeros((128, 32), tl.float32, layout=mma))
        tlx.local_store(tlx.local_view(dsb, 1), tl.zeros((32, 256), tl.bfloat16))
        # The TDM wait-count pass takes the minimum over all paths, so a skippable
        # prologue issue lowers the loop's wait to one outstanding load. Issue
        # unconditionally whenever every split is long enough.
        for initial in tl.static_range(STAGES - 1):
            if outer_count // Q_SPLITS > initial:
                _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer_start + initial, N, OUTER_M, STAGES, True, lse_desc,
                            delta_desc, lseb, db, True)
            elif outer_start + initial < outer_stop:
                _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer_start + initial, N, OUTER_M, STAGES, True, lse_desc,
                            delta_desc, lseb, db, True)
        tl.debug_barrier()
        # occ1-outer-begin: statically unrolled BM32 phases, one outer tile ahead.
        for outer in tl.range(outer_start, outer_stop):
            if outer + STAGES - 1 < outer_stop:
                _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer + STAGES - 1, N, OUTER_M, STAGES, True, lse_desc,
                            delta_desc, lseb, db, True)
            tlx.async_amd_descriptor_wait(4 * (STAGES - 1))
            if outer + STAGES - 1 >= outer_stop:
                tlx.async_amd_descriptor_wait(0)
            early_qt = None
            for phase in tl.static_range(PHASES):
                step = outer * PHASES + phase
                qv = tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * 32, 0], [32, 128])
                dov = tlx.local_slice(tlx.local_view(dob, outer % STAGES), [phase * 32, 0], [32, 128])
                if phase == 0:
                    qt = tlx.local_load(tlx.local_trans(qv), layout=rhs, relaxed=True)
                else:
                    qt1 = tlx.local_load(
                        tlx.local_trans(tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * 32, 32],
                                                        [32, 32])), layout=rhs, relaxed=True)
                    qt2 = tlx.local_load(
                        tlx.local_trans(tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * 32, 64],
                                                        [32, 32])), layout=rhs, relaxed=True)
                    qt3 = tlx.local_load(
                        tlx.local_trans(tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * 32, 96],
                                                        [32, 32])), layout=rhs, relaxed=True)
                    qt = tl.cat(tl.cat(early_qt, qt1, dim=0), tl.cat(qt2, qt3, dim=0), dim=0)
                qt = tlx.require_layout(qt, rhs, pin=False)
                scores = tl.dot(k, qt, tlx.zeros((256, 32), tl.float32, layout=mma))
                dot = tlx.local_load(tlx.local_trans(dov), layout=rhs, relaxed=True)
                dp = tl.dot(v, dot, tlx.zeros((256, 32), tl.float32, layout=mma))
                lse = tl.reshape(
                    tlx.local_load(tlx.local_slice(tlx.local_view(lseb, outer % STAGES), [0, phase * 32], [1, 32])),
                    (32, ))
                delta = tl.reshape(
                    tlx.local_load(tlx.local_slice(tlx.local_view(db, outer % STAGES), [0, phase * 32], [1, 32])),
                    (32, ))
                if LOG2_LSE:
                    p = tl.exp2(scores * (SM_SCALE * 1.4426950408889634) - lse[None, :])
                else:
                    p = tl.exp2(scores * (SM_SCALE * 1.4426950408889634) - lse[None, :] * 1.4426950408889634)
                ds = (p * (dp - delta[None, :])).to(tl.bfloat16)
                # Read the first dV half before dS stores enter the DS queue.
                do_out = tlx.local_load(tlx.local_slice(tlx.local_view(dob, outer % STAGES), [phase * 32, 0], [32, 64]),
                                        layout=rhs, relaxed=True)
                tlx.local_store(tlx.local_view(dsb, phase % 2), tl.trans(ds))
                p_native = ()
                for dv_band in tl.static_range(4):
                    if dv_band < 2:
                        # LLVM otherwise re-clusters both conversions before dV.
                        tlx.amd_sched_barrier(0)
                        p_piece = tlx.extract_slice(p, [128, 32], [dv_band * 128, 0])
                        p_native += (tlx.require_layout(p_piece.to(tl.bfloat16), lhs, pin=False), )
                        tlx.amd_sched_barrier(0)
                    a = p_native[dv_band % 2]
                    b = tlx.extract_slice(do_out, [32, 32], [0, (dv_band // 2) * 32])
                    updated_dv = _wmma(a, b, tlx.require_layout(dvs[dv_band], mma, pin=False), ORDERED)
                    dvs = dvs[:dv_band] + (updated_dv, ) + dvs[dv_band + 1:]
                pv = tl.cat(p_native[0], p_native[1], dim=0)
                do_out = tlx.local_load(
                    tlx.local_slice(tlx.local_view(dob, outer % STAGES), [phase * 32, 64], [32, 64]), layout=rhs,
                    relaxed=True)
                # Cover initial bridge dS/Q reads with the final dV half.
                previous = tlx.local_view(dsb, 1 - phase % 2)
                kb_view = tlx.local_view(kb, 0)
                ds_band = tlx.local_load(tlx.local_slice(previous, [0, 0], [32, 32]), layout=dq_lhs, relaxed=True)
                q_part = tlx.local_load(tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * 32, 0], [32, 32]),
                                        layout=rhs, relaxed=True)
                dvs = _occ1_dv_half(pv, do_out, dvs, mma, lhs, rhs, False, ORDERED)
                ds = tlx.require_layout(ds, lhs, pin=False)
                dq0 = tlx.zeros((32, 64), tl.float32, layout=dq_mma)
                dq1 = tlx.zeros((32, 64), tl.float32, layout=dq_mma)
                k0, k1 = _occ1_k_band(kb_view, resident, 0, K_BANDS, dq_rhs)
                for band in tl.static_range(8):
                    if SCHED:
                        tlx.amd_sched_barrier(0)
                    if PREFETCH and band < 7 and not OCC1_LOAD_AFTER_DK:
                        next_ds = tlx.local_load(tlx.local_slice(previous, [0, (band + 1) * 32], [32, 32]),
                                                 layout=dq_lhs, relaxed=True)
                        next_k0, next_k1 = _occ1_k_band(kb_view, resident, band + 1, K_BANDS, dq_rhs)
                        if band % 2 == 0 and band < 6:
                            next_q = tlx.local_load(
                                tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                [phase * 32, ((band + 2) // 2) * 32], [32, 32]), layout=rhs,
                                relaxed=True)
                    if phase + 1 < PHASES and band == 6:
                        # The next phase shares this completed TDM outer.
                        early_qt = tlx.local_load(
                            tlx.local_trans(
                                tlx.local_slice(tlx.local_view(qb, outer % STAGES), [(phase + 1) * 32, 0], [32, 32])),
                            layout=rhs, relaxed=True)
                    ds_current = tlx.extract_slice(ds, [128, 32], [(band % 2) * 128, 0])
                    if OCC1_DQ_FIRST:
                        dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                        dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                    updated = _wmma(ds_current, q_part, tlx.require_layout(dks[band], mma, pin=False), ORDERED)
                    dks = dks[:band] + (updated, ) + dks[band + 1:]
                    if PREFETCH and band < 7 and OCC1_LOAD_AFTER_DK:
                        tlx.amd_sched_barrier(0)
                        next_ds = tlx.local_load(tlx.local_slice(previous, [0, (band + 1) * 32], [32, 32]),
                                                 layout=dq_lhs, relaxed=True)
                        next_k0, next_k1 = _occ1_k_band(kb_view, resident, band + 1, K_BANDS, dq_rhs)
                        if band % 2 == 0 and band < 6:
                            next_q = tlx.local_load(
                                tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                [phase * 32, ((band + 2) // 2) * 32], [32, 32]), layout=rhs,
                                relaxed=True)
                        tlx.amd_sched_barrier(0)
                    if not OCC1_DQ_FIRST:
                        dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                        dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                    if band < 7:
                        if PREFETCH:
                            ds_band, k0, k1 = next_ds, next_k0, next_k1
                            if band % 2 == 1:
                                q_part = next_q
                        else:
                            ds_band = tlx.local_load(tlx.local_slice(previous, [0, (band + 1) * 32], [32, 32]),
                                                     layout=dq_lhs, relaxed=True)
                            k0, k1 = _occ1_k_band(kb_view, resident, band + 1, K_BANDS, dq_rhs)
                            if band % 2 == 1:
                                q_part = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                    [phase * 32, ((band + 1) // 2) * 32], [32, 32]), layout=rhs,
                                    relaxed=True)
                dq = tl.cat(dq0, dq1, dim=1)
                _dq_atomic(dq, DQA, tl.maximum(step - 1, step_start), SM_SCALE if PACKED_DQ else 1.0, 32, dq_mma,
                           PACKED_DQ, False, True, True, OCC1_LOAD_AFTER_DK)
                if PHASE_IGLP >= 0:
                    tlx.amd_iglp_opt(PHASE_IGLP)
                # dS(i) is visible to all waves before phase i+1; all Q/dO reads
                # finish before this outer ring slot can be overwritten.
                tlx.workgroup_barrier()
        # occ1-outer-end
        dq0 = tlx.zeros((32, 64), tl.float32, layout=dq_mma)
        dq1 = tlx.zeros((32, 64), tl.float32, layout=dq_mma)
        for band in tl.static_range(8):
            ds_band = tlx.local_load(tlx.local_slice(tlx.local_view(dsb, 1), [0, band * 32], [32, 32]), layout=dq_lhs)
            k0, k1 = _occ1_k_band(tlx.local_view(kb, 0), resident, band, K_BANDS, dq_rhs)
            dq0 = _wmma(ds_band, k0, dq0, ORDERED)
            dq1 = _wmma(ds_band, k1, dq1, ORDERED)
        _dq_atomic(tl.cat(dq0, dq1, dim=1), DQA, step_stop - 1, SM_SCALE if PACKED_DQ else 1.0, 32, dq_mma, PACKED_DQ,
                   False, True, True)
        dk = tl.cat(tl.cat(tl.cat(dks[0], dks[1], dim=0), tl.cat(dks[2], dks[3], dim=0), dim=1),
                    tl.cat(tl.cat(dks[4], dks[5], dim=0), tl.cat(dks[6], dks[7], dim=0), dim=1), dim=1)
        dv = tl.cat(tl.cat(tl.cat(dvs[0], dvs[1], dim=0), tl.cat(dvs[2], dvs[3], dim=0), dim=1),
                    tl.cat(tl.cat(dvs[4], dvs[5], dim=0), tl.cat(dvs[6], dvs[7], dim=0), dim=1), dim=1)
        if TAIL_SPLITS > 1:
            # Program IDs make this branch uniform across the whole workgroup.
            if is_tail:
                compact_base = (split.to(tl.int64) * TAIL_OWNERS + owner.to(tl.int64) - TAIL_BASE -
                                kv_tile.to(tl.int64)) * BN * 128
                DK += compact_base
                DV += compact_base
                # Each split writes every element of its private FP32 slice once.
                output_layout: tl.constexpr = tlx.layout(shape=((8, 4, 4), (4, 8)), stride=((4, 32, 128), (1, 512)))
                thread = tlx.thread_id(0)
                lane_base = (((thread >> 3) & 15) * 128 + (thread & 7) * 4)
                lane_base = tl.inline_asm_elementwise("", "=v,0", [lane_base], dtype=tl.int32, is_pure=True, pack=1)
                lane_base = lane_base & ((1 << 29) - 4)
                offsets = lane_base + ((tl.arange(0, 128)[:, None] & (16 | 32 | 64)) * 128 +
                                       (tl.arange(0, 32)[None, :] & (1 | 2)))
                offsets = tlx.require_layout(offsets, output_layout)
                for band in tl.static_range(8):
                    native = tlx.require_layout(dks[band], mma)
                    value = tlx.require_layout(native, output_layout)
                    base = DK + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                    tlx.buffer_store(value, base, offsets, contiguity=4)
                for band in tl.static_range(8):
                    native = tlx.require_layout(dvs[band], mma)
                    value = tlx.require_layout(native, output_layout)
                    base = DV + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                    tlx.buffer_store(value, base, offsets, contiguity=4)
            else:
                tl.store(TAIL_DK + kv_offsets, dk * SM_SCALE)
                tl.store(TAIL_DV + kv_offsets, dv)
        elif SPLIT_KV_STORES:
            # Each split writes every element of its private FP32 slice once.
            output_layout: tl.constexpr = tlx.layout(shape=((8, 4, 4), (4, 8)), stride=((4, 32, 128), (1, 512)))
            thread = tlx.thread_id(0)
            lane_base = (((thread >> 3) & 15) * 128 + (thread & 7) * 4)
            lane_base = tl.inline_asm_elementwise("", "=v,0", [lane_base], dtype=tl.int32, is_pure=True, pack=1)
            lane_base = lane_base & ((1 << 29) - 4)
            offsets = lane_base + ((tl.arange(0, 128)[:, None] & (16 | 32 | 64)) * 128 + (tl.arange(0, 32)[None, :] &
                                                                                          (1 | 2)))
            offsets = tlx.require_layout(offsets, output_layout)
            for band in tl.static_range(8):
                native = tlx.require_layout(dks[band], mma)
                value = tlx.require_layout(native, output_layout)
                base = DK + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                tlx.buffer_store(value, base, offsets, contiguity=4)
            for band in tl.static_range(8):
                native = tlx.require_layout(dvs[band], mma)
                value = tlx.require_layout(native, output_layout)
                base = DV + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                tlx.buffer_store(value, base, offsets, contiguity=4)
        elif Q_SPLITS > 1:
            # Band coordinates use a 32-column tensor, independent of the
            # global 128-column stride. Pin native ownership before converting
            # one band, then retire its DEV-scope atomics before the next band.
            output_layout: tl.constexpr = tlx.layout(shape=((32, 4), (32, )), stride=((1, 32), (128, )))
            thread = tlx.thread_id(0)
            atomic_base = (thread >> 5) * 128 + (thread & 31)
            atomic_base = tl.inline_asm_elementwise("", "=v,0", [atomic_base], dtype=tl.int32, is_pure=True, pack=1)
            atomic_base = atomic_base & ((1 << 29) - 1)
            offsets = atomic_base + ((tl.arange(0, 128)[:, None] & ~3) * 128 + tl.zeros((1, 32), tl.int32))
            offsets = tlx.require_layout(offsets, output_layout)
            for band in tl.static_range(8):
                native = tlx.require_layout(dks[band], mma)
                value = tlx.require_layout(native, output_layout)
                base = DK + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                tlx.buffer_atomic_add(base, offsets, value, sem="relaxed", scope="gpu")
            for band in tl.static_range(8):
                native = tlx.require_layout(dvs[band], mma)
                value = tlx.require_layout(native, output_layout)
                base = DV + (kv_tile * 256 + (band % 2) * 128) * 128 + (band // 2) * 32
                tlx.buffer_atomic_add(base, offsets, value, sem="relaxed", scope="gpu")
        else:
            tl.store(DK + kv_offsets, dk * SM_SCALE)
            tl.store(DV + kv_offsets, dv)
        tlx.async_amd_descriptor_wait(0)
    else:
        if TILE_MAJOR:
            kv_head = tl.program_id(1)
            kv_tile = tl.program_id(0)
        else:
            kv_head = tl.program_id(0)
            kv_tile = tl.program_id(1)
        batch_split = tl.program_id(2)
        batch = batch_split // Q_SPLITS
        split = batch_split % Q_SPLITS
        GROUP: tl.constexpr = HQ // HK
        OM: tl.constexpr = OUTER_M if OUTER_M else BM
        PHASES: tl.constexpr = OM // BM
        DS_SLOTS: tl.constexpr = 3 if PINGPONG else 2
        # At BN128 the usual fragmented layout partitions query columns across
        # wave groups. Ping-pong instead partitions KV rows and keeps four full
        # row fragments, so either group can finish its front independently.
        PAIRED_FRAGMENTS: tl.constexpr = FRAGMENTED and (not PINGPONG or BN == 256)
        TDM_PER_STAGE: tl.constexpr = 4 if ASYNC_STATS else 2
        steps: tl.constexpr = GROUP * N // BM
        step_start = split * steps // Q_SPLITS
        step_stop = (split + 1) * steps // Q_SPLITS
        outer_start = step_start // PHASES
        outer_stop = tl.cdiv(step_stop, PHASES)
        q_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N * 128
        kv_base = (batch.to(tl.int64) * HK + kv_head) * N * 128
        stats_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N
        Q += q_base
        DO += q_base
        DQA += q_base
        if DQ_SLICES > 1:
            batch_count = (tl.num_programs(2).to(tl.uint32) // Q_SPLITS).to(tl.int64)
            slice_elems = batch_count * HQ * N * 128
            # Keep the slice displacement in the scalar 64-bit resource base;
            # the per-query-head-group i32 buffer offsets stay unchanged.
            DQA += (kv_tile.to(tl.uint32) // 16).to(tl.int64) * slice_elems
        K += kv_base
        V += kv_base
        DK += kv_base
        DV += kv_base
        LSE += stats_base
        Delta += stats_base

        if NW == 16:
            if BN == 256:
                kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (4, 0), (0, 1)))
            else:
                kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (0, 1), (0, 2)))
            dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2), (1, 0), (2, 0)), transposed=False)
        elif NW == 4:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0)))
            dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2)), transposed=False)
        elif NW == 8:
            if FRAGMENTED and BN == 128 and not PINGPONG:
                kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (0, 1)))
            else:
                kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (4, 0)))
            dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2), (1, 0)), transposed=False)
        else:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), ))
            dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), ), transposed=False)
        lhs: tl.constexpr = tlx.dot_operand_layout(0, kv_mma, k_width=8)
        rhs: tl.constexpr = tlx.dot_operand_layout(1, kv_mma, k_width=8)
        dq_lhs: tl.constexpr = tlx.dot_operand_layout(0, dq_mma, k_width=8)
        dq_rhs: tl.constexpr = tlx.dot_operand_layout(1, dq_mma, k_width=8)
        if PARTITION_LDS:
            # Two 64 KiB LDS partitions: K/V split by rows, dS by columns. Waves
            # on different SIMD pairs that read one partition in the same cycle
            # halve LDS bandwidth, so the pairs walk the delayed-dQ bands in
            # opposite piece order. Wave bit 0 selects the SIMD pair.
            tl.static_assert(PAIRED_FRAGMENTS and BN == 256 and RESIDENT_KV == 0)
            kv_layout: tl.constexpr = tlx.partitioned_shared_layout_encoding(
                2, 1, 0, tlx.padded_shared_layout_encoding.with_identity_for([(128, 8)], [BN // 2, 128]))
            kb = tlx.local_alloc((BN, 128), tl.bfloat16, 1, layout=kv_layout)
            vb = tlx.local_alloc((BN, 128), tl.bfloat16, 1, layout=kv_layout)
            ds_layout: tl.constexpr = tlx.partitioned_shared_layout_encoding(2, 1, 1, _ds_shared_layout(BM, BN // 2))
            dq_piece_flip = ((tlx.thread_id(0) // 32) & 1) * 4
        else:
            kb = tlx.local_alloc((BN, 128), tl.bfloat16, 1)
            vb = tlx.local_alloc((BN, 128), tl.bfloat16, 1)
            ds_layout: tl.constexpr = _ds_shared_layout(BM, BN)
            dq_piece_flip = 0
        qb = tlx.local_alloc((OM, 128), tl.bfloat16, STAGES)
        dob = tlx.local_alloc((OM, 128), tl.bfloat16, STAGES)
        dsb = tlx.local_alloc((BM, BN), tl.bfloat16, DS_SLOTS, layout=ds_layout)
        if DELAYED:
            # Seed the previous slot: phase zero can execute the same bridge as
            # every later phase, publishing a harmless zero dQ into tile zero.
            tlx.local_store(tlx.local_view(dsb, (step_start + DS_SLOTS - 1) % DS_SLOTS), tl.zeros((BM, BN),
                                                                                                  tl.bfloat16))

        lse_desc = tl.make_tensor_descriptor(LSE, [1, GROUP * N], [GROUP * N, 1], [1, OM])
        delta_desc = tl.make_tensor_descriptor(Delta, [1, GROUP * N], [GROUP * N, 1], [1, OM])
        lseb = tlx.local_alloc((1, OM), tl.float32, STAGES)
        db = tlx.local_alloc((1, OM), tl.float32, STAGES)
        q_desc = tl.make_tensor_descriptor(Q, [GROUP * N, 128], [128, 1], [OM, 128])
        do_desc = tl.make_tensor_descriptor(DO, [GROUP * N, 128], [128, 1], [OM, 128])
        if TDM:
            k_desc = tl.make_tensor_descriptor(K, [N, 128], [128, 1], [BN, 128])
            v_desc = tl.make_tensor_descriptor(V, [N, 128], [128, 1], [BN, 128])
            tlx.async_amd_descriptor_load(k_desc, tlx.local_view(kb, 0), [kv_tile * BN, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_load(v_desc, tlx.local_view(vb, 0), [kv_tile * BN, 0], clamp_bounds=False)
            tlx.async_amd_descriptor_wait(0)
        else:
            kv_offsets = (kv_tile * BN + tl.arange(0, BN))[:, None] * 128 + tl.arange(0, 128)[None, :]
            tlx.async_load(K + kv_offsets, tlx.local_view(kb, 0))
            tlx.async_load(V + kv_offsets, tlx.local_view(vb, 0))
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(0)

        if (RESIDENT_KV == 1 or RESIDENT_KV == 2):
            k_resident = tlx.local_load(tlx.local_view(kb, 0), layout=lhs)
        if CACHE_SCORE_K:
            # Native eight-wave WMMA ownership spans 128 rows. Caching the full
            # score operand avoids the LDS exchanges caused by 64/128-row joins.
            # Only the score GEMM uses this view; delayed dQ still streams K.
            k_score_cache = tlx.local_load(tlx.local_view(kb, 0), layout=lhs)
        if RESIDENT_KV:
            v_resident = tlx.local_load(tlx.local_view(vb, 0), layout=lhs)
        if (RESIDENT_KV == 2 or RESIDENT_KV == 3):
            # Keep the reused leading 3/4 of K in delayed-dQ ownership. The last
            # quarter remains a streamed LDS operand, limiting the resident bank.
            k_dq_lo = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [0, 0], [BN // 2, 128]), layout=dq_rhs)
            k_dq_mid = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [BN // 2, 0], [BN // 4, 128]),
                                      layout=dq_rhs)
        if RESIDENT_KV == 3 and BN == 256:
            k_dq_band6 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [192, 0], [32, 128]), layout=dq_rhs)

        dk0 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
        dk1 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
        dk2 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
        dk3 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
        if PAIRED_FRAGMENTS:
            dks = (tlx.zeros(
                (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
                    (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
                        (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma),
                   tlx.zeros((BN // 2, 32), tl.float32,
                             layout=kv_mma), tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma),
                   tlx.zeros((BN // 2, 32), tl.float32,
                             layout=kv_mma), tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma))
        dv = tlx.zeros((BN, 128), tl.float32, layout=kv_mma)
        for initial in tl.static_range(STAGES - 1):
            if Q_SPLITS == 1 or outer_start + initial < outer_stop:
                _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer_start + initial, N, OM, STAGES, TDM, lse_desc,
                            delta_desc, lseb, db, ASYNC_STATS, PINGPONG)

        if PINGPONG:
            tlx.async_amd_descriptor_wait(0)
            tlx.workgroup_barrier()
            # Measured wave placement puts waves w and w+4 on the same SIMD.
            group_b = tlx.thread_id(0) >= 128
            tlx.cond_barrier(group_b)

        for outer in tl.range(outer_start, outer_stop, loop_unroll_factor=UNROLL):
            # The next tile is in flight while this tile computes its five GEMMs.
            if not PINGPONG and outer + STAGES - 1 < outer_stop:
                _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer + STAGES - 1, N, OM, STAGES, TDM, lse_desc,
                            delta_desc, lseb, db, ASYNC_STATS)
            if PINGPONG:
                pass
            elif TDM:
                tlx.async_amd_descriptor_wait(TDM_PER_STAGE * (STAGES - 1))
            else:
                tlx.async_load_wait_group(STAGES - 1)
            # The last iterations have fewer outstanding copies. A fixed nonzero
            # wait count cannot establish readiness for those draining tiles.
            if not PINGPONG and outer + STAGES - 1 >= outer_stop:
                if TDM:
                    tlx.async_amd_descriptor_wait(0)
                else:
                    tlx.async_load_wait_group(0)
            for phase in tl.static_range(PHASES):
                step = outer * PHASES + phase
                if Q_SPLITS == 1 or PHASES == 1 or (step >= step_start and step < step_stop):
                    qv = tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, 0], [BM, 128])
                    dov = tlx.local_slice(tlx.local_view(dob, outer % STAGES), [phase * BM, 0], [BM, 128])
                    if FRONT_REGIONS:
                        scores = _front_regions(tlx.local_view(kb, 0), tlx.local_trans(qv), kv_mma, lhs, rhs, BM, BN,
                                                FRONT_REGIONS == 2)
                        dp = _front_regions(tlx.local_view(vb, 0), tlx.local_trans(dov), kv_mma, lhs, rhs, BM, BN,
                                            FRONT_REGIONS == 2)
                    else:
                        if CACHE_SCORE_K:
                            k = k_score_cache
                        elif (RESIDENT_KV == 1 or RESIDENT_KV == 2):
                            k = k_resident
                        else:
                            k = tlx.local_load(tlx.local_view(kb, 0), layout=lhs, relaxed=True)
                        qt = tlx.local_load(tlx.local_trans(qv), layout=rhs, relaxed=True)
                        scores = tl.dot(k, qt, tlx.zeros((BN, BM), tl.float32, layout=kv_mma))
                        if RESIDENT_KV:
                            v = v_resident
                        else:
                            v = tlx.local_load(tlx.local_view(vb, 0), layout=lhs, relaxed=True)
                        dot = tlx.local_load(tlx.local_trans(dov), layout=rhs, relaxed=True)
                        dp = tl.dot(v, dot, tlx.zeros((BN, BM), tl.float32, layout=kv_mma))
                    rows = step * BM + tl.arange(0, BM)
                    if ASYNC_STATS:
                        lse = tl.reshape(
                            tlx.local_load(
                                tlx.local_slice(tlx.local_view(lseb, outer % STAGES), [0, phase * BM], [1, BM])),
                            (BM, ))
                        delta = tl.reshape(
                            tlx.local_load(tlx.local_slice(tlx.local_view(db, outer % STAGES), [0, phase * BM],
                                                           [1, BM])), (BM, ))
                    else:
                        lse = tl.load(LSE + rows)
                        delta = tl.load(Delta + rows)
                    if LOG2_LSE:
                        p = tl.exp2(scores * (SM_SCALE * 1.4426950408889634) - lse[None, :])
                    else:
                        p = tl.exp2(scores * (SM_SCALE * 1.4426950408889634) - lse[None, :] * 1.4426950408889634)
                    ds = (p * (dp - delta[None, :])).to(tl.bfloat16)
                    tlx.local_store(tlx.local_view(dsb, step % DS_SLOTS), tl.trans(ds))
                    p = tlx.require_layout(p.to(tl.bfloat16), lhs, pin=False)
                    do_out = tlx.local_load(dov, layout=rhs, relaxed=True)
                    dv = tl.dot(p, do_out, dv)
                    ds = tlx.require_layout(ds, lhs, pin=False)
                    # early-q-seed-begin
                    if BRIDGE_REGIONS == 3 or BRIDGE_REGIONS == 4:
                        # Start the first Q read after p/dO registers are dead.
                        # This is the sole live dK Q panel, reused by bands 0 and 1.
                        tlx.amd_sched_barrier(0)
                        q_part = tlx.local_load(
                            tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, 0], [BM, 32]), layout=rhs,
                            relaxed=True)
                    # early-q-seed-end
                    if PINGPONG:
                        # Leading front(i) rendezvous with trailing bridge(i-1).
                        # Its last Q/dO reads are complete before slot i+1 reuses
                        # the two-stage ring's slot i-1. Only leading waves issue
                        # the refill; their wait precedes the next rendezvous.
                        # early-q-front-join
                        tlx.workgroup_barrier()
                        if outer + STAGES - 1 < outer_stop:
                            _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer + STAGES - 1, N, OM, STAGES, TDM,
                                        lse_desc, delta_desc, lseb, db, ASYNC_STATS, PINGPONG)
                    if not DELAYED:
                        # Immediate dQ consumes the publication from this phase
                        # across all waves. The delayed path reads the other slot,
                        # whose publication is covered by the preceding end barrier.
                        tlx.workgroup_barrier()

                    if PAIRED_FRAGMENTS and (BRIDGE_REGIONS == 1 or BRIDGE_REGIONS == 2):
                        # Regions stay between the existing front/end rendezvous.
                        # Mode 1 fences the original reads and per-WMMA ordering;
                        # mode 2 seeds the current operands, issues the next reads
                        # under this block, then advances after its full fence.
                        # The FP32 path retains its eight dQ atomic groups between
                        # dK fragments. No workgroup synchronization is added.
                        if DELAYED:
                            dq_step = tl.maximum(step - 1, step_start)
                            dq_slot = (step + DS_SLOTS - 1) % DS_SLOTS
                        else:
                            dq_step = step
                            dq_slot = step % DS_SLOTS
                        previous = tlx.local_view(dsb, dq_slot)
                        ds_band = tlx.local_load(tlx.local_slice(previous, [0, dq_piece_flip * 32], [BM, 32]),
                                                 layout=dq_lhs, relaxed=True)
                        dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        if BRIDGE_REGIONS == 2:
                            k0 = tlx.local_load(
                                tlx.local_slice(tlx.local_view(kb, 0), [dq_piece_flip * 32, 0], [32, 64]),
                                layout=dq_rhs, relaxed=True)
                            k1 = tlx.local_load(
                                tlx.local_slice(tlx.local_view(kb, 0), [dq_piece_flip * 32, 64], [32, 64]),
                                layout=dq_rhs, relaxed=True)
                        if SPREAD_DQ:
                            for band in tl.static_range(8):
                                tlx.amd_sched_barrier(0)
                                if band < 7:
                                    next_ds = tlx.local_load(
                                        tlx.local_slice(previous, [0, ((band + 1) ^ dq_piece_flip) * 32], [BM, 32]),
                                        layout=dq_lhs, relaxed=True)
                                if BRIDGE_REGIONS == 2:
                                    if band < 7:
                                        next_k0 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0),
                                                            [((band + 1) ^ dq_piece_flip) * 32, 0], [32, 64]),
                                            layout=dq_rhs, relaxed=True)
                                        next_k1 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0),
                                                            [((band + 1) ^ dq_piece_flip) * 32, 64], [32, 64]),
                                            layout=dq_rhs, relaxed=True)
                                else:
                                    k0 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 0],
                                                        [32, 64]), layout=dq_rhs, relaxed=True)
                                    k1 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 64],
                                                        [32, 64]), layout=dq_rhs, relaxed=True)
                                dq0 = _wmma(ds_band, k0, dq0, ORDERED and BRIDGE_REGIONS == 1)
                                dq1 = _wmma(ds_band, k1, dq1, ORDERED and BRIDGE_REGIONS == 1)
                                if band < 7:
                                    ds_band = next_ds
                                    if BRIDGE_REGIONS == 2:
                                        k0 = next_k0
                                        k1 = next_k1
                            atomic_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2, 2), (8, BM // 32, 2)),
                                                                     stride=((1, 1024, 16, 32, 2048), (128, 4096, 64)))
                            dq = tlx.require_layout(tl.cat(dq0, dq1, dim=1) * SM_SCALE, atomic_layout)
                            offsets = tlx.require_layout(_dq_offsets(dq_step, BM, dq_mma, False, REMATERIALIZE),
                                                         atomic_layout)
                            dq_groups = tl.trans(tl.reshape(tl.trans(dq), (128 * BM // 8, 8)))
                            offset_groups = tl.trans(tl.reshape(tl.trans(offsets), (128 * BM // 8, 8)))
                            if BRIDGE_REGIONS == 2:
                                q_part = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, 0], [BM, 32]),
                                    layout=rhs, relaxed=True)
                            for band in tl.static_range(8):
                                tlx.amd_sched_barrier(0)
                                ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                                if BRIDGE_REGIONS == 2:
                                    if band < 7 and (not REUSE_Q or (band + 1) % 2 == 0):
                                        next_q = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                            [phase * BM, ((band + 1) // 2) * 32], [BM, 32]), layout=rhs,
                                            relaxed=True)
                                elif not REUSE_Q or band % 2 == 0:
                                    q_part = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                        [phase * BM, (band // 2) * 32], [BM, 32]), layout=rhs,
                                        relaxed=True)
                                updated = _wmma(ds_current, q_part, dks[band], ORDERED and BRIDGE_REGIONS == 1)
                                dks = dks[:band] + (updated, ) + dks[band + 1:]
                                _dq_atomic_group(dq_groups, offset_groups, DQA, band, BM, CLUSTER_DQ)
                                tlx.amd_sched_barrier(0)
                                if BRIDGE_REGIONS == 2 and band < 7 and (not REUSE_Q or (band + 1) % 2 == 0):
                                    q_part = next_q
                        else:
                            if BRIDGE_REGIONS == 2:
                                q_part = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, 0], [BM, 32]),
                                    layout=rhs, relaxed=True)
                            for band in tl.static_range(8):
                                tlx.amd_sched_barrier(0)
                                ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                                if BRIDGE_REGIONS == 2:
                                    if band < 7 and (not REUSE_Q or (band + 1) % 2 == 0):
                                        next_q = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                            [phase * BM, ((band + 1) // 2) * 32], [BM, 32]), layout=rhs,
                                            relaxed=True)
                                    if band < 7:
                                        next_k0 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0),
                                                            [((band + 1) ^ dq_piece_flip) * 32, 0], [32, 64]),
                                            layout=dq_rhs, relaxed=True)
                                        next_k1 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0),
                                                            [((band + 1) ^ dq_piece_flip) * 32, 64], [32, 64]),
                                            layout=dq_rhs, relaxed=True)
                                else:
                                    if not REUSE_Q or band % 2 == 0:
                                        q_part = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                            [phase * BM, (band // 2) * 32], [BM, 32]), layout=rhs,
                                            relaxed=True)
                                    k0 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 0],
                                                        [32, 64]), layout=dq_rhs, relaxed=True)
                                    k1 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 64],
                                                        [32, 64]), layout=dq_rhs, relaxed=True)
                                if band < 7:
                                    next_ds = tlx.local_load(
                                        tlx.local_slice(previous, [0, ((band + 1) ^ dq_piece_flip) * 32], [BM, 32]),
                                        layout=dq_lhs, relaxed=True)
                                updated = _wmma(ds_current, q_part, dks[band], ORDERED and BRIDGE_REGIONS == 1)
                                dks = dks[:band] + (updated, ) + dks[band + 1:]
                                dq0 = _wmma(ds_band, k0, dq0, ORDERED and BRIDGE_REGIONS == 1)
                                dq1 = _wmma(ds_band, k1, dq1, ORDERED and BRIDGE_REGIONS == 1)
                                if band < 7:
                                    ds_band = next_ds
                                    if BRIDGE_REGIONS == 2:
                                        k0 = next_k0
                                        k1 = next_k1
                                if BRIDGE_REGIONS == 2 and band < 7 and (not REUSE_Q or (band + 1) % 2 == 0):
                                    q_part = next_q
                            dq = tl.cat(dq0, dq1, dim=1)
                            _dq_atomic(dq, DQA, dq_step, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE, CLUSTER_DQ)
                    elif PAIRED_FRAGMENTS:
                        if DELAYED:
                            dq_step = tl.maximum(step - 1, step_start)
                            dq_slot = (step + DS_SLOTS - 1) % DS_SLOTS
                        else:
                            dq_step = step
                            dq_slot = step % DS_SLOTS
                        previous = tlx.local_view(dsb, dq_slot)
                        if SPREAD_DQ:
                            # Finish the previous dQ before the current dK chain.
                            # Its eight register slices can then issue between dK
                            # fragments without carrying dQ into the next front.
                            dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                            dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                            ds_band = tlx.local_load(tlx.local_slice(previous, [0, dq_piece_flip * 32], [BM, 32]),
                                                     layout=dq_lhs, relaxed=True)
                            for band in tl.static_range(8):
                                if band < 7:
                                    next_ds = tlx.local_load(
                                        tlx.local_slice(previous, [0, ((band + 1) ^ dq_piece_flip) * 32], [BM, 32]),
                                        layout=dq_lhs, relaxed=True)
                                k0 = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 0], [32, 64]),
                                    layout=dq_rhs, relaxed=True)
                                k1 = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 64], [32, 64]),
                                    layout=dq_rhs, relaxed=True)
                                dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                                dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                                if band < 7:
                                    ds_band = next_ds
                            atomic_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2, 2), (8, BM // 32, 2)),
                                                                     stride=((1, 1024, 16, 32, 2048), (128, 4096, 64)))
                            dq = tlx.require_layout(tl.cat(dq0, dq1, dim=1) * SM_SCALE, atomic_layout)
                            offsets = tlx.require_layout(_dq_offsets(dq_step, BM, dq_mma, False, REMATERIALIZE),
                                                         atomic_layout)
                            dq_groups = tl.trans(tl.reshape(tl.trans(dq), (128 * BM // 8, 8)))
                            offset_groups = tl.trans(tl.reshape(tl.trans(offsets), (128 * BM // 8, 8)))
                            for band in tl.static_range(8):
                                ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                                if not REUSE_Q or band % 2 == 0:
                                    q_part = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                        [phase * BM, (band // 2) * 32], [BM, 32]), layout=rhs,
                                        relaxed=True)
                                updated = _wmma(ds_current, q_part, dks[band], ORDERED)
                                dks = dks[:band] + (updated, ) + dks[band + 1:]
                                _dq_atomic_group(dq_groups, offset_groups, DQA, band, BM, CLUSTER_DQ)
                                # Compiler-only fence retains each VMEM group next
                                # to its dK fragment; it emits no workgroup barrier.
                                tlx.amd_sched_barrier(0)
                        else:
                            # Start delayed-dQ reads before dK, then consume one K=32 band
                            # between native dK fragments. Independent head-column halves
                            # supply two dQ accumulator chains.
                            ds_band = tlx.local_load(tlx.local_slice(previous, [0, dq_piece_flip * 32], [BM, 32]),
                                                     layout=dq_lhs, relaxed=True)
                            dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                            dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                            for band in tl.static_range(8):
                                ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                                if BRIDGE_REGIONS != 4 and (BRIDGE_REGIONS != 3 or band != 0) and (not REUSE_Q
                                                                                                   or band % 2 == 0):
                                    # The adjacent low/high KV fragments consume the
                                    # same Q panel. Retain it across their dQ band.
                                    q_part = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                        [phase * BM, (band // 2) * 32], [BM, 32]), layout=rhs,
                                        relaxed=True)
                                if band < BN // 32:
                                    if band + 1 < BN // 32:
                                        next_ds = tlx.local_load(
                                            tlx.local_slice(previous, [0, ((band + 1) ^ dq_piece_flip) * 32], [BM, 32]),
                                            layout=dq_lhs, relaxed=True)
                                    if (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < BN // 64:
                                        k0 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 0])
                                        k1 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 64])
                                    elif (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < 3 * BN // 128:
                                        k0 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 0])
                                        k1 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 64])
                                    elif RESIDENT_KV == 3 and BN == 256 and band == 6:
                                        k0 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 0])
                                        k1 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 64])
                                    else:
                                        k0 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 0],
                                                            [32, 64]), layout=dq_rhs, relaxed=True)
                                        k1 = tlx.local_load(
                                            tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 64],
                                                            [32, 64]), layout=dq_rhs, relaxed=True)
                                # early-q-lookahead-begin
                                if BRIDGE_REGIONS == 4 and band < 7 and band % 2 == 1:
                                    # Keep the current Q panel live for this dK while
                                    # the next panel arrives under independent WMMAs.
                                    next_q = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(qb, outer % STAGES),
                                                        [phase * BM, ((band + 1) // 2) * 32], [BM, 32]), layout=rhs,
                                        relaxed=True)
                                # early-q-lookahead-end
                                updated = _wmma(ds_current, q_part, dks[band], ORDERED)
                                dks = dks[:band] + (updated, ) + dks[band + 1:]
                                if band < BN // 32:
                                    dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                                    dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                                    if band + 1 < BN // 32:
                                        ds_band = next_ds
                                if BRIDGE_REGIONS == 4 and band < 7 and band % 2 == 1:
                                    q_part = next_q
                            dq = tl.cat(dq0, dq1, dim=1)
                            _dq_atomic(dq, DQA, dq_step, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE, CLUSTER_DQ)
                    else:
                        dq = tlx.zeros((BM, 128), tl.float32, layout=dq_mma)
                        # Fragment current dK and delayed dQ so both WMMA chains are visible
                        # to LLVM. dK consumes registers; dQ reads the other dS LDS slot.
                        for band in tl.static_range(4):
                            q_part = tlx.local_load(
                                tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, band * 32], [BM, 32]),
                                layout=rhs, relaxed=True)
                            if band == 0:
                                dk0 = tl.dot(ds, q_part, dk0)
                            elif band == 1:
                                dk1 = tl.dot(ds, q_part, dk1)
                            elif band == 2:
                                dk2 = tl.dot(ds, q_part, dk2)
                            else:
                                dk3 = tl.dot(ds, q_part, dk3)
                            if FRAGMENT_FENCES:
                                tlx.amd_sched_barrier(FRAGMENT_FENCE_MASK)
                            if DELAYED:
                                dq_step = tl.maximum(step - 1, step_start)
                                dq_slot = (step + DS_SLOTS - 1) % DS_SLOTS
                            else:
                                dq_step = step
                                dq_slot = step % DS_SLOTS
                            # BN/4 is at least the native reduction width (32).
                            ds_part = tlx.local_load(
                                tlx.local_slice(tlx.local_view(dsb, dq_slot), [0, band * (BN // 4)], [BM, BN // 4]),
                                layout=dq_lhs, relaxed=True)
                            k_part = tlx.local_load(
                                tlx.local_slice(tlx.local_view(kb, 0), [band * (BN // 4), 0], [BN // 4, 128]),
                                layout=dq_rhs, relaxed=True)
                            dq = tl.dot(ds_part, k_part, dq)
                            if FRAGMENT_FENCES:
                                tlx.amd_sched_barrier(FRAGMENT_FENCE_MASK)
                        _dq_atomic(dq, DQA, dq_step, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE, CLUSTER_DQ)
                    if PINGPONG:
                        tlx.async_amd_descriptor_wait(0)
                        # Leading bridge(i) rendezvous with trailing front(i).
                        tlx.workgroup_barrier()
                    else:
                        tl.debug_barrier()

        if PINGPONG:
            tlx.cond_barrier(~group_b)
            if BN == 128 and BM == 64 and PACKED_DQ:
                # The closing cond_barrier only realigns barrier counts: the
                # trailing group may still be in its last half step. Any epilogue
                # LDS write (here, the packed-dQ layout conversion) must follow a
                # full join, or it can overwrite LDS the trailing group still reads.
                tlx.workgroup_barrier()

        if DELAYED:
            if PAIRED_FRAGMENTS:
                dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                for band in tl.static_range(BN // 32):
                    ds_band = tlx.local_load(
                        tlx.local_slice(tlx.local_view(dsb, (step_stop - 1) % DS_SLOTS),
                                        [0, (band ^ dq_piece_flip) * 32], [BM, 32]), layout=dq_lhs)
                    if (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < BN // 64:
                        k0 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 0])
                        k1 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 64])
                    elif (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < 3 * BN // 128:
                        k0 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 0])
                        k1 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 64])
                    elif RESIDENT_KV == 3 and BN == 256 and band == 6:
                        k0 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 0])
                        k1 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 64])
                    else:
                        k0 = tlx.local_load(
                            tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 0], [32, 64]),
                            layout=dq_rhs, relaxed=True)
                        k1 = tlx.local_load(
                            tlx.local_slice(tlx.local_view(kb, 0), [(band ^ dq_piece_flip) * 32, 64], [32, 64]),
                            layout=dq_rhs, relaxed=True)
                    dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                    dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                dq = tl.cat(dq0, dq1, dim=1)
            else:
                dq = tlx.zeros((BM, 128), tl.float32, layout=dq_mma)
                for band in tl.static_range(4):
                    ds_part = tlx.local_load(
                        tlx.local_slice(tlx.local_view(dsb, (step_stop - 1) % DS_SLOTS), [0, band * (BN // 4)],
                                        [BM, BN // 4]), layout=dq_lhs)
                    k_part = tlx.local_load(
                        tlx.local_slice(tlx.local_view(kb, 0), [band * (BN // 4), 0], [BN // 4, 128]), layout=dq_rhs)
                    dq = tl.dot(ds_part, k_part, dq)
            _dq_atomic(dq, DQA, step_stop - 1, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE, CLUSTER_DQ)
        if PAIRED_FRAGMENTS:
            dk0 = tl.cat(dks[0], dks[1], dim=0)
            dk1 = tl.cat(dks[2], dks[3], dim=0)
            dk2 = tl.cat(dks[4], dks[5], dim=0)
            dk3 = tl.cat(dks[6], dks[7], dim=0)
        dk = tl.cat(tl.cat(dk0, dk1, dim=1), tl.cat(dk2, dk3, dim=1), dim=1)
        offsets = (kv_tile * BN + tl.arange(0, BN))[:, None] * 128 + tl.arange(0, 128)[None, :]
        if Q_SPLITS > 1:
            # Native KV WMMA lanes advance in rows. Redistribute once in the
            # epilogue so each FP32 atomic instruction writes a contiguous row
            # segment, instead of scattering its lanes across 128-column strides.
            if PINGPONG and BN == 128:
                # Full-row fragments retain native ownership at this width.
                output_layout: tl.constexpr = kv_mma
            elif PINGPONG:
                # Preserve wave ownership after the phase-shifted loop, avoiding
                # an LDS exchange. Each wave owns two groups of 16 KV rows at BN256.
                # (Cross-wave conversions are also correct now that the AMD sink
                # pass moves them past the closing cond_barrier.)
                output_layout: tl.constexpr = tlx.layout(shape=((32, NW), (4, 16, BN // (16 * NW))),
                                                         stride=((1, 16 * 128), (32, 128, 128 * 128)))
            else:
                output_layout: tl.constexpr = tlx.layout(shape=((32, NW), (4, BN // NW)),
                                                         stride=((1, 128), (32, 128 * NW)))
            offsets = tlx.require_layout(offsets, output_layout)
            dk = tlx.require_layout(dk, output_layout)
            dv = tlx.require_layout(dv, output_layout)
            tlx.buffer_atomic_add(DK, offsets, dk, sem="relaxed", scope="gpu")
            tlx.buffer_atomic_add(DV, offsets, dv, sem="relaxed", scope="gpu")
        else:
            tl.store(DK + offsets, dk * SM_SCALE)
            tl.store(DV + offsets, dv)


@triton.jit
def _convert_outputs(DQA, DQ, DKA, DVA, DK, DV, SIZE: tl.constexpr, KV_SIZE: tl.constexpr, SM_SCALE: tl.constexpr,
                     BLOCK: tl.constexpr, PACKED: tl.constexpr = False, PARTIALS: tl.constexpr = False,
                     DQ_SLICES: tl.constexpr = 1, SCALE_DQ: tl.constexpr = False, KV_SLICES: tl.constexpr = 1):
    block_base = tl.program_id(0).to(tl.int64) * BLOCK
    if PACKED:
        # A contiguous memory run contains logical low bits 0..3 interleaved
        # with logical bit 7. Expose that run to AxisInfo before the transpose.
        rows = tl.program_id(0).to(tl.int64) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
        logical_outer = ((rows & 7) << 4) + ((rows & ~7) << 5)
        memory_rows = ((rows & ~(1 | 2 | 4 | 32)) | ((rows & 1) << 5) | ((rows & 32) >> 5) | ((rows & 2) << 1) |
                       ((rows & 4) >> 1))
        memory_outer = memory_rows * 32
        columns = tl.arange(0, 32)
        memory_offsets = memory_outer[:, None] + columns[None, :]
        if SIZE % 256 == 0:
            # Every 256-element logical group is complete for BHND128 inputs.
            # Keep the load predicate uniform over the contiguous memory run.
            logical_load = logical_outer[:, None]
        else:
            logical_load = logical_outer[:, None] + (columns[None, :] >> 1) + ((columns[None, :] & 1) << 7)
        offsets = (logical_outer[:, None, None] + tl.arange(0, 2)[None, :, None] * 128 +
                   tl.arange(0, 16)[None, None, :])
    else:
        # The bit-4/bit-10 swap affects only the outer index. The inner
        # sixteen values remain explicitly contiguous for vector buffer loads.
        outer_rows = tl.program_id(0).to(tl.int64) * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
        memory_rows = ((outer_rows & ~(64 | 1)) | ((outer_rows & 64) >> 6) | ((outer_rows & 1) << 6))
        memory_outer = memory_rows * 16
        outer = outer_rows * 16
        memory_offsets = memory_outer[:, None] + tl.arange(0, 16)[None, :]
        offsets = outer[:, None] + tl.arange(0, 16)[None, :]
        logical_load = offsets
    dq = tl.load(DQA + memory_offsets, logical_load < SIZE, 0)
    if DQ_SLICES > 1:
        dq = dq.to(tl.float32)
        for slice_index in tl.static_range(1, DQ_SLICES):
            dq += tl.load(DQA + slice_index * SIZE + memory_offsets, logical_load < SIZE, 0).to(tl.float32)
    if SCALE_DQ:
        dq *= SM_SCALE
    if PACKED:
        dq = tl.permute(tl.reshape(dq, (BLOCK // 32, 16, 2)), (0, 2, 1))
    tl.store(DQ + offsets, dq, offsets < SIZE)
    if PARTIALS:
        # A full register tile is 128 threads x eight values, with no
        # redundant lanes. Fractional and smaller tiles keep the plain loop.
        KV_BLOCK: tl.constexpr = BLOCK * KV_SIZE // SIZE
        BALANCED: tl.constexpr = (KV_SLICES > 1 and SIZE != KV_SIZE and SIZE % KV_SIZE == 0
                                  and (BLOCK * KV_SIZE) % SIZE == 0 and KV_BLOCK >= 1024 and KV_BLOCK % 1024 == 0)
        if BALANCED:
            partial_offsets = tl.program_id(0).to(tl.int64) * KV_BLOCK + tl.arange(0, KV_BLOCK)
            partial_layout: tl.constexpr = tlx.layout(shape=((128, ), (8, KV_BLOCK // 1024)), stride=((8, ), (1, 1024)))
            partial_offsets = tlx.require_layout(partial_offsets, partial_layout)
        else:
            partial_offsets = block_base + tl.arange(0, BLOCK)
        dk = tl.load(DKA + partial_offsets, partial_offsets < KV_SIZE, 0)
        dv = tl.load(DVA + partial_offsets, partial_offsets < KV_SIZE, 0)
        for partial_slice in tl.static_range(1, KV_SLICES):
            dk += tl.load(DKA + partial_slice * KV_SIZE + partial_offsets, partial_offsets < KV_SIZE, 0)
            dv += tl.load(DVA + partial_slice * KV_SIZE + partial_offsets, partial_offsets < KV_SIZE, 0)
        tl.store(DK + partial_offsets, dk * SM_SCALE, partial_offsets < KV_SIZE)
        tl.store(DV + partial_offsets, dv, partial_offsets < KV_SIZE)


def _tail_partition(config, b, hk, n):
    """Prefix/tail KV owners for the measured 192-CU occupancy-one policy."""
    if config.tail_splits == 1:
        return 0, 0
    nt = n // config.block_n
    owners = b * hk * nt
    if not nt or 192 % nt or owners % 192 == 0:
        raise ValueError("tail splits require NT dividing 192 and a nonempty dispatch tail")
    return owners // 192 * 192, owners % 192


def allocate_workspace(q, k, *, config=None, packed_dq=None):
    """Allocate the five legacy buffers and optional FP32 dK/dV partials."""
    config = config or select_config(q.shape[-2], q.shape[0], q.shape[1], k.shape[1],
                                     packed_dq=True if packed_dq is None else packed_dq)
    if packed_dq is None:
        packed_dq = config.packed_dq
    dq_slices = _dq_slices(config, q.shape[-2])
    dqa_shape = (dq_slices, ) + q.shape if dq_slices > 1 else q.shape
    workspace = (torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
                 torch.empty(dqa_shape, device=q.device, dtype=torch.bfloat16 if packed_dq else torch.float32),
                 torch.empty_like(q), torch.empty_like(k), torch.empty_like(k))
    tail_base, tail_owners = _tail_partition(config, q.shape[0], k.shape[1], q.shape[-2])
    if config.q_splits > 1 or config.tail_splits > 1:
        if config.tail_splits > 1:
            partial_shape = (config.tail_splits, tail_owners, config.block_n, 128)
        else:
            partial_shape = (config.q_splits, ) + k.shape if config.occ1 and config.packed_dq else k.shape
        workspace += (torch.empty(partial_shape, device=k.device,
                                  dtype=torch.float32), torch.empty(partial_shape, device=k.device,
                                                                    dtype=torch.float32))
    if config.log2_lse:
        workspace += (torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32), )
    return workspace


def fa_backward(q, k, v, o, do, lse, sm_scale=None, causal=False, *, config=None, workspace=None):
    """Return dQ/dK/dV for contiguous BHND BF16 inputs, equal lengths, D128."""
    b, hq, n, d = q.shape
    hk = k.shape[1]
    if min(b, hq, hk) < 1 or hq % hk:
        raise ValueError("invalid batch/head counts")
    if q.device.type != "cuda":
        raise ValueError("inputs must be on a GPU")
    config = config or select_config(n, b, hq, hk)
    compiler_options = _compiler_options(config, n)
    if config.block_n not in (128, 256) or config.block_m not in (16, 32, 64) or config.num_warps not in (2, 4, 8, 16):
        raise ValueError("unsupported WMMA tile or wave count")
    if causal or d != 128 or n < 256 or n % config.block_n or n % config.block_m:
        raise ValueError("gfx1250 backward requires non-causal D128 and tile-aligned N >= 256")
    if config.stages not in (2, 3, 4) or config.unroll not in (1, 2, 4, 8):
        raise ValueError("unsupported pipeline depth or unroll")
    if not isinstance(config.fragment_fence_mask, int) or not 0 <= config.fragment_fence_mask <= 0xFFF:
        raise ValueError("fragment scheduling mask must be an integer in [0, 0xfff]")
    outer_m = config.outer_m or config.block_m
    if outer_m not in (16, 32, 64, 128) or outer_m < config.block_m or n % outer_m:
        raise ValueError("outer Q tile must be an aligned multiple of the compute tile")
    if config.async_stats and not config.use_tdm:
        raise ValueError("asynchronous statistics require TDM")
    if config.occ1:
        if (config.num_warps != 4 or config.block_n != 256 or config.block_m != 32 or outer_m not in (64, 128)
                or config.waves_per_eu != 1 or not config.use_tdm or not config.async_stats or not config.delayed
                or not config.tile_major or not config.cluster_dq or config.pingpong or config.spread_dq
                or config.partition_lds or config.front_regions or config.bridge_regions or config.cache_score_k
                or config.resident_kv or config.rematerialize):
            raise ValueError(
                "OCC1 requires four waves, BM32 BN256 outer64/128, one wave/EU, delayed TDM and cluster dQ")
        if not isinstance(config.occ1_k_bands, int) or not 0 <= config.occ1_k_bands <= 7:
            raise ValueError("OCC1 resident K bands must be an integer in [0, 7]")
        if (hq // hk) * n // outer_m < config.q_splits:
            raise ValueError("every OCC1 split must contain an outer tile")
    elif (config.occ1_k_bands or config.occ1_sched or config.occ1_dq_first or config.occ1_load_after_dk
          or config.phase_iglp != -1 or not config.occ1_prefetch):
        raise ValueError("OCC1 scheduling options require OCC1")
    if (config.occ1_dq_first or config.occ1_load_after_dk) and not config.occ1_prefetch:
        raise ValueError("OCC1 WMMA reorder experiments require band prefetch")
    if not isinstance(config.phase_iglp, int) or config.phase_iglp not in (-1, 0, 1, 2, 3):
        raise ValueError("phase_iglp must be -1 or an IGLP strategy in [0, 3]")
    if config.pingpong and (not config.use_tdm or not config.delayed or not config.fragmented or config.num_warps != 8
                            or config.stages != 2 or outer_m != config.block_m):
        raise ValueError("ping-pong requires delayed fragmented TDM, eight waves, two stages and one compute phase")
    if config.spread_dq and (config.packed_dq or not config.pingpong or not config.fragmented or not config.delayed
                             or config.num_warps != 8 or config.block_n != 256 or config.block_m not in (32, 64)
                             or config.resident_kv != 0):
        raise ValueError("spread dQ requires FP32 ping-pong, eight waves, BM32/64 BN256 and streamed KV")
    if config.resident_kv not in (0, 1, 2, 3, 4):
        raise ValueError("unsupported resident operand policy")
    if config.front_regions not in (0, 1, 2):
        raise ValueError("front_regions must be zero, one, or two")
    if config.bridge_regions not in (0, 1, 2, 3, 4):
        raise ValueError("bridge_regions must be zero, one, two, three, or four")
    if config.bridge_regions in (3, 4) and (config.block_m != 32 or not config.cache_score_k or config.q_splits != 1
                                            or not config.reuse_q or config.spread_dq):
        raise ValueError("early Q read requires cached BM32, one Q split, reused Q and no spread dQ")
    if config.front_regions and (config.cache_score_k or config.resident_kv != 0):
        raise ValueError("front regions stream K/V from LDS")
    if (config.bridge_regions
            or config.partition_lds) and not (config.fragmented and config.block_n == 256 and config.num_warps == 8
                                              and config.pingpong and config.resident_kv == 0):
        raise ValueError(
            "bridge regions and LDS partitions require paired BN256 eight-wave ping-pong with streamed K/V")
    if config.cache_score_k and (config.block_n != 256 or config.num_warps != 8 or config.resident_kv != 0):
        raise ValueError("score K cache requires BN256, eight waves and streamed dQ K/V")
    if not isinstance(config.q_splits, int) or not 1 <= config.q_splits <= 4:
        raise ValueError("q_splits must be an integer in [1, 4]")
    if not isinstance(config.tail_splits, int) or config.tail_splits not in (1, 2, 3, 4):
        raise ValueError("tail_splits must be an integer in [1, 4]")
    if config.tail_splits > 1:
        if not config.occ1 or config.q_splits != 1:
            raise ValueError("tail splits require single-split OCC1")
        _tail_partition(config, b, hk, n)
        if (hq // hk) * n // outer_m // config.tail_splits < config.stages - 1:
            raise ValueError("tail splits require at least STAGES-1 outer tiles per split")
    if config.q_splits > (hq // hk) * n // config.block_m:
        raise ValueError("every Q split must contain a compute tile")
    if config.pingpong and config.block_m == 64 and config.block_n == 256 and config.q_splits == 1:
        # The joined dK/dV epilogue spills with this tile. A phase-shifted
        # spill is not a supported route: use BM32 or multiple query splits.
        raise ValueError("BM64 BN256 ping-pong requires at least two Q splits")
    if config.pingpong and (hq // hk) * n // config.block_m // config.q_splits < 2:
        # A one-step split has no steady-state phase overlap, and repeated
        # one-step ping-pong dispatches were observed to hang. The cause is
        # not yet understood, so reject the configuration explicitly.
        raise ValueError("ping-pong requires at least two steps per Q split")
    if config.preprocess_block not in (64, 128, 256) or config.convert_block not in (1024, 2048, 4096, 8192):
        raise ValueError("unsupported preprocess/conversion block")
    if not isinstance(config.waves_per_eu, int) or not 0 <= config.waves_per_eu <= 16:
        raise ValueError("waves_per_eu must be an integer in [0, 16]")
    tensors = (q, k, v, o, do, lse)
    if (b < 1 or hq < 1 or hk < 1 or hq % hk or k.shape != (b, hk, n, d) or v.shape != k.shape or o.shape != q.shape
            or do.shape != q.shape or lse.shape != (b, hq, n)):
        raise ValueError("inconsistent attention tensor shapes")
    if q.device.type != "cuda" or any(not t.is_contiguous() or t.device != q.device for t in tensors):
        raise ValueError("inputs must be contiguous on the same GPU")
    if any(t.dtype != torch.bfloat16 for t in tensors[:-1]) or lse.dtype != torch.float32:
        raise ValueError("inputs must be BF16, with FP32 LSE")
    if triton.runtime.driver.active.get_current_target().arch != "gfx1250":
        raise ValueError("requires gfx1250")
    if (hq // hk) * n * 128 >= (1 << 29):
        raise ValueError("one query-head group's FP32 dQ workspace must fit in the signed 2-GiB buffer range")
    workspace = workspace if workspace is not None else allocate_workspace(q, k, config=config)
    partials = config.q_splits > 1 or config.tail_splits > 1
    tail_base, tail_owners = _tail_partition(config, b, hk, n)
    base_buffers = 7 if partials else 5
    if len(workspace) != base_buffers + int(config.log2_lse):
        raise ValueError("workspace does not match q_splits")
    dq_slices = _dq_slices(config, n)
    dqa_shape = (dq_slices, ) + q.shape if dq_slices > 1 else q.shape
    delta, dqa, dq, dk, dv = workspace[:5]
    dka, dva = workspace[5:7] if partials else (None, None)
    lse_log2 = workspace[base_buffers] if config.log2_lse else None
    specifications = [(delta, q.shape[:-1], torch.float32),
                      (dqa, dqa_shape, torch.bfloat16 if config.packed_dq else torch.float32),
                      (dq, q.shape, torch.bfloat16), (dk, k.shape, torch.bfloat16), (dv, k.shape, torch.bfloat16)]
    if partials:
        if config.tail_splits > 1:
            partial_shape = (config.tail_splits, tail_owners, config.block_n, d)
        else:
            partial_shape = (config.q_splits, ) + k.shape if config.occ1 and config.packed_dq else k.shape
        specifications += [(dka, partial_shape, torch.float32), (dva, partial_shape, torch.float32)]
    if config.log2_lse:
        specifications.append((lse_log2, q.shape[:-1], torch.float32))
    for value, shape, dtype in specifications:
        if value.shape != shape or value.dtype != dtype or value.device != q.device or not value.is_contiguous():
            raise ValueError("invalid backward workspace")
    # Zero the atomic accumulators in their own launch. On MI450 a kernel that
    # mixes the O/dO loads with these stores, right after the previous call's
    # atomics, can stall forever in unsynchronized loops.
    _run_bwd_preprocess(o, do, delta, lse=lse if config.log2_lse else None, lse_log2=lse_log2,
                        block_m=config.preprocess_block, native_lse=config.log2_lse)
    _run_bwd_zero_dq(dqa, b * hq, n, d, block_m=config.preprocess_block, dq_slices=dq_slices,
                     dk_acc=None if config.tail_splits > 1 or config.occ1 and config.packed_dq else dka,
                     dv_acc=None if config.tail_splits > 1 or config.occ1 and config.packed_dq else dva, heads=hq,
                     kv_heads=hk)
    grid = (n // config.block_n, hk, b * config.q_splits) if config.tile_major else (hk, n // config.block_n,
                                                                                     b * config.q_splits)
    if config.tail_splits > 1:
        grid = (tail_base + tail_owners * config.tail_splits, 1, 1)
    _delayed_bwd_kernel[grid](
        q, k, v, do, lse_log2 if config.log2_lse else lse, delta, dqa, dka if dka is not None else dk,
        dva if dva is not None else dv, HQ=hq, HK=hk, N=n, SM_SCALE=d**-0.5 if sm_scale is None else sm_scale,
        BM=config.block_m, BN=config.block_n, NW=config.num_warps, TDM=config.use_tdm, STAGES=config.stages,
        DELAYED=config.delayed, UNROLL=config.unroll, FRAGMENT_FENCES=config.fragment_fences,
        FRAGMENT_FENCE_MASK=config.fragment_fence_mask, FRAGMENTED=config.fragmented, RESIDENT_KV=config.resident_kv,
        ORDERED=config.ordered, PACKED_DQ=config.packed_dq, REMATERIALIZE=config.rematerialize, OUTER_M=config.outer_m,
        ASYNC_STATS=config.async_stats, Q_SPLITS=config.q_splits, REUSE_Q=config.reuse_q, PINGPONG=config.pingpong,
        SPREAD_DQ=config.spread_dq, LOG2_LSE=config.log2_lse, CACHE_SCORE_K=config.cache_score_k,
        TILE_MAJOR=config.tile_major, PARTITION_LDS=config.partition_lds, FRONT_REGIONS=config.front_regions,
        BRIDGE_REGIONS=config.bridge_regions, CLUSTER_DQ=config.cluster_dq, DQ_SLICES=dq_slices, OCC1=config.occ1,
        OCC1_K_BANDS=config.occ1_k_bands, OCC1_PREFETCH=config.occ1_prefetch, OCC1_SCHED=config.occ1_sched,
        PHASE_IGLP=config.phase_iglp, SPLIT_KV_STORES=config.occ1 and config.packed_dq and config.q_splits > 1,
        TAIL_DK=dk if config.tail_splits > 1 else None, TAIL_DV=dv if config.tail_splits > 1 else None,
        TAIL_SPLITS=config.tail_splits, TAIL_BASE=tail_base, TAIL_OWNERS=tail_owners,
        OCC1_DQ_FIRST=config.occ1_dq_first, OCC1_LOAD_AFTER_DK=config.occ1_load_after_dk, **compiler_options)
    if config.tail_splits > 1:
        convert_size = tail_owners * config.block_n * d
        convert_dk = dk.reshape(-1)[tail_base * config.block_n * d:]
        convert_dv = dv.reshape(-1)[tail_base * config.block_n * d:]
        kv_slices = config.tail_splits
    else:
        convert_size = k.numel()
        convert_dk, convert_dv = dk, dv
        kv_slices = config.q_splits if config.occ1 and config.packed_dq else 1
    _convert_outputs[(triton.cdiv(q.numel(),
                                  config.convert_block), )](dqa, dq, dka if dka is not None else dqa,
                                                            dva if dva is not None else dqa, convert_dk, convert_dv,
                                                            q.numel(), convert_size,
                                                            d**-0.5 if sm_scale is None else sm_scale,
                                                            config.convert_block, PACKED=config.packed_dq,
                                                            PARTIALS=partials, DQ_SLICES=dq_slices, SCALE_DQ=config.occ1
                                                            and not config.packed_dq, KV_SLICES=kv_slices, num_warps=4)
    return dq, dk, dv
