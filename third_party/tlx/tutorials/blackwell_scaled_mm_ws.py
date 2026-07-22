# TLX warp-specialized FP8 scaled_mm for Blackwell (SM100). Computes
# D = (A @ B^T) * scales in bf16 from A[M,K] / B[N,K] fp8 e4m3, for three scaling
# recipes (scale_mode=):
#   "blockwise" (DeepSeek): scale_a[M,K//128] M-major + scale_b[N//128,K//128]. The
#       scale changes along K, so each 128-wide K group's dot must be rescaled by
#       sa[m,g]*sb[n,g] BEFORE it is summed -- a per-group rescale-then-accumulate.
#   "rowwise":    scale_a[M] per-row, scale_b[N] per-col (constant along K).
#   "tensorwise": scalar scale_a, scale_b (constant along K).
#   The two K-independent recipes share one scale across the whole K reduction, so they
#   sum all of K in a single accumulator and scale once at the end -- far cheaper than
#   blockwise's per-group promotion.
#
# Work is split across specialized warps so each stays lean and the stages overlap:
#   LOAD   : bulk-copies A/B for one K group into SMEM.
#   SFLoad : streams the blockwise scales, one K group at a time (idle for row/tensorwise).
#   MMA    : runs the tensor-core dots, producing accumulator partials.
#   PROMO  : rescales / sums those partials and writes the output tile.
# Pipeline depths and tiling (NUM_SMEM_BUFFERS / NUM_ACC_BUFFERS / GROUP_SIZE_M /
# num_warps) are autotuned per (M, N, K, SCALE_MODE).

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

# The load/MMA warps do almost no arithmetic, so cap them at 48 registers and give the
# freed registers to the PROMO warpgroup, which needs them for the heavy fp32 math.
LOAD_REGS = tl.constexpr(48)
MMA_REGS = tl.constexpr(48)


def _autotune_configs():
    # Autotune only the in-kernel knobs (no host TMA-descriptor/grid effect -> no
    # pre_hook). EPI_SUB=2 / NUM_CTAS=1 are fixed; NUM_PROMO_WARPS must equal num_warps.
    # Curated around the square-GEMM optimum (4/4/8/8), kept small to bound first-call cost.
    specs = [(4, 4, 8, 8),  # proven optimum for square (e.g. 4096^3)
             (4, 4, 1, 8),  # GSM=1: better L2 for tall/skinny M
             (4, 4, 16, 8),  # GSM=16: wide-N shapes
             (4, 3, 8, 8),  # shallower acc pipeline
             (3, 4, 8, 8),  # shallower A/B pipeline (frees SMEM)
             (2, 2, 1, 4),  # small-shape: fewer buffers/warps
             ]
    return [
        triton.Config({"NUM_SMEM_BUFFERS": nsmem, "NUM_ACC_BUFFERS": nacc, "GROUP_SIZE_M": gsm, "NUM_PROMO_WARPS": nw},
                      num_warps=nw, num_stages=1) for (nsmem, nacc, gsm, nw) in specs
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K", "SCALE_MODE"])
@triton.jit
def _blackwell_scaled_mm_ws_kernel(
    a_desc,  # TMA desc for A [M, K] fp8, block [BLOCK_M, BLOCK_K]
    b_desc,  # TMA desc for B [N, K] fp8, block [BLOCK_N, BLOCK_K]
    c_desc,  # TMA desc for C [M, N] bf16, block [BLOCK_M, BLOCK_N]
    scale_a_ptr,  # [M, K//128] fp32, M-major (stride (1, M))
    scale_b_ptr,  # [N//128, K//128] fp32, row-major
    M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # == 128 (one DeepSeek K-group per iteration)
    GROUP_SIZE_M: tl.constexpr, NUM_SMEM_BUFFERS: tl.constexpr,  # A/B load pipeline depth
    NUM_ACC_BUFFERS: tl.constexpr,  # TMEM accumulator pipeline depth (autotuned)
    NUM_SMS: tl.constexpr, NUM_CTAS: tl.constexpr = 1,  # 2 -> a pair of SMs cooperate on one MMA
    USE_CLC: tl.constexpr = False,  # True -> hardware dynamic-persistent tile scheduler;
    # wins on ragged/grouped/variable-M work. Default off (static grid-stride).
    EPI_SUB: tl.constexpr = 2,  # split the N tile into this many epilogue sub-tiles
    NUM_PROMO_WARPS: tl.constexpr = 8,  # = num_warps; for acc_empty warp-barrier (per-thread arrive)
    SCALE_MODE: tl.constexpr = 0,  # 0=BLOCKWISE (K-dependent, per-group promotion);
    # 1=ROWWISE and 2=TENSORWISE (both K-independent: accumulate ALL K in one accumulator,
    # then a single scaled epilogue -- rowwise out=acc*sa[m]*sb[n], tensorwise out=acc*sa*sb).
):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Pad num_pid_m to a multiple of NUM_CTAS so a cluster's paired CTAs get
    # adjacent M-tiles with the SAME pid_n (needed for B-multicast / 2-SM MMA).
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_tiles = num_pid_m * num_pid_n
    k_groups = tl.cdiv(K, BLOCK_K)  # number of 128-K groups
    num_k_groups_scale = K // 128  # stride unit for scale_b row-major

    start_pid = tl.program_id(axis=0)

    # ---- Cluster setup (2-CTA) ----
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        # CTA0 waits for both CTAs to finish loading their B-half before the 2-SM MMA.
        cta_bars = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=2)
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # TLX two_ctas is N-split ONLY. The 2 SMs split the output N of a shared-A tile, each holding
    # half of B.
    BLOCK_N_CTA: tl.constexpr = BLOCK_N // NUM_CTAS  # each CTA loads/holds this N-slice of B

    # ---- SMEM operand buffers (A/B), multi-buffered load pipeline ----
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_N_CTA, BLOCK_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)

    # ---- TMEM accumulator pipeline: NUM_ACC_BUFFERS rotating [BLOCK_M,BLOCK_N] fp32 slots ----
    acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, NUM_ACC_BUFFERS, storage=tlx.storage_kind.tmem)
    # ---- Sub-tiled promotion: handle the N tile in EPI_SUB pieces so only one sub-tile's
    # fp32 partials are live at a time, keeping register/SMEM pressure down. ----
    SUB_N: tl.constexpr = BLOCK_N // EPI_SUB
    # SMEM store buffers, one per sub-tile (multi-buffered TMA store)
    c_smem = tlx.local_alloc((BLOCK_M, SUB_N), tlx.dtype_of(c_desc), EPI_SUB)

    # ---- Scale (SF) SMEM pipeline: stream the blockwise scales one K group at a time,
    # NUM_SF_BUF slots deep, so scale SMEM stays small regardless of K. Holding every K
    # group's scales at once would grow with K and overflow SMEM on large-K problems.
    NUM_SF_BUF: tl.constexpr = 4  # K-group pipeline depth
    sa_smem = tlx.local_alloc((BLOCK_M, ), tl.float32, NUM_SF_BUF)
    sb_smem = tlx.local_alloc((1, ), tl.float32, NUM_SF_BUF)

    # ---- Barriers ----
    # A/B load pipeline: producer(load) -> consumer(mma)
    ab_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)  # signalled when A/B are loaded
    ab_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)  # mma releases
    # scale (SF) pipeline: producer(sf_load) -> consumer(promo), per K-group.
    sf_full = tlx.alloc_barriers(NUM_SF_BUF, arrive_count=1)  # signalled when the scales are loaded
    # sf_empty: a per-thread barrier -- each PROMO thread signals only after its own scale
    # read, so SFLoad can't overwrite a slot while a slower reader is still using it (and it
    # avoids a full-warpgroup sync).
    sf_empty = tlx.alloc_warp_barrier(NUM_SF_BUF, num_warps=NUM_PROMO_WARPS, num_arrivals=1)
    # accumulator pipeline: producer(mma) -> consumer(promo)
    acc_full = tlx.alloc_barriers(NUM_ACC_BUFFERS, arrive_count=1)  # mma commits a slot
    # acc_empty: a per-thread barrier -- each PROMO thread frees the accumulator slot right
    # after its own read, so MMA can refill it without waiting on a full-warpgroup sync.
    acc_empty = tlx.alloc_warp_barrier(NUM_ACC_BUFFERS, num_warps=NUM_PROMO_WARPS, num_arrivals=1)

    dsize_a: tl.constexpr = tlx.size_of(tlx.dtype_of(a_desc))
    dsize_b: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))

    # ---- Optional dynamic-persistent tile scheduler ----
    # A dedicated scheduler warp asks the hardware for the next tile and hands it to the
    # four work warps, so tiles are dealt out on demand instead of by a fixed stride --
    # this keeps the SMs balanced when tiles do uneven work (ragged / grouped / variable-M).
    clc_multi_ctas: tl.constexpr = NUM_CTAS == 2
    if USE_CLC:
        clc_context = tlx.clc_create_context(num_consumers=5)

    with tlx.async_tasks():
        # ============ SCHED warp (hands out the next tile to run) ============
        if USE_CLC:
            with tlx.async_task(num_warps=1, num_regs=LOAD_REGS):
                tile_id = start_pid
                clc_phase_producer = 1
                clc_phase_consumer = 0
                while tile_id != -1:
                    # Request the next tile, then read the answer (-1 means "no more, stop").
                    # This warp does no math, so the request latency hides behind the other
                    # warps' compute on the current tile.
                    tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=clc_multi_ctas)
                    clc_phase_producer ^= 1
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=clc_multi_ctas)
                    clc_phase_consumer ^= 1

        # ================= LOAD warp (TMA A/B per K-group) =================
        with tlx.async_task(num_warps=1, num_regs=LOAD_REGS):
            ld_cnt = 0
            tile_id = start_pid
            clc_phase = 0
            while tile_id != -1:
                pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_M
                # each CTA loads its N-slice of B; the pair covers the full BLOCK_N
                offs_bn = pid_n * BLOCK_N + cluster_cta_rank * BLOCK_N_CTA
                for g in range(0, k_groups):
                    buf, phase = get_bufidx_phase(ld_cnt, NUM_SMEM_BUFFERS)
                    offs_k = g * BLOCK_K
                    # wait slot empty (mma done with it) — phase^1 on first round
                    tlx.barrier_wait(ab_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(
                        ab_full[buf],
                        dsize_a * BLOCK_M * BLOCK_K + dsize_b * BLOCK_N_CTA * BLOCK_K,
                    )
                    tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, offs_k], ab_full[buf])
                    tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_bn, offs_k], ab_full[buf])
                    ld_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase, multi_ctas=clc_multi_ctas)
                    clc_phase ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # ============ SFLoad warp (loads the blockwise scales) ============
        # A separate warp for the scale loads keeps their global-memory latency hidden and
        # off PROMO's critical path.
        with tlx.async_task(num_warps=1, num_regs=LOAD_REGS):
            sf_cnt = 0
            tile_id = start_pid
            clc_phase = 0
            while tile_id != -1:
                pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_M
                # which 128-wide scale_b N-block this tile falls in. BLOCK_N==128 -> nblk==pid_n;
                # BLOCK_N<128 -> adjacent tiles share (re-read) the same block. (constexpr fold)
                nblk = pid_n * BLOCK_N // 128
                if SCALE_MODE == 0:  # BLOCKWISE: pipeline per-(row,K-group) sa + per-block sb.
                    # ROWWISE loads its tiny K-independent sa/sb directly in PROMO.
                    for g in range(0, k_groups):
                        sf_buf, sf_phase = get_bufidx_phase(sf_cnt, NUM_SF_BUF)
                        tlx.barrier_wait(sf_empty[sf_buf], sf_phase ^ 1)  # slot drained by PROMO
                        # sb is one scalar per group; write it to SMEM and fence before the sa
                        # bulk load so both scales become visible to PROMO together.
                        sb_val = tl.load(scale_b_ptr + nblk * num_k_groups_scale + g + tl.arange(0, 1))
                        tlx.local_store(sb_smem[sf_buf], sb_val)
                        tlx.fence_async_shared()
                        tlx.barrier_expect_bytes(sf_full[sf_buf], 4 * BLOCK_M)
                        tlx.async_load(scale_a_ptr + offs_am + g * M, sa_smem[sf_buf], bulk=True,
                                       barrier=sf_full[sf_buf])
                        sf_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase, multi_ctas=clc_multi_ctas)
                    clc_phase ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # ================= MMA warp (producer of acc pipeline) =================
        with tlx.async_task(num_warps=1, num_regs=MMA_REGS):
            ld_cnt = 0
            acc_cnt = 0
            tile_id = start_pid
            clc_phase = 0
            while tile_id != -1:
                if SCALE_MODE == 0:
                    # BLOCKWISE: every K group has its own scale, so its dot must be kept
                    # separate rather than summed in the tensor core. Write each group's
                    # product to its own accumulator slot; PROMO rescales it, then sums.
                    for g in range(0, k_groups):
                        ld_buf, ld_phase = get_bufidx_phase(ld_cnt, NUM_SMEM_BUFFERS)
                        ac_buf, ac_phase = get_bufidx_phase(acc_cnt, NUM_ACC_BUFFERS)
                        # wait operands loaded
                        tlx.barrier_wait(ab_full[ld_buf], ld_phase)
                        # wait this acc slot is free (promo drained the one N groups ago)
                        tlx.barrier_wait(acc_empty[ac_buf], ac_phase ^ 1)
                        # 2-SM MMA: both CTAs signal they've loaded their B-half, CTA0 waits
                        # for both before issuing the pair MMA (reads the peer's B-half too).
                        if NUM_CTAS == 2:
                            tlx.barrier_arrive(cta_bars[ld_buf], arrive_count=1, remote_cta_rank=0)
                            tlx.barrier_wait(cta_bars[ld_buf], phase=ld_phase, pred=pred_cta0)
                        # FRESH MMA (use_acc=False): overwrite the slot with this group's dot.
                        # mBarriers=[ab_empty] releases the operand slot when MMA has read it.
                        tlx.async_dot(
                            buffers_A[ld_buf],
                            tlx.local_trans(buffers_B[ld_buf]),  # B is [N,K]; dot wants A[M,K]@B[K,N]
                            acc_tmem[ac_buf],
                            use_acc=False,
                            mBarriers=[ab_empty[ld_buf]],
                            two_ctas=NUM_CTAS == 2,
                            out_dtype=tl.float32,
                        )
                        # commit: signal promo this slot is full. tcgen05_commit makes
                        # acc_full track MMA completion (separate bar from the dot).
                        tlx.tcgen05_commit(acc_full[ac_buf])
                        ld_cnt += 1
                        acc_cnt += 1
                else:
                    # ROWWISE (K-independent scale): accumulate ALL K into ONE acc slot
                    # (use_acc=True after g=0), commit ONCE per tile. Scale applied once
                    # in the PROMO epilogue -> no per-group promotion, like a plain GEMM.
                    ac_buf, ac_phase = get_bufidx_phase(acc_cnt, NUM_ACC_BUFFERS)
                    tlx.barrier_wait(acc_empty[ac_buf], ac_phase ^ 1)  # one slot per tile
                    for g in range(0, k_groups):
                        ld_buf, ld_phase = get_bufidx_phase(ld_cnt, NUM_SMEM_BUFFERS)
                        tlx.barrier_wait(ab_full[ld_buf], ld_phase)
                        tlx.async_dot(
                            buffers_A[ld_buf],
                            tlx.local_trans(buffers_B[ld_buf]),
                            acc_tmem[ac_buf],
                            use_acc=(g != 0),  # accumulate across the whole K
                            mBarriers=[ab_empty[ld_buf]],
                            out_dtype=tl.float32,
                        )
                        ld_cnt += 1
                    tlx.tcgen05_commit(acc_full[ac_buf])
                    acc_cnt += 1
                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase, multi_ctas=clc_multi_ctas)
                    clc_phase ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

        # ============ PROMOTION warpgroup (rescale + accumulate + store) ============
        # The "default" task is the full num_warps warpgroup; it inherits the registers the
        # lean load/MMA warps gave up and spends them on the fp32 rescale math.
        with tlx.async_task("default"):
            acc_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            sf_cnt = 0
            while tile_id != -1:
                # PROMO just reads the next tile id; the scheduler warp requests it.
                pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_M
                offs_bn = pid_n * BLOCK_N

                if SCALE_MODE == 0:
                    # ===================== BLOCKWISE promotion =====================
                    # One [BLOCK_M, SUB_N] register running-sum per N sub-tile (EPI_SUB
                    # of them), loop-carried across the K-group loop.
                    accs = [tl.zeros((BLOCK_M, SUB_N), dtype=tl.float32) for _ in range(EPI_SUB)]
                    for g in range(0, k_groups):
                        ac_buf, ac_phase = get_bufidx_phase(acc_cnt, NUM_ACC_BUFFERS)
                        # Scales pipelined per K-group by SFLoad; slot released below once read.
                        sf_buf, sf_phase = get_bufidx_phase(sf_cnt, NUM_SF_BUF)
                        tlx.barrier_wait(sf_full[sf_buf], sf_phase)
                        sa = tlx.local_load(sa_smem[sf_buf])  # [BLOCK_M]
                        sb = tlx.local_load(sb_smem[sf_buf])  # [1]
                        sasb = sa * sb  # [BLOCK_M] (broadcast [1])
                        tlx.barrier_wait(acc_full[ac_buf], ac_phase)
                        # Read the partial straight from tensor memory; the deep accumulator
                        # pipeline already hides this read's latency, so overlapping it by
                        # hand would buy nothing.
                        ps = [
                            tlx.local_load(tlx.local_slice(acc_tmem[ac_buf], [0, s * SUB_N], [BLOCK_M, SUB_N]))
                            for s in range(EPI_SUB)
                        ]
                        accs = [accs[s] + ps[s] * sasb[:, None] for s in range(EPI_SUB)]
                        tlx.barrier_arrive(acc_empty[ac_buf])  # per-thread, no warpgroup sync
                        # free the scale slot now that sa/sb are consumed -- per-thread, so a
                        # slower reader is never overwritten.
                        tlx.barrier_arrive(sf_empty[sf_buf])  # per-thread, no warpgroup sync
                        acc_cnt += 1
                        sf_cnt += 1
                    # sub-tiled store, deferred TMA-store wait (double-buffered c_smem):
                    # this tile's stores drain during the NEXT tile's compute.
                    for s in tl.static_range(EPI_SUB):
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(c_smem[s], accs[s].to(tlx.dtype_of(c_desc)))
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(c_desc, c_smem[s], [offs_am, offs_bn + s * SUB_N])
                else:
                    # =============== ROWWISE / TENSORWISE epilogue (K-independent) ===============
                    # Single accumulator (all K already summed by MMA). Apply the scale ONCE:
                    #   SCALE_MODE==1 ROWWISE:    out[m,n] = acc * sa[m] * sb[n] (per-row/col).
                    #   SCALE_MODE==2 TENSORWISE: out[m,n] = acc * (sa * sb)     (two scalars).
                    ac_buf, ac_phase = get_bufidx_phase(acc_cnt, NUM_ACC_BUFFERS)
                    if SCALE_MODE == 1:
                        off_m = offs_am + tl.arange(0, BLOCK_M)
                        sa = tl.load(scale_a_ptr + off_m, mask=off_m < M, other=0.0)  # [BLOCK_M] per-row
                    else:
                        sa_sc = tl.load(scale_a_ptr) * tl.load(scale_b_ptr)  # scalar sa*sb (tensorwise)
                    tlx.barrier_wait(acc_full[ac_buf], ac_phase)
                    for s in tl.static_range(EPI_SUB):
                        ps = tlx.local_load(tlx.local_slice(acc_tmem[ac_buf], [0, s * SUB_N], [BLOCK_M, SUB_N]))
                        if SCALE_MODE == 1:
                            off_n = offs_bn + s * SUB_N + tl.arange(0, SUB_N)
                            sb = tl.load(scale_b_ptr + off_n, mask=off_n < N, other=0.0)  # [SUB_N] per-col
                            out = ps * sa[:, None] * sb[None, :]
                        else:
                            out = ps * sa_sc  # tensorwise: single scalar broadcast
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(c_smem[s], out.to(tlx.dtype_of(c_desc)))
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(c_desc, c_smem[s], [offs_am, offs_bn + s * SUB_N])
                    tlx.barrier_arrive(acc_empty[ac_buf])  # slot free after all TMEM loads
                    acc_cnt += 1

                if USE_CLC:
                    tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer, multi_ctas=clc_multi_ctas)
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                    if tile_id >= num_tiles:
                        tile_id = -1

            # drain the last tile's in-flight TMA stores before the kernel exits
            tlx.async_descriptor_store_wait(0)


@triton.jit
def _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


def blackwell_scaled_mm_ws(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, BLOCK_M=None, BLOCK_N=128, NUM_CTAS=1,
                           USE_CLC=False, EPI_SUB=2, persistent=True, scale_mode="blockwise"):
    # NOTE: NUM_SMEM_BUFFERS, NUM_ACC_BUFFERS, GROUP_SIZE_M, num_warps are now
    # @triton.autotune-managed (keyed on M,N,K,SCALE_MODE) -- see _autotune_configs().
    # NUM_CTAS=1 only for the autotuned path (autotune configs carry no ctas_per_cga).
    """
    a: [M,K] fp8_e4m3, b: [N,K] fp8_e4m3 (row-major, computes A @ B^T)
    scale_mode="blockwise":  scale_a [M, K//128] M-major, scale_b [N//128, K//128] row-major (DeepSeek).
    scale_mode="rowwise":    scale_a [M] (or [M,1]) per-row, scale_b [N] (or [1,N]) per-col; K-independent.
    scale_mode="tensorwise": scale_a, scale_b are scalars (one fp32 each); K-independent.
    NUM_CTAS=2 -> cluster of 2 SMs cooperate on the MMA (blockwise only).
    BLOCK_M=None -> occupancy-aware M tile (64 on small shapes, 128 otherwise); pass an int to force.
    """
    SCALE_MODE = {"blockwise": 0, "rowwise": 1, "tensorwise": 2}[scale_mode]
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb and K % 128 == 0 and N % 128 == 0
    # Blockwise uses one 128-wide scale_b block per N tile, so a tile must fit within a
    # single block -> BLOCK_N must divide 128 (BLOCK_N<128 tiles re-read the shared block).
    assert SCALE_MODE != 0 or 128 % BLOCK_N == 0, "blockwise requires BLOCK_N to divide 128"
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    if SCALE_MODE == 1:
        # rowwise: kernel reads scale_a[off_m], scale_b[off_n] directly -> need contiguous 1-D.
        scale_a = scale_a.reshape(-1).contiguous()
        scale_b = scale_b.reshape(-1).contiguous()
        assert scale_a.numel() == M and scale_b.numel() == N, (
            f"rowwise expects scale_a [M]={M}, scale_b [N]={N}; got {scale_a.numel()}, {scale_b.numel()}")
    elif SCALE_MODE == 2:
        # tensorwise: kernel reads scale_a[0], scale_b[0] -> need contiguous 1-element tensors.
        scale_a = scale_a.reshape(-1).contiguous()
        scale_b = scale_b.reshape(-1).contiguous()
        assert scale_a.numel() == 1 and scale_b.numel() == 1, (
            f"tensorwise expects scalar scale_a/scale_b; got {scale_a.numel()}, {scale_b.numel()}")

    BLOCK_K = 128
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Occupancy-aware M tile. The 128x128 tile leaves most SMs idle on small problems
    # (e.g. 1024^3 -> 64 tiles on ~148 SMs, <50% of the GPU). A 64-tall tile doubles the
    # tile count to fill those SMs -- but only when the doubled count still fits in ONE
    # wave, so we don't trade idle SMs for a second, poorly-occupied wave. Larger shapes
    # keep 128, whose higher MMA efficiency wins once the GPU is already full.
    if BLOCK_M is None:
        num_pid_n = triton.cdiv(N, BLOCK_N)
        tiles_128 = triton.cdiv(M, 128) * num_pid_n
        tiles_64 = triton.cdiv(M, 64) * num_pid_n
        BLOCK_M = 64 if (tiles_128 < num_sms and tiles_64 <= num_sms) else 128

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device=a.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_M, BLOCK_K])
    # each CTA TMA-loads its N-slice of B (two_ctas is N-split; the pair covers full BLOCK_N)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_N // NUM_CTAS, BLOCK_K])
    # c_desc block = sub-tile width (epilogue stores N in EPI_SUB pieces)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_M, BLOCK_N // EPI_SUB])

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_tiles = num_pid_m * triton.cdiv(N, BLOCK_N)
    if USE_CLC:
        # CLC dynamic-persistent: launch ALL tiles; the hardware scheduler cancels
        # the unstarted CTAs and hands their tiles to the ~num_sms that actually run.
        grid = (num_tiles, )
        stride_sms = num_sms
    elif persistent:
        # static persistent grid-stride: launch min(SMs, tiles), each strides by NUM_SMS.
        grid = (min(num_sms // NUM_CTAS * NUM_CTAS, num_tiles), )
        stride_sms = grid[0]
    else:
        # non-persistent: one CTA per tile (NUM_SMS==num_tiles ends the grid-stride after one
        # tile). Slower here -- the warp-specialized prologue (register donation + filling the
        # deep pipeline) only pays off when amortized over many tiles per CTA. Kept for
        # experimentation.
        grid = (num_tiles, )
        stride_sms = num_tiles
    # GROUP_SIZE_M, NUM_SMEM_BUFFERS, NUM_ACC_BUFFERS, NUM_PROMO_WARPS (+ launch
    # num_warps) are injected by @triton.autotune from the selected Config.
    _blackwell_scaled_mm_ws_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        scale_a,
        scale_b,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_SMS=stride_sms,
        NUM_CTAS=NUM_CTAS,
        USE_CLC=USE_CLC,
        EPI_SUB=EPI_SUB,
        SCALE_MODE=SCALE_MODE,
    )
    return c


# --------------------------------------------------------------------------
# Correctness harness (fp32 reference)
# --------------------------------------------------------------------------
def _ref(a, b, scale_a, scale_b):
    M, K = a.shape
    N = b.shape[0]
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    G = K // 128
    for g in range(G):
        ak = af[:, g * 128:(g + 1) * 128]
        bk = bf[:, g * 128:(g + 1) * 128]
        partial = ak @ bk.t()  # [M,N]
        sa = scale_a[:, g][:, None]  # [M,1]
        sb = scale_b[:, g].repeat_interleave(128)[None, :]  # [1,N]
        out += partial * sa * sb
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    M = N = K = 4096
    dev = "cuda"
    a = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    # scale_a M-major (outer-dim-major), scale_b row-major
    scale_a = torch.rand(M, K // 128, device=dev, dtype=torch.float32).t().contiguous().t()
    scale_b = torch.rand(N // 128, K // 128, device=dev, dtype=torch.float32)

    out = blackwell_scaled_mm_ws(a, b, scale_a, scale_b)
    ref = _ref(a, b, scale_a, scale_b).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=0.05)
    print("OK blockwise: max abs err", (out.float() - ref.float()).abs().max().item())

    # ---- rowwise: out[m,n] = (A@B^T)[m,n] * sa[m] * sb[n] (K-independent) ----
    sa_row = torch.rand(M, device=dev, dtype=torch.float32)
    sb_row = torch.rand(N, device=dev, dtype=torch.float32)
    out_r = blackwell_scaled_mm_ws(a, b, sa_row, sb_row, scale_mode="rowwise")
    ref_r = ((a.to(torch.float32) @ b.to(torch.float32).t()) * sa_row[:, None] * sb_row[None, :]).to(torch.bfloat16)
    torch.testing.assert_close(out_r, ref_r, atol=1e-1, rtol=0.05)
    print("OK rowwise:   max abs err", (out_r.float() - ref_r.float()).abs().max().item())

    # ---- tensorwise: out[m,n] = (A@B^T)[m,n] * sa * sb (two scalars) ----
    sa_t = torch.rand(1, device=dev, dtype=torch.float32)
    sb_t = torch.rand(1, device=dev, dtype=torch.float32)
    out_t = blackwell_scaled_mm_ws(a, b, sa_t, sb_t, scale_mode="tensorwise")
    ref_t = ((a.to(torch.float32) @ b.to(torch.float32).t()) * sa_t * sb_t).to(torch.bfloat16)
    torch.testing.assert_close(out_t, ref_t, atol=1e-1, rtol=0.05)
    print("OK tensorwise:max abs err", (out_t.float() - ref_t.float()).abs().max().item())
