# TLX warp-specialized FP8 scaled_mm for Blackwell (SM100), mirroring the CUTLASS
# KernelTmaWarpSpecialized structure. Computes D = (A @ B^T) * scales in bf16 from
# A[M,K] / B[N,K] fp8 e4m3, for three scaling recipes (scale_mode=):
#   "blockwise" (DeepSeek): scale_a[M,K//128] M-major + scale_b[N//128,K//128]. The
#       scale is K-dependent, so each 128-K group's dot is rescaled by sa[m,g]*sb[n,g]
#       and summed in a pipelined per-group promotion.
#   "rowwise":    scale_a[M] per-row, scale_b[N] per-col (K-independent).
#   "tensorwise": scalar scale_a, scale_b (K-independent).
#   K-independent modes accumulate all K in one TMEM accumulator and apply the scale
#   once in the epilogue.
#
# Warp-specialized tasks (register-donation split, CUTLASS-style):
#   LOAD   : TMA-loads A/B per 128-K group into SMEM.
#   SFLoad : stages blockwise scales in SMEM (no-op for rowwise/tensorwise).
#   MMA    : producer of the NUM_ACC_BUFFERS-deep accumulator pipeline.
#   PROMO  : consumer -- rescales / accumulates the partials and stores.
# NUM_SMEM_BUFFERS / NUM_ACC_BUFFERS / GROUP_SIZE_M / num_warps are @triton.autotune-
# managed (keyed on M, N, K, SCALE_MODE).

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

# Register split (mirrors CUTLASS setmaxnreg): the lean load/MMA warps cap at 48
# regs and donate the surplus to the promotion warpgroup (the "default" task),
# which inherits it implicitly for the heavy fp32 math.
LOAD_REGS = tl.constexpr(48)
MMA_REGS = tl.constexpr(48)


def _autotune_configs():
    # Autotune the pure IN-KERNEL knobs (no effect on host-side TMA descriptors or
    # grid, so no pre_hook needed). EPI_SUB / NUM_CTAS stay fixed at their proven
    # optima (EPI_SUB=2, NUM_CTAS=1). NUM_PROMO_WARPS must equal the launch num_warps.
    # Curated list centred on the square-GEMM optimum (4/4/8/8) + diversity for other
    # shapes; keep small to bound first-call benchmarking cost.
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
    NUM_SMS: tl.constexpr, NUM_CTAS: tl.constexpr = 1,  # 2 -> cluster of 2 SMs cooperate (CUTLASS 2-SM)
    USE_CLC: tl.constexpr = False,  # True -> CLC dynamic-persistent scheduler (CUTLASS Sched
    # warp); wins on ragged/grouped/variable-M. Default off (static grid-stride).
    EPI_SUB: tl.constexpr = 2,  # N sub-tiles for the promotion/epilogue (CUTLASS epi_n loop)
    K_GROUPS: tl.constexpr = 1,  # = K//BLOCK_K (constexpr): scales staged once per tile
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

    # TLX two_ctas is N-split ONLY (semantic.py: rhs must be [K, N/2], output N is
    # doubled back): the 2 SMs split the output N of a shared-A tile, each holding
    # half of B. CUTLASS's M-split (shared/multicast B) is NOT expressible here.
    BLOCK_N_CTA: tl.constexpr = BLOCK_N // NUM_CTAS  # each CTA loads/holds this N-slice of B

    # ---- SMEM operand buffers (A/B), multi-buffered load pipeline ----
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_N_CTA, BLOCK_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)

    # ---- TMEM accumulator pipeline: NUM_ACC_BUFFERS rotating [BLOCK_M,BLOCK_N] fp32 slots ----
    acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, NUM_ACC_BUFFERS, storage=tlx.storage_kind.tmem)
    # ---- Sub-tiled promotion: N split into EPI_SUB pieces so only a sub-tile
    # partial is live at once (CUTLASS epilogue sub-tiling). EPI_SUB is a param. ----
    SUB_N: tl.constexpr = BLOCK_N // EPI_SUB
    # SMEM store buffers, one per sub-tile (multi-buffered TMA store)
    c_smem = tlx.local_alloc((BLOCK_M, SUB_N), tlx.dtype_of(c_desc), EPI_SUB)

    # ---- Scale (SF) SMEM pipeline: the SFLoad warp stages the WHOLE tile's scales
    # (all K_GROUPS) ONCE per tile; PROMO then reads sa[g]/sb[g] from SMEM per K-group
    # with NO per-K-group barrier -> only a per-TILE sf handshake (cuts ~2 warpgroup
    # BAR.SYNC per K-group in the 8-warp PROMO group). Flat buffer arrays indexed
    # [tbuf*K_GROUPS + g]; NUM_SF_BUF tile-buffers for cross-tile overlap.
    NUM_SF_BUF: tl.constexpr = 3
    sa_smem = tlx.local_alloc((BLOCK_M, ), tl.float32, NUM_SF_BUF * K_GROUPS)
    sb_smem = tlx.local_alloc((1, ), tl.float32, NUM_SF_BUF * K_GROUPS)

    # ---- Barriers ----
    # A/B load pipeline: producer(load) -> consumer(mma)
    ab_full = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)  # TMA fills (expect_bytes)
    ab_empty = tlx.alloc_barriers(NUM_SMEM_BUFFERS, arrive_count=1)  # mma releases
    # scale (SF) pipeline: producer(sf_load) -> consumer(promo), now PER-TILE
    sf_full = tlx.alloc_barriers(NUM_SF_BUF, arrive_count=1)  # cp.async.bulk fills (expect_bytes)
    sf_empty = tlx.alloc_barriers(NUM_SF_BUF, arrive_count=1)  # promo releases
    # accumulator pipeline: producer(mma) -> consumer(promo)
    acc_full = tlx.alloc_barriers(NUM_ACC_BUFFERS, arrive_count=1)  # mma commits a slot
    # acc_empty: PROMO releases each slot with a PER-THREAD (warp-barrier) arrive, so no
    # CTA BAR.SYNC. alloc_warp_barrier sets arrive_count = NUM_PROMO_WARPS*32; every thread
    # arrives independently (safe: each warp's TMEM read is fenced by its own tcgen05.wait::ld).
    acc_empty = tlx.alloc_warp_barrier(NUM_ACC_BUFFERS, num_warps=NUM_PROMO_WARPS, num_arrivals=1)

    dsize_a: tl.constexpr = tlx.size_of(tlx.dtype_of(a_desc))
    dsize_b: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))

    # ---- CLC dynamic-persistent tile scheduler (CUTLASS Sched warp) ----
    # A dedicated 1-warp SCHED partition (below) is the CLC pipeline producer (issues
    # try_cancel) AND a consumer (reads its own next tile to terminate); the 4 work
    # partitions (LOAD, SFLoad, MMA, PROMO) are the other consumers -> 5 CLC consumers.
    # Mirrors CUTLASS's WarpCategory::Sched. Wins on ragged/grouped/variable-M.
    clc_multi_ctas: tl.constexpr = NUM_CTAS == 2
    if USE_CLC:
        clc_context = tlx.clc_create_context(num_consumers=5)

    with tlx.async_tasks():
        # ============ SCHED warp (dedicated CLC producer, CUTLASS Sched) ============
        if USE_CLC:
            with tlx.async_task(num_warps=1, num_regs=LOAD_REGS):
                tile_id = start_pid
                clc_phase_producer = 1
                clc_phase_consumer = 0
                while tile_id != -1:
                    # issue the try_cancel for the NEXT tile, then read it (also our
                    # own termination signal). No GEMM work here -> the try_cancel
                    # latency overlaps the work warps' compute for this tile.
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

        # ============ SFLoad warp (cp.async.bulk scale_a -> SMEM) ============
        # Dedicated scale-load partition (CUTLASS MainloopSFLoad): keeps the scale
        # global load fully off the promotion warp's critical path.
        with tlx.async_task(num_warps=1, num_regs=LOAD_REGS):
            sf_tile = 0
            tile_id = start_pid
            clc_phase = 0
            while tile_id != -1:
                pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_M
                nblk = pid_n  # BLOCK_N==128 -> one N-block per tile; sb row = scale_b[nblk, :]
                if SCALE_MODE == 0:  # BLOCKWISE: stage per-(row,K-group) sa + per-block sb.
                    # ROWWISE loads its tiny K-independent sa/sb directly in PROMO.
                    tbuf, tphase = get_bufidx_phase(sf_tile, NUM_SF_BUF)
                    base = tbuf * K_GROUPS
                    tlx.barrier_wait(sf_empty[tbuf], tphase ^ 1)
                    # sb: K_GROUPS scalars via regular loads -> flat SMEM slots, fenced before
                    # the sa bulk signals sf_full. (These loads are in the SFLoad producer,
                    # already hidden by NUM_SF_BUF prefetch -> not on PROMO's critical path.)
                    for g in range(0, k_groups):
                        sb_val = tl.load(scale_b_ptr + nblk * num_k_groups_scale + g + tl.arange(0, 1))
                        tlx.local_store(sb_smem[base + g], sb_val)
                    tlx.fence_async_shared()
                    tlx.barrier_expect_bytes(sf_full[tbuf], 4 * BLOCK_M * K_GROUPS)
                    for g in range(0, k_groups):
                        tlx.async_load(scale_a_ptr + offs_am + g * M, sa_smem[base + g], bulk=True,
                                       barrier=sf_full[tbuf])
                sf_tile += 1
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
                    # BLOCKWISE: FRESH MMA per K-group into a rotating acc slot -> the
                    # producer of the NUM_ACC_BUFFERS-deep pipeline PROMO rescales+sums.
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

        # ============ PROMOTION warp group (consumer: rescale + accumulate + store) ============
        # "default" task = the launch num_warps warpgroup; it inherits the register
        # surplus donated by the lean load/MMA warps (CUTLASS setmaxnreg 256 analog).
        with tlx.async_task("default"):
            acc_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            sf_tile = 0
            while tile_id != -1:
                # PROMO is a CLC consumer only; the dedicated SCHED warp produces.
                pid_m, pid_n = _pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_M
                offs_bn = pid_n * BLOCK_N

                if SCALE_MODE == 0:
                    # ===================== BLOCKWISE promotion =====================
                    # One [BLOCK_M, SUB_N] register running-sum per N sub-tile (EPI_SUB
                    # of them), loop-carried across the K-group loop.
                    accs = [tl.zeros((BLOCK_M, SUB_N), dtype=tl.float32) for _ in range(EPI_SUB)]
                    # Scales staged in SMEM by SFLoad; wait ONCE per tile, then read
                    # sa[g]/sb[g] per K-group with NO barrier -> fewer warpgroup BAR.SYNC.
                    tbuf, tphase = get_bufidx_phase(sf_tile, NUM_SF_BUF)
                    base = tbuf * K_GROUPS
                    tlx.barrier_wait(sf_full[tbuf], tphase)
                    for g in range(0, k_groups):
                        ac_buf, ac_phase = get_bufidx_phase(acc_cnt, NUM_ACC_BUFFERS)
                        sa = tlx.local_load(sa_smem[base + g])  # [BLOCK_M]
                        sb = tlx.local_load(sb_smem[base + g])  # [1]
                        sasb = sa * sb  # [BLOCK_M] (broadcast [1])
                        tlx.barrier_wait(acc_full[ac_buf], ac_phase)
                        # Synchronous T2R load: overlapping it across K-groups (async-T2R or a
                        # manual SW pipeline) is neutral -- the acc pipeline + warp scheduler
                        # already hide the tcgen05.ld latency.
                        ps = [
                            tlx.local_load(tlx.local_slice(acc_tmem[ac_buf], [0, s * SUB_N], [BLOCK_M, SUB_N]))
                            for s in range(EPI_SUB)
                        ]
                        accs = [accs[s] + ps[s] * sasb[:, None] for s in range(EPI_SUB)]
                        tlx.barrier_arrive(acc_empty[ac_buf])  # warp barrier, no CTA BAR.SYNC
                        acc_cnt += 1
                    tlx.barrier_arrive(sf_empty[tbuf], 1)  # SFLoad may reload the sf buffer
                    sf_tile += 1
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


def blackwell_scaled_mm_ws(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, BLOCK_M=128, BLOCK_N=128, NUM_CTAS=1,
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
    """
    SCALE_MODE = {"blockwise": 0, "rowwise": 1, "tensorwise": 2}[scale_mode]
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb and K % 128 == 0 and N % 128 == 0
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
        # non-persistent (vLLM/CUTLASS model): one CTA per tile (NUM_SMS==num_tiles ends
        # each CTA's grid-stride after one tile). Slower here -- our WS prologue (register
        # donation + deep pipeline fill) only amortizes over many tiles per CTA. Persistent
        # is the default; kept for experimentation.
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
        K_GROUPS=K // BLOCK_K,  # scales staged once per tile
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
