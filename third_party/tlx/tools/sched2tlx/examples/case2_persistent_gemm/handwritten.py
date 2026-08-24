"""Hand-written TLX target for case2 (persistent GEMM).

Mirrors users/wl/wlei/autows/modulo_schedule/case2_persistent_gemm:
  - BM=256, BN=256, BK=64, fp16 in / fp16 out, fp32 acc
  - 1 CTA, persistent over MN tiles, NUM_SMS=148 stride
  - A loaded as [BM, BK]; B loaded as [BN, BK] then transposed to [BK, BN] for MMA
  - 1 MMA per K-iter, single TMEM accumulator hand-off per tile

Modulo schedule (from case2_persistent_gemm/run.log):
  Inner II=1616, 2 SMEM buffers depth=2 (modulo's lifetime analysis)
  Outer (persistent): 2 WGs default (epilogue) + WG1 for everything else
  Inner: WG MEM (loads) + WG TC (MMA)

Structure modeled after blackwell_gemm_ws.py: smem_accum_cnt persists across
tiles so the SMEM ring buffer doesn't need to drain between tiles.
"""

import os

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def get_bufidx_phase(i, n: tl.constexpr):
    """Continuous ring counter -> (buffer index, phase). Because the counter
    runs unbroken across persistent tiles, producer and consumer agree on
    (buf, phase) for every step regardless of how the ring depth relates to
    k_tiles -- so a ring deeper than one tile's K-iterations is safe."""
    return i % n, (i // n) & 1


# Autotune ONLY the SMEM/TMEM ring depths (the same degrees of freedom the modulo
# scheduler chooses) so the comparison against the generated kernel is
# apples-to-apples; BLOCK_M/N/K, num_warps, and everything else stay fixed.
def _autotune_configs():
    # Measured best (do_bench on B200, square/representative shape). TLX_HW_BEST_CONFIG=1
    # uses it directly (fast path, no autotune search); otherwise the full depth grid
    # is swept. Mirrors the tutorial WS GEMM TLX_GEMM_USE_HEURISTIC pattern.
    if os.environ.get("TLX_HW_BEST_CONFIG") == "1":
        return [triton.Config({"NUM_SMEM_BUFFERS": 5, "NUM_TMEM_BUFFERS": 1}, num_warps=4, num_stages=2, num_ctas=1)]
    return [
        triton.Config({"NUM_SMEM_BUFFERS": s, "NUM_TMEM_BUFFERS": t}, num_warps=4, num_stages=2, num_ctas=1)
        for s in (2, 3, 4, 5, 6)
        for t in (1, 2)
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
):
    # ── Allocs (function scope) ──
    # A: [BM, BK] row-major (compatible with MMA's first operand layout)
    # B: [BN, BK] — TMA loads in this layout, transposed for MMA to [BK, BN]
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)
    c_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_desc), 1)

    # ── Mbarriers ──
    # A and B share one "empty" barrier per ring slot: the async_dot consumes
    # both operands, so one MMA-completion signal frees the whole slot (mirrors
    # the tutorial, which gates B reuse on A's empty barrier).
    a_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    ab_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    with tlx.async_tasks():
        # ── default partition (epilogue, runs once per tile) ──
        with tlx.async_task("default"):
            tile_id = start_pid
            tmem_accum = 0
            while tile_id < num_tiles:
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                cur_tmem, read_phase = get_bufidx_phase(tmem_accum, NUM_TMEM_BUFFERS)
                tlx.barrier_wait(tmem_full[cur_tmem], read_phase)
                acc = tlx.local_load(acc_tmem[cur_tmem])
                tlx.barrier_arrive(tmem_empty[cur_tmem], 1)
                c = acc.to(tlx.dtype_of(c_desc))
                tlx.local_store(c_smem[0], c)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(c_desc, c_smem[0], [pid_m * BLOCK_M, pid_n * BLOCK_N])
                tlx.async_descriptor_store_wait(0)
                tmem_accum += 1
                tile_id += NUM_SMS

        # ── TC partition (MMA, K-loop nested inside persistent loop) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tile_id = start_pid
            smem_accum = 0
            tmem_accum = 0
            while tile_id < num_tiles:
                cur_tmem, write_phase = get_bufidx_phase(tmem_accum, NUM_TMEM_BUFFERS)
                # Peeled first K-iter: reserve the accumulator slot (phase^1 makes
                # the very first use a no-op, so no explicit initial signal), then
                # clear it with use_acc=False.
                buf, phase = get_bufidx_phase(smem_accum, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(tmem_empty[cur_tmem], write_phase ^ 1)
                tlx.barrier_wait(a_full[buf], phase)
                tlx.barrier_wait(b_full[buf], phase)
                # B was loaded as [BN, BK]; transpose for MMA → [BK, BN]. The
                # empty barrier is arrived on MMA COMPLETION (via mBarriers), not
                # on issue, so the loader can't overwrite a slot still in use.
                b_t = tlx.local_trans(smem_b[buf])
                tlx.async_dot(smem_a[buf], b_t, acc_tmem[cur_tmem], use_acc=False, mBarriers=[ab_empty[buf]])
                smem_accum += 1
                for _ in range(1, k_tiles):
                    buf, phase = get_bufidx_phase(smem_accum, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(a_full[buf], phase)
                    tlx.barrier_wait(b_full[buf], phase)
                    b_t = tlx.local_trans(smem_b[buf])
                    tlx.async_dot(smem_a[buf], b_t, acc_tmem[cur_tmem], use_acc=True, mBarriers=[ab_empty[buf]])
                    smem_accum += 1
                # tmem_full is arrived when the tile's async MMAs actually drain.
                tlx.tcgen05_commit(tmem_full[cur_tmem])
                tmem_accum += 1
                tile_id += NUM_SMS

        # ── MEM partition (TMA loads, both A and B in one task) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tile_id = start_pid
            smem_accum = 0
            while tile_id < num_tiles:
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                offs_am = pid_m * BLOCK_M
                offs_bn = pid_n * BLOCK_N
                for k_iter in range(k_tiles):
                    buf, phase = get_bufidx_phase(smem_accum, NUM_SMEM_BUFFERS)
                    offs_k = k_iter * BLOCK_K
                    # One empty wait frees the whole slot; load A and B into it.
                    tlx.barrier_wait(ab_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[buf], BLOCK_M * BLOCK_K * 2)
                    tlx.async_descriptor_load(a_desc, smem_a[buf], [offs_am, offs_k], a_full[buf])
                    tlx.barrier_expect_bytes(b_full[buf], BLOCK_N * BLOCK_K * 2)
                    tlx.async_descriptor_load(b_desc, smem_b[buf], [offs_bn, offs_k], b_full[buf])
                    smem_accum += 1
                tile_id += NUM_SMS
