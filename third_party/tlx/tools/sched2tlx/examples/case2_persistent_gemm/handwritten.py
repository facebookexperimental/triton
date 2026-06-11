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

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


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
):
    # ── Allocs (function scope) ──
    # A: [BM, BK] row-major (compatible with MMA's first operand layout)
    # B: [BN, BK] — TMA loads in this layout, transposed for MMA to [BK, BN]
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem)
    c_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_desc), 1)

    # ── Mbarriers ──
    a_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    a_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    b_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    with tlx.async_tasks():
        # ── default partition (epilogue, runs once per tile) ──
        with tlx.async_task("default"):
            tile_id = start_pid
            tmem_phase = 0
            while tile_id < num_tiles:
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                tlx.barrier_wait(tmem_full[0], tmem_phase)
                acc = tlx.local_load(acc_tmem[0])
                tlx.barrier_arrive(tmem_empty[0], 1)
                c = acc.to(tlx.dtype_of(c_desc))
                tlx.local_store(c_smem[0], c)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(c_desc, c_smem[0], [pid_m * BLOCK_M, pid_n * BLOCK_N])
                tlx.async_descriptor_store_wait(0)
                tile_id += NUM_SMS
                tmem_phase ^= 1

        # ── TC partition (MMA, K-loop nested inside persistent loop) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_wait(tmem_empty[0], 1)  # initial empty signal
            tile_id = start_pid
            smem_accum = 0
            tmem_phase = 0
            while tile_id < num_tiles:
                for k_iter in range(k_tiles):
                    buf = smem_accum % NUM_SMEM_BUFFERS
                    phase = (smem_accum // NUM_SMEM_BUFFERS) & 1
                    tlx.barrier_wait(a_full[buf], phase)
                    tlx.barrier_wait(b_full[buf], phase)
                    use_acc = k_iter > 0
                    # B was loaded as [BN, BK]; transpose for MMA → [BK, BN].
                    b_t = tlx.local_trans(smem_b[buf])
                    tlx.async_dot(smem_a[buf], b_t, acc_tmem[0], use_acc=use_acc)
                    tlx.barrier_arrive(a_empty[buf], 1)
                    tlx.barrier_arrive(b_empty[buf], 1)
                    smem_accum += 1
                tlx.barrier_arrive(tmem_full[0], 1)
                tile_id += NUM_SMS
                # Wait for epilogue to release TMEM before next tile.
                if tile_id < num_tiles:
                    tlx.barrier_wait(tmem_empty[0], tmem_phase ^ 1)
                tmem_phase ^= 1

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
                    buf = smem_accum % NUM_SMEM_BUFFERS
                    phase = (smem_accum // NUM_SMEM_BUFFERS) & 1
                    offs_k = k_iter * BLOCK_K
                    # Load A
                    tlx.barrier_wait(a_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[buf], BLOCK_M * BLOCK_K * 2)
                    tlx.async_descriptor_load(a_desc, smem_a[buf], [offs_am, offs_k], a_full[buf])
                    # Load B (as [BN, BK])
                    tlx.barrier_wait(b_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[buf], BLOCK_N * BLOCK_K * 2)
                    tlx.async_descriptor_load(b_desc, smem_b[buf], [offs_bn, offs_k], b_full[buf])
                    smem_accum += 1
                tile_id += NUM_SMS
