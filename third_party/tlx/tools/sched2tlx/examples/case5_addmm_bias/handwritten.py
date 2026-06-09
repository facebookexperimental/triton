"""Hand-written TLX-WS target for case5 (addmm_persistent_2d_bias).

Persistent GEMM with a 2D bias add in the epilogue. The bias TMA load is
INDEPENDENT of the GEMM critical path (A/B load → MMA), so the partition can
overlap it with the K-loop. This file shows the structural target sched2tlx
should converge to:

  default:  for tile_id: load bias (TMA→SMEM), wait acc full, addmm+truncf, store
  TC:       for tile_id: wait acc empty; K-loop {wait A,B full; MMA; arrive A,B empty}
  MEM:      for tile_id: K-loop {wait A,B empty; TMA load A,B}

The bias load fits in either default's body or its own MEM2 partition; the
schedule_graph today puts the bias load on wg=0 (same as A/B), which is
sub-optimal — that's a modulo cost-model question, not an emitter question.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def addmm_persistent_2d_bias(
    a_desc,
    b_desc,
    bias_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS_AB: tl.constexpr,
    NUM_BUFFERS_ACC: tl.constexpr,
):
    a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS_AB)
    b_smem = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS_AB)
    bias_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)
    acc_tmem = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_BUFFERS_ACC, tlx.storage_kind.tmem
    )
    c_smem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)

    a_full = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_AB, arrive_count=1)
    a_empty = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_AB, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_AB, arrive_count=1)
    b_empty = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_AB, arrive_count=1)
    bias_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    bias_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    acc_full = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_ACC, arrive_count=1)
    acc_empty = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_ACC, arrive_count=1)

    start_pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    k_tiles_n = (K + BLOCK_K - 1) // BLOCK_K

    with tlx.async_tasks():
        # ── default: bias load + correction (acc + bias) + epilogue store ──
        with tlx.async_task("default"):
            it = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                tmem_buf = it % NUM_BUFFERS_ACC
                tmem_phase = (it // NUM_BUFFERS_ACC) & 1
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N

                tlx.barrier_wait(bias_empty[0], (it & 1) ^ 1)
                tlx.barrier_expect_bytes(bias_full[0], BLOCK_M * BLOCK_N * 2)
                tlx.async_descriptor_load(
                    bias_desc, bias_smem[0], [offs_m, offs_n], bias_full[0]
                )
                tlx.barrier_wait(bias_full[0], it & 1)
                bias = tlx.local_load(bias_smem[0])
                tlx.barrier_arrive(bias_empty[0], 1)

                tlx.barrier_wait(acc_full[tmem_buf], tmem_phase)
                acc = tlx.local_load(acc_tmem[tmem_buf])
                tlx.barrier_arrive(acc_empty[tmem_buf], 1)

                out = (acc + bias.to(tl.float32)).to(tl.float16)
                tlx.local_store(c_smem[0], out)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(c_desc, c_smem[0], [offs_m, offs_n])
                tlx.async_descriptor_store_wait(0)
                it += 1

        # ── TC: K-loop MMAs into acc_tmem[tmem_buf] ──
        with tlx.async_task(num_warps=1, num_regs=24):
            smem_accum = 0
            it = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                tmem_buf = it % NUM_BUFFERS_ACC
                tmem_phase = (it // NUM_BUFFERS_ACC) & 1
                tlx.barrier_wait(acc_empty[tmem_buf], tmem_phase ^ 1)
                for k in range(k_tiles_n):
                    buf = smem_accum % NUM_BUFFERS_AB
                    phase = (smem_accum // NUM_BUFFERS_AB) & 1
                    tlx.barrier_wait(a_full[buf], phase)
                    tlx.barrier_wait(b_full[buf], phase)
                    use_acc = k > 0
                    tlx.async_dot(
                        a_smem[buf],
                        b_smem[buf],
                        acc_tmem[tmem_buf],
                        use_acc=use_acc,
                        mBarriers=[a_empty[buf], b_empty[buf]],
                    )
                    smem_accum += 1
                tlx.tcgen05_commit(acc_full[tmem_buf])
                it += 1

        # ── MEM: K-loop TMA loads for A and B ──
        with tlx.async_task(num_warps=1, num_regs=24):
            smem_accum = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                offs_am = pid_m * BLOCK_M
                offs_bn = pid_n * BLOCK_N
                for k in range(k_tiles_n):
                    buf = smem_accum % NUM_BUFFERS_AB
                    phase = (smem_accum // NUM_BUFFERS_AB) & 1
                    offs_k = k * BLOCK_K
                    tlx.barrier_wait(a_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[buf], BLOCK_M * BLOCK_K * 2)
                    tlx.async_descriptor_load(
                        a_desc, a_smem[buf], [offs_am, offs_k], a_full[buf]
                    )
                    tlx.barrier_wait(b_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[buf], BLOCK_K * BLOCK_N * 2)
                    tlx.async_descriptor_load(
                        b_desc, b_smem[buf], [offs_k, offs_bn], b_full[buf]
                    )
                    smem_accum += 1
