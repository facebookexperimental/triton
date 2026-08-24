"""Hand-written DATA-PARTITIONED TLX target for case2 (persistent GEMM).

This is `handwritten.py` extended with **data partitioning** so it can run a
real 256×256 output tile (`handwritten.py` is limited to 128×128 because a
single 256-row TMEM accumulator hits `blockM must be 64 or 128 but got 256`).

The 256-row tile is split along M into `NUM_MMA_GROUPS=2` groups of `GM=128`
rows. Each group has its OWN 128×256 TMEM accumulator (`acc_tmem[group]`), so
the pair occupies the full Blackwell TMEM (128 lanes × 512 cols = 2 × 256).

Differences from `handwritten.py` (kept as small as possible):
  * A is loaded per-group as [GM, BK]; `smem_a` is `NUM_SMEM_BUFFERS` buffers
    PER group, indexed `g*NUM_SMEM_BUFFERS + buf`. B ([BN, BK], transposed for
    MMA) is shared across both groups, exactly one ring of `NUM_SMEM_BUFFERS`.
  * Two TMEM accumulators `acc_tmem[0]`, `acc_tmem[1]` with per-group
    `tmem_full` / `tmem_empty` hand-off barriers.
  * The MMA waits `tmem_empty[group]` at the START of each tile (peeled first
    K-iter), phase = `tmem_write_phase ^ 1` from a per-tile counter. This is the
    one detail that makes the 2nd (column-offset) accumulator correct — placing
    the wait at the end of the loop instead races the epilogue.
  * B has no dedicated `empty` barrier: it is reused by both groups' MMAs, so it
    is safe to overwrite once the LAST group's `a_empty` fires (the MMA mBarrier
    signals when the dot has read BOTH its A and the shared B operand).

Mirrors blackwell_gemm_ws.py's per-group choreography, inlined and stripped of
autotuning / Split-K / 2-CTA / epilogue-subtiling so it reads like handwritten.py.
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
    NUM_MMA_GROUPS: tl.constexpr,
):
    GM: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS  # rows per MMA group (64 or 128)

    # ── Allocs (function scope) ──
    # A: per-group [GM, BK]; NUM_SMEM_BUFFERS buffers per group → idx g*NSB + buf.
    smem_a = tlx.local_alloc((GM, BLOCK_K), tlx.dtype_of(a_desc), NUM_SMEM_BUFFERS * NUM_MMA_GROUPS)
    # B: [BN, BK] shared across groups, transposed to [BK, BN] for MMA.
    smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_desc), NUM_SMEM_BUFFERS)
    # Per-group TMEM accumulators: acc_tmem[0] = rows 0..GM, acc_tmem[1] = rows GM..BM.
    acc_tmem = tlx.local_alloc((GM, BLOCK_N), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)
    # Single epilogue SMEM buffer, reused per group (keeps SMEM under the limit).
    c_smem = tlx.local_alloc((GM, BLOCK_N), tlx.dtype_of(c_desc), 1)

    # ── Mbarriers ──
    a_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    a_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1)
    b_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # Per-group TMEM hand-off barriers (index = MMA group).
    tmem_full = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    with tlx.async_tasks():
        # ── default partition (epilogue, per group) ──
        with tlx.async_task("default"):
            tile_id = start_pid
            tmem_accum_cnt = 0
            while tile_id < num_tiles:
                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                tmem_read_phase = tmem_accum_cnt & 1
                for g in tl.static_range(NUM_MMA_GROUPS):
                    tlx.barrier_wait(tmem_full[g], tmem_read_phase)
                    acc = tlx.local_load(acc_tmem[g])
                    tlx.barrier_arrive(tmem_empty[g], 1)
                    c = acc.to(tlx.dtype_of(c_desc))
                    tlx.local_store(c_smem[0], c)
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        c_desc, c_smem[0], [pid_m * BLOCK_M + g * GM, pid_n * BLOCK_N])
                    tlx.async_descriptor_store_wait(0)
                tile_id += NUM_SMS
                tmem_accum_cnt += 1

        # ── TC partition (2 MMA groups, K-loop nested inside persistent loop) ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tile_id = start_pid
            smem_accum = 0
            tmem_accum_cnt = 0
            while tile_id < num_tiles:
                tmem_write_phase = tmem_accum_cnt & 1
                # Peeled first K-iter (use_acc=False clears the accumulator).
                buf = smem_accum % NUM_SMEM_BUFFERS
                phase = (smem_accum // NUM_SMEM_BUFFERS) & 1
                tlx.barrier_wait(b_full[buf], phase)
                for g in tl.static_range(NUM_MMA_GROUPS):
                    a_buf = g * NUM_SMEM_BUFFERS + buf
                    tlx.barrier_wait(a_full[a_buf], phase)
                    # Wait for epilogue to release this group's TMEM before writing.
                    tlx.barrier_wait(tmem_empty[g], tmem_write_phase ^ 1)
                    b_t = tlx.local_trans(smem_b[buf])
                    tlx.async_dot(smem_a[a_buf], b_t, acc_tmem[g], use_acc=False,
                                  mBarriers=[a_empty[a_buf]], out_dtype=tl.float32)
                smem_accum += 1
                # Remaining K-iters (use_acc=True).
                for _k_iter in range(1, k_tiles):
                    buf = smem_accum % NUM_SMEM_BUFFERS
                    phase = (smem_accum // NUM_SMEM_BUFFERS) & 1
                    tlx.barrier_wait(b_full[buf], phase)
                    for g in tl.static_range(NUM_MMA_GROUPS):
                        a_buf = g * NUM_SMEM_BUFFERS + buf
                        tlx.barrier_wait(a_full[a_buf], phase)
                        b_t = tlx.local_trans(smem_b[buf])
                        tlx.async_dot(smem_a[a_buf], b_t, acc_tmem[g], use_acc=True,
                                      mBarriers=[a_empty[a_buf]], out_dtype=tl.float32)
                    smem_accum += 1
                # Completion handshake: per group, wait its last MMA's operand-read
                # to finish, then signal the epilogue for that group's accumulator.
                last_buf = (smem_accum - 1) % NUM_SMEM_BUFFERS
                last_phase = ((smem_accum - 1) // NUM_SMEM_BUFFERS) & 1
                for g in tl.static_range(NUM_MMA_GROUPS):
                    a_buf = g * NUM_SMEM_BUFFERS + last_buf
                    tlx.barrier_wait(a_empty[a_buf], last_phase)
                    tlx.barrier_arrive(tmem_full[g], 1)
                tile_id += NUM_SMS
                tmem_accum_cnt += 1

        # ── MEM partition (per-group A loads + shared B) ──
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
                    # Load A for group 0.
                    a_buf0 = buf  # 0 * NUM_SMEM_BUFFERS + buf
                    tlx.barrier_wait(a_empty[a_buf0], phase ^ 1)
                    tlx.barrier_expect_bytes(a_full[a_buf0], GM * BLOCK_K * 2)
                    tlx.async_descriptor_load(a_desc, smem_a[a_buf0], [offs_am, offs_k], a_full[a_buf0])
                    # Load shared B once. It is reused by ALL groups, so it is safe
                    # to overwrite only once the LAST group's A buffer is empty (the
                    # MMA that read B last has fired its a_empty mBarrier).
                    last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
                    tlx.barrier_wait(a_empty[last_a_buf], phase ^ 1)
                    tlx.barrier_expect_bytes(b_full[buf], BLOCK_N * BLOCK_K * 2)
                    tlx.async_descriptor_load(b_desc, smem_b[buf], [offs_bn, offs_k], b_full[buf])
                    # Load A for the remaining groups.
                    for g in tl.static_range(1, NUM_MMA_GROUPS):
                        a_bufg = g * NUM_SMEM_BUFFERS + buf
                        tlx.barrier_wait(a_empty[a_bufg], phase ^ 1)
                        tlx.barrier_expect_bytes(a_full[a_bufg], GM * BLOCK_K * 2)
                        tlx.async_descriptor_load(
                            a_desc, smem_a[a_bufg], [offs_am + g * GM, offs_k], a_full[a_bufg])
                    smem_accum += 1
                tile_id += NUM_SMS
