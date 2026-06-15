"""Hand-written FA forward TLX target for case3.

Adapted from third_party/tlx/tutorials/blackwell_fa_ws.py with NUM_MMA_GROUPS=1
(no M splitting), and SMEM (instead of TMEM-reuse) for the small alpha/l/m
buffers. The barrier protocol is verbatim from the tutorial.

Layout:
  q_smem  : [BLOCK_M, HEAD_DIM] bf16  — Q resident through K-loop
  kv_smem : [BLOCK_N, HEAD_DIM] bf16  — ring buffer (NUM_BUFFERS_KV deep)
  qk_tmem : [BLOCK_M, BLOCK_N] f32    — Q@K result (single buffer)
  p_tmem  : [BLOCK_M, BLOCK_N] bf16   — softmax output, MMA operand
  acc_tmem: [BLOCK_M, HEAD_DIM] f32   — running attention output
  alpha/l/m_smem : [BLOCK_M] f32      — per-iter softmax stats hand-off

4 async tasks (matches tutorial):
  default — read alpha+acc each iter, apply correction (acc*=alpha), final epilogue
  softmax — read qk, online softmax math, write alpha + p
  TC      — Q@K, P@V (two MMAs per iter)
  MEM     — TMA loads (Q once, K+V every iter)
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def fa_fwd_kernel(
    Q,
    K,
    V,
    Out,
    M_lse,
    sm_scale,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
):
    # Tensor descriptors.
    desc_q = tl.make_tensor_descriptor(Q, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(K, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(V, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(Out, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    # SMEM buffers.
    q_tile = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.bfloat16, 1)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.bfloat16, NUM_BUFFERS_KV)
    alpha_smem = tlx.local_alloc((BLOCK_M, ), tl.float32, 1)
    l_smem = tlx.local_alloc((BLOCK_M, ), tl.float32, 1)
    m_smem = tlx.local_alloc((BLOCK_M, ), tl.float32, 1)

    # TMEM buffers.
    qk_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, 1, tlx.storage_kind.tmem)
    p_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.bfloat16, 1, tlx.storage_kind.tmem)
    acc_tmem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)

    # Mbarriers (mirrors tutorial's barrier set, single-buffered).
    q_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    qk_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    p_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    acc_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    acc_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    alpha_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    alpha_empty = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    l_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    qo_offset = pid_bh * N_CTX + pid_m * BLOCK_M
    kv_offset_base = pid_bh * N_CTX
    n_kv = N_CTX // BLOCK_N

    with tlx.async_tasks():
        # ── default: correction + epilogue ──
        with tlx.async_task("default"):
            for k_iter in range(n_kv):
                phase = k_iter & 1
                tlx.barrier_wait(alpha_full[0], phase)
                alpha = tlx.local_load(alpha_smem[0])
                tlx.barrier_arrive(alpha_empty[0], 1)
                acc = tlx.local_load(acc_tmem[0])
                acc = acc * alpha[:, None]
                tlx.local_store(acc_tmem[0], acc)
                tlx.barrier_arrive(acc_full[0], 1)
            # Epilogue: read final stats and acc, normalize, store.
            tlx.barrier_wait(l_full[0], 0)
            l = tlx.local_load(l_smem[0])
            m = tlx.local_load(m_smem[0])
            phase = (n_kv - 1) & 1
            tlx.barrier_wait(acc_empty[0], phase)
            acc = tlx.local_load(acc_tmem[0])
            acc = acc / l[:, None]
            desc_o.store([qo_offset, 0], acc.to(tl.bfloat16))
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            tl.store(M_lse + pid_bh * N_CTX + offs_m, m + tl.math.log2(l))

        # ── softmax: qk → online stats + p ──
        with tlx.async_task(num_warps=4, num_regs=152):
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            qk_scale = sm_scale * 1.44269504
            for k_iter in range(n_kv):
                phase = k_iter & 1
                tlx.barrier_wait(qk_full[0], phase)
                qk = tlx.local_load(qk_tmem[0])
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                alpha = tl.math.exp2(m_i - m_ij)
                tlx.barrier_wait(alpha_empty[0], phase ^ 1)
                tlx.local_store(alpha_smem[0], alpha)
                tlx.barrier_arrive(alpha_full[0], 1)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                p_bf16 = p.to(tl.bfloat16)
                tlx.local_store(p_tmem[0], p_bf16)
                tlx.barrier_arrive(p_full[0], 1)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
            # Hand off l, m to default for epilogue.
            tlx.local_store(l_smem[0], l_i)
            tlx.local_store(m_smem[0], m_i)
            tlx.barrier_arrive(l_full[0], 1)

        # ── TC: Q@K then P@V ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_wait(q_full[0], 0)
            for k_iter in range(n_kv):
                k_buf = (k_iter * 2) % NUM_BUFFERS_KV
                k_phase = ((k_iter * 2) // NUM_BUFFERS_KV) & 1
                v_buf = (k_iter * 2 + 1) % NUM_BUFFERS_KV
                v_phase = ((k_iter * 2 + 1) // NUM_BUFFERS_KV) & 1
                phase = k_iter & 1
                # Q @ K^T → qk_tmem
                tlx.barrier_wait(kv_fulls[k_buf], k_phase)
                k_t = tlx.local_trans(kv_tiles[k_buf])
                tlx.async_dot(
                    q_tile[0],
                    k_t,
                    qk_tmem[0],
                    use_acc=False,
                    mBarriers=[qk_full[0], kv_empties[k_buf]],
                )
                # P @ V → acc_tmem (with correction already applied by default)
                tlx.barrier_wait(kv_fulls[v_buf], v_phase)
                tlx.barrier_wait(p_full[0], phase)
                tlx.barrier_wait(acc_full[0], phase)
                tlx.async_dot(
                    p_tmem[0],
                    kv_tiles[v_buf],
                    acc_tmem[0],
                    use_acc=True,
                    mBarriers=[acc_empty[0], kv_empties[v_buf]],
                )

        # ── MEM: TMA loads ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_expect_bytes(q_full[0], BLOCK_M * HEAD_DIM * 2)
            tlx.async_descriptor_load(desc_q, q_tile[0], [qo_offset, 0], q_full[0])
            for k_iter in range(n_kv):
                k_buf = (k_iter * 2) % NUM_BUFFERS_KV
                k_phase = ((k_iter * 2) // NUM_BUFFERS_KV) & 1
                v_buf = (k_iter * 2 + 1) % NUM_BUFFERS_KV
                v_phase = ((k_iter * 2 + 1) // NUM_BUFFERS_KV) & 1
                tlx.barrier_wait(kv_empties[k_buf], k_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[k_buf], BLOCK_N * HEAD_DIM * 2)
                tlx.async_descriptor_load(
                    desc_k,
                    kv_tiles[k_buf],
                    [kv_offset_base + k_iter * BLOCK_N, 0],
                    kv_fulls[k_buf],
                )
                tlx.barrier_wait(kv_empties[v_buf], v_phase ^ 1)
                tlx.barrier_expect_bytes(kv_fulls[v_buf], BLOCK_N * HEAD_DIM * 2)
                tlx.async_descriptor_load(
                    desc_v,
                    kv_tiles[v_buf],
                    [kv_offset_base + k_iter * BLOCK_N, 0],
                    kv_fulls[v_buf],
                )
