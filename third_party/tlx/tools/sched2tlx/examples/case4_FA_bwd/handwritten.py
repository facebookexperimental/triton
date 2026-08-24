"""Hand-written WS TLX FA backward — the ground-truth perf target for case4.

Faithful SINGLE-CTA, non-causal port of `_attn_bwd_ws` from
tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py (see PORT_SPEC.md), with
the common base of the optimizations kept:
  - 4 warp groups: load / mma / compute(softmax+dS) / reduction(dQ).
  - the proven skewed prolog/main/epilog 5-dot MMA pipeline.
  - single-buffered TMEM with STORAGE-ALIASING (qk/p share; dP/dS/dQ share) so
    HEAD_DIM=128 fits in 512 TMEM cols (128+128+dv128+dk128).

Pragmatic simplifications vs the tutorial (keep it a clean 1-CTA base):
  - M/D loaded directly in the compute task (no sM/sD SMEM + m/d barriers).
  - dK/dV/dQ written with plain tl.store / tl.atomic_add (no SMEM staging /
    EPILOGUE_SUBTILE / store_reduce). dQ atomic-adds across N-tiles.
  - case4 scale convention (matches handwritten_nows.py): Q pre-scaled by
    sm=1/sqrt(HD); exp2(qkT*log2e - m); M=logsumexp_e*log2e; D=rowsum(dO*O);
    no dQ/dK post-scale.
"""

import os

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def get_bufidx_phase(i, n: tl.constexpr):
    return i % n, (i // n) & 1


# To keep the comparison against the generated kernel apples-to-apples, we
# autotune ONLY the SMEM Q ring depth (the
# same degree of freedom the modulo scheduler chooses); the TMEM buffers are
# single-buffered with storage-aliasing, and BLOCK_M/N, HEAD_DIM, num_warps, and
# everything else stay fixed.
def _autotune_configs():
    # Measured best (do_bench on B200, square/representative shape). TLX_HW_BEST_CONFIG=1
    # uses it directly (fast path, no autotune search); otherwise the full depth grid
    # is swept. Mirrors the tutorial WS GEMM TLX_GEMM_USE_HEURISTIC pattern.
    # NOTE: depth 2 is the FA default — case4 wasn't in the logged autotune run, so this
    # is the conservative default, not a measured winner.
    if os.environ.get("TLX_HW_BEST_CONFIG") == "1":
        return [triton.Config({"NUM_BUFFERS_Q": 2}, num_warps=4, num_stages=1, num_ctas=1)]
    return [
        triton.Config({"NUM_BUFFERS_Q": s}, num_warps=4, num_stages=1, num_ctas=1)
        for s in (2, 3, 4, 5, 6)
    ]


# dQ is accumulated via tl.atomic_add across N-tiles; autotune re-runs the
# kernel per config, so dQ must be re-zeroed before each trial or configs
# accumulate into it and corrupt the result.
@triton.autotune(configs=_autotune_configs(), key=["N_CTX"], reset_to_zero=["dQ"])
@triton.jit
def fa_bwd_dkdv_ws(
    Q,
    K,
    V,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_m,
    stride_n,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    base = off_hz * N_CTX * HEAD_DIM
    m_base = off_hz * N_CTX
    start_n = pid_n * BLOCK_N
    num_steps = N_CTX // BLOCK_M

    k_desc = tl.make_tensor_descriptor(
        K + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    v_desc = tl.make_tensor_descriptor(
        V + base, [N_CTX, HEAD_DIM], [stride_n, 1], [BLOCK_N, HEAD_DIM]
    )
    q_desc = tl.make_tensor_descriptor(
        Q + base, [N_CTX, HEAD_DIM], [stride_m, 1], [BLOCK_M, HEAD_DIM]
    )
    do_desc = tl.make_tensor_descriptor(
        dO + base, [N_CTX, HEAD_DIM], [stride_m, 1], [BLOCK_M, HEAD_DIM]
    )

    # SMEM
    k_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, 1)
    v_smem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, 1)
    q_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.float16, NUM_BUFFERS_Q)
    do_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), tl.float16, 1)
    ds_smem = tlx.local_alloc((BLOCK_N, BLOCK_M), tl.float16, 1)

    # TMEM with storage-aliasing (single-buffered).
    qk_p = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tmem = tlx.local_alloc(
        (BLOCK_N, BLOCK_M), tl.float32, 1, tlx.storage_kind.tmem, reuse=qk_p
    )
    p_tmem = tlx.local_alloc(
        (BLOCK_N, BLOCK_M), tl.float16, 1, tlx.storage_kind.tmem, reuse=qk_p
    )
    dp_dq = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    dp_tmem = tlx.local_alloc(
        (BLOCK_N, BLOCK_M), tl.float32, 1, tlx.storage_kind.tmem, reuse=dp_dq
    )
    dsT_tmem = tlx.local_alloc(
        (BLOCK_N, BLOCK_M), tl.float16, 1, tlx.storage_kind.tmem, reuse=dp_dq
    )
    dq_tmem = tlx.local_alloc(
        (BLOCK_M, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem, reuse=dp_dq
    )
    dp_dq.set_buffer_overlap(
        tlx.reuse_group(
            dp_tmem, dsT_tmem, dq_tmem, group_type=tlx.reuse_group_type.shared
        )
    )
    dv_tmem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    dk_tmem = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)

    # Barriers
    kv_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    do_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    do_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    qk_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    qk_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    p_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dp_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    ds_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dsT_tmem_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dq_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dq_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    # dP/dS/dQ share one TMEM slot, so "dp slot free" == "dq slot free" — the
    # reduction task's dq read is what frees it (matches tutorial REUSE path).
    dp_empties = dq_empties
    dv_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dv_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dk_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dk_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    k_mma_done = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # ── compute: softmax pT + dS; then dK/dV epilogue store ──
        with tlx.async_task("default"):
            for blk in range(num_steps):
                ds_buf = blk % 1
                tlx.barrier_wait(qk_fulls[0], blk & 1)
                qkT = tlx.local_load(qk_tmem[0])
                offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
                m = tl.load(M + m_base + offs_m)
                pT = tl.math.exp2(qkT * LOG2E - m[None, :])
                tlx.local_store(p_tmem[0], pT.to(tl.float16))
                tlx.barrier_arrive(qk_empties[0], 1)
                tlx.barrier_arrive(p_fulls[0], 1)

                tlx.barrier_wait(dp_fulls[0], blk & 1)
                dpT = tlx.local_load(dp_tmem[0])
                Di = tl.load(D + m_base + offs_m)
                dsT = (pT * (dpT - Di[None, :])).to(tl.float16)
                tlx.local_store(dsT_tmem[0], dsT)
                tlx.local_store(ds_smem[0], dsT)
                tlx.fence_async_shared()
                tlx.barrier_arrive(ds_fulls[0], 1)
                tlx.barrier_arrive(dsT_tmem_fulls[0], 1)

            # epilogue: dV, dK to global (plain stores)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, HEAD_DIM)
            out_off = base + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            tlx.barrier_wait(dv_fulls[0], 0)
            dv = tlx.local_load(dv_tmem[0])
            tl.store(dV + out_off, dv.to(tl.float16))
            tlx.barrier_wait(dk_fulls[0], 0)
            tlx.barrier_wait(k_mma_done[0], 0)
            dk = tlx.local_load(dk_tmem[0])
            tl.store(dK + out_off, dk.to(tl.float16))

        # ── mma: skewed prolog/main/epilog 5-dot pipeline ──
        with tlx.async_task(num_warps=1, num_regs=88):
            tlx.barrier_wait(kv_full[0], 0)
            blk = 0
            # Prolog: qkT, dpT, dV
            q_buf, q_ph = get_bufidx_phase(blk, NUM_BUFFERS_Q)
            tlx.barrier_wait(q_fulls[q_buf], q_ph)
            qT = tlx.local_trans(q_smem[q_buf])
            tlx.async_dot(
                k_smem[0], qT, qk_tmem[0], use_acc=False, mBarriers=[qk_fulls[0]]
            )
            tlx.barrier_wait(do_fulls[0], 0)
            doT = tlx.local_trans(do_smem[0])
            tlx.async_dot(
                v_smem[0], doT, dp_tmem[0], use_acc=False, mBarriers=[dp_fulls[0]]
            )
            tlx.barrier_wait(p_fulls[0], 0)
            tlx.async_dot(
                p_tmem[0],
                do_smem[0],
                dv_tmem[0],
                use_acc=False,
                mBarriers=[do_empties[0]],
            )
            blk += 1
            for j in range(1, num_steps):
                q_buf, q_ph = get_bufidx_phase(blk, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_fulls[q_buf], q_ph)
                tlx.barrier_wait(qk_empties[0], (blk & 1) ^ 1)
                qT = tlx.local_trans(q_smem[q_buf])
                tlx.async_dot(
                    k_smem[0], qT, qk_tmem[0], use_acc=False, mBarriers=[qk_fulls[0]]
                )
                pblk = blk - 1
                q_buf_p, _ = get_bufidx_phase(pblk, NUM_BUFFERS_Q)
                # dK += dsT @ q  (read dsT from TMEM before dq overwrites the slot)
                tlx.barrier_wait(dsT_tmem_fulls[0], pblk & 1)
                tlx.async_dot(
                    dsT_tmem[0],
                    q_smem[q_buf_p],
                    dk_tmem[0],
                    use_acc=(j - 1) > 0,
                    mBarriers=[q_empties[q_buf_p]],
                )
                # dQ = trans(dsT) @ k
                tlx.barrier_wait(ds_fulls[0], pblk & 1)
                tlx.barrier_wait(dq_empties[0], (pblk & 1) ^ 1)
                dsT_view = tlx.local_trans(ds_smem[0])
                tlx.async_dot(
                    dsT_view,
                    k_smem[0],
                    dq_tmem[0],
                    use_acc=False,
                    mBarriers=[dq_fulls[0]],
                )
                # dpT, dV
                tlx.barrier_wait(do_fulls[0], blk & 1)
                tlx.barrier_wait(dp_empties[0], (blk & 1) ^ 1)
                doT = tlx.local_trans(do_smem[0])
                tlx.async_dot(
                    v_smem[0], doT, dp_tmem[0], use_acc=False, mBarriers=[dp_fulls[0]]
                )
                tlx.barrier_wait(p_fulls[0], blk & 1)
                tlx.async_dot(
                    p_tmem[0],
                    do_smem[0],
                    dv_tmem[0],
                    use_acc=True,
                    mBarriers=[do_empties[0]],
                )
                blk += 1
            tlx.tcgen05_commit(dv_fulls[0])
            # Epilog: last dK, dQ
            pblk = blk - 1
            q_buf_p, _ = get_bufidx_phase(pblk, NUM_BUFFERS_Q)
            tlx.barrier_wait(dsT_tmem_fulls[0], pblk & 1)
            tlx.async_dot(
                dsT_tmem[0],
                q_smem[q_buf_p],
                dk_tmem[0],
                use_acc=num_steps > 1,
                mBarriers=[q_empties[q_buf_p], dk_fulls[0]],
            )
            tlx.barrier_wait(ds_fulls[0], pblk & 1)
            tlx.barrier_wait(dq_empties[0], (pblk & 1) ^ 1)
            dsT_view = tlx.local_trans(ds_smem[0])
            tlx.async_dot(
                dsT_view, k_smem[0], dq_tmem[0], use_acc=False, mBarriers=[dq_fulls[0]]
            )
            tlx.tcgen05_commit(k_mma_done[0])

        # ── reduction: dQ atomic-add to global ──
        with tlx.async_task(num_warps=4, num_regs=88):
            offs_d = tl.arange(0, HEAD_DIM)
            for blk in range(num_steps):
                tlx.barrier_wait(dq_fulls[0], blk & 1)
                dq = tlx.local_load(dq_tmem[0])
                tlx.barrier_arrive(dq_empties[0], 1)
                offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
                dq_off = base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
                tl.atomic_add(dQ + dq_off, dq.to(tl.float16))

        # ── load: TMA loads K,V once; Q,dO per iter ──
        with tlx.async_task(num_warps=1, num_regs=24):
            tlx.barrier_expect_bytes(kv_full[0], 2 * BLOCK_N * HEAD_DIM * 2)
            tlx.async_descriptor_load(k_desc, k_smem[0], [start_n, 0], kv_full[0])
            tlx.async_descriptor_load(v_desc, v_smem[0], [start_n, 0], kv_full[0])
            for blk in range(num_steps):
                q_buf, q_ph = get_bufidx_phase(blk, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_buf], q_ph ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_buf], BLOCK_M * HEAD_DIM * 2)
                tlx.async_descriptor_load(
                    q_desc, q_smem[q_buf], [blk * BLOCK_M, 0], q_fulls[q_buf]
                )
                tlx.barrier_wait(do_empties[0], (blk & 1) ^ 1)
                tlx.barrier_expect_bytes(do_fulls[0], BLOCK_M * HEAD_DIM * 2)
                tlx.async_descriptor_load(
                    do_desc, do_smem[0], [blk * BLOCK_M, 0], do_fulls[0]
                )
