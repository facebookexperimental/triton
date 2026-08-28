# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors
"""Kimi Delta Attention (KDA): warp-specialized chunkwise forward + backward.

Gated delta rule with per-channel decay, chunked; one CTA per (sequence, head)
carries state S in [K, V]. Per chunk of length C=BT, gamma = cumsum_t(g),
Gamma = sum_t(g):
    k_hat = k*exp(-gamma) ; k_bar = k*exp(gamma) ; q_bar = scale*q*exp(gamma)
    A_{t,r} = beta_r <k_hat_r, k_bar_t>   (r < t)
    U = (I + tril(A,-1))^{-1} (V - k_bar @ S)
    O = q_bar @ S + tril(q_bar @ k_hat^T, 0) @ (diag(beta) U)
    S = diag(exp(Gamma)) S + k_tilde^T @ (diag(beta) U), k_tilde=k*exp(Gamma-gamma)
Backward is the VJP of the above (2-pass: store S_in, then reverse scan).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase


@triton.jit
def _add_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _mul_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a, b, c):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


_KDA_GATE_LAYOUT = tlx.layout(
    shape=((4, 8, 4, 2), (2, 2, 8)),
    stride=((256, 1, 32, 16), (128, 8, 1024)),
)

_KDA_RHS_LAYOUT = tlx.layout(
    shape=((32, 4, 2), (32, )),
    stride=((1, 32, 4096), (128, )),
)


@triton.jit
def _kda_ws_fwd_kernel(  # noqa: C901
    Q,
    K,
    V,
    G,
    Beta,
    O,
    SaveS,
    SaveT,
    cu_seqlens,
    n_seq,
    max_chunks,
    stride_bt_tok,
    stride_bt_h,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    LOG2_BT: tl.constexpr,
    NBUF: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
):
    # Persistent kernel: grid=(NUM_SMS,). Each of the 3 warp-spec tasks
    # grid-strides over the (seq, head) work-items and keeps its own running
    # chunk counter `it` — since all three tasks visit the same work-items in
    # the same order, their `it` values (and hence the get_bufidx_phase buffer
    # index/phase) stay in lock-step, preserving the producer->consumer->epilogue
    # handshake across sequence boundaries. This removes the launch wave-tail on
    # uniform batches and load-balances jagged batches over the fixed SM pool.
    #
    # How the chunk is split across the tasks follows its dependency graph rather
    # than a producer/consumer convention:
    #   * producer (1 warp)  -- TMA loads of q/k/v/g.
    #   * compute  (8 warps) -- gate scaling, the carried state, the residual, and
    #     the two accumulating dots: everything on the S recurrence.
    #   * epilogue (4 warps) -- the WY triangular inverse and the output drain.
    #     Neither depends on S, and the inverse is five dependent TMEM round trips
    #     that nothing in the chunk's own algebra can overlap, so running it on a
    #     second warp group puts it in the shadow of the state readout instead of
    #     in series with it.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_work = n_seq * H

    HD: tl.constexpr = H * D
    dtype = Q.dtype.element_ty
    q_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    k_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    v_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    g_buf = tlx.local_alloc((BT, D), dtype, NBUF)  # g staged bf16; cumsum in fp32
    o_buf = tlx.local_alloc((BT, D), dtype, NBUF)

    mma_b_persist = tlx.local_alloc((BT, D), dtype, 1)
    mma_b_work = tlx.local_alloc((BT, D), dtype, 1)
    mma_aqk_b = tlx.local_alloc((BT, BT), dtype, 1)
    # The cumsum operand is a loop invariant. It used to share the WY chain's
    # scratch, which forced a [C,C] SMEM store every chunk; its own 8KB tile lets
    # it be written once for the whole kernel.
    mma_lmat = tlx.local_alloc((BT, BT), dtype, 1)
    mma_b_state = tlx.local_alloc((D, D), dtype, 1)
    mma_lhs_state_reads = tlx.local_alloc((D, D), dtype, 1)
    # Dedicated TMA source for the saved WY inverse: every other [C,C] tile is
    # recycled inside the chunk, and the store must still be readable then.
    mma_tinv_out = tlx.local_alloc((BT, BT), dtype, 1)
    mma_lhs_small = tlx.local_alloc((BT, BT), dtype, 1, tlx.storage_kind.tmem)
    mma_small0 = tlx.local_alloc((BT, BT), tl.float32, 1, tlx.storage_kind.tmem)
    mma_state_reads = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)
    mma_u = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    mma_gamma = tlx.local_alloc((D, BT), tl.float32, 1, tlx.storage_kind.tmem, reuse=mma_state_reads)
    mma_state = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)

    full = tlx.alloc_barriers(num_barriers=NBUF)
    full_qv = tlx.alloc_barriers(num_barriers=NBUF)
    empty = tlx.alloc_barriers(num_barriers=NBUF)
    o_empty = tlx.alloc_barriers(num_barriers=NBUF)
    mma_small0_full = tlx.alloc_barriers(num_barriers=1)
    # One barrier per doubling step. Each is then signalled exactly once per
    # chunk, so its wait parity is simply the chunk phase, and the step count
    # can be changed without re-deriving the parity arithmetic that ties the
    # shared barriers to both the iteration index and the chunk phase.
    mma_inv_a = tlx.alloc_barriers(num_barriers=LOG2_BT)
    mma_inv_b = tlx.alloc_barriers(num_barriers=LOG2_BT)
    mma_aqk_full = tlx.alloc_barriers(num_barriers=1)
    tinv_ready = tlx.alloc_barriers(num_barriers=1)
    state_ready = tlx.alloc_barriers(num_barriers=1)
    mma_gamma_p = tlx.alloc_barriers(num_barriers=1)
    mma_wide_full = tlx.alloc_barriers(num_barriers=1)
    mma_out_full = tlx.alloc_barriers(num_barriers=1)
    mma_gamma_full = tlx.alloc_barriers(num_barriers=1)
    mma_state_full = tlx.alloc_barriers(num_barriers=1)
    # No priming: barrier_wait(bar, p) proceeds while phase != p, so the
    # producer's first empty-wait (phase^1 == 1) passes immediately at phase 0.

    QK_BYTES: tl.constexpr = BT * D * 2  # bf16 q/k/v/g tile

    with tlx.async_tasks():
        # ---- producer: TMA loads + the carried state ----
        # Four warps rather than one: the partition warp count is rounded up to a
        # multiple of four regardless, so a 1-warp producer cost three warps of
        # pure padding. Widened, those warps reach all 128 TMEM lanes and can own
        # the state staging. That takes the staging off the compute task, whose A
        # dot -- the head of the WY chain that task later waits on -- then issues
        # a full staging earlier. This task is otherwise idle: it spends the whole
        # chunk parked on `empty`.
        with tlx.async_task(num_warps=4, registers=56):
            DQ: tl.constexpr = D // 4
            b_state_p = mma_b_state[0]
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                pid_h = wid % H
                seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
                seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
                T = seq_end - seq_beg
                num_chunks = tl.cdiv(T, BT)
                st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
                prev_decay = tl.full([D], 1.0, tl.float32)
                # Per-sequence descriptors: base at this sequence's start and use
                # shape=[T, HD] so the TMA globalDim is the seqlen (<= ~14000), not
                # the packed total_len (which exceeds the TMA globalDim limit at
                # prod). Rows are indexed relative (row = c*BT); the [T, HD] bound
                # also clamps the last partial chunk.
                sob = seq_beg.to(tl.int64) * HD
                q_desc = tl.make_tensor_descriptor(Q + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
                k_desc = tl.make_tensor_descriptor(K + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
                v_desc = tl.make_tensor_descriptor(V + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
                g_desc = tl.make_tensor_descriptor(G + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
                if SAVE_INTERMEDIATE:
                    # TMA-store the entering state straight out of the SMEM tile
                    # that already holds it; the register store path materialized a
                    # [D,D] int64 index tile per chunk.
                    ss_desc = tl.make_tensor_descriptor(
                        SaveS + st_base * D * D,
                        shape=[num_chunks * D, D],
                        strides=[D, 1],
                        block_shape=[D, D],
                    )
                col = pid_h * D
                for c in tl.range(0, num_chunks):
                    bufidx, phase = get_bufidx_phase(it, NBUF)
                    tlx.barrier_wait(empty[bufidx], phase ^ 1)
                    row = c * BT  # relative to this sequence's descriptor base
                    # Two arrivals, not one: the compute task only needs g and k
                    # to run the cumsum and the whole gate branch, so gating that
                    # on half the chunk's bytes lets it start while q/v are still
                    # landing. q/v are waited on just before they are read.
                    tlx.barrier_expect_bytes(full[bufidx], 2 * QK_BYTES)
                    tlx.async_descriptor_load(g_desc, g_buf[bufidx], [row, col], full[bufidx])
                    tlx.async_descriptor_load(k_desc, k_buf[bufidx], [row, col], full[bufidx])
                    tlx.barrier_expect_bytes(full_qv[bufidx], 2 * QK_BYTES)
                    tlx.async_descriptor_load(q_desc, q_buf[bufidx], [row, col], full_qv[bufidx])
                    tlx.async_descriptor_load(v_desc, v_buf[bufidx], [row, col], full_qv[bufidx])
                    # --- carried state: bf16 copy for the readout, decay in place ---
                    # S_{c+1} = diag(e^Gamma_c) (S_c + (beta k_hat_c)^T U_c), so the
                    # accumulator carries the *undecayed* sum and this chunk applies
                    # the previous chunk's decay, a scale this task already holds
                    # from its last iteration. Quartered because a [D, D] fp32 tile
                    # is 128 registers per thread on four warps.
                    tlx.barrier_wait(mma_state_full[0], phase ^ 1)
                    if c == 0:
                        tlx.local_store(mma_state[0], tl.zeros([D, D], tl.float32))
                        tlx.local_store(b_state_p, tl.zeros([D, D], dtype))
                    else:
                        for h in tl.static_range(4):
                            s_h = _mul_f32x2(
                                tlx.local_load(tlx.subslice(mma_state[0], h * DQ, DQ)),
                                prev_decay[:, None],
                            )
                            tlx.local_store(
                                tlx.local_slice(b_state_p, [0, h * DQ], [D, DQ]),
                                s_h.to(dtype),
                            )
                            tlx.local_store(tlx.subslice(mma_state[0], h * DQ, DQ), s_h)
                    # This chunk's per-channel gate total is the next chunk's decay.
                    # Read before releasing the compute task: the cumsum tile aliases
                    # the readout accumulator, which its state-read dot overwrites.
                    tlx.barrier_wait(mma_gamma_p[0], phase)
                    prev_decay = tl.exp2(
                        _mul_f32x2(
                            tl.reshape(
                                tlx.local_load(tlx.subslice(mma_gamma[0], BT - 1, 1)),
                                [D],
                            ),
                            1.4426950408889634,
                        ))
                    tlx.fence_async_shared()
                    tlx.barrier_arrive(state_ready[0], 1)
                    if SAVE_INTERMEDIATE:
                        # save the entering state S_in [D,D] for the backward.
                        tlx.async_descriptor_store(ss_desc, b_state_p, [c * D, 0])
                        tlx.async_descriptor_store_wait(1)
                    it += 1

        # ---- consumer: chunkwise KDA compute ----
        with tlx.async_task("default", registers=168):
            lhs_state_reads = mma_lhs_state_reads[0]
            lhs_wide = tlx.local_slice(lhs_state_reads, [0, 0], [BT, D])
            lhs_tail = tlx.local_slice(lhs_state_reads, [BT, 0], [BT, D])
            b_persist = mma_b_persist[0]
            b_work = mma_b_work[0]
            aqk_b = mma_aqk_b[0]
            b_work_x = tlx.local_slice(b_work, [0, 0], [BT, BT])
            b_work_t = tlx.local_slice(b_work, [0, BT], [BT, BT])
            b_state = mma_b_state[0]
            state_reads = mma_state_reads[0]
            state_readout_tmem = tlx.subslice(state_reads, 0, BT)
            output_tmem = tlx.subslice(state_reads, BT, BT)
            acc_small1 = tlx.subslice(mma_u[0], 0, BT)
            # Aqk parks in the other half of the pseudo-value accumulator: both
            # halves are free until the solve writes U.
            acc_aqk = tlx.subslice(mma_u[0], BT, BT)
            DH: tl.constexpr = D // 2  # noqa: F841
            offs_t = tl.arange(0, BT)
            lower_strict = offs_t[:, None] > offs_t[None, :]
            lower_incl = offs_t[:, None] >= offs_t[None, :]
            eye = tl.where(  # noqa: F841
                offs_t[:, None] == offs_t[None, :], 1.0, 0.0)
            # L[t,r] = (r <= t). Rows past the sequence end need no column mask:
            # the descriptors are bounded by shape=[T, HD], so TMA zero-fills g
            # there and those terms contribute exact zeros.
            tlx.local_store(mma_lmat[0], lower_incl.to(dtype))
            tlx.fence_async_shared()
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                pid_h = wid % H
                seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
                seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
                T = seq_end - seq_beg
                num_chunks = tl.cdiv(T, BT)
                # gamma = cumsum(g) as a triangular matmul. tl.cumsum down 64
                # rows is a cross-thread scan staged through SMEM; ablating it
                # measured 3.39ms of a 16.8ms kernel. L carries the row mask in
                # its columns, so g feeds the MMA straight from its TMA buffer
                # and never has to be masked or staged through registers.
                #
                # Chunk 0's is issued here and every later chunk's from the
                # previous chunk's tail, so the gate branch never waits on it.
                # The pipeline is free: g's only consumer is this dot, so NBUF=1
                # suffices -- the producer has long refilled the buffer -- and
                # gamma's TMEM tile is dead from the residual onward, so no
                # extra columns are needed either. Only the dot *issue* moves;
                # moving any register work across the chunk boundary spills
                # (measured: the full gate branch pipelined costs +50%).
                bufidx0, phase0 = get_bufidx_phase(it, NBUF)
                tlx.barrier_wait(full[bufidx0], phase0)
                tlx.async_dot(
                    tlx.local_trans(g_buf[bufidx0]),
                    tlx.local_trans(mma_lmat[0]),
                    mma_gamma[0],
                    use_acc=False,
                    force_async=True,
                    mBarriers=[mma_gamma_full[0], mma_gamma_p[0]],
                )
                for c in tl.range(0, num_chunks):
                    bufidx, phase = get_bufidx_phase(it, NBUF)
                    rows = c * BT + offs_t
                    row_mask = rows < T
                    full_chunk = (c + 1) * BT <= T
                    # Issued ahead of the gate wait: this gather is the only
                    # global load left on the compute path, so its DRAM latency
                    # is spent while the cumsum is still running.
                    beta_a = tl.load(
                        Beta + (seq_beg + rows) * stride_bt_tok + pid_h * stride_bt_h,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    k_a = tlx.local_load(k_buf[bufidx], layout=_KDA_GATE_LAYOUT)
                    if not full_chunk:
                        k_a = tl.where(row_mask[:, None], k_a, tl.zeros_like(k_a))
                    tlx.barrier_wait(mma_gamma_full[0], phase)
                    tlx.barrier_wait(full_qv[bufidx], phase)
                    q = tlx.local_load(q_buf[bufidx], layout=_KDA_GATE_LAYOUT)
                    v = tlx.local_load(v_buf[bufidx], layout=_KDA_RHS_LAYOUT)
                    tlx.barrier_arrive(empty[bufidx], 1)
                    if not full_chunk:
                        q = tl.where(row_mask[:, None], q, tl.zeros_like(q))
                        v = tl.where(row_mask[:, None], v, 0.0)
                    gamma_a = _mul_f32x2(tl.trans(tlx.local_load(mma_gamma[0])), 1.4426950408889634)
                    k_hat_a = (k_a * tl.exp2(-gamma_a).to(dtype)).to(dtype)
                    exp_gamma = tl.exp2(gamma_a).to(dtype)
                    k_bar_a = (k_a * exp_gamma).to(dtype)
                    beta_k_hat_a = (beta_a[:, None] * k_hat_a).to(dtype)

                    # --- intra matrix A (strictly lower-tri) ---
                    # A_{t,r} = <beta_r*k_hat_r, k_bar_t>  for r<t   [C,C]
                    tlx.local_store(lhs_wide, k_bar_a)
                    tlx.local_store(b_persist, beta_k_hat_a)
                    tlx.fence_async_shared()
                    tlx.async_dot(
                        lhs_wide,
                        tlx.local_trans(b_persist),
                        mma_small0[0],
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_small0_full[0]],
                    )
                    q_bar = (q * (exp_gamma * scale).to(dtype)).to(dtype)
                    tlx.local_store(lhs_tail, q_bar)

                    # Both state-dependent dots are queued ahead of the WY chain.
                    # Neither depends on T, so the readout->rhs and Aqk register
                    # work they unblock is what fills the doubling chain's stalls;
                    # run in T's shadow the chain measured 1.5ms of the 8.7ms.
                    # The readout shares its accumulator with the output tile,
                    # so the previous chunk's epilogue has to have drained it.
                    tlx.barrier_wait(o_empty[bufidx], phase ^ 1)
                    tlx.barrier_wait(state_ready[0], phase)
                    tlx.async_dot(
                        tlx.local_trans(b_state),
                        tlx.local_trans(lhs_state_reads),
                        state_reads,
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_wide_full[0]],
                    )
                    # The WY inverse runs on the epilogue task (below); what is
                    # left here is the state-side work that fills its shadow, plus
                    # one wait on the result.
                    # rhs = v - k_bar @ S. k_bar's tile is dead once the readout
                    # retires, so rhs lands on top of it and becomes the B operand
                    # of the solve.
                    tlx.barrier_wait(mma_wide_full[0], 0)
                    state_readout = tl.trans(tlx.local_load(state_readout_tmem))
                    rhs = _add_f32x2(v.to(tl.float32), -state_readout)
                    tlx.local_store(lhs_wide, rhs.to(dtype))
                    tlx.async_dot(
                        lhs_tail,
                        tlx.local_trans(b_persist),
                        acc_aqk,
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_aqk_full[0]],
                    )
                    tlx.barrier_wait(mma_aqk_full[0], phase)
                    Aqk = tlx.local_load(acc_aqk)
                    Aqk = tl.where(lower_incl, Aqk, 0.0).to(dtype)
                    tlx.local_store(aqk_b, Aqk)
                    tlx.fence_async_shared()
                    tlx.barrier_wait(tinv_ready[0], phase)
                    # U = T @ rhs   [C,D]  (solve the intra-chunk system)
                    tlx.async_dot(
                        mma_lhs_small[0],
                        lhs_wide,
                        mma_u[0],
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_wide_full[0]],
                    )
                    tlx.barrier_wait(mma_wide_full[0], 1)
                    U = tlx.local_load(mma_u[0])
                    # Accumulate the transposed output so it matches the merged
                    # state's N-sliced TMEM result: U^T @ Aqk^T.
                    tlx.local_store(b_work, U.to(dtype))
                    tlx.fence_async_shared()
                    tlx.async_dot(
                        tlx.local_trans(b_work),
                        tlx.local_trans(aqk_b),
                        output_tmem,
                        use_acc=True,
                        force_async=True,
                        mBarriers=[mma_out_full[0]],
                    )

                    # --- inter-chunk state recurrence ---
                    # R <- S + (beta k_hat)^T @ U   [D,D]; the diag(e^{Gamma}) that
                    # turns R into the next chunk's S is applied at the top of that
                    # chunk, where its operand is already resident.
                    tlx.async_dot(
                        tlx.local_trans(b_persist),
                        b_work,
                        mma_state[0],
                        use_acc=True,
                        force_async=True,
                        mBarriers=[mma_state_full[0]],
                    )
                    if c + 1 < num_chunks:
                        nbuf, nphase = get_bufidx_phase(it + 1, NBUF)
                        tlx.barrier_wait(full[nbuf], nphase)
                        tlx.async_dot(
                            tlx.local_trans(g_buf[nbuf]),
                            tlx.local_trans(mma_lmat[0]),
                            mma_gamma[0],
                            use_acc=False,
                            force_async=True,
                            mBarriers=[mma_gamma_full[0], mma_gamma_p[0]],
                        )
                    it += 1

        # ---- epilogue: drain the output accumulator and TMA-store it ----
        # Four warps (one per TMEM lane quarter) so this task can read the
        # accumulator itself. Draining it here instead of on the compute task
        # takes the whole O-dot wait -> TMEM read -> transpose -> SMEM store tail
        # off the compute critical path; it now runs against the next chunk's
        # gate math, and the only thing the compute task waits for is the
        # accumulator being free again.
        with tlx.async_task(num_warps=4, registers=104):
            out_tmem = tlx.subslice(mma_state_reads[0], BT, BT)
            lhs_small = mma_lhs_small[0]
            tinv_out = mma_tinv_out[0]
            b_work_x = tlx.local_slice(mma_b_work[0], [0, 0], [BT, BT])
            b_work_t = tlx.local_slice(mma_b_work[0], [0, BT], [BT, BT])
            acc_small1 = tlx.subslice(mma_u[0], 0, BT)
            offs_t = tl.arange(0, BT)
            lower_strict = offs_t[:, None] > offs_t[None, :]
            eye_b = tl.where(offs_t[:, None] == offs_t[None, :], 1.0, 0.0).to(dtype)
            INV_STEPS: tl.constexpr = LOG2_BT - 3
            BTH: tl.constexpr = BT // 2
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                pid_h = wid % H
                seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
                seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
                T = seq_end - seq_beg
                num_chunks = tl.cdiv(T, BT)
                sob = seq_beg.to(tl.int64) * HD
                st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
                o_desc = tl.make_tensor_descriptor(O + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
                if SAVE_INTERMEDIATE:
                    st_desc = tl.make_tensor_descriptor(
                        SaveT + st_base * BT * BT,
                        shape=[num_chunks * BT, BT],
                        strides=[BT, 1],
                        block_shape=[BT, BT],
                    )
                col = (pid_h * D).to(tl.int32)
                for c in tl.range(0, num_chunks):
                    bufidx, phase = get_bufidx_phase(it, NBUF)
                    # --- WY triangular inverse  T = (I + A)^{-1}  [C,C] ---
                    # A strictly-lower => nilpotent (A^C=0), so the Neumann series
                    # T = I - A + A^2 - ... terminates; here via log2(C) doublings.
                    # Five dependent TMEM round trips that nothing in the chunk's
                    # own algebra can fill: ablating the chain measured 1.5ms of an
                    # 8.7ms kernel. Run on this task it overlaps the compute task's
                    # state readout and residual instead of stalling behind them.
                    tlx.barrier_wait(mma_small0_full[0], phase)
                    A = tlx.local_load(mma_small0[0])
                    A = tl.where(lower_strict, A, 0.0).to(dtype)  # keep t>r
                    X = -A
                    # Step 0 of the chain is (I + X) @ I, whose fp32 accumulator
                    # rounds back to exactly `eye_b + X` in bf16 (disjoint
                    # supports), so it is folded into a register add and only its
                    # independent sibling X @ X is issued.
                    Tinv = eye_b + X
                    Xn = X
                    tlx.local_store(b_work_x, Xn)
                    tlx.fence_async_shared()
                    tlx.async_dot(
                        b_work_x,
                        b_work_x,
                        acc_small1,
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_inv_b[0]],
                    )
                    tlx.barrier_wait(mma_inv_b[0], phase)
                    Xn = tlx.local_load(acc_small1).to(dtype)
                    for i in tl.static_range(1, INV_STEPS):
                        # Xn is strictly lower, so `eye_b + Xn` is exact in bf16
                        # and folds the identity term of `Tinv + Xn @ Tinv` into
                        # the A operand. That drops the [C,C] fp32 accumulator
                        # seed store from every step of this serial chain.
                        tlx.local_store(lhs_small, eye_b + Xn)
                        tlx.local_store(b_work_t, Tinv)
                        if i + 1 < INV_STEPS:
                            tlx.local_store(b_work_x, Xn)
                        tlx.fence_async_shared()
                        tlx.async_dot(
                            lhs_small,
                            b_work_t,
                            mma_small0[0],
                            use_acc=False,
                            force_async=True,
                            mBarriers=[mma_inv_a[i]],
                        )
                        if i + 1 < INV_STEPS:
                            # Xn @ Xn takes both operands from the same SMEM
                            # tile, leaving the TMEM A operand for `eye_b + Xn`.
                            tlx.async_dot(
                                b_work_x,
                                b_work_x,
                                acc_small1,
                                use_acc=False,
                                force_async=True,
                                mBarriers=[mma_inv_b[i]],
                            )
                        tlx.barrier_wait(mma_inv_a[i], phase)
                        Tinv = tlx.local_load(mma_small0[0]).to(dtype)
                        if i + 1 < INV_STEPS:
                            tlx.barrier_wait(mma_inv_b[i], phase)
                            Xn = tlx.local_load(acc_small1).to(dtype)
                    tlx.local_store(lhs_small, Tinv)
                    tlx.fence_async_shared()
                    tlx.barrier_arrive(tinv_ready[0], 1)
                    if SAVE_INTERMEDIATE:
                        # save the WY inverse [C,C] so the backward skips the
                        # Neumann recompute; kept behind tinv_ready because the
                        # solve waiting on it is the compute task's longest stall.
                        tlx.local_store(tinv_out, Tinv)
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(st_desc, tinv_out, [c * BT, 0])

                    # --- drain the output accumulator and TMA-store it ---
                    # In [D, C/2] halves: the whole fp32 tile is 64 registers per
                    # thread on this 4-warp task, which spilled the WY chain.
                    tlx.barrier_wait(mma_out_full[0], phase)
                    for j in tl.static_range(2):
                        o_h = tl.trans(tlx.local_load(tlx.subslice(out_tmem, j * BTH, BTH)))
                        tlx.local_store(
                            tlx.local_slice(o_buf[bufidx], [j * BTH, 0], [BTH, D]),
                            o_h.to(dtype),
                        )
                    tlx.fence_async_shared()
                    row = (c * BT).to(tl.int32)  # relative to this sequence's base
                    tlx.async_descriptor_store(o_desc, o_buf[bufidx], [row, col])
                    tlx.barrier_arrive(o_empty[bufidx], 1)
                    tlx.async_descriptor_store_wait(0)
                    it += 1


@triton.autotune(
    configs=[
        # async MMA needs the "default" compute task to be a full warpgroup.
        # registers=232 on the compute task + 24 on the 1-warp producer keeps the
        # whole 64K register file in the hands of the warps that hold the tiles.
        triton.Config({}, num_warps=8, num_stages=1)
    ],
    key=["H", "D", "BT"],
)
@triton.jit
def _kda_bwd_kernel(  # noqa: C901
    Q,
    K,
    V,
    G,
    DO,
    Beta,
    DQ,
    DK,
    DV,
    DG,
    DBeta,
    States,
    TinvScratch,
    cu_seqlens,
    n_seq,
    total_len,
    stride_bt_tok,
    stride_bt_h,
    max_chunks,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    LOG2_BT: tl.constexpr,
    NBUF: tl.constexpr,
):
    """Reverse VJP over the chunked scan. Each chunk TMA-loads q/k/v/g/do plus
    the forward-saved S_in and WY inverse, recomputes the forward intermediates,
    applies the VJP, and carries the state gradient dS backward.

    Algebraic reformulations that shorten the per-chunk chain:

    * ``dA = -(T^T dU)(T rhs)^T = -drhs @ U^T``. The textbook inverse VJP
      ``-T^T (dU rhs^T) T^T`` is three chained [C,C] dots; both inner factors are
      already materialized (``drhs`` and ``U``), so one dot replaces the chain.
    * dS lives in a TMEM accumulator and is decayed by ``exp(Gamma)`` in place at
      the top of the chunk. That decayed state is exactly ``diag(e^Gamma) dS``,
      so staging *it* for the two dS-consuming dots deletes the whole
      ``k_tilde = k*e^(Gamma-gamma)`` branch: ``k_hat @ dS_s == k_tilde @ dS``
      and ``betaU @ dS_s^T == dk_tilde * e^Gamma``.
    * ``U`` is never staged. Every dot that wants it takes ``betaU``, and both
      dbeta terms keep the beta factor so the single reciprocal lands on the
      [C] result -- beta is then only ever broadcast along dim 0, which is the
      axis its own layout already uses. ``dA0 = dA*beta`` and ``dA =
      dA_raw/beta`` cancel outright, so the masked accumulator *is* ``dA0``.
    * q_bar|k_bar share one [2C,D] tile and do|dKS another, so the two A-matrix
      dots merge into one M=128 dot and the dk_hat / dS-update pairs each merge
      into one K=128 dot.
    * Nothing reduces across warps. Gamma comes out of a transposed cumsum dot as
      a TMEM column, and both dGamma and dbeta's A-matrix term are column sums
      expressed as dots against an all-ones operand. The three axis-0 reductions
      they replace measured ~25% of the kernel between them.

    Tiles stay resident in SMEM/TMEM and are re-read as dot operands rather than
    held in registers; the register-resident form of this kernel spilled roughly
    800KB per chunk to local memory.
    """
    # Persistent kernel: grid=(NUM_SMS,), grid-striding over the (seq, head)
    # work-items. No warp specialization: at NBUF=1 nothing overlapped across the
    # chunk boundary anyway, and folding the producer away hands the whole 64K
    # register file to the 16 compute warps.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_work = n_seq * H
    HD: tl.constexpr = H * D
    BT2: tl.constexpr = 2 * BT
    L2E: tl.constexpr = 1.4426950408889634  # log2(e): rebase gate exps to ex2
    dtype = Q.dtype.element_ty

    # --- SMEM (192KB @ BT=64, D=128) ---------------------------------------
    # qk_buf: TMA lands q in rows [0,C) and k in rows [C,2C); the gate scaling
    # rewrites both halves in place (each thread stores back exactly the elements
    # it loaded), turning the tile into [q_bar; k_bar] with no second allocation.
    qk_buf = tlx.local_alloc((BT2, D), dtype, NBUF)
    r_buf = tlx.local_alloc((BT2, D), dtype, NBUF)  # do (TMA) | dKS
    s_buf = tlx.local_alloc((D, D), dtype, NBUF)  # S_in (TMA)
    g_buf = tlx.local_alloc((BT, D), dtype, NBUF)  # g (TMA)
    # k_hat overwrites g: g's only reader is the cumsum MMA, so the MMA's
    # completion barrier already orders the rewrite.
    kh_buf = tlx.local_alloc((BT, D), dtype, NBUF, reuse=g_buf)
    wa_buf = tlx.local_alloc((BT, D), dtype, NBUF)  # v (TMA) -> betaU -> dU -> dgamma
    # decayed dS, bf16 MMA operand; once its two dots retire the tile is
    # recycled for dU/dv/dq and the dk drain, which is what lets the rhs/U
    # staging buffer go away entirely and drops SMEM to 192KB.
    sd_buf = tlx.local_alloc((D, D), dtype, 1)
    p_buf = tlx.local_alloc((BT2, BT), dtype, 1)  # Aqk -> dAqk | dA0
    # Alone among the TMA destinations this one is 2-deep: at 8KB it is the only
    # buffer the 232KB budget can double, and doing so drops the WAR edge from
    # the next chunk's Tinv load to this chunk's triangular solve. The idle half
    # doubles as dbeta's xa0 scratch (see group F).
    t_buf = tlx.local_alloc((BT, BT), dtype, 2)
    lm_buf = tlx.local_alloc((BT, BT), dtype, 1)  # lower-inclusive ones, hoisted
    # dq/dk/dv/dg are TMA-stored. Gradient rows are H*D elements -- 1024B in
    # production and 768B at H=3, both 16B-aligned -- so the descriptor is
    # legal on every shape in the suite. This deletes the [C,D] int64 offset
    # tile (32 of the 128 registers per thread, held across the whole
    # epilogue), five masked global stores and their convergences; TMA's own
    # bounds clipping also replaces the row mask on the partial chunk.
    dg_buf = tlx.local_alloc((BT, D), tl.float32, 1, reuse=s_buf)
    # Zk gets its own tile so the g/k_hat tile has no reader left once group
    # F retires. That is what frees the gate branch for prefetching; it also
    # carries dbeta's U term between groups D and E.
    zk_buf = tlx.local_alloc((BT, D), dtype, 1)
    one_buf = tlx.local_alloc((BT, BT), dtype, 1)  # all-ones, hoisted

    # --- TMEM (512 cols: 128 for the carried dS + 3 reused [C,D] accumulators) ---
    acc_ds = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)
    acc_a = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    acc_b = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    acc_c = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    # dbeta's U term as a dot instead of an axis-1 reduction: reducing a bf16
    # [C,D] product along dim 1 costs a widening tree (PRMT/IMAD/FADD2, 79 of the
    # body's 176 PRMT). Summing over d in two K=C halves lets the existing
    # all-ones tile serve as B, so no new shared memory is needed.
    acc_dbu = tlx.local_alloc((BT, BT), tl.float32, 1, tlx.storage_kind.tmem)
    # dk_hat needs a target of its own: its two K halves become ready a chunk
    # stage apart, and acc_c_lo is rewritten by the dA dot in between.
    acc_kh = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    # Transposed cumsum lands here so exp(Gamma) can be read out as a TMEM
    # column instead of an axis-0 reduction. It shares acc_c with the A-matrix
    # accumulator, whose first write is a chunk stage later.
    acc_gt = tlx.local_alloc((D, BT), tl.float32, 1, tlx.storage_kind.tmem, reuse=acc_c)

    # Two arrival barriers: the gate branch (g, q, k) is waited on first so the
    # cumsum dot and the exp2/scale work overlap the remaining 60% of the
    # chunk's 123KB, which is otherwise exposed DRAM latency at NBUF=1.
    full = tlx.alloc_barriers(num_barriers=NBUF)
    full_s = tlx.alloc_barriers(num_barriers=NBUF)
    # Dot barriers. Each is signalled exactly once per chunk, so the wait parity
    # is just the chunk phase. Dots whose results are consumed at the same point
    # share one barrier (tcgen05 retires them in issue order, so the last one
    # certifies the rest); dots consumed at different points get their own, so a
    # short dot's result is not held hostage by a long sibling.
    mb = tlx.alloc_barriers(num_barriers=16)

    GATE_BYTES: tl.constexpr = 3 * BT * D * 2
    REST_BYTES: tl.constexpr = 2 * BT * D * 2 + D * D * 2 + BT * BT * 2

    # Dots are issued in groups and waited on once per group: tcgen05 retires a
    # CTA's dots in issue order, so a group's last barrier also certifies every
    # dot issued before it. Each wait is a CTA-wide convergence point, so what
    # the schedule below minimizes is the group count, not the dot count.
    offs_t = tl.arange(0, BT)
    offs_2 = tl.arange(0, BT2)
    lower_incl = offs_t[:, None] >= offs_t[None, :]
    lower_strict = offs_t[:, None] > offs_t[None, :]
    # Aqk keeps its lower triangle; A0 (rows [C,2C)) keeps everything.
    aa_keep = offs_2[:, None] >= offs_t[None, :]
    # L[t,r] = (r <= t): the cumsum operand, and its transpose is the
    # reverse-cumsum operand. Rows past the sequence end need no column
    # mask because TMA zero-fills g there.
    tlx.local_store(lm_buf[0], lower_incl.to(dtype))
    tlx.local_store(one_buf[0], tl.full([BT, BT], 1.0, dtype))
    z_view = tlx.local_slice(sd_buf[0], [0, 0], [BT, D])
    du_view = tlx.local_slice(sd_buf[0], [BT, 0], [BT, D])
    p_lo = tlx.local_slice(p_buf[0], [0, 0], [BT, BT])
    p_hi = tlx.local_slice(p_buf[0], [BT, 0], [BT, BT])
    acc_aa = tlx.local_alloc((BT2, BT), tl.float32, 1, tlx.storage_kind.tmem, reuse=acc_c)
    acc_c_lo = tlx.subslice(acc_c[0], 0, BT)

    it = 0
    for wid in tl.range(pid, n_work, nprog):
        pid_s = wid // H
        pid_h = wid % H
        seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
        seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
        T = seq_end - seq_beg
        num_chunks = tl.cdiv(T, BT)
        st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
        sob = seq_beg.to(tl.int64) * HD
        s_desc = tl.make_tensor_descriptor(
            States + st_base * D * D,
            shape=[num_chunks * D, D],
            strides=[D, 1],
            block_shape=[D, D],
        )
        t_desc = tl.make_tensor_descriptor(
            TinvScratch + st_base * BT * BT,
            shape=[num_chunks * BT, BT],
            strides=[BT, 1],
            block_shape=[BT, BT],
        )
        q_desc = tl.make_tensor_descriptor(Q + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        k_desc = tl.make_tensor_descriptor(K + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        v_desc = tl.make_tensor_descriptor(V + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        do_desc = tl.make_tensor_descriptor(DO + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        g_desc = tl.make_tensor_descriptor(G + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        dq_desc = tl.make_tensor_descriptor(DQ + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        dk_desc = tl.make_tensor_descriptor(DK + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        dv_desc = tl.make_tensor_descriptor(DV + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        dg_desc = tl.make_tensor_descriptor(DG + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D])
        col = pid_h * D
        tlx.local_store(acc_ds[0], tl.zeros([D, D], dtype=tl.float32))

        for ci in tl.range(0, num_chunks):
            c = num_chunks - 1 - ci
            bufidx, phase = get_bufidx_phase(it, NBUF)
            tidx = it % 2
            # Fire this chunk's TMA loads, then wait on them. Dropping the
            # producer warp hands the whole 64K register file to the 8 compute
            # warps (255 regs/thread instead of 168 across 12), which is what
            # keeps the VJP's live tiles out of local memory. Nothing was
            # overlapping across the chunk boundary anyway: at NBUF=1 the
            # producer could not start until `empty` fired at the very end.
            row = c * BT
            if ci == 0:
                # Only the first chunk of a work item fetches its own gate
                # branch; every later one was issued from its predecessor's
                # epilogue, so this arrival is already satisfied by then.
                grow = row
                tlx.barrier_expect_bytes(full[bufidx], GATE_BYTES)
                tlx.async_descriptor_load(g_desc, g_buf[bufidx], [grow, col], full[bufidx])
                tlx.async_descriptor_load(
                    q_desc,
                    tlx.local_slice(qk_buf[bufidx], [0, 0], [BT, D]),
                    [grow, col],
                    full[bufidx],
                )
                tlx.async_descriptor_load(
                    k_desc,
                    tlx.local_slice(qk_buf[bufidx], [BT, 0], [BT, D]),
                    [grow, col],
                    full[bufidx],
                )
            tlx.barrier_expect_bytes(full_s[bufidx], REST_BYTES)
            tlx.async_descriptor_load(v_desc, wa_buf[bufidx], [row, col], full_s[bufidx])
            tlx.async_descriptor_load(
                do_desc,
                tlx.local_slice(r_buf[bufidx], [0, 0], [BT, D]),
                [row, col],
                full_s[bufidx],
            )
            tlx.async_descriptor_load(t_desc, t_buf[tidx], [c * BT, 0], full_s[bufidx])
            # dq/dk stage over sd_buf, which is not rewritten until the state
            # decay, and dg over s_buf -- so this load is the only one whose
            # target an output store still occupies. Draining here rather than
            # at the end of the previous chunk hides it behind the loads
            # already in flight; measured at 1.2% when left exposed.
            tlx.async_descriptor_store_wait(0)
            tlx.async_descriptor_load(s_desc, s_buf[bufidx], [c * D, 0], full_s[bufidx])
            # === group A: gamma = cumsum_t(g) as a triangular matmul ===
            # g feeds the MMA straight from its TMA buffer, no cross-thread
            # tl.cumsum. Only the first chunk of a work item issues its own
            # pair; every later chunk's was issued from its predecessor's tail
            # (bottom of this loop), so the gate branch never pays the MMA
            # latency. mb[0] still takes exactly one arrival per chunk, so its
            # wait parity is unchanged.
            if ci == 0:
                tlx.barrier_wait(full[bufidx], phase)
                tlx.async_dot(
                    lm_buf[0],
                    g_buf[bufidx],
                    acc_a[0],
                    use_acc=False,
                    force_async=True,
                )
                # ... and the same cumsum transposed. Gamma is then the last
                # column of a [D,C] accumulator, i.e. a plain TMEM read already
                # indexed by the channel. Extracting it from the [C,D] copy
                # instead costs an axis-0 reduction plus a transposing
                # broadcast every time it multiplies the [D,D] state.
                tlx.async_dot(
                    tlx.local_trans(g_buf[bufidx]),
                    tlx.local_trans(lm_buf[0]),
                    acc_gt[0],
                    use_acc=False,
                    force_async=True,
                    mBarriers=[mb[0]],
                )
            qk_lo = tlx.local_slice(qk_buf[bufidx], [0, 0], [BT, D])
            qk_hi = tlx.local_slice(qk_buf[bufidx], [BT, 0], [BT, D])
            r_lo = tlx.local_slice(r_buf[bufidx], [0, 0], [BT, D])
            r_hi = tlx.local_slice(r_buf[bufidx], [BT, 0], [BT, D])

            rows = c * BT + offs_t
            rmask = rows < T
            # int32 token offsets: beta is [T, H] and T*H stays far under 2^31
            # on every supported shape, so the 64-bit IMAD.WIDE chain the
            # int64 form generated for this gather buys nothing.
            tok = seq_beg + rows
            b = tl.load(
                Beta + tok * stride_bt_tok + pid_h * stride_bt_h,
                mask=rmask,
                other=0.0,
            ).to(tl.float32)
            tlx.barrier_wait(mb[0], phase)
            # base-2 rebase: exp2(L2E*natural) = exp(natural). The VJP's
            # ln2 (from d/dgamma of ex2) cancels the L2E in dg, so the dg
            # formula below is unchanged.
            gam = tlx.local_load(acc_a[0])
            eg = tl.exp2(_mul_f32x2(gam, L2E))
            # Only the bf16 gates stay live to the epilogue: k_bar/k_hat/q_bar
            # are bf16 anyway, so nothing downstream can see more than 8 mantissa
            # bits of e^gamma, and the fp32 copies were what tipped the register
            # allocator into local memory at 16 warps.
            egb = eg.to(dtype)
            engb = tl.exp2(_mul_f32x2(gam, -L2E)).to(dtype)
            egc = tl.exp2(_mul_f32x2(
                tl.reshape(tlx.local_load(tlx.subslice(acc_gt[0], BT - 1, 1)), [D]),
                L2E,
            ))
            # k_hat=k*e^-gamma  k_bar=k*e^gamma  q_bar=scale*q*e^gamma.
            # The raw q/k stay in registers: the epilogue needs them, and
            # re-deriving dgamma from them is cheaper than reloading the
            # three scaled tiles back out of SMEM.
            q_raw = tlx.local_load(qk_lo)
            k_raw = tlx.local_load(qk_hi)
            # scale rides the gate rather than the epilogue: the same tile
            # builds q_bar here and unscales dq_bar at the end.
            egs = _mul_f32x2(eg, scale).to(dtype)
            tlx.local_store(qk_lo, q_raw * egs)
            tlx.local_store(qk_hi, k_raw * egb)
            tlx.local_store(kh_buf[bufidx], k_raw * engb)

            # === group B: state readout + both A-matrices ===
            # The A-matrix dot reads only the gate branch, which arrived on the
            # first barrier, so it is issued ahead of the S_in wait: the second
            # arrival group is 74KB and mbarrier spin waits are the largest
            # stall bucket in the SASS sampling.
            tlx.fence_async_shared()
            tlx.async_dot(  # [Aqk ; A0] = [q_bar ; k_bar] @ k_hat^T
                qk_buf[bufidx],
                tlx.local_trans(kh_buf[bufidx]),
                acc_aa[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[1]],
            )
            # The decay reads only the accumulator, so it also runs ahead of the
            # S_in wait. The decayed state is both the first term of the outgoing
            # dS and the operand that makes the k_tilde branch disappear; one
            # column strip at a time keeps a [D,D] fp32 tile out of registers.
            for hs in tl.static_range(D // BT):
                strip = tlx.subslice(acc_ds[0], hs * BT, BT)
                ds_s = _mul_f32x2(tlx.local_load(strip), egc[:, None])
                tlx.local_store(strip, ds_s)
                tlx.local_store(
                    tlx.local_slice(sd_buf[0], [0, hs * BT], [D, BT]),
                    ds_s.to(dtype),
                )

            tlx.barrier_wait(full_s[bufidx], phase)
            tlx.async_dot(  # k_bar @ S_in
                qk_hi,
                s_buf[bufidx],
                acc_b[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[9]],
            )
            # r1 = d(exp Gamma) * exp(Gamma), folded straight into dGamma. Read
            # back from the bf16 stage rather than carried across the wait in
            # registers, which measured worse.

            # The state readout and the A-matrices carry separate barriers even
            # though they were issued together: a group's wait blocks on its
            # slowest dot, so results consumed at different points want
            # different barriers. Here the residual is built while the A-matrix
            # dot is still running, and the A-matrix is masked while the
            # triangular solve runs.
            # The A-matrices were issued before the state readout, so tcgen05
            # retires them first: masking them here fills the readout's wait
            # instead of the solve's, which is the shorter of the two.
            tlx.barrier_wait(mb[1], phase)
            tlx.local_store(
                p_buf[0],
                tl.where(aa_keep,
                         tlx.local_load(acc_aa[0]).to(dtype), 0.0),
            )
            tlx.barrier_wait(mb[9], phase)
            rhs = _fma_f32x2(
                tlx.local_load(acc_b[0]),
                -1.0,
                tlx.local_load(wa_buf[bufidx]).to(tl.float32),
            )
            tlx.local_store(wa_buf[bufidx], rhs.to(dtype))
            tlx.fence_async_shared()
            # === group C: U = Tinv @ rhs ===
            tlx.async_dot(
                t_buf[tidx],
                wa_buf[bufidx],
                acc_b[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[2]],
            )
            # r1 = d(exp Gamma) * exp(Gamma), folded straight into dGamma.
            # Placed under the triangular solve: it reads only tiles that are
            # already resident, and sd_buf survives until dU overwrites its
            # upper half after the dbetaU wait.
            r1 = tl.zeros([D], dtype=tl.float32)
            for hs in tl.static_range(D // BT):
                r1 += tl.sum(
                    (tlx.local_load(tlx.local_slice(sd_buf[0], [0, hs * BT], [D, BT])) *
                     tlx.local_load(tlx.local_slice(s_buf[bufidx], [0, hs * BT], [D, BT]))).to(tl.float32),
                    axis=1,
                )
            tlx.barrier_wait(mb[2], phase)
            bb = b.to(dtype)
            # betaU is staged once and reused for dbeta's U term, so that term
            # carries the beta factor and needs no per-element unscale.
            ubb = tlx.local_load(acc_b[0]).to(dtype) * bb[:, None]
            tlx.local_store(wa_buf[bufidx], ubb)

            # === group D: dAqk, Z, dbetaU (both terms into one acc) ===
            tlx.fence_async_shared()
            tlx.async_dot(  # dAqk = do @ betaU^T
                r_lo,
                tlx.local_trans(wa_buf[bufidx]),
                acc_c_lo,
                use_acc=False,
                force_async=True,
            )
            tlx.async_dot(  # Z = betaU @ dS_s^T == dk_tilde * exp(Gamma)
                wa_buf[bufidx],
                tlx.local_trans(sd_buf[0]),
                acc_a[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[10]],
            )
            tlx.async_dot(  # dbetaU = Aqk^T @ do ...
                tlx.local_trans(p_lo),
                r_lo,
                acc_b[0],
                use_acc=False,
                force_async=True,
            )
            tlx.async_dot(  # ... + k_tilde @ dS, accumulated in place
                kh_buf[bufidx],
                sd_buf[0],
                acc_b[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[3]],
            )
            # dAqk and Z retire first (issue order); stage both while the two
            # dbetaU dots are still in flight.
            tlx.async_dot(
                tlx.local_trans(qk_lo),
                r_lo,
                acc_ds[0],
                use_acc=True,
                force_async=True,
            )
            tlx.barrier_wait(mb[10], phase)
            # Z stays in registers: its only consumers are the dk_hat sum and
            # zk, and the SMEM round trip it used to take cost a barrier.
            zb = tlx.local_load(acc_a[0]).to(dtype)
            tlx.local_store(
                p_lo,
                tl.where(lower_incl,
                         tlx.local_load(acc_c_lo).to(dtype), 0.0),
            )
            # dk_hat = dAqk^T@q_bar + dA0^T@k_bar, K-split at the 2C boundary: dAqk
            # is ready here but dA0 not until after dA retires, and the sum is not
            # read until the epilogue.
            tlx.fence_async_shared()
            tlx.async_dot(
                tlx.local_trans(p_lo),
                qk_lo,
                acc_kh[0],
                use_acc=False,
                force_async=True,
            )
            tlx.barrier_wait(mb[3], phase)
            dbu = tlx.local_load(acc_b[0])
            tlx.local_store(du_view, dbu.to(dtype) * bb[:, None])

            # === group E: drhs = Tinv^T @ dU ===
            tlx.fence_async_shared()
            tlx.async_dot(
                tlx.local_trans(t_buf[tidx]),
                du_view,
                acc_b[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[4]],
            )
            # dS = decay*dS + q_bar^T@do + k_bar^T@dKS. The q_bar half reads only
            # tiles that are already resident, so it is split off by K -- both
            # halves keep M=2C, so there is no half-rate M=C penalty -- and issued
            # here rather than waiting behind dKS at the end of the chunk. Its
            # result is not read until the next chunk, which is the most slack any
            # dot here has; placement was swept across the group boundaries and
            # this one is the minimum (issuing it earlier delays the solve, later
            # delays the A-matrix halves that gate the epilogue).
            # sum_d dbetaU*U with U still in registers: betaU*rb == U exactly, so
            # reloading the tile that was just written buys nothing but a
            # read-after-write barrier on it. Reduced under the solve rather than
            # in front of it -- dbeta is a [C] output, not an operand of any dot.
            tlx.local_store(zk_buf[0], dbu.to(dtype) * ubb)
            tlx.fence_async_shared()
            for ds in tl.static_range(D // BT):
                tlx.async_dot(
                    tlx.local_slice(zk_buf[0], [0, ds * BT], [BT, BT]),
                    one_buf[0],
                    acc_dbu[0],
                    use_acc=tl.constexpr(ds > 0),
                    force_async=True,
                )
            tlx.barrier_wait(mb[4], phase)
            # do and S_in have both been resident since the top of the chunk and
            # acc_a was drained by the Z readout, so this dot owes nothing to the
            # solve -- it is issued before dv has even been formed.
            tlx.async_dot(  # dq_bar = do @ S_in^T
                r_lo,
                tlx.local_trans(s_buf[bufidx]),
                acc_a[0],
                use_acc=False,
                force_async=True,
            )
            dvb = tlx.local_load(acc_b[0]).to(dtype)  # v enters linearly
            tlx.local_store(du_view, dvb)  # dU was drained by the drhs dot
            tlx.local_store(r_hi, -dvb)  # dKS
            tlx.fence_async_shared()
            tlx.async_descriptor_store(dv_desc, du_view, [row, col])

            # === group F: dA, plus the state halves of dq_bar/dk_bar ===
            tlx.async_dot(  # dA = dKS @ U^T = -(T^T dU)(T rhs)^T
                r_hi,
                tlx.local_trans(wa_buf[bufidx]),
                acc_c_lo,
                use_acc=False,
                force_async=True,
                mBarriers=[mb[5]],
            )
            tlx.async_dot(  # dk_bar = dKS @ S_in^T
                r_hi,
                tlx.local_trans(s_buf[bufidx]),
                acc_b[0],
                use_acc=False,
                force_async=True,
            )
            tlx.barrier_wait(mb[5], phase)
            # dA0 is dA*beta and dA is dA_raw/beta, so the two cancel: the
            # masked accumulator is already dA0. dbeta's A-matrix term keeps the
            # beta factor too and the single reciprocal lands on the [C] result,
            # which never needs beta broadcast along the column axis.
            tlx.local_store(zk_buf[0], zb * (k_raw * engb))
            damb = tl.where(lower_strict, tlx.local_load(acc_c_lo).to(dtype), 0.0)
            # xa0 parks in the idle half of the 2-deep Tinv buffer -- the next
            # chunk does not load into it until after this chunk's last wait --
            # which frees dbeta's column-sum dot from waiting on p_lo and lets
            # it issue with group G instead of at the tail.
            tlx.local_store(t_buf[1 - tidx], damb * tlx.local_load(p_hi))
            tlx.local_store(p_hi, damb)

            # === group G: A-matrix halves, dk_hat, and the dS handoff ===
            tlx.fence_async_shared()
            # sum_t (dA0*A0)[t, r] as trans(X) @ ones: every output column
            # repeats the column sum, so collapsing them is an axis-1 (in-warp)
            # reduction rather than a cross-warp one.
            tlx.async_dot(
                tlx.local_trans(t_buf[1 - tidx]),
                one_buf[0],
                acc_dbu[0],
                use_acc=True,
                force_async=True,
            )
            # dk_hat is issued first because the epilogue consumes it first:
            # p1 is then built while the two A-matrix halves are still running.
            tlx.async_dot(  # ... + the dA0 half, which only lands now
                tlx.local_trans(p_hi),
                qk_hi,
                acc_kh[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[11]],
            )
            tlx.async_dot(  # dk_bar += dA0 @ k_hat
                p_hi,
                kh_buf[bufidx],
                acc_b[0],
                use_acc=True,
                force_async=True,
            )
            tlx.async_dot(  # dq_bar += dAqk @ k_hat
                p_lo,
                kh_buf[bufidx],
                acc_a[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[6]],
            )
            # --- state grad handed to chunk c-1 (the ONLY sequential line) ---
            # dS = exp(Gamma)*dS + q_bar^T@do + k_bar^T@dKS, the decay
            # having already been folded into the accumulator above.
            tlx.async_dot(
                tlx.local_trans(qk_hi),
                r_hi,
                acc_ds[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[7]],
            )

            # --- un-scale the decays -------------------------------
            # dk_tilde*ktexp == Z*e^-gamma, so dk_hat and dk_tilde share
            # one e^-gamma multiply; dgamma then collapses to
            #   (dk_bar*e^gamma - (dk_hat+Z)*e^-gamma) * k + dq * q
            # which reads the raw q/k already in registers instead of
            # pulling the three scaled tiles back out of SMEM.
            # r1 is reduced out of the [D,C] state strips, so it is indexed along
            # dim 0 while dg wants it along dim 1; bridging that is a
            # convert_layout worth 6 of the chunk's barriers. Materialize it
            # under this dot group rather than after the last wait.
            # Drain dv before the first epilogue wait, not between the two:
            # it only has to retire before dq overwrites the tile, and every
            # cycle it spends in front of a wait it would otherwise spend
            # exposed. Sweeping the position, here beats after mb[11] by
            # 0.6% and after mb[6] by 0.3%; moving it up into group F or G
            # costs 2-5% because the store has not been issued long enough.
            tlx.async_descriptor_store_wait(0)
            tlx.barrier_wait(mb[11], phase)
            p1 = (tlx.local_load(acc_kh[0]).to(dtype) + zb) * engb
            tlx.barrier_wait(mb[6], phase)
            if ci + 1 < num_chunks:
                # q_bar/k_bar and k_hat have no readers left, so the successor's
                # gate branch -- 49KB, otherwise fully exposed at the top of the
                # next chunk -- transfers under the rest of this epilogue.
                grow = row - BT
                tlx.barrier_expect_bytes(full[bufidx], GATE_BYTES)
                tlx.async_descriptor_load(g_desc, g_buf[bufidx], [grow, col], full[bufidx])
                tlx.async_descriptor_load(
                    q_desc,
                    tlx.local_slice(qk_buf[bufidx], [0, 0], [BT, D]),
                    [grow, col],
                    full[bufidx],
                )
                tlx.async_descriptor_load(
                    k_desc,
                    tlx.local_slice(qk_buf[bufidx], [BT, 0], [BT, D]),
                    [grow, col],
                    full[bufidx],
                )
            p2 = tlx.local_load(acc_b[0]).to(dtype) * egb
            dq = tlx.local_load(acc_a[0]).to(dtype) * egs
            dgamma = (p2 - p1) * k_raw + dq * q_raw
            tlx.local_store(z_view, p1 + p2)
            tlx.local_store(du_view, dq)
            tlx.fence_async_shared()
            tlx.async_descriptor_store(dk_desc, z_view, [row, col])
            tlx.async_descriptor_store(dq_desc, du_view, [row, col])

            # dg = reverse_cumsum_incl(dgamma) + dGamma. Both halves are
            # triangular matmuls into the same accumulator: L^T[t,r]=1 for r>=t
            # gives the reverse scan, and an all-ones operand turns
            # dGamma = sum_t(dk_tilde*k_tilde) into a column broadcast. Done as
            # an axis-0 tl.sum instead, that term costs a 16-warp cross-warp
            # reduction; here it rides the idle tensor core. wa_buf/kh_buf/p_lo
            # have all been drained by group F, so no buffer is added.
            tlx.local_store(wa_buf[bufidx], dgamma)
            tlx.fence_async_shared()
            # The scan reads only dgamma, so it goes out before the other two
            # operands are staged; the accumulate order into acc_a is preserved
            # because a CTA's dots retire in issue order.
            tlx.async_dot(
                tlx.local_trans(lm_buf[0]),
                wa_buf[bufidx],
                acc_a[0],
                use_acc=False,
                force_async=True,
            )
            tlx.fence_async_shared()
            tlx.async_dot(
                one_buf[0],
                zk_buf[0],
                acc_a[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[8]],
            )
            # r1 is reduced out of the [D,C] state strips, so it is indexed
            # along dim 0 while dg wants it along dim 1; bridging that is a
            # convert_layout. Run here it sits in the shadow of the dg dots
            # rather than in front of them.
            r1t = tl.zeros([BT, D], dtype=tl.float32) + r1[None, :]
            tlx.barrier_wait(mb[8], phase)
            # add.f32x2: dg is the one [C,D] fp32 elementwise pass left, and
            # halving its instruction count is worth 1% at production clocks.
            tlx.local_store(dg_buf[0], _add_f32x2(tlx.local_load(acc_a[0]), r1t))
            tlx.fence_async_shared()
            tlx.async_descriptor_store(dg_desc, dg_buf[0], [row, col])
            # dbeta last: its [C] result is reduced along dim 1 but stored along
            # dim 0, and that convert_layout runs under the dg store rather than
            # in front of it.
            # U is never staged: every dot that wants it takes betaU instead and
            # divides the [C,C]/[C] result by beta afterwards. beta is a sigmoid
            # so it is only ever zero on rows past the sequence end, where betaU
            # is zero too and the guarded reciprocal keeps the term at zero.
            # One division rather than a reciprocal and a scale. 1/BT is a
            # power of two so folding it in is exact, and this sits on the
            # fully exposed tail after the dg store.
            rbn = (1.0 / BT) / tl.where(b > 0.0, b, 1.0)
            tl.store(
                DBeta + tok * stride_bt_tok + pid_h * stride_bt_h,
                tl.sum(tlx.local_load(acc_dbu[0]), axis=1) * rbn,
                mask=rmask,
            )
            # The successor's cumsum pair is issued here, once acc_a and acc_gt
            # have been drained, so the next chunk's gate branch starts on a
            # result that is already in flight.
            if ci + 1 < num_chunks:
                nbufidx, nphase = get_bufidx_phase(it + 1, NBUF)
                tlx.barrier_wait(full[nbufidx], nphase)
                tlx.async_dot(
                    lm_buf[0],
                    g_buf[nbufidx],
                    acc_a[0],
                    use_acc=False,
                    force_async=True,
                )
                tlx.async_dot(
                    tlx.local_trans(g_buf[nbufidx]),
                    tlx.local_trans(lm_buf[0]),
                    acc_gt[0],
                    use_acc=False,
                    force_async=True,
                    mBarriers=[mb[0]],
                )
            # The dS dot was issued ahead of the dg pair, so mb[8] above already
            # certified it -- tcgen05 retires a CTA's dots in issue order.
            it += 1


def _ensure_allocator() -> None:
    # Set every call (not gated): the TMA scratch allocator does not persist
    # across the autograd fwd->bwd boundary, so the bwd's descriptor kernels
    # need it re-set even after the fwd already set it.
    def _alloc(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(_alloc)


def kimi_delta_attention_ws_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    save_intermediate: bool = True,
):
    """Warp-specialized chunkwise KDA forward. q/k/v/g: [1,T,H,D]; beta:[1,T,H].

    Training path (``save_intermediate=True``, the default): the forward also
    stores each chunk's entering state ``S_in`` [D,D] and WY inverse ``Tinv``
    [C,C] to bf16 scratch and returns ``(o, states, tinv)``, so the backward
    reuses them instead of recomputing the state scan and the Neumann
    inverse. Inference path (``save_intermediate=False``): skips that save
    epilogue and returns just ``o`` (byte-identical output, no scratch).
    """
    assert q.dim() == 4 and q.shape[0] == 1
    _, T, H, D = q.shape
    assert D == 128, "KDA TLX currently requires head_dim == 128"
    BT = chunk_size
    assert BT == 64, "async MMA kernel currently requires chunk_size == 64"
    log2_bt = BT.bit_length() - 1
    _ensure_allocator()

    q, k, v = (x.contiguous() for x in (q, k, v))
    # g staged as bf16 in SMEM (frees room for the epilogue o_buf); the kernel
    # casts back to fp32 before the decay cumsum.
    g = g.contiguous().to(torch.bfloat16)
    beta = beta.contiguous().to(torch.float32)
    o = torch.empty_like(v)

    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    beta2 = beta.view(T, H)

    if save_intermediate:
        seqlens = (cu[1:] - cu[:-1]).tolist()
        max_chunks = max((int(s) + BT - 1) // BT for s in seqlens)
        states = torch.empty(n_seq * H, max_chunks, D, D, dtype=torch.bfloat16, device=q.device)
        tinv = torch.empty(n_seq * H, max_chunks, BT, BT, dtype=torch.bfloat16, device=q.device)
        save_s, save_t = states.view(-1), tinv.view(-1)
    else:
        max_chunks = 0
        states = tinv = None
        save_s = save_t = o.view(T, H * D)  # dummy; SAVE_INTERMEDIATE skips stores

    nbuf = 1  # NBUF=2 exceeds the per-CTA SMEM limit

    # Persistent grid: one program per SM (capped at the work-item count), each
    # grid-striding over the n_seq*H (seq, head) work-items.
    n_work = n_seq * H
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(num_sms, n_work), )
    _kda_ws_fwd_kernel[grid](
        q.view(T, H * D),
        k.view(T, H * D),
        v.view(T, H * D),
        g.view(T, H * D),
        beta2,
        o.view(T, H * D),
        save_s,
        save_t,
        cu,
        n_seq,
        max_chunks,
        beta2.stride(0),
        beta2.stride(1),
        float(scale),
        H=H,
        D=D,
        BT=BT,
        LOG2_BT=log2_bt,
        NBUF=nbuf,
        SAVE_INTERMEDIATE=save_intermediate,
        # The explicit dot layout covers all 8 warps: producer 1 + epilogue 1
        # + compute 6. Additional compiler stages do not help this manual
        # mbarrier pipeline.
        num_warps=8,
        num_stages=1,
    )
    if save_intermediate:
        return o, states, tinv
    return o


def kimi_delta_attention_ws(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, None]:
    del cu_seqlens_cpu
    assert cu_seqlens is not None
    # Inference-only entry: skip the backward-scratch save epilogue.
    o = kimi_delta_attention_ws_fwd(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, save_intermediate=False)
    return o, None


def kimi_delta_attention_ws_bwd(q, k, v, g, beta, do, states, tinv, *, scale=1.0, cu_seqlens, chunk_size=64):
    """Chunkwise KDA backward (single warp-spec kernel). `states` (S_in per chunk)
    and `tinv` (WY inverse per chunk) are the bf16 scratch saved by the forward, so
    the backward runs no state scan and skips the Neumann inverse recompute.
    Returns dq, dk, dv, dg, dbeta."""
    _, T, H, D = q.shape
    assert D == 128, "KDA TLX currently requires head_dim == 128"
    BT = chunk_size
    assert BT == 64, "async MMA kernel currently requires chunk_size == 64"
    log2_bt = BT.bit_length() - 1
    _ensure_allocator()
    q, k, v, do = (x.contiguous() for x in (q, k, v, do))
    g = g.contiguous().to(torch.bfloat16)
    beta = beta.contiguous().to(torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty(1, T, H, D, dtype=torch.float32, device=q.device)
    dbeta = torch.empty(1, T, H, dtype=torch.float32, device=q.device)

    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    max_chunks = states.shape[1]

    HD = H * D
    q2, k2, v2, g2, do2 = (x.view(T, HD) for x in (q, k, v, g, do))
    dq2, dk2, dv2, dg2 = (x.view(T, HD) for x in (dq, dk, dv, dg))
    beta2 = beta.view(T, H)
    dbeta2 = dbeta.view(T, H)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(num_sms, n_seq * H), )
    _kda_bwd_kernel[grid](
        q2,
        k2,
        v2,
        g2,
        do2,
        beta2,
        dq2,
        dk2,
        dv2,
        dg2,
        dbeta2,
        states.view(-1),
        tinv.view(-1),
        cu,
        n_seq,
        T,
        beta2.stride(0),
        beta2.stride(1),
        max_chunks,
        float(scale),
        H=H,
        D=D,
        BT=BT,
        LOG2_BT=log2_bt,
        NBUF=1,
    )
    return dq, dk, dv, dg, dbeta


class _KDAWSFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, cu_seqlens):
        o, states, tinv = kimi_delta_attention_ws_fwd(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens,
                                                      save_intermediate=True)
        ctx.save_for_backward(q, k, v, g, beta, cu_seqlens, states, tinv)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, cu_seqlens, states, tinv = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = kimi_delta_attention_ws_bwd(q, k, v, g, beta, do, states, tinv, scale=ctx.scale,
                                                            cu_seqlens=cu_seqlens)
        return dq, dk, dv, dg, dbeta, None, None


def kimi_delta_attention_ws_autograd(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None):
    """Full fwd+bwd autograd entry (warp-spec fwd + warp-spec bwd)."""
    del cu_seqlens_cpu
    assert cu_seqlens is not None
    o = _KDAWSFunction.apply(q, k, v, g, beta, scale, cu_seqlens)
    return o, None


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, space="full"):
    """Full fwd+bwd autograd entry for `tlx.ops.kimi_delta_attention`."""
    del space
    return kimi_delta_attention_ws_autograd(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens,
                                            cu_seqlens_cpu=cu_seqlens_cpu)
