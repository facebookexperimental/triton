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
The fast path centers gamma at half of each chunk's total so the reciprocal
exponentials stay within 60 nats, and carries S scaled by exp(Gamma/2).
The main forward kernel checks that regime on every chunk (sum_t |g| <= 120,
beta >= 2^-60, ||A||_inf <= 1/2); flagged heads are recomputed by the
recurrent update, which has a matching VJP. ||A||_inf <= 1/2 also bounds the
truncation of the doubling inverse (see the epilogue).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.jit import constexpr_function


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



def _make_tlx_layout(*args, **kwargs):
    return tlx.layout(*args, **kwargs)


_make_layout_cf = constexpr_function(_make_tlx_layout)


# Match the Blackwell dot-operand #linear layout for a [64, 128] tile:
# lane bits cover rows [1, 2, 4, 8] and column 32, warp bits cover rows
# [16, 32] and column 64, and each thread owns 32 contiguous columns.

# The D=64 tile uses the same 256 threads and moves column bit 16 from a
# register into the final warp bit. Each thread then holds 16 columns.

# Row slice of the transposed D128 decay TMEM layout. Move the 64 beta
# scalars before broadcasting; converting the FP32 [64,128] broadcast needs
# 32 KB of shared-memory scratch on the current compiler.


# Stable-kernel tuples keep the launch ABI compact. Config fields are:
# (H, D, V token stride, beta token stride, beta head stride, scale,
#  save intermediates, save work, packed scratch, final output,
#  gradient token stride [backward only]). Metadata is (cu, order, mask,
# n_seq, max_chunks); the opaque work payload is U on stable heads and beta*U
# on fast heads. The matching device selection determines its representation.


@triton.jit
def _stable_chunk_metadata(metadata, head, H: tl.constexpr, PACKED: tl.constexpr):
    cu, order, _selection, n_seq, max_chunks = metadata
    seq = tl.program_id(0) // H
    begin = tl.load(cu + seq).to(tl.int64)
    length = (tl.load(cu + seq + 1).to(tl.int64) - begin).to(tl.int32)
    chunks = tl.cdiv(length, 64)
    if PACKED:
        prefix = tl.load(order + n_seq + seq).to(tl.int64)
        base = prefix * H + head * chunks
    else:
        base = (seq.to(tl.int64) * H + head) * max_chunks
    return begin, length, chunks, base


@triton.jit
def _stable_token_rows(begin, chunk, count):
    rows = tl.arange(0, 64)
    return begin + chunk * 64 + rows, rows < count


@triton.jit
def _stable_point(inputs, token, head, Config: tl.constexpr):
    q, k, v, g, beta = inputs
    H: tl.constexpr = Config[0]
    D: tl.constexpr = Config[1]
    cols = tl.arange(0, D)
    offset = token.to(tl.int64) * (H * D) + head * D + cols
    # Triton TR003: exact D64/D128 head tiles declare their bounds.
    qr = tl.load(q + offset, mask=cols < D).to(tl.float32)
    kr = tl.load(k + offset, mask=cols < D).to(tl.float32)
    vr = tl.load(
        v + token.to(tl.int64) * Config[2] + head * D + cols, mask=cols < D
    ).to(tl.float32)
    ar = tl.exp2(tl.load(g + offset, mask=cols < D).to(tl.float32) * 1.4426950408889634)
    br = tl.load(beta + token.to(tl.int64) * Config[3] + head * Config[4]).to(
        tl.float32
    )
    return qr, kr, vr, ar, br


@triton.jit
def _stable_forward_step(inputs, outputs, token, head, state, Config: tl.constexpr):
    q, k, v, a, beta = _stable_point(inputs, token, head, Config)
    decayed = a[:, None] * state
    u = (v - tl.sum(k[:, None] * decayed, axis=0)).to(inputs[0].dtype.element_ty)
    update = (u * beta.to(u.dtype)).to(u.dtype)
    state = decayed + k[:, None] * update.to(tl.float32)[None, :]
    offset = token.to(tl.int64) * (Config[0] * Config[1]) + head * Config[1]
    tl.store(
        outputs[0] + offset + tl.arange(0, Config[1]),
        tl.sum(q[:, None] * state, axis=0) * Config[5],
    )
    if Config[7]:
        # Stable heads carry unscaled U, including beta=0, in the opaque work
        # payload. The matching backward selects the same head on the device.
        tl.store(outputs[3] + offset + tl.arange(0, Config[1]), u)
    return state


@triton.jit
def _kda_stable_fwd_kernel(inputs, outputs, metadata, Config: tl.constexpr):
    H: tl.constexpr = Config[0]
    D: tl.constexpr = Config[1]
    head = tl.program_id(0) % H
    # The main kernel ran the fast path on every head and flagged the ones
    # whose decay range, beta or ||A||_inf leave its numerical regime.
    flags = metadata[2] + 2 * tl.program_id(0)
    selected = (tl.load(flags) | tl.load(flags + 1)) != 0
    begin, length, chunks, base = _stable_chunk_metadata(metadata, head, H, Config[8])
    chunks = tl.where(selected, chunks, 0)
    carried = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)
    tlx.local_store(carried[0], tl.zeros((D, D), tl.float32))
    rr, cc = tl.arange(0, D)[:, None], tl.arange(0, D)[None, :]
    for ci in range(chunks):
        chunk = ci
        count = tl.minimum(64, length - chunk * 64)
        state = tlx.local_load(carried[0])
        if Config[6]:
            tl.store(outputs[1] + (base + chunk) * D * D + rr * D + cc, state)
            # The stable VJP uses U and S_in rather than the inverse scratch.
            rows = tl.arange(0, 64)
            tl.store(
                outputs[2] + (base + chunk) * 4096 + rows[:, None] * 64 + rows[None, :],
                0.0,
            )
        for step in range(count):
            local = step
            token = begin + chunk * 64 + local
            state = _stable_forward_step(inputs, outputs, token, head, state, Config)
        tlx.local_store(carried[0], state)
    if Config[9] and selected:
        state_base = tl.program_id(0).to(tl.int64) * D * D
        tl.store(outputs[4] + state_base + rr * D + cc, tlx.local_load(carried[0]))


@triton.jit
def _stable_residual_tile(
    inputs,
    payload,
    begin,
    chunk,
    count,
    head,
    entering,
    Config: tl.constexpr,
    HAS_PAYLOAD: tl.constexpr,
    SCALED_PAYLOAD,
):
    D: tl.constexpr = Config[1]
    rows = tl.arange(0, 64)
    cols = tl.arange(0, D)
    tokens, valid = _stable_token_rows(begin, chunk, count)
    offsets = tokens[:, None] * (Config[0] * D) + head * D + cols[None, :]
    if HAS_PAYLOAD:
        residuals = tl.load(
            payload + offsets, valid[:, None] & (cols[None, :] < D), 0.0
        )
        raw_payload = residuals
        if SCALED_PAYLOAD:
            beta_decode = (
                tl.load(inputs[4] + tokens * Config[3] + head * Config[4], valid, 1.0)
                .to(residuals.dtype)
                .to(tl.float32)
            )
            residuals = (
                residuals.to(tl.float32)
                / tl.where(beta_decode > 0.0, beta_decode, 1.0)[:, None]
            ).to(residuals.dtype)
    else:
        residuals = tl.zeros((64, D), inputs[0].dtype.element_ty)
        state = entering
        for step in range(count):
            local = step
            token = begin + chunk * 64 + local
            _q, k, v, a, beta = _stable_point(inputs, token, head, Config)
            decayed = a[:, None] * state
            u = (v - tl.sum(k[:, None] * decayed, axis=0)).to(residuals.dtype)
            update = (u * beta.to(u.dtype)).to(u.dtype)
            state = decayed + k[:, None] * update.to(tl.float32)[None, :]
            residuals = tl.where(rows[:, None] == step, u[None, :], residuals)
    beta = tl.load(inputs[4] + tokens * Config[3] + head * Config[4], valid, 0.0)
    updates = (residuals * beta.to(residuals.dtype)[:, None]).to(residuals.dtype)
    if HAS_PAYLOAD:
        if SCALED_PAYLOAD:
            updates = raw_payload
    return residuals, updates


@triton.jit
def _stable_state_before(
    inputs,
    buffers,
    entering,
    begin,
    chunk,
    head,
    step,
    Config: tl.constexpr,
):
    rows = tl.arange(0, 64)
    updates = tlx.local_load(buffers[0]).to(tl.float32)
    state = entering
    for previous in range(step + 1):
        token = begin + chunk * 64 + previous
        _q, key, _v, decay, _beta = _stable_point(inputs, token, head, Config)
        state = decay[:, None] * state
        if previous < step:
            update = tl.sum(tl.where(rows[:, None] == previous, updates, 0.0), axis=0)
            state += key[:, None] * update[None, :]
    return state


@triton.jit
def _stable_backward_point(
    inputs, outputs, buffers, decayed, token, head, step, Config: tl.constexpr
):
    D: tl.constexpr = Config[1]
    q, k, _v, a, beta = _stable_point(inputs, token, head, Config)
    cols = tl.arange(0, D)
    offset = token.to(tl.int64) * (Config[0] * D) + head * D
    do = tl.load(outputs[5] + offset + cols, mask=cols < D).to(tl.float32)
    work = tlx.local_load(buffers[1]).to(tl.float32)
    u = tl.sum(tl.where(tl.arange(0, 64)[:, None] == step, work, 0.0), axis=0)
    ds = tlx.local_load(buffers[2]) + Config[5] * q[:, None] * do[None, :]
    x = tl.sum(k[:, None] * ds, axis=0)
    rounded_beta = beta.to(inputs[0].dtype.element_ty).to(tl.float32)
    update = (
        (u.to(inputs[0].dtype.element_ty) * beta.to(inputs[0].dtype.element_ty))
        .to(inputs[0].dtype.element_ty)
        .to(tl.float32)
    )
    dv = rounded_beta * x
    dk = tl.sum(ds * update[None, :], axis=1) - tl.sum(decayed * dv[None, :], axis=1)
    before_grad = ds - k[:, None] * dv[None, :]
    dg = tl.sum(before_grad * decayed, axis=1)
    dq = Config[5] * (
        tl.sum(decayed * do[None, :], axis=1) + k * tl.sum(update * do, axis=0)
    )
    grad_offset = token.to(tl.int64) * Config[10] + head * D
    tl.store(outputs[0] + grad_offset + cols, dq)
    tl.store(outputs[1] + grad_offset + cols, dk)
    tl.store(outputs[2] + grad_offset + cols, dv)
    tl.store(outputs[3] + offset + cols, dg)
    tl.store(outputs[4] + token.to(tl.int64) * Config[0] + head, tl.sum(u * x, axis=0))
    tlx.local_store(buffers[2], before_grad * a[:, None])


@triton.jit
def _stable_backward_chunk(
    inputs,
    outputs,
    buffers,
    payload,
    entering,
    begin,
    chunk,
    count,
    head,
    Config: tl.constexpr,
    HAS_PAYLOAD: tl.constexpr,
    SCALED_PAYLOAD,
):
    u, update = _stable_residual_tile(
        inputs,
        payload,
        begin,
        chunk,
        count,
        head,
        entering,
        Config,
        HAS_PAYLOAD,
        SCALED_PAYLOAD,
    )
    tlx.local_store(buffers[0], update)
    tlx.local_store(buffers[1], u)
    tlx.fence_async_shared()
    for jj in range(count):
        step = count - 1 - jj
        decayed = _stable_state_before(
            inputs,
            buffers,
            entering,
            begin,
            chunk,
            head,
            step,
            Config,
        )
        local = step
        token = begin + chunk * 64 + local
        _stable_backward_point(
            inputs, outputs, buffers, decayed, token, head, step, Config
        )


@triton.jit
def _stable_initial_gradient(final, D: tl.constexpr, HAS_FINAL: tl.constexpr):
    if HAS_FINAL:
        rr, cc = tl.arange(0, D)[:, None], tl.arange(0, D)[None, :]
        base = tl.program_id(0).to(tl.int64) * D * D
        result = tl.load(final + base + rr * D + cc, mask=(rr < D) & (cc < D)).to(
            tl.float32
        )
    else:
        result = tl.zeros((D, D), tl.float32)
    return result


@triton.jit
def _kda_stable_bwd_kernel(
    inputs,
    outputs,
    scratch,
    metadata,
    Config: tl.constexpr,
    HAS_PAYLOAD: tl.constexpr,
    HAS_FINAL: tl.constexpr,
):
    H: tl.constexpr = Config[0]
    D: tl.constexpr = Config[1]
    head = tl.program_id(0) % H
    begin, length, chunks, base = _stable_chunk_metadata(metadata, head, H, Config[8])
    # A fast forward head stores (I + A)^-1, whose first diagonal entry is
    # exactly 1; the bounded forward zero-fills its inverse slots.
    fast_forward = (
        tl.load(metadata[2] + base * 4096, mask=chunks > 0, other=1.0).to(tl.float32)
        == 1.0
    )
    scaled_payload = fast_forward
    if not HAS_FINAL:
        chunks = tl.where(fast_forward, 0, chunks)
    update = tlx.local_alloc((64, D), inputs[0].dtype.element_ty, 1)
    residual = tlx.local_alloc((64, D), inputs[0].dtype.element_ty, 1)
    gradient = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)
    buffers = (update[0], residual[0], gradient[0])
    rr, cc = tl.arange(0, D)[:, None], tl.arange(0, D)[None, :]
    tlx.local_store(gradient[0], _stable_initial_gradient(scratch[2], D, HAS_FINAL))
    for ci in range(chunks):
        chunk = chunks - 1 - ci
        count = tl.minimum(64, length - chunk * 64)
        entering = tl.load(
            scratch[0] + (base + chunk) * D * D + rr * D + cc, mask=(rr < D) & (cc < D)
        ).to(tl.float32)
        if HAS_FINAL:
            # A fast forward head saved its state scaled by exp(Gamma/2).
            tokens, valid = _stable_token_rows(begin, chunk, count)
            gamma_total = tl.sum(
                tl.load(
                    inputs[3] + tokens[:, None] * (H * D) + head * D + cc,
                    mask=valid[:, None] & (cc < D),
                    other=0.0,
                ).to(tl.float32),
                axis=0,
            )
            unshift = tl.exp2(gamma_total * -0.7213475204444817)
            entering = tl.where(scaled_payload, entering * unshift[:, None], entering)
        _stable_backward_chunk(
            inputs,
            outputs,
            buffers,
            scratch[1],
            entering,
            begin,
            chunk,
            count,
            head,
            Config,
            HAS_PAYLOAD,
            scaled_payload,
        )


@triton.jit
def get_bufidx_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    """Keep the existing warp-spec ring phase on beta builds without warp_spec.py."""
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase


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
    SavedBetaU,
    FinalS,
    cu_seqlens,
    WorkOrder,
    UnsafeFlags,
    n_seq,
    max_chunks,
    stride_bt_tok,
    stride_bt_h,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    NBUF: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    SAVE_BETA_U: tl.constexpr,
    PACKED_SCRATCH: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
    SORT_WORK: tl.constexpr,
    BETA_IN_PRODUCER: tl.constexpr,
    FUSE_A_AQK: tl.constexpr,
    PRODUCER_REGISTERS: tl.constexpr,
    COMPUTE_REGISTERS: tl.constexpr,
    EPILOGUE_REGISTERS: tl.constexpr,
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
    _KDA_BETA_ROW_LAYOUT: tl.constexpr = _make_layout_cf(
        shape=((4, 8, 4, 2), (2, 2, 8)),
        stride=((2, 0, 0, 0), (1, 0, 8)),
    )
    _KDA_DOT_LAYOUT: tl.constexpr = _make_layout_cf(
        shape=((16, 2, 4, 2), (32,)),
        stride=((128, 32, 2048, 64), (1,)),
    )
    _KDA_DOT_LAYOUT_D64: tl.constexpr = _make_layout_cf(
        shape=((16, 2, 4, 2), (16,)),
        stride=((64, 32, 1024, 16), (1,)),
    )
    _KDA_GATE_INPUT_LAYOUT: tl.constexpr = _make_layout_cf(
        shape=((4, 8, 4, 2), (2, 2, 8)),
        stride=((256, 1, 32, 16), (128, 8, 1024)),
    )
    # D=64 decay check: one channel per producer thread (two warps repeat the
    # other two) and 16 rows in registers, so the sum never leaves the thread.
    if D == 64:
        _KDA_GATE_COLUMN_LAYOUT: tl.constexpr = _make_layout_cf(
            shape=((32, 2, 2), (16,)), stride=((1, 32, 0), (64,))
        )
    # Rows of the epilogue's [C, C] TMEM tile: each warp owns 16 of them, with
    # the two column halves of a row in lanes l and l + 16.
    _KDA_TILE_ROW_LAYOUT: tl.constexpr = _make_layout_cf(
        shape=((16, 2, 4), (1,)), stride=((1, 0, 16), (0,))
    )
    # Rows l, l + 16, l + 32 and l + 48: one from each of those warps.
    _KDA_WARP_ROWS_LAYOUT: tl.constexpr = _make_layout_cf(
        shape=((16, 2, 4), (4,)), stride=((1, 0, 0), (16,))
    )
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_work = n_seq * H

    HD: tl.constexpr = H * D
    WORK_WIDTH: tl.constexpr = D if D >= 2 * BT else 2 * BT
    dtype = Q.dtype.element_ty
    q_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    k_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    v_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    g_buf = tlx.local_alloc((BT, D), dtype, NBUF)  # g staged bf16; cumsum in fp32
    if BETA_IN_PRODUCER:
        beta_buf = tlx.local_alloc((BT,), tl.float32, NBUF)
    o_buf = tlx.local_alloc((BT, D), dtype, NBUF)
    if SAVE_BETA_U:
        # Keep beta*U until the epilogue has issued its TMA store. The compute
        # task's U operand is shared with the WY inverse and cannot be stored
        # asynchronously after the next chunk starts reusing it.
        beta_u_buf = tlx.local_alloc((BT, D), dtype, 1)

    mma_b_persist = tlx.local_alloc((BT, D), dtype, 1)
    mma_b_work = tlx.local_alloc((BT, WORK_WIDTH), dtype, 1)
    mma_aqk_b = tlx.local_alloc((BT, BT), dtype, 1)
    # The cumsum operand is a loop invariant. It used to share the WY chain's
    # scratch, which forced a [C,C] SMEM store every chunk; its own 8KB tile lets
    # it be written once for the whole kernel.
    mma_lmat = tlx.local_alloc((BT, BT), dtype, 1)
    mma_b_state = tlx.local_alloc((D, D), dtype, 1)
    mma_lhs_state_reads = tlx.local_alloc((WORK_WIDTH, D), dtype, 1)
    mma_lhs_small = tlx.local_alloc((BT, BT), dtype, 1, tlx.storage_kind.tmem)
    mma_small0 = tlx.local_alloc((BT, BT), tl.float32, 1, tlx.storage_kind.tmem)
    # The state read and gamma views share storage after their MMA readers retire.
    mma_state_reads_storage_alias = tlx.storage_alias_spec(
        storage=tlx.storage_kind.tmem
    )
    mma_state_reads = tlx.local_alloc(
        (D, WORK_WIDTH),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
        reuse=mma_state_reads_storage_alias,
    )
    mma_u = tlx.local_alloc((BT, WORK_WIDTH), tl.float32, 1, tlx.storage_kind.tmem)
    mma_gamma = tlx.local_alloc(
        (D, BT),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
        reuse=mma_state_reads_storage_alias,
    )
    mma_state_reads_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            mma_state_reads,
            mma_gamma,
            group_type=tlx.reuse_group_type.shared,
        )
    )
    mma_state = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)

    full = tlx.alloc_barriers(num_barriers=NBUF)
    full_qv = tlx.alloc_barriers(num_barriers=NBUF)
    if BETA_IN_PRODUCER:
        beta_ready = tlx.alloc_barriers(num_barriers=NBUF)
    empty = tlx.alloc_barriers(num_barriers=NBUF)
    o_empty = tlx.alloc_barriers(num_barriers=NBUF)
    if SAVE_INTERMEDIATE:
        # Aqk becomes SaveT's asynchronous TMA source. o_empty signals when
        # the store is issued, so a separate barrier protects its completion.
        save_t_empty = tlx.alloc_barriers(num_barriers=1)
    if SAVE_BETA_U:
        beta_u_full = tlx.alloc_barriers(num_barriers=1)
        beta_u_empty = tlx.alloc_barriers(num_barriers=1)
    mma_small0_full = tlx.alloc_barriers(num_barriers=1)
    # One barrier per doubling step, so a wait parity never has to be derived
    # from the iteration index: the first three steps signal once per chunk,
    # and the optional fourth tracks its own count.
    mma_inv_a = tlx.alloc_barriers(num_barriers=4)
    mma_inv_b = tlx.alloc_barriers(num_barriers=4)
    mma_aqk_full = tlx.alloc_barriers(num_barriers=1)
    tinv_ready = tlx.alloc_barriers(num_barriers=1)
    state_ready = tlx.alloc_barriers(num_barriers=1)
    mma_gamma_p = tlx.alloc_barriers(num_barriers=1)
    mma_wide_full = tlx.alloc_barriers(num_barriers=1)
    mma_out_full = tlx.alloc_barriers(num_barriers=1)
    mma_gamma_full = tlx.alloc_barriers(num_barriers=1)
    mma_state_full = tlx.alloc_barriers(num_barriers=1)
    # Per-row warp maxima of ||A||'s row sums, for the inverse epilogue.
    row_max = tlx.local_alloc((BT,), tl.float32, 1)
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
        with tlx.async_task(num_warps=4, registers=PRODUCER_REGISTERS):
            DQ: tl.constexpr = D // 4
            b_state_p = mma_b_state[0]
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                if SORT_WORK:
                    pid_s = tl.load(WorkOrder + pid_s)
                pid_h = wid % H
                seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
                seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
                T = seq_end - seq_beg
                num_chunks = tl.cdiv(T, BT)
                if PACKED_SCRATCH:
                    chunk_prefix = tl.load(WorkOrder + n_seq + pid_s)
                    st_base = (chunk_prefix * H + pid_h * num_chunks).to(tl.int64)
                else:
                    st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
                prev_decay = tl.full([D], 1.0, tl.float32)
                decay_unsafe = tl.zeros([D], tl.int32)
                beta_unsafe = tl.zeros([BT], tl.int32)
                # Per-sequence descriptors: base at this sequence's start and use
                # shape=[T, HD] so the TMA globalDim is the seqlen (<= ~14000), not
                # the packed total_len (which exceeds the TMA globalDim limit at
                # prod). Rows are indexed relative (row = c*BT); the [T, HD] bound
                # also clamps the last partial chunk.
                sob = seq_beg.to(tl.int64) * HD
                q_desc = tl.make_tensor_descriptor(
                    Q + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
                )
                k_desc = tl.make_tensor_descriptor(
                    K + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
                )
                v_desc = tl.make_tensor_descriptor(
                    V + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
                )
                g_desc = tl.make_tensor_descriptor(
                    G + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
                )
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
                    tlx.async_descriptor_load(
                        g_desc, g_buf[bufidx], [row, col], full[bufidx]
                    )
                    tlx.async_descriptor_load(
                        k_desc, k_buf[bufidx], [row, col], full[bufidx]
                    )
                    tlx.barrier_expect_bytes(full_qv[bufidx], 2 * QK_BYTES)
                    tlx.async_descriptor_load(
                        q_desc, q_buf[bufidx], [row, col], full_qv[bufidx]
                    )
                    tlx.async_descriptor_load(
                        v_desc, v_buf[bufidx], [row, col], full_qv[bufidx]
                    )
                    if BETA_IN_PRODUCER:
                        beta_rows = row + tl.arange(0, BT)
                        beta_values = tl.load(
                            Beta
                            + (seq_beg + beta_rows) * stride_bt_tok
                            + pid_h * stride_bt_h,
                            mask=beta_rows < T,
                            other=0.0,
                        ).to(tl.float32)
                        tlx.local_store(beta_buf[bufidx], beta_values)
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(beta_ready[bufidx], 1)
                    if D == 64:
                        # Fast-path regime: centered gate exponents stay within
                        # 60 nats when sum_t |g| <= 120. The legacy TMEM lowering
                        # cannot slice a 64-row accumulator, so read g from SMEM.
                        tlx.barrier_wait(full[bufidx], phase)
                        chunk_decay = tl.zeros([D], tl.float32)
                        for r in tl.static_range(BT // 16):
                            g_rows = tlx.local_load(
                                tlx.local_slice(g_buf[bufidx], [r * 16, 0], [16, D]),
                                layout=_KDA_GATE_COLUMN_LAYOUT,
                            )
                            chunk_decay += tl.sum(tl.abs(g_rows.to(tl.float32)), axis=0)
                        decay_unsafe |= (~(chunk_decay <= 120.0)).to(tl.int32)
                    # The fast VJP divides by beta; zero and tiny values take
                    # the bounded recurrence. Masked rows read as 1.
                    beta_rows = row + tl.arange(0, BT)
                    beta_check = tl.load(
                        Beta
                        + (seq_beg + beta_rows) * stride_bt_tok
                        + pid_h * stride_bt_h,
                        mask=beta_rows < T,
                        other=1.0,
                    ).to(tl.float32)
                    beta_unsafe |= (~(beta_check >= 8.673617379884035e-19)).to(tl.int32)
                    # --- carried state: bf16 copy for the readout, decay in place ---
                    # The accumulator carries R = S_mid + (beta k_hat)^T U of the
                    # previous chunk, with gates centered at half the chunk total.
                    # This chunk's centered state is exp(Gamma_prev/2) R
                    # exp(Gamma/2). Quartered because a [D, D] fp32 tile is 128
                    # registers per thread on four warps.
                    tlx.barrier_wait(mma_state_full[0], phase ^ 1)
                    # This chunk's per-channel gate total also scales the state.
                    # Read before releasing the compute task: the cumsum tile aliases
                    # the readout accumulator, which its state-read dot overwrites.
                    tlx.barrier_wait(mma_gamma_p[0], phase)
                    if D == 64:
                        # The legacy TMEM lowering cannot load one column from
                        # a 64-row accumulator. Load its full tile and extract
                        # the final cumsum column with a register reduction.
                        gamma_tile = tlx.local_load(mma_gamma[0])
                        gamma_total = tl.sum(
                            tl.where(
                                tl.arange(0, BT)[None, :] == BT - 1,
                                gamma_tile,
                                0.0,
                            ),
                            axis=1,
                        )
                    else:
                        gamma_total = tl.reshape(
                            tlx.local_load(tlx.subslice(mma_gamma[0], BT - 1, 1)),
                            [D],
                        )
                    chunk_shift = tl.exp2(_mul_f32x2(gamma_total, 0.7213475204444817))
                    if D == 128:
                        # Fast-path regime: centered gate exponents stay within
                        # 60 nats when sum_t |g| <= 120. One channel per TMEM lane:
                        # |g_t| = |gamma_t - gamma_(t-1)| from two column slices
                        # one apart, the last pair starting on t = 48 with its
                        # repeated first step masked.
                        cols = tl.arange(0, 16)[None, :]
                        chunk_decay = tl.zeros([D], tl.float32)
                        for j in tl.static_range(4):
                            step = tl.abs(
                                tlx.local_load(
                                    tlx.subslice(mma_gamma[0], 1 + 16 * j - j // 3, 16)
                                )
                                - tlx.local_load(
                                    tlx.subslice(mma_gamma[0], 16 * j - j // 3, 16)
                                )
                            )
                            if j == 3:
                                step = tl.where(cols == 0, 0.0, step)
                            if j == 0:
                                # t = 0 starts from a zero gate total.
                                step += tl.where(
                                    cols == 0,
                                    tl.abs(
                                        tlx.local_load(
                                            tlx.subslice(mma_gamma[0], 0, 16)
                                        )
                                    ),
                                    0.0,
                                )
                            chunk_decay += tl.sum(step, axis=1)
                        decay_unsafe |= (~(chunk_decay <= 120.0)).to(tl.int32)
                    if SAVE_INTERMEDIATE:
                        # The previous chunk's state save still reads b_state_p.
                        tlx.async_descriptor_store_wait(0)
                    if c == 0:
                        tlx.local_store(mma_state[0], tl.zeros([D, D], tl.float32))
                        tlx.local_store(b_state_p, tl.zeros([D, D], dtype))
                    else:
                        for h in tl.static_range(4):
                            # Two multiplies: the combined factor can underflow
                            # FP32 while R and the centered state stay in range.
                            s_h = _mul_f32x2(
                                _mul_f32x2(
                                    tlx.local_load(
                                        tlx.subslice(mma_state[0], h * DQ, DQ)
                                    ),
                                    prev_decay[:, None],
                                ),
                                chunk_shift[:, None],
                            )
                            tlx.local_store(
                                tlx.local_slice(b_state_p, [0, h * DQ], [D, DQ]),
                                s_h.to(dtype),
                            )
                            tlx.local_store(tlx.subslice(mma_state[0], h * DQ, DQ), s_h)
                    prev_decay = chunk_shift
                    tlx.fence_async_shared()
                    tlx.barrier_arrive(state_ready[0], 1)
                    if SAVE_INTERMEDIATE:
                        # Save the centered entering state the readout uses; the
                        # backward consumes it as is. It drains before the next
                        # chunk rewrites b_state_p, so this task moves straight on
                        # to the next chunk's loads.
                        tlx.async_descriptor_store(ss_desc, b_state_p, [c * D, 0])
                    it += 1
                if SAVE_INTERMEDIATE:
                    # The next work item rebuilds this sequence's descriptor.
                    tlx.async_descriptor_store_wait(0)

                if OUTPUT_FINAL_STATE:
                    # mma_state holds R = S_mid + (beta*k_hat)^T @ U after the
                    # last chunk. The next chunk normally applies the other half
                    # of its decay; there is no next chunk here, so materialize
                    # exp(Gamma/2)*R before writing the final FP32 [K,V] state. Read one
                    # V-column quarter at a time to stay within the producer's
                    # register budget; no additional shared memory is needed.
                    state_base = (pid_s * H + pid_h).to(tl.int64) * D * D
                    state_rows = tl.arange(0, D)[:, None]
                    state_cols = tl.arange(0, DQ)[None, :]
                    if num_chunks > 0:
                        tlx.barrier_wait(mma_state_full[0], (it - 1) & 1)
                        for h in tl.static_range(4):
                            s_h = _mul_f32x2(
                                tlx.local_load(tlx.subslice(mma_state[0], h * DQ, DQ)),
                                prev_decay[:, None],
                            )
                            tl.store(
                                FinalS
                                + state_base
                                + state_rows * D
                                + state_cols
                                + h * DQ,
                                s_h,
                            )
                    else:
                        # Jagged inputs may include empty sequences. They start
                        # at zero and have no state-update dot to wait for.
                        for h in tl.static_range(4):
                            tl.store(
                                FinalS
                                + state_base
                                + state_rows * D
                                + state_cols
                                + h * DQ,
                                tl.full((D, DQ), 0.0, tl.float32),
                            )
                tl.store(
                    UnsafeFlags + 2 * (pid_s * H + pid_h),
                    tl.maximum(
                        tl.max(decay_unsafe, axis=0), tl.max(beta_unsafe, axis=0)
                    ),
                )

        # ---- consumer: chunkwise KDA compute ----
        with tlx.async_task("default", registers=COMPUTE_REGISTERS):
            lhs_state_reads = mma_lhs_state_reads[0]
            lhs_wide = tlx.local_slice(lhs_state_reads, [0, 0], [BT, D])
            lhs_tail = tlx.local_slice(lhs_state_reads, [BT, 0], [BT, D])
            b_persist = mma_b_persist[0]
            b_work = mma_b_work[0]
            if D == 64:
                b_work_u = tlx.local_slice(b_work, [0, 0], [BT, D])
                u_acc = tlx.subslice(mma_u[0], 0, D)
            else:
                b_work_u = b_work
                u_acc = mma_u[0]
            aqk_b = mma_aqk_b[0]
            b_work_x = tlx.local_slice(b_work, [0, 0], [BT, BT])
            b_work_t = tlx.local_slice(b_work, [0, BT], [BT, BT])
            b_state = mma_b_state[0]
            state_reads = mma_state_reads[0]
            state_readout_tmem = tlx.subslice(state_reads, 0, BT)
            output_tmem = tlx.subslice(state_reads, BT, BT)
            acc_small1 = tlx.subslice(mma_u[0], 0, BT)
            # The D128 forward computes A^T and Aqk^T together in these two
            # halves; both are free until the solve writes U.
            acc_aqk = tlx.subslice(mma_u[0], BT, BT)
            DH: tl.constexpr = D // 2  # noqa: F841
            offs_t = tl.arange(0, BT)
            lower_strict = offs_t[:, None] > offs_t[None, :]
            lower_incl = offs_t[:, None] >= offs_t[None, :]
            eye = tl.where(  # noqa: F841
                offs_t[:, None] == offs_t[None, :], 1.0, 0.0
            )
            # L[t,r] = (r <= t). Rows past the sequence end need no column mask:
            # the descriptors are bounded by shape=[T, HD], so TMA zero-fills g
            # there and those terms contribute exact zeros.
            tlx.local_store(mma_lmat[0], lower_incl.to(dtype))
            tlx.fence_async_shared()
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                if SORT_WORK:
                    pid_s = tl.load(WorkOrder + pid_s)
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
                # Empty jagged sequences have no producer arrivals. Starting
                # the speculative first dot for them would wait forever on
                # `full` even though the chunk loop below has no work.
                if num_chunks > 0:
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
                    if BETA_IN_PRODUCER:
                        tlx.barrier_wait(beta_ready[bufidx], phase)
                        beta_a = tlx.local_load(beta_buf[bufidx]).to(tl.float32)
                    else:
                        beta_a = tl.load(
                            Beta
                            + (seq_beg + rows) * stride_bt_tok
                            + pid_h * stride_bt_h,
                            mask=row_mask,
                            other=0.0,
                        ).to(tl.float32)
                    if D == 128:
                        beta_a = tlx.require_layout(beta_a, _KDA_BETA_ROW_LAYOUT)
                    beta_column = tl.reshape(beta_a, (BT, 1))
                    if D == 128:
                        beta_tile = tlx.require_layout(
                            tl.broadcast_to(beta_column, (BT, D)),
                            _KDA_GATE_INPUT_LAYOUT,
                        )
                    else:
                        beta_tile = beta_column
                    k_a = tlx.local_load(k_buf[bufidx])
                    if not full_chunk:
                        k_a = tl.where(row_mask[:, None], k_a, tl.zeros_like(k_a))
                    tlx.barrier_wait(mma_gamma_full[0], phase)
                    tlx.barrier_wait(full_qv[bufidx], phase)
                    q = tlx.local_load(q_buf[bufidx])
                    if D == 128:
                        v = tlx.local_load(v_buf[bufidx], layout=_KDA_DOT_LAYOUT)
                    else:
                        v = tlx.local_load(v_buf[bufidx], layout=_KDA_DOT_LAYOUT_D64)
                    tlx.barrier_arrive(empty[bufidx], 1)
                    if not full_chunk:
                        q = tl.where(row_mask[:, None], q, tl.zeros_like(q))
                        v = tl.where(row_mask[:, None], v, 0.0)
                    gamma_values = tlx.local_load(mma_gamma[0])
                    gamma_center = 0.5 * tl.sum(
                        tl.where(offs_t[None, :] == BT - 1, gamma_values, 0.0), axis=1
                    )
                    gamma_a = _mul_f32x2(
                        tl.trans(gamma_values) - gamma_center[None, :],
                        1.4426950408889634,
                    )
                    k_hat_a = (k_a * tl.exp2(-gamma_a).to(dtype)).to(dtype)
                    exp_gamma = tl.exp2(gamma_a).to(dtype)
                    k_bar_a = (k_a * exp_gamma).to(dtype)
                    beta_k_hat_a = (beta_tile * k_hat_a).to(dtype)

                    # --- intra matrix A (strictly lower-tri) ---
                    # A_{t,r} = <beta_r*k_hat_r, k_bar_t>  for r<t   [C,C]
                    tlx.local_store(lhs_wide, k_bar_a)
                    tlx.local_store(b_persist, beta_k_hat_a)
                    tlx.fence_async_shared()
                    if not FUSE_A_AQK:
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
                    if FUSE_A_AQK:
                        # Concatenate k_bar and q_bar in the shared readout
                        # tile. A single K=128 dot writes A^T and Aqk^T to
                        # the two accumulator halves without changing either
                        # matrix's reduction order or rounded BF16 operands.
                        tlx.fence_async_shared()
                        tlx.async_dot(
                            b_persist,
                            tlx.local_trans(lhs_state_reads),
                            mma_u[0],
                            use_acc=False,
                            force_async=True,
                            mBarriers=[mma_small0_full[0], mma_aqk_full[0]],
                        )

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
                    if not FUSE_A_AQK:
                        tlx.async_dot(
                            lhs_tail,
                            tlx.local_trans(b_persist),
                            acc_aqk,
                            use_acc=False,
                            force_async=True,
                            mBarriers=[mma_aqk_full[0]],
                        )
                    tlx.barrier_wait(mma_aqk_full[0], phase)
                    if FUSE_A_AQK:
                        Aqk = tl.trans(tlx.local_load(acc_aqk))
                    else:
                        Aqk = tlx.local_load(acc_aqk)
                    Aqk = tl.where(lower_incl, Aqk, 0.0).to(dtype)
                    if SAVE_INTERMEDIATE:
                        tlx.barrier_wait(save_t_empty[0], phase ^ 1)
                    tlx.local_store(aqk_b, Aqk)
                    tlx.fence_async_shared()
                    tlx.barrier_wait(tinv_ready[0], phase)
                    # U = T @ rhs   [C,D]  (solve the intra-chunk system)
                    tlx.async_dot(
                        mma_lhs_small[0],
                        lhs_wide,
                        u_acc,
                        use_acc=False,
                        force_async=True,
                        mBarriers=[mma_wide_full[0]],
                    )
                    tlx.barrier_wait(mma_wide_full[0], 1)
                    U = tlx.local_load(u_acc)
                    # Accumulate the transposed output so it matches the merged
                    # state's N-sliced TMEM result: U^T @ Aqk^T.
                    tlx.local_store(b_work_u, U.to(dtype))
                    tlx.fence_async_shared()
                    tlx.async_dot(
                        tlx.local_trans(b_work_u),
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
                        b_work_u,
                        mma_state[0],
                        use_acc=True,
                        force_async=True,
                        mBarriers=[mma_state_full[0]],
                    )
                    if SAVE_BETA_U:
                        # The two dots have been issued. Stage the forward's
                        # rounded beta*U for the backward without extending
                        # the state recurrence or waiting on the TMA store.
                        tlx.barrier_wait(beta_u_empty[0], phase ^ 1)
                        tlx.local_store(
                            beta_u_buf[0], U.to(dtype) * beta_column.to(dtype)
                        )
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(beta_u_full[0], 1)
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
        with tlx.async_task(num_warps=4, registers=EPILOGUE_REGISTERS):
            out_tmem = tlx.subslice(mma_state_reads[0], BT, BT)
            lhs_small = mma_lhs_small[0]
            save_t_buf = mma_aqk_b[0]
            b_work_x = tlx.local_slice(mma_b_work[0], [0, 0], [BT, BT])
            b_work_t = tlx.local_slice(mma_b_work[0], [0, BT], [BT, BT])
            acc_small1 = tlx.subslice(mma_u[0], 0, BT)
            offs_t = tl.arange(0, BT)
            lower_strict = offs_t[:, None] > offs_t[None, :]
            eye_b = tl.where(offs_t[:, None] == offs_t[None, :], 1.0, 0.0).to(dtype)
            BTH: tl.constexpr = BT // 2
            # Fourth-factor barriers fire only on chunks that need that factor,
            # so their wait parity follows this count instead of the chunk.
            extra_steps = 0
            it = 0
            for wid in tl.range(pid, n_work, nprog):
                pid_s = wid // H
                if SORT_WORK:
                    pid_s = tl.load(WorkOrder + pid_s)
                pid_h = wid % H
                seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
                seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
                T = seq_end - seq_beg
                num_chunks = tl.cdiv(T, BT)
                sob = seq_beg.to(tl.int64) * HD
                if PACKED_SCRATCH:
                    chunk_prefix = tl.load(WorkOrder + n_seq + pid_s)
                    st_base = (chunk_prefix * H + pid_h * num_chunks).to(tl.int64)
                else:
                    st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
                o_desc = tl.make_tensor_descriptor(
                    O + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
                )
                if SAVE_BETA_U:
                    beta_u_desc = tl.make_tensor_descriptor(
                        SavedBetaU + sob,
                        shape=[T, HD],
                        strides=[HD, 1],
                        block_shape=[BT, D],
                    )
                if SAVE_INTERMEDIATE:
                    st_desc = tl.make_tensor_descriptor(
                        SaveT + st_base * BT * BT,
                        shape=[num_chunks * BT, BT],
                        strides=[BT, 1],
                        block_shape=[BT, BT],
                    )
                col = (pid_h * D).to(tl.int32)
                inverse_unsafe = tl.zeros([BT], tl.int32)
                for c in tl.range(0, num_chunks):
                    bufidx, phase = get_bufidx_phase(it, NBUF)
                    # --- WY triangular inverse  T = (I + A)^{-1}  [C,C] ---
                    tlx.barrier_wait(mma_small0_full[0], phase)
                    if FUSE_A_AQK:
                        A = tl.trans(tlx.local_load(acc_small1))
                    else:
                        A = tlx.local_load(mma_small0[0])
                    A = tl.where(lower_strict, A, 0.0)
                    X = (-A).to(dtype)
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
                    # (I + A)^-1 = prod_k (I + X^(2^k)) truncates to the powers
                    # below 2^K after K factors; the omitted tail is
                    # rho^(2^K) / (1 - rho) times the solve's input, rho =
                    # ||A||_inf. ||A||_inf <= 1/2 bounds ||(I + A)^-1||_inf by 2
                    # and is the fast-path condition (any non-finite row fails
                    # it too). Three factors keep the tail below 2.1e-5 when
                    # rho <= 1/4; a fourth keeps it below 3.1e-5 up to 1/2.
                    row_sum = tl.sum(tl.abs(A), axis=1)
                    inverse_unsafe |= (~(row_sum <= 0.5)).to(tl.int32)
                    extra_phase = extra_steps & 1
                    tlx.barrier_wait(mma_inv_b[0], phase)
                    Xn = tlx.local_load(acc_small1).to(dtype)
                    # Factors (I + X^2) and (I + X^4); X^4 and X^8 alongside.
                    for i in tl.static_range(1, 3):
                        if i == 1:
                            # The four warps must agree on the fourth factor. A
                            # generic max adds three barriers per chunk; here
                            # each row posts its warp's max, and the barrier
                            # ahead of this step's dots publishes the posts.
                            tlx.local_store(
                                row_max[0],
                                tlx.warp_redux(
                                    tlx.require_layout(row_sum, _KDA_TILE_ROW_LAYOUT),
                                    "max",
                                ),
                            )
                        tlx.local_store(lhs_small, eye_b + Xn)
                        tlx.local_store(b_work_t, Tinv)
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
                        if i == 1:
                            tlx.async_dot(
                                b_work_x,
                                b_work_x,
                                acc_small1,
                                use_acc=False,
                                force_async=True,
                                mBarriers=[mma_inv_b[i]],
                            )
                            extra = (
                                tl.max(
                                    tlx.local_load(
                                        row_max[0], layout=_KDA_WARP_ROWS_LAYOUT
                                    ),
                                    axis=0,
                                )
                                > 0.25
                            )
                        else:
                            if extra:
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
                        if i == 1:
                            tlx.barrier_wait(mma_inv_b[i], phase)
                            Xn = tlx.local_load(acc_small1).to(dtype)
                    if extra:
                        # Factor (I + X^8).
                        tlx.barrier_wait(mma_inv_b[2], extra_phase)
                        Xn = tlx.local_load(acc_small1).to(dtype)
                        tlx.local_store(lhs_small, eye_b + Xn)
                        tlx.local_store(b_work_t, Tinv)
                        tlx.fence_async_shared()
                        tlx.async_dot(
                            lhs_small,
                            b_work_t,
                            mma_small0[0],
                            use_acc=False,
                            force_async=True,
                            mBarriers=[mma_inv_a[3]],
                        )
                        tlx.barrier_wait(mma_inv_a[3], extra_phase)
                        Tinv = tlx.local_load(mma_small0[0]).to(dtype)
                        extra_steps += 1
                    tlx.local_store(lhs_small, Tinv)
                    tlx.fence_async_shared()
                    tlx.barrier_arrive(tinv_ready[0], 1)

                    # --- drain the output accumulator and TMA-store it ---
                    # In [D, C/2] halves: the whole fp32 tile is 64 registers per
                    # thread on this 4-warp task, which spilled the WY chain.
                    tlx.barrier_wait(mma_out_full[0], phase)
                    if SAVE_INTERMEDIATE:
                        # The output dot has finished reading Aqk. Reuse its
                        # shared-memory tile as the SaveT source, and keep the
                        # next chunk behind o_empty until both stores are issued.
                        tlx.local_store(save_t_buf, tlx.local_load(lhs_small).to(dtype))
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(st_desc, save_t_buf, [c * BT, 0])
                    for j in tl.static_range(2):
                        o_h = tl.trans(
                            tlx.local_load(tlx.subslice(out_tmem, j * BTH, BTH))
                        )
                        tlx.local_store(
                            tlx.local_slice(o_buf[bufidx], [j * BTH, 0], [BTH, D]),
                            o_h.to(dtype),
                        )
                    tlx.fence_async_shared()
                    row = (c * BT).to(tl.int32)  # relative to this sequence's base
                    tlx.async_descriptor_store(o_desc, o_buf[bufidx], [row, col])
                    tlx.barrier_arrive(o_empty[bufidx], 1)
                    if SAVE_BETA_U:
                        tlx.barrier_wait(beta_u_full[0], phase)
                        tlx.async_descriptor_store(
                            beta_u_desc, beta_u_buf[0], [row, col]
                        )
                    tlx.async_descriptor_store_wait(0)
                    if SAVE_INTERMEDIATE:
                        tlx.barrier_arrive(save_t_empty[0], 1)
                    if SAVE_BETA_U:
                        tlx.barrier_arrive(beta_u_empty[0], 1)
                    it += 1
                tl.store(
                    UnsafeFlags + 2 * (pid_s * H + pid_h) + 1,
                    tl.max(inverse_unsafe, axis=0),
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
    SavedBetaU,
    DFinalS,
    cu_seqlens,
    WorkOrder,
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
    HAS_SAVED_BETA_U: tl.constexpr,
    PACKED_SCRATCH: tl.constexpr,
    HAS_FINAL_STATE_GRAD: tl.constexpr,
    SORT_WORK: tl.constexpr,
    GRAD_G_BF16: tl.constexpr,
    UNIT_SCALE_EGS: tl.constexpr,
    STAGE_EGB: tl.constexpr,
    COMPACT_D64: tl.constexpr,
    STAGE_DG_IN_S: tl.constexpr,
    SPLIT_DG_EPILOGUE: tl.constexpr,
    GRAD_STRIDE: tl.constexpr,
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
    # Persistent kernel: grid-striding over the (seq, head) work-items. No
    # separate producer warp: at NBUF=1 its transfers did not overlap the next
    # chunk, so all registers can serve the 8 compute warps (4 in compact D64).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_work = n_seq * H
    HD: tl.constexpr = H * D
    BT2: tl.constexpr = 2 * BT
    L2E: tl.constexpr = 1.4426950408889634  # log2(e): rebase gate exps to ex2
    dtype = Q.dtype.element_ty

    # --- SMEM (D128 ~225KB, compact D64 ~115KB at BT=64) -------------------
    # qk_buf: TMA lands q in rows [0,C) and k in rows [C,2C); the gate scaling
    # rewrites both halves in place (each thread stores back exactly the elements
    # it loaded), turning the tile into [q_bar; k_bar] with no second allocation.
    qk_buf = tlx.local_alloc((BT2, D), dtype, NBUF)
    r_buf = tlx.local_alloc((BT2, D), dtype, NBUF)  # do (TMA) | dKS
    s_buf_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    s_buf = tlx.local_alloc(
        (D, D), dtype, NBUF, reuse=s_buf_storage_alias
    )  # S_in (TMA)
    g_buf_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    g_buf = tlx.local_alloc((BT, D), dtype, NBUF, reuse=g_buf_storage_alias)  # g (TMA)
    # k_hat overwrites g: g's only reader is the cumsum MMA, so the MMA's
    # completion barrier already orders the rewrite.
    kh_buf = tlx.local_alloc((BT, D), dtype, NBUF, reuse=g_buf_storage_alias)
    g_buf_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            g_buf,
            kh_buf,
            group_type=tlx.reuse_group_type.shared,
        )
    )
    wa_buf = tlx.local_alloc((BT, D), dtype, NBUF)  # v (TMA) -> betaU -> dU -> dgamma
    # Decayed dS, bf16 MMA operand; once its two dots retire, its space is
    # recycled for dv and dk. D128 also uses its second half for dU and dq.
    # D64 can use the unused upper half of r_buf for dU/dq. Recycle the state
    # tile for dv after the state dots retire, then for dk after dv is stored.
    # This saves one [C,D] shared tile, allowing two 4-warp CTAs per SM.
    if COMPACT_D64:
        sd_buf = tlx.local_alloc((BT, D), dtype, 1)
    else:
        sd_buf = tlx.local_alloc((BT2, D), dtype, 1)
    p_buf = tlx.local_alloc((BT2, BT), dtype, 1)  # Aqk -> dAqk | dA0
    # Alone among the TMA destinations this one is 2-deep: at 8KB it is the only
    # buffer the 232KB budget can double, and doing so drops the WAR edge from
    # the next chunk's Tinv load to this chunk's triangular solve. D64 stages
    # Zk in the idle half.
    t_buf = tlx.local_alloc((BT, BT), dtype, 2)
    lm_buf_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)
    lm_buf = tlx.local_alloc(
        (BT, BT), dtype, 1, reuse=lm_buf_storage_alias
    )  # lower-inclusive ones, hoisted
    # dq/dk/dv/dg are TMA-stored. Gradient rows are H*D elements -- 1024B in
    # production and 768B at H=3, both 16B-aligned -- so the descriptor is
    # legal on every shape in the suite. This deletes the [C,D] int64 offset
    # tile (32 of the 128 registers per thread, held across the whole
    # epilogue), five masked global stores and their convergences; TMA's own
    # bounds clipping also replaces the row mask on the partial chunk.
    dg_buf = tlx.local_alloc((BT, D), tl.float32, 1, reuse=s_buf_storage_alias)
    s_buf_storage_alias.set_buffer_overlap(tlx.reuse_group(s_buf, dg_buf))
    # D128 stages Zk separately so the gate branch can be prefetched after
    # group F; D64 recycles the idle Tinv slot once its last dot has retired.
    if not COMPACT_D64:
        zk_buf = tlx.local_alloc((BT, D), dtype, 1)
    # A dS_end seed adds layout-conversion scratch to the backward, and D=128
    # otherwise runs out of SMEM. Only for that specialization, share the two
    # constant tiles: the cumsum consumes L before the dbeta reductions need
    # ones, and the epilogue can switch back to L after those dots retire.
    if HAS_FINAL_STATE_GRAD or COMPACT_D64:
        one_buf = tlx.local_alloc((BT, BT), dtype, 1, reuse=lm_buf_storage_alias)
        lm_buf_storage_alias.set_buffer_overlap(tlx.reuse_group(lm_buf, one_buf))
    else:
        one_buf = tlx.local_alloc((BT, BT), dtype, 1)
        lm_buf_storage_alias.set_buffer_overlap(tlx.reuse_group(lm_buf))

    # --- TMEM (512 cols: 128 for the carried dS + 3 reused [C,D] accumulators) ---
    acc_ds = tlx.local_alloc((D, D), tl.float32, 1, tlx.storage_kind.tmem)
    acc_a = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    acc_b = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    # These sequential accumulators need [C,C] and [2C,C] views over the same
    # storage. Let TLX derive a [2C,C] backing allocation from all peer views,
    # instead of treating the first, smaller view as the physical owner.
    acc_c_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    acc_c = tlx.local_alloc(
        (BT, BT),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
        reuse=acc_c_alias,
    )
    # dk_hat needs a target of its own: its two K halves become ready a chunk
    # stage apart, and acc_c_lo is rewritten by the dA dot in between.
    acc_kh = tlx.local_alloc((BT, D), tl.float32, 1, tlx.storage_kind.tmem)
    # Transposed cumsum lands here so exp(Gamma) can be read out as a TMEM
    # column instead of an axis-0 reduction. It shares acc_c with the A-matrix
    # accumulator, whose first write is a chunk stage later.
    acc_gt = tlx.local_alloc(
        (D, BT),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
        reuse=acc_c_alias,
    )

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
    lower_incl = offs_t[:, None] >= offs_t[None, :]
    lower_strict = offs_t[:, None] > offs_t[None, :]
    # L[t,r] = (r <= t): the cumsum operand, and its transpose is the
    # reverse-cumsum operand. Rows past the sequence end need no column
    # mask because TMA zero-fills g there.
    tlx.local_store(lm_buf[0], lower_incl.to(dtype))
    if not HAS_FINAL_STATE_GRAD:
        tlx.local_store(one_buf[0], tl.full([BT, BT], 1.0, dtype))
    z_view = tlx.local_slice(sd_buf[0], [0, 0], [BT, D])
    if COMPACT_D64:
        du_view = tlx.local_slice(r_buf[0], [BT, 0], [BT, D])
        dv_view = z_view
    else:
        du_view = tlx.local_slice(sd_buf[0], [BT, 0], [BT, D])
        dv_view = du_view
    # D128 uses the entire state tile; D64 consumes only its first C rows.
    if D == 64:
        state_view = z_view
    else:
        state_view = sd_buf[0]
    p_lo = tlx.local_slice(p_buf[0], [0, 0], [BT, BT])
    p_hi = tlx.local_slice(p_buf[0], [BT, 0], [BT, BT])
    acc_aa = tlx.local_alloc(
        (BT2, BT),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
        reuse=acc_c_alias,
    )
    acc_c_alias.set_buffer_overlap(tlx.reuse_group(acc_c, acc_gt, acc_aa))
    acc_c_lo = acc_c[0]

    it = 0
    for wid in tl.range(pid, n_work, nprog):
        if HAS_FINAL_STATE_GRAD or COMPACT_D64:
            # The previous work item finishes with ones in the shared tile.
            tlx.local_store(lm_buf[0], lower_incl.to(dtype))
            tlx.fence_async_shared()
        pid_s = wid // H
        if SORT_WORK:
            pid_s = tl.load(WorkOrder + pid_s)
        pid_h = wid % H
        seq_beg = tl.load(cu_seqlens + pid_s).to(tl.int32)
        seq_end = tl.load(cu_seqlens + pid_s + 1).to(tl.int32)
        T = seq_end - seq_beg
        num_chunks = tl.cdiv(T, BT)
        if PACKED_SCRATCH:
            chunk_prefix = tl.load(WorkOrder + n_seq + pid_s)
            st_base = (chunk_prefix * H + pid_h * num_chunks).to(tl.int64)
        else:
            st_base = (pid_s * H + pid_h).to(tl.int64) * max_chunks
        sob = seq_beg.to(tl.int64) * HD
        grad_sob = seq_beg.to(tl.int64) * GRAD_STRIDE
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
        q_desc = tl.make_tensor_descriptor(
            Q + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
        )
        k_desc = tl.make_tensor_descriptor(
            K + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
        )
        if HAS_SAVED_BETA_U:
            beta_u_desc = tl.make_tensor_descriptor(
                SavedBetaU + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
            )
        else:
            v_desc = tl.make_tensor_descriptor(
                V + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
            )
        do_desc = tl.make_tensor_descriptor(
            DO + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
        )
        g_desc = tl.make_tensor_descriptor(
            G + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
        )
        dq_desc = tl.make_tensor_descriptor(
            DQ + grad_sob,
            shape=[T, HD],
            strides=[GRAD_STRIDE, 1],
            block_shape=[BT, D],
        )
        dk_desc = tl.make_tensor_descriptor(
            DK + grad_sob,
            shape=[T, HD],
            strides=[GRAD_STRIDE, 1],
            block_shape=[BT, D],
        )
        dv_desc = tl.make_tensor_descriptor(
            DV + grad_sob,
            shape=[T, HD],
            strides=[GRAD_STRIDE, 1],
            block_shape=[BT, D],
        )
        dg_desc = tl.make_tensor_descriptor(
            DG + sob, shape=[T, HD], strides=[HD, 1], block_shape=[BT, D]
        )
        col = pid_h * D
        if HAS_FINAL_STATE_GRAD:
            # The reverse VJP starts at dS_end. The first chunk in the reverse
            # scan applies its gate decay to this seed in the existing path.
            # Stage one K-row strip in the existing FP32 dg/s_buf allocation;
            # adding a separate layout-conversion scratch exceeds the D=128
            # backward's shared-memory budget.
            state_base = (pid_s * H + pid_h).to(tl.int64) * D * D
            state_rows = tl.arange(0, D)[:, None]
            state_cols = tl.arange(0, BT)[None, :]
            for h in tl.static_range(D // BT):
                initial_ds = tl.load(
                    DFinalS + state_base + state_rows * D + h * BT + state_cols,
                    mask=(state_rows < D) & (state_cols < BT),
                )
                tlx.local_store(dg_buf[0], tl.trans(initial_ds))
                tlx.fence_async_shared()
                tlx.local_store(
                    tlx.subslice(acc_ds[0], h * BT, BT),
                    tl.trans(tlx.local_load(dg_buf[0])),
                )
        else:
            tlx.local_store(acc_ds[0], tl.zeros([D, D], dtype=tl.float32))

        previous_shift = tl.full([D], 1.0, tl.float32)
        for ci in tl.range(0, num_chunks):
            c = num_chunks - 1 - ci
            bufidx, phase = get_bufidx_phase(it, NBUF)
            tidx = it % 2
            if COMPACT_D64:
                # The next chunk's Tinv lands here only after mb[8] retires
                # the dGamma dot that reads Zk.
                zk_view = t_buf[1 - tidx]
            else:
                zk_view = zk_buf[0]
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
                tlx.async_descriptor_load(
                    g_desc, g_buf[bufidx], [grow, col], full[bufidx]
                )
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
            if HAS_SAVED_BETA_U:
                tlx.async_descriptor_load(
                    beta_u_desc, wa_buf[bufidx], [row, col], full_s[bufidx]
                )
            else:
                tlx.async_descriptor_load(
                    v_desc, wa_buf[bufidx], [row, col], full_s[bufidx]
                )
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
            # Only the bf16 gates stay live to the epilogue: k_bar/k_hat/q_bar
            # are bf16 anyway, so nothing downstream can see more than 8 mantissa
            # bits of e^gamma, and the fp32 copies were what tipped the register
            # allocator into local memory at 16 warps.
            if D == 64:
                gamma_last = tl.sum(
                    tl.where(
                        tl.arange(0, BT)[None, :] == BT - 1,
                        tlx.local_load(acc_gt[0]),
                        0.0,
                    ),
                    axis=1,
                )
            else:
                gamma_last = tl.reshape(
                    tlx.local_load(tlx.subslice(acc_gt[0], BT - 1, 1)), [D]
                )
            gam -= 0.5 * gamma_last[None, :]
            eg = tl.exp2(_mul_f32x2(gam, L2E))
            egb = eg.to(dtype)
            engb = tl.exp2(_mul_f32x2(gam, -L2E)).to(dtype)
            chunk_shift = tl.exp2(_mul_f32x2(gamma_last, 0.7213475204444817))
            # Two half-chunk decays: their product underflows FP32 for strong
            # gates while the decayed dS itself stays representable.
            next_shift = previous_shift
            previous_shift = chunk_shift
            # k_hat=k*e^-gamma  k_bar=k*e^gamma  q_bar=scale*q*e^gamma.
            # The raw q/k are only needed to form the scaled MMA operands.
            # Keep the backward's long dependency chain free of these two
            # live register tiles; the epilogue uses the scaled SMEM values.
            q_raw = tlx.local_load(qk_lo)
            k_raw = tlx.local_load(qk_hi)
            # scale rides the gate rather than the epilogue: the same tile
            # builds q_bar here and unscales dq_bar at the end.
            # At unit query scale, the BF16 query gate is exactly the already
            # rounded key gate. Sharing it removes one persistent register tile
            # from the long reverse recurrence without changing its values.
            if UNIT_SCALE_EGS:
                egs = egb
            else:
                egs = _mul_f32x2(eg, scale).to(dtype)
            tlx.local_store(qk_lo, q_raw * egs)
            tlx.local_store(qk_hi, k_raw * egb)
            tlx.local_store(kh_buf[bufidx], k_raw * engb)
            if STAGE_EGB:
                egb_even, egb_odd = tl.split(tl.reshape(egb, (BT, D // 2, 2)))
                # L is no longer an MMA operand after mb[0] retires. Keep
                # half the rounded gate here until the gradient epilogue.
                tlx.local_store(lm_buf[0], egb_odd)

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
                ds_s = _mul_f32x2(
                    _mul_f32x2(tlx.local_load(strip), chunk_shift[:, None]),
                    next_shift[:, None],
                )
                tlx.local_store(strip, ds_s)
                tlx.local_store(
                    tlx.local_slice(sd_buf[0], [0, hs * BT], [D, BT]),
                    ds_s.to(dtype),
                )

            tlx.barrier_wait(full_s[bufidx], phase)
            if not HAS_SAVED_BETA_U:
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
            # Form the triangular A-mask only when it is used. The equal
            # per-chunk offset keeps this comparison equivalent while avoiding
            # a long-lived [2C,C] mask in the persistent scan. Masking the
            # chunk counter also prevents overflow in very long scans.
            chunk_base = (it & 0x3FF) * BT2
            aa_row = tl.arange(0, BT2) + chunk_base
            aa_col = tl.arange(0, BT) + chunk_base
            aa_keep = aa_row[:, None] >= aa_col[None, :]
            tlx.local_store(
                p_buf[0],
                tl.where(aa_keep, tlx.local_load(acc_aa[0]).to(dtype), 0.0),
            )
            if not HAS_SAVED_BETA_U:
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
                    (
                        tlx.local_load(
                            tlx.local_slice(sd_buf[0], [0, hs * BT], [D, BT])
                        )
                        * tlx.local_load(
                            tlx.local_slice(s_buf[bufidx], [0, hs * BT], [D, BT])
                        )
                    ).to(tl.float32),
                    axis=1,
                )
            if not HAS_SAVED_BETA_U:
                tlx.barrier_wait(mb[2], phase)
            bb = b.to(dtype)
            if not HAS_SAVED_BETA_U:
                tlx.local_store(
                    wa_buf[bufidx], tlx.local_load(acc_b[0]).to(dtype) * bb[:, None]
                )

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
                tlx.local_trans(state_view),
                acc_a[0],
                use_acc=False,
                force_async=True,
                mBarriers=[mb[10]],
            )
            # The same product seeds dk_hat's accumulator, so the epilogue reads
            # dk_hat + Z directly instead of carrying Z in FP32 registers.
            tlx.async_dot(
                wa_buf[bufidx],
                tlx.local_trans(state_view),
                acc_kh[0],
                use_acc=False,
                force_async=True,
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
                state_view,
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
            # zk = Z * k_hat feeds dGamma's ones-dot; staging it now ends Z's
            # register lifetime here.
            tlx.local_store(
                zk_view,
                (
                    tlx.local_load(acc_a[0])
                    * tlx.local_load(kh_buf[bufidx]).to(tl.float32)
                ).to(dtype),
            )
            # dAqk takes the A0 half: nothing reads A0 here, while the dbetaU dot
            # may still be reading Aqk from p_lo.
            tlx.local_store(
                p_hi,
                tl.where(lower_incl, tlx.local_load(acc_c_lo).to(dtype), 0.0),
            )
            # dk_hat = Z + dAqk^T@q_bar + dA0^T@k_bar, K-split at the 2C boundary:
            # dAqk is ready here but dA0 not until after dA retires, and the sum
            # is not read until the epilogue.
            tlx.fence_async_shared()
            tlx.async_dot(
                tlx.local_trans(p_hi),
                qk_lo,
                acc_kh[0],
                use_acc=True,
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
            drhs = tlx.local_load(acc_b[0])
            dvb = drhs.to(dtype)
            tlx.local_store(dv_view, dvb)  # dU was drained by the drhs dot
            tlx.local_store(r_hi, -dvb)  # dKS
            tlx.fence_async_shared()
            tlx.async_descriptor_store(dv_desc, dv_view, [row, col])

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
            # dbeta = sum_d(betaU * drhs) / beta^2, reduced while group F runs.
            # wa_buf still holds betaU until dgamma replaces it.
            safe_beta = tl.where(bb > 0.0, bb, 1.0).to(tl.float32)
            dbeta = tl.sum(
                tlx.local_load(wa_buf[bufidx]).to(tl.float32) * drhs, axis=1
            ) / (safe_beta * safe_beta)
            tl.store(
                DBeta + tok * stride_bt_tok + pid_h * stride_bt_h, dbeta, mask=rmask
            )
            tlx.barrier_wait(mb[5], phase)
            # dA0 is dA*beta and dA is dA_raw/beta, so the two cancel: the
            # masked accumulator is already dA0. It takes the Aqk half, whose
            # dbetaU reader retired before mb[3].
            damb = tl.where(lower_strict, tlx.local_load(acc_c_lo).to(dtype), 0.0)
            tlx.local_store(p_lo, damb)

            # === group G: A-matrix halves, dk_hat, and the dS handoff ===
            tlx.fence_async_shared()
            # dk_hat is issued first because the epilogue consumes it first,
            # while the two A-matrix halves are still running.
            tlx.async_dot(  # ... + the dA0 half, which only lands now
                tlx.local_trans(p_lo),
                qk_hi,
                acc_kh[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[11]],
            )
            tlx.async_dot(  # dk_bar += dA0 @ k_hat
                p_lo,
                kh_buf[bufidx],
                acc_b[0],
                use_acc=True,
                force_async=True,
            )
            tlx.async_dot(  # dq_bar += dAqk @ k_hat
                p_hi,
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
            # Form dgamma from the scaled q_bar/k_bar/k_hat tiles while they
            # still reside in SMEM. This ends raw q/k's live ranges at the
            # beginning of the chunk and leaves no global reloads in the tail.
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
            # dgamma's three terms cancel and dk sums two gate products, so
            # both are formed in FP32; dq is a single gate product. Each
            # accumulator is consumed as soon as it is read to bound the live
            # FP32 tiles.
            dk_hat_total = tlx.local_load(acc_kh[0])
            dgamma = -(dk_hat_total * tlx.local_load(kh_buf[bufidx]).to(tl.float32))
            dk_out = dk_hat_total * engb.to(tl.float32)
            if STAGE_EGB:
                egb_grad = tl.interleave(egb_even, tlx.local_load(lm_buf[0]))
            else:
                egb_grad = egb
            tlx.barrier_wait(mb[6], phase)
            dk_bar = tlx.local_load(acc_b[0])
            dgamma += dk_bar * tlx.local_load(qk_hi).to(tl.float32)
            tlx.local_store(
                z_view, (dk_out + dk_bar * egb_grad.to(tl.float32)).to(dtype)
            )
            dq_bar = tlx.local_load(acc_a[0])
            dgamma += dq_bar * tlx.local_load(qk_lo).to(tl.float32)
            tlx.local_store(du_view, dq_bar.to(dtype) * egs)
            tlx.local_store(wa_buf[bufidx], dgamma.to(dtype))
            if ci + 1 < num_chunks:
                # q_bar/k_bar and k_hat have no readers left, so the successor's
                # gate branch -- 49KB, otherwise fully exposed at the top of the
                # next chunk -- transfers under the rest of this epilogue.
                grow = row - BT
                tlx.barrier_expect_bytes(full[bufidx], GATE_BYTES)
                tlx.async_descriptor_load(
                    g_desc, g_buf[bufidx], [grow, col], full[bufidx]
                )
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
            if HAS_FINAL_STATE_GRAD or STAGE_EGB or COMPACT_D64:
                # Group G's last dot (mb[6]) has finished using the ones tile.
                tlx.local_store(lm_buf[0], lower_incl.to(dtype))
            tlx.fence_async_shared()
            # The scan reads only dgamma, so it goes out before the other two
            # operands are staged; the accumulate order into acc_a is preserved
            # because a CTA's dots retire in issue order.
            if HAS_FINAL_STATE_GRAD or COMPACT_D64:
                tlx.async_dot(
                    tlx.local_trans(lm_buf[0]),
                    wa_buf[bufidx],
                    acc_a[0],
                    use_acc=False,
                    force_async=True,
                    mBarriers=[mb[12]],
                )
                # The scan must finish reading L before this shared tile can
                # become the all-ones operand for dGamma's dot.
                tlx.barrier_wait(mb[12], phase)
                tlx.local_store(one_buf[0], tl.full([BT, BT], 1.0, dtype))
            else:
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
                zk_view,
                acc_a[0],
                use_acc=True,
                force_async=True,
                mBarriers=[mb[8]],
            )
            # r1 is reduced out of the [D,C] state strips, so it is indexed
            # along dim 0 while dg wants it along dim 1; bridging that is a
            # convert_layout. Run here it sits in the shadow of the dg dots
            # rather than in front of them.
            if not SPLIT_DG_EPILOGUE:
                r1t = tl.zeros([BT, D], dtype=tl.float32) + r1[None, :]
            tlx.barrier_wait(mb[8], phase)
            # add.f32x2: dg is the one [C,D] fp32 elementwise pass left, and
            # halving its instruction count is worth 1% at production clocks.
            if GRAD_G_BF16:
                # The original FP32 dG is immediately cast to BF16 by
                # autograd when the decay op produced BF16. The state dots
                # have retired, so reuse S_in for the rounded dG tile. The
                # next beta*U load then has a separate SMEM source; S_in is
                # reloaded after the store-wait in the next chunk.
                if STAGE_DG_IN_S:
                    dg_source = tlx.local_slice(s_buf[bufidx], [0, 0], [BT, D])
                else:
                    dg_source = wa_buf[bufidx]
                if SPLIT_DG_EPILOGUE:
                    for half in tl.static_range(2):
                        dg_half = tlx.local_slice(
                            dg_source, [0, half * (D // 2)], [BT, D // 2]
                        )
                        r1_half = tl.gather(
                            r1, half * (D // 2) + tl.arange(0, D // 2), 0
                        )
                        r1_tile = (
                            tl.zeros([BT, D // 2], dtype=tl.float32) + r1_half[None, :]
                        )
                        tlx.local_store(
                            dg_half,
                            _add_f32x2(
                                tlx.local_load(
                                    tlx.local_slice(
                                        acc_a[0], [0, half * (D // 2)], [BT, D // 2]
                                    )
                                ),
                                r1_tile,
                            ).to(dtype),
                        )
                else:
                    tlx.local_store(
                        dg_source,
                        _add_f32x2(tlx.local_load(acc_a[0]), r1t).to(dtype),
                    )
                tlx.fence_async_shared()
                tlx.async_descriptor_store(dg_desc, dg_source, [row, col])
            else:
                tlx.local_store(dg_buf[0], _add_f32x2(tlx.local_load(acc_a[0]), r1t))
                tlx.fence_async_shared()
                tlx.async_descriptor_store(dg_desc, dg_buf[0], [row, col])
            # The successor's cumsum pair is issued here, once acc_a and acc_gt
            # have been drained, so the next chunk's gate branch starts on a
            # result that is already in flight.
            if ci + 1 < num_chunks:
                if HAS_FINAL_STATE_GRAD or COMPACT_D64:
                    # mb[8] retired dGamma's ones dot. Seed the successor's
                    # eagerly issued cumsum with L, including its transpose.
                    tlx.local_store(lm_buf[0], lower_incl.to(dtype))
                    tlx.fence_async_shared()
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


def _sequence_work_order(
    cpu_offsets: Optional[torch.Tensor],
    heads: int,
    device: torch.device,
    *,
    pack_chunk_offsets: bool = False,
) -> Optional[torch.Tensor]:
    """Sort jagged work and optionally append per-sequence chunk offsets.

    The first ``n_seq`` entries keep the existing scheduling order. Packed
    training scratch uses the trailing prefix to find each sequence's state
    and inverse without allocating slots for its neighbors' longest chunk.
    """
    if cpu_offsets is None:
        return None
    n_seq = cpu_offsets.numel() - 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if n_seq * heads <= num_sms and not pack_chunk_offsets:
        return None
    lengths = cpu_offsets[1:] - cpu_offsets[:-1]
    if n_seq * heads <= num_sms:
        order = torch.arange(n_seq, dtype=torch.int32)
    else:
        order = torch.argsort(lengths, stable=True, descending=True).to(torch.int32)
    if pack_chunk_offsets:
        chunk_counts = torch.div(lengths + 63, 64, rounding_mode="floor")
        prefix = torch.cat(
            (torch.zeros(1, dtype=torch.int32), chunk_counts.cumsum(0).to(torch.int32))
        )
        order = torch.cat((order, prefix))
    return order.to(device=device, dtype=torch.int32, non_blocking=True)


def _ensure_allocator() -> None:
    # Set every call (not gated): the TMA scratch allocator does not persist
    # across the autograd fwd->bwd boundary, so the bwd's descriptor kernels
    # need it re-set even after the fwd already set it.
    def _alloc(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(_alloc)


def kimi_delta_attention_ws_fwd(  # noqa: C901
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = 1.0,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    work_order: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    save_intermediate: bool = True,
    save_beta_u: bool = False,
    pack_saved_chunks: bool = False,
    output_final_state: bool = True,
    max_seq_length: Optional[int] = None,
):
    """Warp-specialized chunkwise KDA forward. q/k/v/g: [1,T,H,D]; beta:[1,T,H].

    Training path (``save_intermediate=True``, the default): the forward also
    stores each chunk's entering state [D,D] and WY inverse ``Tinv`` [C,C] to
    bf16 scratch. Fast heads store the state scaled by exp(Gamma/2), the
    operand their readout uses; bounded heads store the physical ``S_in`` and
    zero-fill their inverse slots, which records the choice for the backward.
    ``save_beta_u=True`` additionally saves an opaque work payload: rounded
    beta*U for fast heads and unscaled U for bounded heads.
    ``pack_saved_chunks=True`` sizes the saved state/inverse by each jagged
    sequence's actual chunk count using the already-available CPU offsets.
    By default the training path
    returns ``(o, states, tinv, S_end)``;
    the inference path (``save_intermediate=False``) returns ``(o, S_end)``
    without saving backward scratch. The terminal FP32 state has shape
    [N,H,D,D], matching ``chunk_kda``. Pass ``output_final_state=False`` to
    skip the write and retain the original return signatures.
    """
    assert q.dim() == 4 and q.shape[0] == 1
    _, T, H, D = q.shape
    assert D in (64, 128), "KDA TLX supports head_dim 64 or 128"
    BT = chunk_size
    assert BT == 64, "async MMA kernel currently requires chunk_size == 64"
    if save_beta_u and not save_intermediate:
        raise ValueError("Saving additional WS intermediates requires training mode")
    if pack_saved_chunks and (
        not save_intermediate or cu_seqlens_cpu is None or max_seq_length is not None
    ):
        raise ValueError("Compact saved chunks require CPU offsets in training mode")
    _ensure_allocator()

    q, k, v = (x.contiguous() for x in (q, k, v))
    # g staged as bf16 in SMEM (frees room for the epilogue o_buf); the kernel
    # casts back to fp32 before the decay cumsum.
    g = g.contiguous().to(torch.bfloat16)
    beta = beta.contiguous().to(torch.float32)
    o = torch.empty_like(v)
    saved_beta_u = torch.empty_like(v) if save_beta_u else None

    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    if pack_saved_chunks and work_order is None:
        work_order = _sequence_work_order(
            cu_seqlens_cpu, H, q.device, pack_chunk_offsets=True
        )
    if work_order is not None:
        assert work_order.shape == ((2 * n_seq + 1) if pack_saved_chunks else n_seq,)
        assert work_order.device == q.device and work_order.dtype == torch.int32
        assert work_order.is_contiguous()
    beta2 = beta.view(T, H)
    final_state = (
        torch.empty(n_seq, H, D, D, dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )

    if save_intermediate:
        # Production already passes the corresponding host offsets. Use them
        # for scratch sizing instead of synchronizing the GPU merely to read
        # the same sequence lengths back to Python on every AC recomputation.
        sizing_offsets = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu
        if cu_seqlens_cpu is not None:
            assert cu_seqlens_cpu.device.type == "cpu"
            assert cu_seqlens_cpu.numel() == n_seq + 1
            assert int(cu_seqlens_cpu[-1]) == T
        if max_seq_length is not None:
            # The GEO-exported op uses the batch maximum as a fixed bound for
            # saved intermediates. The normal training path retains its exact
            # host-offset sizing and existing allocation behavior.
            if cu_seqlens_cpu is None:
                raise ValueError("GEO export requires CPU sequence offsets")
            lengths = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
            if lengths.numel() and int(lengths.max()) > max_seq_length:
                raise ValueError(
                    "GEO max_seq_length is smaller than an actual sequence"
                )
            max_chunks = triton.cdiv(max_seq_length, BT)
        else:
            seqlens = (sizing_offsets[1:] - sizing_offsets[:-1]).tolist()
            chunks = [(int(s) + BT - 1) // BT for s in seqlens]
            max_chunks = max(chunks)
        if pack_saved_chunks:
            total_chunks = sum(chunks)
            states = torch.empty(
                total_chunks * H, D, D, dtype=torch.bfloat16, device=q.device
            )
            tinv = torch.empty(
                total_chunks * H, BT, BT, dtype=torch.bfloat16, device=q.device
            )
        else:
            states = torch.empty(
                n_seq * H, max_chunks, D, D, dtype=torch.bfloat16, device=q.device
            )
            tinv = torch.empty(
                n_seq * H, max_chunks, BT, BT, dtype=torch.bfloat16, device=q.device
            )
        save_s, save_t = states.view(-1), tinv.view(-1)
    else:
        max_chunks = 0
        states = tinv = None
        save_s = save_t = o.view(T, H * D)  # dummy; SAVE_INTERMEDIATE skips stores

    nbuf = 1  # NBUF=2 exceeds the per-CTA SMEM limit
    fuse_a_aqk = (
        save_intermediate
        and D == 128
        and q.dtype == torch.bfloat16
        and os.environ.get("KDA_WS_EXPERIMENT_FUSE_A_AQK") == "1"
    )

    # Launch up to eight programs per SM so uneven jagged sequence lengths do not
    # leave one long (seq, head) holding up the entire grid-strided tail.
    # Each program still handles a disjoint set of (seq, head) work-items.
    register_budget = (56, 168, 104)
    if (
        save_intermediate
        and os.environ.get("KDA_WS_EXPERIMENT_FWD_EPILOGUE_REGISTERS") == "1"
    ):
        # Keep the same total CTA register budget while giving the dependent
        # inverse/output drain more room than the compute task needs. D64's
        # smaller state tile lets the epilogue retain a larger budget.
        register_budget = (64, 152, 128) if D == 128 else (56, 136, 168)
        if fuse_a_aqk:
            # Give the inverse/output drain more registers without increasing
            # the fused CTA's total register allocation.
            register_budget = (56, 152, 152)
    n_work = n_seq * H
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(8 * num_sms, n_work),)
    # Per (sequence, head): [decay range or beta, ||A||_inf], written by the
    # main kernel and read by the bounded recurrence.
    unsafe_flags = torch.empty(2 * n_work, dtype=torch.int32, device=q.device)
    _kda_ws_fwd_kernel[grid](
        q.view(T, H * D),
        k.view(T, H * D),
        v.view(T, H * D),
        g.view(T, H * D),
        beta2,
        o.view(T, H * D),
        save_s,
        save_t,
        saved_beta_u.view(T, H * D) if saved_beta_u is not None else o.view(T, H * D),
        final_state.view(-1) if final_state is not None else o.view(-1),
        cu,
        work_order if work_order is not None else cu,
        unsafe_flags,
        n_seq,
        max_chunks,
        beta2.stride(0),
        beta2.stride(1),
        float(scale),
        H=H,
        D=D,
        BT=BT,
        NBUF=nbuf,
        SAVE_INTERMEDIATE=save_intermediate,
        SAVE_BETA_U=save_beta_u,
        PACKED_SCRATCH=pack_saved_chunks,
        OUTPUT_FINAL_STATE=output_final_state,
        SORT_WORK=work_order is not None,
        BETA_IN_PRODUCER=(
            D == 128 and os.environ.get("KDA_WS_EXPERIMENT_BETA_PRODUCER") == "1"
        ),
        FUSE_A_AQK=fuse_a_aqk,
        PRODUCER_REGISTERS=register_budget[0],
        COMPUTE_REGISTERS=register_budget[1],
        EPILOGUE_REGISTERS=register_budget[2],
        # The explicit dot layout covers all 8 warps: producer 1 + epilogue 1
        # + compute 6. Additional compiler stages do not help this manual
        # mbarrier pipeline.
        num_warps=8,
        num_stages=1,
    )
    stable_config = (
        H,
        D,
        v.stride(-3),
        beta2.stride(0),
        beta2.stride(1),
        float(scale),
        save_intermediate,
        save_beta_u,
        pack_saved_chunks,
        output_final_state,
    )
    _kda_stable_fwd_kernel[(n_work,)](
        (q, k, v, g, beta2),
        (
            o,
            save_s,
            save_t,
            saved_beta_u if saved_beta_u is not None else o,
            final_state if final_state is not None else o,
        ),
        (
            cu,
            work_order if work_order is not None else cu,
            unsafe_flags,
            n_seq,
            max_chunks,
        ),
        Config=stable_config,
        num_warps=8,
        num_stages=1,
    )
    if save_intermediate:
        result = (
            (o, states, tinv, final_state) if output_final_state else (o, states, tinv)
        )
        if saved_beta_u is not None:
            result += (saved_beta_u,)
        return result
    if output_final_state:
        return o, final_state
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
    output_final_state: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert cu_seqlens is not None
    # Inference-only entry: skip the backward-scratch save epilogue.
    result = kimi_delta_attention_ws_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        work_order=_sequence_work_order(cu_seqlens_cpu, q.shape[2], q.device),
        save_intermediate=False,
        output_final_state=output_final_state,
    )
    if output_final_state:
        return result
    o = result
    return o, None


def kimi_delta_attention_ws_bwd(
    q,
    k,
    v,
    g,
    beta,
    do,
    states,
    tinv,
    *,
    scale=1.0,
    cu_seqlens,
    chunk_size=64,
    saved_beta_u: Optional[torch.Tensor] = None,
    d_final_state=None,
    work_order: Optional[torch.Tensor] = None,
    grad_g_bf16: bool = False,
    grad_beta_bf16: bool = False,
    packed_qkv_gradient: Optional[torch.Tensor] = None,
):
    """Chunkwise KDA backward (single warp-spec kernel). `states` (entering state
    per chunk) and `tinv` (WY inverse per fast-path chunk) are the bf16 scratch
    saved by the forward, whose zero-filled inverse slots mark the heads it ran
    through the bounded recurrence. Those heads use the matching bounded VJP,
    as does every head when the optional FP32 `d_final_state` seeds the reverse
    scan. Returns dq, dk, dv, dg, dbeta."""
    _, T, H, D = q.shape
    assert D in (64, 128), "KDA TLX supports head_dim 64 or 128"
    if grad_g_bf16:
        assert g.dtype == torch.bfloat16
    if grad_beta_bf16:
        assert beta.dtype == torch.bfloat16
    BT = chunk_size
    assert BT == 64, "async MMA kernel currently requires chunk_size == 64"
    log2_bt = BT.bit_length() - 1
    _ensure_allocator()
    if do is None:
        do = torch.zeros_like(v)
    q, k, v, do = (x.contiguous() for x in (q, k, v, do))
    g = g.contiguous().to(torch.bfloat16)
    beta = beta.contiguous().to(torch.float32)
    if packed_qkv_gradient is None:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
    else:
        assert packed_qkv_gradient.shape == (1, T, 3 * H * D)
        assert packed_qkv_gradient.dtype == q.dtype
        assert packed_qkv_gradient.device == q.device
        assert packed_qkv_gradient.is_contiguous()
        packed_gradient_2d = packed_qkv_gradient.view(T, 3 * H * D)
        dq = packed_gradient_2d[:, : H * D].view(1, T, H, D)
        dk = packed_gradient_2d[:, H * D : 2 * H * D].view(1, T, H, D)
        dv = packed_gradient_2d[:, 2 * H * D :].view(1, T, H, D)
    dg = torch.empty(
        1,
        T,
        H,
        D,
        dtype=torch.bfloat16 if grad_g_bf16 else torch.float32,
        device=q.device,
    )
    dbeta = torch.empty(
        1,
        T,
        H,
        dtype=torch.bfloat16 if grad_beta_bf16 else torch.float32,
        device=q.device,
    )

    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
    pack_saved_chunks = states.ndim == 3
    if work_order is not None:
        assert work_order.shape == ((2 * n_seq + 1) if pack_saved_chunks else n_seq,)
        assert work_order.device == q.device and work_order.dtype == torch.int32
        assert work_order.is_contiguous()
    assert states.ndim in (3, 4) and states.shape[-2:] == (D, D)
    assert tinv.ndim == states.ndim and tinv.shape[-2:] == (BT, BT)
    max_chunks = 0 if pack_saved_chunks else states.shape[1]
    if pack_saved_chunks:
        assert work_order is not None and work_order.shape == (2 * n_seq + 1,)
    if d_final_state is not None:
        assert d_final_state.shape == (n_seq, H, D, D)
        d_final_state = d_final_state.contiguous()
    if saved_beta_u is not None:
        assert saved_beta_u.shape == v.shape
        assert saved_beta_u.dtype == v.dtype and saved_beta_u.device == v.device
        saved_beta_u = saved_beta_u.contiguous()
    HD = H * D
    q2, k2, v2, g2, do2 = (x.view(T, HD) for x in (q, k, v, g, do))
    dq2, dk2, dv2, dg2 = (x.view(T, HD) for x in (dq, dk, dv, dg))
    beta2 = beta.view(T, H)
    dbeta2 = dbeta.view(T, H)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    # Match the forward grid's larger work queue for variable sequence lengths.
    grid = (min(8 * num_sms, n_seq * H),)
    compact_d64 = (
        D == 64
        and d_final_state is None
        and os.environ.get("KDA_WS_EXPERIMENT_D64_TWO_CTA") == "1"
    )
    # The bounded VJP below handles every head that seeds dS_end, and every
    # head the forward routed to its bounded recurrence; it overwrites the
    # latter's fast-path gradients.
    if d_final_state is None:
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
            (saved_beta_u.view(T, HD) if saved_beta_u is not None else v2),
            states.view(-1),
            cu,
            work_order if work_order is not None else cu,
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
            HAS_SAVED_BETA_U=saved_beta_u is not None,
            PACKED_SCRATCH=pack_saved_chunks,
            HAS_FINAL_STATE_GRAD=False,
            SORT_WORK=work_order is not None,
            GRAD_G_BF16=grad_g_bf16,
            UNIT_SCALE_EGS=(
                os.environ.get("KDA_WS_EXPERIMENT_UNIT_SCALE_EGS") == "1"
                and float(scale) == 1.0
                and D == 128
            ),
            STAGE_EGB=(
                os.environ.get("KDA_WS_EXPERIMENT_STAGE_EGB") == "1"
                and D == 128
                and float(scale) == 1.0
            ),
            COMPACT_D64=compact_d64,
            STAGE_DG_IN_S=(
                grad_g_bf16
                and os.environ.get("KDA_WS_EXPERIMENT_STAGE_DG_IN_S", "1") == "1"
            ),
            SPLIT_DG_EPILOGUE=(
                D == 64
                and grad_g_bf16
                and os.environ.get("KDA_WS_EXPERIMENT_SPLIT_DG_D64") == "1"
            ),
            GRAD_STRIDE=dq2.stride(0),
            num_warps=4 if compact_d64 else 8,
            num_stages=1,
        )
    stable_config = (
        H,
        D,
        HD,
        beta2.stride(0),
        beta2.stride(1),
        float(scale),
        True,
        saved_beta_u is not None,
        pack_saved_chunks,
        True,
        dq2.stride(0),
    )
    _kda_stable_bwd_kernel[(n_seq * H,)](
        (q2, k2, v2, g2, beta2),
        (dq2, dk2, dv2, dg2, dbeta2, do2),
        (
            states,
            saved_beta_u if saved_beta_u is not None else v2,
            d_final_state if d_final_state is not None else states,
        ),
        (
            cu,
            work_order if work_order is not None else cu,
            tinv.view(-1),
            n_seq,
            max_chunks,
        ),
        Config=stable_config,
        HAS_PAYLOAD=saved_beta_u is not None,
        HAS_FINAL=d_final_state is not None,
        num_warps=8,
        num_stages=1,
    )
    return dq, dk, dv, dg, dbeta


class _KDAWSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, g, beta, scale, cu_seqlens, cu_seqlens_cpu, output_final_state
    ):
        # The forward and backward kernels both consume BF16 g. Keep the
        # converted tensor for backward instead of saving the larger FP32
        # original and converting the same token data a second time.
        g_for_kernel = g.contiguous().to(torch.bfloat16)
        pack_saved_chunks = (
            cu_seqlens_cpu is not None
            and os.environ.get("KDA_WS_EXPERIMENT_PACK_SAVED_CHUNKS") == "1"
        )
        work_order = _sequence_work_order(
            cu_seqlens_cpu,
            q.shape[2],
            q.device,
            pack_chunk_offsets=pack_saved_chunks,
        )
        # Benchmark-only opt-in while evaluating the extra training activation.
        # The production choice needs a typed model configuration after EMS A/B.
        save_beta_u = (
            q.shape[3] == 128
            or (
                q.shape[3] == 64
                and os.environ.get("KDA_WS_EXPERIMENT_SAVE_D64_BETA_U") == "1"
            )
        ) and os.environ.get("KDA_WS_EXPERIMENT_SAVE_BETA_U") == "1"
        result = kimi_delta_attention_ws_fwd(
            q,
            k,
            v,
            g_for_kernel,
            beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            work_order=work_order,
            save_intermediate=True,
            save_beta_u=save_beta_u,
            pack_saved_chunks=pack_saved_chunks,
            output_final_state=output_final_state,
        )
        if output_final_state:
            o, states, tinv, final_state = result[:4]
        else:
            o, states, tinv = result[:3]
        saved_beta_u = result[-1] if save_beta_u else None
        # Once beta*U is available, the backward only needs V's shape/dtype.
        # Use the saved intermediate for both roles so the V projection result
        # need not remain live throughout the rest of the training forward.
        v_or_saved_beta_u = saved_beta_u if saved_beta_u is not None else v
        ctx.save_for_backward(
            q,
            k,
            v_or_saved_beta_u,
            g_for_kernel,
            beta,
            cu_seqlens,
            states,
            tinv,
        )
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        ctx.work_order = work_order
        ctx.save_beta_u = save_beta_u
        direct_bf16_grads = os.environ.get("KDA_WS_EXPERIMENT_BF16_INPUT_GRADS") == "1"
        # Autograd would cast both FP32 gradients to the original BF16 input
        # dtypes. Let the TLX kernel write those rounded gradients directly.
        ctx.grad_g_bf16 = direct_bf16_grads and g.dtype == torch.bfloat16
        ctx.grad_beta_bf16 = direct_bf16_grads and beta.dtype == torch.bfloat16
        if output_final_state:
            # When the caller discards S_end, PyTorch otherwise materializes a
            # dense zero dS_end and needlessly selects the more expensive seeded
            # backward. The output-only path retains its original kernel.
            ctx.set_materialize_grads(False)
            return o, final_state
        return o

    @staticmethod
    def backward(ctx, *grad_outputs):
        do = grad_outputs[0]
        d_final_state = grad_outputs[1] if ctx.output_final_state else None
        q, k, v_or_saved_beta_u, g, beta, cu_seqlens, states, tinv = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = kimi_delta_attention_ws_bwd(
            q,
            k,
            v_or_saved_beta_u,
            g,
            beta,
            do,
            states,
            tinv,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            saved_beta_u=v_or_saved_beta_u if ctx.save_beta_u else None,
            d_final_state=d_final_state,
            work_order=ctx.work_order,
            grad_g_bf16=ctx.grad_g_bf16,
            grad_beta_bf16=ctx.grad_beta_bf16,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None


@torch.library.custom_op("tlx::kda_ws_training_geo", mutates_args=())
def _kda_ws_training_geo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    scale: float,
    max_seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_for_kernel = g.contiguous().to(torch.bfloat16)
    o, states, tinv, final_state = kimi_delta_attention_ws_fwd(
        q,
        k,
        v,
        g_for_kernel,
        beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        work_order=_sequence_work_order(cu_seqlens_cpu, q.shape[2], q.device),
        save_intermediate=True,
        output_final_state=True,
        max_seq_length=max_seq_length,
    )
    return o, states, tinv, final_state


@_kda_ws_training_geo.register_fake
def _fake_kda_ws_training_geo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    scale: float,
    max_seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, heads, dim = q.shape
    n_seq = cu_seqlens.numel() - 1
    max_chunks = triton.cdiv(max_seq_length, 64)
    return (
        torch.empty_like(v),
        torch.empty(
            n_seq * heads,
            max_chunks,
            dim,
            dim,
            dtype=torch.bfloat16,
            device=q.device,
        ),
        torch.empty(
            n_seq * heads,
            max_chunks,
            64,
            64,
            dtype=torch.bfloat16,
            device=q.device,
        ),
        torch.empty(n_seq, heads, dim, dim, dtype=torch.float32, device=q.device),
    )


def _setup_kda_ws_training_geo(ctx, inputs, output) -> None:
    q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, scale, _max_seq_length = inputs
    _o, states, tinv, _final_state = output
    g_for_kernel = g.contiguous().to(torch.bfloat16)
    ctx.save_for_backward(q, k, v, g_for_kernel, beta, cu_seqlens, states, tinv)
    ctx.scale = scale
    ctx.work_order = _sequence_work_order(cu_seqlens_cpu, q.shape[2], q.device)
    direct_bf16_grads = os.environ.get("KDA_WS_EXPERIMENT_BF16_INPUT_GRADS") == "1"
    ctx.grad_g_bf16 = direct_bf16_grads and g.dtype == torch.bfloat16
    ctx.grad_beta_bf16 = direct_bf16_grads and beta.dtype == torch.bfloat16
    ctx.set_materialize_grads(False)


def _backward_kda_ws_training_geo(
    ctx, grad_o, _grad_states, _grad_tinv, grad_final_state
):
    q, k, v, g, beta, cu_seqlens, states, tinv = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = kimi_delta_attention_ws_bwd(
        q,
        k,
        v,
        g,
        beta,
        grad_o,
        states,
        tinv,
        scale=ctx.scale,
        cu_seqlens=cu_seqlens,
        d_final_state=grad_final_state,
        work_order=ctx.work_order,
        grad_g_bf16=ctx.grad_g_bf16,
        grad_beta_bf16=ctx.grad_beta_bf16,
    )
    return dq, dk, dv, dg, dbeta, None, None, None, None


_kda_ws_training_geo.register_autograd(
    _backward_kda_ws_training_geo, setup_context=_setup_kda_ws_training_geo
)


@torch._dynamo.disable
def kimi_delta_attention_ws_autograd(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=1.0,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    output_final_state=True,
    max_seq_length=None,
):
    """Keep the custom VJP and host-offset scratch sizing outside AOT tracing.

    The caller can still compile its projections and output epilogue without
    requiring an outer activation checkpoint to preserve the WS saved state.
    """
    assert cu_seqlens is not None
    if torch.compiler.is_exporting() and output_final_state:
        assert max_seq_length is not None
        o, _states, _tinv, final_state = _kda_ws_training_geo(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            cu_seqlens_cpu,
            scale,
            max_seq_length,
        )
        return o, final_state
    result = _KDAWSFunction.apply(
        q, k, v, g, beta, scale, cu_seqlens, cu_seqlens_cpu, output_final_state
    )
    if output_final_state:
        return result
    o = result
    return o, None


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, space="full"):
    """Full fwd+bwd autograd entry for `tlx.ops.kimi_delta_attention`."""
    del space
    return kimi_delta_attention_ws_autograd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        output_final_state=False,
    )
