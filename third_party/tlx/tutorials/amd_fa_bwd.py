"""BF16 Flash-Attention backward kernel families for AMD gfx950.

The public ``fa_backward`` wrapper supports dense contiguous BF16 D64 tensors
with matching positive batches, Hkv dividing Hq, SQ/SKV at least 256 and
aligned to 64, SQ no larger than 8,388,544, and SKV no larger than 16,777,152.
Bottom-right causal D64 additionally requires SQ no larger than SKV. It also
supports the two validated equal-head configurations in ``SUPPORTED_SHAPES``
and the causal/non-causal HipKittens GQA contract: positive B/Hq/Hkv, Hkv dividing
Hq, N a positive multiple of 256, and D=128. ``GQA_BENCHMARK_SHAPES`` records
the published performance series; it is not an allow-list. Other
configurations are not part of this submission's public contract yet. Run
this file with pytest for correctness.

Each launch topology has one stable JIT entry.  Constexpr schedule kwargs pick
the tuned split, persistent, staged, peeled, or hoisted implementation behind
that entry; algorithms with different output ownership or launch counts remain
separate instead of being hidden in one monolithic kernel.

The validated short D128 MFMA/LDS path is opt-in for ``(16,27,200,128)`` through
``TLX_FA_BWD_ENABLE_EXACT_D128=1``.  With no D128 opt-in flag, the older split
path is used.  The other fused D128 short-context kernels are narrow, opt-in
experiments selected by ``TLX_FA_BWD_ENABLE_PERSISTENT_D128=1`` (combined) or
``TLX_FA_BWD_ENABLE_PERSISTENT_D128_PIPE=1`` (async Q/dO pipeline); they are
not production dispatches and currently spill on the generic TLX lowering.
"""

import ast
import dataclasses
import math
import os
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip_cdna4

# Public equal-head correctness contract for this submission. The kernels
# have broader D128 and D256 families internally, but these are the two MHA
# tuples validated end-to-end on gfx950.
SUPPORTED_SHAPES = {
    (16, 27, 200, 128),
    (32, 1, 2600, 256),
}
# Published HipKittens performance series. The supported GQA space is defined
# by _is_supported_gqa_shape rather than limited to these benchmark points.
# Signatures are (batch, query heads, KV heads, sequence, dimension).
GQA_BENCHMARK_SHAPES = {
    (16, 64, 8, 1024, 128),
    (16, 64, 8, 2048, 128),
    (16, 64, 8, 4096, 128),
    (16, 64, 8, 8192, 128),
    (15, 64, 8, 16384, 128),
}
_GQA_SHAPE_CONSTRAINT = ("B >= 1, Hq >= 1, Hkv >= 1, Hq % Hkv == 0, "
                         "N >= 256, N % 256 == 0, and D == 128")


def _is_supported_gqa_shape(shape):
    """Return whether a (B, Hq, Hkv, N, D) signature matches HipKittens."""
    if len(shape) != 5:
        return False
    batch, hq, hk, n_ctx, head_dim = shape
    return (batch >= 1 and hq >= 1 and hk >= 1 and hq % hk == 0 and n_ctx >= 256 and n_ctx % 256 == 0
            and head_dim == 128)


# Gluon pins CDNA4's 16x16x32 MFMA for these BF16 tiles.  Leaving Triton to
# infer the instruction shape selects 32x32x16 on this checkout, which doubles
# register pressure for the D-sliced kernels and prevents the accumulator from
# using AGPRs on gfx950.
_CDNA4_MATRIX_INSTR_NONKDIM = 16


@dataclasses.dataclass(frozen=True)
class ReferenceCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    do: torch.Tensor
    lse: torch.Tensor
    sm_scale: float
    causal: bool
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def kernel_args(self):
        return (self.q, self.k, self.v, self.o, self.do, self.lse, self.sm_scale, self.causal)


def make_reference_case(shape, causal, seed=0):
    """Build forward state and FP32 autograd gradients one head at a time."""
    batch, heads, n_ctx, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty(shape, device="cuda", dtype=torch.float32)
    dk = torch.empty_like(dq)
    dv = torch.empty_like(dq)
    sm_scale = head_dim**-0.5
    causal_mask = None
    if causal:
        causal_mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1)

    for batch_idx in range(batch):
        for head_idx in range(heads):
            q_ref = q[batch_idx, head_idx].float().requires_grad_(True)
            k_ref = k[batch_idx, head_idx].float().requires_grad_(True)
            v_ref = v[batch_idx, head_idx].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            o_ref = torch.matmul(probs, v_ref)
            grads = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do[batch_idx, head_idx].float())
            with torch.no_grad():
                o[batch_idx, head_idx].copy_(o_ref)
                lse[batch_idx, head_idx].copy_(lse_ref)
                dq[batch_idx, head_idx].copy_(grads[0])
                dk[batch_idx, head_idx].copy_(grads[1])
                dv[batch_idx, head_idx].copy_(grads[2])

    return ReferenceCase(q, k, v, o, do, lse, sm_scale, causal, (dq, dk, dv))


def _make_gqa_smoke_case(shape=(1, 8, 1, 512, 128), causal=False, seed=0, sm_scale=None):
    """Build a small supported GQA reference case."""
    assert _is_supported_gqa_shape(shape)
    batch, hq, hk, n_ctx, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (batch, hq, n_ctx, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, hk, n_ctx, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (batch, hk, n_ctx, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    do = torch.randn(
        q.shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    o = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    causal_mask = None
    if causal:
        causal_mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1)
    group_size = hq // hk

    for batch_idx in range(batch):
        for query_head in range(hq):
            kv_head = query_head // group_size
            q_ref = q[batch_idx, query_head].float().requires_grad_(True)
            k_ref = k[batch_idx, kv_head].float().requires_grad_(True)
            v_ref = v[batch_idx, kv_head].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            o_ref = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
            grads = torch.autograd.grad(
                o_ref,
                (q_ref, k_ref, v_ref),
                do[batch_idx, query_head].float(),
            )
            with torch.no_grad():
                o[batch_idx, query_head].copy_(o_ref)
                lse[batch_idx, query_head].copy_(lse_ref)
                dq[batch_idx, query_head].copy_(grads[0])
                dk[batch_idx, kv_head].add_(grads[1])
                dv[batch_idx, kv_head].add_(grads[2])

    return ReferenceCase(q, k, v, o, do, lse, sm_scale, causal, (dq, dk, dv))


@triton.jit
def _attn_bwd_preprocess_kernel(
    O,
    DO,
    Delta,
    DQ_ACC,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ZERO_DQ: tl.constexpr,
):
    batch_head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    # Keep the per-program offsets below the signed 2-GiB buffer boundary.
    # The B=15, H=64, N=16384 HK stress case crosses that boundary at batch
    # eight when batch/head is folded into a 32-bit element offset.
    tensor_base = batch_head.to(tl.int64) * N * D
    delta_base = batch_head.to(tl.int64) * N
    offsets = rows[:, None] * D + cols[None, :]
    mask = rows[:, None] < N
    o = tl.load(O + tensor_base + offsets, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + tensor_base + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(Delta + delta_base + rows, tl.sum(o * do, axis=1), mask=rows < N)
    if ZERO_DQ:
        tl.store(DQ_ACC + tensor_base + offsets, 0.0, mask=mask)


def _run_bwd_preprocess(o, do, delta, dq_acc=None):
    batch, heads, n_ctx, head_dim = o.shape
    block_m = 64
    grid = (triton.cdiv(n_ctx, block_m), batch * heads)
    _attn_bwd_preprocess_kernel[grid](
        o,
        do,
        delta,
        dq_acc if dq_acc is not None else delta,
        N=n_ctx,
        D=head_dim,
        BLOCK_M=block_m,
        ZERO_DQ=dq_acc is not None,
        num_warps=4,
    )


@triton.jit
def _attn_bwd_dkdv_d128_single_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    n0 = pid_n * BLOCK
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D
    row_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    row_mask = offs_n[:, None] < N

    if BLOCK == 32:
        # Gluon's BM32 causal schedule drops the row bit at 32 and uses the
        # phase-shifted [16, 0]/[8, 0] bases.  Keeping those bases out of the
        # BM64 descriptor is required: a row bit beyond the logical tile shape
        # makes the memdesc reinterpret verifier reject the layout.
        shared_offset_bases: tl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [8, 0],
            [1, 0],
            [2, 0],
            [4, 0],
        ]
    else:
        shared_offset_bases: tl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ]
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 32)], shared_offset_bases,
                                                                               [BLOCK, D])
    k_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)
    v_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)
    q_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)
    do_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)

    # Gluon's BM32 causal schedule uses the same eight-value direct-to-LDS
    # ownership as its rectangular full-attention path.  Keep the older
    # register-staged async load for the BM64 fallback used by other lengths.
    if BLOCK == 32:
        qdo_async_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
            stride=((8, 16, 32, 64, 2048, 1024, 128), (1, 2, 4, 512, 256)),
        )
    else:
        qdo_async_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((8, 16, 32, 64, 2048, 1024, 128, 256), (1, 2, 4, 512)),
        )

    if BLOCK == 32:
        if (n0 + BLOCK) > N:
            # As with the Q/dO copies below, masked direct-to-LDS K/V copies
            # leave OOB rows unspecified.  Clear the one tail tile before the
            # async transaction so masked MFMA lanes see numerical zeros.
            tlx.local_store(tlx.local_view(k_buffers, 0), tl.zeros((BLOCK, D), tl.bfloat16))
            tlx.local_store(tlx.local_view(v_buffers, 0), tl.zeros((BLOCK, D), tl.bfloat16))
            tl.debug_barrier()
        kv_offsets = row_ptrs.to(tl.int32)
        kv_offsets = tlx.require_layout(kv_offsets, qdo_async_layout)
        kv_load_mask = tl.broadcast_to(row_mask, kv_offsets.shape)
        kv_load_mask = tlx.require_layout(kv_load_mask, qdo_async_layout)
        k_token = tlx.buffer_load_to_local(tlx.local_view(k_buffers, 0), K, kv_offsets, mask=kv_load_mask)
        v_token = tlx.buffer_load_to_local(tlx.local_view(v_buffers, 0), V, kv_offsets, mask=kv_load_mask)
    else:
        k_token = tlx.async_load(K + row_ptrs, tlx.local_view(k_buffers, 0), mask=row_mask, other=0.0)
        v_token = tlx.async_load(V + row_ptrs, tlx.local_view(v_buffers, 0), mask=row_mask, other=0.0)
    tlx.async_load_commit_group([k_token, v_token])
    kv_wait = tlx.async_load_wait_group(0)
    k_tile = tlx.local_load(tlx.local_view(k_buffers, 0), token=kv_wait)
    v_tile = tlx.local_load(tlx.local_view(v_buffers, 0), token=kv_wait)

    dk = tl.zeros((BLOCK, D), tl.float32)
    dv = tl.zeros((BLOCK, D), tl.float32)
    start_m_block = pid_n if IS_CAUSAL else 0
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(start_m_block, num_m_blocks):
        tl.debug_barrier()
        offs_m = m_block * BLOCK + tl.arange(0, BLOCK)
        qdo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        qdo_mask = offs_m[:, None] < N
        if BLOCK == 32:
            if (m_block + 1) * BLOCK > N:
                # Masked direct-to-LDS copies leave OOB rows untouched. Clear
                # the reused Q/dO tile before the final partial causal block
                # so invalid rows cannot feed 0 * NaN into dP, dV, or dK.
                tlx.local_store(tlx.local_view(q_buffers, 0), tl.zeros((BLOCK, D), tl.bfloat16))
                tlx.local_store(tlx.local_view(do_buffers, 0), tl.zeros((BLOCK, D), tl.bfloat16))
                tl.debug_barrier()
            qdo_offsets = qdo_ptrs.to(tl.int32)
            qdo_offsets = tlx.require_layout(qdo_offsets, qdo_async_layout)
            qdo_load_mask = tl.broadcast_to(qdo_mask, qdo_offsets.shape)
            qdo_load_mask = tlx.require_layout(qdo_load_mask, qdo_async_layout)
            q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, 0), Q, qdo_offsets, mask=qdo_load_mask)
            do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, 0), DO, qdo_offsets, mask=qdo_load_mask)
        else:
            q_token = tlx.async_load(Q + qdo_ptrs, tlx.local_view(q_buffers, 0), mask=qdo_mask, other=0.0)
            do_token = tlx.async_load(DO + qdo_ptrs, tlx.local_view(do_buffers, 0), mask=qdo_mask, other=0.0)
        tlx.async_load_commit_group([q_token, do_token])
        qdo_wait = tlx.async_load_wait_group(0)
        q_tile = tlx.local_load(tlx.local_view(q_buffers, 0), token=qdo_wait)
        do_tile = tlx.local_load(tlx.local_view(do_buffers, 0), token=qdo_wait)
        q_t = tlx.local_load(tlx.local_trans(tlx.local_view(q_buffers, 0)), token=qdo_wait)
        do_t = tlx.local_load(tlx.local_trans(tlx.local_view(do_buffers, 0)), token=qdo_wait)

        scores_t = tl.dot(k_tile, q_t)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[:, None] <= offs_m[None, :])
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)
        dp_t = tl.dot(v_tile, do_t)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
        dk = tl.dot(ds_t.to(tl.bfloat16), q_tile, dk)

    dk *= SM_SCALE
    tl.store(DK + row_ptrs, dk.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + row_ptrs, dv.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d128_pipeline_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    n0 = pid_n * BLOCK
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D
    row_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    row_mask = offs_n[:, None] < N
    k_tile = tl.load(K + row_ptrs, mask=row_mask, other=0.0)
    v_tile = tl.load(V + row_ptrs, mask=row_mask, other=0.0)

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, D],
    )
    q_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 2, layout=shared_layout)
    do_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 2, layout=shared_layout)

    first_m = tl.arange(0, BLOCK)
    first_ptrs = tensor_base + first_m[:, None] * D + offs_d[None, :]
    first_mask = first_m[:, None] < N
    q_token = tlx.async_load(Q + first_ptrs, tlx.local_view(q_buffers, 0), mask=first_mask, other=0.0)
    do_token = tlx.async_load(DO + first_ptrs, tlx.local_view(do_buffers, 0), mask=first_mask, other=0.0)
    tlx.async_load_commit_group([q_token, do_token])

    dk = tl.zeros((BLOCK, D), tl.float32)
    dv = tl.zeros((BLOCK, D), tl.float32)
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(0, num_m_blocks):
        tl.debug_barrier()
        current_slot = m_block % 2
        next_slot = 1 - current_slot
        next_m = (m_block + 1) * BLOCK + tl.arange(0, BLOCK)
        next_ptrs = tensor_base + next_m[:, None] * D + offs_d[None, :]
        next_mask = next_m[:, None] < N
        q_token = tlx.async_load(
            Q + next_ptrs,
            tlx.local_view(q_buffers, next_slot),
            mask=next_mask,
            other=0.0,
        )
        do_token = tlx.async_load(
            DO + next_ptrs,
            tlx.local_view(do_buffers, next_slot),
            mask=next_mask,
            other=0.0,
        )
        tlx.async_load_commit_group([q_token, do_token])
        qdo_wait = tlx.async_load_wait_group(1)

        q_view = tlx.local_view(q_buffers, current_slot)
        do_view = tlx.local_view(do_buffers, current_slot)
        q_tile = tlx.local_load(q_view, token=qdo_wait)
        do_tile = tlx.local_load(do_view, token=qdo_wait)
        q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait)
        do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait)

        offs_m = m_block * BLOCK + tl.arange(0, BLOCK)
        scores_t = tl.dot(k_tile, q_t)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)
        dp_t = tl.dot(v_tile, do_t)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
        dk = tl.dot(ds_t.to(tl.bfloat16), q_tile, dk)

    tlx.async_load_wait_group(0)
    dk *= SM_SCALE
    tl.store(DK + row_ptrs, dk.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + row_ptrs, dv.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d128_rect_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BM32/BN64 short-context dK/dV port of Gluon's rectangular path.

    The existing TLX pipeline uses one square tile for both ownership axes.
    Gluon's short non-causal winner uses a 32-row Q producer with a 64-row KV
    tile instead: K/V stay resident in registers while the smaller Q/dO ring
    reduces the per-CTA live range.  This kernel keeps the same arithmetic and
    async ring, but makes M and N independent so that schedule can be selected
    without changing the 64x64 dQ owner.
    """
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(D == 128)
    tl.static_assert(not IS_CAUSAL)
    tl.static_assert(0 < N)
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D
    row_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    row_mask = offs_n[:, None] < N
    k_tile = tl.load(K + row_ptrs, mask=row_mask, other=0.0)
    v_tile = tl.load(V + row_ptrs, mask=row_mask, other=0.0)

    # BM32's phase-shifted row bases match Gluon's two/four-wave descriptor;
    # the [32, 0] basis used by BM64 is outside this logical tile.
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [8, 0],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        [BLOCK_M, D],
    )
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=shared_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=shared_layout)

    # Match Gluon's CDNA4 direct-to-LDS ownership for BM32/D128.  The register
    # layout gives every lane an 8-value contiguous vector while the two warp
    # bits cover the row dimension.  Keeping offsets in this explicit layout
    # lets gfx950 issue a coalesced global->LDS transaction without first
    # materializing a register-side Q/dO tile.
    qdo_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 2048, 1024, 128, 256), (1, 2, 4, 512)),
    )

    first_m = tl.arange(0, BLOCK_M)
    first_mask = first_m[:, None] < N
    first_offsets = (tensor_base + first_m[:, None] * D + offs_d[None, :]).to(tl.int32)
    first_offsets = tlx.require_layout(first_offsets, qdo_async_layout)
    first_load_mask = tl.broadcast_to(first_mask, first_offsets.shape)
    first_load_mask = tlx.require_layout(first_load_mask, qdo_async_layout)
    q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, 0), Q, first_offsets, mask=first_load_mask)
    do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, 0), DO, first_offsets, mask=first_load_mask)
    tlx.async_load_commit_group([q_token, do_token])

    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK_M)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(0, num_m_blocks):
        tl.debug_barrier()
        current_slot = m_block % 2
        next_slot = 1 - current_slot
        next_m = (m_block + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
        next_mask = next_m[:, None] < N
        if ((m_block + 1) * BLOCK_M + BLOCK_M) > N:
            # A masked direct-to-LDS copy leaves OOB rows untouched. Clear the
            # reused slot before issuing the final partial Q/dO tile so the
            # score/dP products cannot observe stale values.
            tlx.local_store(tlx.local_view(q_buffers, next_slot), tl.zeros((BLOCK_M, D), tl.bfloat16))
            tlx.local_store(tlx.local_view(do_buffers, next_slot), tl.zeros((BLOCK_M, D), tl.bfloat16))
            tl.debug_barrier()
        next_offsets = (tensor_base + next_m[:, None] * D + offs_d[None, :]).to(tl.int32)
        next_offsets = tlx.require_layout(next_offsets, qdo_async_layout)
        next_load_mask = tl.broadcast_to(next_mask, next_offsets.shape)
        next_load_mask = tlx.require_layout(next_load_mask, qdo_async_layout)
        next_q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, next_slot), Q, next_offsets,
                                                mask=next_load_mask)
        next_do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, next_slot), DO, next_offsets,
                                                 mask=next_load_mask)
        tlx.async_load_commit_group([next_q_token, next_do_token])
        qdo_wait = tlx.async_load_wait_group(1)

        q_view = tlx.local_view(q_buffers, current_slot)
        do_view = tlx.local_view(do_buffers, current_slot)
        q_tile = tlx.local_load(q_view, token=qdo_wait)
        do_tile = tlx.local_load(do_view, token=qdo_wait)
        q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait)
        do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait)

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        scores_t = tl.dot(k_tile, q_t)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)
        dp_t = tl.dot(v_tile, do_t)
        ds_t = p_t * (dp_t - delta[None, :])
        dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
        dk = tl.dot(ds_t.to(tl.bfloat16), q_tile, dk)

    tlx.async_load_wait_group(0)
    dk *= SM_SCALE
    tl.store(DK + row_ptrs, dk.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + row_ptrs, dv.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d128_split_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINED: tl.constexpr,
    RECTANGULAR: tl.constexpr,
):
    """Stable KV-owned D128 split entry configured by schedule kwargs."""
    if PIPELINED:
        if RECTANGULAR:
            _attn_bwd_dkdv_d128_rect_impl(
                Q,
                K,
                V,
                DO,
                LSE,
                Delta,
                DK,
                DV,
                SM_SCALE,
                IS_CAUSAL,
                N,
                D,
                BLOCK_M,
                BLOCK_N,
            )
        else:
            tl.static_assert(BLOCK_M == BLOCK_N)
            _attn_bwd_dkdv_d128_pipeline_impl(
                Q,
                K,
                V,
                DO,
                LSE,
                Delta,
                DK,
                DV,
                SM_SCALE,
                IS_CAUSAL,
                N,
                D,
                BLOCK_N,
            )
    else:
        tl.static_assert(not RECTANGULAR and BLOCK_M == BLOCK_N)
        _attn_bwd_dkdv_d128_single_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            BLOCK_N,
        )


@triton.jit
def _attn_bwd_dq_d128_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    m0 = pid_m * BLOCK
    offs_m = m0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D
    qdo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
    qdo_mask = offs_m[:, None] < N
    q_tile = tl.load(Q + qdo_ptrs, mask=qdo_mask, other=0.0)
    do_tile = tl.load(DO + qdo_ptrs, mask=qdo_mask, other=0.0)
    lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
    delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, D],
    )
    k_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)
    v_buffers = tlx.local_alloc((BLOCK, D), tl.bfloat16, 1, layout=shared_layout)
    dq = tl.zeros((BLOCK, D), tl.float32)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    end_n_block = pid_m + 1 if IS_CAUSAL else num_n_blocks
    log2e: tl.constexpr = 1.4426950408889634

    for n_block in range(0, end_n_block):
        tl.debug_barrier()
        offs_n = n_block * BLOCK + tl.arange(0, BLOCK)
        kv_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
        kv_mask = offs_n[:, None] < N
        k_token = tlx.async_load(K + kv_ptrs, tlx.local_view(k_buffers, 0), mask=kv_mask, other=0.0)
        v_token = tlx.async_load(V + kv_ptrs, tlx.local_view(v_buffers, 0), mask=kv_mask, other=0.0)
        tlx.async_load_commit_group([k_token, v_token])
        kv_wait = tlx.async_load_wait_group(0)
        k_tile = tlx.local_load(tlx.local_view(k_buffers, 0), token=kv_wait)
        k_t = tlx.local_load(tlx.local_trans(tlx.local_view(k_buffers, 0)), token=kv_wait)
        v_t = tlx.local_load(tlx.local_trans(tlx.local_view(v_buffers, 0)), token=kv_wait)

        scores = tl.dot(q_tile, k_t)
        scores = scores * (SM_SCALE * log2e) - lse[:, None] * log2e
        valid = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, float("-inf"))
        p = tl.math.exp2(scores)
        dp = tl.dot(do_tile, v_t)
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(tl.bfloat16), k_tile, dq)

    dq *= SM_SCALE
    tl.store(DQ + qdo_ptrs, dq.to(tl.bfloat16), mask=qdo_mask)


# TODO: Revisit or remove this generic persistent variant once TLX can express
# the CDNA4 Gluon ownership without the current register spills. It is kept
# only as an opt-in correctness/performance experiment; split D128 remains the
# default and the exact MFMA/LDS route is the measured path.
@triton.jit
def _attn_bwd_dkdv_dq_d128_persistent_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fuse D128 dK/dV and dQ for one short, MHA sequence.

    BLOCK_N covers the complete supported short sequence. K/V are loaded once
    into LDS, while each BM16 Q/dO tile produces dS, dQ, dK, and dV. The
    launch uses eight warps because TLX's generic blocked layout otherwise
    spills the full 256-row K/V image on gfx950. This is retained as an
    experimental comparison path; the split implementation is faster on the
    current compiler and remains the default dispatch.
    """
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(D == 128)
    tl.static_assert(0 < N and N <= BLOCK_N)
    batch_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D

    kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_N, D],
    )
    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)
    v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)

    key_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    key_mask = offs_n[:, None] < N
    k_token = tlx.async_load(K + key_ptrs, tlx.local_view(k_buffer, 0), mask=key_mask, other=0.0)
    v_token = tlx.async_load(V + key_ptrs, tlx.local_view(v_buffer, 0), mask=key_mask, other=0.0)
    tlx.async_load_commit_group([k_token, v_token])
    kv_wait = tlx.async_load_wait_group(0)
    k_tile = tlx.local_load(tlx.local_view(k_buffer, 0), token=kv_wait)
    v_tile = tlx.local_load(tlx.local_view(v_buffer, 0), token=kv_wait)

    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK_M)
    log2e: tl.constexpr = 1.4426950408889634

    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    for m_block in range(0, num_m_blocks):
        tl.debug_barrier()
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        qdo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        qdo_mask = offs_m[:, None] < N
        # A 16-row local-transpose layout is not representable by the current
        # four-warp TLX inference.  Direct register loads preserve the short
        # tile schedule while K/V stream through the resident LDS buffer.
        q_tile = tl.load(Q + qdo_ptrs, mask=qdo_mask, other=0.0)
        do_tile = tl.load(DO + qdo_ptrs, mask=qdo_mask, other=0.0)
        q_t = tl.trans(q_tile)
        do_t = tl.trans(do_tile)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = tl.dot(k_tile, q_t)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = key_mask & (offs_m[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[:, None] <= offs_m[None, :])
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)
        dp_t = tl.dot(v_tile, do_t)
        ds_t = p_t * (dp_t - delta[None, :])
        ds_bf16 = ds_t.to(tl.bfloat16)

        dq_part = tl.dot(tl.trans(ds_bf16), k_tile) * SM_SCALE
        tl.store(DQ + qdo_ptrs, dq_part.to(tl.bfloat16), mask=qdo_mask)

        dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
        dk = tl.dot(ds_bf16, q_tile, dk)

    dk *= SM_SCALE
    tl.store(DK + key_ptrs, dk.to(tl.bfloat16), mask=key_mask)
    tl.store(DV + key_ptrs, dv.to(tl.bfloat16), mask=key_mask)


# TODO: Re-benchmark this async-ring variant after generic TLX register
# allocation and LDS layout inference improve. Promote it only if it becomes
# competitive with the exact or split D128 route; otherwise remove the
# experiment and its flag together with the combined persistent variant.
@triton.jit
def _attn_bwd_dkdv_dq_d128_persistent_pipeline_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Async-pipelined variant of the short persistent D128 experiment.

    K/V stay resident in one LDS tile. Q and dO use a two-slot async ring, so
    the next BM16 global copy overlaps the current score/dP/dQ/dK/dV MFMA
    chain. This is intentionally a separate opt-in experiment: the measured
    split path remains the production default until this schedule wins.
    """
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(D == 128)
    tl.static_assert(0 < N and N <= BLOCK_N)
    batch_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    tensor_base = batch_head * N * D
    key_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    key_mask = offs_n[:, None] < N

    kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_N, D],
    )
    qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [8, 0],
            [4, 0],
            [1, 0],
            [2, 0],
        ],
        [BLOCK_M, D],
    )
    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)
    v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_layout)

    k_token = tlx.async_load(K + key_ptrs, tlx.local_view(k_buffer, 0), mask=key_mask, other=0.0)
    v_token = tlx.async_load(V + key_ptrs, tlx.local_view(v_buffer, 0), mask=key_mask, other=0.0)
    tlx.async_load_commit_group([k_token, v_token])
    kv_wait = tlx.async_load_wait_group(0)
    k_tile = tlx.local_load(tlx.local_view(k_buffer, 0), token=kv_wait)
    v_tile = tlx.local_load(tlx.local_view(v_buffer, 0), token=kv_wait)

    first_m = tl.arange(0, BLOCK_M)
    first_ptrs = tensor_base + first_m[:, None] * D + offs_d[None, :]
    first_mask = first_m[:, None] < N
    q_token = tlx.async_load(Q + first_ptrs, tlx.local_view(q_buffers, 0), mask=first_mask, other=0.0)
    do_token = tlx.async_load(DO + first_ptrs, tlx.local_view(do_buffers, 0), mask=first_mask, other=0.0)
    tlx.async_load_commit_group([q_token, do_token])

    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK_M)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(0, num_m_blocks):
        current_slot = m_block % 2
        next_slot = 1 - current_slot
        next_m = (m_block + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
        next_ptrs = tensor_base + next_m[:, None] * D + offs_d[None, :]
        next_mask = next_m[:, None] < N
        next_q_token = tlx.async_load(
            Q + next_ptrs,
            tlx.local_view(q_buffers, next_slot),
            mask=next_mask,
            other=0.0,
        )
        next_do_token = tlx.async_load(
            DO + next_ptrs,
            tlx.local_view(do_buffers, next_slot),
            mask=next_mask,
            other=0.0,
        )
        tlx.async_load_commit_group([next_q_token, next_do_token])
        qdo_wait = tlx.async_load_wait_group(1)

        q_view = tlx.local_view(q_buffers, current_slot)
        do_view = tlx.local_view(do_buffers, current_slot)
        q_tile = tlx.local_load(q_view, token=qdo_wait)
        do_tile = tlx.local_load(do_view, token=qdo_wait)
        q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait)
        do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait)

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = tl.dot(k_tile, q_t)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = key_mask & (offs_m[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[:, None] <= offs_m[None, :])
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)
        dp_t = tl.dot(v_tile, do_t)
        ds_t = p_t * (dp_t - delta[None, :])
        ds_bf16 = ds_t.to(tl.bfloat16)

        dq_part = tl.dot(tl.trans(ds_bf16), k_tile) * SM_SCALE
        qdo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(DQ + qdo_ptrs, dq_part.to(tl.bfloat16), mask=offs_m[:, None] < N)

        dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
        dk = tl.dot(ds_bf16, q_tile, dk)

    tlx.async_load_wait_group(0)
    dk *= SM_SCALE
    tl.store(DK + key_ptrs, dk.to(tl.bfloat16), mask=key_mask)
    tl.store(DV + key_ptrs, dv.to(tl.bfloat16), mask=key_mask)


@triton.jit
def _attn_bwd_gqa_issue_qdo_async(
    q_outer,
    do_outer,
    lse_outer,
    delta_outer,
    Q,
    DO,
    LSE,
    Delta,
    off_h_kv,
    pid_n,
    step,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTER_M: tl.constexpr,
    ASYNC_LAYOUT: tl.constexpr,
    STATS_ASYNC_LAYOUT: tl.constexpr,
    Q_BATCH_FITS_BUFFER: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Map a compact step to one absolute Q/dO/stats tile and issue it."""
    buffer_base_alignment_bytes: tl.constexpr = 16
    group_size: tl.constexpr = HQ // HK
    n_outer_blocks: tl.constexpr = N // OUTER_M
    first_outer_block = pid_n * (BLOCK_N // OUTER_M) if IS_CAUSAL else 0
    active_outer_blocks = n_outer_blocks - first_outer_block
    total_outer_steps = group_size * active_outer_blocks
    refill_step = step % total_outer_steps
    group_idx = refill_step // active_outer_blocks
    outer_block = first_outer_block + refill_step % active_outer_blocks
    off_h_q = off_h_kv * group_size + group_idx

    outer_slice = tl.arange(0, OUTER_M // BLOCK_M)
    inner_m = tlx.rematerialized_range(0, BLOCK_M, 10, placement=step)
    offs_d = tlx.rematerialized_range(0, D, 11, placement=step)
    if Q_BATCH_FITS_BUFFER:
        offs_m = outer_block * OUTER_M + outer_slice[:, None] * BLOCK_M + inner_m[None, :]
        q_base = off_h_q * N * D
        qdo_offsets = (q_base + offs_m[:, :, None] * D + offs_d[None, None, :]).to(tl.int32)
        stats_base = off_h_q * N
        stats_offsets = (stats_base + outer_block * OUTER_M + tl.arange(0, OUTER_M)).to(tl.int32)
        qdo_offsets = tlx.require_layout(qdo_offsets, ASYNC_LAYOUT, pin=False)
        stats_offsets = tlx.require_layout(stats_offsets, STATS_ASYNC_LAYOUT, pin=False)
        q_token = tlx.buffer_load_to_local(q_outer, Q, qdo_offsets)
        tlx.async_load_commit_group([q_token])
        lse_token = tlx.buffer_load_to_local(lse_outer, LSE, stats_offsets)
        delta_token = tlx.buffer_load_to_local(delta_outer, Delta, stats_offsets)
        do_token = tlx.buffer_load_to_local(do_outer, DO, qdo_offsets)
        tlx.async_load_commit_group([lse_token, delta_token, do_token])
    else:
        offs_m = outer_slice[:, None] * BLOCK_M + inner_m[None, :]
        q_outer_base = (off_h_q.to(tl.int64) * N + outer_block * OUTER_M) * D
        qdo_offsets = (offs_m[:, :, None] * D + offs_d[None, None, :]).to(tl.int32)
        stats_outer_base = off_h_q.to(tl.int64) * N + outer_block * OUTER_M
        stats_offsets = tl.arange(0, OUTER_M).to(tl.int32)
        qdo_offsets = tlx.require_layout(qdo_offsets, ASYNC_LAYOUT, pin=False)
        stats_offsets = tlx.require_layout(stats_offsets, STATS_ASYNC_LAYOUT, pin=False)
        q_outer_ptr = tl.multiple_of(Q + q_outer_base, buffer_base_alignment_bytes)
        do_outer_ptr = tl.multiple_of(DO + q_outer_base, buffer_base_alignment_bytes)
        lse_outer_ptr = tl.multiple_of(LSE + stats_outer_base, buffer_base_alignment_bytes)
        delta_outer_ptr = tl.multiple_of(Delta + stats_outer_base, buffer_base_alignment_bytes)
        q_token = tlx.buffer_load_to_local(q_outer, q_outer_ptr, qdo_offsets)
        tlx.async_load_commit_group([q_token])
        lse_token = tlx.buffer_load_to_local(lse_outer, lse_outer_ptr, stats_offsets)
        delta_token = tlx.buffer_load_to_local(delta_outer, delta_outer_ptr, stats_offsets)
        do_token = tlx.buffer_load_to_local(do_outer, do_outer_ptr, qdo_offsets)
        tlx.async_load_commit_group([lse_token, delta_token, do_token])


@triton.jit
def _attn_bwd_gqa_load_qdo_slice(
    q_tiles,
    do_tiles,
    phase,
):
    """Select one 16-row Q/dO slice without starting wide operand loads."""
    q_slice = tlx.local_view(q_tiles, phase)
    do_slice = tlx.local_view(do_tiles, phase)
    return q_slice, do_slice


@triton.jit
def _attn_bwd_gqa_dv_fragmented_half(
    dv_lhs,
    dv_rhs,
    dv,
    MMA_ND: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    FIRST_HALF: tl.constexpr,
):
    """Update four dV fragments while preserving the native VGPR bank."""
    dv_lhs = tlx.require_layout(dv_lhs, P_ND_LAYOUT, pin=False)
    dv_rhs = tlx.require_layout(dv_rhs, Q_OUT_LAYOUT, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    lhs0 = tlx.extract_slice(dv_lhs, [128, 16], [0, 0])
    lhs1 = tlx.extract_slice(dv_lhs, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(dv_rhs, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(dv_rhs, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(dv_rhs, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(dv_rhs, [16, 32], [0, 96])

    c00 = tlx.extract_slice(dv, [128, 32], [0, 0])
    c10 = tlx.extract_slice(dv, [128, 32], [128, 0])
    c01 = tlx.extract_slice(dv, [128, 32], [0, 32])
    c11 = tlx.extract_slice(dv, [128, 32], [128, 32])
    c02 = tlx.extract_slice(dv, [128, 32], [0, 64])
    c12 = tlx.extract_slice(dv, [128, 32], [128, 64])
    c03 = tlx.extract_slice(dv, [128, 32], [0, 96])
    c13 = tlx.extract_slice(dv, [128, 32], [128, 96])

    if FIRST_HALF:
        c00 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs0,
            c00,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c10 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs0,
            c10,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c01 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs1,
            c01,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c11 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs1,
            c11,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
    else:
        c02 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs2,
            c02,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c12 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs2,
            c12,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c03 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs3,
            c03,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )
        c13 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs3,
            c13,
            accumulator_role="persistent",
            accumulator_register_class="vgpr",
        )

    row0 = tl.cat(
        tl.cat(c00, c01, dim=1),
        tl.cat(c02, c03, dim=1),
        dim=1,
    )
    row1 = tl.cat(
        tl.cat(c10, c11, dim=1),
        tl.cat(c12, c13, dim=1),
        dim=1,
    )
    new_dv = tl.cat(row0, row1, dim=0)
    return tlx.require_layout(new_dv, MMA_ND, pin=False)


@triton.jit
def _attn_bwd_gqa_front(
    dv,
    q_slice,
    do_slice,
    k_buffer,
    v_operand,
    ds_stage,
    lse_values,
    delta_values,
    pid_n,
    step,
    SM_SCALE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
):
    """Apply the optional causal scaled-score mask, then publish current dS to LDS."""
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    v_operand = tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False)
    k_nm = tlx.local_load(
        k_buffer,
        layout=K_NM_LAYOUT,
        relaxed=True,
    )
    q_t = tlx.local_load(
        tlx.local_trans(q_slice),
        layout=QT_LAYOUT,
        relaxed=True,
    )
    scores = tl.dot(
        k_nm,
        q_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    scores = tlx.amd_register_resident(scores, register_class="vgpr", registers_per_group=16)
    if IS_CAUSAL:
        # Mask after scaling so custom zero or negative scales cannot turn an
        # invalid raw-score sentinel into a NaN or a finite probability.
        lse_log2 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse_values],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        lse_full = tlx.require_layout(
            tl.broadcast_to(lse_log2[None, :], (BLOCK_N, BLOCK_M)),
            MMA_NM,
            pin=False,
        )
        scale_full = tlx.require_layout(
            tl.full(
                (BLOCK_N, BLOCK_M),
                SM_SCALE * 1.4426950408889634,
                tl.float32,
            ),
            MMA_NM,
            pin=False,
        )
        scaled_scores = scores * scale_full - lse_full
        n_m_blocks: tl.constexpr = N // BLOCK_M
        m_block = step % n_m_blocks
        query_fragment = m_block - pid_n * (BLOCK_N // BLOCK_M)
        if query_fragment < BLOCK_N // BLOCK_M:
            offs_n = pid_n * BLOCK_N + tlx.rematerialized_range(0, BLOCK_N, 32, placement=step)
            offs_m = m_block * BLOCK_M + tlx.rematerialized_range(0, BLOCK_M, 33, placement=step)
            valid = offs_n[:, None] <= offs_m[None, :]
            valid = tlx.require_layout(valid, MMA_NM, pin=False)
            neg_inf = tlx.require_layout(
                tl.full((BLOCK_N, BLOCK_M), float("-inf"), dtype=tl.float32),
                MMA_NM,
                pin=False,
            )
            scaled_scores = tl.where(valid, scaled_scores, neg_inf)
            scaled_scores = tlx.require_layout(scaled_scores, MMA_NM, pin=False)
    do_t = tlx.local_load(
        tlx.local_trans(do_slice),
        layout=QT_LAYOUT,
        relaxed=True,
    )
    dp = tl.dot(
        v_operand,
        do_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    dp = tlx.amd_register_resident(dp, register_class="vgpr", registers_per_group=16)

    q_out = tlx.local_load(q_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    q_out = tlx.amd_register_resident(q_out, register_class="vgpr", registers_per_group=4)
    do_out = tlx.local_load(do_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    if not IS_CAUSAL:
        # Keep LSE scaling on an independent scalar VALU chain.  Broadcasting
        # the multiply lets LLVM pair an LSE lane with a score fragment in a
        # packed multiply and creates a false cross-fragment dependency.
        lse_log2 = tl.inline_asm_elementwise(
            "v_mul_f32_e32 $0, 0x3fb8aa3b, $1;",
            "=v,v",
            [lse_values],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        # These are soft requirements. Layout propagation does not yet infer
        # the score MFMA layout for broadcast/full operands from elementwise users.
        lse_full = tlx.require_layout(
            tl.broadcast_to(lse_log2[None, :], (BLOCK_N, BLOCK_M)),
            MMA_NM,
            pin=False,
        )
        scale_full = tlx.require_layout(
            tl.full(
                (BLOCK_N, BLOCK_M),
                SM_SCALE * 1.4426950408889634,
                tl.float32,
            ),
            MMA_NM,
            pin=False,
        )
        scaled_scores = scores * scale_full - lse_full
    p = tl.math.exp2(scaled_scores)
    delta_full = tlx.require_layout(
        tl.broadcast_to(delta_values[None, :], (BLOCK_N, BLOCK_M)),
        MMA_NM,
        pin=False,
    )
    ds = p * (dp - delta_full)

    p_nd = tl.reshape(p.to(tl.bfloat16), (2, 2, 2, 2, 16, BLOCK_M))
    p_nd = tl.permute(p_nd, (0, 2, 3, 1, 4, 5))
    p_nd = tl.reshape(p_nd, (BLOCK_N, BLOCK_M))
    p_nd = tlx.require_layout(p_nd, P_ND_LAYOUT, pin=False)
    dv = _attn_bwd_gqa_dv_fragmented_half(
        p_nd,
        do_out,
        dv,
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        True,
    )

    ds_bf16 = ds.to(tl.bfloat16)
    tlx.local_store(ds_stage, tl.trans(ds_bf16))
    ds_nd = tl.reshape(ds_bf16, (2, 2, 2, 2, 16, BLOCK_M))
    ds_nd = tl.permute(ds_nd, (0, 2, 3, 1, 4, 5))
    ds_nd = tl.reshape(ds_nd, (BLOCK_N, BLOCK_M))
    ds_nd = tlx.require_layout(ds_nd, P_ND_LAYOUT, pin=False)
    return dv, ds_nd, q_out, p_nd, do_out


@triton.jit
def _attn_bwd_gqa_store_dq_native(
    dq,
    DQ_ACC,
    off_h_kv,
    step,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
    Q_BATCH_FITS_BUFFER: tl.constexpr,
):
    """Accumulate one dQ partial in the native MFMA ownership."""
    buffer_base_alignment_bytes: tl.constexpr = 16
    group_size: tl.constexpr = HQ // HK
    n_m_blocks: tl.constexpr = N // BLOCK_M
    group_idx = step // n_m_blocks
    m_block = step % n_m_blocks
    off_h_q = off_h_kv * group_size + group_idx
    start_m = m_block * BLOCK_M

    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    dq_scale = tlx.require_layout(
        tl.full((BLOCK_M, D), SM_SCALE, dtype=tl.float32),
        MMA_MD,
        pin=False,
    )
    dq = dq * dq_scale
    local_m = tlx.rematerialized_range(0, BLOCK_M, 14, placement=step)
    offs_d = tlx.rematerialized_range(0, D, 13, placement=step)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    if Q_BATCH_FITS_BUFFER:
        row_base = off_h_q * N + start_m
        swizzled = (row_base * D + ((local_m[:, None] << 1) | d_swizzled[None, :])).to(tl.int32)
        swizzled = tl.max_contiguous(swizzled, [1, 2])
        swizzled = tlx.require_layout(swizzled, MMA_MD, pin=False)
        tlx.buffer_atomic_add(
            DQ_ACC,
            swizzled,
            dq.to(tl.bfloat16),
            sem="relaxed",
            contiguity=2,
        )
    else:
        dq_ptr = tl.multiple_of(
            DQ_ACC + (off_h_q.to(tl.int64) * N + start_m) * D,
            buffer_base_alignment_bytes,
        )
        swizzled = ((local_m[:, None] << 1) | d_swizzled[None, :]).to(tl.int32)
        swizzled = tl.max_contiguous(swizzled, [1, 2])
        swizzled = tlx.require_layout(swizzled, MMA_MD, pin=False)
        tlx.buffer_atomic_add(
            dq_ptr,
            swizzled,
            dq.to(tl.bfloat16),
            sem="relaxed",
            contiguity=2,
        )


@triton.jit
def _attn_bwd_gqa_dq_prefetched(
    prev_ds,
    ds0,
    ds1,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_resident,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    COORDINATE_GROUP: tl.constexpr = None,
):
    """Reduce eight K=32 bands into two independent native dQ chains."""
    ds0 = tlx.require_layout(ds0, DS_MD_LAYOUT, pin=False)
    ds1 = tlx.require_layout(ds1, DS_MD_LAYOUT, pin=False)
    k_resident_lo = tlx.require_layout(k_resident_lo, K_MD_LAYOUT, pin=False)
    k_resident_mid = tlx.require_layout(k_resident_mid, K_MD_LAYOUT, pin=False)
    k_resident_band6 = tlx.require_layout(k_resident_band6, K_MD_LAYOUT, pin=False)
    v_resident = tlx.require_layout(v_resident, V_LAYOUT, pin=False)
    dq0 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)
    dq1 = tlx.zeros((16, 64), tl.float32, layout=MMA_MD)

    ds2 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 64], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k7 = tlx.local_load(
        tlx.local_slice(k_buffer, [224, 0], [32, 128]),
        layout=K_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k00 = tlx.extract_slice(k_resident_lo, [32, 64], [0, 0])
    k01 = tlx.extract_slice(k_resident_lo, [32, 64], [0, 64])
    k10 = tlx.extract_slice(k_resident_lo, [32, 64], [32, 0])
    k11 = tlx.extract_slice(k_resident_lo, [32, 64], [32, 64])
    dq0 = tlx.amd_scheduled_mfma(
        ds0,
        k00,
        dq0,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )
    dq1 = tlx.amd_scheduled_mfma(
        ds0,
        k01,
        dq1,
        resident_operand=1,
        accumulator_role="transient",
        initialize=True,
    )

    ds3 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 96], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    dq0 = tlx.amd_scheduled_mfma(ds1, k10, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds1, k11, dq1, resident_operand=1, accumulator_role="transient")
    k20 = tlx.extract_slice(k_resident_lo, [32, 64], [64, 0])
    k21 = tlx.extract_slice(k_resident_lo, [32, 64], [64, 64])
    dq0 = tlx.amd_scheduled_mfma(ds2, k20, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds2, k21, dq1, resident_operand=1, accumulator_role="transient")

    ds4 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 128], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k30 = tlx.extract_slice(k_resident_lo, [32, 64], [96, 0])
    k31 = tlx.extract_slice(k_resident_lo, [32, 64], [96, 64])
    dq0 = tlx.amd_scheduled_mfma(ds3, k30, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds3, k31, dq1, resident_operand=1, accumulator_role="transient")

    ds5 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 160], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k40 = tlx.extract_slice(k_resident_mid, [32, 64], [0, 0])
    k41 = tlx.extract_slice(k_resident_mid, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds4, k40, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds4, k41, dq1, resident_operand=1, accumulator_role="transient")

    ds6 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 192], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k50 = tlx.extract_slice(k_resident_mid, [32, 64], [32, 0])
    k51 = tlx.extract_slice(k_resident_mid, [32, 64], [32, 64])
    dq0 = tlx.amd_scheduled_mfma(ds5, k50, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds5, k51, dq1, resident_operand=1, accumulator_role="transient")

    ds7 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 224], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    k60 = tlx.extract_slice(k_resident_band6, [32, 64], [0, 0])
    k61 = tlx.extract_slice(k_resident_band6, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds6, k60, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds6, k61, dq1, resident_operand=1, accumulator_role="transient")
    k70 = tlx.extract_slice(k7, [32, 64], [0, 0])
    k71 = tlx.extract_slice(k7, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds7, k70, dq0, resident_operand=1, accumulator_role="transient")
    dq1 = tlx.amd_scheduled_mfma(ds7, k71, dq1, resident_operand=1, accumulator_role="transient")
    dq0, dq1, v_resident = tlx.amd_mfma_commit((dq0, dq1), v_resident)
    dq = tl.join(dq0, dq1)
    dq = tl.permute(dq, (0, 2, 1))
    dq = tl.reshape(dq, (16, 128))
    dq = tlx.require_layout(dq, MMA_MD, pin=False)
    return dq, v_resident


@triton.jit
def _attn_bwd_gqa_dq(
    prev_ds,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_resident,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
    COORDINATE_GROUP: tl.constexpr = None,
):
    """Drain entry when no dK work is available to hide the first dS reads."""
    ds0 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 0], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    ds1 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 32], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
        rematerialize_coordinates_group=COORDINATE_GROUP,
    )
    return _attn_bwd_gqa_dq_prefetched(
        prev_ds,
        ds0,
        ds1,
        k_buffer,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_resident,
        MMA_MD,
        DS_MD_LAYOUT,
        K_MD_LAYOUT,
        V_LAYOUT,
        COORDINATE_GROUP,
    )


@triton.jit
def _attn_bwd_gqa_dk_fragmented_prefetch(
    dk_lhs,
    dk_rhs,
    dk,
    prev_ds,
    MMA_ND: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
):
    """Update fragmented dK while prefetching the first two dQ bands."""
    dk_lhs = tlx.require_layout(dk_lhs, P_ND_LAYOUT, pin=False)
    dk_rhs = tlx.require_layout(dk_rhs, Q_OUT_LAYOUT, pin=False)
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    lhs0 = tlx.extract_slice(dk_lhs, [128, 16], [0, 0])
    lhs1 = tlx.extract_slice(dk_lhs, [128, 16], [128, 0])
    rhs0 = tlx.extract_slice(dk_rhs, [16, 32], [0, 0])
    rhs1 = tlx.extract_slice(dk_rhs, [16, 32], [0, 32])
    rhs2 = tlx.extract_slice(dk_rhs, [16, 32], [0, 64])
    rhs3 = tlx.extract_slice(dk_rhs, [16, 32], [0, 96])

    c00 = tlx.extract_slice(dk, [128, 32], [0, 0])
    c10 = tlx.extract_slice(dk, [128, 32], [128, 0])
    c01 = tlx.extract_slice(dk, [128, 32], [0, 32])
    c11 = tlx.extract_slice(dk, [128, 32], [128, 32])
    c02 = tlx.extract_slice(dk, [128, 32], [0, 64])
    c12 = tlx.extract_slice(dk, [128, 32], [128, 64])
    c03 = tlx.extract_slice(dk, [128, 32], [0, 96])
    c13 = tlx.extract_slice(dk, [128, 32], [128, 96])

    c00 = tlx.amd_scheduled_mfma(lhs0, rhs0, c00, accumulator_role="persistent")
    c10 = tlx.amd_scheduled_mfma(lhs1, rhs0, c10, accumulator_role="persistent")
    ds0 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 0], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    c01 = tlx.amd_scheduled_mfma(lhs0, rhs1, c01, accumulator_role="persistent")
    c11 = tlx.amd_scheduled_mfma(lhs1, rhs1, c11, accumulator_role="persistent")
    c02 = tlx.amd_scheduled_mfma(lhs0, rhs2, c02, accumulator_role="persistent")
    c12 = tlx.amd_scheduled_mfma(lhs1, rhs2, c12, accumulator_role="persistent")
    ds1 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 32], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    c03 = tlx.amd_scheduled_mfma(lhs0, rhs3, c03, accumulator_role="persistent")
    c13 = tlx.amd_scheduled_mfma(lhs1, rhs3, c13, accumulator_role="persistent")

    row0 = tl.cat(
        tl.cat(c00, c01, dim=1),
        tl.cat(c02, c03, dim=1),
        dim=1,
    )
    row1 = tl.cat(
        tl.cat(c10, c11, dim=1),
        tl.cat(c12, c13, dim=1),
        dim=1,
    )
    new_dk = tl.cat(row0, row1, dim=0)
    new_dk = tlx.require_layout(new_dk, MMA_ND, pin=False)
    return new_dk, ds0, ds1


@triton.jit
def _attn_bwd_gqa_bridge(
    prev_ds,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_resident,
    dk_lhs,
    dk_rhs,
    dk,
    MMA_ND: tl.constexpr,
    MMA_MD: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    V_LAYOUT: tl.constexpr,
):
    """Interleave independent current-dK and previous-dQ chains."""
    new_dk, ds0, ds1 = _attn_bwd_gqa_dk_fragmented_prefetch(
        dk_lhs,
        dk_rhs,
        dk,
        prev_ds,
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        DS_MD_LAYOUT,
    )
    dq, v_resident = _attn_bwd_gqa_dq_prefetched(
        prev_ds,
        ds0,
        ds1,
        k_buffer,
        k_resident_lo,
        k_resident_mid,
        k_resident_band6,
        v_resident,
        MMA_MD,
        DS_MD_LAYOUT,
        K_MD_LAYOUT,
        V_LAYOUT,
    )
    return new_dk, dq, v_resident


@triton.jit
def _attn_bwd_gqa_phase(
    q_tiles,
    do_tiles,
    lse_tiles,
    delta_tiles,
    ds_buffers,
    k_buffer,
    k_resident_lo,
    k_resident_mid,
    k_resident_band6,
    v_operand,
    dk,
    dv,
    DQ_ACC,
    off_h_kv,
    pid_n,
    outer_step,
    phase,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    MMA_MD: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
    DS_MD_LAYOUT: tl.constexpr,
    K_MD_LAYOUT: tl.constexpr,
    DIRECT_DK: tl.constexpr,
    REDIRECT_DUMMY_DQ: tl.constexpr,
    Q_BATCH_FITS_BUFFER: tl.constexpr,
):
    """Run one absolute outer-step phase with direct dK or the lagged bridge.

    ``REDIRECT_DUMMY_DQ`` is used only by MHA phase 0, whose seeded zero dS
    produces a harmless dQ before any real previous phase exists.
    """
    tl.static_assert(not (DIRECT_DK and REDIRECT_DUMMY_DQ))
    step = outer_step * 4 + phase
    cur_stage = phase % 2
    q_slice, do_slice = _attn_bwd_gqa_load_qdo_slice(
        q_tiles,
        do_tiles,
        phase,
    )
    # Re-anchor lane/warp coordinates at the adjacent statistics loads and
    # share the copy. The group number is only a local equivalence tag, not a
    # tuning parameter. Entry coordinates otherwise cross the register-heavy
    # phase; rematerializing each load independently duplicates the anchors.
    stats_coordinate_group: tl.constexpr = 20
    lse_values = tlx.local_load(
        tlx.local_view(lse_tiles, phase),
        relaxed=True,
        rematerialize_coordinates_group=stats_coordinate_group,
    )
    delta_values = tlx.local_load(
        tlx.local_view(delta_tiles, phase),
        relaxed=True,
        rematerialize_coordinates_group=stats_coordinate_group,
    )
    dv, dk_lhs, dk_rhs, dv_lhs, dv_rhs = _attn_bwd_gqa_front(
        dv,
        q_slice,
        do_slice,
        k_buffer,
        v_operand,
        tlx.local_view(ds_buffers, cur_stage),
        lse_values,
        delta_values,
        pid_n,
        step,
        SM_SCALE,
        N,
        BLOCK_M,
        BLOCK_N,
        IS_CAUSAL,
        MMA_NM,
        MMA_ND,
        K_NM_LAYOUT,
        QT_LAYOUT,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
    )
    dv = _attn_bwd_gqa_dv_fragmented_half(
        dv_lhs,
        dv_rhs,
        dv,
        MMA_ND,
        P_ND_LAYOUT,
        Q_OUT_LAYOUT,
        False,
    )
    if DIRECT_DK:
        dk_lhs = tlx.require_layout(dk_lhs, P_ND_LAYOUT, pin=False)
        dk_rhs = tlx.require_layout(dk_rhs, Q_OUT_LAYOUT, pin=False)
        dk = tlx.require_layout(dk, MMA_ND, pin=False)
        dk = tl.dot(dk_lhs, dk_rhs, dk)
        dk = tlx.require_layout(dk, MMA_ND, pin=False)
    else:
        prev_stage = 1 - cur_stage
        dk, dq, v_operand = _attn_bwd_gqa_bridge(
            tlx.local_view(ds_buffers, prev_stage),
            k_buffer,
            k_resident_lo,
            k_resident_mid,
            k_resident_band6,
            v_operand,
            dk_lhs,
            dk_rhs,
            dk,
            MMA_ND,
            MMA_MD,
            DS_MD_LAYOUT,
            K_MD_LAYOUT,
            P_ND_LAYOUT,
            Q_OUT_LAYOUT,
            K_NM_LAYOUT,
        )
        if REDIRECT_DUMMY_DQ:
            # Full attention starts at step zero.  Causal workgroups after KV
            # tile zero start later, so keep the seeded dummy atomic inside
            # this workgroup's first active query tile instead of touching the
            # preceding (wholly masked) tile.
            first_active_dq_step = pid_n * (BLOCK_N // BLOCK_M) if IS_CAUSAL else 0
            dq_step = tl.maximum(step - 1, first_active_dq_step)
        else:
            dq_step = step - 1
        _attn_bwd_gqa_store_dq_native(
            dq,
            DQ_ACC,
            off_h_kv,
            dq_step,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            MMA_MD,
            Q_BATCH_FITS_BUFFER,
        )
    dk = tlx.require_layout(dk, MMA_ND, pin=False)
    dv = tlx.require_layout(dv, MMA_ND, pin=False)
    v_operand = tlx.require_layout(v_operand, K_NM_LAYOUT, pin=False)
    tl.debug_barrier()
    return dk, dv, v_operand


@triton.jit
def _attn_bwd_dkdv_dq_d128_gqa_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ_ACC,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Four-wave outer64 causal/full GQA backward bridge."""
    tl.static_assert(D == 128)
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(N % 256 == 0)
    tl.static_assert(HQ % HK == 0)

    # Keep adjacent workgroups on different KV heads for the same batch and
    # N tile to improve the dQ atomic/cache locality of the GQA bridge.
    off_h_kv = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_z = tl.program_id(2)
    OUTER_M: tl.constexpr = 64

    # AMD buffer instructions use signed 32-bit byte offsets. Rebase every
    # global tensor to this program's batch with 64-bit pointer arithmetic;
    # individual Q and KV tiles are rebased again below so large head counts
    # cannot push a valid access past the 2-GiB buffer-offset boundary.
    q_batch_base = off_z.to(tl.int64) * HQ * N * D
    kv_batch_base = off_z.to(tl.int64) * HK * N * D
    stats_batch_base = off_z.to(tl.int64) * HQ * N
    Q = Q + q_batch_base
    DO = DO + q_batch_base
    DQ_ACC = DQ_ACC + q_batch_base
    LSE = LSE + stats_batch_base
    Delta = Delta + stats_batch_base

    # Buffer offsets are signed 32-bit byte offsets. Preserve the original
    # batch-relative address schedule whenever a complete Q or KV batch fits;
    # exceptionally large head counts use per-tile 64-bit pointer rebasing.
    max_bf16_buffer_elements: tl.constexpr = 1 << 30
    buffer_base_alignment_bytes: tl.constexpr = 16
    q_batch_fits_buffer: tl.constexpr = HQ * N * D <= max_bf16_buffer_elements
    kv_batch_fits_buffer: tl.constexpr = HK * N * D <= max_bf16_buffer_elements

    # Native score/dP, persistent dK/dV, and delayed-dQ ownerships.
    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    k_nm_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_nd_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_out_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    ds_md_layout: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_md_layout: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    # Inverse producer layouts for one 64-row Q/dO outer tile and K tile.
    qdo_async_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2),
        ),
        stride=(
            (8, 16, 32, 128, 64, 512, 256, 1024),
            (1, 2, 4, 2048, 4096),
        ),
    )
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    # Native ownership after inverse-rotating the persistent dK/dV MFMA
    # accumulators.  Store from this layout directly: converting back to the
    # async-load layout lowers through a full LDS write/read transpose.
    kv_native_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (128, 256, 512, 1024, 8192, 4, 2048, 4096),
            (1, 2, 8, 16, 32, 64, 16384),
        ),
    )

    qdo_smem_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 4],
            [0, 0, 8],
            [0, 0, 16],
            [0, 0, 32],
            [0, 1, 0],
            [0, 0, 64],
            [0, 4, 0],
            [0, 2, 0],
            [0, 8, 0],
            [1, 0, 0],
            [2, 0, 0],
        ],
        [OUTER_M // BLOCK_M, BLOCK_M, D],
    ))
    qdo_slice_smem_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32), (1024, 16)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [0, 64],
            [4, 0],
            [2, 0],
            [8, 0],
        ],
        [BLOCK_M, D],
    ))
    stats_smem_layout: tl.constexpr = (tlx.shared_linear_layout_encoding(
        offset_bases=[
            [1],
            [2],
            [4],
            [8],
            [16],
            [32],
        ],
        block_bases=[],
        alignment=16,
    ))
    stats_slice_smem_layout: tl.constexpr = (tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8]],
        block_bases=[],
        alignment=16,
    ))
    ds_smem_layout: tl.constexpr = (tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
            [4, 0],
            [0, 8],
            [8, 0],
            [0, 32],
            [0, 16],
            [0, 4],
            [0, 64],
            [0, 128],
        ],
        [BLOCK_M, BLOCK_N],
    ))
    # Preserve the existing TLX physical K image: a direct-to-LDS rank-3
    # producer is reinterpreted as the bank-rotated rank-2 dQ consumer.
    k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 4],
            [0, 1, 0],
            [0, 2, 0],
            [0, 4, 0],
            [0, 8, 0],
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [8, 0, 0],
            [16, 0, 0],
            [32, 0, 0],
            [64, 0, 0],
            [128, 0, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    k_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 64],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    k_raw_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 8, 2)),
        stride=((8, 512), (1, 2048, 16384)),
    )

    k_raw_buffer = tlx.local_alloc(
        (BLOCK_N, D // 8, 8),
        tl.bfloat16,
        1,
        layout=k_raw_smem_layout,
    )
    k_buffer = tlx.local_reinterpret(
        tlx.local_view(k_raw_buffer, 0),
        tl.bfloat16,
        [BLOCK_N, D],
        layout=k_smem_layout,
    )
    q_buffers = tlx.local_alloc(
        (OUTER_M // BLOCK_M, BLOCK_M, D),
        tl.bfloat16,
        2,
        layout=qdo_smem_layout,
    )
    do_buffers = tlx.local_alloc(
        (OUTER_M // BLOCK_M, BLOCK_M, D),
        tl.bfloat16,
        2,
        layout=qdo_smem_layout,
    )
    lse_buffers = tlx.local_alloc(
        (OUTER_M, ),
        tl.float32,
        2,
        layout=stats_smem_layout,
    )
    delta_buffers = tlx.local_alloc(
        (OUTER_M, ),
        tl.float32,
        2,
        layout=stats_smem_layout,
    )
    ds_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.bfloat16,
        2,
        layout=ds_smem_layout,
    )

    # K direct-to-LDS group.
    raw_n = tl.arange(0, BLOCK_N)
    raw_dg = tl.arange(0, D // 8)
    raw_v = tl.arange(0, 8)
    k_phys = raw_n[:, None, None] * D + raw_dg[None, :, None] * 8
    k_d_base = ((k_phys & 0x8) | (((k_phys >> 9) & 0x3) << 4) | ((((k_phys >> 4) ^ (k_phys >> 8)) & 0x1) << 6))
    k_n = (((k_phys >> 5) & 0x7) | (((k_phys >> 8) & 0x1) << 3) | (((k_phys >> 11) & 0xf) << 4))
    if kv_batch_fits_buffer:
        K = K + kv_batch_base
        V = V + kv_batch_base
        DK = DK + kv_batch_base
        DV = DV + kv_batch_base
        kv_offset_base = off_h_kv * N * D
        kv_tile_n = pid_n * BLOCK_N
    else:
        kv_program_base = kv_batch_base + (off_h_kv.to(tl.int64) * N + pid_n.to(tl.int64) * BLOCK_N) * D
        # The tile offsets are aligned by D=128 elements. Preserve a concrete
        # byte-alignment proof so direct-to-LDS vectorization remains legal.
        K = tl.multiple_of(K + kv_program_base, buffer_base_alignment_bytes)
        V = tl.multiple_of(V + kv_program_base, buffer_base_alignment_bytes)
        DK = tl.multiple_of(DK + kv_program_base, buffer_base_alignment_bytes)
        DV = tl.multiple_of(DV + kv_program_base, buffer_base_alignment_bytes)
        kv_offset_base = 0
        kv_tile_n = 0
    k_offsets = kv_offset_base + (kv_tile_n + k_n) * D + k_d_base + raw_v[None, None, :]
    k_offsets = tl.multiple_of(k_offsets, [1, 1, 8])
    k_offsets = tl.max_contiguous(k_offsets, [1, 1, 8])
    k_offsets = tlx.require_layout(k_offsets.to(tl.int32), k_raw_async_layout, pin=False)
    k_token = tlx.buffer_load_to_local(tlx.local_view(k_raw_buffer, 0), K, k_offsets)
    tlx.async_load_commit_group([k_token])

    _attn_bwd_gqa_issue_qdo_async(
        tlx.local_view(q_buffers, 0),
        tlx.local_view(do_buffers, 0),
        tlx.local_view(lse_buffers, 0),
        tlx.local_view(delta_buffers, 0),
        Q,
        DO,
        LSE,
        Delta,
        off_h_kv,
        pid_n,
        0,
        HQ,
        HK,
        N,
        D,
        BLOCK_M,
        BLOCK_N,
        OUTER_M,
        qdo_async_layout,
        stats_async_layout,
        q_batch_fits_buffer,
        IS_CAUSAL,
    )
    _attn_bwd_gqa_issue_qdo_async(
        tlx.local_view(q_buffers, 1),
        tlx.local_view(do_buffers, 1),
        tlx.local_view(lse_buffers, 1),
        tlx.local_view(delta_buffers, 1),
        Q,
        DO,
        LSE,
        Delta,
        off_h_kv,
        pid_n,
        1,
        HQ,
        HK,
        N,
        D,
        BLOCK_M,
        BLOCK_N,
        OUTER_M,
        qdo_async_layout,
        stats_async_layout,
        q_batch_fits_buffer,
        IS_CAUSAL,
    )
    initial_wait = tlx.async_load_wait_group(2)
    tl.debug_barrier()

    offs_n = kv_tile_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    v_offsets = kv_offset_base + offs_n[:, None] * D + offs_d[None, :]
    v_offsets = tlx.require_layout(v_offsets.to(tl.int32), k_nm_layout, pin=False)
    v_operand = tlx.buffer_load(V, v_offsets)
    v_operand = tlx.require_layout(v_operand, k_nm_layout, pin=False)

    k_resident_lo = tlx.local_load(
        tlx.local_slice(k_buffer, [0, 0], [128, D]),
        token=initial_wait,
        layout=k_md_layout,
        relaxed=True,
    )
    k_resident_mid = tlx.local_load(
        tlx.local_slice(k_buffer, [128, 0], [64, D]),
        token=initial_wait,
        layout=k_md_layout,
        relaxed=True,
    )
    k_resident_band6 = tlx.local_load(
        tlx.local_slice(k_buffer, [192, 0], [32, D]),
        token=initial_wait,
        layout=k_md_layout,
        relaxed=True,
    )
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    n_outer_blocks: tl.constexpr = N // OUTER_M
    first_outer_block = pid_n * (BLOCK_N // OUTER_M) if IS_CAUSAL else 0
    active_outer_blocks = n_outer_blocks - first_outer_block
    total_outer_steps = (HQ // HK) * active_outer_blocks
    continuous_bridge: tl.constexpr = HQ == HK
    if continuous_bridge:
        # Seed delayed dQ so MHA phase 0 can use the steady bridge.  The dummy
        # zero dQ is redirected to the first tile and leaves it unchanged.
        tlx.local_store(
            tlx.local_view(ds_buffers, 1),
            tl.zeros((BLOCK_M, BLOCK_N), tl.bfloat16),
        )
        tl.debug_barrier()

    # MHA carries delayed dQ across outer64 boundaries.  Grouped-query shapes
    # retain the per-group prologue/drain, which better overlaps Q/dO refill.
    for outer_step in tl.range(0, total_outer_steps, loop_unroll_factor=1):
        outer_stage = outer_step % 2
        group_idx = outer_step // active_outer_blocks
        outer_block = first_outer_block + outer_step % active_outer_blocks
        global_outer_step = group_idx * n_outer_blocks + outer_block
        q_outer = tlx.local_view(q_buffers, outer_stage)
        do_outer = tlx.local_view(do_buffers, outer_stage)
        lse_outer = tlx.local_view(lse_buffers, outer_stage)
        delta_outer = tlx.local_view(delta_buffers, outer_stage)
        q_tiles = tlx.local_reinterpret(
            q_outer,
            tl.bfloat16,
            [4, BLOCK_M, D],
            layout=qdo_slice_smem_layout,
        )
        do_tiles = tlx.local_reinterpret(
            do_outer,
            tl.bfloat16,
            [4, BLOCK_M, D],
            layout=qdo_slice_smem_layout,
        )
        lse_tiles = tlx.local_reinterpret(
            lse_outer,
            tl.float32,
            [4, BLOCK_M],
            layout=stats_slice_smem_layout,
        )
        delta_tiles = tlx.local_reinterpret(
            delta_outer,
            tl.float32,
            [4, BLOCK_M],
            layout=stats_slice_smem_layout,
        )
        # Phase 0 publishes stage 0.  MHA consumes seeded/prior stage 1;
        # grouped-query shapes use direct dK and drain at the outer boundary.
        phase0: tl.constexpr = 0
        dk, dv, v_operand = _attn_bwd_gqa_phase(
            q_tiles,
            do_tiles,
            lse_tiles,
            delta_tiles,
            ds_buffers,
            k_buffer,
            k_resident_lo,
            k_resident_mid,
            k_resident_band6,
            v_operand,
            dk,
            dv,
            DQ_ACC,
            off_h_kv,
            pid_n,
            global_outer_step,
            phase0,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
            IS_CAUSAL,
            mma_nm,
            mma_nd,
            mma_md,
            k_nm_layout,
            qt_layout,
            p_nd_layout,
            q_out_layout,
            ds_md_layout,
            k_md_layout,
            not continuous_bridge,
            continuous_bridge,
            q_batch_fits_buffer,
        )

        for phase in tl.range(1, 4, loop_unroll_factor=1):
            dk, dv, v_operand = _attn_bwd_gqa_phase(
                q_tiles,
                do_tiles,
                lse_tiles,
                delta_tiles,
                ds_buffers,
                k_buffer,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                dk,
                dv,
                DQ_ACC,
                off_h_kv,
                pid_n,
                global_outer_step,
                phase,
                SM_SCALE,
                HQ,
                HK,
                N,
                D,
                BLOCK_M,
                BLOCK_N,
                IS_CAUSAL,
                mma_nm,
                mma_nd,
                mma_md,
                k_nm_layout,
                qt_layout,
                p_nd_layout,
                q_out_layout,
                ds_md_layout,
                k_md_layout,
                False,
                False,
                q_batch_fits_buffer,
            )

        _attn_bwd_gqa_issue_qdo_async(
            q_outer,
            do_outer,
            lse_outer,
            delta_outer,
            Q,
            DO,
            LSE,
            Delta,
            off_h_kv,
            pid_n,
            outer_step + 2,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
            OUTER_M,
            qdo_async_layout,
            stats_async_layout,
            q_batch_fits_buffer,
            IS_CAUSAL,
        )
        if not continuous_bridge:
            # Grouped-query reuse makes this drain useful work while the next
            # Q/dO/stats refill is in flight.
            dq, v_operand = _attn_bwd_gqa_dq(
                tlx.local_view(ds_buffers, 1),
                k_buffer,
                k_resident_lo,
                k_resident_mid,
                k_resident_band6,
                v_operand,
                mma_md,
                ds_md_layout,
                k_md_layout,
                k_nm_layout,
            )
            _attn_bwd_gqa_store_dq_native(
                dq,
                DQ_ACC,
                off_h_kv,
                global_outer_step * 4 + 3,
                SM_SCALE,
                HQ,
                HK,
                N,
                D,
                BLOCK_M,
                mma_md,
                q_batch_fits_buffer,
            )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()

    if continuous_bridge:
        # Only the last MHA dS has no following dK with which to braid dQ.
        # Re-anchor this standalone drain after the causal outer loop; 21 is
        # only a local equivalence tag shared by its LDS loads.
        dq, v_operand = _attn_bwd_gqa_dq(
            tlx.local_view(ds_buffers, 1),
            k_buffer,
            k_resident_lo,
            k_resident_mid,
            k_resident_band6,
            v_operand,
            mma_md,
            ds_md_layout,
            k_md_layout,
            k_nm_layout,
            21 if IS_CAUSAL else None,
        )
        _attn_bwd_gqa_store_dq_native(
            dq,
            DQ_ACC,
            off_h_kv,
            N // BLOCK_M - 1,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            mma_md,
            q_batch_fits_buffer,
        )
    tlx.async_load_wait_group(0)

    # Store the unique dK/dV tile after every query head in the group has
    # contributed.
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    dk = tl.reshape(dk, (2, 2, 2, 2, 16, D))
    dk = tl.permute(dk, (0, 3, 1, 2, 4, 5))
    dk = tl.reshape(dk, (BLOCK_N, D))
    dk = tlx.require_layout(dk, kv_native_layout, pin=False)
    dk = dk * SM_SCALE
    store_n = kv_tile_n + tlx.rematerialized_range(0, BLOCK_N, 30)
    store_d = tlx.rematerialized_range(0, D, 31)
    key_offsets = kv_offset_base + store_n[:, None] * D + store_d[None, :]
    key_offsets = tlx.require_layout(key_offsets.to(tl.int32), kv_native_layout, pin=False)
    tlx.buffer_store(dk.to(tl.bfloat16), DK, key_offsets)

    # dK is dead before constructing the dV view, so its temporary registers
    # are available to the dV epilogue.
    dv = tl.reshape(dv, (2, 2, 2, 2, 16, D))
    dv = tl.permute(dv, (0, 3, 1, 2, 4, 5))
    dv = tl.reshape(dv, (BLOCK_N, D))
    dv = tlx.require_layout(dv, kv_native_layout, pin=False)
    tlx.buffer_store(dv.to(tl.bfloat16), DV, key_offsets)


@triton.jit
def _attn_bwd_dq_native_convert_kernel(
    DQ_ACC,
    DQ,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Restore the logical [M,D] order of native-swizzled BF16 dQ."""
    native_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (256, 512, 1024, 8, 2, 2048, 16, 32),
            (1, 128, 4, 64, 4096, 8192),
        ),
    )
    store_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (256, 512, 1024, 8, 128, 2048, 16, 32),
            (1, 2, 4, 64, 4096, 8192),
        ),
    )
    pid_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    max_bf16_buffer_elements: tl.constexpr = 1 << 30
    head_fits_buffer: tl.constexpr = N * D <= max_bf16_buffer_elements
    tensor_base = batch_head.to(tl.int64) * N * D
    if head_fits_buffer:
        tile_m_base = pid_m * BLOCK_M
    else:
        tensor_base += pid_m.to(tl.int64) * BLOCK_M * D
        tile_m_base = 0
    DQ_ACC = DQ_ACC + tensor_base
    DQ = DQ + tensor_base
    native_m = tile_m_base + tl.arange(0, BLOCK_M)
    native_d = tl.arange(0, D)
    local_m = native_m & 15
    tile_m = native_m - local_m
    d_swizzled = ((native_d & 1)
                  | ((native_d & 2) << 6)
                  | ((native_d & 12) << 3)
                  | ((native_d & 48) << 5)
                  | ((native_d & 64) << 2))
    native_offsets = (tile_m[:, None] * D + (local_m[:, None] << 1) + d_swizzled[None, :]).to(tl.int32)
    native_offsets = tl.max_contiguous(native_offsets, [1, 4])
    # Reading the swizzled accumulator needs native MFMA ownership, but this
    # requirement may remain soft and be absorbed by layout propagation.
    native_offsets = tlx.require_layout(native_offsets, native_layout, pin=False)
    values = tlx.buffer_load(DQ_ACC, native_offsets, contiguity=4)
    # Keep this one hard: softening the native-to-store conversion makes the
    # current compiler materialize the transpose through 32 KiB of LDS.
    values = tlx.require_layout(values, store_layout, pin=True)

    store_m = tile_m_base + tl.arange(0, BLOCK_M)
    store_d = tl.arange(0, D)
    store_offsets = (store_m[:, None] * D + store_d[None, :]).to(tl.int32)
    tl.store(DQ + store_offsets, values)


@triton.jit
def _attn_bwd_dkdv_dq_d128_exact_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCHEDULED_MFMA: tl.constexpr,
):
    """Four-warp MFMA/LDS port of Gluon's short D128 owner.

    This is deliberately a narrow BF16 kernel.  The LDS K/V tile and the
    BM16 Q/dO ring use the same eight-value direct-to-LDS ownership as the
    CDNA4 Gluon kernel.  Explicit MFMA and dot-operand layouts keep score,
    dP, dQ, dK, and dV in their native wave ownership; dS is exchanged through
    LDS so no register transpose is needed between the dQ and dK consumers.
    """
    tl.static_assert(BLOCK_M == 16)
    tl.static_assert(BLOCK_N == 256)
    tl.static_assert(D == 128)
    tl.static_assert(N == 200)
    batch_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_d_half = tl.arange(0, D // 2)
    tensor_base = batch_head * N * D
    key_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    key_mask = offs_n[:, None] < N

    # Four warps × 64 lanes own 256 elements of the BM16 Q/dO tile.  The
    # value bits describe the eight contiguous BF16 elements copied by each
    # lane; the first six thread bits are lane bits and the last two are warp
    # bits on gfx950.
    qdo_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2)),
        stride=((8, 16, 32, 64, 1024, 512, 128, 256), (1, 2, 4)),
    )
    # K/V use the same cooperative ownership, with seven value bits covering
    # the full 256x128 tile.
    kv_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
        stride=((8, 16, 32, 64, 2048, 4096, 128, 256), (1, 2, 4, 1024, 512, 8192, 16384)),
    )

    qdo_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [8, 0],
            [4, 0],
            [1, 0],
            [2, 0],
        ],
        [BLOCK_M, D],
    )
    v_pad: tl.constexpr = 16 if IS_CAUSAL else 32
    kv_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, v_pad)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_N, D],
    )
    # The MHA N=200 path uses Gluon's bank-rotated physical K image.  Direct
    # global-to-LDS writes target a rank-3 [N, D/8, 8] image; the subsequent
    # reinterpretation exposes the same bytes as a rank-2 [N, D] tile whose
    # SharedLinear mapping gives the MFMA transpose-read path its bank rotation.
    k_raw_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 4],
            [0, 1, 0],
            [0, 2, 0],
            [0, 4, 0],
            [0, 8, 0],
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [8, 0, 0],
            [16, 0, 0],
            [32, 0, 0],
            [64, 0, 0],
            [128, 0, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    k_tiled_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 64],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    # Four warps cooperatively populate the rank-3 image.  The shape/stride
    # expands to the same register/lane/warp bases as Gluon's
    # DistributedLinearLayout, including the final eight-contiguous BF16
    # values owned by each lane.
    k_raw_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 8, 2)),
        stride=((8, 512), (1, 2048, 16384)),
    )
    # dS is written as [N,M] and read both in its native ownership for dK and
    # through a descriptor transpose for dQ.  The interval pad keeps the two
    # consumers bank-disjoint, matching Gluon's 0x120-byte stripe pitch.
    ds_smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(128, 16)],
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 4],
            [0, 8],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
        ],
        [BLOCK_N, BLOCK_M],
    )

    # These are the exact CDNA4 MFMA ownerships used by Gluon.  The score and
    # dK/dV paths distribute waves over N, while dQ distributes them over D.
    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    # Q and dO share a two-stage producer cadence. One complete async group
    # remains ahead of the current score/dP consumers without extending the
    # causal Q-address live range by another tile.
    Q_STAGES: tl.constexpr = 2
    Q_LOOKAHEAD: tl.constexpr = 1
    # A direct resident V load removes one LDS round trip and wins for the
    # full-attention schedule. In the causal schedule its VMEM burst competes
    # with the Q/dO producer, so retain async V-to-LDS there.
    DIRECT_V: tl.constexpr = not IS_CAUSAL
    # Preserve each large-dK pair / small-dQ pair as a distinct scheduling
    # group. Mask zero prevents LLVM from dissolving the intended LDS-read,
    # partial-wait, and MFMA grouping while still emitting no runtime barrier.
    MFMA_BRAID_BARRIER_MASK: tl.constexpr = 0
    k_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    v_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    dot_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    pt_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    do_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    dst_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=8)
    q_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=8)
    ds_op0_md: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=8)
    k_op1_md: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=8)

    k_raw_buffer = tlx.local_alloc((BLOCK_N, D // 8, 8), tl.bfloat16, 1, layout=k_raw_smem_layout)
    k_buffer = tlx.local_reinterpret(tlx.local_view(k_raw_buffer, 0), tl.bfloat16, [BLOCK_N, D],
                                     layout=k_tiled_smem_layout)
    if not DIRECT_V:
        v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_smem_layout)
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, Q_STAGES, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    if DIRECT_V:
        # V is loaded directly into its persistent score/dP MFMA operand.
        # Only the two alternating dS stages need this allocation.
        ds_buffers = tlx.local_alloc(
            (BLOCK_N, BLOCK_M),
            tl.bfloat16,
            2,
            layout=ds_smem_layout,
        )
    else:
        # V is consumed into the persistent v_nm register value before the
        # first dS publish. Reuse its now-dead 64-KiB LDS image for both dS
        # stages, so lagging dQ does not increase the allocation.
        ds_buffers = tlx.local_alloc(
            (BLOCK_N, BLOCK_M),
            tl.bfloat16,
            2,
            layout=ds_smem_layout,
            reuse=v_buffer,
        )

    # Masked direct-to-LDS copies leave OOB rows untouched.  Clear only the
    # aligned K suffix needed by the N=200 tile; this preserves the valid K
    # rows while preventing undefined LDS values from entering dQ MFMA.
    # The current TLX AMD buffer-op conversion cannot lower a rank-3
    # memdesc_subslice in the same pipeline as a direct-to-LDS copy.  Keep the
    # physical image initialized with the same lane ownership for now; the
    # valid K rows are overwritten by the async copy below and the aligned tail
    # remains zero without relying on masked-copy fallback semantics.
    tlx.local_store(
        tlx.local_view(k_raw_buffer, 0),
        tlx.zeros((BLOCK_N, D // 8, 8), tl.bfloat16, layout=k_raw_async_layout),
    )
    if not DIRECT_V:
        tlx.local_store(
            tlx.local_view(v_buffer, 0),
            tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=kv_async_layout),
        )
    tl.debug_barrier()
    # Invert the physical XOR view used by k_tiled_smem_layout.  The final
    # value axis remains eight contiguous BF16 elements, so the compiler can
    # lower one 128-bit direct-to-LDS transaction per lane.
    raw_n = tl.arange(0, BLOCK_N)
    raw_dg = tl.arange(0, D // 8)
    raw_v = tl.arange(0, 8)
    k_phys = raw_n[:, None, None] * D + raw_dg[None, :, None] * 8
    k_d_base = ((k_phys & 0x8) | (((k_phys >> 9) & 0x3) << 4) | ((((k_phys >> 4) ^ (k_phys >> 8)) & 0x1) << 6))
    k_n = (((k_phys >> 5) & 0x7) | (((k_phys >> 8) & 0x1) << 3) | (((k_phys >> 11) & 0xf) << 4))
    k_offsets = (tensor_base + k_n * D + k_d_base + raw_v[None, None, :])
    # The last raw axis is the eight-element BF16 vector issued by each lane.
    # Preserve Gluon's alignment/contiguity facts before lowering the offsets
    # to the direct-to-LDS buffer operation; without them AxisInfo can split
    # the vector and leave an unresolved descriptor conversion in LLIR.
    k_offsets = tl.multiple_of(k_offsets, [1, 1, 8])
    k_offsets = tl.max_contiguous(k_offsets, [1, 1, 8])
    # These exact-path requirements stay soft: they select the transient
    # async/MFMA ownership but do not pin it through the loop-carried dot chain.
    # Fixed output ownership below continues to use the pin=True default.
    k_offsets = tlx.require_layout(k_offsets.to(tl.int32), k_raw_async_layout, pin=False)
    k_load_mask = tlx.require_layout(tl.broadcast_to(k_n < N, k_offsets.shape), k_raw_async_layout, pin=False)
    k_token = tlx.buffer_load_to_local(
        tlx.local_view(k_raw_buffer, 0),
        K,
        k_offsets,
        mask=k_load_mask,
    )
    if DIRECT_V:
        tlx.async_load_commit_group([k_token])
        # Match V's final dot-operand ownership at the global load. It is read
        # once and remains live for the full query walk, so an LDS round trip
        # provides no reuse. Mask the N=200 tail before constructing the
        # resident BF16 fragments.
        v_offsets = tlx.require_layout(key_ptrs.to(tl.int32), v_op0_nm, pin=False)
        v_load_mask = tlx.require_layout(tl.broadcast_to(key_mask, v_offsets.shape), v_op0_nm, pin=False)
        v_zero = tlx.zeros((BLOCK_N, D), tl.bfloat16, layout=v_op0_nm)
        v_nm = tlx.buffer_load(V, v_offsets, mask=v_load_mask, other=v_zero)
        v_nm = tlx.require_layout(v_nm, v_op0_nm, pin=False)
    else:
        key_offsets = tlx.require_layout(key_ptrs.to(tl.int32), kv_async_layout, pin=False)
        key_load_mask = tlx.require_layout(
            tl.broadcast_to(key_mask, key_offsets.shape),
            kv_async_layout,
            pin=False,
        )
        v_token = tlx.buffer_load_to_local(
            tlx.local_view(v_buffer, 0),
            V,
            key_offsets,
            mask=key_load_mask,
        )
        tlx.async_load_commit_group([k_token, v_token])
    kv_wait = tlx.async_load_wait_group(0)
    k_nm = tlx.local_load(k_buffer, token=kv_wait, layout=k_op0_nm)
    if not DIRECT_V:
        v_nm = tlx.local_load(tlx.local_view(v_buffer, 0), token=kv_wait, layout=v_op0_nm)
        # Every wave must finish reading V before its aliased LDS allocation
        # is reused by the first dS stage.
        tl.debug_barrier()

    # The baseline keeps dK as four 32-column groups, each containing two
    # native output tiles per wave. The scheduled-MFMA experiment splits those
    # groups once more along N so each value is exactly one persistent native
    # accumulator per wave and can be braided with one delayed-dQ pair.
    if SCHEDULED_MFMA:
        dk_0_lo = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_0_hi = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_1_lo = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_1_hi = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_2_lo = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_2_hi = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_3_lo = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
        dk_3_hi = tlx.zeros((BLOCK_N // 2, 32), tl.float32, layout=mma_nd)
    else:
        dk_0 = tlx.zeros((BLOCK_N, 32), tl.float32, layout=mma_nd)
        dk_1 = tlx.zeros((BLOCK_N, 32), tl.float32, layout=mma_nd)
        dk_2 = tlx.zeros((BLOCK_N, 32), tl.float32, layout=mma_nd)
        dk_3 = tlx.zeros((BLOCK_N, 32), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK_M)
    # N=200 has seven live native K=32 reduction bands. Rows 224:255 are
    # entirely out of bounds, so issuing an eighth dQ MFMA pair only
    # accumulates zeros.
    num_dq_bands: tl.constexpr = tl.cdiv(N, 32)
    log2e: tl.constexpr = 1.4426950408889634

    first_m = tl.arange(0, BLOCK_M)
    first_ptrs = tensor_base + first_m[:, None] * D + offs_d[None, :]
    first_mask = first_m[:, None] < N
    first_offsets = tlx.require_layout(first_ptrs.to(tl.int32), qdo_async_layout, pin=False)
    first_load_mask = tlx.require_layout(tl.broadcast_to(first_mask, first_offsets.shape), qdo_async_layout, pin=False)
    first_q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, 0), Q, first_offsets, mask=first_load_mask)
    first_do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, 0), DO, first_offsets, mask=first_load_mask)
    tlx.async_load_commit_group([first_q_token, first_do_token])

    # Compile phase 0 as a prologue and phases 1..last as the steady bridge.
    # The static outer loop removes the previous-dS path from the prologue.
    for bridge_phase in tl.static_range(0, 2):
        for m_block in range(
                0 if bridge_phase == 0 else 1,
                1 if bridge_phase == 0 else num_m_blocks,
        ):
            if IS_CAUSAL:
                current_slot = m_block % Q_STAGES
                current_do_slot = m_block % 2
                # The final phase consumes the outstanding current group but
                # has no future consumer. Drain it with wait(0) instead of
                # issuing and clearing an unused out-of-range Q/dO refill.
                if m_block + 1 < num_m_blocks:
                    next_slot = (m_block + Q_LOOKAHEAD) % Q_STAGES
                    next_do_slot = 1 - current_do_slot
                    next_q_m = (m_block + Q_LOOKAHEAD) * BLOCK_M + tl.arange(0, BLOCK_M)
                    next_do_m = (m_block + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
                    next_q_ptrs = tensor_base + next_q_m[:, None] * D + offs_d[None, :]
                    next_do_ptrs = tensor_base + next_do_m[:, None] * D + offs_d[None, :]
                    next_q_mask = next_q_m[:, None] < N
                    next_do_mask = next_do_m[:, None] < N
                    if (m_block + Q_LOOKAHEAD + 1) * BLOCK_M > N:
                        tlx.local_store(
                            tlx.local_view(q_buffers, next_slot),
                            tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=qdo_async_layout),
                        )
                    if (m_block + 2) * BLOCK_M > N:
                        tlx.local_store(
                            tlx.local_view(do_buffers, next_do_slot),
                            tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=qdo_async_layout),
                        )
                    if ((m_block + Q_LOOKAHEAD + 1) * BLOCK_M > N or (m_block + 2) * BLOCK_M > N):
                        tl.debug_barrier()
                    next_q_offsets = tlx.require_layout(next_q_ptrs.to(tl.int32), qdo_async_layout, pin=False)
                    next_q_load_mask = tlx.require_layout(
                        tl.broadcast_to(next_q_mask, next_q_offsets.shape),
                        qdo_async_layout,
                        pin=False,
                    )
                    next_do_offsets = tlx.require_layout(next_do_ptrs.to(tl.int32), qdo_async_layout, pin=False)
                    next_do_load_mask = tlx.require_layout(
                        tl.broadcast_to(next_do_mask, next_do_offsets.shape),
                        qdo_async_layout,
                        pin=False,
                    )
                    next_q_token = tlx.buffer_load_to_local(
                        tlx.local_view(q_buffers, next_slot),
                        Q,
                        next_q_offsets,
                        mask=next_q_load_mask,
                    )
                    next_do_token = tlx.buffer_load_to_local(
                        tlx.local_view(do_buffers, next_do_slot),
                        DO,
                        next_do_offsets,
                        mask=next_do_load_mask,
                    )
                    tlx.async_load_commit_group([next_q_token, next_do_token])
                    qdo_wait = tlx.async_load_wait_group(1)
                else:
                    qdo_wait = tlx.async_load_wait_group(0)
                do_slot = current_do_slot
            else:
                current_slot = m_block % 2
                next_slot = 1 - current_slot
                next_m = (m_block + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
                next_ptrs = tensor_base + next_m[:, None] * D + offs_d[None, :]
                next_mask = next_m[:, None] < N
                if (m_block + 2) * BLOCK_M > N:
                    tlx.local_store(
                        tlx.local_view(q_buffers, next_slot),
                        tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=qdo_async_layout),
                    )
                    tlx.local_store(
                        tlx.local_view(do_buffers, next_slot),
                        tlx.zeros((BLOCK_M, D), tl.bfloat16, layout=qdo_async_layout),
                    )
                    tl.debug_barrier()
                next_offsets = tlx.require_layout(next_ptrs.to(tl.int32), qdo_async_layout, pin=False)
                next_load_mask = tlx.require_layout(
                    tl.broadcast_to(next_mask, next_offsets.shape),
                    qdo_async_layout,
                    pin=False,
                )
                next_q_token = tlx.buffer_load_to_local(
                    tlx.local_view(q_buffers, next_slot),
                    Q,
                    next_offsets,
                    mask=next_load_mask,
                )
                next_do_token = tlx.buffer_load_to_local(
                    tlx.local_view(do_buffers, next_slot),
                    DO,
                    next_offsets,
                    mask=next_load_mask,
                )
                tlx.async_load_commit_group([next_q_token, next_do_token])
                qdo_wait = tlx.async_load_wait_group(1)
                do_slot = current_slot

            q_view = tlx.local_view(q_buffers, current_slot)
            do_view = tlx.local_view(do_buffers, do_slot)
            q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait, layout=qt_op1_nm)
            do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait, layout=dot_op1_nm)
            q_nd = tlx.local_load(q_view, token=qdo_wait, layout=q_op1_nd)
            do_nd = tlx.local_load(do_view, token=qdo_wait, layout=do_op1_nd)
            q_nd_0 = tlx.extract_slice(q_nd, [BLOCK_M, 32], [0, 0])
            q_nd_1 = tlx.extract_slice(q_nd, [BLOCK_M, 32], [0, 32])
            q_nd_2 = tlx.extract_slice(q_nd, [BLOCK_M, 32], [0, 64])
            q_nd_3 = tlx.extract_slice(q_nd, [BLOCK_M, 32], [0, 96])

            offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
            delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
            score_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
            score_acc = tl.dot(k_nm, q_t, acc=score_acc, out_dtype=score_acc.dtype)
            lse_full = tl.broadcast_to(lse[None, :] * log2e, (BLOCK_N, BLOCK_M))
            lse_full = tlx.require_layout(lse_full, mma_nm, pin=False)
            qk_scale_full = tlx.require_layout(
                tl.full((BLOCK_N, BLOCK_M), SM_SCALE * log2e, dtype=tl.float32),
                mma_nm,
                pin=False,
            )
            scores_t = score_acc * qk_scale_full - lse_full
            valid = key_mask & (offs_m[None, :] < N)
            if IS_CAUSAL:
                valid = valid & (offs_n[:, None] <= offs_m[None, :])
            valid = tlx.require_layout(valid, mma_nm, pin=False)
            neg_inf = tlx.require_layout(
                tl.full((BLOCK_N, BLOCK_M), float("-inf"), dtype=tl.float32),
                mma_nm,
                pin=False,
            )
            scores_t = tl.where(valid, scores_t, neg_inf)
            p_t = tlx.require_layout(tl.math.exp2(scores_t), mma_nm, pin=False)

            dpt_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
            dpt_acc = tl.dot(v_nm, do_t, acc=dpt_acc, out_dtype=dpt_acc.dtype)
            delta_full = tl.broadcast_to(delta[None, :], (BLOCK_N, BLOCK_M))
            delta_full = tlx.require_layout(delta_full, mma_nm, pin=False)
            ds_t = p_t * (dpt_acc - delta_full)
            # Ordinary casts retain the score MFMA ownership. The LDS store
            # reconciles that transient register layout with its shared consumer.
            ds_bf16 = ds_t.to(tl.bfloat16)

            # Keep the current dS value in dK operand ownership.  Rotate the
            # four 2-way N subdimensions into the native CDNA4 dK operand
            # order before requesting that ownership.  This is the same
            # register-only permutation used by V171; unlike a direct
            # score-layout -> dK-layout conversion, it gives propagation an
            # aligned representation and avoids publishing/reloading current
            # dS merely to redistribute it through LDS.
            pt_nd = tl.reshape(p_t.to(tl.bfloat16), (2, 2, 2, 2, 16, BLOCK_M))
            pt_nd = tl.permute(pt_nd, (0, 2, 3, 1, 4, 5))
            pt_nd = tl.reshape(pt_nd, (BLOCK_N, BLOCK_M))
            pt_nd = tlx.require_layout(pt_nd, pt_op0_nd, pin=False)
            ds_nd = tl.reshape(ds_bf16, (2, 2, 2, 2, 16, BLOCK_M))
            ds_nd = tl.permute(ds_nd, (0, 2, 3, 1, 4, 5))
            ds_nd = tl.reshape(ds_nd, (BLOCK_N, BLOCK_M))
            ds_nd = tlx.require_layout(ds_nd, dst_op0_nd, pin=False)
            if SCHEDULED_MFMA:
                ds_nd_lo = tlx.extract_slice(
                    ds_nd,
                    [BLOCK_N // 2, BLOCK_M],
                    [0, 0],
                )
                ds_nd_hi = tlx.extract_slice(
                    ds_nd,
                    [BLOCK_N // 2, BLOCK_M],
                    [BLOCK_N // 2, 0],
                )

            current_ds_slot = m_block % 2
            tlx.local_store(tlx.local_view(ds_buffers, current_ds_slot), ds_bf16)
            # dV and dK are register-ready while the current publication and
            # previous-dQ LDS reads progress.
            dv = tl.dot(pt_nd, do_nd, acc=dv, out_dtype=dv.dtype)
            if bridge_phase == 1:
                previous_ds_slot = (m_block - 1) % 2
                previous_ds_full = tlx.local_load(
                    tlx.local_trans(tlx.local_view(ds_buffers, previous_ds_slot)),
                    layout=ds_op0_md,
                    relaxed=True,
                )
                previous_ds_0 = tlx.extract_slice(
                    previous_ds_full,
                    [BLOCK_M, 32],
                    [0, 0],
                )
                dq_a = tlx.zeros((BLOCK_M, D // 2), tl.float32, layout=mma_md)
                dq_b = tlx.zeros((BLOCK_M, D // 2), tl.float32, layout=mma_md)

                # Launch the first delayed-dQ band reads before current dK.
                # The two output halves are independent accumulator chains;
                # keeping the K reduction in 32-row bands prevents all of K
                # and previous dS from becoming live at once.
                k_a_0 = tlx.local_load(
                    tlx.local_slice(k_buffer, [0, 0], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                k_b_0 = tlx.local_load(
                    tlx.local_slice(k_buffer, [0, D // 2], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )

            # Explicit V171-style ready-work braid. Each dK group lowers to
            # two independent 32x32x16 MFMAs per wave. Each dQ band contributes
            # one 16x16x32 update to each of the independent A/B chains. The
            # first four dQ bands are separated by the four current-dK groups;
            # the remaining bands drain after all eight dK updates have issued.
            # Start the next band's K-A read before each current dQ pair. It
            # remains behind the current K-A/K-B operands in the LDS queue, so
            # partial waits can preserve useful read work across both chains.
            if SCHEDULED_MFMA:
                dk_0_lo = tlx.amd_scheduled_mfma(
                    ds_nd_lo,
                    q_nd_0,
                    dk_0_lo,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_0_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_0,
                    dk_0_hi,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
            else:
                dk_0 = tl.dot(ds_nd, q_nd_0, acc=dk_0, out_dtype=dk_0.dtype)

            if bridge_phase == 1:
                previous_ds_1 = tlx.extract_slice(
                    previous_ds_full,
                    [BLOCK_M, 32],
                    [0, 32],
                )
                k_a_1 = tlx.local_load(
                    tlx.local_slice(k_buffer, [32, 0], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                if SCHEDULED_MFMA:
                    dq_a = tlx.amd_scheduled_mfma(
                        previous_ds_0,
                        k_a_0,
                        dq_a,
                        resident_operand=1,
                        accumulator_role="transient",
                        initialize=True,
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_0,
                        k_b_0,
                        dq_b,
                        resident_operand=1,
                        accumulator_role="transient",
                        initialize=True,
                    )
                else:
                    tlx.amd_sched_barrier(MFMA_BRAID_BARRIER_MASK)
                    dq_a = tl.dot(previous_ds_0, k_a_0, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_0, k_b_0, acc=dq_b, out_dtype=dq_b.dtype)
                k_b_1 = tlx.local_load(
                    tlx.local_slice(k_buffer, [32, D // 2], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                previous_ds_2 = tlx.extract_slice(
                    previous_ds_full,
                    [BLOCK_M, 32],
                    [0, 64],
                )
                k_a_2 = tlx.local_load(
                    tlx.local_slice(k_buffer, [64, 0], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
            if SCHEDULED_MFMA:
                dk_1_lo = tlx.amd_scheduled_mfma(
                    ds_nd_lo,
                    q_nd_1,
                    dk_1_lo,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_1_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_1,
                    dk_1_hi,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
            else:
                dk_1 = tl.dot(ds_nd, q_nd_1, acc=dk_1, out_dtype=dk_1.dtype)

            if bridge_phase == 1:
                if SCHEDULED_MFMA:
                    dq_a = tlx.amd_scheduled_mfma(
                        previous_ds_1,
                        k_a_1,
                        dq_a,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_1,
                        k_b_1,
                        dq_b,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                else:
                    tlx.amd_sched_barrier(MFMA_BRAID_BARRIER_MASK)
                    dq_a = tl.dot(previous_ds_1, k_a_1, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_1, k_b_1, acc=dq_b, out_dtype=dq_b.dtype)
                k_b_2 = tlx.local_load(
                    tlx.local_slice(k_buffer, [64, D // 2], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                previous_ds_3 = tlx.extract_slice(
                    previous_ds_full,
                    [BLOCK_M, 32],
                    [0, 96],
                )
                k_a_3 = tlx.local_load(
                    tlx.local_slice(k_buffer, [96, 0], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
            if SCHEDULED_MFMA:
                dk_2_lo = tlx.amd_scheduled_mfma(
                    ds_nd_lo,
                    q_nd_2,
                    dk_2_lo,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_2_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_2,
                    dk_2_hi,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
            else:
                dk_2 = tl.dot(ds_nd, q_nd_2, acc=dk_2, out_dtype=dk_2.dtype)

            if bridge_phase == 1:
                if SCHEDULED_MFMA:
                    dq_a = tlx.amd_scheduled_mfma(
                        previous_ds_2,
                        k_a_2,
                        dq_a,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_2,
                        k_b_2,
                        dq_b,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                else:
                    tlx.amd_sched_barrier(MFMA_BRAID_BARRIER_MASK)
                    dq_a = tl.dot(previous_ds_2, k_a_2, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_2, k_b_2, acc=dq_b, out_dtype=dq_b.dtype)
                k_b_3 = tlx.local_load(
                    tlx.local_slice(k_buffer, [96, D // 2], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                previous_ds_4 = tlx.extract_slice(
                    previous_ds_full,
                    [BLOCK_M, 32],
                    [0, 128],
                )
                k_a_4 = tlx.local_load(
                    tlx.local_slice(k_buffer, [128, 0], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
            if SCHEDULED_MFMA:
                dk_3_lo = tlx.amd_scheduled_mfma(
                    ds_nd_lo,
                    q_nd_3,
                    dk_3_lo,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_3_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_3,
                    dk_3_hi,
                    accumulator_role="persistent",
                    initialize=bridge_phase == 0,
                )
            else:
                dk_3 = tl.dot(ds_nd, q_nd_3, acc=dk_3, out_dtype=dk_3.dtype)

            if bridge_phase == 1:
                if SCHEDULED_MFMA:
                    dq_a = tlx.amd_scheduled_mfma(
                        previous_ds_3,
                        k_a_3,
                        dq_a,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_3,
                        k_b_3,
                        dq_b,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                else:
                    tlx.amd_sched_barrier(MFMA_BRAID_BARRIER_MASK)
                    dq_a = tl.dot(previous_ds_3, k_a_3, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_3, k_b_3, acc=dq_b, out_dtype=dq_b.dtype)
                k_b_4 = tlx.local_load(
                    tlx.local_slice(k_buffer, [128, D // 2], [32, D // 2]),
                    token=kv_wait,
                    layout=k_op1_md,
                    relaxed=True,
                )
                if SCHEDULED_MFMA:
                    dq_a = tlx.amd_scheduled_mfma(
                        previous_ds_4,
                        k_a_4,
                        dq_a,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_4,
                        k_b_4,
                        dq_b,
                        resident_operand=1,
                        accumulator_role="transient",
                    )
                else:
                    dq_a = tl.dot(previous_ds_4, k_a_4, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_4, k_b_4, acc=dq_b, out_dtype=dq_b.dtype)
                for dq_band in tl.static_range(5, num_dq_bands):
                    previous_ds_band = tlx.extract_slice(
                        previous_ds_full,
                        [BLOCK_M, 32],
                        [0, dq_band * 32],
                    )
                    k_a_band = tlx.local_load(
                        tlx.local_slice(
                            k_buffer,
                            [dq_band * 32, 0],
                            [32, D // 2],
                        ),
                        token=kv_wait,
                        layout=k_op1_md,
                        relaxed=True,
                    )
                    k_b_band = tlx.local_load(
                        tlx.local_slice(
                            k_buffer,
                            [dq_band * 32, D // 2],
                            [32, D // 2],
                        ),
                        token=kv_wait,
                        layout=k_op1_md,
                        relaxed=True,
                    )
                    if SCHEDULED_MFMA:
                        dq_a = tlx.amd_scheduled_mfma(
                            previous_ds_band,
                            k_a_band,
                            dq_a,
                            resident_operand=1,
                            accumulator_role="transient",
                        )
                        dq_b = tlx.amd_scheduled_mfma(
                            previous_ds_band,
                            k_b_band,
                            dq_b,
                            resident_operand=1,
                            accumulator_role="transient",
                        )
                    else:
                        dq_a = tl.dot(previous_ds_band, k_a_band, acc=dq_a, out_dtype=dq_a.dtype)
                        dq_b = tl.dot(previous_ds_band, k_b_band, acc=dq_b, out_dtype=dq_b.dtype)

                if SCHEDULED_MFMA:
                    dq_a, dq_b, _ = tlx.amd_mfma_commit(
                        (dq_a, dq_b),
                        k_b_band,
                    )

                dq_scale_half = tlx.require_layout(
                    tl.full((BLOCK_M, D // 2), SM_SCALE, dtype=tl.float32),
                    mma_md,
                    pin=False,
                )
                dq_a = dq_a * dq_scale_half
                dq_b = dq_b * dq_scale_half
                previous_offs_m = (m_block - 1) * BLOCK_M + tl.arange(0, BLOCK_M)
                previous_q_ptrs_a = (tensor_base + previous_offs_m[:, None] * D + offs_d_half[None, :])
                previous_q_ptrs_b = previous_q_ptrs_a + D // 2
                previous_q_ptrs_a = tlx.require_layout(previous_q_ptrs_a, mma_md, pin=False)
                previous_q_ptrs_b = tlx.require_layout(previous_q_ptrs_b, mma_md, pin=False)
                previous_q_mask = tl.broadcast_to(
                    previous_offs_m[:, None] < N,
                    previous_q_ptrs_a.shape,
                )
                previous_q_mask = tlx.require_layout(previous_q_mask, mma_md, pin=False)
                dq_a_ptrs = tlx.require_layout(DQ + previous_q_ptrs_a, mma_md, pin=False)
                dq_b_ptrs = tlx.require_layout(DQ + previous_q_ptrs_b, mma_md, pin=False)
                tl.store(dq_a_ptrs, dq_a.to(tl.bfloat16), mask=previous_q_mask)
                tl.store(dq_b_ptrs, dq_b.to(tl.bfloat16), mask=previous_q_mask)

            # Both relaxed consumers must finish before the next phase can
            # recycle the older alternating stage.
            tl.debug_barrier()

    # Drain dQ for the last phase after the final current-dK/previous-dQ bridge.
    last_m_block: tl.constexpr = num_m_blocks - 1
    last_ds_slot: tl.constexpr = last_m_block % 2
    last_ds_full = tlx.local_load(
        tlx.local_trans(tlx.local_view(ds_buffers, last_ds_slot)),
        layout=ds_op0_md,
        relaxed=True,
    )
    last_dq_a = tlx.zeros((BLOCK_M, D // 2), tl.float32, layout=mma_md)
    last_dq_b = tlx.zeros((BLOCK_M, D // 2), tl.float32, layout=mma_md)
    for dq_band in tl.static_range(0, num_dq_bands):
        last_ds_band = tlx.extract_slice(
            last_ds_full,
            [BLOCK_M, 32],
            [0, dq_band * 32],
        )
        last_k_a_band = tlx.local_load(
            tlx.local_slice(
                k_buffer,
                [dq_band * 32, 0],
                [32, D // 2],
            ),
            token=kv_wait,
            layout=k_op1_md,
            relaxed=True,
        )
        last_k_b_band = tlx.local_load(
            tlx.local_slice(
                k_buffer,
                [dq_band * 32, D // 2],
                [32, D // 2],
            ),
            token=kv_wait,
            layout=k_op1_md,
            relaxed=True,
        )
        last_dq_a = tl.dot(last_ds_band, last_k_a_band, acc=last_dq_a, out_dtype=last_dq_a.dtype)
        last_dq_b = tl.dot(last_ds_band, last_k_b_band, acc=last_dq_b, out_dtype=last_dq_b.dtype)
    last_dq_scale = tlx.require_layout(
        tl.full((BLOCK_M, D // 2), SM_SCALE, dtype=tl.float32),
        mma_md,
        pin=False,
    )
    last_dq_a = last_dq_a * last_dq_scale
    last_dq_b = last_dq_b * last_dq_scale
    last_offs_m = last_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    last_q_ptrs_a = tensor_base + last_offs_m[:, None] * D + offs_d_half[None, :]
    last_q_ptrs_b = last_q_ptrs_a + D // 2
    last_q_ptrs_a = tlx.require_layout(last_q_ptrs_a, mma_md, pin=False)
    last_q_ptrs_b = tlx.require_layout(last_q_ptrs_b, mma_md, pin=False)
    last_q_mask = tl.broadcast_to(last_offs_m[:, None] < N, last_q_ptrs_a.shape)
    last_q_mask = tlx.require_layout(last_q_mask, mma_md, pin=False)
    last_dq_a_ptrs = tlx.require_layout(DQ + last_q_ptrs_a, mma_md, pin=False)
    last_dq_b_ptrs = tlx.require_layout(DQ + last_q_ptrs_b, mma_md, pin=False)
    tl.store(last_dq_a_ptrs, last_dq_a.to(tl.bfloat16), mask=last_q_mask)
    tl.store(last_dq_b_ptrs, last_dq_b.to(tl.bfloat16), mask=last_q_mask)

    tlx.async_load_wait_group(0)
    # Undo the current-dS operand rotation once, after the persistent dK
    # accumulation is complete. Keeping the inverse outside the phase loop
    # preserves logical [N, D] store order without lengthening the bridge.
    if SCHEDULED_MFMA:
        dk_0 = tl.cat(dk_0_lo, dk_0_hi, dim=0)
        dk_1 = tl.cat(dk_1_lo, dk_1_hi, dim=0)
        dk_2 = tl.cat(dk_2_lo, dk_2_hi, dim=0)
        dk_3 = tl.cat(dk_3_lo, dk_3_hi, dim=0)
    dk = tl.cat(tl.cat(dk_0, dk_1, dim=1), tl.cat(dk_2, dk_3, dim=1), dim=1)
    dk = tl.reshape(dk, (2, 2, 2, 2, 16, D))
    dk = tl.permute(dk, (0, 3, 1, 2, 4, 5))
    dk = tl.reshape(dk, (BLOCK_N, D))
    dk_mma = tlx.require_layout(dk, mma_nd)
    dk_mma *= SM_SCALE
    # Causal and non-causal modes use Gluon's whole-tile epilogue: pin each
    # completed accumulator in native MFMA ownership before narrowing, then
    # pin each final store to eight contiguous BF16 values per lane. The hard
    # ``require_layout`` anchors make each conversion boundary explicit;
    # Coalesce and AMD OptimizeEpilogue lower the ordinary cast to the same
    # MFMA -> BF16 -> vector handoff as a layout-preserving cast.
    # Gluon's newer full-attention D64-half epilogue raised TLX from 496 to 503
    # VGPR for N=200, so the lower-resource whole-tile epilogue remains used.
    # Pinning dV's final store ownership as well removes the non-causal spills
    # without adding causal spills on gfx950.
    dk_bf16 = dk_mma.to(tl.bfloat16)
    dk_vec = tlx.require_layout(dk_bf16, kv_async_layout)
    if IS_CAUSAL:
        # Do not reuse the range fragments from the prologue V load here.
        # Rebuilding each epilogue address gives LLVM/RA short, independent
        # live ranges instead of carrying fourteen VGPR address leaves through
        # the register-heavy dK/dQ bridge.
        dk_offs_n = tlx.rematerialized_range(0, BLOCK_N, 0)
        dk_offs_d = tlx.rematerialized_range(0, D, 1)
        dk_key_ptrs = (tensor_base + dk_offs_n[:, None] * D + dk_offs_d[None, :])
        dk_key_mask = dk_offs_n[:, None] < N
    else:
        dk_key_ptrs = key_ptrs
        dk_key_mask = key_mask
    tl.store(DK + dk_key_ptrs, dk_vec, mask=dk_key_mask)
    dv = tl.reshape(dv, (2, 2, 2, 2, 16, D))
    dv = tl.permute(dv, (0, 3, 1, 2, 4, 5))
    dv = tl.reshape(dv, (BLOCK_N, D))
    dv_mma = tlx.require_layout(dv, mma_nd)
    dv_bf16 = dv_mma.to(tl.bfloat16)
    dv_vec = tlx.require_layout(dv_bf16, kv_async_layout)
    if IS_CAUSAL:
        dv_offs_n = tlx.rematerialized_range(0, BLOCK_N, 2)
        dv_offs_d = tlx.rematerialized_range(0, D, 3)
        dv_key_ptrs = (tensor_base + dv_offs_n[:, None] * D + dv_offs_d[None, :])
        dv_key_mask = dv_offs_n[:, None] < N
    else:
        dv_key_ptrs = key_ptrs
        dv_key_mask = key_mask
    tl.store(DV + dv_key_ptrs, dv_vec, mask=dv_key_mask)


@triton.jit
def _attn_bwd_dkdv_dq_d128_combined_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT: tl.constexpr,
    PIPELINED: tl.constexpr,
    SCHEDULED_MFMA: tl.constexpr,
):
    """Stable one-CTA D128 combined entry configured by schedule kwargs."""
    if EXACT:
        tl.static_assert(not PIPELINED)
        _attn_bwd_dkdv_dq_d128_exact_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DQ,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
            SCHEDULED_MFMA,
        )
    elif PIPELINED:
        _attn_bwd_dkdv_dq_d128_persistent_pipeline_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DQ,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
        )
    else:
        _attn_bwd_dkdv_dq_d128_persistent_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DQ,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
        )


def _select_d128_dkdv_config(shape, causal):
    """Choose the dK/dV tile independently from the dQ tile.

    Gluon's short D128 paths use BM32 for the streamed Q/dO rows: a square
    BM32/BN32 tile with two waves for causal attention and a rectangular
    BM32/BN64 tile with four waves for full attention.  Longer sequences keep
    the historical 64x64 fallback.
    """
    if len(shape) == 4 and shape[-1] == 128 and 128 <= shape[-2] < 256:
        if causal:
            return 32, 32, 2
        return 32, 64, 4
    return 64, 64, _d128_num_warps(causal)


_D128_PERSISTENT_EXPERIMENT_SHAPE = (16, 27, 200, 128)
_D128_PERSISTENT_ENABLE_ENV = "TLX_FA_BWD_ENABLE_PERSISTENT_D128"
_D128_PERSISTENT_PIPE_ENABLE_ENV = "TLX_FA_BWD_ENABLE_PERSISTENT_D128_PIPE"
_D128_EXACT_ENABLE_ENV = "TLX_FA_BWD_ENABLE_EXACT_D128"
_D128_SINK_INSTS_ENV = "TLX_FA_BWD_SINK_INSTS_TO_AVOID_SPILLS"
_D128_REGCLASS_PRIORITY_ENV = "TLX_FA_BWD_REGCLASS_PRIORITY_TRUMPS_GLOBALNESS"
_D128_REVERSE_LOCAL_ENV = "TLX_FA_BWD_REVERSE_LOCAL_ASSIGNMENT"
# TODO: Keep these experiment flags out of the production dispatch contract;
# consolidate or delete them when the generic persistent kernels are revisited.


def _d128_persistent_short_supported(shape, causal):
    del causal  # The combined kernel uses the same ownership for both masks.
    return tuple(shape) == _D128_PERSISTENT_EXPERIMENT_SHAPE


def _d128_exact_layout_supported(shape, causal):
    del causal
    return tuple(shape) == (16, 27, 200, 128)


@dataclasses.dataclass(frozen=True)
class _D128Dispatch:
    entry: object
    block_m: int
    block_n: int
    num_warps: int
    pipelined: bool = False
    rectangular: bool = False
    exact: bool = False


def _select_d128_dispatch(shape, causal):
    """Select one stable topology entry plus its constexpr schedule kwargs."""
    # The exact Gluon-derived schedule is validated only for the requested
    # square D128 target and is explicit opt-in because some gfx950 compiler
    # revisions cannot lower its SharedLinear path.  The generic persistent
    # branches below are only exact-shape experiments; they do not widen the
    # public shape contract. Check the exact opt-in first so persistent
    # experiment switches cannot suppress it. There is intentionally no
    # persistent-specific disable switch; legacy disable names are ignored.
    # With no opt-in branch selected, use the split fallback.
    exact_enabled = os.environ.get(_D128_EXACT_ENABLE_ENV, "") == "1"
    if _d128_exact_layout_supported(shape, causal) and exact_enabled:
        return _D128Dispatch(
            _attn_bwd_dkdv_dq_d128_combined_kernel,
            block_m=16,
            block_n=256,
            num_warps=4,
            exact=True,
        )
    if (os.environ.get(_D128_PERSISTENT_PIPE_ENABLE_ENV, "") == "1"
            and _d128_persistent_short_supported(shape, causal)):
        return _D128Dispatch(
            _attn_bwd_dkdv_dq_d128_combined_kernel,
            block_m=16,
            block_n=256,
            num_warps=4,
            pipelined=True,
        )
    if (os.environ.get(_D128_PERSISTENT_ENABLE_ENV, "") == "1" and _d128_persistent_short_supported(shape, causal)):
        return _D128Dispatch(
            _attn_bwd_dkdv_dq_d128_combined_kernel,
            block_m=16,
            block_n=256,
            num_warps=8,
        )
    block_m, block_n, num_warps = _select_d128_dkdv_config(shape, causal)
    return _D128Dispatch(
        _attn_bwd_dkdv_d128_split_kernel,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        pipelined=not causal,
        rectangular=block_m != block_n,
    )


def _d128_num_warps(causal=False):
    # The causal triangular loop benefits from the four-wave 16x16x32 layout;
    # the non-causal pipe remains memory-bound and is best at two waves.
    return 4 if causal else 2


def _matrix_instr_nonkdim():
    return _CDNA4_MATRIX_INSTR_NONKDIM


def _d128_regalloc_options():
    """Return cache-keyed LLVM register-allocation experiments for D128."""
    return {
        "sink_insts_to_avoid_spills": os.environ.get(_D128_SINK_INSTS_ENV, "") == "1",
        "regclass_priority_trumps_globalness": os.environ.get(_D128_REGCLASS_PRIORITY_ENV, "") == "1",
        "reverse_local_assignment": os.environ.get(_D128_REVERSE_LOCAL_ENV, "") == "1",
    }


def _run_bwd_d128(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal):
    batch, heads, n_ctx, head_dim = q.shape
    dispatch = _select_d128_dispatch(tuple(q.shape), causal)
    if dispatch.entry is _attn_bwd_dkdv_dq_d128_combined_kernel:
        # A single KV-owner CTA covers the complete short key tile.  The
        # combined kernel computes dQ from the same dS tile and stores it
        # directly, so no second Q-parallel launch or reduction is needed.
        dispatch.entry[(1, batch * heads)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            dq,
            SM_SCALE=sm_scale,
            IS_CAUSAL=causal,
            N=n_ctx,
            D=head_dim,
            BLOCK_M=dispatch.block_m,
            BLOCK_N=dispatch.block_n,
            EXACT=dispatch.exact,
            PIPELINED=dispatch.pipelined,
            # The exact-D128 opt-in selects its validated scheduled-MFMA
            # implementation directly; it has no second hidden opt-in.
            SCHEDULED_MFMA=dispatch.exact,
            num_warps=dispatch.num_warps,
            num_stages=1,
            matrix_instr_nonkdim=_matrix_instr_nonkdim(),
            **_d128_regalloc_options(),
        )
        return
    batch_heads = batch * heads
    assert dispatch.entry is _attn_bwd_dkdv_d128_split_kernel
    dkdv_grid = (triton.cdiv(n_ctx, dispatch.block_n), batch_heads)
    dispatch.entry[dkdv_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        N=n_ctx,
        D=head_dim,
        BLOCK_M=dispatch.block_m,
        BLOCK_N=dispatch.block_n,
        PIPELINED=dispatch.pipelined,
        RECTANGULAR=dispatch.rectangular,
        num_warps=dispatch.num_warps,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )
    dq_block = 64
    dq_grid = (triton.cdiv(n_ctx, dq_block), batch_heads)
    _attn_bwd_dq_d128_kernel[dq_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        N=n_ctx,
        D=head_dim,
        BLOCK=dq_block,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _run_bwd_d128_gqa(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq_acc,
    dq,
    dk,
    dv,
    sm_scale,
    causal,
):
    batch, hq, n_ctx, head_dim = q.shape
    hk = k.shape[1]
    assert _is_supported_gqa_shape((batch, hq, hk, n_ctx, head_dim))
    _attn_bwd_dkdv_dq_d128_gqa_kernel[(hk, triton.cdiv(n_ctx, 256), batch)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_acc,
        dk,
        dv,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        HQ=hq,
        HK=hk,
        N=n_ctx,
        D=head_dim,
        BLOCK_M=16,
        BLOCK_N=256,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
        # The source bridge directly emits fragmented persistent current-dK
        # and transient delayed-dQ MFMAs. The chains read different alternating
        # dS stages, so they are independent and can be interleaved. No TTGIR
        # dot preset is needed because the bridge contains scheduled fragments
        # rather than a pair of ordinary tt.dot operations.
        # Register-allocation experiments remain independent explicit opt-ins.
        reverse_local_assignment=(os.environ.get(_D128_REVERSE_LOCAL_ENV, "0") == "1"),
        sink_insts_to_avoid_spills=(os.environ.get(_D128_SINK_INSTS_ENV, "0") == "1"),
        regclass_priority_trumps_globalness=(os.environ.get(_D128_REGCLASS_PRIORITY_ENV, "") == "1"),
    )
    _attn_bwd_dq_native_convert_kernel[(triton.cdiv(n_ctx, 128), batch * hq)](
        dq_acc,
        dq,
        N=n_ctx,
        D=head_dim,
        BLOCK_M=128,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


@triton.jit
def _attn_bwd_dkdv_d256_staged_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DS,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    num_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    if IS_CAUSAL:
        zigzag_half = pid_n // 2
        pid_n = tl.where(pid_n % 2 == 0, zigzag_half, num_blocks - 1 - zigzag_half)

    n0 = pid_n * BLOCK
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HALF_D)
    tensor_base = batch_head * N * D
    scratch_base = batch_head * N_PAD * N_PAD
    row_mask = offs_n[:, None] < N
    lo_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    hi_ptrs = lo_ptrs + HALF_D

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, HALF_D],
    )
    k_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    k_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    v_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    v_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    q_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    q_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    do_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    do_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)

    kv_tokens = [
        tlx.async_load(K + lo_ptrs, tlx.local_view(k_lo_buffer, 0), mask=row_mask, other=0.0),
        tlx.async_load(K + hi_ptrs, tlx.local_view(k_hi_buffer, 0), mask=row_mask, other=0.0),
        tlx.async_load(V + lo_ptrs, tlx.local_view(v_lo_buffer, 0), mask=row_mask, other=0.0),
        tlx.async_load(V + hi_ptrs, tlx.local_view(v_hi_buffer, 0), mask=row_mask, other=0.0),
    ]
    tlx.async_load_commit_group(kv_tokens)
    tlx.async_load_wait_group(0)

    dk_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dk_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    start_m_block = pid_n if IS_CAUSAL else 0
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(start_m_block, num_blocks):
        tl.debug_barrier()
        offs_m = m_block * BLOCK + tl.arange(0, BLOCK)
        qdo_mask = offs_m[:, None] < N
        qdo_lo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        qdo_hi_ptrs = qdo_lo_ptrs + HALF_D
        qdo_tokens = [
            tlx.async_load(Q + qdo_lo_ptrs, tlx.local_view(q_lo_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(Q + qdo_hi_ptrs, tlx.local_view(q_hi_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(DO + qdo_lo_ptrs, tlx.local_view(do_lo_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(DO + qdo_hi_ptrs, tlx.local_view(do_hi_buffer, 0), mask=qdo_mask, other=0.0),
        ]
        tlx.async_load_commit_group(qdo_tokens)
        qdo_wait = tlx.async_load_wait_group(0)

        k_lo = tlx.local_load(tlx.local_view(k_lo_buffer, 0), token=qdo_wait)
        q_lo_t = tlx.local_load(tlx.local_trans(tlx.local_view(q_lo_buffer, 0)), token=qdo_wait)
        scores_t = tl.dot(k_lo, q_lo_t)
        k_hi = tlx.local_load(tlx.local_view(k_hi_buffer, 0), token=qdo_wait)
        q_hi_t = tlx.local_load(tlx.local_trans(tlx.local_view(q_hi_buffer, 0)), token=qdo_wait)
        scores_t = tl.dot(k_hi, q_hi_t, scores_t)

        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[:, None] <= offs_m[None, :])
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)

        v_lo = tlx.local_load(tlx.local_view(v_lo_buffer, 0), token=qdo_wait)
        do_lo_t = tlx.local_load(tlx.local_trans(tlx.local_view(do_lo_buffer, 0)), token=qdo_wait)
        dp_t = tl.dot(v_lo, do_lo_t)
        v_hi = tlx.local_load(tlx.local_view(v_hi_buffer, 0), token=qdo_wait)
        do_hi_t = tlx.local_load(tlx.local_trans(tlx.local_view(do_hi_buffer, 0)), token=qdo_wait)
        dp_t = tl.dot(v_hi, do_hi_t, dp_t)
        ds_t = p_t * (dp_t - delta[None, :])
        ds_bf16 = ds_t.to(tl.bfloat16)

        scratch_ptrs = scratch_base + offs_n[:, None] * N_PAD + offs_m[None, :]
        tl.store(DS + scratch_ptrs, ds_bf16)

        p_bf16 = p_t.to(tl.bfloat16)
        do_lo = tlx.local_load(tlx.local_view(do_lo_buffer, 0), token=qdo_wait)
        dv_lo = tl.dot(p_bf16, do_lo, dv_lo)
        q_lo = tlx.local_load(tlx.local_view(q_lo_buffer, 0), token=qdo_wait)
        dk_lo = tl.dot(ds_bf16, q_lo, dk_lo)
        do_hi = tlx.local_load(tlx.local_view(do_hi_buffer, 0), token=qdo_wait)
        dv_hi = tl.dot(p_bf16, do_hi, dv_hi)
        q_hi = tlx.local_load(tlx.local_view(q_hi_buffer, 0), token=qdo_wait)
        dk_hi = tl.dot(ds_bf16, q_hi, dk_hi)

    dk_lo *= SM_SCALE
    dk_hi *= SM_SCALE
    tl.store(DK + lo_ptrs, dk_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DK + hi_ptrs, dk_hi.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + lo_ptrs, dv_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + hi_ptrs, dv_hi.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d256_peel_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DS,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_D: tl.constexpr,
):
    tl.static_assert(not IS_CAUSAL, "the peeled D256 producer is non-causal only")
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    n0 = pid_n * BLOCK
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HALF_D)
    tensor_base = batch_head * N * D
    scratch_base = batch_head * N_PAD * N_PAD
    row_mask = offs_n[:, None] < N
    lo_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    hi_ptrs = lo_ptrs + HALF_D
    k_lo = tl.load(K + lo_ptrs, mask=row_mask, other=0.0)
    k_hi = tl.load(K + hi_ptrs, mask=row_mask, other=0.0)
    v_lo = tl.load(V + lo_ptrs, mask=row_mask, other=0.0)
    v_hi = tl.load(V + hi_ptrs, mask=row_mask, other=0.0)

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, HALF_D],
    )
    q_lo_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)
    q_hi_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)
    do_lo_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)
    do_hi_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)

    first_m = tl.arange(0, BLOCK)
    first_mask = first_m[:, None] < N
    first_lo_ptrs = tensor_base + first_m[:, None] * D + offs_d[None, :]
    first_tokens = [
        tlx.async_load(Q + first_lo_ptrs, tlx.local_view(q_lo_buffers, 0), mask=first_mask, other=0.0),
        tlx.async_load(
            Q + first_lo_ptrs + HALF_D,
            tlx.local_view(q_hi_buffers, 0),
            mask=first_mask,
            other=0.0,
        ),
        tlx.async_load(DO + first_lo_ptrs, tlx.local_view(do_lo_buffers, 0), mask=first_mask, other=0.0),
        tlx.async_load(
            DO + first_lo_ptrs + HALF_D,
            tlx.local_view(do_hi_buffers, 0),
            mask=first_mask,
            other=0.0,
        ),
    ]
    tlx.async_load_commit_group(first_tokens)

    dk_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dk_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    num_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(0, num_blocks):
        tl.debug_barrier()
        current_slot = m_block % 2
        next_slot = 1 - current_slot
        next_m = (m_block + 1) * BLOCK + tl.arange(0, BLOCK)
        next_mask = next_m[:, None] < N
        next_lo_ptrs = tensor_base + next_m[:, None] * D + offs_d[None, :]
        next_tokens = [
            tlx.async_load(
                Q + next_lo_ptrs,
                tlx.local_view(q_lo_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
            tlx.async_load(
                Q + next_lo_ptrs + HALF_D,
                tlx.local_view(q_hi_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
            tlx.async_load(
                DO + next_lo_ptrs,
                tlx.local_view(do_lo_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
            tlx.async_load(
                DO + next_lo_ptrs + HALF_D,
                tlx.local_view(do_hi_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
        ]
        tlx.async_load_commit_group(next_tokens)
        qdo_wait = tlx.async_load_wait_group(1)

        q_lo_view = tlx.local_view(q_lo_buffers, current_slot)
        q_hi_view = tlx.local_view(q_hi_buffers, current_slot)
        do_lo_view = tlx.local_view(do_lo_buffers, current_slot)
        do_hi_view = tlx.local_view(do_hi_buffers, current_slot)
        q_lo_t = tlx.local_load(tlx.local_trans(q_lo_view), token=qdo_wait)
        scores_t = tl.dot(k_lo, q_lo_t)
        q_hi_t = tlx.local_load(tlx.local_trans(q_hi_view), token=qdo_wait)
        scores_t = tl.dot(k_hi, q_hi_t, scores_t)

        offs_m = m_block * BLOCK + tl.arange(0, BLOCK)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)

        do_lo_t = tlx.local_load(tlx.local_trans(do_lo_view), token=qdo_wait)
        dp_t = tl.dot(v_lo, do_lo_t)
        do_hi_t = tlx.local_load(tlx.local_trans(do_hi_view), token=qdo_wait)
        dp_t = tl.dot(v_hi, do_hi_t, dp_t)
        ds_t = p_t * (dp_t - delta[None, :])
        ds_bf16 = ds_t.to(tl.bfloat16)
        scratch_ptrs = scratch_base + offs_n[:, None] * N_PAD + offs_m[None, :]
        tl.store(DS + scratch_ptrs, ds_bf16)

        p_bf16 = p_t.to(tl.bfloat16)
        do_lo = tlx.local_load(do_lo_view, token=qdo_wait)
        dv_lo = tl.dot(p_bf16, do_lo, dv_lo)
        q_lo = tlx.local_load(q_lo_view, token=qdo_wait)
        dk_lo = tl.dot(ds_bf16, q_lo, dk_lo)
        do_hi = tlx.local_load(do_hi_view, token=qdo_wait)
        dv_hi = tl.dot(p_bf16, do_hi, dv_hi)
        q_hi = tlx.local_load(q_hi_view, token=qdo_wait)
        dk_hi = tl.dot(ds_bf16, q_hi, dk_hi)

    tlx.async_load_wait_group(0)
    dk_lo *= SM_SCALE
    dk_hi *= SM_SCALE
    tl.store(DK + lo_ptrs, dk_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DK + hi_ptrs, dk_hi.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + lo_ptrs, dv_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + hi_ptrs, dv_hi.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d256_hoist_impl(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DS,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_D: tl.constexpr,
):
    tl.static_assert(IS_CAUSAL, "the AGPR-hoisted D256 producer is causal only")
    pid_n = tl.program_id(0)
    batch_head = tl.program_id(1)
    num_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    zigzag_half = pid_n // 2
    pid_n = tl.where(pid_n % 2 == 0, zigzag_half, num_blocks - 1 - zigzag_half)
    n0 = pid_n * BLOCK
    offs_n = n0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HALF_D)
    tensor_base = batch_head * N * D
    scratch_base = batch_head * N_PAD * N_PAD
    row_mask = offs_n[:, None] < N
    lo_ptrs = tensor_base + offs_n[:, None] * D + offs_d[None, :]
    hi_ptrs = lo_ptrs + HALF_D
    k_lo = tl.load(K + lo_ptrs, mask=row_mask, other=0.0)
    k_hi = tl.load(K + hi_ptrs, mask=row_mask, other=0.0)
    v_lo = tl.load(V + lo_ptrs, mask=row_mask, other=0.0)
    v_hi = tl.load(V + hi_ptrs, mask=row_mask, other=0.0)

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, HALF_D],
    )
    q_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    q_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    do_lo_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)
    do_hi_buffer = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 1, layout=shared_layout)

    dk_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dk_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dv_hi = tl.zeros((BLOCK, HALF_D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for m_block in range(pid_n, num_blocks):
        tl.debug_barrier()
        offs_m = m_block * BLOCK + tl.arange(0, BLOCK)
        qdo_mask = offs_m[:, None] < N
        qdo_lo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        qdo_hi_ptrs = qdo_lo_ptrs + HALF_D
        qdo_tokens = [
            tlx.async_load(Q + qdo_lo_ptrs, tlx.local_view(q_lo_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(Q + qdo_hi_ptrs, tlx.local_view(q_hi_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(DO + qdo_lo_ptrs, tlx.local_view(do_lo_buffer, 0), mask=qdo_mask, other=0.0),
            tlx.async_load(DO + qdo_hi_ptrs, tlx.local_view(do_hi_buffer, 0), mask=qdo_mask, other=0.0),
        ]
        tlx.async_load_commit_group(qdo_tokens)
        qdo_wait = tlx.async_load_wait_group(0)

        q_lo_view = q_lo_buffer[0]
        q_hi_view = q_hi_buffer[0]
        do_lo_view = do_lo_buffer[0]
        do_hi_view = do_hi_buffer[0]
        q_lo_t = tlx.local_load(tlx.local_trans(q_lo_view), token=qdo_wait)
        scores_t = tl.dot(k_lo, q_lo_t)
        q_hi_t = tlx.local_load(tlx.local_trans(q_hi_view), token=qdo_wait)
        scores_t = tl.dot(k_hi, q_hi_t, scores_t)

        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
        valid = (offs_n[:, None] < N) & (offs_m[None, :] < N)
        valid = valid & (offs_n[:, None] <= offs_m[None, :])
        scores_t = tl.where(valid, scores_t, float("-inf"))
        p_t = tl.math.exp2(scores_t)

        do_lo_t = tlx.local_load(tlx.local_trans(do_lo_view), token=qdo_wait)
        dp_t = tl.dot(v_lo, do_lo_t)
        do_hi_t = tlx.local_load(tlx.local_trans(do_hi_view), token=qdo_wait)
        dp_t = tl.dot(v_hi, do_hi_t, dp_t)
        ds_t = p_t * (dp_t - delta[None, :])
        ds_bf16 = ds_t.to(tl.bfloat16)
        scratch_ptrs = scratch_base + offs_n[:, None] * N_PAD + offs_m[None, :]
        tl.store(DS + scratch_ptrs, ds_bf16)

        p_bf16 = p_t.to(tl.bfloat16)
        do_lo = tlx.local_load(do_lo_view, token=qdo_wait)
        dv_lo = tl.dot(p_bf16, do_lo, dv_lo)
        q_lo = tlx.local_load(q_lo_view, token=qdo_wait)
        dk_lo = tl.dot(ds_bf16, q_lo, dk_lo)
        do_hi = tlx.local_load(do_hi_view, token=qdo_wait)
        dv_hi = tl.dot(p_bf16, do_hi, dv_hi)
        q_hi = tlx.local_load(q_hi_view, token=qdo_wait)
        dk_hi = tl.dot(ds_bf16, q_hi, dk_hi)

    dk_lo *= SM_SCALE
    dk_hi *= SM_SCALE
    tl.store(DK + lo_ptrs, dk_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DK + hi_ptrs, dk_hi.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + lo_ptrs, dv_lo.to(tl.bfloat16), mask=row_mask)
    tl.store(DV + hi_ptrs, dv_hi.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _attn_bwd_dkdv_d256_producer_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    DS,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_D: tl.constexpr,
    STAGED: tl.constexpr,
    PIPELINED: tl.constexpr,
):
    """Stable D256 dS-producing entry configured by residency/pipeline kwargs."""
    if STAGED:
        tl.static_assert(not PIPELINED)
        _attn_bwd_dkdv_d256_staged_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DS,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            N_PAD,
            BLOCK,
            HALF_D,
        )
    elif PIPELINED:
        _attn_bwd_dkdv_d256_peel_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DS,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            N_PAD,
            BLOCK,
            HALF_D,
        )
    else:
        _attn_bwd_dkdv_d256_hoist_impl(
            Q,
            K,
            V,
            DO,
            LSE,
            Delta,
            DK,
            DV,
            DS,
            SM_SCALE,
            IS_CAUSAL,
            N,
            D,
            N_PAD,
            BLOCK,
            HALF_D,
        )


@triton.jit
def _attn_bwd_dq_from_ds_d256_kernel(
    DS,
    K,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    m0 = pid_m * BLOCK
    offs_m = m0 + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HALF_D)
    tensor_base = batch_head * N * D
    scratch_base = batch_head * N_PAD * N_PAD
    num_blocks: tl.constexpr = tl.cdiv(N, BLOCK)
    end_n_block = pid_m + 1 if IS_CAUSAL else num_blocks

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK, HALF_D],
    )
    k_lo_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)
    k_hi_buffers = tlx.local_alloc((BLOCK, HALF_D), tl.bfloat16, 2, layout=shared_layout)

    first_n = tl.arange(0, BLOCK)
    first_mask = first_n[:, None] < N
    first_lo_ptrs = tensor_base + first_n[:, None] * D + offs_d[None, :]
    first_tokens = [
        tlx.async_load(K + first_lo_ptrs, tlx.local_view(k_lo_buffers, 0), mask=first_mask, other=0.0),
        tlx.async_load(
            K + first_lo_ptrs + HALF_D,
            tlx.local_view(k_hi_buffers, 0),
            mask=first_mask,
            other=0.0,
        ),
    ]
    tlx.async_load_commit_group(first_tokens)

    dq_lo = tl.zeros((BLOCK, HALF_D), tl.float32)
    dq_hi = tl.zeros((BLOCK, HALF_D), tl.float32)

    for n_block in range(0, end_n_block):
        tl.debug_barrier()
        current_slot = n_block % 2
        next_slot = 1 - current_slot
        next_n = (n_block + 1) * BLOCK + tl.arange(0, BLOCK)
        next_mask = next_n[:, None] < N
        next_lo_ptrs = tensor_base + next_n[:, None] * D + offs_d[None, :]
        next_tokens = [
            tlx.async_load(
                K + next_lo_ptrs,
                tlx.local_view(k_lo_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
            tlx.async_load(
                K + next_lo_ptrs + HALF_D,
                tlx.local_view(k_hi_buffers, next_slot),
                mask=next_mask,
                other=0.0,
            ),
        ]
        tlx.async_load_commit_group(next_tokens)
        k_wait = tlx.async_load_wait_group(1)

        offs_n = n_block * BLOCK + tl.arange(0, BLOCK)
        ds_ptrs = scratch_base + offs_n[None, :] * N_PAD + offs_m[:, None]
        ds_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        ds = tl.load(DS + ds_ptrs, mask=ds_mask, other=0.0)
        k_lo = tlx.local_load(tlx.local_view(k_lo_buffers, current_slot), token=k_wait)
        k_hi = tlx.local_load(tlx.local_view(k_hi_buffers, current_slot), token=k_wait)
        dq_lo = tl.dot(ds, k_lo, dq_lo)
        dq_hi = tl.dot(ds, k_hi, dq_hi)

    tlx.async_load_wait_group(0)
    dq_lo *= SM_SCALE
    dq_hi *= SM_SCALE
    out_mask = offs_m[:, None] < N
    out_lo_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
    tl.store(DQ + out_lo_ptrs, dq_lo.to(tl.bfloat16), mask=out_mask)
    tl.store(DQ + out_lo_ptrs + HALF_D, dq_hi.to(tl.bfloat16), mask=out_mask)


@dataclasses.dataclass(frozen=True)
class _D256Dispatch:
    entry: object
    num_warps: int
    staged: bool = False
    pipelined: bool = False


def _select_d256_dispatch(causal):
    """Select the stable dS producer entry plus constexpr schedule kwargs."""
    if not causal:
        return _D256Dispatch(_attn_bwd_dkdv_d256_producer_kernel, num_warps=4, pipelined=True)
    # With the CDNA4 16x16x32 selection below, MFMA accumulators are assigned to
    # AGPRs on gfx950 and the Gluon-matching hoisted K/V schedule wins.  Keep a
    # staged escape hatch for compiler/resource experiments; production dispatch
    # uses the hoisted path for this gfx950-only tutorial.
    if os.environ.get("TLX_FA_BWD_FORCE_STAGED", "") == "1":
        return _D256Dispatch(_attn_bwd_dkdv_d256_producer_kernel, num_warps=2, staged=True)
    return _D256Dispatch(_attn_bwd_dkdv_d256_producer_kernel, num_warps=4)


def _run_bwd_d256(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal, poison_scratch=False):
    batch, heads, n_ctx, head_dim = q.shape
    block = 64
    half_d = 128
    n_pad = triton.cdiv(n_ctx, block) * block
    ds = torch.empty((batch, heads, n_pad, n_pad), device=q.device, dtype=q.dtype)
    if poison_scratch:
        ds.fill_(float("nan"))
    grid = (triton.cdiv(n_ctx, block), batch * heads)
    dispatch = _select_d256_dispatch(causal)
    dispatch.entry[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        ds,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        N=n_ctx,
        D=head_dim,
        N_PAD=n_pad,
        BLOCK=block,
        HALF_D=half_d,
        STAGED=dispatch.staged,
        PIPELINED=dispatch.pipelined,
        num_warps=dispatch.num_warps,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )
    _attn_bwd_dq_from_ds_d256_kernel[grid](
        ds,
        k,
        dq,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        N=n_ctx,
        D=head_dim,
        N_PAD=n_pad,
        BLOCK=block,
        HALF_D=half_d,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _validate_inputs(q, k, v, o, do, lse):
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be rank-4 B,H,N,D tensors")
    batch, hq, n_ctx, head_dim = q.shape
    k_batch, hk, k_ctx, k_dim = k.shape
    mha_shape = (tuple(q.shape) in SUPPORTED_SHAPES and tuple(k.shape) == tuple(q.shape))
    gqa_signature = (batch, hq, hk, n_ctx, head_dim)
    gqa_shape = (_is_supported_gqa_shape(gqa_signature) and (k_batch, k_ctx, k_dim) == (
        batch,
        n_ctx,
        head_dim,
    ))
    d64_shape = _is_supported_d64_shape(tuple(q.shape), tuple(k.shape))
    if not (mha_shape or gqa_shape or d64_shape):
        supported = sorted(SUPPORTED_SHAPES)
        raise ValueError(f"supported MHA shapes are {supported}; supported GQA shapes "
                         f"satisfy {_GQA_SHAPE_CONSTRAINT}; {_D64_SHAPE_CONSTRAINT}; "
                         f"got q={tuple(q.shape)}, "
                         f"k={tuple(k.shape)}")
    q_tensors = {"q": q, "o": o, "do": do}
    for name, tensor in q_tensors.items():
        if tensor.device != q.device or tensor.shape != q.shape:
            raise ValueError(f"{name} must match q shape and device")
        if tensor.dtype is not torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous B,H,N,D")
    kv_tensors = {"k": k, "v": v}
    for name, tensor in kv_tensors.items():
        if tensor.device != q.device or tensor.shape != k.shape:
            raise ValueError(f"{name} must match k shape and q device")
        if tensor.dtype is not torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous B,H,N,D")
    if lse.device != q.device or lse.shape != q.shape[:-1] or lse.dtype is not torch.float32:
        raise ValueError("lse must be FP32 B,H,N on the same device")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous B,H,N")
    arch = torch.cuda.get_device_properties(q.device).gcnArchName
    if not arch.startswith("gfx950"):
        raise ValueError(f"gfx950 is required, got {arch}")


def fa_backward(q, k, v, o, do, lse, sm_scale, causal):
    _validate_inputs(q, k, v, o, do, lse)
    if _is_supported_d64_shape(tuple(q.shape), tuple(k.shape)):
        sm_scale = _validate_d64_sm_scale(sm_scale)
        if causal and q.shape[2] > k.shape[2]:
            raise ValueError("D64 bottom-right causal attention requires SQ <= SKV")
        dispatch = _select_d64_dispatch_for_device(q, k, v, o, do, lse, sm_scale, causal)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        _run_bwd_d64(
            q,
            k,
            v,
            o,
            do,
            lse,
            delta,
            dq,
            dk,
            dv,
            sm_scale,
            causal,
            dispatch,
        )
        return dq, dk, dv
    gqa_signature = (
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[3],
    )
    if _is_supported_gqa_shape(gqa_signature):
        dq_acc = torch.zeros_like(q)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        _run_bwd_preprocess(o, do, delta)
        _run_bwd_d128_gqa(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_acc,
            dq,
            dk,
            dv,
            sm_scale,
            causal,
        )
        return dq, dk, dv
    if q.shape[-1] == 128:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        _run_bwd_preprocess(o, do, delta)
        _run_bwd_d128(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal)
        return dq, dk, dv
    if q.shape[-1] == 256:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        _run_bwd_preprocess(o, do, delta)
        _run_bwd_d256(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal)
        return dq, dk, dv
    raise NotImplementedError("TLX Flash-Attention backward kernels are not implemented yet")


@pytest.mark.parametrize("shape", sorted(SUPPORTED_SHAPES))
def test_fa_backward_rejects_fp16(shape):
    q = torch.empty(shape, device="cuda", dtype=torch.float16)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="bfloat16"):
        fa_backward(q, q, q, q, q, lse, 0.5, False)


def test_fa_backward_rejects_unsupported_shape():
    shape = (1, 1, 128, 128)
    q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="supported MHA shapes"):
        fa_backward(q, q, q, q, q, lse, 0.5, False)


def test_fa_backward_rejects_noncontiguous_lse():
    batch, heads, n_ctx, head_dim = (16, 27, 200, 128)
    q = torch.empty((batch, heads, n_ctx, head_dim), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((batch, heads, 2 * n_ctx), device="cuda", dtype=torch.float32)[..., ::2]
    assert lse.shape == (batch, heads, n_ctx)
    assert not lse.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        fa_backward(q, q, q, q, q, lse, 0.5, False)


@pytest.mark.parametrize("causal", [False, True])
def test_make_reference_case(causal):
    case = make_reference_case((1, 1, 8, 4), causal, seed=17)
    assert case.q.shape == (1, 1, 8, 4)
    assert case.o.dtype is torch.bfloat16
    assert case.lse.dtype is torch.float32
    assert len(case.grads) == 3
    for grad in case.grads:
        assert grad.shape == case.q.shape
        assert torch.isfinite(grad).all()
    assert case.kernel_args == (
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        case.sm_scale,
        causal,
    )


@pytest.mark.parametrize("shape", sorted(SUPPORTED_SHAPES))
def test_bwd_preprocess(shape):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    o = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    _run_bwd_preprocess(o, do, actual)
    expected = (o.float() * do.float()).sum(-1)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _snr_db(actual, expected):
    signal = torch.linalg.vector_norm(expected.float())
    noise = torch.linalg.vector_norm(actual.float() - expected.float())
    return 20.0 * torch.log10(signal / noise).item()


def test_d128_dispatch_uses_one_entry_per_launch_topology(monkeypatch):
    """Schedule kwargs select implementations behind stable topology entries."""
    monkeypatch.delenv(_D128_EXACT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)

    full = _select_d128_dispatch((16, 27, 200, 128), False)
    causal = _select_d128_dispatch((16, 27, 200, 128), True)
    assert full.entry is causal.entry is _attn_bwd_dkdv_d128_split_kernel
    assert (full.pipelined, full.rectangular, full.block_m, full.block_n) == (True, True, 32, 64)
    assert (causal.pipelined, causal.rectangular, causal.block_m, causal.block_n) == (False, False, 32, 32)

    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "1")
    exact = _select_d128_dispatch((16, 27, 200, 128), False)
    assert exact.entry is _attn_bwd_dkdv_dq_d128_combined_kernel
    assert exact.exact and not exact.pipelined


def test_gqa_benchmark_shapes_match_hk_series():
    assert GQA_BENCHMARK_SHAPES == {
        (16, 64, 8, 1024, 128),
        (16, 64, 8, 2048, 128),
        (16, 64, 8, 4096, 128),
        (16, 64, 8, 8192, 128),
        (15, 64, 8, 16384, 128),
    }


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 256, 128),
        (2, 6, 3, 512, 128),
        (1, 6, 2, 768, 128),
        (1, 12, 3, 1024, 128),
        (16, 64, 8, 4096, 128),
        (1, 520, 8, 16384, 128),
    ],
)
def test_gqa_supported_shape_constraint(shape):
    assert _is_supported_gqa_shape(shape)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 8, 1, 256, 128),
        (1, 0, 1, 256, 128),
        (1, 8, 0, 256, 128),
        (1, 27, 8, 256, 128),
        (1, 8, 1, 128, 128),
        (1, 8, 1, 200, 128),
        (1, 8, 1, 384, 128),
        (1, 8, 1, 256, 256),
    ],
)
def test_gqa_unsupported_shape_constraint(shape):
    assert not _is_supported_gqa_shape(shape)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 128), id="group1"),
        pytest.param((1, 1, 1, 512, 128), id="group1-two-kv-tiles"),
        pytest.param((1, 2, 1, 256, 128), id="group2"),
        pytest.param((2, 3, 1, 256, 128), id="group3-batch2"),
        pytest.param((1, 4, 1, 256, 128), id="group4"),
        pytest.param((1, 8, 1, 512, 128), id="group8-two-kv-tiles"),
        pytest.param((2, 2, 2, 512, 128), id="mha-hkv2-batch2"),
        pytest.param((1, 4, 2, 256, 128), id="group2-hkv2"),
        pytest.param((2, 6, 3, 512, 128), id="group2-hkv3-batch2"),
    ],
)
def test_gqa_supported_shapes_end_to_end_gfx950(shape, causal):
    case = _make_gqa_smoke_case(shape, causal=causal, seed=17)
    actual_grads = fa_backward(*case.kernel_args)
    assert len(actual_grads) == len(case.grads) == 3
    for actual, expected in zip(actual_grads, case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_gqa_causal_zero_scale_gfx950():
    case = _make_gqa_smoke_case((1, 2, 1, 256, 128), causal=True, seed=17, sm_scale=0.0)
    actual_grads = fa_backward(*case.kernel_args)
    assert len(actual_grads) == len(case.grads) == 3
    for actual, expected in zip(actual_grads, case.grads):
        assert torch.isfinite(actual).all()
        if torch.count_nonzero(expected).item() == 0:
            assert torch.count_nonzero(actual).item() == 0
        else:
            assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_gqa_causal_negative_scale_gfx950():
    case = _make_gqa_smoke_case((1, 2, 1, 256, 128), causal=True, seed=17, sm_scale=-0.125)
    actual_grads = fa_backward(*case.kernel_args)
    assert len(actual_grads) == len(case.grads) == 3
    for actual, expected in zip(actual_grads, case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    ("q_shape", "k_shape"),
    [
        pytest.param((1, 6, 256, 128), (1, 4, 256, 128), id="nondivisible-heads"),
        pytest.param((1, 8, 384, 128), (1, 1, 384, 128), id="nonmultiple-sequence"),
    ],
)
def test_fa_backward_rejects_unsupported_gqa_signature(q_shape, k_shape, causal):
    q = torch.empty(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q_shape[:-1], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="supported GQA shapes"):
        fa_backward(q, k, k, q, q, lse, 0.5, causal)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 128), id="group1"),
        pytest.param((1, 1, 1, 512, 128), id="group1-two-kv-tiles"),
        pytest.param((1, 64, 8, 4096, 128), id="group8"),
    ],
)
def test_gqa_main_kernel_is_spill_free_gfx950(shape, causal, monkeypatch):
    batch, hq, hk, n_ctx, head_dim = shape
    q = torch.zeros((batch, hq, n_ctx, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((batch, hk, n_ctx, head_dim), device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros((batch, hq, n_ctx), device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(lse)
    dq_acc = torch.zeros_like(q)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)

    monkeypatch.delenv(_D128_REVERSE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(_D128_SINK_INSTS_ENV, raising=False)
    monkeypatch.delenv(_D128_REGCLASS_PRIORITY_ENV, raising=False)
    _attn_bwd_dkdv_dq_d128_gqa_kernel.device_caches.clear()
    _run_bwd_d128_gqa(
        q,
        k,
        k,
        q,
        lse,
        delta,
        dq_acc,
        dq,
        dk,
        dv,
        head_dim**-0.5,
        causal,
    )
    device = torch.cuda.current_device()
    compiled = tuple(_attn_bwd_dkdv_dq_d128_gqa_kernel.device_caches[device][0].values())
    assert len(compiled) == 1
    assert compiled[0].n_spills == 0
    ttir = compiled[0].asm["ttir"]
    amdgcn = compiled[0].asm["amdgcn"]
    assert "scratch_load" not in amdgcn
    assert "scratch_store" not in amdgcn
    expected_mask_regions = 2 if causal else 0
    assert ttir.count("amdg.rematerialized_range 0 to 256 identity 32") == expected_mask_regions
    assert ttir.count("amdg.rematerialized_range 0 to 16 identity 33") == expected_mask_regions
    if causal:
        llir = compiled[0].asm["llir"]
        for mask in ("0x1fff01ff", "0x3fff03ff", "0x7fff07ff", "0xffff0fff"):
            assert mask not in amdgcn
        assert "~{vcc},~{scc}" not in llir


def test_d256_dispatch_uses_one_configured_producer_entry(monkeypatch):
    monkeypatch.delenv("TLX_FA_BWD_FORCE_STAGED", raising=False)
    full = _select_d256_dispatch(False)
    causal = _select_d256_dispatch(True)
    assert full.entry is causal.entry is _attn_bwd_dkdv_d256_producer_kernel
    assert (full.staged, full.pipelined, full.num_warps) == (False, True, 4)
    assert (causal.staged, causal.pipelined, causal.num_warps) == (False, False, 4)

    monkeypatch.setenv("TLX_FA_BWD_FORCE_STAGED", "1")
    staged = _select_d256_dispatch(True)
    assert staged.entry is _attn_bwd_dkdv_d256_producer_kernel
    assert (staged.staged, staged.pipelined, staged.num_warps) == (True, False, 2)


def test_d128_short_causal_uses_gluon_matching_tile_config():
    """Short D128 mirrors Gluon's causal and full-attention tile schedules."""
    assert _select_d128_dkdv_config((16, 27, 200, 128), True) == (32, 32, 2)
    assert _select_d128_dkdv_config((16, 27, 200, 128), False) == (32, 64, 4)
    assert _select_d128_dkdv_config((16, 27, 260, 128), True) == (64, 64, 4)


def test_d128_persistent_short_is_exact_experiment_shape():
    assert _d128_persistent_short_supported((16, 27, 200, 128), False)
    assert _d128_persistent_short_supported((16, 27, 200, 128), True)
    assert not _d128_persistent_short_supported((1, 1, 128, 128), False)
    assert not _d128_persistent_short_supported((1, 1, 256, 128), False)
    assert not _d128_persistent_short_supported((1, 1, 260, 128), False)
    assert not _d128_persistent_short_supported((32, 1, 2600, 256), False)
    assert not _d128_persistent_short_supported((1, 1, 200, 64), False)


def test_d128_legacy_disable_flags_are_ignored(monkeypatch):
    """Legacy disable knobs must not override the explicit exact opt-in."""
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    monkeypatch.setenv("TLX_FA_BWD_DISABLE_PERSISTENT_D128", "1")
    monkeypatch.setenv("TLX_FA_BWD_DISABLE_EXACT_D128", "1")
    dispatch = _select_d128_dispatch((16, 27, 200, 128), False)
    assert dispatch.entry is _attn_bwd_dkdv_dq_d128_combined_kernel
    assert dispatch.exact


def test_d128_exact_layout_dispatch_is_narrow_and_opt_in(monkeypatch):
    """The exact MFMA/LDS port is opt-in and narrow to the validated target."""
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    for causal in (False, True):
        dispatch = _select_d128_dispatch((16, 27, 200, 128), causal)
        assert dispatch.entry is _attn_bwd_dkdv_dq_d128_combined_kernel
        assert dispatch.exact
    assert _select_d128_dispatch((16, 27, 201, 128), False).entry is _attn_bwd_dkdv_d128_split_kernel
    assert not _select_d128_dispatch((32, 1, 2600, 256), False).exact

    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "0")
    assert _select_d128_dispatch((16, 27, 200, 128), False).entry is _attn_bwd_dkdv_d128_split_kernel


def test_d128_persistent_short_dispatches_combined_when_opted_in(monkeypatch):
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "0")
    monkeypatch.setenv(_D128_PERSISTENT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    dispatch = _select_d128_dispatch((16, 27, 200, 128), False)
    assert dispatch.entry is _attn_bwd_dkdv_dq_d128_combined_kernel
    assert not dispatch.exact and not dispatch.pipelined
    assert dispatch.num_warps == 8


def test_d128_persistent_pipeline_dispatches_only_when_opted_in(monkeypatch):
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "0")
    monkeypatch.setenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    dispatch = _select_d128_dispatch((16, 27, 200, 128), False)
    assert dispatch.entry is _attn_bwd_dkdv_dq_d128_combined_kernel
    assert not dispatch.exact and dispatch.pipelined
    assert dispatch.num_warps == 4


def test_d128_persistent_short_keeps_non_target_shapes_on_split(monkeypatch):
    monkeypatch.delenv(_D128_EXACT_ENABLE_ENV, raising=False)
    monkeypatch.setenv(_D128_PERSISTENT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    assert _select_d128_dispatch((1, 1, 128, 128), False).entry is _attn_bwd_dkdv_d128_split_kernel
    assert _select_d128_dispatch((1, 1, 256, 128), True).entry is _attn_bwd_dkdv_d128_split_kernel


def test_shape_specific_launch_configs():
    assert _d128_num_warps(False) == 2
    assert _d128_num_warps(True) == 4
    assert _matrix_instr_nonkdim() == 16


def test_d128_regalloc_options_are_independent_opt_ins(monkeypatch):
    for name in (_D128_SINK_INSTS_ENV, _D128_REGCLASS_PRIORITY_ENV, _D128_REVERSE_LOCAL_ENV):
        monkeypatch.delenv(name, raising=False)
    assert _d128_regalloc_options() == {
        "sink_insts_to_avoid_spills": False,
        "regclass_priority_trumps_globalness": False,
        "reverse_local_assignment": False,
    }

    monkeypatch.setenv(_D128_SINK_INSTS_ENV, "1")
    monkeypatch.setenv(_D128_REGCLASS_PRIORITY_ENV, "1")
    assert _d128_regalloc_options() == {
        "sink_insts_to_avoid_spills": True,
        "regclass_priority_trumps_globalness": True,
        "reverse_local_assignment": False,
    }


@pytest.mark.parametrize("causal", [False, True])
def test_fa_backward_b16_h27_n200_d128(causal, monkeypatch):
    monkeypatch.delenv(_D128_EXACT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    case = make_reference_case((16, 27, 200, 128), causal)
    dq, dk, dv = fa_backward(*case.kernel_args)
    for actual, expected in zip((dq, dk, dv), case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


def test_fa_backward_b16_h27_n200_d128_causal_repeated(monkeypatch):
    """Repeat the causal BM32 tail to catch stale or non-finite LDS rows."""
    monkeypatch.delenv(_D128_EXACT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    case = make_reference_case((16, 27, 200, 128), True)
    for _ in range(5):
        dq, dk, dv = fa_backward(*case.kernel_args)
        for actual, expected in zip((dq, dk, dv), case.grads):
            assert torch.isfinite(actual).all()
            assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True])
def test_fa_backward_b16_h27_n200_d128_persistent_opt_in(causal, monkeypatch):
    # Keep the fused port exercised without allowing its experimental status to
    # change the production/default correctness test above.
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "0")
    monkeypatch.setenv(_D128_PERSISTENT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    case = make_reference_case((16, 27, 200, 128), causal)
    dq, dk, dv = fa_backward(*case.kernel_args)
    for actual, expected in zip((dq, dk, dv), case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True])
def test_fa_backward_b16_h27_n200_d128_persistent_pipeline_opt_in(causal, monkeypatch):
    # The async Q/dO ring has separate ownership and wait-group ordering from
    # the older combined experiment, so keep a BF16 runtime check for both
    # masks in addition to the selector-only coverage.
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "0")
    monkeypatch.setenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    case = make_reference_case((16, 27, 200, 128), causal)
    dq, dk, dv = fa_backward(*case.kernel_args)
    for actual, expected in zip((dq, dk, dv), case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True])
def test_fa_backward_b16_h27_n200_d128_exact_layout(causal, monkeypatch):
    monkeypatch.setenv(_D128_EXACT_ENABLE_ENV, "1")
    monkeypatch.delenv(_D128_PERSISTENT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(_D128_PERSISTENT_PIPE_ENABLE_ENV, raising=False)
    case = make_reference_case((16, 27, 200, 128), causal)
    dq, dk, dv = fa_backward(*case.kernel_args)
    for actual, expected in zip((dq, dk, dv), case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True])
def test_fa_backward_b32_h1_n2600_d256(causal):
    case = make_reference_case((32, 1, 2600, 256), causal)
    dq, dk, dv = fa_backward(*case.kernel_args)
    for actual, expected in zip((dq, dk, dv), case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.parametrize("causal", [False, True])
def test_d256_scratch_producer_covers_consumer(causal):
    case = make_reference_case((32, 1, 2600, 256), causal)
    dq = torch.empty_like(case.q)
    dk = torch.empty_like(case.k)
    dv = torch.empty_like(case.v)
    delta = torch.empty(case.q.shape[:-1], device=case.q.device, dtype=torch.float32)
    _run_bwd_preprocess(case.o, case.do, delta)
    _run_bwd_d256(
        case.q,
        case.k,
        case.v,
        case.do,
        case.lse,
        delta,
        dq,
        dk,
        dv,
        case.sm_scale,
        causal,
        poison_scratch=True,
    )
    assert torch.isfinite(dq).all()


_D64_VALIDATION_SHAPES = {
    "mha_square_16k_noncausal": (2, 16384, 16384, 32, 32, 64, False),
    "mha_square_16k_causal": (2, 16384, 16384, 32, 32, 64, True),
    "gqa8_square_16k_noncausal": (2, 16384, 16384, 32, 4, 64, False),
    "gqa8_square_16k_causal": (2, 16384, 16384, 32, 4, 64, True),
    "gqa8_square_4k_causal": (4, 4096, 4096, 48, 6, 64, True),
    "gqa8_rect_4k_16k_causal": (4, 4096, 16384, 48, 6, 64, True),
    "gqa8_rect_4k_8k_causal": (4, 4096, 8192, 48, 6, 64, True),
    "gqa8_rect_4k_12k_causal": (4, 4096, 12288, 48, 6, 64, True),
}
_D64_CAUSAL_GQA8_VALIDATION_CASES = (
    "gqa8_square_16k_causal",
    "gqa8_square_4k_causal",
    "gqa8_rect_4k_16k_causal",
    "gqa8_rect_4k_8k_causal",
    "gqa8_rect_4k_12k_causal",
)

# Explicit AMD buffer operations encode their per-resource byte offset as a
# nonnegative signed i32. Keep every D64 per-head resource within the same
# limit enforced by ConvertToBufferOps. The FP32 dQ accumulator is the widest
# Q-side resource; K/V and their outputs are BF16.
_AMD_BUFFER_MAX_ADDRESSABLE_BYTES = (1 << 31) - 1
_D64_HEAD_DIM = 64
_D64_SEQUENCE_ALIGNMENT = 64


def _max_aligned_d64_sequence(element_size_bytes):
    rows = _AMD_BUFFER_MAX_ADDRESSABLE_BYTES // (_D64_HEAD_DIM * element_size_bytes)
    return rows - rows % _D64_SEQUENCE_ALIGNMENT


_D64_MAX_QUERY_SEQUENCE = _max_aligned_d64_sequence(torch.float32.itemsize)
_D64_MAX_KV_SEQUENCE = _max_aligned_d64_sequence(torch.bfloat16.itemsize)
_D64_SHAPE_CONSTRAINT = ("D64 shapes require matching positive batches, positive Hq/Hkv, "
                         "Hq % Hkv == 0, SQ/SKV >= 256, SQ/SKV multiples of 64, "
                         f"SQ <= {_D64_MAX_QUERY_SEQUENCE}, SKV <= {_D64_MAX_KV_SEQUENCE}, and D == 64")


def _is_supported_d64_shape(q_shape, k_shape):
    """Return whether dense BF16 D64 tensors match the gfx950 contract."""
    if len(q_shape) != 4 or len(k_shape) != 4:
        return False
    batch, hq, sq, d = q_shape
    k_batch, hkv, skv, k_d = k_shape
    return (batch >= 1 and batch == k_batch and hq >= 1 and hkv >= 1 and hq % hkv == 0 and sq >= 256 and skv >= 256
            and sq % 64 == 0 and skv % 64 == 0 and sq <= _D64_MAX_QUERY_SEQUENCE and skv <= _D64_MAX_KV_SEQUENCE
            and d == k_d == 64)


def _is_d64_fused_n256_eligible(q_shape, k_shape, causal, *, arch, cu_count):
    if not _is_supported_d64_shape(q_shape, k_shape):
        return False
    if causal or arch is None or not arch.startswith("gfx950"):
        return False
    if cu_count is None or cu_count < 1:
        return False
    batch, hq, sq, _d = q_shape
    _k_batch, hkv, skv, _k_d = k_shape
    group_size = hq // hkv
    owner_ctas = batch * hq * triton.cdiv(skv, 256)
    return (group_size in (1, 8) and sq >= 4096 and skv >= 4096 and sq % 64 == 0 and skv % 256 == 0
            and owner_ctas >= cu_count)


_D64_MHA_POSITIVE = 0
_D64_GQA_SIGNED = 1
_D64_LSE_NATURAL_LOG = 0
_D64_LSE_NEG_LOG2E = 1
_D64_DELTA_POSITIVE = 0
_D64_DELTA_NEGATED = 1
_D64_CAUSAL_GQA8_KV_SPLITS = 4

# With zero QK scores, selected dQ internally represents LSE as
# -log(valid_keys) / scale. Keep that unavoidable entropy term within half
# the FP32 range so the input-dependent score contribution retains headroom.
_D64_RECIP_LSE_ENTROPY_LIMIT = torch.finfo(torch.float32).max / 2.0

_D64_MHA_POSITIVE_JIT = tl.constexpr(_D64_MHA_POSITIVE)
_D64_GQA_SIGNED_JIT = tl.constexpr(_D64_GQA_SIGNED)
_D64_LSE_NATURAL_LOG_JIT = tl.constexpr(_D64_LSE_NATURAL_LOG)
_D64_LSE_NEG_LOG2E_JIT = tl.constexpr(_D64_LSE_NEG_LOG2E)
_D64_DELTA_POSITIVE_JIT = tl.constexpr(_D64_DELTA_POSITIVE)
_D64_DELTA_NEGATED_JIT = tl.constexpr(_D64_DELTA_NEGATED)
_D64_CAUSAL_GQA8_KV_SPLITS_JIT = tl.constexpr(_D64_CAUSAL_GQA8_KV_SPLITS)

_D64_GQA_SPLIT_FAST = "split_fast"
_D64_GQA_XCD = "xcd"
_D64_GQA_XCD_N_FAST = "xcd_n_fast"

_D64_GQA_INDEPENDENT_D32 = "independent_d32"
_D64_GQA_INTERLEAVED_D32 = "interleaved_d32"
_D64_GQA_DIRECT_D64 = "direct_d64"

_D64_GQA_INDEPENDENT_D32_JIT = tl.constexpr(_D64_GQA_INDEPENDENT_D32)
_D64_GQA_INTERLEAVED_D32_JIT = tl.constexpr(_D64_GQA_INTERLEAVED_D32)
_D64_GQA_DIRECT_D64_JIT = tl.constexpr(_D64_GQA_DIRECT_D64)


@dataclasses.dataclass(frozen=True)
class _D64DQLaunch:
    launch_tiles: int
    skip_owner_tail: bool
    owner_pid_base: int
    launch_q_tiles: int
    owner_fragments: int
    grid_owner_m: int


@dataclasses.dataclass(frozen=True)
class _D64Dispatch:
    family: str
    owner_rows: int
    key_rows: int
    kv_splits: int
    selected_causal: bool = False
    stat_mode: int = _D64_MHA_POSITIVE
    dq_logical_n: int = 64
    dq_use_xcd: bool = False
    dq_launches: tuple[_D64DQLaunch, ...] = ()
    gqa_grid_mode: str | None = None
    cyclic_query_split: bool = False
    dkdv_lifetime: str | None = None


_D64_NONCAUSAL_FAMILIES = frozenset({"noncausal_direct_n256", "noncausal_fused_n256"})
_D64_RETAINED_CAUSAL_FAMILIES = frozenset({"causal_m192", "causal_m256"})
_D64_SELECTED_CAUSAL_FAMILIES = frozenset({"causal_scheduled_mha", "causal_scheduled_gqa8"})
_D64_DISPATCH_FAMILIES = (_D64_NONCAUSAL_FAMILIES | _D64_RETAINED_CAUSAL_FAMILIES | _D64_SELECTED_CAUSAL_FAMILIES)

_D64_DQ_KV_STAGES = 2


def _d64_selected_causal_owner_rows(sq, skv, group_size):
    deep_square = (sq == skv and sq % 256 == 0 and (sq >= 16384 or (group_size == 8 and sq >= 4096)))
    deep_gqa8_rectangle = (group_size == 8 and sq >= 4096 and skv >= 2 * sq and sq % 256 == 0)
    return 256 if deep_square or deep_gqa8_rectangle else 192


def _d64_selected_causal_logical_n(sq, skv, group_size):
    # Deep GQA8 rectangles use two N32 slices to keep the live score/dP
    # footprint of four-fragment M256 owners scratch-free while publishing K/V
    # once per owner; N64 remains better for shallower rectangular recurrences.
    return 32 if sq == skv or (group_size == 8 and skv >= 2 * sq) else 64


def _d64_causal_gqa8_batch_stats4(sq, skv, dispatch):
    """Stage four adjacent statistics tiles for long direct-D64 walks."""
    return (dispatch.dkdv_lifetime == _D64_GQA_DIRECT_D64 and not dispatch.cyclic_query_split and sq % 256 == 0
            and skv >= sq)


def _invalid_d64_dispatch(dispatch, reason):
    raise ValueError(f"invalid D64 dispatch {dispatch.family!r}: {reason}")


def _require_d64_dispatch_variant(dispatch, family, *, stat_mode=None, kv_splits=None):
    if dispatch.family != family:
        _invalid_d64_dispatch(dispatch, f"family must be {family!r} for this route")
    if stat_mode is not None and dispatch.stat_mode != stat_mode:
        _invalid_d64_dispatch(
            dispatch,
            f"stat_mode must be {stat_mode}, got {dispatch.stat_mode}",
        )
    if kv_splits is not None and dispatch.kv_splits != kv_splits:
        _invalid_d64_dispatch(
            dispatch,
            f"kv_splits must be {kv_splits}, got {dispatch.kv_splits}",
        )


def _validate_d64_dispatch(q_shape, k_shape, causal, dispatch):
    """Reject dispatch records that would violate a D64 kernel ABI."""
    if not isinstance(dispatch, _D64Dispatch):
        raise ValueError("D64 dispatch must be a _D64Dispatch record")
    if not _is_supported_d64_shape(q_shape, k_shape):
        raise ValueError(f"unsupported D64 dispatch shapes q={q_shape}, k={k_shape}")

    family = dispatch.family
    if family not in _D64_DISPATCH_FAMILIES:
        raise ValueError(f"unknown D64 dispatch family {family!r}")
    family_is_causal = family not in _D64_NONCAUSAL_FAMILIES
    if causal != family_is_causal:
        requirement = "causal" if family_is_causal else "noncausal"
        _invalid_d64_dispatch(dispatch, f"family requires {requirement} attention")

    _batch, hq, sq, _d = q_shape
    _k_batch, hkv, skv, _k_d = k_shape
    group_size = hq // hkv

    def require_field(name, expected):
        actual = getattr(dispatch, name)
        if actual != expected:
            _invalid_d64_dispatch(dispatch, f"{name} must be {expected!r}, got {actual!r}")

    def require_unselected_defaults():
        require_field("selected_causal", False)
        require_field("stat_mode", _D64_MHA_POSITIVE)
        require_field("dq_logical_n", 64)
        require_field("dq_use_xcd", False)
        require_field("dq_launches", ())
        require_field("gqa_grid_mode", None)
        require_field("cyclic_query_split", False)
        require_field("dkdv_lifetime", None)

    if family in _D64_NONCAUSAL_FAMILIES:
        require_field("owner_rows", 32)
        require_field("key_rows", 256)
        require_field("kv_splits", 8 if group_size == 8 else 1)
        require_unselected_defaults()
        return

    retained_owner_rows = (256 if sq == skv and sq >= 16384 and sq % 256 == 0 else 192)
    if family in _D64_RETAINED_CAUSAL_FAMILIES:
        expected_family = "causal_m256" if retained_owner_rows == 256 else "causal_m192"
        require_field("family", expected_family)
        require_field("owner_rows", retained_owner_rows)
        require_field("key_rows", 32 if sq == skv else 64)
        require_field("kv_splits", 4 if group_size == 8 else 1)
        require_unselected_defaults()
        return

    expected_family = ("causal_scheduled_gqa8" if group_size == 8 else "causal_scheduled_mha")
    if group_size not in (1, 8):
        _invalid_d64_dispatch(dispatch, "selected causal family requires MHA or GQA8")
    owner_rows = _d64_selected_causal_owner_rows(sq, skv, group_size)
    require_field("family", expected_family)
    require_field("owner_rows", owner_rows)
    require_field("selected_causal", True)
    require_field("dq_logical_n", _d64_selected_causal_logical_n(sq, skv, group_size))
    require_field("dq_use_xcd", _d64_use_dq_xcd(_batch, hkv, sq, skv, owner_rows))

    def require_valid_dq_launches():
        owners = triton.cdiv(sq, owner_rows)
        fragments = owner_rows // 64
        valid_dq_launches = {
            (_D64DQLaunch(owners, True, 0, 0, fragments, 0), ),
        }
        if (owner_rows == 192 and sq >= 8192 and dispatch.dq_use_xcd and sq % 192 == 128 and owners > 1):
            valid_dq_launches.add((
                _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
                _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
            ))
        if dispatch.dq_launches not in valid_dq_launches:
            _invalid_d64_dispatch(dispatch, "dq_launches must match a full or peeled owner plan")

    if family == "causal_scheduled_mha":
        require_field("key_rows", 64)
        require_field("kv_splits", 1)
        require_field("stat_mode", _D64_MHA_POSITIVE)
        require_field("gqa_grid_mode", None)
        require_field("cyclic_query_split", False)
        require_field("dkdv_lifetime", None)
        require_valid_dq_launches()
        return

    require_field("key_rows", 128)
    require_field("kv_splits", 4)
    require_field("stat_mode", _D64_GQA_SIGNED)
    if dispatch.gqa_grid_mode not in {
            _D64_GQA_SPLIT_FAST,
            _D64_GQA_XCD,
            _D64_GQA_XCD_N_FAST,
    }:
        _invalid_d64_dispatch(dispatch, "unknown GQA grid mode")
    if (dispatch.gqa_grid_mode != _D64_GQA_SPLIT_FAST and (_batch * hkv) % 8 != 0):
        _invalid_d64_dispatch(dispatch, "GQA XCD grid requires B * Hkv divisible by 8")
    require_field("dkdv_lifetime", _d64_gqa_lifetime(sq, skv))
    if dispatch.cyclic_query_split and not (dispatch.gqa_grid_mode == _D64_GQA_XCD_N_FAST and sq == skv and skv >= 16384
                                            and (_batch * hkv) % 8 == 0):
        _invalid_d64_dispatch(dispatch, "cyclic_query_split requires a deep square XCD N-fast grid")
    require_valid_dq_launches()


def _validate_d64_sm_scale(sm_scale):
    try:
        value = float(sm_scale)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("D64 sm_scale must be finite and nonzero") from None
    if not math.isfinite(value) or value == 0.0:
        raise ValueError("D64 sm_scale must be finite and nonzero")
    return value


def _is_d64_scheduled_causal_eligible(
    q_shape,
    k_shape,
    causal,
    *,
    arch,
    cu_count,
    sm_scale,
    bases_aligned_16,
):
    if not _is_supported_d64_shape(q_shape, k_shape) or not causal:
        return False
    if arch is None or not arch.startswith("gfx950"):
        return False
    if cu_count is None or cu_count < 1 or not bases_aligned_16:
        return False
    try:
        scale = float(sm_scale)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(scale) or scale <= 0.0:
        return False

    batch, hq, sq, _d = q_shape
    _k_batch, hkv, skv, _k_d = k_shape
    if sq > skv or sq % 64 != 0 or skv % 64 != 0:
        return False
    if math.log(skv) > scale * _D64_RECIP_LSE_ENTROPY_LIMIT:
        return False
    group_size = hq // hkv
    if group_size == 1:
        if sq < 4096 or skv < 4096:
            return False
    elif group_size == 8:
        if sq < 1024 or skv < 1024 or skv % 128 != 0:
            return False
    else:
        return False

    owner_rows = _d64_selected_causal_owner_rows(sq, skv, group_size)
    if batch * hq * triton.cdiv(sq, owner_rows) < 2 * cu_count:
        return False
    if group_size == 8:
        producer_owners = batch * hkv * 4 * triton.cdiv(skv, 128)
    else:
        producer_owners = batch * hkv * triton.cdiv(skv, 64)
    return producer_owners >= 2 * cu_count


def _select_d64_dispatch(
    q_shape,
    k_shape,
    causal,
    *,
    arch=None,
    cu_count=None,
    sm_scale=None,
    bases_aligned_16=False,
):
    """Select D64 ownership from tensor shape and device structure."""
    if not _is_supported_d64_shape(q_shape, k_shape):
        raise ValueError(f"unsupported D64 dispatch shapes q={q_shape}, k={k_shape}")
    _batch, _hq, sq, _d = q_shape
    _k_batch, hkv, skv, _k_d = k_shape
    group_size = _hq // hkv
    if not causal:
        fused = _is_d64_fused_n256_eligible(q_shape, k_shape, causal, arch=arch, cu_count=cu_count)
        return _D64Dispatch(
            "noncausal_fused_n256" if fused else "noncausal_direct_n256",
            owner_rows=32,
            key_rows=256,
            kv_splits=8 if group_size == 8 else 1,
        )
    if sq == skv and sq >= 16384 and sq % 256 == 0:
        retained = _D64Dispatch(
            "causal_m256",
            owner_rows=256,
            key_rows=32,
            kv_splits=4 if group_size == 8 else 1,
        )
    else:
        retained = _D64Dispatch(
            "causal_m192",
            owner_rows=192,
            key_rows=32 if sq == skv else 64,
            kv_splits=4 if group_size == 8 else 1,
        )
    if not _is_d64_scheduled_causal_eligible(
            q_shape,
            k_shape,
            causal,
            arch=arch,
            cu_count=cu_count,
            sm_scale=sm_scale,
            bases_aligned_16=bases_aligned_16,
    ):
        return retained

    owner_rows = _d64_selected_causal_owner_rows(sq, skv, group_size)
    dq_logical_n = _d64_selected_causal_logical_n(sq, skv, group_size)
    dq_use_xcd = _d64_use_dq_xcd(_batch, hkv, sq, skv, owner_rows)
    dq_launches = _d64_dq_launch_plan(
        _batch,
        _hq,
        hkv,
        sq,
        skv,
        owner_rows,
        cu_count,
        True,
    )
    if group_size == 8:
        grid_mode, cyclic_query_split = _d64_gqa_grid_policy(_batch, hkv, sq, skv, cu_count)
        return _D64Dispatch(
            "causal_scheduled_gqa8",
            owner_rows=owner_rows,
            key_rows=128,
            kv_splits=_D64_CAUSAL_GQA8_KV_SPLITS,
            selected_causal=True,
            stat_mode=_D64_GQA_SIGNED,
            dq_logical_n=dq_logical_n,
            dq_use_xcd=dq_use_xcd,
            dq_launches=dq_launches,
            gqa_grid_mode=grid_mode,
            cyclic_query_split=cyclic_query_split,
            dkdv_lifetime=_d64_gqa_lifetime(sq, skv),
        )
    return _D64Dispatch(
        "causal_scheduled_mha",
        owner_rows=owner_rows,
        key_rows=64,
        kv_splits=1,
        selected_causal=True,
        stat_mode=_D64_MHA_POSITIVE,
        dq_logical_n=dq_logical_n,
        dq_use_xcd=dq_use_xcd,
        dq_launches=dq_launches,
    )


def _d64_causal_stat_values(o, do, lse, sm_scale, stat_mode):
    positive = torch.sum(o.float() * do.float(), dim=-1)
    if stat_mode == _D64_MHA_POSITIVE:
        return positive, None
    if stat_mode == _D64_GQA_SIGNED:
        return -positive, -lse.float() * math.log2(math.e)
    raise ValueError(f"unknown D64 stat mode {stat_mode!r}")


def _d64_causal_owner_interval(physical_owner, sq, owner_rows):
    owners = triton.cdiv(sq, owner_rows)
    if not 0 <= physical_owner < owners:
        raise ValueError("physical owner is outside the dQ grid")
    pad = owners * owner_rows - sq
    reverse_owner = owners - 1 - physical_owner
    raw = reverse_owner * owner_rows
    return max(raw - pad, 0), min(raw + owner_rows - pad, sq)


def _d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows):
    single_fragment_tail = (owner_rows == 192 and sq == skv and sq % 192 == 64 and 4096 <= sq < 5120 and hkv % 8 == 0)
    return (batch * hkv) % 8 == 0 and not single_fragment_tail


def _d64_gqa_grid_policy(batch, hkv, sq, skv, cu_count):
    cyclic = (sq == skv and skv >= 16384 and batch * hkv * 4 >= cu_count and (batch * hkv) % 8 == 0)
    if cyclic:
        return _D64_GQA_XCD_N_FAST, True
    if (batch * hkv) % 8 != 0:
        return _D64_GQA_SPLIT_FAST, cyclic
    return _D64_GQA_XCD, False


def _d64_decode_gqa_pid(pid, batch, hkv, skv, grid_mode):
    nt = triton.cdiv(skv, 128)
    value = pid
    if grid_mode == _D64_GQA_SPLIT_FAST:
        split = value % 4
        value //= 4
        out_hkv = value % hkv
        value //= hkv
        n = value % nt
        out_batch = value // nt
        return out_batch, out_hkv, split, n
    xcd = value % 8
    value //= 8
    if grid_mode == _D64_GQA_XCD_N_FAST:
        n = value % nt
        value //= nt
        split = value % 4
        bkv_group = value // 4
    elif grid_mode == _D64_GQA_XCD:
        split = value % 4
        value //= 4
        n = value % nt
        bkv_group = value // nt
    else:
        raise ValueError(f"unknown GQA grid mode {grid_mode!r}")
    bkv = bkv_group * 8 + xcd
    return bkv // hkv, bkv % hkv, split, n


def _d64_causal_physical_frontier(n0, sq, skv, block_m, block_n):
    diff = skv - sq
    start_m_blk = max((n0 - diff) // block_m, 0)
    masked = tuple(m_blk for m_blk in range(start_m_blk, triton.cdiv(sq, block_m))
                   if n0 + block_n - 1 > m_blk * block_m + diff)
    return start_m_blk, masked


def _d64_gqa_split_ownership(split, query_blocks, cyclic):
    if not 0 <= split < 4:
        raise ValueError("GQA split must be in [0, 4)")
    if cyclic:
        return tuple((head, m_blk) for head in range(8) for m_blk in range(query_blocks) if m_blk % 4 == split)
    return tuple((head, m_blk) for head in (2 * split, 2 * split + 1) for m_blk in range(query_blocks))


def _d64_gqa_lifetime(sq, skv):
    if skv > sq:
        # Shallow rectangles retain the D32 peel for their frequently odd
        # bottom-right frontier.  At two or more query lengths, the longer
        # recurrence amortizes full-width D64 ownership and avoids the D32
        # split/join schedule without increasing the compiled VGPR footprint.
        if skv < 2 * sq:
            return _D64_GQA_INTERLEAVED_D32
        return _D64_GQA_DIRECT_D64
    if sq == skv and sq >= 1024:
        # Four-tile statistics staging amortizes the full-width recurrence
        # across square walks while avoiding the D32 split/join epilogue.
        return _D64_GQA_DIRECT_D64
    raise ValueError("selected GQA8 shape has no lifetime mode")


def _d64_decode_dq_pid(
    pid,
    batch,
    hq,
    hkv,
    launch_tiles,
    use_xcd,
    owner_pid_base=0,
):
    group = hq // hkv
    value = pid
    if use_xcd:
        xcd = value % 8
        value //= 8
        q_in_group = value % group
        value //= group
        local_owner = value % launch_tiles
        bkv_group = value // launch_tiles
        bkv = bkv_group * 8 + xcd
        out_hkv = bkv % hkv
        out_batch = bkv // hkv
    else:
        out_hkv = value % hkv
        value //= hkv
        q_in_group = value % group
        value //= group
        local_owner = value % launch_tiles
        out_batch = value // launch_tiles
    return out_batch, out_hkv * group + q_in_group, owner_pid_base + local_owner


def _d64_encode_dq_pid(
    batch_id,
    hq_id,
    physical_owner,
    batch,
    hq,
    hkv,
    launch_tiles,
    use_xcd,
    owner_pid_base=0,
):
    group = hq // hkv
    hkv_id, q_in_group = divmod(hq_id, group)
    local_owner = physical_owner - owner_pid_base
    if use_xcd:
        bkv = batch_id * hkv + hkv_id
        bkv_group, xcd = divmod(bkv, 8)
        return (((bkv_group * launch_tiles + local_owner) * group + q_in_group) * 8 + xcd)
    return ((batch_id * launch_tiles + local_owner) * group + q_in_group) * hkv + hkv_id


def _d64_dq_launch_plan(
    batch,
    hq,
    hkv,
    sq,
    skv,
    owner_rows,
    cu_count,
    host_skip_owner_tail,
):
    owners = triton.cdiv(sq, owner_rows)
    fragments = owner_rows // 64
    use_xcd = _d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows)
    peel = (owner_rows == 192 and sq >= 8192 and use_xcd and batch * hq >= cu_count and host_skip_owner_tail
            and sq % 192 == 128 and owners > 1)
    if peel:
        return (
            _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
            _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
        )
    return (_D64DQLaunch(owners, host_skip_owner_tail, 0, 0, fragments, 0), )


# These intentionally duplicate the scalar formulas in the JIT kernels as
# host-testable Python reference models.  Keep them separate: sharing helpers
# across ordinary Python and TLX JIT control flow would blur that boundary.
def _d64_causal_dq_key_blocks(owner_start, owner_rows, sq, skv, block_n):
    """Python reference model for the JIT dQ bottom-right key frontier."""
    return min(skv, owner_start + owner_rows + skv - sq + block_n - 1) // block_n


def _d64_causal_dkdv_first_query_block(key_start, sq, skv, block_m):
    """Python reference model for the JIT dK/dV bottom-right query frontier."""
    return max(0, key_start - (skv - sq)) // block_m


def _d64_causal_triangular_tail_schedule(owner_fragments, valid_fragments, tail_step):
    """Python reference for one uniform causal-tail fragment visit."""
    return tuple("skip" if fragment < tail_step or fragment >= valid_fragments else "masked" if fragment ==
                 tail_step else "unmasked" for fragment in range(owner_fragments))


def _make_d64_gqa_smoke_case(shape=(1, 1, 1, 256, 256, 64), causal=False, seed=0, sm_scale=None):
    """Build a small D64 MHA/GQA reference with bottom-right causality."""
    batch, hq, hkv, sq, skv, head_dim = shape
    assert batch >= 1 and hq >= 1 and hkv >= 1 and hq % hkv == 0
    assert head_dim == 64
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn((batch, hq, sq, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, hkv, skv, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, hkv, skv, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(q.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    group_size = hq // hkv
    causal_mask = None
    if causal:
        query_positions = torch.arange(sq, device="cuda")[:, None]
        key_positions = torch.arange(skv, device="cuda")[None, :]
        causal_mask = key_positions > query_positions + (skv - sq)

    for batch_idx in range(batch):
        for query_head in range(hq):
            kv_head = query_head // group_size
            q_ref = q[batch_idx, query_head].float().requires_grad_(True)
            k_ref = k[batch_idx, kv_head].float().requires_grad_(True)
            v_ref = v[batch_idx, kv_head].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            o_ref = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
            grads = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do[batch_idx, query_head].float())
            with torch.no_grad():
                o[batch_idx, query_head].copy_(o_ref)
                lse[batch_idx, query_head].copy_(lse_ref)
                dq[batch_idx, query_head].copy_(grads[0])
                dk[batch_idx, kv_head].add_(grads[1])
                dv[batch_idx, kv_head].add_(grads[2])

    return ReferenceCase(q, k, v, o, do, lse, sm_scale, causal, (dq, dk, dv))


def _make_d64_aten_case(shape, seed, causal=False, sm_scale=None):
    """Build a full-size D64 case without materializing a dense score matrix."""
    batch, hq, hkv, sq, skv, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    def random(tensor_shape):
        return torch.randn(
            tensor_shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()

    q = random((batch, hq, sq, head_dim))
    k = random((batch, hkv, skv, head_dim))
    v = random((batch, hkv, skv, head_dim))
    do = random(q.shape)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, causal, False, scale=sm_scale)
    out, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do,
        q,
        k,
        v,
        out,
        lse,
        cum_q,
        cum_k,
        max_q,
        max_k,
        0.0,
        causal,
        rng,
        unused,
        scale=sm_scale,
    )
    return ReferenceCase(
        q,
        k,
        v,
        out.contiguous(),
        do,
        lse.contiguous(),
        sm_scale,
        causal,
        tuple(reference),
    )


@triton.jit
def _attn_bwd_d64_fused_n256_update(
    q_t,
    do_t,
    q_nd,
    do_nd,
    k_nm,
    kt_dm,
    v_nm,
    lse,
    delta,
    dq_acc_base,
    offs_m,
    dk,
    dv,
    SM_SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    mma_dm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    ds_op0_nd: tl.constexpr,
    ds_op1_dm: tl.constexpr,
):
    log2e: tl.constexpr = 1.4426950408889634
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    scores = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
    scores = tl.dot(k_nm, q_t, acc=scores, out_dtype=tl.float32)
    scale_full = tlx.require_layout(
        tl.full((BLOCK_N, BLOCK_M), SM_SCALE * log2e, tl.float32),
        mma_nm,
        pin=False,
    )
    lse_full = tlx.require_layout(
        tl.broadcast_to(lse[None, :] * log2e, (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    p = tlx.require_layout(tl.math.exp2(scores * scale_full - lse_full), mma_nm, pin=False)

    # Score and dK/dV use compatible CDNA4 16x16x32 MFMA ownership. Preserve
    # that ownership instead of introducing a generic linear permutation.
    p_nd = tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)
    dv = tl.dot(p_nd, do_nd, acc=dv, out_dtype=tl.float32)

    dp = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
    dp = tl.dot(v_nm, do_t, acc=dp, out_dtype=tl.float32)
    delta_full = tlx.require_layout(
        tl.broadcast_to(delta[None, :], (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    ds = p * (dp - delta_full)
    ds_bf16 = ds.to(tl.bfloat16)

    ds_nd = tlx.require_layout(ds_bf16, ds_op0_nd, pin=False)
    dk = tl.dot(ds_nd, q_nd, acc=dk, out_dtype=tl.float32)

    ds_dm = tlx.require_layout(ds_bf16, ds_op1_dm, pin=False)
    dq = tlx.zeros((D, BLOCK_M), tl.float32, layout=mma_dm)
    dq = tl.dot(kt_dm, ds_dm, acc=dq, out_dtype=tl.float32)
    dq_scale = tlx.require_layout(tl.full((D, BLOCK_M), SM_SCALE, tl.float32), mma_dm, pin=False)
    dq = dq * dq_scale
    offs_d = tl.arange(0, D)
    dq_offsets = offs_m[None, :] * D + offs_d[:, None]
    dq_offsets = tlx.require_layout(dq_offsets.to(tl.int32), mma_dm, pin=False)
    tlx.buffer_atomic_add(
        dq_acc_base,
        dq_offsets,
        dq,
        sem="relaxed",
        contiguity=1,
    )
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    return dk, dv


@triton.jit
def _attn_bwd_d64_fused_n256_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ_ACC,
    DK_OWNER,
    DV_OWNER,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 32 and BLOCK_N == 256 and D == 64)
    tl.static_assert(SQ % 64 == 0 and SKV % 256 == 0)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert((HQ // HKV == 1 and KV_SPLITS == 1) or (HQ // HKV == 8 and KV_SPLITS == 8))
    pid_n = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_b = tl.program_id(2)
    group_size: tl.constexpr = HQ // HKV
    pid_hkv = pid_hq // group_size
    pid_split = pid_hq % group_size
    q_head = (pid_b * HQ + pid_hq).to(tl.int64)
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    q_base = q_head * SQ * D
    kv_base = kv_head * SKV * D
    stats_base = q_head * SQ
    dq_acc_base = DQ_ACC + q_base

    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_dm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    kv_async_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
        ),
        stride=(
            (8, 16, 32, 64, 128, 256, 4096, 8192),
            (1, 2, 4, 512, 1024, 2048),
        ),
    )
    k_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    qt_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    v_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    do_t_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=4)
    do_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=4)
    ds_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=4)
    q_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=4)
    kt_op0_dm: tl.constexpr = tlx.dot_operand_layout(0, mma_dm, k_width=4)
    ds_op1_dm: tl.constexpr = tlx.dot_operand_layout(1, mma_dm, k_width=4)

    kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_N, D],
    )
    qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_M, D],
    )
    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)
    v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_layout)
    q_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 8, layout=qdo_layout, reuse=k_buffer)
    do_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 8, layout=qdo_layout, reuse=v_buffer)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    key_ptrs = kv_base + offs_n[:, None] * D + offs_d[None, :]
    key_mask = offs_n[:, None] < SKV
    k_token = tlx.async_load(
        K + key_ptrs,
        tlx.local_view(k_buffer, 0),
        mask=key_mask,
        other=0.0,
    )
    v_token = tlx.async_load(
        V + key_ptrs,
        tlx.local_view(v_buffer, 0),
        mask=key_mask,
        other=0.0,
    )
    tlx.async_load_commit_group([k_token, v_token])
    kv_wait = tlx.async_load_wait_group(0)
    k_nm = tlx.local_load(tlx.local_view(k_buffer, 0), token=kv_wait, layout=k_op0_nm)
    kt_dm = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buffer, 0)),
        token=kv_wait,
        layout=kt_op0_dm,
    )
    v_nm = tlx.local_load(tlx.local_view(v_buffer, 0), token=kv_wait, layout=v_op0_nm)
    tl.debug_barrier()

    first_m = tl.arange(0, BLOCK_M)
    first_ptrs = q_base + first_m[:, None] * D + offs_d[None, :]
    first_q_token = tlx.async_load(Q + first_ptrs, tlx.local_view(q_ring, 0))
    first_do_token = tlx.async_load(DO + first_ptrs, tlx.local_view(do_ring, 0))
    tlx.async_load_commit_group([first_q_token, first_do_token])

    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = SQ // BLOCK_M
    for m_block in range(0, num_m_blocks):
        current_slot = m_block % 2
        next_slot = 1 - current_slot
        if m_block + 1 < num_m_blocks:
            next_m = (m_block + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
            next_ptrs = q_base + next_m[:, None] * D + offs_d[None, :]
            next_q_token = tlx.async_load(Q + next_ptrs, tlx.local_view(q_ring, next_slot))
            next_do_token = tlx.async_load(DO + next_ptrs, tlx.local_view(do_ring, next_slot))
            tlx.async_load_commit_group([next_q_token, next_do_token])
            qdo_wait = tlx.async_load_wait_group(1)
        else:
            qdo_wait = tlx.async_load_wait_group(0)

        q_view = tlx.local_view(q_ring, current_slot)
        do_view = tlx.local_view(do_ring, current_slot)
        q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait, layout=qt_op1_nm)
        do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait, layout=do_t_op1_nm)
        q_nd = tlx.local_load(q_view, token=qdo_wait, layout=q_op1_nd)
        do_nd = tlx.local_load(do_view, token=qdo_wait, layout=do_op1_nd)
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        lse = tl.load(LSE + stats_base + offs_m)
        delta = tl.load(Delta + stats_base + offs_m)
        dk, dv = _attn_bwd_d64_fused_n256_update(
            q_t,
            do_t,
            q_nd,
            do_nd,
            k_nm,
            kt_dm,
            v_nm,
            lse,
            delta,
            dq_acc_base,
            offs_m,
            dk,
            dv,
            SM_SCALE,
            D,
            BLOCK_M,
            BLOCK_N,
            mma_nm,
            mma_nd,
            mma_dm,
            p_op0_nd,
            ds_op0_nd,
            ds_op1_dm,
        )
        tl.debug_barrier()

    output_offsets = offs_n[:, None] * D + offs_d[None, :]
    output_offsets = tlx.require_layout(output_offsets.to(tl.int32), kv_async_layout, pin=False)
    dk_scale = tlx.require_layout(tl.full((BLOCK_N, D), SM_SCALE, tl.float32), mma_nd, pin=False)
    dk_out = tlx.require_layout(
        (dk * dk_scale).to(tl.bfloat16),
        kv_async_layout,
        pin=False,
    )
    dv_out = tlx.require_layout(
        dv.to(tl.bfloat16),
        kv_async_layout,
        pin=False,
    )
    if KV_SPLITS == 1:
        output_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    else:
        output_head = ((pid_b * HKV + pid_hkv) * KV_SPLITS + pid_split).to(tl.int64)
    output_base = output_head * SKV * D
    tlx.buffer_store(dk_out, DK_OWNER + output_base, output_offsets)
    tlx.buffer_store(dv_out, DV_OWNER + output_base, output_offsets)


@triton.jit
def _attn_bwd_d64_fused_dq_convert_kernel(
    DQ_ACC,
    DQ,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    base = batch_head.to(tl.int64) * N * D
    offsets = rows[:, None] * D + cols[None, :]
    mask = rows[:, None] < N
    values = tl.load(DQ_ACC + base + offsets, mask=mask, other=0.0)
    tl.store(DQ + base + offsets, values.to(tl.bfloat16), mask=mask)


@triton.jit
def _attn_bwd_dq_d64_causal_load_q64(
    Q,
    O,
    DO,
    LSE,
    DELTA,
    LSE_TERM,
    q_base,
    stats_base,
    row_start,
    store_end,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    D: tl.constexpr,
    STAT_MODE: tl.constexpr,
    q_op0_mn: tl.constexpr,
):
    rows = row_start + tl.arange(0, 64)
    lse = tl.load(LSE + stats_base + rows, mask=rows < store_end, other=0.0)
    cols = tl.arange(0, D)
    offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, q_op0_mn, pin=False)
    mask = tlx.require_layout(tl.broadcast_to(rows[:, None] < store_end, offsets.shape), q_op0_mn, pin=False)
    zero = tlx.zeros((64, D), tl.bfloat16, layout=q_op0_mn)
    do = tlx.buffer_load(DO + q_base, offsets, mask=mask, other=zero, contiguity=8)
    o = tlx.buffer_load(O + q_base, offsets, mask=mask, other=zero, contiguity=8)
    product = o.to(tl.float32) * do.to(tl.float32)
    product = tlx.release_layout(product)
    positive = tl.sum(product, axis=1)
    # Q is independent of the O*dO reduction. Loading it after that short-lived
    # product keeps the scheduler from extending Q's interval through the
    # reduction prologue of every resident owner fragment.
    q = tlx.buffer_load(Q + q_base, offsets, mask=mask, other=zero, contiguity=8)
    stat_mask = rows < store_end
    if STAT_MODE == _D64_MHA_POSITIVE_JIT:
        score_lse_term = -lse * (1.0 / SM_SCALE)
        tl.store(DELTA + stats_base + rows, positive, mask=stat_mask)
    else:
        tl.static_assert(STAT_MODE == _D64_GQA_SIGNED_JIT)
        signed = -positive
        # dQ and the producer share the log2-domain term. Publish it once per
        # row instead of rescaling it in every resident K owner.
        producer_lse_term = -lse * 1.4426950408889634
        # Form both independent statistics before either publication. This
        # keeps their arithmetic available to the scheduler ahead of the
        # side-effecting stores and shortens the dQ epilogue dependency chain.
        tl.store(DELTA + stats_base + rows, signed, mask=stat_mask)
        tl.store(LSE_TERM + stats_base + rows, producer_lse_term, mask=stat_mask)
        score_lse_term = producer_lse_term
        # Q is score-only and remains resident across every owned K slice.
        # Fold the constant log2 softmax scale here once per owner rather
        # than multiplying every score fragment after its MFMA.
        q_scale = tlx.require_layout(
            tl.full((64, D), SM_SCALE * 1.4426950408889634, tl.float32),
            q_op0_mn,
            pin=False,
        )
        q = tlx.require_layout(
            (q.to(tl.float32) * q_scale).to(tl.bfloat16),
            q_op0_mn,
            pin=False,
        )
    q = tlx.require_layout(q, q_op0_mn, pin=False)
    do = tlx.require_layout(do, q_op0_mn, pin=False)
    return q, do, score_lse_term, positive, rows


@triton.jit
def _attn_bwd_dq_d64_causal_step(
    dq,
    q,
    do,
    row_lse,
    row_delta,
    rows,
    k_source,
    v_source,
    kv_source,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    N_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    SCORE_PRE_SCALED: tl.constexpr,
    mma_mn: tl.constexpr,
    mma_md: tl.constexpr,
    kt_op1_mn: tl.constexpr,
    vt_op1_mn: tl.constexpr,
    ds_op0_md: tl.constexpr,
    k_op1_md: tl.constexpr,
):
    log2e: tl.constexpr = 1.4426950408889634
    dq = tlx.require_layout(dq, mma_md, pin=False)
    q = tlx.require_layout(q, tlx.dot_operand_layout(0, mma_mn, k_width=8), pin=False)
    do = tlx.require_layout(do, tlx.dot_operand_layout(0, mma_mn, k_width=8), pin=False)
    kt = tlx.require_layout(k_source, kt_op1_mn, pin=False)
    vt = tlx.require_layout(v_source, vt_op1_mn, pin=False)
    k_nd = tlx.require_layout(kv_source, k_op1_md, pin=False)
    row_lse_full = tlx.require_layout(
        tl.broadcast_to(row_lse[:, None], (64, BLOCK_N)),
        mma_mn,
        pin=False,
    )
    scores = tlx.zeros((64, BLOCK_N), tl.float32, layout=mma_mn)
    scores = scores + row_lse_full
    scores = tl.dot(q, kt, acc=scores, out_dtype=tl.float32)
    if BLOCK_N == 64:
        # End the two-group score-MFMA allocation interval before the exp tail.
        scores = tlx.amd_register_handoff(scores, register_class="vgpr", registers_per_group=2)
        scores = tlx.require_layout(scores, mma_mn, pin=False)

    # Match the independent MFMA cadence used by the tuned reference: issue
    # dP while the score recurrence is ready, before its scale/exp tail.
    row_delta_full = tlx.require_layout(
        tl.broadcast_to(row_delta[:, None], (64, BLOCK_N)),
        mma_mn,
        pin=False,
    )
    dp = tlx.zeros((64, BLOCK_N), tl.float32, layout=mma_mn)
    dp = dp - row_delta_full
    dp = tl.dot(do, vt, acc=dp, out_dtype=tl.float32)

    if not SCORE_PRE_SCALED:
        scale = tlx.require_layout(
            tl.full((64, BLOCK_N), SM_SCALE * log2e, tl.float32),
            mma_mn,
            pin=False,
        )
        scores = scores * scale
    if BLOCK_N == 32:
        # This complete-step path consumes the native one-group result directly;
        # unlike the split score/dP/finish path, it has no helper boundary.
        scores = tlx.require_layout(scores, mma_mn, pin=False)
    if APPLY_MASK:
        cols = n0 + N_OFFSET + tl.arange(0, BLOCK_N)
        valid = cols[None, :] <= rows[:, None] + (SKV - SQ)
        valid = tlx.require_layout(valid, mma_mn, pin=False)
        negative_inf = tlx.require_layout(
            tl.full((64, BLOCK_N), float("-inf"), tl.float32),
            mma_mn,
            pin=False,
        )
        scores = tl.where(valid, scores, negative_inf)
    p = tlx.require_layout(tl.math.exp2(scores), mma_mn, pin=False)
    ds = tlx.amd_register_handoff(
        p * dp,
        register_class="vgpr",
        registers_per_group=2,
    )
    ds = tlx.require_layout(ds.to(tl.bfloat16), ds_op0_md, pin=False)
    dq = tl.dot(ds, k_nd, acc=dq, out_dtype=tl.float32)
    return tlx.require_layout(dq, mma_md, pin=False)


@triton.jit
def _attn_bwd_dq_d64_causal_score32(
    q,
    row_lse,
    kt,
    SM_SCALE: tl.constexpr,
    mma_mn: tl.constexpr,
    q_op0_mn: tl.constexpr,
    kt_op1_mn: tl.constexpr,
    SCORE_PRE_SCALED: tl.constexpr,
):
    log2e: tl.constexpr = 1.4426950408889634
    q = tlx.require_layout(q, q_op0_mn, pin=False)
    kt = tlx.require_layout(kt, kt_op1_mn, pin=False)
    row_lse_full = tlx.require_layout(tl.broadcast_to(row_lse[:, None], (64, 32)), mma_mn, pin=False)
    scores = tlx.zeros((64, 32), tl.float32, layout=mma_mn) + row_lse_full
    scores = tl.dot(q, kt, acc=scores, out_dtype=tl.float32)
    if not SCORE_PRE_SCALED:
        scores *= tlx.require_layout(tl.full((64, 32), SM_SCALE * log2e, tl.float32), mma_mn, pin=False)
    return tlx.require_layout(scores, mma_mn, pin=False)


@triton.jit
def _attn_bwd_dq_d64_causal_dp32(
    do,
    row_delta,
    vt,
    mma_mn: tl.constexpr,
    q_op0_mn: tl.constexpr,
    vt_op1_mn: tl.constexpr,
):
    do = tlx.require_layout(do, q_op0_mn, pin=False)
    vt = tlx.require_layout(vt, vt_op1_mn, pin=False)
    row_delta_full = tlx.require_layout(tl.broadcast_to(row_delta[:, None], (64, 32)), mma_mn, pin=False)
    dp = tlx.zeros((64, 32), tl.float32, layout=mma_mn) - row_delta_full
    dp = tl.dot(do, vt, acc=dp, out_dtype=tl.float32)
    return tlx.require_layout(dp, mma_mn, pin=False)


@triton.jit
def _attn_bwd_dq_d64_causal_finish32(
    dq,
    scores,
    dp,
    k_nd,
    mma_mn: tl.constexpr,
    mma_md: tl.constexpr,
    ds_op0_md: tl.constexpr,
    k_op1_md: tl.constexpr,
):
    dq = tlx.require_layout(dq, mma_md, pin=False)
    scores = tlx.require_layout(scores, mma_mn, pin=False)
    dp = tlx.require_layout(dp, mma_mn, pin=False)
    k_nd = tlx.require_layout(k_nd, k_op1_md, pin=False)
    p = tlx.require_layout(tl.math.exp2(scores), mma_mn, pin=False)
    ds = tlx.amd_register_handoff(
        p * dp,
        register_class="vgpr",
        registers_per_group=2,
    )
    ds = tlx.require_layout(ds.to(tl.bfloat16), ds_op0_md, pin=False)
    dq = tl.dot(ds, k_nd, acc=dq, out_dtype=tl.float32)
    return tlx.require_layout(dq, mma_md, pin=False)


@triton.jit
def _attn_bwd_dq_d64_causal_m256_unmasked_n32(
    dq0,
    dq1,
    dq2,
    dq3,
    q0,
    q1,
    q2,
    do0,
    do1,
    do2,
    lse0,
    lse1,
    lse2,
    lse3,
    delta0,
    delta1,
    delta2,
    delta3,
    rows1,
    rows2,
    rows3,
    q3,
    do3,
    k_view,
    v_view,
    kv_wait,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    N_OFFSET: tl.constexpr,
    SCORE_PRE_SCALED: tl.constexpr,
    mma_mn: tl.constexpr,
    mma_md: tl.constexpr,
    q_op0_mn: tl.constexpr,
    kt_op1_mn: tl.constexpr,
    vt_op1_mn: tl.constexpr,
    ds_op0_md: tl.constexpr,
    k_op1_md: tl.constexpr,
):
    """Consume one unmasked N32 half of a four-fragment D64 owner."""
    tl.static_assert(D == 64)
    k_slice = tlx.local_slice(k_view, [N_OFFSET, 0], [32, D])
    v_slice = tlx.local_slice(v_view, [N_OFFSET, 0], [32, D])
    kt = tlx.local_load(tlx.local_trans(k_slice), token=kv_wait, layout=kt_op1_mn)
    k_nd = tlx.local_load(k_slice, token=kv_wait, layout=k_op1_md)
    scores0 = _attn_bwd_dq_d64_causal_score32(q0, lse0, kt, SM_SCALE, mma_mn, q_op0_mn, kt_op1_mn, SCORE_PRE_SCALED)
    vt = tlx.local_load(tlx.local_trans(v_slice), token=kv_wait, layout=vt_op1_mn)
    dp0 = _attn_bwd_dq_d64_causal_dp32(do0, delta0, vt, mma_mn, q_op0_mn, vt_op1_mn)
    dq0 = _attn_bwd_dq_d64_causal_finish32(dq0, scores0, dp0, k_nd, mma_mn, mma_md, ds_op0_md, k_op1_md)
    dq1 = _attn_bwd_dq_d64_causal_step(dq1, q1, do1, lse1, delta1, rows1, kt, vt, k_nd, n0, SM_SCALE, SQ, SKV, D,
                                       N_OFFSET, 32, False, SCORE_PRE_SCALED, mma_mn, mma_md, kt_op1_mn, vt_op1_mn,
                                       ds_op0_md, k_op1_md)
    dq3 = _attn_bwd_dq_d64_causal_step(dq3, q3, do3, lse3, delta3, rows3, kt, vt, k_nd, n0, SM_SCALE, SQ, SKV, D,
                                       N_OFFSET, 32, False, SCORE_PRE_SCALED, mma_mn, mma_md, kt_op1_mn, vt_op1_mn,
                                       ds_op0_md, k_op1_md)
    scores2 = _attn_bwd_dq_d64_causal_score32(q2, lse2, kt, SM_SCALE, mma_mn, q_op0_mn, kt_op1_mn, SCORE_PRE_SCALED)
    dp2 = _attn_bwd_dq_d64_causal_dp32(do2, delta2, vt, mma_mn, q_op0_mn, vt_op1_mn)
    dq2 = _attn_bwd_dq_d64_causal_finish32(dq2, scores2, dp2, k_nd, mma_mn, mma_md, ds_op0_md, k_op1_md)
    return dq0, dq1, dq2, dq3


@triton.jit
def _attn_bwd_dq_d64_causal_nslice(
    dq0,
    dq1,
    dq2,
    dq3,
    q0,
    q1,
    q2,
    do0,
    do1,
    do2,
    lse0,
    lse1,
    lse2,
    lse3,
    delta0,
    delta1,
    delta2,
    delta3,
    rows0,
    rows1,
    rows2,
    rows3,
    q3,
    do3,
    k_view,
    v_view,
    kv_wait,
    valid_fragments,
    tail_step,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    OWNER_FRAGMENTS: tl.constexpr,
    SKIP_OWNER_TAIL: tl.constexpr,
    N_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    TRIANGULAR_TAIL: tl.constexpr,
    SCORE_PRE_SCALED: tl.constexpr,
    mma_mn: tl.constexpr,
    mma_md: tl.constexpr,
    q_op0_mn: tl.constexpr,
    kt_op1_mn: tl.constexpr,
    vt_op1_mn: tl.constexpr,
    ds_op0_md: tl.constexpr,
    k_op1_md: tl.constexpr,
):
    dq0 = tlx.require_layout(dq0, mma_md, pin=False)
    dq1 = tlx.require_layout(dq1, mma_md, pin=False)
    dq2 = tlx.require_layout(dq2, mma_md, pin=False)
    dq3 = tlx.require_layout(dq3, mma_md, pin=False)
    k_slice = tlx.local_slice(k_view, [N_OFFSET, 0], [BLOCK_N, D])
    v_slice = tlx.local_slice(v_view, [N_OFFSET, 0], [BLOCK_N, D])
    k_source = tlx.local_load(tlx.local_trans(k_slice), token=kv_wait, layout=kt_op1_mn)
    kv_source = tlx.local_load(k_slice, token=kv_wait, layout=k_op1_md)
    v_source = tlx.local_load(tlx.local_trans(v_slice), token=kv_wait, layout=vt_op1_mn)
    if TRIANGULAR_TAIL:
        # Tail K/V block j intersects fragment j. Earlier fragments are
        # wholly future and skipped; later valid fragments are fully causal.
        if tail_step == 0:
            dq0 = _attn_bwd_dq_d64_causal_step(
                dq0,
                q0,
                do0,
                lse0,
                delta0,
                rows0,
                k_source,
                v_source,
                kv_source,
                n0,
                SM_SCALE,
                SQ,
                SKV,
                D,
                N_OFFSET,
                BLOCK_N,
                True,
                SCORE_PRE_SCALED,
                mma_mn,
                mma_md,
                kt_op1_mn,
                vt_op1_mn,
                ds_op0_md,
                k_op1_md,
            )
            dq0 = tlx.require_layout(dq0, mma_md, pin=False)
        dq0 = tlx.require_layout(dq0, mma_md, pin=False)

        if not SKIP_OWNER_TAIL or valid_fragments >= 2:
            if tail_step == 0:
                dq1 = _attn_bwd_dq_d64_causal_step(
                    dq1,
                    q1,
                    do1,
                    lse1,
                    delta1,
                    rows1,
                    k_source,
                    v_source,
                    kv_source,
                    n0,
                    SM_SCALE,
                    SQ,
                    SKV,
                    D,
                    N_OFFSET,
                    BLOCK_N,
                    False,
                    SCORE_PRE_SCALED,
                    mma_mn,
                    mma_md,
                    kt_op1_mn,
                    vt_op1_mn,
                    ds_op0_md,
                    k_op1_md,
                )
                dq1 = tlx.require_layout(dq1, mma_md, pin=False)
            elif tail_step == 1:
                dq1 = _attn_bwd_dq_d64_causal_step(
                    dq1,
                    q1,
                    do1,
                    lse1,
                    delta1,
                    rows1,
                    k_source,
                    v_source,
                    kv_source,
                    n0,
                    SM_SCALE,
                    SQ,
                    SKV,
                    D,
                    N_OFFSET,
                    BLOCK_N,
                    True,
                    SCORE_PRE_SCALED,
                    mma_mn,
                    mma_md,
                    kt_op1_mn,
                    vt_op1_mn,
                    ds_op0_md,
                    k_op1_md,
                )
                dq1 = tlx.require_layout(dq1, mma_md, pin=False)
        dq1 = tlx.require_layout(dq1, mma_md, pin=False)

        if OWNER_FRAGMENTS >= 3:
            if not SKIP_OWNER_TAIL or valid_fragments >= 3:
                if tail_step < 2:
                    dq2 = _attn_bwd_dq_d64_causal_step(
                        dq2,
                        q2,
                        do2,
                        lse2,
                        delta2,
                        rows2,
                        k_source,
                        v_source,
                        kv_source,
                        n0,
                        SM_SCALE,
                        SQ,
                        SKV,
                        D,
                        N_OFFSET,
                        BLOCK_N,
                        False,
                        SCORE_PRE_SCALED,
                        mma_mn,
                        mma_md,
                        kt_op1_mn,
                        vt_op1_mn,
                        ds_op0_md,
                        k_op1_md,
                    )
                    dq2 = tlx.require_layout(dq2, mma_md, pin=False)
                elif tail_step == 2:
                    dq2 = _attn_bwd_dq_d64_causal_step(
                        dq2,
                        q2,
                        do2,
                        lse2,
                        delta2,
                        rows2,
                        k_source,
                        v_source,
                        kv_source,
                        n0,
                        SM_SCALE,
                        SQ,
                        SKV,
                        D,
                        N_OFFSET,
                        BLOCK_N,
                        True,
                        SCORE_PRE_SCALED,
                        mma_mn,
                        mma_md,
                        kt_op1_mn,
                        vt_op1_mn,
                        ds_op0_md,
                        k_op1_md,
                    )
                    dq2 = tlx.require_layout(dq2, mma_md, pin=False)
        dq2 = tlx.require_layout(dq2, mma_md, pin=False)

        if OWNER_FRAGMENTS == 4:
            if not SKIP_OWNER_TAIL or valid_fragments >= 4:
                if tail_step < 3:
                    dq3 = _attn_bwd_dq_d64_causal_step(
                        dq3,
                        q3,
                        do3,
                        lse3,
                        delta3,
                        rows3,
                        k_source,
                        v_source,
                        kv_source,
                        n0,
                        SM_SCALE,
                        SQ,
                        SKV,
                        D,
                        N_OFFSET,
                        BLOCK_N,
                        False,
                        SCORE_PRE_SCALED,
                        mma_mn,
                        mma_md,
                        kt_op1_mn,
                        vt_op1_mn,
                        ds_op0_md,
                        k_op1_md,
                    )
                    dq3 = tlx.require_layout(dq3, mma_md, pin=False)
                elif tail_step == 3:
                    dq3 = _attn_bwd_dq_d64_causal_step(
                        dq3,
                        q3,
                        do3,
                        lse3,
                        delta3,
                        rows3,
                        k_source,
                        v_source,
                        kv_source,
                        n0,
                        SM_SCALE,
                        SQ,
                        SKV,
                        D,
                        N_OFFSET,
                        BLOCK_N,
                        True,
                        SCORE_PRE_SCALED,
                        mma_mn,
                        mma_md,
                        kt_op1_mn,
                        vt_op1_mn,
                        ds_op0_md,
                        k_op1_md,
                    )
                    dq3 = tlx.require_layout(dq3, mma_md, pin=False)
        dq3 = tlx.require_layout(dq3, mma_md, pin=False)
    else:
        dq0 = _attn_bwd_dq_d64_causal_step(
            dq0,
            q0,
            do0,
            lse0,
            delta0,
            rows0,
            k_source,
            v_source,
            kv_source,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            N_OFFSET,
            BLOCK_N,
            APPLY_MASK,
            SCORE_PRE_SCALED,
            mma_mn,
            mma_md,
            kt_op1_mn,
            vt_op1_mn,
            ds_op0_md,
            k_op1_md,
        )
        dq0 = tlx.require_layout(dq0, mma_md, pin=False)
        if not SKIP_OWNER_TAIL or valid_fragments >= 2:
            dq1 = _attn_bwd_dq_d64_causal_step(
                dq1,
                q1,
                do1,
                lse1,
                delta1,
                rows1,
                k_source,
                v_source,
                kv_source,
                n0,
                SM_SCALE,
                SQ,
                SKV,
                D,
                N_OFFSET,
                BLOCK_N,
                APPLY_MASK,
                SCORE_PRE_SCALED,
                mma_mn,
                mma_md,
                kt_op1_mn,
                vt_op1_mn,
                ds_op0_md,
                k_op1_md,
            )
            dq1 = tlx.require_layout(dq1, mma_md, pin=False)
        if OWNER_FRAGMENTS >= 3:
            if not SKIP_OWNER_TAIL or valid_fragments >= 3:
                dq2 = _attn_bwd_dq_d64_causal_step(
                    dq2,
                    q2,
                    do2,
                    lse2,
                    delta2,
                    rows2,
                    k_source,
                    v_source,
                    kv_source,
                    n0,
                    SM_SCALE,
                    SQ,
                    SKV,
                    D,
                    N_OFFSET,
                    BLOCK_N,
                    APPLY_MASK,
                    SCORE_PRE_SCALED,
                    mma_mn,
                    mma_md,
                    kt_op1_mn,
                    vt_op1_mn,
                    ds_op0_md,
                    k_op1_md,
                )
                dq2 = tlx.require_layout(dq2, mma_md, pin=False)
        if OWNER_FRAGMENTS == 4:
            if not SKIP_OWNER_TAIL or valid_fragments >= 4:
                dq3 = _attn_bwd_dq_d64_causal_step(
                    dq3,
                    q3,
                    do3,
                    lse3,
                    delta3,
                    rows3,
                    k_source,
                    v_source,
                    kv_source,
                    n0,
                    SM_SCALE,
                    SQ,
                    SKV,
                    D,
                    N_OFFSET,
                    BLOCK_N,
                    APPLY_MASK,
                    SCORE_PRE_SCALED,
                    mma_mn,
                    mma_md,
                    kt_op1_mn,
                    vt_op1_mn,
                    ds_op0_md,
                    k_op1_md,
                )
                dq3 = tlx.require_layout(dq3, mma_md, pin=False)
    dq0 = tlx.require_layout(dq0, mma_md, pin=False)
    dq1 = tlx.require_layout(dq1, mma_md, pin=False)
    dq2 = tlx.require_layout(dq2, mma_md, pin=False)
    dq3 = tlx.require_layout(dq3, mma_md, pin=False)
    return dq0, dq1, dq2, dq3


@triton.jit
def _attn_bwd_dq_d64_causal_full_tail_block(
    dq0,
    dq1,
    dq2,
    dq3,
    q0,
    q1,
    q2,
    do0,
    do1,
    do2,
    do3,
    lse0,
    lse1,
    lse2,
    lse3,
    delta0,
    delta1,
    delta2,
    delta3,
    rows0,
    rows1,
    rows2,
    rows3,
    q3,
    k_buffers,
    v_buffers,
    K,
    V,
    kv_base,
    bulk_end_block,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    OWNER_FRAGMENTS: tl.constexpr,
    SKIP_OWNER_TAIL: tl.constexpr,
    LOGICAL_N: tl.constexpr,
    KV_PIPELINE_STAGES: tl.constexpr,
    TAIL_STEP: tl.constexpr,
    SCORE_PRE_SCALED: tl.constexpr,
    kv_async_layout: tl.constexpr,
    mma_mn: tl.constexpr,
    mma_md: tl.constexpr,
    q_op0_mn: tl.constexpr,
    kt_op1_mn: tl.constexpr,
    vt_op1_mn: tl.constexpr,
    ds_op0_md: tl.constexpr,
    k_op1_md: tl.constexpr,
):
    """Consume one statically selected triangular block of a full owner."""
    tl.static_assert(KV_PIPELINE_STAGES == 2)
    n_block = bulk_end_block + TAIL_STEP
    offs_n = tl.arange(0, 64)
    offs_d = tl.arange(0, D)
    current_slot = n_block % 2
    next_slot = 1 - current_slot
    tl.debug_barrier()
    if TAIL_STEP + 1 < OWNER_FRAGMENTS:
        next_offsets = ((n_block + 1) * 64 * D + offs_n[:, None] * D + offs_d[None, :]).to(tl.int32)
        next_offsets = tlx.require_layout(next_offsets, kv_async_layout, pin=False)
        next_k = tlx.buffer_load_to_local(tlx.local_view(k_buffers, next_slot), K + kv_base, next_offsets)
        next_v = tlx.buffer_load_to_local(tlx.local_view(v_buffers, next_slot), V + kv_base, next_offsets)
        tlx.async_load_commit_group([next_k, next_v])
        kv_wait = tlx.async_load_wait_group(1)
    else:
        kv_wait = tlx.async_load_wait_group(0)
    k_view = tlx.local_view(k_buffers, current_slot)
    v_view = tlx.local_view(v_buffers, current_slot)
    n0 = n_block * 64
    if LOGICAL_N == 32:
        dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
            dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
            rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, OWNER_FRAGMENTS, TAIL_STEP, n0, SM_SCALE, SQ,
            SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 32, True, True, SCORE_PRE_SCALED, mma_mn, mma_md, q_op0_mn,
            kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
        dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
            dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
            rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, OWNER_FRAGMENTS, TAIL_STEP, n0, SM_SCALE, SQ,
            SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 32, 32, True, True, SCORE_PRE_SCALED, mma_mn, mma_md, q_op0_mn,
            kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
    else:
        dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
            dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
            rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, OWNER_FRAGMENTS, TAIL_STEP, n0, SM_SCALE, SQ,
            SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 64, True, True, SCORE_PRE_SCALED, mma_mn, mma_md, q_op0_mn,
            kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
    return dq0, dq1, dq2, dq3


@triton.jit
def _attn_bwd_dq_d64_causal_store_q64(
    DQ,
    q_base,
    dq,
    row_start,
    store_end,
    SM_SCALE: tl.constexpr,
    D: tl.constexpr,
    out_layout: tl.constexpr,
):
    rows = row_start + tl.arange(0, 64)
    cols = tl.arange(0, D)
    local_rows = tl.arange(0, 64)
    offsets = (local_rows[:, None] * D + cols[None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, out_layout, pin=False)
    mask = tl.broadcast_to(rows[:, None] < store_end, offsets.shape)
    mask = tlx.require_layout(mask, out_layout, pin=False)
    dq = tlx.require_layout(
        dq,
        tlx.amd_mfma_layout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        ),
        pin=False,
    )
    scale = tlx.require_layout(
        tl.full((64, D), SM_SCALE, tl.float32),
        tlx.amd_mfma_layout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        ),
        pin=False,
    )
    out = tlx.require_layout((dq * scale).to(tl.bfloat16), out_layout, pin=False)
    tlx.buffer_store(out, DQ + q_base + row_start * D, offsets, mask=mask)


@triton.jit
def _attn_bwd_dq_d64_causal_impl(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    LSE_TERM,
    DQ,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    OWNER_ROWS: tl.constexpr,
    LOGICAL_N: tl.constexpr,
    USE_DQ_XCD: tl.constexpr,
    SKIP_OWNER_TAIL: tl.constexpr,
    OWNER_PID_BASE: tl.constexpr,
    LAUNCH_Q_TILES: tl.constexpr,
    OWNER_FRAGMENTS: tl.constexpr,
    GRID_OWNER_M: tl.constexpr,
    KV_PIPELINE_STAGES: tl.constexpr,
    STAT_MODE: tl.constexpr,
):
    tl.static_assert(D == 64)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert(SQ % 64 == 0 and SKV % 64 == 0 and SQ <= SKV)
    tl.static_assert(OWNER_ROWS == 192 or OWNER_ROWS == 256)
    tl.static_assert(OWNER_FRAGMENTS == 2 or OWNER_FRAGMENTS == 3 or OWNER_FRAGMENTS == 4)
    tl.static_assert(LOGICAL_N == 32 or LOGICAL_N == 64)
    tl.static_assert(KV_PIPELINE_STAGES == 2)
    tl.static_assert(STAT_MODE == _D64_MHA_POSITIVE_JIT or STAT_MODE == _D64_GQA_SIGNED_JIT)
    score_pre_scaled: tl.constexpr = STAT_MODE == _D64_GQA_SIGNED_JIT

    grid_owner_m: tl.constexpr = (OWNER_FRAGMENTS * 64 if GRID_OWNER_M == 0 else GRID_OWNER_M)
    tl.static_assert(grid_owner_m == OWNER_ROWS)
    num_owners: tl.constexpr = tl.cdiv(SQ, grid_owner_m)
    launch_q_tiles: tl.constexpr = (num_owners if LAUNCH_Q_TILES == 0 else LAUNCH_Q_TILES)
    group: tl.constexpr = HQ // HKV
    value = tl.program_id(0)
    if USE_DQ_XCD:
        xcd = value % 8
        value //= 8
        q_in_group = value % group
        value //= group
        local_owner = value % launch_q_tiles
        bkv_group = value // launch_q_tiles
        bkv = bkv_group * 8 + xcd
        pid_hkv = bkv % HKV
        pid_b = bkv // HKV
    else:
        pid_hkv = value % HKV
        value //= HKV
        q_in_group = value % group
        value //= group
        local_owner = value % launch_q_tiles
        pid_b = value // launch_q_tiles
    pid_hq = pid_hkv * group + q_in_group
    physical_owner = OWNER_PID_BASE + local_owner

    reverse_owner = num_owners - 1 - physical_owner
    pad: tl.constexpr = num_owners * grid_owner_m - SQ
    raw_start = reverse_owner * grid_owner_m
    owner_start = tl.maximum(raw_start - pad, 0)
    owner_end = raw_start + grid_owner_m - pad
    store_end = tl.minimum(owner_end, SQ)

    q_head = (pid_b * HQ + pid_hq).to(tl.int64)
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    q_base = q_head * SQ * D
    kv_base = kv_head * SKV * D
    stats_base = q_head * SQ

    mma_mn: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_md: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    q_op0_mn: tl.constexpr = tlx.dot_operand_layout(0, mma_mn, k_width=8)
    kt_op1_mn: tl.constexpr = tlx.dot_operand_layout(1, mma_mn, k_width=8)
    vt_op1_mn: tl.constexpr = tlx.dot_operand_layout(1, mma_mn, k_width=8)
    ds_op0_md: tl.constexpr = tlx.dot_operand_layout(0, mma_md, k_width=4)
    k_op1_md: tl.constexpr = tlx.dot_operand_layout(1, mma_md, k_width=4)

    kv_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((8, 64, 128, 256, 512, 16, 32, 2048), (1, 2, 4, 1024)),
    )
    out_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((64, 128, 256, 512, 8, 16, 1024, 2048), (1, 2, 4, 32)),
    )
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 8)],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
        ],
        [64, 64],
    )
    k_buffers = tlx.local_alloc((64, 64), tl.bfloat16, KV_PIPELINE_STAGES, layout=shared_layout)
    v_buffers = tlx.local_alloc((64, 64), tl.bfloat16, KV_PIPELINE_STAGES, layout=shared_layout)

    q0, do0, lse0, delta0, rows0 = _attn_bwd_dq_d64_causal_load_q64(
        Q,
        O,
        DO,
        LSE,
        DELTA,
        LSE_TERM,
        q_base,
        stats_base,
        owner_start,
        store_end,
        SM_SCALE,
        SQ,
        D,
        STAT_MODE,
        q_op0_mn,
    )
    q1, do1, lse1, delta1, rows1 = _attn_bwd_dq_d64_causal_load_q64(
        Q,
        O,
        DO,
        LSE,
        DELTA,
        LSE_TERM,
        q_base,
        stats_base,
        owner_start + 64,
        store_end,
        SM_SCALE,
        SQ,
        D,
        STAT_MODE,
        q_op0_mn,
    )
    if OWNER_FRAGMENTS >= 3:
        q2, do2, lse2, delta2, rows2 = _attn_bwd_dq_d64_causal_load_q64(
            Q,
            O,
            DO,
            LSE,
            DELTA,
            LSE_TERM,
            q_base,
            stats_base,
            owner_start + 128,
            store_end,
            SM_SCALE,
            SQ,
            D,
            STAT_MODE,
            q_op0_mn,
        )
    else:
        q2, do2, lse2, delta2, rows2 = q0, do0, lse0, delta0, rows0
    if OWNER_FRAGMENTS == 4:
        q3, do3, lse3, delta3, rows3 = _attn_bwd_dq_d64_causal_load_q64(
            Q,
            O,
            DO,
            LSE,
            DELTA,
            LSE_TERM,
            q_base,
            stats_base,
            owner_start + 192,
            store_end,
            SM_SCALE,
            SQ,
            D,
            STAT_MODE,
            q_op0_mn,
        )
        q3 = tlx.require_layout(q3, q_op0_mn, pin=False)
        # One 64x64 bf16 fragment over four waves is eight 32-bit registers
        # per thread, so one group keeps the complete fragment resident.
        q3 = tlx.amd_register_resident(q3, register_class="agpr", registers_per_group=8)
    else:
        q3, do3, lse3, delta3, rows3 = q0, do0, lse0, delta0, rows0

    dq0 = tlx.zeros((64, 64), tl.float32, layout=mma_md)
    dq1 = tlx.zeros((64, 64), tl.float32, layout=mma_md)
    dq2 = tlx.zeros((64, 64), tl.float32, layout=mma_md)
    dq3 = tlx.zeros((64, 64), tl.float32, layout=mma_md)

    offs_n = tl.arange(0, 64)
    offs_d = tl.arange(0, D)
    first_offsets = (offs_n[:, None] * D + offs_d[None, :]).to(tl.int32)
    first_offsets = tlx.require_layout(first_offsets, kv_async_layout, pin=False)
    first_k = tlx.buffer_load_to_local(tlx.local_view(k_buffers, 0), K + kv_base, first_offsets)
    first_v = tlx.buffer_load_to_local(tlx.local_view(v_buffers, 0), V + kv_base, first_offsets)
    tlx.async_load_commit_group([first_k, first_v])

    bulk_end_block = (owner_start + (SKV - SQ)) // 64
    end_n_block = tl.minimum(
        (owner_end - 1 + (SKV - SQ)) // 64 + 1,
        SKV // 64,
    )
    valid_fragments = (store_end - owner_start + 63) // 64
    for n_block in range(0, bulk_end_block):
        current_slot = n_block % 2
        next_slot = 1 - current_slot
        tl.debug_barrier()
        # Every bulk block precedes at least one owner-tail block, so the
        # successor is in range and becomes the tail prologue on the final
        # iteration.
        next_offsets = ((n_block + 1) * 64 * D + offs_n[:, None] * D + offs_d[None, :]).to(tl.int32)
        next_offsets = tlx.require_layout(next_offsets, kv_async_layout, pin=False)
        next_k = tlx.buffer_load_to_local(
            tlx.local_view(k_buffers, next_slot),
            K + kv_base,
            next_offsets,
        )
        next_v = tlx.buffer_load_to_local(
            tlx.local_view(v_buffers, next_slot),
            V + kv_base,
            next_offsets,
        )
        tlx.async_load_commit_group([next_k, next_v])
        kv_wait = tlx.async_load_wait_group(1)
        k_view = tlx.local_view(k_buffers, current_slot)
        v_view = tlx.local_view(v_buffers, current_slot)
        n0 = n_block * 64
        if LOGICAL_N == 32 and OWNER_FRAGMENTS == 4:
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_m256_unmasked_n32(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, n0, SM_SCALE, SQ, SKV, D, 0, score_pre_scaled,
                mma_mn, mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_m256_unmasked_n32(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, n0, SM_SCALE, SQ, SKV, D, 32, score_pre_scaled,
                mma_mn, mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
        elif LOGICAL_N == 32:
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, 0, n0, SM_SCALE, SQ, SKV,
                D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 32, False, False, score_pre_scaled, mma_mn, mma_md, q_op0_mn,
                kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, 0, n0, SM_SCALE, SQ, SKV,
                D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 32, 32, False, False, score_pre_scaled, mma_mn, mma_md, q_op0_mn,
                kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
        else:
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, 0, n0, SM_SCALE, SQ, SKV,
                D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 64, False, False, score_pre_scaled, mma_mn, mma_md, q_op0_mn,
                kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)

    static_full_tail: tl.constexpr = SQ % OWNER_ROWS == 0 and SKV > SQ
    dynamic_tail_end = bulk_end_block if static_full_tail else end_n_block
    for n_block in range(bulk_end_block, dynamic_tail_end):
        current_slot = n_block % 2
        next_slot = 1 - current_slot
        tl.debug_barrier()
        if n_block + 1 < end_n_block:
            next_offsets = ((n_block + 1) * 64 * D + offs_n[:, None] * D + offs_d[None, :]).to(tl.int32)
            next_offsets = tlx.require_layout(next_offsets, kv_async_layout, pin=False)
            next_k = tlx.buffer_load_to_local(
                tlx.local_view(k_buffers, next_slot),
                K + kv_base,
                next_offsets,
            )
            next_v = tlx.buffer_load_to_local(
                tlx.local_view(v_buffers, next_slot),
                V + kv_base,
                next_offsets,
            )
            tlx.async_load_commit_group([next_k, next_v])
            kv_wait = tlx.async_load_wait_group(1)
        else:
            kv_wait = tlx.async_load_wait_group(0)
        k_view = tlx.local_view(k_buffers, current_slot)
        v_view = tlx.local_view(v_buffers, current_slot)
        n0 = n_block * 64
        if LOGICAL_N == 32:
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, n_block - bulk_end_block,
                n0, SM_SCALE, SQ, SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 32, True, True, score_pre_scaled, mma_mn,
                mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, n_block - bulk_end_block,
                n0, SM_SCALE, SQ, SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 32, 32, True, True, score_pre_scaled,
                mma_mn, mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
        else:
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_nslice(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, lse0, lse1, lse2, lse3, delta0, delta1, delta2, delta3,
                rows0, rows1, rows2, rows3, q3, do3, k_view, v_view, kv_wait, valid_fragments, n_block - bulk_end_block,
                n0, SM_SCALE, SQ, SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, 0, 64, True, True, score_pre_scaled, mma_mn,
                mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
    if static_full_tail:
        for tail_step in tl.static_range(0, OWNER_FRAGMENTS):
            dq0, dq1, dq2, dq3 = _attn_bwd_dq_d64_causal_full_tail_block(
                dq0, dq1, dq2, dq3, q0, q1, q2, do0, do1, do2, do3, lse0, lse1, lse2, lse3, delta0, delta1, delta2,
                delta3, rows0, rows1, rows2, rows3, q3, k_buffers, v_buffers, K, V, kv_base, bulk_end_block, SM_SCALE,
                SQ, SKV, D, OWNER_FRAGMENTS, SKIP_OWNER_TAIL, LOGICAL_N, KV_PIPELINE_STAGES, tail_step,
                score_pre_scaled, kv_async_layout, mma_mn, mma_md, q_op0_mn, kt_op1_mn, vt_op1_mn, ds_op0_md, k_op1_md)
    tlx.async_load_wait_group(0)
    tl.debug_barrier()

    _attn_bwd_dq_d64_causal_store_q64(DQ, q_base, dq0, owner_start, store_end, SM_SCALE, D, out_layout)
    if not SKIP_OWNER_TAIL or owner_start + 64 < store_end:
        _attn_bwd_dq_d64_causal_store_q64(DQ, q_base, dq1, owner_start + 64, store_end, SM_SCALE, D, out_layout)
    if OWNER_FRAGMENTS >= 3:
        if not SKIP_OWNER_TAIL or owner_start + 128 < store_end:
            _attn_bwd_dq_d64_causal_store_q64(DQ, q_base, dq2, owner_start + 128, store_end, SM_SCALE, D, out_layout)
    if OWNER_FRAGMENTS == 4:
        if not SKIP_OWNER_TAIL or owner_start + 192 < store_end:
            _attn_bwd_dq_d64_causal_store_q64(DQ, q_base, dq3, owner_start + 192, store_end, SM_SCALE, D, out_layout)


@triton.jit
def _attn_bwd_dq_d64_causal_mha_kernel(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DQ,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    OWNER_ROWS: tl.constexpr,
    LOGICAL_N: tl.constexpr,
    USE_DQ_XCD: tl.constexpr,
    SKIP_OWNER_TAIL: tl.constexpr,
    OWNER_PID_BASE: tl.constexpr,
    LAUNCH_Q_TILES: tl.constexpr,
    OWNER_FRAGMENTS: tl.constexpr,
    GRID_OWNER_M: tl.constexpr,
    KV_PIPELINE_STAGES: tl.constexpr,
):
    _attn_bwd_dq_d64_causal_impl(Q, K, V, O, DO, LSE, DELTA, DELTA, DQ, SM_SCALE, HQ, HKV, SQ, SKV, D, OWNER_ROWS,
                                 LOGICAL_N, USE_DQ_XCD, SKIP_OWNER_TAIL, OWNER_PID_BASE, LAUNCH_Q_TILES,
                                 OWNER_FRAGMENTS, GRID_OWNER_M, KV_PIPELINE_STAGES, _D64_MHA_POSITIVE_JIT)


@triton.jit
def _attn_bwd_dq_d64_causal_gqa8_kernel(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    LSE_TERM,
    DQ,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    OWNER_ROWS: tl.constexpr,
    LOGICAL_N: tl.constexpr,
    USE_DQ_XCD: tl.constexpr,
    SKIP_OWNER_TAIL: tl.constexpr,
    OWNER_PID_BASE: tl.constexpr,
    LAUNCH_Q_TILES: tl.constexpr,
    OWNER_FRAGMENTS: tl.constexpr,
    GRID_OWNER_M: tl.constexpr,
    KV_PIPELINE_STAGES: tl.constexpr,
):
    _attn_bwd_dq_d64_causal_impl(Q, K, V, O, DO, LSE, DELTA, LSE_TERM, DQ, SM_SCALE, HQ, HKV, SQ, SKV, D, OWNER_ROWS,
                                 LOGICAL_N, USE_DQ_XCD, SKIP_OWNER_TAIL, OWNER_PID_BASE, LAUNCH_Q_TILES,
                                 OWNER_FRAGMENTS, GRID_OWNER_M, KV_PIPELINE_STAGES, _D64_GQA_SIGNED_JIT)


@triton.jit
def _d64_mha_issue_stage(
    Q,
    DO,
    LSE,
    Delta,
    q_dst,
    do_dst,
    lse_dst,
    delta_dst,
    m_blk,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats_async_layout: tl.constexpr,
):
    """Stage one complete BM32 Q/dO/natural-stat tile."""
    rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    q_offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    q_offsets = tlx.require_layout(q_offsets, q_async_layout, pin=False)
    stats_offsets = tlx.require_layout(rows.to(tl.int32), stats_async_layout, pin=False)
    q_token = tlx.buffer_load_to_local(q_dst, Q, q_offsets)
    do_token = tlx.buffer_load_to_local(do_dst, DO, q_offsets)
    lse = tlx.buffer_load(LSE, stats_offsets)
    delta = tlx.buffer_load(Delta, stats_offsets)
    tlx.local_store(lse_dst, lse)
    tlx.local_store(delta_dst, delta)
    tlx.async_load_commit_group([q_token, do_token])


@triton.jit
def _d64_mha_positive_front(
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK: tl.constexpr,
    mma_nm: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
):
    """Reconstruct positive-ABI P and dS from one staged BM32 tile."""
    log2e: tl.constexpr = 1.4426950408889634
    q_t = tlx.local_load(tlx.local_trans(q_view), token=stage_wait, layout=q_t_op1_nm)
    do_t = tlx.local_load(tlx.local_trans(do_view), token=stage_wait, layout=q_t_op1_nm)
    lse = tlx.local_load(lse_view, token=stage_wait, relaxed=True)
    delta = tlx.local_load(delta_view, token=stage_wait, relaxed=True)

    scores = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
    scores = tl.dot(k_nm, q_t, acc=scores, out_dtype=tl.float32)
    score_scale = tlx.require_layout(
        tl.full((BLOCK_N, BLOCK_M), SM_SCALE, tl.float32),
        mma_nm,
        pin=False,
    )
    lse_nm = tlx.require_layout(
        tl.broadcast_to(lse[None, :], (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    log2e_nm = tlx.require_layout(
        tl.full((BLOCK_N, BLOCK_M), log2e, tl.float32),
        mma_nm,
        pin=False,
    )
    # P = exp2((QK * SM_SCALE - LSE) * log2(e)).
    scores = (scores * score_scale - lse_nm) * log2e_nm
    if APPLY_CAUSAL_MASK:
        rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = n0 + tl.arange(0, BLOCK_N)
        valid = cols[:, None] <= rows[None, :] + (SKV - SQ)
        valid = tlx.require_layout(valid, mma_nm, pin=False)
        negative_inf = tlx.require_layout(
            tl.full(
                (BLOCK_N, BLOCK_M),
                float("-inf"),
                tl.float32,
            ),
            mma_nm,
            pin=False,
        )
        scores = tl.where(valid, scores, negative_inf)
    p = tlx.require_layout(tl.math.exp2(scores), mma_nm, pin=False)

    dp = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
    dp = tl.dot(v_nm, do_t, acc=dp, out_dtype=tl.float32)
    delta_nm = tlx.require_layout(
        tl.broadcast_to(delta[None, :], (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    # dS = P * (dO @ V.T - Delta).
    ds = p * (dp - delta_nm)
    p_nd = tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)
    ds_nd = tlx.require_layout(ds.to(tl.bfloat16), p_op0_nd, pin=False)
    return p_nd, ds_nd


@triton.jit
def _d64_mha_step(
    dk,
    dv,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    p_nd, ds_nd = _d64_mha_positive_front(
        q_view,
        do_view,
        lse_view,
        delta_view,
        stage_wait,
        k_nm,
        v_nm,
        m_blk,
        n0,
        SM_SCALE,
        SQ,
        SKV,
        BLOCK_M,
        BLOCK_N,
        APPLY_CAUSAL_MASK,
        mma_nm,
        q_t_op1_nm,
        p_op0_nd,
    )
    do_nd = tlx.local_load(do_view, token=stage_wait, layout=q_op1_nd)
    q_nd = tlx.local_load(q_view, token=stage_wait, layout=q_op1_nd)
    dv = tl.dot(p_nd, do_nd, acc=dv, out_dtype=tl.float32)
    dk = tl.dot(ds_nd, q_nd, acc=dk, out_dtype=tl.float32)
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_mha_consume(
    dk,
    dv,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    if n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ):
        dk, dv = _d64_mha_step(
            dk,
            dv,
            q_view,
            do_view,
            lse_view,
            delta_view,
            stage_wait,
            k_nm,
            v_nm,
            m_blk,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            BLOCK_M,
            BLOCK_N,
            True,
            mma_nm,
            mma_nd,
            q_t_op1_nm,
            p_op0_nd,
            q_op1_nd,
        )
        dk = tlx.require_layout(dk, mma_nd, pin=False)
        dv = tlx.require_layout(dv, mma_nd, pin=False)
    else:
        dk, dv = _d64_mha_step(
            dk,
            dv,
            q_view,
            do_view,
            lse_view,
            delta_view,
            stage_wait,
            k_nm,
            v_nm,
            m_blk,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            BLOCK_M,
            BLOCK_N,
            False,
            mma_nm,
            mma_nd,
            q_t_op1_nm,
            p_op0_nd,
            q_op1_nd,
        )
        dk = tlx.require_layout(dk, mma_nd, pin=False)
        dv = tlx.require_layout(dv, mma_nd, pin=False)
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _attn_bwd_dkdv_d64_causal_mha_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LSE_MODE: tl.constexpr,
    DELTA_MODE: tl.constexpr,
):
    """Two-wave BM32/BN64 causal MHA owner with direct publication."""
    tl.static_assert(HQ == HKV)
    tl.static_assert(D == 64)
    tl.static_assert(BLOCK_M == 32)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(SQ % BLOCK_M == 0 and SKV % BLOCK_N == 0)
    tl.static_assert(SQ % 64 == 0 and SKV % 64 == 0)
    tl.static_assert(SQ <= SKV)
    tl.static_assert(LSE_MODE == _D64_LSE_NATURAL_LOG_JIT)
    tl.static_assert(DELTA_MODE == _D64_DELTA_POSITIVE_JIT)

    value = tl.program_id(0)
    nt = SKV // BLOCK_N
    pid_n = value % nt
    value //= nt
    pid_hkv = value % HKV
    pid_b = value // HKV
    n0 = pid_n * BLOCK_N

    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 1],
    )
    k_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    v_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    q_t_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=4)
    q_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=4)

    # Two waves cooperatively copy one BM32xD64 tile with D8 vectors.
    q_async_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
        stride=((8, 64, 128, 256, 512, 16, 32), (1, 2, 4, 1024)),
    )
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((32, 4), ()),
        stride=((1, 0), ()),
    )
    out_layout: tl.constexpr = tlx.layout(
        shape=((2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
        stride=(
            (64, 128, 256, 512, 8, 16, 1024),
            (1, 2, 4, 32, 2048),
        ),
    )
    qdo_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 16],
            [0, 32],
            [16, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8], [16]],
        block_bases=[],
        alignment=4,
    )
    q_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    do_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    lse_ring = tlx.local_alloc((BLOCK_M, ), tl.float32, 2, layout=stats_smem_layout)
    delta_ring = tlx.local_alloc((BLOCK_M, ), tl.float32, 2, layout=stats_smem_layout)

    # K/V are loaded once in their final MFMA operand layouts and remain
    # resident through the complete bottom-right query frontier.
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    kv_base = kv_head * SKV * D
    kv_rows = n0 + tl.arange(0, BLOCK_N)
    kv_cols = tl.arange(0, D)
    kv_offsets = (kv_rows[:, None] * D + kv_cols[None, :]).to(tl.int32)
    k_offsets = tlx.require_layout(kv_offsets, k_op0_nm, pin=False)
    v_offsets = tlx.require_layout(kv_offsets, v_op0_nm, pin=False)
    k_nm = tlx.buffer_load(K + kv_base, k_offsets)
    k_nm = tlx.require_layout(k_nm, k_op0_nm, pin=False)
    v_nm = tlx.buffer_load(V + kv_base, v_offsets)
    v_nm = tlx.require_layout(v_nm, v_op0_nm, pin=False)

    q_head = (pid_b * HQ + pid_hkv).to(tl.int64)
    q_base = Q + q_head * SQ * D
    do_base = DO + q_head * SQ * D
    stats_base = q_head * SQ
    lse_base = LSE + stats_base
    delta_base = Delta + stats_base
    start_m_blk = tl.maximum((n0 - (SKV - SQ)) // BLOCK_M, 0)

    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    full_pairs: tl.constexpr = SQ // (2 * BLOCK_M)
    _d64_mha_issue_stage(
        q_base,
        do_base,
        lse_base,
        delta_base,
        tlx.local_view(q_ring, 0),
        tlx.local_view(do_ring, 0),
        tlx.local_view(lse_ring, 0),
        tlx.local_view(delta_ring, 0),
        start_m_blk,
        D,
        BLOCK_M,
        q_async_layout,
        stats_async_layout,
    )
    for m_pair in range(start_m_blk // 2, full_pairs):
        m_blk_a = m_pair * 2
        m_blk_b = m_blk_a + 1
        _d64_mha_issue_stage(
            q_base,
            do_base,
            lse_base,
            delta_base,
            tlx.local_view(q_ring, 1),
            tlx.local_view(do_ring, 1),
            tlx.local_view(lse_ring, 1),
            tlx.local_view(delta_ring, 1),
            m_blk_b,
            D,
            BLOCK_M,
            q_async_layout,
            stats_async_layout,
        )
        stage_wait = tlx.async_load_wait_group(1)
        dk, dv = _d64_mha_consume(
            dk,
            dv,
            tlx.local_view(q_ring, 0),
            tlx.local_view(do_ring, 0),
            tlx.local_view(lse_ring, 0),
            tlx.local_view(delta_ring, 0),
            stage_wait,
            k_nm,
            v_nm,
            m_blk_a,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            BLOCK_M,
            BLOCK_N,
            mma_nm,
            mma_nd,
            q_t_op1_nm,
            p_op0_nd,
            q_op1_nd,
        )
        has_next = m_pair + 1 < full_pairs
        if has_next:
            # Retire every relaxed view before overwriting the ping slot.
            tl.debug_barrier()
            _d64_mha_issue_stage(
                q_base,
                do_base,
                lse_base,
                delta_base,
                tlx.local_view(q_ring, 0),
                tlx.local_view(do_ring, 0),
                tlx.local_view(lse_ring, 0),
                tlx.local_view(delta_ring, 0),
                m_blk_a + 2,
                D,
                BLOCK_M,
                q_async_layout,
                stats_async_layout,
            )
            stage_wait = tlx.async_load_wait_group(1)
        else:
            stage_wait = tlx.async_load_wait_group(0)
        dk, dv = _d64_mha_consume(
            dk,
            dv,
            tlx.local_view(q_ring, 1),
            tlx.local_view(do_ring, 1),
            tlx.local_view(lse_ring, 1),
            tlx.local_view(delta_ring, 1),
            stage_wait,
            k_nm,
            v_nm,
            m_blk_b,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            BLOCK_M,
            BLOCK_N,
            mma_nm,
            mma_nd,
            q_t_op1_nm,
            p_op0_nd,
            q_op1_nd,
        )
        if has_next:
            # Slot one is reused at the top of the next pair.
            tl.debug_barrier()

    dk_scale = tlx.require_layout(
        tl.full((BLOCK_N, D), SM_SCALE, tl.float32),
        mma_nd,
        pin=False,
    )
    dk = tlx.require_layout(dk, mma_nd, pin=False) * dk_scale
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    output_offsets = (tl.arange(0, BLOCK_N)[:, None] * D + tl.arange(0, D)[None, :]).to(tl.int32)
    output_offsets = tlx.require_layout(output_offsets, out_layout, pin=False)
    dk_out = tlx.require_layout(dk.to(tl.bfloat16), out_layout, pin=False)
    dv_out = tlx.require_layout(dv.to(tl.bfloat16), out_layout, pin=False)
    output_base = kv_head * SKV * D + n0 * D
    tlx.buffer_store(dk_out, DK + output_base, output_offsets)
    tlx.buffer_store(dv_out, DV + output_base, output_offsets)


@triton.jit
def _d64_gqa8_issue_stage(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_dst,
    do_dst,
    lse_dst,
    delta_dst,
    m_blk,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats_async_layout: tl.constexpr,
):
    """Issue one complete Q/dO/signed-stat tile as one async group."""
    rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    q_offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    q_offsets = tlx.require_layout(q_offsets, q_async_layout, pin=False)
    stats_offsets = tlx.require_layout(rows.to(tl.int32), stats_async_layout, pin=False)
    q_token = tlx.buffer_load_to_local(q_dst, Q, q_offsets)
    do_token = tlx.buffer_load_to_local(do_dst, DO, q_offsets)
    lse_token = tlx.buffer_load_to_local(lse_dst, LSE_TERM, stats_offsets)
    delta_token = tlx.buffer_load_to_local(delta_dst, Delta, stats_offsets)
    tlx.async_load_commit_group([q_token, do_token, lse_token, delta_token])


@triton.jit
def _d64_gqa8_issue_qdo_stage(
    Q,
    DO,
    q_dst,
    do_dst,
    m_blk,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    q_async_layout: tl.constexpr,
):
    """Issue one Q/dO tile as an async group."""
    rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    q_offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    q_offsets = tlx.require_layout(q_offsets, q_async_layout, pin=False)
    q_token = tlx.buffer_load_to_local(q_dst, Q, q_offsets)
    do_token = tlx.buffer_load_to_local(do_dst, DO, q_offsets)
    tlx.async_load_commit_group([q_token, do_token])


@triton.jit
def _d64_gqa8_issue_stats4_qdo_stage(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_dst,
    do_dst,
    lse_dst,
    delta_dst,
    qdo_m_blk,
    first_stats_m_blk,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats4_async_layout: tl.constexpr,
):
    """Issue Q/dO plus four contiguous statistics tiles as one async group."""
    rows = qdo_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    q_offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    q_offsets = tlx.require_layout(q_offsets, q_async_layout, pin=False)
    stats_offsets = first_stats_m_blk * BLOCK_M + tl.arange(0, 4 * BLOCK_M)
    stats_offsets = tl.reshape(stats_offsets, (4, BLOCK_M)).to(tl.int32)
    stats_offsets = tlx.require_layout(stats_offsets, stats4_async_layout, pin=False)
    q_token = tlx.buffer_load_to_local(q_dst, Q, q_offsets)
    do_token = tlx.buffer_load_to_local(do_dst, DO, q_offsets)
    lse_token = tlx.buffer_load_to_local(lse_dst, LSE_TERM, stats_offsets)
    delta_token = tlx.buffer_load_to_local(delta_dst, Delta, stats_offsets)
    tlx.async_load_commit_group([q_token, do_token, lse_token, delta_token])


@triton.jit
def _d64_gqa8_signed_front(
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK,
    LATE_DO_T: tl.constexpr,
    STATS_RANK2: tl.constexpr,
    mma_nm: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
):
    """Compute signed-ABI P and dS from one staged BM64 query tile."""
    q_t = tlx.local_load(tlx.local_trans(q_view), token=stage_wait, layout=q_t_op1_nm)
    if not LATE_DO_T:
        do_t = tlx.local_load(tlx.local_trans(do_view), token=stage_wait, layout=q_t_op1_nm)
    lse_term = tlx.local_load(lse_view, token=stage_wait, relaxed=True)
    negative_delta = tlx.local_load(delta_view, token=stage_wait, relaxed=True)
    if STATS_RANK2:
        lse_term = tl.reshape(lse_term, (BLOCK_M, ))
        negative_delta = tl.reshape(negative_delta, (BLOCK_M, ))

    # Selected GQA publishes the log2-domain -LSE term. K is resident and
    # pre-scaled once, so no per-score or per-owner scaling remains here.
    scores = tlx.require_layout(
        tl.broadcast_to(lse_term[None, :], (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    scores = tl.dot(k_nm, q_t, acc=scores, out_dtype=tl.float32)
    if APPLY_CAUSAL_MASK:
        rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = n0 + tl.arange(0, BLOCK_N)
        valid = cols[:, None] <= rows[None, :] + (SKV - SQ)
        valid = tlx.require_layout(valid, mma_nm, pin=False)
        negative_inf = tlx.require_layout(
            tl.full(
                (BLOCK_N, BLOCK_M),
                float("-inf"),
                tl.float32,
            ),
            mma_nm,
            pin=False,
        )
        scores = tl.where(valid, scores, negative_inf)
    p = tlx.require_layout(tl.math.exp2(scores), mma_nm, pin=False)

    # Selected GQA publishes negative Delta, so this accumulator is exactly
    # dO@V^T+delta before the P product.
    dp = tlx.require_layout(
        tl.broadcast_to(tl.reshape(negative_delta, (1, BLOCK_M)), (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    if LATE_DO_T:
        # The full-width gradient recurrence benefits when dO starts after the
        # independent score/exp work and does not remain live across it.
        do_t = tlx.local_load(tlx.local_trans(do_view), token=stage_wait, layout=q_t_op1_nm)
    dp = tl.dot(v_nm, do_t, acc=dp, out_dtype=tl.float32)
    ds = p * dp
    # Direct D64 keeps each native pair together; split D32 starts independent
    # intervals so its narrower recurrences retain scheduler flexibility.
    handoff_group: tl.constexpr = 2 if LATE_DO_T else 1
    ds = tlx.amd_register_handoff(
        ds,
        register_class="vgpr",
        registers_per_group=handoff_group,
    )
    p_nd = tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)
    ds_nd = tlx.require_layout(ds.to(tl.bfloat16), p_op0_nd, pin=False)
    return p_nd, ds_nd


@triton.jit
def _d64_gqa8_signed_front_loaded_stats(
    q_view,
    do_view,
    lse_values,
    delta_values,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK,
    BATCHED_STATS: tl.constexpr,
    STATS_STEP: tl.constexpr,
    mma_nm: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
):
    """Signed GQA front for register or four-tile shared statistics."""
    q_t = tlx.local_load(tlx.local_trans(q_view), token=stage_wait, layout=q_t_op1_nm)
    if BATCHED_STATS:
        tl.static_assert(STATS_STEP >= 0 and STATS_STEP < 4)
        lse_view = tlx.local_slice(lse_values, [STATS_STEP, 0], [1, BLOCK_M])
        lse_term = tl.reshape(tlx.local_load(lse_view, relaxed=True), (BLOCK_M, ))
    else:
        lse_term = lse_values
    scores = tlx.require_layout(
        tl.broadcast_to(lse_term[None, :], (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    scores = tl.dot(k_nm, q_t, acc=scores, out_dtype=tl.float32)
    if APPLY_CAUSAL_MASK:
        rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = n0 + tl.arange(0, BLOCK_N)
        valid = cols[:, None] <= rows[None, :] + (SKV - SQ)
        valid = tlx.require_layout(valid, mma_nm, pin=False)
        negative_inf = tlx.require_layout(
            tl.full((BLOCK_N, BLOCK_M), float("-inf"), tl.float32),
            mma_nm,
            pin=False,
        )
        scores = tl.where(valid, scores, negative_inf)
    p = tlx.require_layout(tl.math.exp2(scores), mma_nm, pin=False)
    if BATCHED_STATS:
        delta_view = tlx.local_slice(delta_values, [STATS_STEP, 0], [1, BLOCK_M])
        negative_delta = tl.reshape(tlx.local_load(delta_view, relaxed=True), (BLOCK_M, ))
    else:
        negative_delta = delta_values
    dp = tlx.require_layout(
        tl.broadcast_to(tl.reshape(negative_delta, (1, BLOCK_M)), (BLOCK_N, BLOCK_M)),
        mma_nm,
        pin=False,
    )
    do_t = tlx.local_load(tlx.local_trans(do_view), token=stage_wait, layout=q_t_op1_nm)
    dp = tl.dot(v_nm, do_t, acc=dp, out_dtype=tl.float32)
    ds = tlx.amd_register_handoff(
        p * dp,
        register_class="vgpr",
        registers_per_group=2,
    )
    p_nd = tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)
    ds_nd = tlx.require_layout(ds.to(tl.bfloat16), p_op0_nd, pin=False)
    return p_nd, ds_nd


@triton.jit
def _d64_gqa8_direct_d64_step(
    dk,
    dv,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK,
    STATS_RANK2: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    p_nd, ds_nd = _d64_gqa8_signed_front(
        q_view,
        do_view,
        lse_view,
        delta_view,
        stage_wait,
        k_nm,
        v_nm,
        m_blk,
        n0,
        SM_SCALE,
        SQ,
        SKV,
        D,
        BLOCK_M,
        BLOCK_N,
        APPLY_CAUSAL_MASK,
        True,
        STATS_RANK2,
        mma_nm,
        q_t_op1_nm,
        p_op0_nd,
    )
    p_nd = tlx.require_layout(p_nd, p_op0_nd, pin=False)
    ds_nd = tlx.require_layout(ds_nd, p_op0_nd, pin=False)
    do_nd = tlx.local_load(do_view, token=stage_wait, layout=q_op1_nd)
    dv = tl.dot(p_nd, do_nd, acc=dv, out_dtype=tl.float32)
    # Retire the dV operand before materializing Q for the independent dK
    # update. This preserves the algorithmic order while shortening the
    # simultaneous operand lifetime in the full-D recurrence.
    q_nd = tlx.local_load(q_view, token=stage_wait, layout=q_op1_nd)
    dk = tl.dot(ds_nd, q_nd, acc=dk, out_dtype=tl.float32)
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_d32_step(
    dk_d0,
    dk_d1,
    dv_d0,
    dv_d1,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK,
    INTERLEAVED_D32: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
    dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
    dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
    dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
    p_nd, ds_nd = _d64_gqa8_signed_front(
        q_view,
        do_view,
        lse_view,
        delta_view,
        stage_wait,
        k_nm,
        v_nm,
        m_blk,
        n0,
        SM_SCALE,
        SQ,
        SKV,
        D,
        BLOCK_M,
        BLOCK_N,
        APPLY_CAUSAL_MASK,
        False,
        False,
        mma_nm,
        q_t_op1_nm,
        p_op0_nd,
    )
    if INTERLEAVED_D32:
        # Interleave low dV/dK then high dV/dK recurrences.
        do_d0 = tlx.local_load(
            tlx.local_slice(do_view, [0, 0], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dv_d0 = tl.dot(p_nd, do_d0, acc=dv_d0, out_dtype=tl.float32)
        q_d0 = tlx.local_load(
            tlx.local_slice(q_view, [0, 0], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dk_d0 = tl.dot(ds_nd, q_d0, acc=dk_d0, out_dtype=tl.float32)
        do_d1 = tlx.local_load(
            tlx.local_slice(do_view, [0, D // 2], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dv_d1 = tl.dot(p_nd, do_d1, acc=dv_d1, out_dtype=tl.float32)
        q_d1 = tlx.local_load(
            tlx.local_slice(q_view, [0, D // 2], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dk_d1 = tl.dot(ds_nd, q_d1, acc=dk_d1, out_dtype=tl.float32)
    else:
        # Independent D32 keeps both dV recurrences separate from both dK
        # recurrences, shortening each scheduler-visible chain.
        do_d0 = tlx.local_load(
            tlx.local_slice(do_view, [0, 0], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dv_d0 = tl.dot(p_nd, do_d0, acc=dv_d0, out_dtype=tl.float32)
        do_d1 = tlx.local_load(
            tlx.local_slice(do_view, [0, D // 2], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dv_d1 = tl.dot(p_nd, do_d1, acc=dv_d1, out_dtype=tl.float32)
        q_d0 = tlx.local_load(
            tlx.local_slice(q_view, [0, 0], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dk_d0 = tl.dot(ds_nd, q_d0, acc=dk_d0, out_dtype=tl.float32)
        q_d1 = tlx.local_load(
            tlx.local_slice(q_view, [0, D // 2], [BLOCK_M, D // 2]),
            token=stage_wait,
            layout=q_op1_nd,
        )
        dk_d1 = tl.dot(ds_nd, q_d1, acc=dk_d1, out_dtype=tl.float32)
    return (
        tlx.require_layout(dk_d0, mma_nd, pin=False),
        tlx.require_layout(dk_d1, mma_nd, pin=False),
        tlx.require_layout(dv_d0, mma_nd, pin=False),
        tlx.require_layout(dv_d1, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_direct_consume(
    dk,
    dv,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STATS_RANK2: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    apply_causal_mask = n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)
    dk, dv = _d64_gqa8_direct_d64_step(
        dk,
        dv,
        q_view,
        do_view,
        lse_view,
        delta_view,
        stage_wait,
        k_nm,
        v_nm,
        m_blk,
        n0,
        SM_SCALE,
        SQ,
        SKV,
        D,
        BLOCK_M,
        BLOCK_N,
        apply_causal_mask,
        STATS_RANK2,
        mma_nm,
        mma_nd,
        q_t_op1_nm,
        p_op0_nd,
        q_op1_nd,
    )
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_direct_consume_loaded_stats(
    dk,
    dv,
    q_view,
    do_view,
    lse_values,
    delta_values,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK,
    BATCHED_STATS: tl.constexpr,
    STATS_STEP: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    p_nd, ds_nd = _d64_gqa8_signed_front_loaded_stats(
        q_view,
        do_view,
        lse_values,
        delta_values,
        stage_wait,
        k_nm,
        v_nm,
        m_blk,
        n0,
        SM_SCALE,
        SQ,
        SKV,
        D,
        BLOCK_M,
        BLOCK_N,
        APPLY_CAUSAL_MASK,
        BATCHED_STATS,
        STATS_STEP,
        mma_nm,
        q_t_op1_nm,
        p_op0_nd,
    )
    p_nd = tlx.require_layout(p_nd, p_op0_nd, pin=False)
    ds_nd = tlx.require_layout(ds_nd, p_op0_nd, pin=False)
    do_nd = tlx.local_load(do_view, token=stage_wait, layout=q_op1_nd)
    dv = tl.dot(p_nd, do_nd, acc=dv, out_dtype=tl.float32)
    # Match the ordinary statistics path: Q is independent of dV, so keep it
    # out of the live operand set until the following dK recurrence.
    q_nd = tlx.local_load(
        q_view,
        token=stage_wait,
        layout=q_op1_nd,
        rematerialize_coordinates=True,
    )
    dk = tl.dot(ds_nd, q_nd, acc=dk, out_dtype=tl.float32)
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_d32_consume(
    dk_d0,
    dk_d1,
    dv_d0,
    dv_d1,
    q_view,
    do_view,
    lse_view,
    delta_view,
    stage_wait,
    k_nm,
    v_nm,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INTERLEAVED_D32: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
    dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
    dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
    dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
    apply_causal_mask = n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)
    dk_d0, dk_d1, dv_d0, dv_d1 = _d64_gqa8_d32_step(dk_d0, dk_d1, dv_d0, dv_d1, q_view, do_view, lse_view, delta_view,
                                                    stage_wait, k_nm, v_nm, m_blk, n0, SM_SCALE, SQ, SKV, D, BLOCK_M,
                                                    BLOCK_N, apply_causal_mask, INTERLEAVED_D32, mma_nm, mma_nd,
                                                    q_t_op1_nm, p_op0_nd, q_op1_nd)
    dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
    dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
    dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
    dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
    return dk_d0, dk_d1, dv_d0, dv_d1


@triton.jit
def _d64_gqa8_all_head_mblock(
    dk,
    dv,
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_ring,
    do_ring,
    lse_all,
    delta_all,
    k_nm,
    v_nm,
    group_stats_base,
    group_q_base,
    m_blk,
    n0,
    SM_SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK: tl.constexpr,
    q_async_layout: tl.constexpr,
    all_head_stats_layout: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    """Consume one query block for all eight heads with a static mask mode."""
    dk = tlx.require_layout(dk, mma_nd, pin=False)
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    stat_heads = tl.arange(0, 8)
    stats_offsets = stat_heads[:, None] * SQ + rows[None, :]
    stats_offsets = tlx.require_layout(stats_offsets.to(tl.int32), all_head_stats_layout, pin=False)
    lse_values = tlx.buffer_load(LSE_TERM + group_stats_base, stats_offsets, contiguity=2)
    delta_values = tlx.buffer_load(Delta + group_stats_base, stats_offsets, contiguity=2)
    tlx.local_store(lse_all, lse_values)
    tlx.local_store(delta_all, delta_values)

    cols = tl.arange(0, D)
    q_offsets = (rows[:, None] * D + cols[None, :]).to(tl.int32)
    q_offsets = tlx.require_layout(q_offsets, q_async_layout, pin=False)
    q_token = tlx.buffer_load_to_local(tlx.local_view(q_ring, 0), Q + group_q_base, q_offsets)
    do_token = tlx.buffer_load_to_local(tlx.local_view(do_ring, 0), DO + group_q_base, q_offsets)
    tlx.async_load_commit_group([q_token, do_token])
    tl.debug_barrier()

    for local_head in range(0, 8):
        current_slot = local_head % 2
        if local_head + 1 < 8:
            next_head = local_head + 1
            next_slot = 1 - current_slot
            next_head_base = next_head.to(tl.int64) * SQ * D
            next_q_token = tlx.buffer_load_to_local(
                tlx.local_view(q_ring, next_slot),
                Q + group_q_base + next_head_base,
                q_offsets,
            )
            next_do_token = tlx.buffer_load_to_local(
                tlx.local_view(do_ring, next_slot),
                DO + group_q_base + next_head_base,
                q_offsets,
            )
            tlx.async_load_commit_group([next_q_token, next_do_token])
            stage_wait = tlx.async_load_wait_group(1)
        else:
            stage_wait = tlx.async_load_wait_group(0)
        lse_view = tlx.local_dynamic_slice(lse_all, [local_head, 0], [1, BLOCK_M])
        delta_view = tlx.local_dynamic_slice(delta_all, [local_head, 0], [1, BLOCK_M])
        lse_term = tl.reshape(tlx.local_load(lse_view, relaxed=True), (BLOCK_M, ))
        negative_delta = tl.reshape(tlx.local_load(delta_view, relaxed=True), (BLOCK_M, ))
        dk, dv = _d64_gqa8_direct_consume_loaded_stats(
            dk,
            dv,
            tlx.local_view(q_ring, current_slot),
            tlx.local_view(do_ring, current_slot),
            lse_term,
            negative_delta,
            stage_wait,
            k_nm,
            v_nm,
            m_blk,
            n0,
            SM_SCALE,
            SQ,
            SKV,
            D,
            BLOCK_M,
            BLOCK_N,
            APPLY_CAUSAL_MASK,
            False,
            0,
            mma_nm,
            mma_nd,
            q_t_op1_nm,
            p_op0_nd,
            q_op1_nd,
        )
        tl.debug_barrier()
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_all_head_direct_d64_impl(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_ring,
    do_ring,
    lse_all,
    delta_all,
    k_nm,
    v_nm,
    pid_b,
    pid_hkv,
    off_split,
    n0,
    start_m_blk,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    q_async_layout: tl.constexpr,
    all_head_stats_layout: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    """Walk query blocks outside heads and stage all eight heads' statistics."""
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = SQ // BLOCK_M
    split_advance = (off_split - (start_m_blk % KV_SPLITS) + KV_SPLITS) % KV_SPLITS
    first_m_blk = start_m_blk + split_advance
    group_head = (pid_b * HQ + pid_hkv * 8).to(tl.int64)
    group_stats_base = group_head * SQ
    group_q_base = group_head * SQ * D

    if first_m_blk < num_m_blocks:
        first_is_masked = n0 + BLOCK_N - 1 > first_m_blk * BLOCK_M + (SKV - SQ)
        if first_is_masked:
            dk, dv = _d64_gqa8_all_head_mblock(dk, dv, Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_all, delta_all,
                                               k_nm, v_nm, group_stats_base, group_q_base, first_m_blk, n0, SM_SCALE,
                                               SQ, SKV, D, BLOCK_M, BLOCK_N, True, q_async_layout,
                                               all_head_stats_layout, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
            dk = tlx.require_layout(dk, mma_nd, pin=False)
            dv = tlx.require_layout(dv, mma_nd, pin=False)
        else:
            dk, dv = _d64_gqa8_all_head_mblock(dk, dv, Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_all, delta_all,
                                               k_nm, v_nm, group_stats_base, group_q_base, first_m_blk, n0, SM_SCALE,
                                               SQ, SKV, D, BLOCK_M, BLOCK_N, False, q_async_layout,
                                               all_head_stats_layout, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
            dk = tlx.require_layout(dk, mma_nd, pin=False)
            dv = tlx.require_layout(dv, mma_nd, pin=False)
        for m_blk in range(first_m_blk + KV_SPLITS, num_m_blocks, KV_SPLITS):
            dk, dv = _d64_gqa8_all_head_mblock(dk, dv, Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_all, delta_all,
                                               k_nm, v_nm, group_stats_base, group_q_base, m_blk, n0, SM_SCALE, SQ, SKV,
                                               D, BLOCK_M, BLOCK_N, False, q_async_layout, all_head_stats_layout,
                                               mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_direct_d64_impl(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_ring,
    do_ring,
    lse_ring,
    delta_ring,
    k_nm,
    v_nm,
    pid_b,
    pid_hkv,
    off_split,
    n0,
    start_m_blk,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    CYCLIC_QUERY_SPLIT: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats_async_layout: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    """Carry only full-width FP32 dK/dV through the complete owner walk."""
    tl.static_assert(KV_SPLITS == _D64_CAUSAL_GQA8_KV_SPLITS_JIT)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = SQ // BLOCK_M
    full_pairs: tl.constexpr = num_m_blocks // 2
    has_odd: tl.constexpr = (num_m_blocks % 2) != 0

    if CYCLIC_QUERY_SPLIT:
        for local_head in range(0, 8):
            q_head = (pid_b * HQ + pid_hkv * 8 + local_head).to(tl.int64)
            q_base = Q + q_head * SQ * D
            do_base = DO + q_head * SQ * D
            stats_base = q_head * SQ
            lse_base = LSE_TERM + stats_base
            delta_base = Delta + stats_base
            split_advance = (off_split - (start_m_blk % KV_SPLITS) + KV_SPLITS) % KV_SPLITS
            first_m_blk = start_m_blk + split_advance
            if first_m_blk < num_m_blocks:
                _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 0),
                                      tlx.local_view(do_ring, 0), tlx.local_view(lse_ring, 0),
                                      tlx.local_view(delta_ring, 0), first_m_blk, D, BLOCK_M, q_async_layout,
                                      stats_async_layout)
                sequence_blocks = (num_m_blocks - first_m_blk + KV_SPLITS - 1) // KV_SPLITS
                sequence_pairs = sequence_blocks // 2
                for sequence_pair in range(0, sequence_pairs):
                    m_blk_a = first_m_blk + sequence_pair * 2 * KV_SPLITS
                    m_blk_b = m_blk_a + KV_SPLITS
                    # Issue the complete next stage before consuming current.
                    _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 1),
                                          tlx.local_view(do_ring, 1), tlx.local_view(lse_ring, 1),
                                          tlx.local_view(delta_ring, 1), m_blk_b, D, BLOCK_M, q_async_layout,
                                          stats_async_layout)
                    stage_wait = tlx.async_load_wait_group(1)
                    dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                                      tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring,
                                                                                                  0), stage_wait, k_nm,
                                                      v_nm, m_blk_a, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N, False,
                                                      mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                    has_next_pair = sequence_pair + 1 < sequence_pairs
                    has_odd_block = (sequence_blocks % 2) != 0
                    if has_next_pair or has_odd_block:
                        # All relaxed views are retired before slot reuse.
                        tl.debug_barrier()
                        _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 0),
                                              tlx.local_view(do_ring, 0), tlx.local_view(lse_ring, 0),
                                              tlx.local_view(delta_ring, 0), m_blk_a + 2 * KV_SPLITS, D, BLOCK_M,
                                              q_async_layout, stats_async_layout)
                        stage_wait = tlx.async_load_wait_group(1)
                    else:
                        stage_wait = tlx.async_load_wait_group(0)
                    dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 1), tlx.local_view(do_ring, 1),
                                                      tlx.local_view(lse_ring, 1), tlx.local_view(delta_ring,
                                                                                                  1), stage_wait, k_nm,
                                                      v_nm, m_blk_b, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N, False,
                                                      mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                    if has_next_pair:
                        tl.debug_barrier()
                dk = tlx.require_layout(dk, mma_nd, pin=False)
                dv = tlx.require_layout(dv, mma_nd, pin=False)
                if (sequence_blocks % 2) != 0:
                    stage_wait = tlx.async_load_wait_group(0)
                    m_blk_tail = first_m_blk + (sequence_blocks - 1) * KV_SPLITS
                    dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                                      tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring,
                                                                                                  0), stage_wait, k_nm,
                                                      v_nm, m_blk_tail, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N,
                                                      False, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                    dk = tlx.require_layout(dk, mma_nd, pin=False)
                    dv = tlx.require_layout(dv, mma_nd, pin=False)
                dk = tlx.require_layout(dk, mma_nd, pin=False)
                dv = tlx.require_layout(dv, mma_nd, pin=False)
            if local_head + 1 < 8:
                tl.debug_barrier()
    else:
        pair_start = (start_m_blk // 2) * 2
        for local_head in tl.static_range(0, 2):
            query_in_group = off_split * 2 + local_head
            q_head = (pid_b * HQ + pid_hkv * 8 + query_in_group).to(tl.int64)
            q_base = Q + q_head * SQ * D
            do_base = DO + q_head * SQ * D
            stats_base = q_head * SQ
            lse_base = LSE_TERM + stats_base
            delta_base = Delta + stats_base
            _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring,
                                                                                        0), tlx.local_view(do_ring, 0),
                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0), pair_start, D, BLOCK_M,
                                  q_async_layout, stats_async_layout)
            for m_pair in range(pair_start // 2, full_pairs):
                m_blk_a = m_pair * 2
                m_blk_b = m_blk_a + 1
                _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 1),
                                      tlx.local_view(do_ring, 1), tlx.local_view(lse_ring, 1),
                                      tlx.local_view(delta_ring, 1), m_blk_b, D, BLOCK_M, q_async_layout,
                                      stats_async_layout)
                stage_wait = tlx.async_load_wait_group(1)
                dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0),
                                                  stage_wait, k_nm, v_nm, m_blk_a, n0, SM_SCALE, SQ, SKV, D, BLOCK_M,
                                                  BLOCK_N, False, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                has_following = (m_pair + 1 < full_pairs) or has_odd
                if has_following:
                    tl.debug_barrier()
                    _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 0),
                                          tlx.local_view(do_ring, 0), tlx.local_view(lse_ring, 0),
                                          tlx.local_view(delta_ring, 0), m_blk_a + 2, D, BLOCK_M, q_async_layout,
                                          stats_async_layout)
                    stage_wait = tlx.async_load_wait_group(1)
                else:
                    stage_wait = tlx.async_load_wait_group(0)
                dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 1), tlx.local_view(do_ring, 1),
                                                  tlx.local_view(lse_ring, 1), tlx.local_view(delta_ring, 1),
                                                  stage_wait, k_nm, v_nm, m_blk_b, n0, SM_SCALE, SQ, SKV, D, BLOCK_M,
                                                  BLOCK_N, False, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                if m_pair + 1 < full_pairs:
                    tl.debug_barrier()
            if has_odd:
                stage_wait = tlx.async_load_wait_group(0)
                dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring,
                                                                                              0), stage_wait, k_nm,
                                                  v_nm, num_m_blocks - 1, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N,
                                                  False, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
            if local_head + 1 < 2:
                tl.debug_barrier()
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_async_stats4_direct_d64_impl(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_ring,
    do_ring,
    lse_ring,
    delta_ring,
    lse_batch_ring,
    delta_batch_ring,
    k_nm,
    v_nm,
    pid_b,
    pid_hkv,
    off_split,
    n0,
    start_m_blk,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats_async_layout: tl.constexpr,
    stats4_async_layout: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    """Direct-D64 walk with four-stage Q/dO and double-buffered statistics."""
    tl.static_assert(SQ % (4 * BLOCK_M) == 0)
    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = SQ // BLOCK_M
    pair_start = (start_m_blk // 2) * 2

    for local_head in tl.static_range(0, 2):
        query_in_group = off_split * 2 + local_head
        q_head = (pid_b * HQ + pid_hkv * 8 + query_in_group).to(tl.int64)
        q_base = Q + q_head * SQ * D
        do_base = DO + q_head * SQ * D
        stats_base = q_head * SQ
        lse_base = LSE_TERM + stats_base
        delta_base = Delta + stats_base

        # The causal frontier advances in BM128 units. Peel one pair when it
        # starts halfway through a four-tile vector-statistics batch.
        batch_start = pair_start + (pair_start % 4)
        dk = tlx.require_layout(dk, mma_nd, pin=False)
        dv = tlx.require_layout(dv, mma_nd, pin=False)
        if batch_start != pair_start:
            _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring,
                                                                                        0), tlx.local_view(do_ring, 0),
                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0), pair_start, D, BLOCK_M,
                                  q_async_layout, stats_async_layout)
            _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring,
                                                                                        1), tlx.local_view(do_ring, 1),
                                  tlx.local_view(lse_ring, 1), tlx.local_view(delta_ring, 1), pair_start + 1, D,
                                  BLOCK_M, q_async_layout, stats_async_layout)
            stage_wait = tlx.async_load_wait_group(1)
            dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                              tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0), stage_wait,
                                              k_nm, v_nm, pair_start, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N, False,
                                              mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
            stage_wait = tlx.async_load_wait_group(0)
            dk, dv = _d64_gqa8_direct_consume(dk, dv, tlx.local_view(q_ring, 1), tlx.local_view(do_ring, 1),
                                              tlx.local_view(lse_ring, 1), tlx.local_view(delta_ring, 1), stage_wait,
                                              k_nm, v_nm, pair_start + 1, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N,
                                              False, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
            dk = tlx.require_layout(dk, mma_nd, pin=False)
            dv = tlx.require_layout(dv, mma_nd, pin=False)
            tl.debug_barrier()
        dk = tlx.require_layout(dk, mma_nd, pin=False)
        dv = tlx.require_layout(dv, mma_nd, pin=False)

        if batch_start < num_m_blocks:
            first_quad = batch_start // 4
            initial_stats_slot = first_quad % 2
            _d64_gqa8_issue_stats4_qdo_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 0),
                                             tlx.local_view(do_ring, 0),
                                             tlx.local_view(lse_batch_ring, initial_stats_slot),
                                             tlx.local_view(delta_batch_ring, initial_stats_slot), batch_start,
                                             batch_start, D, BLOCK_M, q_async_layout, stats4_async_layout)
            # Keep the async refill and the dV/dK recurrence in one loop so
            # their overlapping lifetimes remain visible to the scheduler.
            for m_quad in range(first_quad, num_m_blocks // 4):
                m0 = m_quad * 4
                stats_slot = m_quad % 2
                lse_batch = tlx.local_view(lse_batch_ring, stats_slot)
                delta_batch = tlx.local_view(delta_batch_ring, stats_slot)
                for step in tl.static_range(0, 4):
                    current_slot = step
                    has_next_quad = m_quad + 1 < num_m_blocks // 4
                    if step + 1 < 4:
                        next_slot = step + 1
                        _d64_gqa8_issue_qdo_stage(q_base, do_base, tlx.local_view(q_ring, next_slot),
                                                  tlx.local_view(do_ring, next_slot), m0 + step + 1, D, BLOCK_M,
                                                  q_async_layout)
                        stage_wait = tlx.async_load_wait_group(1)
                    elif has_next_quad:
                        next_stats_slot = 1 - stats_slot
                        _d64_gqa8_issue_stats4_qdo_stage(q_base, do_base, lse_base, delta_base,
                                                         tlx.local_view(q_ring, 0), tlx.local_view(do_ring, 0),
                                                         tlx.local_view(lse_batch_ring, next_stats_slot),
                                                         tlx.local_view(delta_batch_ring, next_stats_slot), m0 + 4,
                                                         m0 + 4, D, BLOCK_M, q_async_layout, stats4_async_layout)
                        stage_wait = tlx.async_load_wait_group(1)
                    else:
                        stage_wait = tlx.async_load_wait_group(0)
                    apply_causal_mask = n0 + BLOCK_N - 1 > (m0 + step) * BLOCK_M + (SKV - SQ)
                    dk, dv = _d64_gqa8_direct_consume_loaded_stats(dk, dv, tlx.local_view(q_ring, current_slot),
                                                                   tlx.local_view(do_ring,
                                                                                  current_slot), lse_batch, delta_batch,
                                                                   stage_wait, k_nm, v_nm, m0 + step, n0, SM_SCALE, SQ,
                                                                   SKV, D, BLOCK_M, BLOCK_N, apply_causal_mask, True,
                                                                   step, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
                    if step % 2 == 1:
                        tl.debug_barrier()
            dk = tlx.require_layout(dk, mma_nd, pin=False)
            dv = tlx.require_layout(dv, mma_nd, pin=False)
        if local_head + 1 < 2:
            tl.debug_barrier()
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _d64_gqa8_d32_impl(
    Q,
    DO,
    LSE_TERM,
    Delta,
    q_ring,
    do_ring,
    lse_ring,
    delta_ring,
    k_nm,
    v_nm,
    pid_b,
    pid_hkv,
    off_split,
    n0,
    start_m_blk,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INTERLEAVED_D32: tl.constexpr,
    q_async_layout: tl.constexpr,
    stats_async_layout: tl.constexpr,
    mma_nm: tl.constexpr,
    mma_nd: tl.constexpr,
    q_t_op1_nm: tl.constexpr,
    p_op0_nd: tl.constexpr,
    q_op1_nd: tl.constexpr,
):
    """Carry only the selected low/high FP32 recurrences until epilogue."""
    dk_d0 = tlx.zeros((BLOCK_N, D // 2), tl.float32, layout=mma_nd)
    dk_d1 = tlx.zeros((BLOCK_N, D // 2), tl.float32, layout=mma_nd)
    dv_d0 = tlx.zeros((BLOCK_N, D // 2), tl.float32, layout=mma_nd)
    dv_d1 = tlx.zeros((BLOCK_N, D // 2), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = SQ // BLOCK_M
    full_pairs: tl.constexpr = num_m_blocks // 2
    has_odd: tl.constexpr = (num_m_blocks % 2) != 0
    peel_frontier = (start_m_blk % 2) != 0
    pair_start = start_m_blk + (start_m_blk % 2)

    for local_head in tl.static_range(0, 2):
        query_in_group = off_split * 2 + local_head
        q_head = (pid_b * HQ + pid_hkv * 8 + query_in_group).to(tl.int64)
        q_base = Q + q_head * SQ * D
        do_base = DO + q_head * SQ * D
        stats_base = q_head * SQ
        lse_base = LSE_TERM + stats_base
        delta_base = Delta + stats_base
        if peel_frontier:
            _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring,
                                                                                        0), tlx.local_view(do_ring, 0),
                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0), start_m_blk, D, BLOCK_M,
                                  q_async_layout, stats_async_layout)
            stage_wait = tlx.async_load_wait_group(0)
            dk_d0, dk_d1, dv_d0, dv_d1 = _d64_gqa8_d32_consume(dk_d0, dk_d1, dv_d0, dv_d1, tlx.local_view(q_ring, 0),
                                                               tlx.local_view(do_ring, 0), tlx.local_view(lse_ring, 0),
                                                               tlx.local_view(delta_ring, 0), stage_wait, k_nm, v_nm,
                                                               start_m_blk, n0, SM_SCALE, SQ, SKV, D, BLOCK_M, BLOCK_N,
                                                               INTERLEAVED_D32, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd,
                                                               q_op1_nd)
            dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
            dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
            dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
            dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
            # Every relaxed view is retired before slot-zero reuse.
            tl.debug_barrier()
        if pair_start < num_m_blocks:
            _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring,
                                                                                        0), tlx.local_view(do_ring, 0),
                                  tlx.local_view(lse_ring, 0), tlx.local_view(delta_ring, 0), pair_start, D, BLOCK_M,
                                  q_async_layout, stats_async_layout)
            for m_pair in range(pair_start // 2, full_pairs):
                m_blk_a = m_pair * 2
                m_blk_b = m_blk_a + 1
                # Issue all next-stage requests before any current-stage read.
                _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 1),
                                      tlx.local_view(do_ring, 1), tlx.local_view(lse_ring, 1),
                                      tlx.local_view(delta_ring, 1), m_blk_b, D, BLOCK_M, q_async_layout,
                                      stats_async_layout)
                stage_wait = tlx.async_load_wait_group(1)
                dk_d0, dk_d1, dv_d0, dv_d1 = _d64_gqa8_d32_consume(dk_d0, dk_d1, dv_d0,
                                                                   dv_d1, tlx.local_view(q_ring, 0),
                                                                   tlx.local_view(do_ring, 0),
                                                                   tlx.local_view(lse_ring, 0),
                                                                   tlx.local_view(delta_ring, 0), stage_wait, k_nm,
                                                                   v_nm, m_blk_a, n0, SM_SCALE, SQ, SKV, D, BLOCK_M,
                                                                   BLOCK_N, INTERLEAVED_D32, mma_nm, mma_nd, q_t_op1_nm,
                                                                   p_op0_nd, q_op1_nd)
                has_following = (m_pair + 1 < full_pairs) or has_odd
                if has_following:
                    # Retire every view before overwriting the ping slot.
                    tl.debug_barrier()
                    _d64_gqa8_issue_stage(q_base, do_base, lse_base, delta_base, tlx.local_view(q_ring, 0),
                                          tlx.local_view(do_ring, 0), tlx.local_view(lse_ring, 0),
                                          tlx.local_view(delta_ring, 0), m_blk_a + 2, D, BLOCK_M, q_async_layout,
                                          stats_async_layout)
                    stage_wait = tlx.async_load_wait_group(1)
                else:
                    stage_wait = tlx.async_load_wait_group(0)
                dk_d0, dk_d1, dv_d0, dv_d1 = _d64_gqa8_d32_consume(dk_d0, dk_d1, dv_d0,
                                                                   dv_d1, tlx.local_view(q_ring, 1),
                                                                   tlx.local_view(do_ring, 1),
                                                                   tlx.local_view(lse_ring, 1),
                                                                   tlx.local_view(delta_ring, 1), stage_wait, k_nm,
                                                                   v_nm, m_blk_b, n0, SM_SCALE, SQ, SKV, D, BLOCK_M,
                                                                   BLOCK_N, INTERLEAVED_D32, mma_nm, mma_nd, q_t_op1_nm,
                                                                   p_op0_nd, q_op1_nd)
                if m_pair + 1 < full_pairs:
                    tl.debug_barrier()
            dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
            dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
            dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
            dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
            if has_odd:
                stage_wait = tlx.async_load_wait_group(0)
                dk_d0, dk_d1, dv_d0, dv_d1 = _d64_gqa8_d32_consume(dk_d0, dk_d1, dv_d0,
                                                                   dv_d1, tlx.local_view(q_ring, 0),
                                                                   tlx.local_view(do_ring, 0),
                                                                   tlx.local_view(lse_ring, 0),
                                                                   tlx.local_view(delta_ring, 0), stage_wait, k_nm,
                                                                   v_nm, num_m_blocks - 1, n0, SM_SCALE, SQ, SKV, D,
                                                                   BLOCK_M, BLOCK_N, INTERLEAVED_D32, mma_nm, mma_nd,
                                                                   q_t_op1_nm, p_op0_nd, q_op1_nd)
                dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
                dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
                dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
                dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
        dk_d0 = tlx.require_layout(dk_d0, mma_nd, pin=False)
        dk_d1 = tlx.require_layout(dk_d1, mma_nd, pin=False)
        dv_d0 = tlx.require_layout(dv_d0, mma_nd, pin=False)
        dv_d1 = tlx.require_layout(dv_d1, mma_nd, pin=False)
        if local_head + 1 < 2:
            tl.debug_barrier()

    # This is the only D32 join: both selected recurrences die in epilogue.
    dk = tl.join(dk_d0, dk_d1)
    dk = tl.permute(dk, (0, 2, 1))
    dk = tl.reshape(dk, (BLOCK_N, D))
    dv = tl.join(dv_d0, dv_d1)
    dv = tl.permute(dv, (0, 2, 1))
    dv = tl.reshape(dv, (BLOCK_N, D))
    return (
        tlx.require_layout(dk, mma_nd, pin=False),
        tlx.require_layout(dv, mma_nd, pin=False),
    )


@triton.jit
def _attn_bwd_dkdv_d64_causal_gqa8_kernel(
    Q,
    K,
    V,
    DO,
    LSE_TERM,
    Delta,
    DK_PART,
    DV_PART,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    USE_GQA_XCD: tl.constexpr,
    USE_XCD_N_FAST: tl.constexpr,
    CYCLIC_QUERY_SPLIT: tl.constexpr,
    BATCH_STATS4: tl.constexpr,
    LIFETIME_MODE: tl.constexpr,
    LSE_MODE: tl.constexpr,
    DELTA_MODE: tl.constexpr,
):
    """Resident-K/V causal D64 GQA8 producer with four fixed partials."""
    tl.static_assert(D == 64 and BLOCK_M == 64 and (BLOCK_N == 64 or BLOCK_N == 128))
    tl.static_assert(HQ == HKV * 8)
    tl.static_assert(SQ % BLOCK_M == 0 and SKV % BLOCK_N == 0)
    tl.static_assert(SQ <= SKV)
    tl.static_assert(KV_SPLITS == _D64_CAUSAL_GQA8_KV_SPLITS_JIT)
    tl.static_assert(LSE_MODE == _D64_LSE_NEG_LOG2E_JIT)
    tl.static_assert(DELTA_MODE == _D64_DELTA_NEGATED_JIT)
    tl.static_assert(LIFETIME_MODE == _D64_GQA_INDEPENDENT_D32_JIT or LIFETIME_MODE == _D64_GQA_INTERLEAVED_D32_JIT
                     or LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT)
    if USE_XCD_N_FAST:
        tl.static_assert(USE_GQA_XCD)
    if CYCLIC_QUERY_SPLIT:
        tl.static_assert(USE_GQA_XCD and LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT)
    if BATCH_STATS4:
        tl.static_assert(not CYCLIC_QUERY_SPLIT and LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT)

    value = tl.program_id(0)
    num_n: tl.constexpr = SKV // BLOCK_N
    if USE_GQA_XCD:
        xcd = value % 8
        value //= 8
        if USE_XCD_N_FAST:
            pid_n = value % num_n
            value //= num_n
            off_split = value % KV_SPLITS
            bkv_group = value // KV_SPLITS
        else:
            off_split = value % KV_SPLITS
            value //= KV_SPLITS
            pid_n = value % num_n
            bkv_group = value // num_n
        bkv = bkv_group * 8 + xcd
        pid_hkv = bkv % HKV
        pid_b = bkv // HKV
    else:
        off_split = value % KV_SPLITS
        value //= KV_SPLITS
        pid_hkv = value % HKV
        value //= HKV
        pid_n = value % num_n
        pid_b = value // num_n
    n0 = pid_n * BLOCK_N

    mma_nm: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mma_nd: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    k_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    v_op0_nm: tl.constexpr = tlx.dot_operand_layout(0, mma_nm, k_width=8)
    q_t_op1_nm: tl.constexpr = tlx.dot_operand_layout(1, mma_nm, k_width=8)
    p_op0_nd: tl.constexpr = tlx.dot_operand_layout(0, mma_nd, k_width=4)
    q_op1_nd: tl.constexpr = tlx.dot_operand_layout(1, mma_nd, k_width=4)

    q_async_layout: tl.constexpr = tlx.layout(
        shape=(
            (2, 2, 2, 2, 2, 2, 2, 2),
            (2, 2, 2, 2),
        ),
        stride=(
            (8, 64, 128, 256, 512, 16, 32, 2048),
            (1, 2, 4, 1024),
        ),
    )
    stats_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 0), ()),
    )
    all_head_stats_layout: tl.constexpr = tlx.layout(
        shape=((8, 32), (2, )),
        stride=((64, 2), (1, )),
    )
    stats4_async_layout: tl.constexpr = tlx.layout(
        shape=((64, 4), ()),
        stride=((1, 64), ()),
    )
    if BLOCK_N == 64:
        out_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((64, 128, 256, 512, 8, 16, 1024, 2048), (1, 2, 4, 32)),
        )
    else:
        out_layout: tl.constexpr = tlx.layout(
            shape=(
                (2, 2, 2, 2, 2, 2, 2, 2),
                (2, 2, 2, 2, 2),
            ),
            stride=(
                (64, 128, 256, 512, 8, 16, 1024, 2048),
                (1, 2, 4, 32, 4096),
            ),
        )
    qdo_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
        ],
        block_bases=[],
        alignment=16,
    )
    stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=4,
    )
    # XOR head ownership into otherwise row-major bank bits.  Fixed-head
    # consumers retain full row rank, while the cooperative all-head store is
    # injective within each 32-lane half-wave.
    all_head_stats_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 8], [4, 16]],
        block_bases=[],
        alignment=8,
    )
    stats4_smem_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0]],
        block_bases=[],
        alignment=16,
    )
    qdo_stages: tl.constexpr = 4 if BATCH_STATS4 else 2
    q_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, qdo_stages, layout=qdo_smem_layout)
    do_ring = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, qdo_stages, layout=qdo_smem_layout)
    if CYCLIC_QUERY_SPLIT:
        lse_buffer = tlx.local_alloc((8, BLOCK_M), tl.float32, 1, layout=all_head_stats_smem_layout)
        delta_buffer = tlx.local_alloc((8, BLOCK_M), tl.float32, 1, layout=all_head_stats_smem_layout)
        lse_all = tlx.local_view(lse_buffer, 0)
        delta_all = tlx.local_view(delta_buffer, 0)
    else:
        lse_ring = tlx.local_alloc((BLOCK_M, ), tl.float32, 2, layout=stats_smem_layout)
        delta_ring = tlx.local_alloc((BLOCK_M, ), tl.float32, 2, layout=stats_smem_layout)
        if BATCH_STATS4:
            lse_batch_ring = tlx.local_alloc((4, BLOCK_M), tl.float32, 2, layout=stats4_smem_layout)
            delta_batch_ring = tlx.local_alloc((4, BLOCK_M), tl.float32, 2, layout=stats4_smem_layout)

    # K/V are loaded once into their final MFMA operand layouts and remain
    # resident through every owned head/BM64 visit. They never round-trip LDS.
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    kv_base = kv_head * SKV * D
    kv_rows = n0 + tl.arange(0, BLOCK_N)
    kv_cols = tl.arange(0, D)
    kv_offsets = (kv_rows[:, None] * D + kv_cols[None, :]).to(tl.int32)
    k_offsets = tlx.require_layout(kv_offsets, k_op0_nm, pin=False)
    v_offsets = tlx.require_layout(kv_offsets, v_op0_nm, pin=False)
    k_nm = tlx.buffer_load(K + kv_base, k_offsets)
    # K participates only in the score MFMA. Folding the constant softmax
    # scale into this resident operand amortizes it over every owned query
    # tile and head.
    k_scale = tlx.require_layout(
        tl.full((BLOCK_N, D), SM_SCALE * 1.4426950408889634, tl.float32),
        k_op0_nm,
        pin=False,
    )
    k_nm = (k_nm.to(tl.float32) * k_scale).to(tl.bfloat16)
    k_nm = tlx.require_layout(k_nm, k_op0_nm, pin=False)
    v_nm = tlx.buffer_load(V + kv_base, v_offsets)
    v_nm = tlx.require_layout(v_nm, v_op0_nm, pin=False)

    # Compute the bottom-right physical frontier once per resident K/V owner.
    start_m_blk = tl.maximum((n0 - (SKV - SQ)) // BLOCK_M, 0)
    # Implementations branch on this exact predicate so an unmasked BM64 has
    # no elementwise causal arithmetic:
    # n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)
    if LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT:
        if CYCLIC_QUERY_SPLIT:
            dk, dv = _d64_gqa8_all_head_direct_d64_impl(Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_all, delta_all,
                                                        k_nm, v_nm, pid_b, pid_hkv, off_split, n0, start_m_blk,
                                                        SM_SCALE, HQ, SQ, SKV, D, BLOCK_M, BLOCK_N, KV_SPLITS,
                                                        q_async_layout, all_head_stats_layout, mma_nm, mma_nd,
                                                        q_t_op1_nm, p_op0_nd, q_op1_nd)
        elif BATCH_STATS4:
            dk, dv = _d64_gqa8_async_stats4_direct_d64_impl(Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_ring,
                                                            delta_ring, lse_batch_ring, delta_batch_ring, k_nm, v_nm,
                                                            pid_b, pid_hkv, off_split, n0, start_m_blk, SM_SCALE, HQ,
                                                            SQ, SKV, D, BLOCK_M, BLOCK_N, q_async_layout,
                                                            stats_async_layout, stats4_async_layout, mma_nm, mma_nd,
                                                            q_t_op1_nm, p_op0_nd, q_op1_nd)
        else:
            dk, dv = _d64_gqa8_direct_d64_impl(Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_ring, delta_ring, k_nm,
                                               v_nm, pid_b, pid_hkv, off_split, n0, start_m_blk, SM_SCALE, HQ, SQ, SKV,
                                               D, BLOCK_M, BLOCK_N, KV_SPLITS, False, q_async_layout,
                                               stats_async_layout, mma_nm, mma_nd, q_t_op1_nm, p_op0_nd, q_op1_nd)
    else:
        tl.static_assert(not CYCLIC_QUERY_SPLIT)
        INTERLEAVED_D32: tl.constexpr = (LIFETIME_MODE == _D64_GQA_INTERLEAVED_D32_JIT)
        dk, dv = _d64_gqa8_d32_impl(Q, DO, LSE_TERM, Delta, q_ring, do_ring, lse_ring, delta_ring, k_nm, v_nm, pid_b,
                                    pid_hkv, off_split, n0, start_m_blk, SM_SCALE, HQ, SQ, SKV, D, BLOCK_M, BLOCK_N,
                                    INTERLEAVED_D32, q_async_layout, stats_async_layout, mma_nm, mma_nd, q_t_op1_nm,
                                    p_op0_nd, q_op1_nd)

    dk_scale = tlx.require_layout(tl.full((BLOCK_N, D), SM_SCALE, tl.float32), mma_nd, pin=False)
    dk = tlx.require_layout(dk, mma_nd, pin=False) * dk_scale
    dv = tlx.require_layout(dv, mma_nd, pin=False)
    output_offsets = (tl.arange(0, BLOCK_N)[:, None] * D + tl.arange(0, D)[None, :]).to(tl.int32)
    output_offsets = tlx.require_layout(output_offsets, out_layout, pin=False)
    dk_out = tlx.require_layout(dk.to(tl.bfloat16), out_layout, pin=False)
    dv_out = tlx.require_layout(dv.to(tl.bfloat16), out_layout, pin=False)
    partial_head = ((pid_b * HKV + pid_hkv) * KV_SPLITS + off_split).to(tl.int64)
    partial_base = partial_head * SKV * D + n0 * D
    tlx.buffer_store(dk_out, DK_PART + partial_base, output_offsets)
    tlx.buffer_store(dv_out, DV_PART + partial_base, output_offsets)


@triton.jit
def _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel(
    DK_PART,
    DV_PART,
    DK,
    DV,
    HKV: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Reduce split 0,1,2,3 in FP32, then narrow exactly once."""
    tl.static_assert(D == 64 and (BLOCK_N == 64 or BLOCK_N == 128) and SKV % BLOCK_N == 0)
    tl.static_assert(KV_SPLITS == _D64_CAUSAL_GQA8_KV_SPLITS_JIT)
    pid_n = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_b = tl.program_id(2)
    n0 = pid_n * BLOCK_N
    if BLOCK_N == 64:
        out_layout: tl.constexpr = tlx.layout(
            shape=((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2)),
            stride=((64, 128, 256, 512, 8, 16, 1024, 2048), (1, 2, 4, 32)),
        )
    else:
        out_layout: tl.constexpr = tlx.layout(
            shape=(
                (2, 2, 2, 2, 2, 2, 2, 2),
                (2, 2, 2, 2, 2),
            ),
            stride=(
                (64, 128, 256, 512, 8, 16, 1024, 2048),
                (1, 2, 4, 32, 4096),
            ),
        )
    offsets = (tl.arange(0, BLOCK_N)[:, None] * D + tl.arange(0, D)[None, :]).to(tl.int32)
    offsets = tlx.require_layout(offsets, out_layout, pin=False)
    owner = (pid_b * HKV + pid_hkv).to(tl.int64)
    split_stride: tl.constexpr = SKV * D
    partial_base = owner * KV_SPLITS * split_stride + n0 * D

    dk_acc = tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)
    dv_acc = tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)
    for split in tl.static_range(0, KV_SPLITS):
        dk_part = tlx.buffer_load(DK_PART + partial_base + split * split_stride, offsets)
        dv_part = tlx.buffer_load(DV_PART + partial_base + split * split_stride, offsets)
        dk_acc += dk_part.to(tl.float32)
        dv_acc += dv_part.to(tl.float32)

    output_base = owner * SKV * D + n0 * D
    dk_out = tlx.require_layout(dk_acc.to(tl.bfloat16), out_layout, pin=False)
    dv_out = tlx.require_layout(dv_acc.to(tl.bfloat16), out_layout, pin=False)
    tlx.buffer_store(dk_out, DK + output_base, offsets)
    tlx.buffer_store(dv_out, DV + output_base, offsets)


@triton.jit
def _attn_bwd_dq_d64_update(
    q_tile,
    do_tile,
    lse,
    delta,
    dq,
    k_tile,
    k_t,
    v_t,
    offs_m,
    offs_n,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
):
    log2e: tl.constexpr = 1.4426950408889634
    scores = tl.dot(q_tile, k_t)
    scores = scores * (SM_SCALE * log2e) - lse[:, None] * log2e
    valid = (offs_m[:, None] < SQ) & (offs_n[None, :] < SKV)
    if IS_CAUSAL:
        valid = valid & (offs_n[None, :] <= offs_m[:, None] + (SKV - SQ))
    scores = tl.where(valid, scores, float("-inf"))
    p = tl.math.exp2(scores)
    dp = tl.dot(do_tile, v_t)
    ds = p * (dp - delta[:, None])
    return tl.dot(ds.to(tl.bfloat16), k_tile, dq)


@triton.jit
def _attn_bwd_dq_d64_direct_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OWNER_ROWS: tl.constexpr,
):
    """Query-owned D64 dQ with shared K/V and direct fragment stores."""
    tl.static_assert(D == 64)
    tl.static_assert(BLOCK_M == 32 or BLOCK_M == 64)
    tl.static_assert(BLOCK_N == 32 or BLOCK_N == 64 or BLOCK_N == 256)
    tl.static_assert(OWNER_ROWS == BLOCK_M or OWNER_ROWS == 192 or OWNER_ROWS == 256)
    tl.static_assert(HQ % HKV == 0)
    pid_owner = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_b = tl.program_id(2)
    group_size: tl.constexpr = HQ // HKV
    pid_hkv = pid_hq // group_size
    offs_d = tl.arange(0, D)
    q_head = (pid_b * HQ + pid_hq).to(tl.int64)
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    q_base = q_head * SQ * D
    kv_base = kv_head * SKV * D
    stats_base = q_head * SQ

    num_owners: tl.constexpr = tl.cdiv(SQ, OWNER_ROWS)
    logical_owner = num_owners - 1 - pid_owner if IS_CAUSAL else pid_owner
    owner_start = logical_owner * OWNER_ROWS
    offs_m0 = owner_start + tl.arange(0, BLOCK_M)
    qdo_ptrs0 = q_base + offs_m0[:, None] * D + offs_d[None, :]
    qdo_mask0 = offs_m0[:, None] < SQ
    q0 = tl.load(Q + qdo_ptrs0, mask=qdo_mask0, other=0.0)
    do0 = tl.load(DO + qdo_ptrs0, mask=qdo_mask0, other=0.0)
    lse0 = tl.load(LSE + stats_base + offs_m0, mask=offs_m0 < SQ, other=0.0)
    delta0 = tl.load(Delta + stats_base + offs_m0, mask=offs_m0 < SQ, other=0.0)
    dq0 = tl.zeros((BLOCK_M, D), tl.float32)
    if OWNER_ROWS >= 128:
        offs_m1 = owner_start + 64 + tl.arange(0, 64)
        qdo_ptrs1 = q_base + offs_m1[:, None] * D + offs_d[None, :]
        qdo_mask1 = offs_m1[:, None] < SQ
        q1 = tl.load(Q + qdo_ptrs1, mask=qdo_mask1, other=0.0)
        do1 = tl.load(DO + qdo_ptrs1, mask=qdo_mask1, other=0.0)
        lse1 = tl.load(LSE + stats_base + offs_m1, mask=offs_m1 < SQ, other=0.0)
        delta1 = tl.load(Delta + stats_base + offs_m1, mask=offs_m1 < SQ, other=0.0)
        dq1 = tl.zeros((64, D), tl.float32)
        offs_m2 = owner_start + 128 + tl.arange(0, 64)
        qdo_ptrs2 = q_base + offs_m2[:, None] * D + offs_d[None, :]
        qdo_mask2 = offs_m2[:, None] < SQ
        q2 = tl.load(Q + qdo_ptrs2, mask=qdo_mask2, other=0.0)
        do2 = tl.load(DO + qdo_ptrs2, mask=qdo_mask2, other=0.0)
        lse2 = tl.load(LSE + stats_base + offs_m2, mask=offs_m2 < SQ, other=0.0)
        delta2 = tl.load(Delta + stats_base + offs_m2, mask=offs_m2 < SQ, other=0.0)
        dq2 = tl.zeros((64, D), tl.float32)
    if OWNER_ROWS == 256:
        offs_m3 = owner_start + 192 + tl.arange(0, 64)
        qdo_ptrs3 = q_base + offs_m3[:, None] * D + offs_d[None, :]
        qdo_mask3 = offs_m3[:, None] < SQ
        q3 = tl.load(Q + qdo_ptrs3, mask=qdo_mask3, other=0.0)
        do3 = tl.load(DO + qdo_ptrs3, mask=qdo_mask3, other=0.0)
        lse3 = tl.load(LSE + stats_base + offs_m3, mask=offs_m3 < SQ, other=0.0)
        delta3 = tl.load(Delta + stats_base + offs_m3, mask=offs_m3 < SQ, other=0.0)
        dq3 = tl.zeros((64, D), tl.float32)

    if BLOCK_N == 32:
        row_bases: tl.constexpr = [[16, 0], [8, 0], [1, 0], [2, 0], [4, 0]]
    else:
        if BLOCK_N == 64:
            row_bases: tl.constexpr = [[16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
        else:
            row_bases: tl.constexpr = [
                [16, 0],
                [32, 0],
                [64, 0],
                [128, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]] + row_bases,
        [BLOCK_N, D],
    )
    k_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=shared_layout)
    v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=shared_layout)
    if IS_CAUSAL:
        owner_key_end = tl.minimum(SKV, owner_start + OWNER_ROWS + (SKV - SQ))
        num_n_blocks = (owner_key_end + BLOCK_N - 1) // BLOCK_N
    else:
        num_n_blocks: tl.constexpr = tl.cdiv(SKV, BLOCK_N)

    for n_block in range(0, num_n_blocks):
        tl.debug_barrier()
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_ptrs = kv_base + offs_n[:, None] * D + offs_d[None, :]
        kv_mask = offs_n[:, None] < SKV
        k_token = tlx.async_load(K + kv_ptrs, tlx.local_view(k_buffer, 0), mask=kv_mask, other=0.0)
        v_token = tlx.async_load(V + kv_ptrs, tlx.local_view(v_buffer, 0), mask=kv_mask, other=0.0)
        tlx.async_load_commit_group([k_token, v_token])
        kv_wait = tlx.async_load_wait_group(0)
        k_tile = tlx.local_load(tlx.local_view(k_buffer, 0), token=kv_wait)
        k_t = tlx.local_load(tlx.local_trans(tlx.local_view(k_buffer, 0)), token=kv_wait)
        v_t = tlx.local_load(tlx.local_trans(tlx.local_view(v_buffer, 0)), token=kv_wait)
        dq0 = _attn_bwd_dq_d64_update(q0, do0, lse0, delta0, dq0, k_tile, k_t, v_t, offs_m0, offs_n, SM_SCALE,
                                      IS_CAUSAL, SQ, SKV)
        if OWNER_ROWS >= 128:
            dq1 = _attn_bwd_dq_d64_update(q1, do1, lse1, delta1, dq1, k_tile, k_t, v_t, offs_m1, offs_n, SM_SCALE,
                                          IS_CAUSAL, SQ, SKV)
            dq2 = _attn_bwd_dq_d64_update(q2, do2, lse2, delta2, dq2, k_tile, k_t, v_t, offs_m2, offs_n, SM_SCALE,
                                          IS_CAUSAL, SQ, SKV)
        if OWNER_ROWS == 256:
            dq3 = _attn_bwd_dq_d64_update(q3, do3, lse3, delta3, dq3, k_tile, k_t, v_t, offs_m3, offs_n, SM_SCALE,
                                          IS_CAUSAL, SQ, SKV)

    tl.store(DQ + qdo_ptrs0, (dq0 * SM_SCALE).to(tl.bfloat16), mask=qdo_mask0)
    if OWNER_ROWS >= 128:
        tl.store(DQ + qdo_ptrs1, (dq1 * SM_SCALE).to(tl.bfloat16), mask=qdo_mask1)
        tl.store(DQ + qdo_ptrs2, (dq2 * SM_SCALE).to(tl.bfloat16), mask=qdo_mask2)
    if OWNER_ROWS == 256:
        tl.store(DQ + qdo_ptrs3, (dq3 * SM_SCALE).to(tl.bfloat16), mask=qdo_mask3)


@triton.jit
def _attn_bwd_dkdv_d64_direct_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """KV-owned D64 dK/dV with deterministic FP32 GQA accumulation."""
    tl.static_assert(D == 64)
    tl.static_assert(BLOCK_M == 64 and BLOCK_N == 64)
    tl.static_assert(HQ % HKV == 0)
    tl.static_assert((HQ // HKV) % KV_SPLITS == 0)
    pid_n = tl.program_id(0)
    pid_hkv_split = tl.program_id(1)
    pid_b = tl.program_id(2)
    group_size: tl.constexpr = HQ // HKV
    heads_per_split: tl.constexpr = group_size // KV_SPLITS
    pid_hkv = pid_hkv_split // KV_SPLITS
    pid_split = pid_hkv_split % KV_SPLITS
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    kv_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    kv_base = kv_head * SKV * D
    kv_ptrs = kv_base + offs_n[:, None] * D + offs_d[None, :]
    kv_mask = offs_n[:, None] < SKV
    k_tile = tl.load(K + kv_ptrs, mask=kv_mask, other=0.0)
    v_tile = tl.load(V + kv_ptrs, mask=kv_mask, other=0.0)

    shared_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 32)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
        [BLOCK_M, D],
    )
    q_buffer = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 1, layout=shared_layout)
    do_buffer = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 1, layout=shared_layout)
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    num_m_blocks: tl.constexpr = tl.cdiv(SQ, BLOCK_M)
    if IS_CAUSAL:
        first_m_block = tl.maximum(0, pid_n * BLOCK_N - (SKV - SQ)) // BLOCK_M
    else:
        first_m_block: tl.constexpr = 0
    causal_shift: tl.constexpr = SKV - SQ
    log2e: tl.constexpr = 1.4426950408889634

    for local_group_head in tl.static_range(0, heads_per_split):
        group_head = pid_split * heads_per_split + local_group_head
        pid_hq = pid_hkv * group_size + group_head
        q_head = (pid_b * HQ + pid_hq).to(tl.int64)
        q_base = q_head * SQ * D
        stats_base = q_head * SQ
        for m_block in range(first_m_block, num_m_blocks):
            tl.debug_barrier()
            offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            qdo_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
            qdo_mask = offs_m[:, None] < SQ
            q_token = tlx.async_load(Q + qdo_ptrs, tlx.local_view(q_buffer, 0), mask=qdo_mask, other=0.0)
            do_token = tlx.async_load(DO + qdo_ptrs, tlx.local_view(do_buffer, 0), mask=qdo_mask, other=0.0)
            tlx.async_load_commit_group([q_token, do_token])
            qdo_wait = tlx.async_load_wait_group(0)
            q_tile = tlx.local_load(tlx.local_view(q_buffer, 0), token=qdo_wait)
            do_tile = tlx.local_load(tlx.local_view(do_buffer, 0), token=qdo_wait)
            q_t = tlx.local_load(tlx.local_trans(tlx.local_view(q_buffer, 0)), token=qdo_wait)
            do_t = tlx.local_load(tlx.local_trans(tlx.local_view(do_buffer, 0)), token=qdo_wait)
            lse = tl.load(LSE + stats_base + offs_m, mask=offs_m < SQ, other=0.0)
            delta = tl.load(Delta + stats_base + offs_m, mask=offs_m < SQ, other=0.0)
            scores_t = tl.dot(k_tile, q_t)
            scores_t = scores_t * (SM_SCALE * log2e) - lse[None, :] * log2e
            valid = (offs_n[:, None] < SKV) & (offs_m[None, :] < SQ)
            if IS_CAUSAL:
                valid = valid & (offs_n[:, None] <= offs_m[None, :] + causal_shift)
            scores_t = tl.where(valid, scores_t, float("-inf"))
            p_t = tl.math.exp2(scores_t)
            dp_t = tl.dot(v_tile, do_t)
            ds_t = p_t * (dp_t - delta[None, :])
            dv = tl.dot(p_t.to(tl.bfloat16), do_tile, dv)
            dk = tl.dot(ds_t.to(tl.bfloat16), q_tile, dk)

    dk *= SM_SCALE
    if KV_SPLITS == 1:
        output_ptrs = kv_ptrs
    else:
        partial_head = ((pid_b * HKV + pid_hkv) * KV_SPLITS + pid_split).to(tl.int64)
        output_ptrs = partial_head * SKV * D + offs_n[:, None] * D + offs_d[None, :]
    tl.store(DK + output_ptrs, dk.to(tl.bfloat16), mask=kv_mask)
    tl.store(DV + output_ptrs, dv.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def _attn_bwd_dkdv_d64_reduce_kernel(
    DK_PART,
    DV_PART,
    DK,
    DV,
    HKV: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce fixed-order BF16 GQA partials in FP32 and narrow once."""
    tl.static_assert(D == 64 and BLOCK_N == 64)
    pid_n = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask = offs_n[:, None] < SKV
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)
    for split in tl.static_range(0, KV_SPLITS):
        partial_head = ((pid_b * HKV + pid_hkv) * KV_SPLITS + split).to(tl.int64)
        partial_ptrs = partial_head * SKV * D + offs_n[:, None] * D + offs_d[None, :]
        dk += tl.load(DK_PART + partial_ptrs, mask=mask, other=0.0).to(tl.float32)
        dv += tl.load(DV_PART + partial_ptrs, mask=mask, other=0.0).to(tl.float32)
    output_head = (pid_b * HKV + pid_hkv).to(tl.int64)
    output_ptrs = output_head * SKV * D + offs_n[:, None] * D + offs_d[None, :]
    tl.store(DK + output_ptrs, dk.to(tl.bfloat16), mask=mask)
    tl.store(DV + output_ptrs, dv.to(tl.bfloat16), mask=mask)


def _launch_bwd_d64_dq(q, k, v, do, lse, delta, dq, sm_scale, causal, dispatch):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    dq_block_m = 32 if dispatch.owner_rows == 32 else 64
    _attn_bwd_dq_d64_direct_kernel[(triton.cdiv(sq, dispatch.owner_rows), hq, batch)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        HQ=hq,
        HKV=hkv,
        SQ=sq,
        SKV=skv,
        D=head_dim,
        BLOCK_M=dq_block_m,
        BLOCK_N=dispatch.key_rows,
        OWNER_ROWS=dispatch.owner_rows,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _launch_bwd_d64_causal_dq(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    lse_term,
    dq,
    sm_scale,
    dispatch,
):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    if dispatch.stat_mode == _D64_MHA_POSITIVE:
        if lse_term is not None:
            raise ValueError("MHA positive dQ must not receive lse_term")
        kernel = _attn_bwd_dq_d64_causal_mha_kernel
    elif dispatch.stat_mode == _D64_GQA_SIGNED:
        if lse_term is None:
            raise ValueError("GQA signed dQ requires lse_term")
        kernel = _attn_bwd_dq_d64_causal_gqa8_kernel
    else:
        raise ValueError(f"unknown dQ stat mode {dispatch.stat_mode!r}")

    for launch in dispatch.dq_launches:
        grid = (batch * hq * launch.launch_tiles, )
        args = (q, k, v, o, do, lse, delta)
        if dispatch.stat_mode == _D64_GQA_SIGNED:
            args += (lse_term, )
        args += (dq, )
        kernel[grid](
            *args,
            SM_SCALE=sm_scale,
            HQ=hq,
            HKV=hkv,
            SQ=sq,
            SKV=skv,
            D=head_dim,
            OWNER_ROWS=dispatch.owner_rows,
            LOGICAL_N=dispatch.dq_logical_n,
            USE_DQ_XCD=dispatch.dq_use_xcd,
            SKIP_OWNER_TAIL=launch.skip_owner_tail,
            OWNER_PID_BASE=launch.owner_pid_base,
            LAUNCH_Q_TILES=launch.launch_q_tiles,
            OWNER_FRAGMENTS=launch.owner_fragments,
            GRID_OWNER_M=launch.grid_owner_m,
            KV_PIPELINE_STAGES=_D64_DQ_KV_STAGES,
            num_warps=4,
            matrix_instr_nonkdim=_matrix_instr_nonkdim(),
        )


def _launch_bwd_d64_causal_mha_dkdv(q, k, v, do, lse, delta, dk, dv, sm_scale, dispatch):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    _require_d64_dispatch_variant(
        dispatch,
        "causal_scheduled_mha",
        stat_mode=_D64_MHA_POSITIVE,
        kv_splits=1,
    )
    grid = (batch * hkv * triton.cdiv(skv, 64), )
    _attn_bwd_dkdv_d64_causal_mha_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        SM_SCALE=sm_scale,
        HQ=hq,
        HKV=hkv,
        SQ=sq,
        SKV=skv,
        D=head_dim,
        BLOCK_M=32,
        BLOCK_N=64,
        LSE_MODE=_D64_LSE_NATURAL_LOG,
        DELTA_MODE=_D64_DELTA_POSITIVE,
        num_warps=2,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _launch_bwd_d64_causal_gqa8_dkdv(
    q,
    k,
    v,
    do,
    lse_term,
    delta,
    dk_part,
    dv_part,
    sm_scale,
    dispatch,
):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    _require_d64_dispatch_variant(
        dispatch,
        "causal_scheduled_gqa8",
        stat_mode=_D64_GQA_SIGNED,
        kv_splits=_D64_CAUSAL_GQA8_KV_SPLITS,
    )
    use_gqa_xcd = dispatch.gqa_grid_mode != _D64_GQA_SPLIT_FAST
    use_xcd_n_fast = dispatch.gqa_grid_mode == _D64_GQA_XCD_N_FAST
    kv_splits = dispatch.kv_splits
    block_n = 128
    batch_stats4 = _d64_causal_gqa8_batch_stats4(sq, skv, dispatch)
    grid = (batch * hkv * kv_splits * triton.cdiv(skv, block_n), )
    _attn_bwd_dkdv_d64_causal_gqa8_kernel[grid](
        q,
        k,
        v,
        do,
        lse_term,
        delta,
        dk_part,
        dv_part,
        SM_SCALE=sm_scale,
        HQ=hq,
        HKV=hkv,
        SQ=sq,
        SKV=skv,
        D=head_dim,
        BLOCK_M=64,
        BLOCK_N=block_n,
        KV_SPLITS=kv_splits,
        USE_GQA_XCD=use_gqa_xcd,
        USE_XCD_N_FAST=use_xcd_n_fast,
        CYCLIC_QUERY_SPLIT=dispatch.cyclic_query_split,
        BATCH_STATS4=batch_stats4,
        LIFETIME_MODE=dispatch.dkdv_lifetime,
        LSE_MODE=_D64_LSE_NEG_LOG2E,
        DELTA_MODE=_D64_DELTA_NEGATED,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv):
    batch, hkv, skv, head_dim = dk.shape
    block_n = 64
    _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel[(triton.cdiv(skv, block_n), hkv, batch)](
        dk_part,
        dv_part,
        dk,
        dv,
        HKV=hkv,
        SKV=skv,
        D=head_dim,
        BLOCK_N=block_n,
        KV_SPLITS=dk_part.shape[2],
        num_warps=4,
    )


def _allocate_bwd_d64_kv_partials(k, kv_splits):
    if kv_splits == 1:
        return None, None
    batch, hkv, skv, head_dim = k.shape
    partial_shape = (batch, hkv, kv_splits, skv, head_dim)
    dk_part = torch.empty(partial_shape, device=k.device, dtype=k.dtype)
    return dk_part, torch.empty_like(dk_part)


def _allocate_bwd_d64_causal_gqa8_workspaces(q, k):
    batch, hq, sq, _d = q.shape
    _kb, hkv, skv, head_dim = k.shape
    lse_term = torch.empty((batch, hq, sq), device=q.device, dtype=torch.float32)
    partial_shape = (batch, hkv, _D64_CAUSAL_GQA8_KV_SPLITS, skv, head_dim)
    dk_part = torch.empty(partial_shape, device=k.device, dtype=torch.bfloat16)
    return lse_term, dk_part, torch.empty_like(dk_part)


def _allocate_bwd_d64_fused_workspaces(q, k, dispatch):
    _require_d64_dispatch_variant(dispatch, "noncausal_fused_n256")
    dq_acc = torch.empty_like(q, dtype=torch.float32)
    if dispatch.kv_splits == 1:
        return dq_acc, None, None
    dk_part, dv_part = _allocate_bwd_d64_kv_partials(k, dispatch.kv_splits)
    return dq_acc, dk_part, dv_part


def _launch_bwd_d64_fused_n256(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq_acc,
    dk_owner,
    dv_owner,
    sm_scale,
    dispatch,
):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    _attn_bwd_d64_fused_n256_kernel[(triton.cdiv(skv, 256), hq, batch)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_acc,
        dk_owner,
        dv_owner,
        SM_SCALE=sm_scale,
        HQ=hq,
        HKV=hkv,
        SQ=sq,
        SKV=skv,
        D=head_dim,
        BLOCK_M=32,
        BLOCK_N=256,
        KV_SPLITS=dispatch.kv_splits,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _launch_bwd_d64_fused_dq_convert(dq_acc, dq):
    batch, hq, sq, head_dim = dq.shape
    block_m = 64
    _attn_bwd_d64_fused_dq_convert_kernel[(triton.cdiv(sq, block_m), batch * hq)](
        dq_acc,
        dq,
        N=sq,
        D=head_dim,
        BLOCK_M=block_m,
        num_warps=4,
    )


def _launch_bwd_d64_dkdv(q, k, v, do, lse, delta, dk_target, dv_target, sm_scale, causal, dispatch):
    batch, hq, sq, head_dim = q.shape
    _k_batch, hkv, skv, _k_head_dim = k.shape
    dkdv_block = 64
    _attn_bwd_dkdv_d64_direct_kernel[(triton.cdiv(skv, dkdv_block), hkv * dispatch.kv_splits, batch)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk_target,
        dv_target,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        HQ=hq,
        HKV=hkv,
        SQ=sq,
        SKV=skv,
        D=head_dim,
        BLOCK_M=dkdv_block,
        BLOCK_N=dkdv_block,
        KV_SPLITS=dispatch.kv_splits,
        num_warps=4,
        matrix_instr_nonkdim=_matrix_instr_nonkdim(),
    )


def _launch_bwd_d64_kv_reduce(dk_part, dv_part, dk, dv, dispatch):
    batch, hkv, skv, head_dim = dk.shape
    block_n = 64
    _attn_bwd_dkdv_d64_reduce_kernel[(triton.cdiv(skv, block_n), hkv, batch)](
        dk_part,
        dv_part,
        dk,
        dv,
        HKV=hkv,
        SKV=skv,
        D=head_dim,
        KV_SPLITS=dispatch.kv_splits,
        BLOCK_N=block_n,
        num_warps=4,
    )


def _run_bwd_d64_direct(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal, dispatch):
    _validate_d64_dispatch(tuple(q.shape), tuple(k.shape), causal, dispatch)
    if dispatch.family not in {
            "noncausal_direct_n256",
            "causal_m192",
            "causal_m256",
    }:
        _invalid_d64_dispatch(dispatch, "family cannot use the direct route")
    _launch_bwd_d64_dq(q, k, v, do, lse, delta, dq, sm_scale, causal, dispatch)
    dk_part, dv_part = _allocate_bwd_d64_kv_partials(k, dispatch.kv_splits)
    dk_target = dk if dk_part is None else dk_part
    dv_target = dv if dv_part is None else dv_part
    _launch_bwd_d64_dkdv(q, k, v, do, lse, delta, dk_target, dv_target, sm_scale, causal, dispatch)
    if dk_part is not None:
        _launch_bwd_d64_kv_reduce(dk_part, dv_part, dk, dv, dispatch)


def _select_d64_dispatch_for_device(q, k, v, o, do, lse, sm_scale, causal):
    properties = torch.cuda.get_device_properties(q.device)
    bases_aligned_16 = all(tensor.data_ptr() % 16 == 0 for tensor in (q, k, v, o, do, lse))
    return _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        causal,
        arch=properties.gcnArchName,
        cu_count=properties.multi_processor_count,
        sm_scale=sm_scale,
        bases_aligned_16=bases_aligned_16,
    )


def _run_bwd_d64(q, k, v, o, do, lse, delta, dq, dk, dv, sm_scale, causal, dispatch):
    _validate_d64_dispatch(tuple(q.shape), tuple(k.shape), causal, dispatch)
    if dispatch.family == "causal_scheduled_gqa8":
        lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(q, k)
        _launch_bwd_d64_causal_dq(
            q,
            k,
            v,
            o,
            do,
            lse,
            delta,
            lse_term,
            dq,
            sm_scale,
            dispatch,
        )
        _launch_bwd_d64_causal_gqa8_dkdv(
            q,
            k,
            v,
            do,
            lse_term,
            delta,
            dk_part,
            dv_part,
            sm_scale,
            dispatch,
        )
        _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
        return
    if dispatch.family == "causal_scheduled_mha":
        _launch_bwd_d64_causal_dq(
            q,
            k,
            v,
            o,
            do,
            lse,
            delta,
            None,
            dq,
            sm_scale,
            dispatch,
        )
        _launch_bwd_d64_causal_mha_dkdv(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            sm_scale,
            dispatch,
        )
        return
    if dispatch.family == "noncausal_fused_n256":
        dq_acc, dk_part, dv_part = _allocate_bwd_d64_fused_workspaces(q, k, dispatch)
        dk_owner = dk if dk_part is None else dk_part
        dv_owner = dv if dv_part is None else dv_part
        _run_bwd_preprocess(o, do, delta, dq_acc=dq_acc)
        _launch_bwd_d64_fused_n256(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_acc,
            dk_owner,
            dv_owner,
            sm_scale,
            dispatch,
        )
        _launch_bwd_d64_fused_dq_convert(dq_acc, dq)
        if dk_part is not None:
            _launch_bwd_d64_kv_reduce(dk_part, dv_part, dk, dv, dispatch)
        return

    _run_bwd_preprocess(o, do, delta)
    _run_bwd_d64_direct(q, k, v, do, lse, delta, dq, dk, dv, sm_scale, causal, dispatch)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 256, 64), id="group1"),
        pytest.param((1, 4, 2, 256, 256, 64), id="group2-hkv2"),
        pytest.param((1, 3, 1, 256, 256, 64), id="group3"),
        pytest.param((1, 4, 1, 256, 256, 64), id="group4"),
        pytest.param((1, 16, 2, 256, 256, 64), id="group8-hkv2"),
    ],
)
def test_d64_public_contract_group_matrix_gfx950(shape, causal):
    group_size = shape[1] // shape[2]
    seed = 23 + group_size + int(causal)
    case = _make_d64_gqa_smoke_case(shape, causal=causal, seed=seed)
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_negative_scale_gfx950():
    case = _make_d64_gqa_smoke_case(
        (1, 1, 1, 256, 256, 64),
        causal=True,
        seed=47,
        sm_scale=-0.125,
    )
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        expected_norm = torch.linalg.vector_norm(expected.float())
        assert expected_norm.item() > 0.0, name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / expected_norm
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_bottom_right_rectangular_causal_gqa_public_contract_gfx950():
    case = _make_d64_gqa_smoke_case((1, 8, 1, 256, 512, 64), causal=True, seed=41)
    actual_grads = fa_backward(*case.kernel_args)
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("causal", "expected_family", "seed"),
    (
        pytest.param(False, "noncausal_direct_n256", 127, id="noncausal"),
        pytest.param(True, "causal_m192", 131, id="bottom-right-causal"),
    ),
)
def test_d64_64_aligned_public_fallback_gfx950(monkeypatch, causal, expected_family, seed):
    case = _make_d64_gqa_smoke_case((1, 1, 1, 320, 384, 64), causal=causal, seed=seed)
    assert torch.count_nonzero(case.q).item() > 0
    dispatches = []
    original_run = _run_bwd_d64

    def record_dispatch(*args):
        dispatches.append(args[-1])
        return original_run(*args)

    monkeypatch.setitem(globals(), "_run_bwd_d64", record_dispatch)
    actual_grads = fa_backward(*case.kernel_args)

    assert [dispatch.family for dispatch in dispatches] == [expected_family]
    assert not dispatches[0].selected_causal
    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, case.grads):
        assert torch.isfinite(actual).all(), name
        assert torch.linalg.vector_norm(expected).item() > 0.0, name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("case_name", ["mha_square_16k_causal", "gqa8_square_16k_causal"])
def test_d64_causal_m256_accuracy_gfx950(case_name):
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal and sq == skv == 16384
    dispatch = _select_d64_dispatch((batch, hq, sq, head_dim), (batch, hkv, skv, head_dim), causal)
    assert dispatch.family == "causal_m256"
    assert dispatch.kv_splits == (1 if hq == hkv else 4)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260807 + tuple(_D64_VALIDATION_SHAPES).index(case_name))

    def random(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16).contiguous()

    q = random((batch, hq, sq, head_dim))
    k = random((batch, hkv, skv, head_dim))
    v = random((batch, hkv, skv, head_dim))
    do = random(q.shape)
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, False, scale=sm_scale)
    o, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(do, q, k, v, o, lse, cum_q, cum_k,
                                                                                    max_q, max_k, 0.0, True, rng,
                                                                                    unused, scale=sm_scale)

    actual = fa_backward(q, k, v, o.contiguous(), do, lse.contiguous(), sm_scale, True)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, reference):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def test_d64_causal_stat_contract_formulas():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101)
    q = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    k = torch.randn((5, 4), generator=generator, dtype=torch.float32)
    v = torch.randn((5, 4), generator=generator, dtype=torch.float32)
    o = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    do = torch.randn((3, 4), generator=generator, dtype=torch.float32)
    lse = torch.randn((3, ), generator=generator, dtype=torch.float32)
    sm_scale = 0.5

    delta_mha, lse_term_mha = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_MHA_POSITIVE)
    delta_gqa, lse_term_gqa = _d64_causal_stat_values(o, do, lse, sm_scale, _D64_GQA_SIGNED)
    with pytest.raises(ValueError, match=r"^unknown D64 stat mode 2$"):
        _d64_causal_stat_values(o, do, lse, sm_scale, 2)

    delta_positive = torch.sum(o.float() * do.float(), dim=-1)
    delta_signed = -delta_positive
    lse_term = -lse.float() * math.log2(math.e)

    torch.testing.assert_close(delta_mha, delta_positive)
    torch.testing.assert_close(delta_gqa, delta_signed)
    assert lse_term_mha is None
    torch.testing.assert_close(lse_term_gqa, lse_term)
    assert delta_gqa.dtype is lse_term_gqa.dtype is torch.float32
    assert delta_gqa.is_contiguous() and lse_term_gqa.is_contiguous()

    scores_mha = q.float() @ k.float().mT
    p_mha = torch.exp2((scores_mha * sm_scale - lse.float()[..., None]) * math.log2(math.e))
    ds_mha = p_mha * (do.float() @ v.float().mT - delta_positive[..., None])

    scores_gqa = q.float() @ k.float().mT
    p_gqa = torch.exp2(scores_gqa * (sm_scale * math.log2(math.e)) + lse_term[..., None])
    ds_gqa = p_gqa * (do.float() @ v.float().mT + delta_signed[..., None])

    torch.testing.assert_close(p_mha, p_gqa)
    torch.testing.assert_close(ds_mha, ds_gqa)


def test_d64_causal_owner_interval_exhaustive():
    for sq, owner_rows in (
        (4096, 192),
        (8192, 192),
        (12288, 192),
        (16384, 256),
        (16448, 192),
    ):
        owners = triton.cdiv(sq, owner_rows)
        for invalid_owner in (-1, owners):
            with pytest.raises(
                    ValueError,
                    match=r"^physical owner is outside the dQ grid$",
            ):
                _d64_causal_owner_interval(invalid_owner, sq, owner_rows)
        covered = []
        for physical_owner in range(owners):
            actual = _d64_causal_owner_interval(physical_owner, sq, owner_rows)
            expected = (
                max(
                    (owners - 1 - physical_owner) * owner_rows - (owners * owner_rows - sq),
                    0,
                ),
                min(
                    (owners - 1 - physical_owner) * owner_rows - (owners * owner_rows - sq) + owner_rows,
                    sq,
                ),
            )
            assert actual == expected
            covered.extend(range(*actual))
        assert len(covered) == len(set(covered))
        assert sorted(covered) == list(range(sq))


def test_d64_causal_dq_grid_bijection_exhaustive():
    launch_tiles = 5
    owner_pid_base = 3
    for batch, hq, hkv in ((2, 16, 16), (2, 64, 8)):
        for use_xcd in (
                False,
                _d64_use_dq_xcd(batch, hkv, 8192, 8192, 192),
        ):
            assert use_xcd is (batch * hkv % 8 == 0) if use_xcd else True
            decoded = []
            for pid in range(batch * hq * launch_tiles):
                coords = _d64_decode_dq_pid(
                    pid,
                    batch,
                    hq,
                    hkv,
                    launch_tiles,
                    use_xcd,
                    owner_pid_base,
                )
                decoded.append(coords)
                assert _d64_encode_dq_pid(
                    *coords,
                    batch,
                    hq,
                    hkv,
                    launch_tiles,
                    use_xcd,
                    owner_pid_base,
                ) == pid
            assert len(decoded) == len(set(decoded))
            assert set(decoded) == {(batch_id, hq_id, owner_pid_base + local_owner)
                                    for batch_id in range(batch)
                                    for hq_id in range(hq)
                                    for local_owner in range(launch_tiles)}


def test_d64_causal_dq_xcd_predicate_boundaries():
    assert not _d64_use_dq_xcd(1, 7, 8192, 8192, 192)
    assert _d64_use_dq_xcd(1, 8, 8192, 8192, 192)

    assert not _d64_use_dq_xcd(1, 8, 4096, 4096, 192)
    assert _d64_use_dq_xcd(1, 8, 4096, 4096, 256)
    assert _d64_use_dq_xcd(1, 8, 4096, 4160, 192)
    assert _d64_use_dq_xcd(1, 8, 4160, 4160, 192)
    assert _d64_use_dq_xcd(1, 8, 3904, 3904, 192)
    assert _d64_use_dq_xcd(1, 8, 5248, 5248, 192)
    assert _d64_use_dq_xcd(8, 1, 4096, 4096, 192)


def test_d64_causal_m192_launch_plan(monkeypatch):
    batch, hq, hkv = 4, 64, 8
    sq = skv = 8192
    owner_rows = 192
    cu_count = 256
    owners = triton.cdiv(sq, owner_rows)
    peeled = _d64_dq_launch_plan(
        batch,
        hq,
        hkv,
        sq,
        skv,
        owner_rows,
        cu_count,
        True,
    )
    assert peeled == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    false_boundaries = (
        (batch, hq, hkv, 8000, 8000, owner_rows, cu_count, True),
        (1, 7, 7, sq, skv, owner_rows, 1, True),
        (1, hq, hkv, sq, skv, owner_rows, cu_count, True),
        (batch, hq, hkv, sq, skv, owner_rows, cu_count, False),
        (batch, hq, hkv, 8256, 8256, owner_rows, cu_count, True),
    )
    for args in false_boundaries:
        local_owners = triton.cdiv(args[3], args[5])
        assert _d64_dq_launch_plan(*args) == (_D64DQLaunch(local_owners, args[-1], 0, 0, 3, 0), )

    with monkeypatch.context() as patch:
        patch.setattr(triton, "cdiv", lambda _sq, _owner_rows: 1)
        assert _d64_dq_launch_plan(
            batch,
            hq,
            hkv,
            sq,
            skv,
            owner_rows,
            cu_count,
            True,
        ) == (_D64DQLaunch(1, True, 0, 0, 3, 0), )


def test_d64_causal_m192_launch_coverage_and_order():
    batch, hq, hkv = 4, 64, 8
    sq = skv = 8192
    owner_rows = 192
    owners = triton.cdiv(sq, owner_rows)
    launches = _d64_dq_launch_plan(batch, hq, hkv, sq, skv, owner_rows, 256, True)
    assert launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    decoded_by_launch = []
    for launch in launches:
        decoded_by_launch.append([
            _d64_decode_dq_pid(
                pid,
                batch,
                hq,
                hkv,
                launch.launch_tiles,
                True,
                launch.owner_pid_base,
            ) for pid in range(batch * hq * launch.launch_tiles)
        ])

    assert all(coords[2] < owners - 1 for coords in decoded_by_launch[0])
    assert all(coords[2] == owners - 1 for coords in decoded_by_launch[1])
    combined = decoded_by_launch[0] + decoded_by_launch[1]
    assert len(combined) == len(set(combined))
    assert set(combined) == {(batch_id, hq_id, physical_owner)
                             for batch_id in range(batch)
                             for hq_id in range(hq)
                             for physical_owner in range(owners)}
    assert _d64_causal_owner_interval(owners - 1, sq, owner_rows) == (
        0,
        128,
    )


def test_d64_causal_owner_triangular_tail_schedule_exhaustive():
    for owner_fragments in (2, 3, 4):
        for tail_step in range(owner_fragments):
            modes = _d64_causal_triangular_tail_schedule(owner_fragments, owner_fragments, tail_step)
            assert modes.count("masked") == 1
            assert modes[tail_step] == "masked"
            assert modes[:tail_step] == ("skip", ) * tail_step
            assert modes[tail_step + 1:] == ("unmasked", ) * (owner_fragments - tail_step - 1)

        for valid_fragments in range(1, owner_fragments + 1):
            for tail_step in range(valid_fragments):
                modes = _d64_causal_triangular_tail_schedule(owner_fragments, valid_fragments, tail_step)
                assert modes.count("masked") == 1
                assert modes[tail_step] == "masked"
                assert all(mode == "skip" for mode in modes[valid_fragments:])


def test_d64_structural_dispatch_locked_host_interface():
    assert tuple(field.name for field in dataclasses.fields(_D64Dispatch)) == (
        "family",
        "owner_rows",
        "key_rows",
        "kv_splits",
        "selected_causal",
        "stat_mode",
        "dq_logical_n",
        "dq_use_xcd",
        "dq_launches",
        "gqa_grid_mode",
        "cyclic_query_split",
        "dkdv_lifetime",
    )
    for value in (
            _D64_MHA_POSITIVE,
            _D64_GQA_SIGNED,
            _D64_LSE_NATURAL_LOG,
            _D64_LSE_NEG_LOG2E,
            _D64_DELTA_POSITIVE,
            _D64_DELTA_NEGATED,
    ):
        assert type(value) is int
    assert dataclasses.asdict(_D64Dispatch("retained", 192, 64, 1)) == {
        "family": "retained",
        "owner_rows": 192,
        "key_rows": 64,
        "kv_splits": 1,
        "selected_causal": False,
        "stat_mode": _D64_MHA_POSITIVE,
        "dq_logical_n": 64,
        "dq_use_xcd": False,
        "dq_launches": (),
        "gqa_grid_mode": None,
        "cyclic_query_split": False,
        "dkdv_lifetime": None,
    }


def test_d64_structural_dispatch_dq_launcher_stat_mode_contract():
    q = torch.empty((1, 1, 64, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 1, 64), device="meta", dtype=torch.float32)

    def launch(dispatch, lse_term):
        _launch_bwd_d64_causal_dq(
            q,
            q,
            q,
            q,
            q,
            stats,
            stats,
            lse_term,
            q,
            0.125,
            dispatch,
        )

    mha = _D64Dispatch("causal_scheduled_mha", 192, 64, 1, stat_mode=_D64_MHA_POSITIVE)
    with pytest.raises(ValueError) as exc_info:
        launch(mha, stats)
    assert exc_info.value.args == ("MHA positive dQ must not receive lse_term", )

    gqa = dataclasses.replace(mha, stat_mode=_D64_GQA_SIGNED)
    with pytest.raises(ValueError) as exc_info:
        launch(gqa, None)
    assert exc_info.value.args == ("GQA signed dQ requires lse_term", )

    unknown = dataclasses.replace(mha, stat_mode=7)
    with pytest.raises(ValueError) as exc_info:
        launch(unknown, None)
    assert exc_info.value.args == ("unknown dQ stat mode 7", )


def test_d64_structural_dispatch_direct_route_uses_selected_record(monkeypatch):
    q = torch.empty((1, 8, 256, 64), device="meta", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 256, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 8, 256), device="meta", dtype=torch.float32)
    dispatch = _D64Dispatch("causal_m192", owner_rows=192, key_rows=32, kv_splits=4)
    calls = []

    def forbid_reselection(*args, **kwargs):
        raise AssertionError("retained D64 route reselected dispatch")

    def preprocess(*args, **kwargs):
        calls.append(("preprocess", None))

    def launch_dq(*args, **kwargs):
        calls.append(("dq", args[-1]))

    def allocate(_k, kv_splits):
        calls.append(("allocate", kv_splits))
        return None, None

    def launch_dkdv(*args, **kwargs):
        calls.append(("dkdv", args[-1]))

    monkeypatch.setitem(globals(), "_select_d64_dispatch", forbid_reselection)
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", preprocess)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dq", launch_dq)
    monkeypatch.setitem(globals(), "_allocate_bwd_d64_kv_partials", allocate)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dkdv", launch_dkdv)

    _run_bwd_d64(
        q,
        k,
        k,
        q,
        q,
        stats,
        stats,
        q,
        k,
        k,
        0.125,
        True,
        dispatch,
    )

    assert calls == [
        ("preprocess", None),
        ("dq", dispatch),
        ("allocate", dispatch.kv_splits),
        ("dkdv", dispatch),
    ]


def test_d64_causal_gqa_grid_policy_validation_shapes():
    cu_count = 256
    expected = {
        "gqa8_square_16k_causal": (_D64_GQA_XCD, False),
        "gqa8_square_4k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_16k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_8k_causal": (_D64_GQA_XCD, False),
        "gqa8_rect_4k_12k_causal": (_D64_GQA_XCD, False),
    }
    for case_name, expected_policy in expected.items():
        batch, sq, skv, _hq, hkv, _d, causal = _D64_VALIDATION_SHAPES[case_name]
        assert causal
        assert _d64_gqa_grid_policy(batch, hkv, sq, skv, cu_count) == expected_policy

    cyclic = (8, 8, 16384, 16384, cu_count)
    assert _d64_gqa_grid_policy(*cyclic) == (_D64_GQA_XCD_N_FAST, True)

    # Negate each cyclic conjunct independently while retaining the others.
    assert _d64_gqa_grid_policy(8, 8, 8192, 16384, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(8, 8, 8192, 8192, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(1, 8, 16384, 16384, cu_count) == (
        _D64_GQA_XCD,
        False,
    )
    assert _d64_gqa_grid_policy(10, 7, 16384, 16384, cu_count) == (
        _D64_GQA_SPLIT_FAST,
        False,
    )

    boundary_cases = (
        ((1, 8, 4032, 4032, cu_count), (_D64_GQA_XCD, False)),
        ((1, 8, 4096, 4096, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 8192, 8192, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 8256, 8256, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 4096, 16256, cu_count), (_D64_GQA_XCD, False)),
        ((4, 6, 4096, 16384, cu_count), (_D64_GQA_XCD, False)),
    )
    for arguments, expected_policy in boundary_cases:
        assert _d64_gqa_grid_policy(*arguments) == expected_policy


def test_d64_causal_gqa_grid_bijection_exhaustive():
    for batch, hkv, skv in (
        (2, 8, 256),
        (2, 8, 640),
        (4, 6, 384),
        (4, 6, 512),
        (4, 6, 640),
    ):
        assert (batch * hkv) % 8 == 0
        nt = triton.cdiv(skv, 128)
        total = batch * hkv * 4 * nt

        expected_orders = {
            _D64_GQA_SPLIT_FAST: [(batch_id, hkv_id, split, n)
                                  for batch_id in range(batch)
                                  for n in range(nt)
                                  for hkv_id in range(hkv)
                                  for split in range(4)],
            _D64_GQA_XCD: [(bkv // hkv, bkv % hkv, split, n)
                           for bkv_group in range(batch * hkv // 8)
                           for n in range(nt)
                           for split in range(4)
                           for xcd in range(8)
                           for bkv in (bkv_group * 8 + xcd, )],
            _D64_GQA_XCD_N_FAST: [(bkv // hkv, bkv % hkv, split, n)
                                  for bkv_group in range(batch * hkv // 8)
                                  for split in range(4)
                                  for n in range(nt)
                                  for xcd in range(8)
                                  for bkv in (bkv_group * 8 + xcd, )],
        }
        expected_coords = {(batch_id, hkv_id, split, n)
                           for batch_id in range(batch)
                           for hkv_id in range(hkv)
                           for split in range(4)
                           for n in range(nt)}
        assert len(expected_coords) == total

        for grid_mode, expected_order in expected_orders.items():
            decoded = [_d64_decode_gqa_pid(pid, batch, hkv, skv, grid_mode) for pid in range(total)]
            assert decoded == expected_order
            assert len(decoded) == len(set(decoded)) == total
            assert set(decoded) == expected_coords


def test_d64_causal_gqa_frontier_exhaustive():
    block_m, block_n = 64, 128
    for sq, skv in (
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (16384, 16384),
    ):
        diff = skv - sq
        m_blocks = triton.cdiv(sq, block_m)
        for n0 in range(0, skv, block_n):
            start_m_blk, masked = _d64_causal_physical_frontier(n0, sq, skv, block_m, block_n)
            expected_start = max((n0 - diff) // block_m, 0)
            expected_masked = tuple(m_blk for m_blk in range(expected_start, m_blocks)
                                    if n0 + block_n - 1 > m_blk * block_m + diff)
            assert start_m_blk == expected_start
            assert masked == expected_masked

            pair_start = (start_m_blk // 2) * 2
            batch_start = pair_start + (pair_start % 4)
            assert all(m_blk < batch_start + 2 for m_blk in masked if m_blk >= batch_start)

            # Every omitted block is wholly invalid, and every scheduled block
            # contains at least one valid (m, n) satisfying bottom-right causal.
            for m_blk in range(m_blocks):
                m0 = m_blk * block_m
                m_last = min(sq, m0 + block_m) - 1
                scheduled = m_blk >= start_m_blk
                assert scheduled is (n0 <= m_last + diff)
                tile_needs_mask = n0 + block_n - 1 > m0 + diff
                assert (m_blk in masked) is (scheduled and tile_needs_mask)
                if scheduled and not tile_needs_mask:
                    assert all(0 <= m < sq and 0 <= n < skv and n <= m + skv - sq
                               for m in range(m0, min(m0 + block_m, sq))
                               for n in range(n0, min(n0 + block_n, skv)))

        aligned_n0 = diff + block_n
        start_m_blk, masked = _d64_causal_physical_frontier(aligned_n0, sq, skv, block_m, block_n)
        assert start_m_blk > 0
        assert masked == (start_m_blk, start_m_blk + 1)

        if diff:
            zero_clamp_n0 = diff - block_n
            start_m_blk, masked = _d64_causal_physical_frontier(zero_clamp_n0, sq, skv, block_m, block_n)
            assert start_m_blk == 0
            assert masked == tuple(m_blk for m_blk in range(m_blocks)
                                   if zero_clamp_n0 + block_n - 1 > m_blk * block_m + diff)
            assert masked == ()


def test_d64_causal_dq_bulk_successor_in_range():
    for sq, skv in (
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (16384, 16384),
    ):
        for owner_rows in (192, 256):
            num_owners = triton.cdiv(sq, owner_rows)
            pad = num_owners * owner_rows - sq
            for physical_owner in range(num_owners):
                reverse_owner = num_owners - 1 - physical_owner
                raw_start = reverse_owner * owner_rows
                owner_start = max(raw_start - pad, 0)
                owner_end = raw_start + owner_rows - pad
                bulk_end_block = (owner_start + (skv - sq)) // 64
                end_n_block = min((owner_end - 1 + (skv - sq)) // 64 + 1, skv // 64)
                assert bulk_end_block < end_n_block


def test_d64_causal_mha_frontier_exhaustive():
    block_m, block_n = 32, 64
    for sq, skv in (
        (4096, 4096),
        (16384, 16384),
        (4096, 8192),
    ):
        diff = skv - sq
        m_blocks = triton.cdiv(sq, block_m)
        for n0 in range(0, skv, block_n):
            start_m_blk, masked = _d64_causal_physical_frontier(n0, sq, skv, block_m, block_n)
            expected_start = max((n0 - diff) // block_m, 0)
            expected_masked = tuple(m_blk for m_blk in range(expected_start, m_blocks)
                                    if n0 + block_n - 1 > m_blk * block_m + diff)
            assert start_m_blk == expected_start
            assert masked == expected_masked

            # Exhaust every physical BM32 block against bottom-right validity.
            for m_blk in range(m_blocks):
                m0 = m_blk * block_m
                m_last = min(sq, m0 + block_m) - 1
                any_valid = n0 <= m_last + diff
                all_valid = n0 + block_n - 1 <= m0 + diff
                scheduled = m_blk >= start_m_blk
                assert scheduled is any_valid
                assert (m_blk in masked) is (scheduled and not all_valid)

        aligned_n0 = diff + block_n
        start_m_blk, masked = _d64_causal_physical_frontier(aligned_n0, sq, skv, block_m, block_n)
        assert start_m_blk > 0
        assert masked == (start_m_blk, start_m_blk + 1)

        if diff:
            zero_clamp_n0 = diff - block_n
            start_m_blk, masked = _d64_causal_physical_frontier(zero_clamp_n0, sq, skv, block_m, block_n)
            assert start_m_blk == 0
            assert masked == tuple(m_blk for m_blk in range(m_blocks)
                                   if zero_clamp_n0 + block_n - 1 > m_blk * block_m + diff)
            assert masked == ()


def test_d64_causal_gqa_split_ownership():
    query_blocks = 13
    expected = {(head, m_blk) for head in range(8) for m_blk in range(query_blocks)}
    for cyclic in (False, True):
        by_split = [_d64_gqa_split_ownership(split, query_blocks, cyclic) for split in range(4)]
        for split, owned in enumerate(by_split):
            if cyclic:
                assert {head for head, _m_blk in owned} == set(range(8))
                assert all(m_blk % 4 == split for _head, m_blk in owned)
            else:
                assert {head
                        for head, _m_blk in owned} == {
                            2 * split,
                            2 * split + 1,
                        }
                assert all(0 <= m_blk < query_blocks for _head, m_blk in owned)
        flattened = [item for owned in by_split for item in owned]
        assert len(flattened) == len(set(flattened)) == len(expected)
        assert set(flattened) == expected


def test_d64_causal_gqa_lifetime_policy():
    for sq in (2048, 4096, 8192):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64
    for sq, skv in (
        (1024, 1152),
        (12288, 16384),
    ):
        assert _d64_gqa_lifetime(sq, skv) == _D64_GQA_INTERLEAVED_D32
    for sq, skv in (
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
    ):
        assert _d64_gqa_lifetime(sq, skv) == _D64_GQA_DIRECT_D64
    for sq in (1024, 12288, 16384):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64

    for sq in range(1024, 16385, 1024):
        assert _d64_gqa_lifetime(sq, sq) == _D64_GQA_DIRECT_D64
        assert _d64_gqa_lifetime(sq, sq + 128) == _D64_GQA_INTERLEAVED_D32
        assert _d64_gqa_lifetime(sq, 2 * sq - 128) == _D64_GQA_INTERLEAVED_D32
        assert _d64_gqa_lifetime(sq, 2 * sq) == _D64_GQA_DIRECT_D64
    with pytest.raises(ValueError, match="selected GQA8 shape has no lifetime mode"):
        _d64_gqa_lifetime(960, 960)


def test_d64_causal_gqa_d32_schedule_peels_odd_frontier_before_even_pairs():
    source = _d64_gqa8_d32_impl.src
    assert "pair_start = (start_m_blk // 2) * 2" not in source
    assert "peel_frontier = (start_m_blk % 2) != 0" in source
    assert "pair_start = start_m_blk + (start_m_blk % 2)" in source

    impl = ast.parse(source).body[0]
    scalar_assignments = {
        statement.targets[0].id: statement.value
        for statement in impl.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }

    def evaluate(name, start_m_blk):
        expression = ast.Expression(scalar_assignments[name])
        return eval(
            compile(ast.fix_missing_locations(expression), "<d32-jit>", "eval"),
            {},
            {"start_m_blk": start_m_blk},
        )

    def scheduled_groups(start_m_blk, num_m_blocks):
        groups = []
        if evaluate("peel_frontier", start_m_blk) and start_m_blk < num_m_blocks:
            groups.append((start_m_blk, ))
        pair_start = evaluate("pair_start", start_m_blk)
        for m_pair in range(pair_start // 2, num_m_blocks // 2):
            m_blk_a = m_pair * 2
            groups.append((m_blk_a, m_blk_a + 1))
        if num_m_blocks % 2 and pair_start < num_m_blocks:
            groups.append((num_m_blocks - 1, ))
        return tuple(groups)

    for num_m_blocks in (1, 2, 7, 17, 64):
        for start_m_blk in range(num_m_blocks + 1):
            groups = scheduled_groups(start_m_blk, num_m_blocks)
            scheduled = tuple(block for group in groups for block in group)
            assert scheduled == tuple(range(start_m_blk, num_m_blocks))
            if scheduled:
                assert scheduled[0] == start_m_blk
                assert len(scheduled) == len(set(scheduled))
            expected_groups = []
            next_m_blk = start_m_blk
            if next_m_blk % 2 and next_m_blk < num_m_blocks:
                expected_groups.append((next_m_blk, ))
                next_m_blk += 1
            while next_m_blk + 1 < num_m_blocks:
                expected_groups.append((next_m_blk, next_m_blk + 1))
                next_m_blk += 2
            if next_m_blk < num_m_blocks:
                expected_groups.append((next_m_blk, ))
            assert groups == tuple(expected_groups)
            assert all(len(group) == 1 or (len(group) == 2 and group[0] % 2 == 0) for group in groups)

    sq, skv, n0 = 1088, 1152, 128
    start_m_blk, _masked = _d64_causal_physical_frontier(n0, sq, skv, 64, 128)
    assert start_m_blk == 1
    assert scheduled_groups(start_m_blk, sq // 64) == (
        (1, ),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, ),
    )


def test_d64_causal_gqa8_helper_ast_contract():

    def function_ast(jit_function):
        tree = ast.parse(jit_function.src)
        assert len(tree.body) == 1
        return tree.body[0]

    def dotted_name(call):
        value = call.func
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    def root_call_name(statement):
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            value = statement.value
            if isinstance(value, ast.Call):
                return dotted_name(value)
        return None

    interesting = {
        "_d64_gqa8_issue_stage",
        "_d64_gqa8_d32_consume",
        "tlx.async_load_wait_group",
        "tl.debug_barrier",
    }

    def events(statements):
        return tuple(event for statement in statements if (event := root_call_name(statement)) in interesting)

    issue = function_ast(_d64_gqa8_issue_stage)
    load_tokens = []
    for statement in issue.body:
        if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and dotted_name(statement.value) == "tlx.buffer_load_to_local"):
            load_tokens.append(statement.targets[0].id)
    assert load_tokens == [
        "q_token",
        "do_token",
        "lse_token",
        "delta_token",
    ]
    commits = [
        node for node in ast.walk(issue)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.async_load_commit_group"
    ]
    assert len(commits) == 1
    assert ast.unparse(commits[0].args[0]) == ("[q_token, do_token, lse_token, delta_token]")

    def assert_runtime_causal_mask(jit_function, step_name, mask_arg):
        consume = function_ast(jit_function)
        mask_assignment = next(node for node in consume.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "apply_causal_mask")
        assert ast.unparse(mask_assignment.value) == ("n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)")
        steps = [node for node in ast.walk(consume) if isinstance(node, ast.Call) and dotted_name(node) == step_name]
        assert len(steps) == 1
        assert ast.unparse(steps[0].args[mask_arg]) == "apply_causal_mask"
        assert not any(isinstance(node, ast.If) for node in consume.body)

    assert_runtime_causal_mask(_d64_gqa8_direct_consume, "_d64_gqa8_direct_d64_step", 17)
    assert_runtime_causal_mask(_d64_gqa8_d32_consume, "_d64_gqa8_d32_step", 19)

    signed_front = function_ast(_d64_gqa8_signed_front)
    vgpr_handoffs = [
        node for node in ast.walk(signed_front)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
    ]
    assert len(vgpr_handoffs) == 1
    assert ast.unparse(vgpr_handoffs[0].args[0]) == "ds"
    assert ast.literal_eval(
        next(keyword.value for keyword in vgpr_handoffs[0].keywords if keyword.arg == "register_class")) == "vgpr"
    assert ast.unparse(
        next(keyword.value
             for keyword in vgpr_handoffs[0].keywords
             if keyword.arg == "registers_per_group")) == "handoff_group"
    handoff_assignment = next(
        node for node in signed_front.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "handoff_group")
    assert ast.unparse(handoff_assignment.annotation) == "tl.constexpr"
    assert ast.unparse(handoff_assignment.value) == "2 if LATE_DO_T else 1"

    def assignment_call(statement, target, call_name):
        return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == target
                and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == call_name)

    early_do_index = next(index for index, statement in enumerate(signed_front.body)
                          if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not LATE_DO_T")
    p_index = next(index for index, statement in enumerate(signed_front.body)
                   if assignment_call(statement, "p", "tlx.require_layout"))
    late_do_index = next(index for index, statement in enumerate(signed_front.body)
                         if isinstance(statement, ast.If) and ast.unparse(statement.test) == "LATE_DO_T")
    dp_dot_index = next(index for index, statement in enumerate(signed_front.body)
                        if assignment_call(statement, "dp", "tl.dot"))
    assert early_do_index < p_index < late_do_index < dp_dot_index
    for index in (early_do_index, late_do_index):
        assert any(assignment_call(statement, "do_t", "tlx.local_load") for statement in signed_front.body[index].body)

    for step_function, expected_late_do in (
        (_d64_gqa8_direct_d64_step, "True"),
        (_d64_gqa8_d32_step, "False"),
    ):
        signed_call = next(node for node in ast.walk(function_ast(step_function))
                           if isinstance(node, ast.Call) and dotted_name(node) == "_d64_gqa8_signed_front")
        assert ast.unparse(signed_call.args[16]) == expected_late_do

    step = function_ast(_d64_gqa8_d32_step)
    lifetime_ifs = [
        node for node in step.body if isinstance(node, ast.If) and ast.unparse(node.test) == "INTERLEAVED_D32"
    ]
    assert len(lifetime_ifs) == 1

    def dot_targets(statements):
        return [
            statement.targets[0].id for statement in statements if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tl.dot"
        ]

    lifetime_if = lifetime_ifs[0]
    assert dot_targets(lifetime_if.body) == [
        "dv_d0",
        "dk_d0",
        "dv_d1",
        "dk_d1",
    ]
    assert dot_targets(lifetime_if.orelse) == [
        "dv_d0",
        "dv_d1",
        "dk_d0",
        "dk_d1",
    ]

    for loop_impl, expected_iters in (
        (_d64_gqa8_direct_d64_impl, {"range(0, 8)", "tl.static_range(0, 2)"}),
        (_d64_gqa8_d32_impl, {"tl.static_range(0, 2)"}),
    ):
        local_head_loops = [
            node for node in ast.walk(function_ast(loop_impl))
            if isinstance(node, ast.For) and ast.unparse(node.target) == "local_head"
        ]
        assert {ast.unparse(loop.iter) for loop in local_head_loops} == expected_iters

    kernel_impl = function_ast(_attn_bwd_dkdv_d64_causal_gqa8_kernel)
    qdo_stages = next(
        node for node in kernel_impl.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "qdo_stages")
    assert ast.unparse(qdo_stages.annotation) == "tl.constexpr"
    assert ast.unparse(qdo_stages.value) == "4 if BATCH_STATS4 else 2"

    stats4_impl = function_ast(_d64_gqa8_async_stats4_direct_d64_impl)
    stats4_quad_loop = next(node for node in ast.walk(stats4_impl)
                            if isinstance(node, ast.For) and ast.unparse(node.target) == "m_quad")
    assert ast.unparse(stats4_quad_loop.iter) == "range(first_quad, num_m_blocks // 4)"
    stats4_step_loop = next(node for node in ast.walk(stats4_quad_loop)
                            if isinstance(node, ast.For) and ast.unparse(node.target) == "step")
    stats4_assignments = {
        statement.targets[0].id: ast.unparse(statement.value)
        for statement in stats4_step_loop.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    assert stats4_assignments["current_slot"] == "step"
    assert stats4_assignments["apply_causal_mask"] == ("n0 + BLOCK_N - 1 > "
                                                       "(m0 + step) * BLOCK_M + (SKV - SQ)")
    next_slot_if = next(statement for statement in stats4_step_loop.body
                        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "step + 1 < 4")
    assert any(
        isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "next_slot"
        and ast.unparse(statement.value) == "step + 1" for statement in next_slot_if.body)
    barrier_guard = next(statement for statement in stats4_step_loop.body
                         if isinstance(statement, ast.If) and ast.unparse(statement.test) == "step % 2 == 1")
    assert events(barrier_guard.body) == ("tl.debug_barrier", )

    impl = function_ast(_d64_gqa8_d32_impl)
    head_loop = next(node for node in impl.body
                     if isinstance(node, ast.For) and ast.unparse(node.target) == "local_head")
    peel_if = next(node for node in head_loop.body
                   if isinstance(node, ast.If) and ast.unparse(node.test) == "peel_frontier")
    assert events(peel_if.body) == (
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
        "tl.debug_barrier",
    )
    pair_guard = next(node for node in head_loop.body
                      if isinstance(node, ast.If) and ast.unparse(node.test) == "pair_start < num_m_blocks")
    assert events(pair_guard.body[:1]) == ("_d64_gqa8_issue_stage", )
    pair_loop = next(node for node in pair_guard.body if isinstance(node, ast.For))
    assert ast.unparse(pair_loop.iter) == "range(pair_start // 2, full_pairs)"
    pair_assignments = {
        statement.targets[0].id: ast.unparse(statement.value)
        for statement in pair_loop.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    assert pair_assignments["m_blk_a"] == "m_pair * 2"
    assert pair_assignments["m_blk_b"] == "m_blk_a + 1"
    assert events(pair_loop.body) == (
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
        "_d64_gqa8_d32_consume",
    )
    following_if = next(node for node in pair_loop.body
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "has_following")
    assert events(following_if.body) == (
        "tl.debug_barrier",
        "_d64_gqa8_issue_stage",
        "tlx.async_load_wait_group",
    )
    assert events(following_if.orelse) == ("tlx.async_load_wait_group", )
    tail_if = next(node for node in pair_guard.body if isinstance(node, ast.If) and ast.unparse(node.test) == "has_odd")
    assert events(tail_if.body)[:2] == (
        "tlx.async_load_wait_group",
        "_d64_gqa8_d32_consume",
    )


def test_d64_causal_gqa8_launch_mode_kwargs_same_shape(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    recorder = LaunchRecorder()
    monkeypatch.setitem(globals(), "_attn_bwd_dkdv_d64_causal_gqa8_kernel", recorder)
    q = torch.empty((1, 8, 1024, 64), device="meta", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 1024, 64), device="meta", dtype=torch.bfloat16)
    stats = torch.empty((1, 8, 1024), device="meta", dtype=torch.float32)
    partial = torch.empty((1, 1, 4, 1024, 64), device="meta", dtype=torch.bfloat16)
    expected = (
        (
            _D64_GQA_SPLIT_FAST,
            False,
            False,
            False,
            _D64_GQA_INDEPENDENT_D32,
        ),
        (_D64_GQA_XCD, False, True, False, _D64_GQA_INDEPENDENT_D32),
        (
            _D64_GQA_XCD_N_FAST,
            False,
            True,
            True,
            _D64_GQA_INDEPENDENT_D32,
        ),
        (_D64_GQA_XCD_N_FAST, True, True, True, _D64_GQA_DIRECT_D64),
    )
    for grid_mode, cyclic, _use_xcd, _use_n_fast, lifetime in expected:
        dispatch = _D64Dispatch(
            "causal_scheduled_gqa8",
            owner_rows=192,
            key_rows=128,
            kv_splits=4,
            selected_causal=True,
            stat_mode=_D64_GQA_SIGNED,
            gqa_grid_mode=grid_mode,
            cyclic_query_split=cyclic,
            dkdv_lifetime=lifetime,
        )
        _launch_bwd_d64_causal_gqa8_dkdv(q, k, k, q, stats, stats, partial, partial, 0.125, dispatch)

    assert len(recorder.calls) == len(expected)
    for call, (grid_mode, cyclic, use_xcd, use_n_fast, lifetime) in zip(recorder.calls, expected):
        grid, _args, kwargs = call
        assert grid == (32, )
        assert kwargs["USE_GQA_XCD"] is use_xcd
        assert kwargs["USE_XCD_N_FAST"] is use_n_fast
        assert kwargs["CYCLIC_QUERY_SPLIT"] is cyclic
        assert kwargs["LIFETIME_MODE"] == lifetime
        assert kwargs["LSE_MODE"] == _D64_LSE_NEG_LOG2E
        assert kwargs["DELTA_MODE"] == _D64_DELTA_NEGATED


def test_d64_causal_gqa_workspace_contract(monkeypatch):
    batch, hq, hkv, sq, skv, head_dim = 4, 48, 6, 4096, 16384, 64
    q = torch.empty((batch, hq, sq, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="meta", dtype=torch.bfloat16)
    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(q, k)
    assert tuple(lse_term.shape) == (batch, hq, sq)
    assert lse_term.dtype == torch.float32
    assert lse_term.is_contiguous()
    partial_shape = (batch, hkv, 4, skv, head_dim)
    for partial in (dk_part, dv_part):
        assert tuple(partial.shape) == partial_shape
        assert partial.dtype == torch.bfloat16
        assert partial.is_contiguous()

    def forbidden_allocator(*_args, **_kwargs):
        raise AssertionError("MHA and generic routes must not allocate GQA workspaces")

    monkeypatch.setitem(
        globals(),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        forbidden_allocator,
    )
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_dq", lambda *_args: None)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_mha_dkdv", lambda *_args: None)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dkdv", lambda *_args: None)
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(globals(), "_run_bwd_d64_direct", lambda *_args: None)

    def exercise_without_gqa_workspace(q_shape, k_shape, dispatch):
        q_meta = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
        k_meta = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
        stats = torch.empty(q_shape[:-1], device="meta", dtype=torch.float32)
        _run_bwd_d64(
            q_meta,
            k_meta,
            k_meta,
            q_meta,
            q_meta,
            stats,
            stats,
            q_meta,
            k_meta,
            k_meta,
            0.125,
            True,
            dispatch,
        )

    mha_shape = (4, 64, 4096, 64)
    mha = _select_d64_dispatch(
        mha_shape,
        mha_shape,
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert mha.family == "causal_scheduled_mha"
    exercise_without_gqa_workspace(mha_shape, mha_shape, mha)

    generic_q = (1, 8, 1024, 64)
    generic_k = (1, 1, 1024, 64)
    generic = _select_d64_dispatch(generic_q, generic_k, True)
    assert not generic.selected_causal
    exercise_without_gqa_workspace(generic_q, generic_k, generic)


def test_d64_causal_mha_workspace_and_launch_contract(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    batch, heads, sq, skv, head_dim = 2, 32, 16384, 16384, 64
    q = torch.empty(
        (batch, heads, sq, head_dim),
        device="meta",
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (batch, heads, skv, head_dim),
        device="meta",
        dtype=torch.bfloat16,
    )
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty((batch, heads, sq), device="meta", dtype=torch.float32)
    delta = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    assert dispatch.kv_splits == 1

    launches = []
    real_launcher = globals().get("_launch_bwd_d64_causal_mha_dkdv")

    def record_dq(*args):
        launches.append(("dq", args[7], args[6], args[8], args[-1]))

    def record_mha(*args):
        launches.append((
            "mha",
            args[4],
            args[5],
            args[6],
            args[7],
            args[-1],
        ))

    def reject_forbidden(*_args, **_kwargs):
        raise AssertionError("selected causal MHA must not preprocess, allocate partials, "
                             "reduce, convert, publish atomically, or use the retained producer")

    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_dq", record_dq)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_mha_dkdv", record_mha)
    for name in (
            "_run_bwd_preprocess",
            "_allocate_bwd_d64_causal_gqa8_workspaces",
            "_allocate_bwd_d64_kv_partials",
            "_launch_bwd_d64_causal_gqa8_dkdv",
            "_launch_bwd_d64_causal_gqa8_reduce",
            "_launch_bwd_d64_kv_reduce",
            "_launch_bwd_d64_fused_dq_convert",
            "_launch_bwd_d64_dkdv",
    ):
        monkeypatch.setitem(globals(), name, reject_forbidden)

    _run_bwd_d64(
        q,
        k,
        v,
        o,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        0.125,
        True,
        dispatch,
    )

    assert [launch[0] for launch in launches] == ["dq", "mha"]
    dq_launch, mha_launch = launches
    assert dq_launch[1] is None
    assert dq_launch[2] is delta is mha_launch[2]
    assert dq_launch[3] is dq
    assert mha_launch[1] is lse
    assert mha_launch[3] is dk
    assert mha_launch[4] is dv
    assert dq_launch[4] is mha_launch[5] is dispatch

    # Exercise the real launcher separately so this test covers both the
    # _run_bwd_d64 policy and the kernel launch ABI, not just a wished-for mock.
    assert callable(real_launcher)
    recorder = LaunchRecorder()
    monkeypatch.setitem(globals(), "_attn_bwd_dkdv_d64_causal_mha_kernel", recorder)
    real_launcher(q, k, v, do, lse, delta, dk, dv, 0.125, dispatch)

    assert len(recorder.calls) == 1
    grid, args, kwargs = recorder.calls[0]
    assert grid == (batch * heads * triton.cdiv(skv, 64), )
    assert args == (q, k, v, do, lse, delta, dk, dv)
    assert kwargs["SM_SCALE"] == 0.125
    assert kwargs["HQ"] == kwargs["HKV"] == heads
    assert kwargs["SQ"] == sq
    assert kwargs["SKV"] == skv
    assert kwargs["D"] == head_dim
    assert kwargs["BLOCK_M"] == 32
    assert kwargs["BLOCK_N"] == 64
    assert kwargs["LSE_MODE"] == _D64_LSE_NATURAL_LOG
    assert kwargs["DELTA_MODE"] == _D64_DELTA_POSITIVE
    assert kwargs["num_warps"] == 2
    assert kwargs["matrix_instr_nonkdim"] == _matrix_instr_nonkdim()


def test_d64_causal_selected_dispatch_contract():
    arch = "gfx950:sramecc+:xnack-"
    cu_count = 256

    def is_eligible(q_shape, k_shape, **overrides):
        return _is_d64_scheduled_causal_eligible(
            q_shape,
            k_shape,
            True,
            arch=overrides.get("arch", arch),
            cu_count=overrides.get("cu_count", cu_count),
            sm_scale=overrides.get("sm_scale", 0.125),
            bases_aligned_16=overrides.get("bases_aligned_16", True),
        )

    def select(q_shape, k_shape):
        return _select_d64_dispatch(
            q_shape,
            k_shape,
            True,
            arch=arch,
            cu_count=cu_count,
            sm_scale=0.125,
            bases_aligned_16=True,
        )

    mha_m192 = ((4, 64, 4096, 64), (4, 64, 4096, 64))
    mha_m256 = ((2, 32, 16384, 64), (2, 32, 16384, 64))
    for q_shape, k_shape in (mha_m192, mha_m256):
        assert is_eligible(q_shape, k_shape)
        dispatch = select(q_shape, k_shape)
        assert dispatch.owner_rows == _d64_selected_causal_owner_rows(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        assert dispatch.family == "causal_scheduled_mha"
        assert dispatch.selected_causal
        assert dispatch.stat_mode == _D64_MHA_POSITIVE
        assert dispatch.key_rows == 64
        assert dispatch.dq_logical_n == 32
        assert dispatch.kv_splits == 1
        assert dispatch.dq_use_xcd is _d64_use_dq_xcd(q_shape[0], k_shape[1], q_shape[2], k_shape[2],
                                                      dispatch.owner_rows)
        assert dispatch.dq_launches == _d64_dq_launch_plan(
            q_shape[0],
            q_shape[1],
            k_shape[1],
            q_shape[2],
            k_shape[2],
            dispatch.owner_rows,
            cu_count,
            True,
        )
    assert select(*mha_m192).owner_rows == 192
    assert select(*mha_m256).owner_rows == 256

    peeled = select((4, 64, 8192, 64), (4, 64, 8192, 64))
    owners = triton.cdiv(8192, 192)
    assert peeled.dq_use_xcd
    assert peeled.dq_launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )

    gqa_cases = [
        ((2, 128, 1024, 64), (2, 16, 1024, 64)),
        *[(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        )
          for name in _D64_CAUSAL_GQA8_VALIDATION_CASES
          for batch, sq, skv, hq, hkv, head_dim, causal in (_D64_VALIDATION_SHAPES[name], )
          if causal],
    ]
    for q_shape, k_shape in gqa_cases:
        assert q_shape[1] == 8 * k_shape[1]
        assert is_eligible(q_shape, k_shape)
        dispatch = select(q_shape, k_shape)
        assert dispatch.owner_rows == _d64_selected_causal_owner_rows(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        expected_grid, expected_cyclic = _d64_gqa_grid_policy(q_shape[0], k_shape[1], q_shape[2], k_shape[2], cu_count)
        assert dispatch.family == "causal_scheduled_gqa8"
        assert dispatch.selected_causal
        assert dispatch.stat_mode == _D64_GQA_SIGNED
        assert dispatch.key_rows == 128
        assert dispatch.kv_splits == 4
        assert dispatch.gqa_grid_mode == expected_grid
        assert dispatch.cyclic_query_split is expected_cyclic
        assert dispatch.dkdv_lifetime == _d64_gqa_lifetime(q_shape[2], k_shape[2])
        assert dispatch.dq_logical_n == _d64_selected_causal_logical_n(
            q_shape[2],
            k_shape[2],
            q_shape[1] // k_shape[1],
        )
        assert dispatch.dq_use_xcd is _d64_use_dq_xcd(q_shape[0], k_shape[1], q_shape[2], k_shape[2],
                                                      dispatch.owner_rows)
        assert dispatch.dq_launches == _d64_dq_launch_plan(
            q_shape[0],
            q_shape[1],
            k_shape[1],
            q_shape[2],
            k_shape[2],
            dispatch.owner_rows,
            cu_count,
            True,
        )
    assert [
        select(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        ).owner_rows
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [256, 256, 256, 256, 256]
    assert [
        select(
            (batch, hq, sq, head_dim),
            (batch, hkv, skv, head_dim),
        ).dkdv_lifetime
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
        _D64_GQA_DIRECT_D64,
    ]
    assert [
        _d64_causal_gqa8_batch_stats4(
            sq,
            skv,
            select(
                (batch, hq, sq, head_dim),
                (batch, hkv, skv, head_dim),
            ),
        )
        for case_name in _D64_CAUSAL_GQA8_VALIDATION_CASES
        for batch, sq, skv, hq, hkv, head_dim, _causal in (_D64_VALIDATION_SHAPES[case_name], )
    ] == [True, True, True, True, True]
    negative_cases = (
        (
            (4, 64, 4096, 64),
            (4, 64, 4096, 64),
            {"arch": "gfx942"},
        ),
        (
            (4, 64, 4096, 64),
            (4, 64, 4096, 64),
            {"bases_aligned_16": False},
        ),
        ((4, 64, 4160, 64), (4, 64, 4096, 64), {}),
        ((4, 64, 4096, 64), (4, 32, 4096, 64), {}),
        ((4, 64, 4096, 64), (4, 16, 4096, 64), {}),
        ((4, 64, 4032, 64), (4, 64, 4032, 64), {}),
        ((2, 128, 960, 64), (2, 16, 960, 64), {}),
        ((2, 128, 1024, 64), (2, 16, 1088, 64), {}),
        ((4, 64, 4097, 64), (4, 64, 4097, 64), {}),
        ((1, 8, 1024, 64), (1, 1, 16384, 64), {}),
        ((1, 8, 12288, 64), (1, 1, 12288, 64), {}),
    )
    for q_shape, k_shape, overrides in negative_cases:
        assert not is_eligible(q_shape, k_shape, **overrides)

    m192_neighbor = select((2, 32, 16320, 64), (2, 32, 16320, 64))
    m192_rectangular = select((2, 32, 16384, 64), (2, 32, 16448, 64))
    assert m192_neighbor.owner_rows == 192
    assert m192_neighbor.dq_logical_n == 32
    assert m192_rectangular.owner_rows == 192
    assert m192_rectangular.dq_logical_n == 64


def test_d64_causal_dispatch_allocation_order(monkeypatch):
    q_shape = (1, 8, 256, 64)
    k_shape = (1, 1, 256, 64)
    q = torch.empty(q_shape, dtype=torch.bfloat16)
    k = torch.empty(k_shape, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty(q_shape[:-1], dtype=torch.float32)
    original_empty = torch.empty
    original_empty_like = torch.empty_like
    calls = []
    output_allocations = 0
    stat_allocations = 0
    active_dispatch = None

    def make_meta_inputs(test_q_shape, test_k_shape):
        test_q = torch.empty(test_q_shape, device="meta", dtype=torch.bfloat16)
        test_k = torch.empty(test_k_shape, device="meta", dtype=torch.bfloat16)
        return (
            test_q,
            test_k,
            torch.empty_like(test_k),
            torch.empty_like(test_q),
            torch.empty_like(test_q),
            torch.empty(test_q_shape[:-1], device="meta", dtype=torch.float32),
        )

    selected_mha_q_shape = (4, 64, 4096, 64)
    selected_mha_k_shape = (4, 64, 4096, 64)
    selected_mha_inputs = make_meta_inputs(selected_mha_q_shape, selected_mha_k_shape)
    selected_mha = _select_d64_dispatch(
        selected_mha_q_shape,
        selected_mha_k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    selected_gqa_q_shape = (8, 16, 1024, 64)
    selected_gqa_k_shape = (8, 2, 1024, 64)
    selected_gqa_inputs = make_meta_inputs(selected_gqa_q_shape, selected_gqa_k_shape)
    selected_gqa = _select_d64_dispatch(
        selected_gqa_q_shape,
        selected_gqa_k_shape,
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    retained = _select_d64_dispatch(q_shape, k_shape, True)

    def record_validate(*_args):
        calls.append("validate")

    def record_dispatch(*args):
        calls.append("dispatch")
        assert args[7] is True
        return active_dispatch

    def record_empty_like(tensor, *args, **kwargs):
        nonlocal output_allocations
        if tensor.ndim == 4:
            output_allocations += 1
            if output_allocations == 1:
                calls.append("outputs")
        return original_empty_like(tensor, *args, **kwargs)

    def record_empty(*args, **kwargs):
        nonlocal stat_allocations
        shape = tuple(args[0]) if args else tuple(kwargs["size"])
        if len(shape) == 3:
            stat_allocations += 1
            calls.append("delta" if stat_allocations == 1 else "lse_term")
        elif len(shape) == 5:
            calls.append("partials")
        return original_empty(*args, **kwargs)

    monkeypatch.setitem(globals(), "_validate_inputs", record_validate)
    monkeypatch.setitem(globals(), "_select_d64_dispatch_for_device", record_dispatch)
    monkeypatch.setattr(torch, "empty_like", record_empty_like)
    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setitem(
        globals(),
        "_launch_bwd_d64_causal_dq",
        lambda *_args: calls.append("dq"),
    )
    monkeypatch.setitem(
        globals(),
        "_launch_bwd_d64_causal_mha_dkdv",
        lambda *_args: calls.append("mha_producer"),
    )
    monkeypatch.setitem(
        globals(),
        "_launch_bwd_d64_causal_gqa8_dkdv",
        lambda *_args: calls.append("gqa_producer"),
    )
    monkeypatch.setitem(
        globals(),
        "_launch_bwd_d64_causal_gqa8_reduce",
        lambda *_args: calls.append("reducer"),
    )
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", lambda *_args: None)
    monkeypatch.setitem(
        globals(),
        "_run_bwd_d64_direct",
        lambda *_args: calls.append("retained_generic"),
    )

    def exercise(inputs, dispatch, scale):
        nonlocal active_dispatch, output_allocations, stat_allocations
        active_dispatch = dispatch
        output_allocations = 0
        stat_allocations = 0
        calls.clear()
        result = fa_backward(*inputs, scale, True)
        assert len(result) == 3
        return tuple(calls)

    assert exercise(selected_mha_inputs, selected_mha, 0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "dq",
        "mha_producer",
    )
    assert exercise(selected_gqa_inputs, selected_gqa, 0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "lse_term",
        "partials",
        "dq",
        "gqa_producer",
        "reducer",
    )
    assert exercise((q, k, v, o, do, lse), retained, -0.125) == (
        "validate",
        "dispatch",
        "outputs",
        "delta",
        "retained_generic",
    )

    input_sentinels = tuple(tensor.view(torch.uint8).clone() for tensor in (q, k, v, o, do, lse))
    for invalid in (0.0, -0.0, float("nan"), float("inf"), -float("inf")):
        calls.clear()
        with pytest.raises(ValueError, match="^D64 sm_scale must be finite and nonzero$"):
            fa_backward(q, k, v, o, do, lse, invalid, True)
        assert calls == ["validate"]
        for tensor, sentinel in zip((q, k, v, o, do, lse), input_sentinels, strict=True):
            assert torch.equal(tensor.view(torch.uint8), sentinel)


def test_d64_causal_scale_classification_before_writes(monkeypatch):
    shape = (4, 64, 4096, 64)
    q = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty(shape[:-1], device="meta", dtype=torch.float32)

    class Properties:
        gcnArchName = "gfx950:sramecc+:xnack-"
        multi_processor_count = 256

    calls = []

    def record_empty_like(*_args, **_kwargs):
        calls.append("empty_like")
        return object()

    def record_empty(*_args, **_kwargs):
        calls.append("empty")
        return object()

    def record_run(*args):
        dispatch = args[-1]
        calls.append(("run", getattr(dispatch, "family", None)))

    def record_gqa_workspace(*_args, **_kwargs):
        calls.append("gqa_workspace")
        return object()

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setattr(torch, "empty_like", record_empty_like)
    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setitem(globals(), "_run_bwd_d64", record_run)
    monkeypatch.setitem(
        globals(),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        record_gqa_workspace,
    )

    for invalid in (
            0.0,
            -0.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            10**10000,
    ):
        calls.clear()
        with pytest.raises(ValueError, match="^D64 sm_scale must be finite and nonzero$"):
            fa_backward(q, k, v, o, do, lse, invalid, True)
        assert calls == []

    calls.clear()
    fa_backward(q, k, v, o, do, lse, -0.125, True)
    assert ("run", "causal_m192") in calls
    assert "gqa_workspace" not in calls

    calls.clear()
    fa_backward(q, k, v, o, do, lse, 0.125, True)
    assert ("run", "causal_scheduled_mha") in calls
    assert "gqa_workspace" not in calls


def test_d64_causal_gqa8_tiny_scale_dispatch_before_workspace(monkeypatch):
    q_shape = (8, 16, 1024, 64)
    k_shape = (8, 2, 1024, 64)
    select_kwargs = {
        "arch": "gfx950:sramecc+:xnack-",
        "cu_count": 256,
        "bases_aligned_16": True,
    }
    selected = _select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        sm_scale=0.125,
        **select_kwargs,
    )
    tiny_scale = _select_d64_dispatch(
        q_shape,
        k_shape,
        True,
        sm_scale=1e-38,
        **select_kwargs,
    )
    assert selected.family == "causal_scheduled_gqa8"
    assert tiny_scale.family == "causal_m192"

    class Properties:
        gcnArchName = select_kwargs["arch"]
        multi_processor_count = select_kwargs["cu_count"]

    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    stats = torch.empty(q_shape[:-1], device="meta", dtype=torch.float32)
    retained_dispatches = []

    def reject_selected_workspace(*_args, **_kwargs):
        raise AssertionError("tiny scale must fall back before GQA8 workspace allocation")

    def record_retained(*args):
        retained_dispatches.append(args[-1])

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setitem(
        globals(),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_selected_workspace,
    )
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", lambda *_args: None)
    monkeypatch.setitem(globals(), "_run_bwd_d64_direct", record_retained)

    outputs = fa_backward(q, k, k, q, q, stats, 1e-38, True)
    assert [dispatch.family for dispatch in retained_dispatches] == ["causal_m192"]
    assert all(output.device.type == "meta" for output in outputs)


def test_d64_causal_mha_tiny_scale_uses_retained_dispatch():
    shape = (4, 64, 4096, 64)
    select_kwargs = {
        "arch": "gfx950:sramecc+:xnack-",
        "cu_count": 256,
        "bases_aligned_16": True,
    }
    selected = _select_d64_dispatch(
        shape,
        shape,
        True,
        sm_scale=0.125,
        **select_kwargs,
    )
    tiny_scale = _select_d64_dispatch(
        shape,
        shape,
        True,
        sm_scale=1e-38,
        **select_kwargs,
    )
    assert selected.family == "causal_scheduled_mha"
    assert tiny_scale.family == "causal_m192"


def test_d64_causal_scale_oversized_integer_validation():
    with pytest.raises(ValueError, match=r"^D64 sm_scale must be finite and nonzero$"):
        _validate_d64_sm_scale(10**10000)


def test_d64_causal_scale_oversized_integer_eligibility():
    assert not _is_d64_scheduled_causal_eligible(
        (4, 64, 4096, 64),
        (4, 64, 4096, 64),
        True,
        arch="gfx950",
        cu_count=256,
        sm_scale=10**10000,
        bases_aligned_16=True,
    )


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "family", "owner_rows", "key_rows"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False, "noncausal_direct_n256", 32, 256),
        ((2, 32, 16384, 64), (2, 32, 16384, 64), True, "causal_m256", 256, 32),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True, "causal_m256", 256, 32),
        ((4, 48, 4096, 64), (4, 6, 4096, 64), True, "causal_m192", 192, 32),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True, "causal_m192", 192, 64),
    ],
)
def test_d64_structural_dispatch(q_shape, k_shape, causal, family, owner_rows, key_rows):
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)
    assert (dispatch.family, dispatch.owner_rows, dispatch.key_rows) == (family, owner_rows, key_rows)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "arch", "cu_count", "expected"),
    [
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx950",
            256,
            True,
            id="mha-square-16k",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            False,
            "gfx950:sramecc+:xnack-",
            256,
            True,
            id="gqa8-square-16k",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            256,
            False,
            id="insufficient-owner-grid",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            16,
            True,
            id="sufficient-owner-grid",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            True,
            "gfx950",
            256,
            False,
            id="causal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 8, 16384, 64),
            False,
            "gfx950",
            256,
            False,
            id="group4",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx942",
            256,
            False,
            id="wrong-arch",
        ),
        pytest.param(
            (1, 1, 256, 64),
            (1, 1, 256, 64),
            False,
            "gfx950",
            1,
            False,
            id="short",
        ),
        pytest.param(
            (1, 1, 4095, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            1,
            False,
            id="misaligned-sq",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4095, 64),
            False,
            "gfx950",
            1,
            False,
            id="misaligned-skv",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            None,
            None,
            False,
            id="missing-device-metadata",
        ),
    ],
)
def test_d64_fused_n256_eligibility(q_shape, k_shape, causal, arch, cu_count, expected):
    assert _is_d64_fused_n256_eligible(q_shape, k_shape, causal, arch=arch, cu_count=cu_count) is expected


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "arch", "cu_count", "family"),
    [
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_fused_n256",
            id="mha-square-16k",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_fused_n256",
            id="gqa8-square-16k",
        ),
        pytest.param(
            (1, 1, 4096, 64),
            (1, 1, 4096, 64),
            False,
            "gfx950",
            256,
            "noncausal_direct_n256",
            id="insufficient-owner-grid",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 4, 16384, 64),
            True,
            "gfx950",
            256,
            "causal_m256",
            id="causal",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 8, 16384, 64),
            False,
            "gfx950",
            256,
            "noncausal_direct_n256",
            id="group4",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            "gfx942",
            256,
            "noncausal_direct_n256",
            id="wrong-arch",
        ),
        pytest.param(
            (1, 1, 256, 64),
            (1, 1, 256, 64),
            False,
            "gfx950",
            1,
            "noncausal_direct_n256",
            id="short",
        ),
        pytest.param(
            (2, 32, 16384, 64),
            (2, 32, 16384, 64),
            False,
            None,
            None,
            "noncausal_direct_n256",
            id="missing-device-metadata",
        ),
    ],
)
def test_d64_fused_dispatch_is_structural(q_shape, k_shape, causal, arch, cu_count, family):
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal, arch=arch, cu_count=cu_count)
    assert dispatch.family == family


def test_d64_fused_dispatch_without_device_metadata_uses_direct_fallback():
    dispatch = _select_d64_dispatch((2, 32, 16384, 64), (2, 32, 16384, 64), False)
    assert dispatch.family == "noncausal_direct_n256"


def test_d64_fused_n256_uses_direct_score_and_output_layouts():

    def normalized_assignments(source, names):
        assignments = {name: [] for name in names}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in assignments:
                assignments[target.id].append(ast.unparse(node.value))
        return assignments

    assert normalized_assignments(_attn_bwd_d64_fused_n256_update.src, ("p_nd", "ds_nd")) == {
        "p_nd": ["tlx.require_layout(p.to(tl.bfloat16), p_op0_nd, pin=False)"],
        "ds_nd": ["tlx.require_layout(ds_bf16, ds_op0_nd, pin=False)"],
    }
    assert normalized_assignments(_attn_bwd_d64_fused_n256_kernel.src, ("dk_out", "dv_out")) == {
        "dk_out": ["tlx.require_layout((dk * dk_scale).to(tl.bfloat16), kv_async_layout, pin=False)"],
        "dv_out": ["tlx.require_layout(dv.to(tl.bfloat16), kv_async_layout, pin=False)"],
    }


def test_d64_fused_workspace_contract():
    q = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_mha = torch.empty((2, 32, 4096, 64), device="meta", dtype=torch.bfloat16)
    k_gqa = torch.empty((2, 4, 4096, 64), device="meta", dtype=torch.bfloat16)
    mha = _D64Dispatch("noncausal_fused_n256", 32, 256, 1)
    gqa = _D64Dispatch("noncausal_fused_n256", 32, 256, 8)

    mha_acc, mha_dk, mha_dv = _allocate_bwd_d64_fused_workspaces(q, k_mha, mha)
    gqa_acc, gqa_dk, gqa_dv = _allocate_bwd_d64_fused_workspaces(q, k_gqa, gqa)

    assert mha_acc.shape == q.shape and mha_acc.dtype is torch.float32
    assert mha_acc.is_contiguous()
    assert mha_dk is mha_dv is None
    assert gqa_acc.shape == q.shape and gqa_acc.dtype is torch.float32
    assert gqa_acc.is_contiguous()
    assert gqa_dk.shape == gqa_dv.shape == (2, 4, 8, 4096, 64)
    assert gqa_dk.dtype is gqa_dv.dtype is torch.bfloat16
    assert gqa_dk.is_contiguous() and gqa_dv.is_contiguous()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_preprocess_computes_delta_and_zeros_fp32_dq_gfx950():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(47)
    o = torch.randn((1, 2, 256, 64), generator=generator, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(o.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    delta = torch.empty(o.shape[:-1], device="cuda", dtype=torch.float32)
    dq_acc = torch.full(o.shape, 7.0, device="cuda", dtype=torch.float32)
    _attn_bwd_preprocess_kernel.device_caches.clear()

    _run_bwd_preprocess(o, do, delta, dq_acc=dq_acc)

    expected = torch.sum(o.float() * do.float(), dim=-1)
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(dq_acc).item() == 0

    device = torch.cuda.current_device()
    compiled = tuple(_attn_bwd_preprocess_kernel.device_caches[device][0].values())
    assert len(compiled) == 1
    obj = compiled[0]
    amdgcn = obj.asm["amdgcn"]
    private_segment = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    resources = {
        "n_spills": obj.n_spills,
        "global_scratch_bytes": obj.metadata.global_scratch_size,
        "private_segment_bytes": (private_segment.pop() if len(private_segment) == 1 else None),
        "scratch_load_instructions": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_store_instructions": len(re.findall(r"\bscratch_store", amdgcn)),
    }
    assert resources == {
        "n_spills": 0,
        "global_scratch_bytes": 0,
        "private_segment_bytes": 0,
        "scratch_load_instructions": 0,
        "scratch_store_instructions": 0,
    }


_D64_ZERO_RESOURCE_FIELDS = (
    "n_spills",
    "global_scratch_bytes",
    "private_segment_bytes",
    "scratch_load_instructions",
    "scratch_store_instructions",
)


def _d64_code_object_resource(obj):
    amdgcn = obj.asm["amdgcn"]
    private_segments = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    vector_vgpr_counts = {int(value) for value in re.findall(r";\s+NumVgprs:\s+(\d+)", amdgcn)}
    agpr_counts = {int(value) for value in re.findall(r";\s+NumAgprs:\s+(\d+)", amdgcn)}
    return {
        "vgpr_count": obj.n_regs,
        "vector_vgpr_count": (vector_vgpr_counts.pop() if len(vector_vgpr_counts) == 1 else None),
        "agpr_count": agpr_counts.pop() if len(agpr_counts) == 1 else None,
        "unified_vgpr_count": obj.n_regs,
        "lds_bytes": obj.metadata.shared,
        "n_spills": obj.n_spills,
        "global_scratch_bytes": obj.metadata.global_scratch_size,
        "private_segment_bytes": (private_segments.pop() if len(private_segments) == 1 else None),
        "scratch_load_instructions": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_store_instructions": len(re.findall(r"\bscratch_store", amdgcn)),
    }


def test_d64_structural_dispatch_codegen_resource_contract():

    class Metadata:
        shared = 41472
        global_scratch_size = 0

    class CompiledObject:
        n_regs = 268
        n_spills = 0
        metadata = Metadata()
        asm = {"amdgcn": "\n".join((
            "; NumVgprs: 254",
            "; NumAgprs: 12",
            ".private_segment_fixed_size: 0",
        ))}

    resource = _d64_code_object_resource(CompiledObject())
    assert resource["vgpr_count"] == 268
    assert resource["vector_vgpr_count"] == 254
    assert resource["agpr_count"] == 12
    assert resource["unified_vgpr_count"] == 268
    assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])


def _assert_d64_code_object_scratch_free(name, obj):
    resource = _d64_code_object_resource(obj)
    for field in _D64_ZERO_RESOURCE_FIELDS:
        assert resource[field] == 0, (name, resource)
    return resource


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape, expected_logical_n, expected_use_xcd, seed",
    (
        pytest.param((1, 8, 1, 256, 256, 64), 32, False, 107, id="square_flat"),
        pytest.param((1, 8, 8, 256, 320, 64), 64, True, 108, id="rectangular_xcd"),
    ),
)
def test_d64_causal_common_dq_stat_modes_gfx950(shape, expected_logical_n, expected_use_xcd, seed):
    batch, hq, hkv, sq, skv, _head_dim = shape
    case = _make_d64_gqa_smoke_case(shape, causal=True, seed=seed)
    launches = _d64_dq_launch_plan(batch, hq, hkv, sq, skv, 192, 1, True)

    def dispatch(family, stat_mode):
        return _D64Dispatch(
            family,
            owner_rows=192,
            key_rows=64,
            kv_splits=1,
            selected_causal=True,
            stat_mode=stat_mode,
            dq_logical_n=_d64_selected_causal_logical_n(sq, skv, hq // hkv),
            dq_use_xcd=_d64_use_dq_xcd(batch, hkv, sq, skv, 192),
            dq_launches=launches,
        )

    mha_dispatch = dispatch("causal_scheduled_mha", _D64_MHA_POSITIVE)
    gqa_dispatch = dispatch("causal_scheduled_gqa8", _D64_GQA_SIGNED)
    for mode_dispatch in (mha_dispatch, gqa_dispatch):
        assert mode_dispatch.dq_logical_n == expected_logical_n
        assert mode_dispatch.dq_use_xcd is expected_use_xcd

    delta_mha = torch.empty_like(case.lse)
    delta_gqa = torch.empty_like(case.lse)
    lse_term_gqa = torch.empty_like(case.lse)
    dq_mha = torch.empty_like(case.q)
    dq_gqa = torch.empty_like(case.q)

    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta_mha,
        None,
        dq_mha,
        case.sm_scale,
        mha_dispatch,
    )
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta_gqa,
        lse_term_gqa,
        dq_gqa,
        case.sm_scale,
        gqa_dispatch,
    )
    torch.cuda.synchronize()

    positive = torch.sum(case.o.float() * case.do.float(), dim=-1)
    torch.testing.assert_close(delta_mha, positive, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(delta_gqa, -positive, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        lse_term_gqa,
        -case.lse.float() * math.log2(math.e),
        rtol=1e-5,
        atol=1e-5,
    )
    for name, actual in (("mha", dq_mha), ("gqa", dq_gqa)):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - case.grads[0]) / torch.linalg.vector_norm(case.grads[0])
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_common_dq_peeled_m128_accuracy_gfx950():
    shape = (1, 8, 8, 8192, 8192, 64)
    case = _make_d64_aten_case(shape, seed=137, causal=True)
    batch, hq, hkv, sq, skv, _head_dim = shape
    owner_rows = 192
    owners = triton.cdiv(sq, owner_rows)
    launches = _d64_dq_launch_plan(
        batch,
        hq,
        hkv,
        sq,
        skv,
        owner_rows,
        cu_count=8,
        host_skip_owner_tail=True,
    )
    assert launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )
    assert _d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows)
    peeled_interval = _d64_causal_owner_interval(owners - 1, sq, owner_rows)
    full_interval = _d64_causal_owner_interval(owners - 2, sq, owner_rows)
    assert peeled_interval == (0, 128)
    assert full_interval == (128, 320)

    dispatch = _D64Dispatch(
        family="causal_scheduled_mha",
        owner_rows=owner_rows,
        key_rows=64,
        kv_splits=1,
        selected_causal=True,
        stat_mode=_D64_MHA_POSITIVE,
        dq_logical_n=32,
        dq_use_xcd=True,
        dq_launches=launches,
    )
    delta = torch.empty_like(case.lse)
    dq = torch.empty_like(case.q)
    _attn_bwd_dq_d64_causal_mha_kernel.device_caches.clear()
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        None,
        dq,
        case.sm_scale,
        dispatch,
    )
    torch.cuda.synchronize()

    positive = torch.sum(case.o.float() * case.do.float(), dim=-1)
    torch.testing.assert_close(delta, positive, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(dq).all()
    expected_dq = case.grads[0].float()
    for region_name, row_interval in (
        ("peeled_m128", peeled_interval),
        ("full_m192", full_interval),
        ("all_dq", (0, sq)),
    ):
        row_begin, row_end = row_interval
        actual_region = dq[:, :, row_begin:row_end].float()
        expected_region = expected_dq[:, :, row_begin:row_end]
        relative_l2 = torch.linalg.vector_norm(actual_region -
                                               expected_region) / torch.linalg.vector_norm(expected_region)
        assert relative_l2.item() < 5e-3, (
            region_name,
            relative_l2.item(),
        )
    device = torch.cuda.current_device()
    objects = tuple(_attn_bwd_dq_d64_causal_mha_kernel.device_caches[device][0].values())
    assert len(objects) == 2


def test_d64_causal_common_dq_direct_load_ast_contract():
    helper = ast.parse(_attn_bwd_dq_d64_causal_load_q64.src).body[0]

    def dotted_name(call):
        value = call.func
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    direct_loads = [
        statement.targets[0].id for statement in helper.body if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.buffer_load"
    ]
    assert direct_loads == ["do", "o", "q"]

    stat_mode_branch = next(
        statement for statement in helper.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "STAT_MODE == _D64_MHA_POSITIVE_JIT")
    producer_lse_term = next(
        statement.value
        for statement in ast.walk(stat_mode_branch)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "producer_lse_term")
    assert ast.unparse(producer_lse_term) == "-lse * 1.4426950408889634"
    lse_term_store = next(node for node in ast.walk(stat_mode_branch)
                          if isinstance(node, ast.Call) and dotted_name(node) == "tl.store"
                          and ast.unparse(node.args[0]) == "LSE_TERM + stats_base + rows")
    assert ast.unparse(lse_term_store.args[1]) == "producer_lse_term"

    q_scale = next(statement.value
                   for statement in ast.walk(stat_mode_branch)
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "q_scale")
    q_scale_fill = next(node for node in ast.walk(q_scale)
                        if isinstance(node, ast.Call) and dotted_name(node) == "tl.full")
    assert ast.unparse(q_scale_fill.args[1]) == "SM_SCALE * 1.4426950408889634"
    q_scale_products = [
        node for node in ast.walk(stat_mode_branch)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and ast.unparse(node.right) == "q_scale"
    ]
    assert len(q_scale_products) == 1
    assert ast.unparse(q_scale_products[0].left) == "q.to(tl.float32)"

    handoff = next(statement.value
                   for statement in helper.body
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "product"
                   and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.release_layout")
    assert ast.unparse(handoff.args[0]) == "product"

    reduction = next(statement.value
                     for statement in helper.body
                     if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                     and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "positive")
    assert dotted_name(reduction) == "tl.sum"
    assert ast.unparse(reduction.args[0]) == "product"
    assert ast.literal_eval(reduction.keywords[0].value) == 1

    step = ast.parse(_attn_bwd_dq_d64_causal_step.src).body[0]
    n64_handoff = next(statement for statement in step.body
                       if isinstance(statement, ast.If) and ast.unparse(statement.test) == "BLOCK_N == 64")
    score_tie = next(
        statement.value
        for statement in n64_handoff.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "scores"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(score_tie.args[0]) == "scores"
    assert ast.literal_eval(
        next(keyword.value for keyword in score_tie.keywords if keyword.arg == "registers_per_group")) == 2
    n32_handoff = next(statement for statement in step.body
                       if isinstance(statement, ast.If) and ast.unparse(statement.test) == "BLOCK_N == 32")
    assert not any(
        isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
        for node in ast.walk(n32_handoff))
    assert any(
        isinstance(node, ast.Call) and dotted_name(node) == "tlx.require_layout"
        and ast.unparse(node.args[0]) == "scores" for node in ast.walk(n32_handoff))
    ds_handoff = next(
        statement.value for statement in step.body if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "ds"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(ds_handoff.args[0]) == "p * dp"
    assert ast.literal_eval(
        next(keyword.value for keyword in ds_handoff.keywords if keyword.arg == "registers_per_group")) == 2
    assert not any(isinstance(node, ast.Call) and dotted_name(node) == "tlx.local_load" for node in ast.walk(step))
    score_scale_guard = next(statement for statement in step.body
                             if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not SCORE_PRE_SCALED")
    assert any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and ast.unparse(node.left) == "scores"
        for node in ast.walk(score_scale_guard))

    score32 = ast.parse(_attn_bwd_dq_d64_causal_score32.src).body[0]
    score32_scale_guard = next(
        statement for statement in score32.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not SCORE_PRE_SCALED")
    assert any(
        isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult) and ast.unparse(node.target) == "scores"
        for node in ast.walk(score32_scale_guard))
    split_score_handoffs = [
        node for node in ast.walk(score32)
        if isinstance(node, ast.Call) and dotted_name(node) == "tlx.amd_register_handoff"
    ]
    assert not split_score_handoffs

    finish32 = ast.parse(_attn_bwd_dq_d64_causal_finish32.src).body[0]
    finish32_handoff = next(
        statement.value
        for statement in finish32.body
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "ds"
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.amd_register_handoff")
    assert ast.unparse(finish32_handoff.args[0]) == "p * dp"
    assert ast.literal_eval(
        next(keyword.value for keyword in finish32_handoff.keywords if keyword.arg == "register_class")) == "vgpr"
    assert ast.literal_eval(
        next(keyword.value for keyword in finish32_handoff.keywords if keyword.arg == "registers_per_group")) == 2

    def positional_argument(call, function, name):
        parameter_names = [argument.arg for argument in function.args.args]
        return ast.unparse(call.args[parameter_names.index(name)])

    nslice = ast.parse(_attn_bwd_dq_d64_causal_nslice.src).body[0]
    shared_operand_assignments = [
        statement for statement in ast.walk(nslice)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(
            statement.targets[0], ast.Name) and statement.targets[0].id in {"k_source", "kv_source", "v_source"}
    ]
    assert [statement.targets[0].id for statement in shared_operand_assignments] == [
        "k_source",
        "kv_source",
        "v_source",
    ]
    assert all(
        isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "tlx.local_load"
        for statement in shared_operand_assignments)
    fragment_steps = [
        node for node in ast.walk(nslice)
        if isinstance(node, ast.Call) and dotted_name(node) == "_attn_bwd_dq_d64_causal_step"
    ]
    assert len(fragment_steps) == 11
    assert all([ast.unparse(arg)
                for arg in call.args[6:9]] == ["k_source", "v_source", "kv_source"]
               for call in fragment_steps)
    assert all(positional_argument(call, step, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in fragment_steps)

    m256 = ast.parse(_attn_bwd_dq_d64_causal_m256_unmasked_n32.src).body[0]
    ordered_assignments = [
        statement.targets[0].id for statement in m256.body if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in {
            "kt",
            "k_nd",
            "scores0",
            "vt",
            "dp0",
            "dq0",
            "dq1",
            "dq3",
            "scores2",
            "dp2",
            "dq2",
        }
    ]
    assert ordered_assignments == [
        "kt",
        "k_nd",
        "scores0",
        "vt",
        "dp0",
        "dq0",
        "dq1",
        "dq3",
        "scores2",
        "dp2",
        "dq2",
    ]
    m256_fragment_steps = [
        statement.value for statement in m256.body if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "_attn_bwd_dq_d64_causal_step"
    ]
    assert [ast.unparse(call.args[0]) for call in m256_fragment_steps] == ["dq1", "dq3"]
    assert all(ast.unparse(call.args[15]) == "32" for call in m256_fragment_steps)
    assert all(ast.unparse(call.args[16]) == "False" for call in m256_fragment_steps)
    assert all(
        positional_argument(call, step, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in m256_fragment_steps)
    score32_calls = [
        statement.value for statement in m256.body if isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in {"scores0", "scores2"}
        and isinstance(statement.value, ast.Call) and dotted_name(statement.value) == "_attn_bwd_dq_d64_causal_score32"
    ]
    assert "PACK_TIE" not in {argument.arg for argument in score32.args.args}
    assert all(positional_argument(call, score32, "SCORE_PRE_SCALED") == "SCORE_PRE_SCALED" for call in score32_calls)

    impl = ast.parse(_attn_bwd_dq_d64_causal_impl.src).body[0]
    q3_residency = next(node for node in ast.walk(impl) if isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "q3"
                        and isinstance(node.value, ast.Call) and dotted_name(node.value) == "tlx.amd_register_resident")
    assert ast.unparse(q3_residency.value.args[0]) == "q3"
    assert ast.literal_eval(
        next(keyword.value for keyword in q3_residency.value.keywords if keyword.arg == "register_class")) == "agpr"
    assert ast.literal_eval(
        next(keyword.value for keyword in q3_residency.value.keywords if keyword.arg == "registers_per_group")) == 8
    assert not any(isinstance(node, ast.Name) and node.id == "q3_buffer" for node in ast.walk(impl))
    score_pre_scaled = next(statement for statement in impl.body if isinstance(statement, ast.AnnAssign)
                            and isinstance(statement.target, ast.Name) and statement.target.id == "score_pre_scaled")
    assert ast.unparse(score_pre_scaled.annotation) == "tl.constexpr"
    assert ast.unparse(score_pre_scaled.value) == "STAT_MODE == _D64_GQA_SIGNED_JIT"
    score_consumers = {
        "_attn_bwd_dq_d64_causal_m256_unmasked_n32": m256,
        "_attn_bwd_dq_d64_causal_nslice": nslice,
    }
    consumer_calls = [
        node for node in ast.walk(impl) if isinstance(node, ast.Call) and dotted_name(node) in score_consumers
    ]
    assert len(consumer_calls) == 8
    assert all(
        positional_argument(
            call,
            score_consumers[dotted_name(call)],
            "SCORE_PRE_SCALED",
        ) == "score_pre_scaled" for call in consumer_calls)

    store = ast.parse(_attn_bwd_dq_d64_causal_store_q64.src).body[0]
    offsets = next(statement.value
                   for statement in store.body
                   if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "offsets")
    assert ast.unparse(offsets) == "(local_rows[:, None] * D + cols[None, :]).to(tl.int32)"
    store_call = next(node for node in ast.walk(store)
                      if isinstance(node, ast.Call) and dotted_name(node) == "tlx.buffer_store")
    assert ast.unparse(store_call.args[1]) == "DQ + q_base + row_start * D"


def test_d64_causal_common_dq_launch_order(monkeypatch):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    recorder = LaunchRecorder()
    monkeypatch.setitem(globals(), "_attn_bwd_dq_d64_causal_mha_kernel", recorder)

    def tensors(batch, hq, hkv, sq, skv):
        q = torch.empty((batch, hq, sq, 64), device="meta", dtype=torch.bfloat16)
        k = torch.empty((batch, hkv, skv, 64), device="meta", dtype=torch.bfloat16)
        stats = torch.empty((batch, hq, sq), device="meta", dtype=torch.float32)
        return q, k, stats

    flat_q, flat_k, flat_stats = tensors(1, 8, 1, 256, 256)
    flat_launches = _d64_dq_launch_plan(1, 8, 1, 256, 256, 192, 1, True)
    flat = _D64Dispatch(
        "causal_scheduled_mha",
        192,
        64,
        1,
        True,
        _D64_MHA_POSITIVE,
        32,
        False,
        flat_launches,
    )
    _launch_bwd_d64_causal_dq(
        flat_q,
        flat_k,
        flat_k,
        flat_q,
        flat_q,
        flat_stats,
        flat_stats,
        None,
        flat_q,
        0.125,
        flat,
    )

    batch, hq, hkv, sq = 4, 64, 8, 8192
    xcd_q, xcd_k, xcd_stats = tensors(batch, hq, hkv, sq, sq)
    xcd_launches = _d64_dq_launch_plan(batch, hq, hkv, sq, sq, 192, 256, True)
    xcd = _D64Dispatch(
        "causal_scheduled_mha",
        192,
        64,
        1,
        True,
        _D64_MHA_POSITIVE,
        32,
        True,
        xcd_launches,
    )
    _launch_bwd_d64_causal_dq(
        xcd_q,
        xcd_k,
        xcd_k,
        xcd_q,
        xcd_q,
        xcd_stats,
        xcd_stats,
        None,
        xcd_q,
        0.125,
        xcd,
    )

    assert len(recorder.calls) == 1 + len(xcd_launches)
    flat_grid, _flat_args, flat_kwargs = recorder.calls[0]
    assert flat_grid == (1 * 8 * flat_launches[0].launch_tiles, )
    assert not flat_kwargs["USE_DQ_XCD"]
    assert flat_kwargs["KV_PIPELINE_STAGES"] == _D64_DQ_KV_STAGES

    owners = triton.cdiv(sq, 192)
    assert xcd_launches == (
        _D64DQLaunch(owners - 1, False, 0, owners - 1, 3, 0),
        _D64DQLaunch(1, False, owners - 1, 1, 2, 192),
    )
    for call, launch in zip(recorder.calls[1:], xcd_launches):
        grid, args, kwargs = call
        assert grid == (batch * hq * launch.launch_tiles, )
        assert len(args) == 8
        assert kwargs["USE_DQ_XCD"]
        assert kwargs["OWNER_PID_BASE"] == launch.owner_pid_base
        assert kwargs["LAUNCH_Q_TILES"] == launch.launch_q_tiles
        assert kwargs["OWNER_FRAGMENTS"] == launch.owner_fragments
        assert kwargs["GRID_OWNER_M"] == launch.grid_owner_m
        assert kwargs["KV_PIPELINE_STAGES"] == _D64_DQ_KV_STAGES

    decoded = []
    for launch_index, launch in enumerate(xcd_launches):
        for pid in range(batch * hq * launch.launch_tiles):
            coords = _d64_decode_dq_pid(
                pid,
                batch,
                hq,
                hkv,
                launch.launch_tiles,
                True,
                launch.owner_pid_base,
            )
            decoded.append((launch_index, coords))
    assert all(item[1][2] < owners - 1 for item in decoded if item[0] == 0)
    assert all(item[1][2] == owners - 1 for item in decoded if item[0] == 1)
    assert _d64_causal_owner_interval(0, sq, 192)[0] > _d64_causal_owner_interval(owners - 1, sq, 192)[0]
    assert _d64_causal_owner_interval(owners - 1, sq, 192) == (0, 128)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_common_dq_codegen_gfx950():

    def compile_variant(name, shape, owner_rows, launches, stat_mode, expected_lds, expected_private_segment):
        batch, hq, hkv, sq, skv, _head_dim = shape
        q = torch.zeros((batch, hq, sq, 64), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((batch, hkv, skv, 64), device="cuda", dtype=torch.bfloat16)
        stats = torch.zeros((batch, hq, sq), device="cuda", dtype=torch.float32)
        dq = torch.empty_like(q)
        signed = stat_mode == _D64_GQA_SIGNED
        kernel = (_attn_bwd_dq_d64_causal_gqa8_kernel if signed else _attn_bwd_dq_d64_causal_mha_kernel)
        kernel.device_caches.clear()
        dispatch = _D64Dispatch(
            family="causal_scheduled_gqa8" if signed else "causal_scheduled_mha",
            owner_rows=owner_rows,
            key_rows=64,
            kv_splits=1,
            selected_causal=True,
            stat_mode=stat_mode,
            dq_logical_n=_d64_selected_causal_logical_n(sq, skv, hq // hkv),
            dq_use_xcd=_d64_use_dq_xcd(batch, hkv, sq, skv, owner_rows),
            dq_launches=launches,
        )
        lse_term = torch.empty_like(stats) if signed else None
        _launch_bwd_d64_causal_dq(q, k, k, q, q, stats, stats, lse_term, dq, 0.125, dispatch)
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        objects = tuple(kernel.device_caches[device][0].values())
        assert len(objects) == 1, (name, len(objects))
        obj = objects[0]
        resource = _d64_code_object_resource(obj)
        if expected_private_segment == 0:
            for field in _D64_ZERO_RESOURCE_FIELDS:
                assert resource[field] == 0, (name, resource)
        else:
            expected_spill_resources = {
                264: {
                    "n_spills": 66,
                    "global_scratch_bytes": 0,
                    "private_segment_bytes": 264,
                    "scratch_load_instructions": 81,
                    "scratch_store_instructions": 54,
                },
                292: {
                    "n_spills": 73,
                    "global_scratch_bytes": 0,
                    "private_segment_bytes": 292,
                    "scratch_load_instructions": 73,
                    "scratch_store_instructions": 48,
                },
            }
            assert {
                "n_spills": resource["n_spills"],
                "global_scratch_bytes": resource["global_scratch_bytes"],
                "private_segment_bytes": resource["private_segment_bytes"],
                "scratch_load_instructions": resource["scratch_load_instructions"],
                "scratch_store_instructions": resource["scratch_store_instructions"],
            } == expected_spill_resources[expected_private_segment], (name, resource)
        assert resource["unified_vgpr_count"] == resource["vgpr_count"]
        assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])
        assert resource["lds_bytes"] == expected_lds, (name, resource)
        amdgcn = obj.asm["amdgcn"]
        assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn), name
        assert not re.search(r"\b\w*atomic\w*\b", amdgcn), name
        return resource

    square_sq = 8192
    square_owners = triton.cdiv(square_sq, 192)
    rectangular_shape = (1, 8, 8, 256, 320, 64)
    rectangular_launches = _d64_dq_launch_plan(1, 8, 8, 256, 320, 192, 1, True)
    deep_m192_gqa8_shape = (8, 8, 1, 1024, 2048, 64)
    deep_m192_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 1024, 2048, 192, 256, True)
    short_square_m256_gqa8_shape = (4, 48, 6, 4096, 4096, 64)
    short_square_m256_gqa8_launches = _d64_dq_launch_plan(4, 48, 6, 4096, 4096, 256, 256, True)
    square_m256_gqa8_shape = (1, 8, 1, 16384, 16384, 64)
    square_m256_gqa8_launches = _d64_dq_launch_plan(1, 8, 1, 16384, 16384, 256, 256, True)
    deep_m256_gqa8_shape = (8, 8, 1, 4096, 8192, 64)
    deep_m256_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 4096, 8192, 256, 256, True)
    long_m256_gqa8_shape = (8, 8, 1, 4096, 12288, 64)
    long_m256_gqa8_launches = _d64_dq_launch_plan(8, 8, 1, 4096, 12288, 256, 256, True)
    variants = (
        (
            "peeled_m128_square",
            (1, 8, 8, square_sq, square_sq, 64),
            192,
            (_D64DQLaunch(1, False, square_owners - 1, 1, 2, 192), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m192_square",
            (1, 8, 8, square_sq, square_sq, 64),
            192,
            (_D64DQLaunch(
                square_owners - 1,
                False,
                0,
                square_owners - 1,
                3,
                0,
            ), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m192_rectangular",
            rectangular_shape,
            192,
            rectangular_launches,
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "m256_square",
            (1, 1, 1, 16384, 16384, 64),
            256,
            (_D64DQLaunch(64, True, 0, 0, 4, 0), ),
            _D64_MHA_POSITIVE,
            33536,
            0,
        ),
        (
            "gqa_signed_rectangular",
            rectangular_shape,
            192,
            rectangular_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_deep_m192_n32",
            deep_m192_gqa8_shape,
            192,
            deep_m192_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_short_square_m256_n32",
            short_square_m256_gqa8_shape,
            256,
            short_square_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_square_m256_n32",
            square_m256_gqa8_shape,
            256,
            square_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_deep_m256_n32",
            deep_m256_gqa8_shape,
            256,
            deep_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
        (
            "gqa_signed_long_m256_n32",
            long_m256_gqa8_shape,
            256,
            long_m256_gqa8_launches,
            _D64_GQA_SIGNED,
            33536,
            0,
        ),
    )
    resources = {
        name: compile_variant(
            name,
            shape,
            owner_rows,
            launches,
            stat_mode,
            expected_lds,
            expected_private_segment,
        )
        for (
            name,
            shape,
            owner_rows,
            launches,
            stat_mode,
            expected_lds,
            expected_private_segment,
        ) in variants
    }
    assert tuple(resources) == tuple(variant[0] for variant in variants)
    assert resources["gqa_signed_deep_m192_n32"]["unified_vgpr_count"] <= 224
    assert resources["gqa_signed_short_square_m256_n32"]["vector_vgpr_count"] == 245
    assert resources["gqa_signed_short_square_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_short_square_m256_n32"]["unified_vgpr_count"] == 256
    assert resources["gqa_signed_square_m256_n32"]["vector_vgpr_count"] == 245
    assert resources["gqa_signed_square_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_square_m256_n32"]["unified_vgpr_count"] == 256
    assert resources["gqa_signed_deep_m256_n32"]["vector_vgpr_count"] == 244
    assert resources["gqa_signed_deep_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_deep_m256_n32"]["unified_vgpr_count"] == 252
    assert resources["gqa_signed_long_m256_n32"]["vector_vgpr_count"] == 244
    assert resources["gqa_signed_long_m256_n32"]["agpr_count"] == 8
    assert resources["gqa_signed_long_m256_n32"]["unified_vgpr_count"] == 252


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_mha_accuracy_gfx950(monkeypatch):
    case_name = "mha_square_16k_causal"
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal and hq == hkv and sq == skv == 16384
    real_mha_launcher = globals().get("_launch_bwd_d64_causal_mha_dkdv")
    original_retained = _launch_bwd_d64_dkdv
    producer_calls = []

    def record_mha(*args):
        producer_calls.append(("mha", args[-1]))
        assert callable(real_mha_launcher)
        return real_mha_launcher(*args)

    def record_retained(*args, **kwargs):
        producer_calls.append(("retained", args[-1]))
        return original_retained(*args, **kwargs)

    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_mha_dkdv", record_mha)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dkdv", record_retained)

    case = _make_d64_aten_case(
        (batch, hq, hkv, sq, skv, head_dim),
        seed=223,
        causal=True,
    )
    actual = fa_backward(*case.kernel_args)

    assert [call[0] for call in producer_calls] == ["mha"]
    dispatch = producer_calls[0][1]
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), (case_name, name)
        expected_norm = torch.linalg.vector_norm(expected.float())
        assert torch.isfinite(expected_norm) and expected_norm.item() > 0.0
        error_norm = torch.linalg.vector_norm(result.float() - expected.float())
        assert torch.isfinite(error_norm), (case_name, name)
        relative_l2 = error_norm / expected_norm
        assert relative_l2.item() < 5e-3, (
            case_name,
            name,
            relative_l2.item(),
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape",
    (
        pytest.param(
            (1, 24, 24, 4096, 4096, 64),
            id="square-sq4096-skv4096",
        ),
        pytest.param(
            (1, 24, 24, 4096, 8192, 64),
            id="rectangular-sq4096-skv8192",
        ),
    ),
)
def test_d64_causal_mha_direct_publication_gfx950(monkeypatch, shape):
    real_mha_launcher = globals().get("_launch_bwd_d64_causal_mha_dkdv")
    assert callable(real_mha_launcher)
    case = _make_d64_aten_case(shape, seed=227, causal=True)
    batch, hq, hkv, sq, skv, _head_dim = shape
    dispatch = _select_d64_dispatch(
        tuple(case.q.shape),
        tuple(case.k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=case.sm_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    delta = torch.sum(case.o.float() * case.do.float(), dim=-1)
    dq = torch.empty_like(case.q)
    dk = torch.full_like(case.k, float("nan"))
    dv = torch.full_like(case.v, float("nan"))
    producer_targets = []

    def record_publication(*args):
        producer_targets.append((args[6], args[7]))
        return real_mha_launcher(*args)

    def keep_precomputed_delta(*args):
        assert args[6] is delta
        assert args[7] is None

    def reject_partial_allocation(*_args, **_kwargs):
        raise AssertionError("direct causal MHA publication must not allocate partials")

    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_dq", keep_precomputed_delta)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_mha_dkdv", record_publication)
    monkeypatch.setitem(globals(), "_allocate_bwd_d64_kv_partials", reject_partial_allocation)
    monkeypatch.setitem(
        globals(),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_partial_allocation,
    )
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dkdv", reject_partial_allocation)

    _run_bwd_d64(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        dq,
        dk,
        dv,
        case.sm_scale,
        True,
        dispatch,
    )
    torch.cuda.synchronize()

    assert len(producer_targets) == 1
    assert producer_targets[0][0] is dk
    assert producer_targets[0][1] is dv
    for name, result, expected in (
        ("dk", dk, case.grads[1]),
        ("dv", dv, case.grads[2]),
    ):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def _compile_d64_causal_mha_producer_variant(name, shape):
    batch, hq, hkv, sq, skv, head_dim = shape
    assert hq == hkv
    q = torch.empty((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    do = torch.empty_like(q)
    lse = torch.empty((batch, hq, sq), device="cuda", dtype=torch.float32)
    delta = torch.empty_like(lse)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_mha"
    kernel = globals().get("_attn_bwd_dkdv_d64_causal_mha_kernel")
    assert kernel is not None, name
    kernel.device_caches.clear()
    _launch_bwd_d64_causal_mha_dkdv(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,
        0.125,
        dispatch,
    )
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    objects = tuple(kernel.device_caches[device][0].values())
    assert len(objects) == 1, (name, len(objects))
    return dispatch, objects[0]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "variant_name, shape",
    (
        pytest.param(
            "mha_bm32_bn64_square",
            (1, 24, 24, 4096, 4096, 64),
            id="square-sq4096-skv4096",
        ),
        pytest.param(
            "mha_bm32_bn64_rectangular",
            (1, 24, 24, 4096, 8192, 64),
            id="rectangular-sq4096-skv8192",
        ),
    ),
)
def test_d64_causal_mha_codegen_gfx950(variant_name, shape):
    dispatch, obj = _compile_d64_causal_mha_producer_variant(variant_name, shape)
    assert dispatch.family == "causal_scheduled_mha"
    resource = _assert_d64_code_object_scratch_free(variant_name, obj)
    assert resource["vgpr_count"] is not None
    assert resource["vector_vgpr_count"] is not None
    assert resource["agpr_count"] is not None
    assert resource["unified_vgpr_count"] == resource["vgpr_count"]
    assert resource["unified_vgpr_count"] >= (resource["vector_vgpr_count"] + resource["agpr_count"])
    assert resource["lds_bytes"] == 16896
    amdgcn = obj.asm["amdgcn"]
    assert not re.search(r"\b\w*atomic\w*\b", amdgcn)
    assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_mha_positive_compatibility_gfx950(monkeypatch):
    batch, heads, sq, head_dim = 1, 24, 4096, 64
    generator = torch.Generator(device="cuda")
    generator.manual_seed(109)

    def random(shape):
        return torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()

    q = random((batch, heads, sq, head_dim))
    k = random(q.shape)
    v = random(q.shape)
    do = random(q.shape)
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, False, scale=sm_scale)
    o, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    o = o.contiguous()
    lse = lse.contiguous()
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do,
        q,
        k,
        v,
        o,
        lse,
        cum_q,
        cum_k,
        max_q,
        max_k,
        0.0,
        True,
        rng,
        unused,
        scale=sm_scale,
    )

    calls = []
    original_dq = _launch_bwd_d64_causal_dq
    original_dkdv = _launch_bwd_d64_causal_mha_dkdv

    def record_dq(*args, **kwargs):
        calls.append(("dq", args[7], args[6], args[-1]))
        return original_dq(*args, **kwargs)

    def record_dkdv(*args, **kwargs):
        calls.append(("dkdv", args[5]))
        return original_dkdv(*args, **kwargs)

    def reject_preprocess(*_args, **_kwargs):
        calls.append(("preprocess", ))
        raise AssertionError("selected causal MHA must not preprocess")

    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_dq", record_dq)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_mha_dkdv", record_dkdv)
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", reject_preprocess)

    actual = fa_backward(q, k, v, o, do, lse, sm_scale, True)

    assert [call[0] for call in calls] == ["dq", "dkdv"]
    assert calls[0][1] is None
    assert calls[0][2] is calls[1][1]
    dispatch = calls[0][3]
    assert dispatch.family == "causal_scheduled_mha"
    assert dispatch.stat_mode == _D64_MHA_POSITIVE
    for name, result, expected in zip(("dq", "dk", "dv"), actual, reference):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


def test_d64_causal_gqa8_signed_pipeline_gfx950(monkeypatch):
    original_allocator = _allocate_bwd_d64_causal_gqa8_workspaces
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    assert _launch_bwd_d64_causal_gqa8_reduce is not None

    class Properties:
        gcnArchName = "gfx950:sramecc+:xnack-"
        multi_processor_count = 256

    batch, hq, hkv, sq, skv, head_dim = 4, 48, 6, 4096, 16384, 64
    q = torch.empty((batch, hq, sq, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="meta", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty((batch, hq, sq), device="meta", dtype=torch.float32)
    launches = []
    workspace = {}

    def allocate(q_arg, k_arg):
        lse_term, dk_part, dv_part = original_allocator(q_arg, k_arg)
        workspace.update(
            lse_term=lse_term,
            dk_part=dk_part,
            dv_part=dv_part,
        )
        return lse_term, dk_part, dv_part

    def launch_dq(*args):
        launches.append(("dq", args[6], args[7], args[-1]))

    def launch_producer(*args):
        launches.append(("producer", args[4], args[5], args[6], args[7], args[-1]))

    def launch_reducer(*args):
        launches.append(("reduce", args[0], args[1], args[2], args[3]))

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("selected signed GQA8 must not preprocess, convert, or use the retained producer")

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setitem(globals(), "_allocate_bwd_d64_causal_gqa8_workspaces", allocate)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_dq", launch_dq)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_gqa8_dkdv", launch_producer)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_causal_gqa8_reduce", launch_reducer)
    monkeypatch.setitem(globals(), "_run_bwd_preprocess", reject_legacy)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_fused_dq_convert", reject_legacy)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_dkdv", reject_legacy)

    outputs = fa_backward(q, k, v, o, do, lse, 0.125, True)
    assert [launch[0] for launch in launches] == ["dq", "producer", "reduce"]
    dq_launch, producer_launch, reduce_launch = launches
    dispatch = dq_launch[-1]
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.stat_mode == _D64_GQA_SIGNED
    assert dispatch.kv_splits == 4
    assert dq_launch[1] is producer_launch[2]
    assert dq_launch[2] is workspace["lse_term"] is producer_launch[1]
    assert producer_launch[3] is workspace["dk_part"] is reduce_launch[1]
    assert producer_launch[4] is workspace["dv_part"] is reduce_launch[2]
    partial_shape = (batch, hkv, _D64_CAUSAL_GQA8_KV_SPLITS, skv, head_dim)
    assert tuple(workspace["lse_term"].shape) == (batch, hq, sq)
    assert workspace["lse_term"].dtype == torch.float32
    for partial in (workspace["dk_part"], workspace["dv_part"]):
        assert tuple(partial.shape) == partial_shape
        assert partial.dtype == torch.bfloat16
        assert partial.is_contiguous()
    for output in outputs:
        assert output.dtype == torch.bfloat16
        assert output.is_contiguous()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_tiny_scale_retained_accuracy_gfx950(monkeypatch):
    shape = (8, 16, 2, 1024, 1024, 64)
    case = _make_d64_aten_case(
        shape,
        seed=307,
        causal=True,
        sm_scale=1e-38,
    )
    dispatches = []
    original_run = _run_bwd_d64

    def record_run(*args):
        dispatches.append(args[-1])
        return original_run(*args)

    def reject_selected_workspace(*_args, **_kwargs):
        raise AssertionError("tiny scale must use retained D64 workspaces")

    monkeypatch.setitem(globals(), "_run_bwd_d64", record_run)
    monkeypatch.setitem(
        globals(),
        "_allocate_bwd_d64_causal_gqa8_workspaces",
        reject_selected_workspace,
    )
    actual = fa_backward(*case.kernel_args)

    assert len(dispatches) == 1
    assert dispatches[0].family == "causal_m192"
    assert not dispatches[0].selected_causal
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        error_norm = torch.linalg.vector_norm((result.float() - expected.float()).double())
        reference_norm = torch.linalg.vector_norm(expected.double())
        assert reference_norm.item() > 0.0, name
        relative_l2 = error_norm / reference_norm
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_cyclic_analytic_accuracy_gfx950(monkeypatch):
    batch, hq, hkv, sequence, head_dim = 8, 64, 8, 16384, 64
    sm_scale = head_dim**-0.5
    counts = torch.arange(
        1,
        sequence + 1,
        device="cuda",
        dtype=torch.float64,
    )
    v_values = torch.linspace(
        -1.0,
        1.0,
        sequence,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    prefix_mean = (torch.cumsum(v_values.double(), dim=0) / counts).to(torch.bfloat16)

    q = torch.zeros(
        (batch, hq, sequence, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        (batch, hkv, sequence, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    do = torch.zeros_like(q)
    q[..., 0] = 1.0
    v[..., 0] = v_values
    o[..., 0] = prefix_mean
    do[..., 0] = 1.0
    lse = torch.log(counts).float()[None, None, :].expand(batch, hq, sequence).contiguous()

    dispatches = []
    producer_grids = []
    producer_kernel = _attn_bwd_dkdv_d64_causal_gqa8_kernel
    producer_kernel.device_caches.clear()
    original_select = _select_d64_dispatch_for_device

    def record_select(*args):
        dispatch = original_select(*args)
        dispatches.append(dispatch)
        return dispatch

    class RecordProducerGrid:

        def __getitem__(self, grid):
            producer_grids.append(grid)
            return producer_kernel[grid]

    monkeypatch.setitem(globals(), "_select_d64_dispatch_for_device", record_select)
    monkeypatch.setitem(
        globals(),
        "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
        RecordProducerGrid(),
    )
    dq, dk, dv = fa_backward(q, k, v, o, do, lse, sm_scale, True)
    torch.cuda.synchronize()

    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.cyclic_query_split
    assert dispatch.gqa_grid_mode == _D64_GQA_XCD_N_FAST
    assert dispatch.dkdv_lifetime == _D64_GQA_DIRECT_D64
    expected_grid = batch * hkv * 4 * triton.cdiv(sequence, 128)
    assert producer_grids == [(expected_grid, )]

    for name, result in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert torch.isfinite(result).all(), name
    assert torch.count_nonzero(dq).item() == 0
    assert torch.count_nonzero(dk[..., 1:]).item() == 0
    assert torch.count_nonzero(dv[..., 1:]).item() == 0

    inverse_counts = counts.reciprocal()
    harmonic_tail = torch.flip(
        torch.cumsum(torch.flip(inverse_counts, dims=(0, )), dim=0),
        dims=(0, ),
    )
    weighted_output_tail = torch.flip(
        torch.cumsum(
            torch.flip(prefix_mean.double() * inverse_counts, dims=(0, )),
            dim=0,
        ),
        dims=(0, ),
    )
    expected_dk = sm_scale * 8.0 * (v_values.double() * harmonic_tail - weighted_output_tail)
    expected_dv = 8.0 * harmonic_tail
    for name, result, expected in (
        ("dk", dk[..., 0], expected_dk),
        ("dv", dv[..., 0], expected_dv),
    ):
        expected = expected[None, None, :].expand_as(result)
        relative_l2 = torch.linalg.vector_norm(result.double() - expected)
        relative_l2 /= torch.linalg.vector_norm(expected)
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    objects = tuple(producer_kernel.device_caches[device][0].values())
    assert len(objects) == 1
    resources = _assert_d64_code_object_scratch_free("cyclic_analytic", objects[0])
    assert resources["lds_bytes"] == 36864
    assert not re.search(r"\b\w*atomic\w*\b", objects[0].asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("case_name", _D64_CAUSAL_GQA8_VALIDATION_CASES)
def test_d64_causal_gqa8_accuracy_gfx950(case_name):
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    batch, sq, skv, hq, hkv, head_dim, causal = _D64_VALIDATION_SHAPES[case_name]
    assert causal
    case = _make_d64_aten_case(
        (batch, hq, hkv, sq, skv, head_dim),
        seed=211 + _D64_CAUSAL_GQA8_VALIDATION_CASES.index(case_name),
        causal=True,
    )
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), (case_name, name)
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (case_name, name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_two_stage_m192_accuracy_gfx950():
    shape = (12, 8, 1, 1024, 3072, 64)
    case = _make_d64_aten_case(shape, seed=1203, causal=True)
    dispatch = _select_d64_dispatch_for_device(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        case.sm_scale,
        True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.owner_rows == 192

    kernel = _attn_bwd_dq_d64_causal_gqa8_kernel
    kernel.device_caches.clear()
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float())
        relative_l2 /= torch.linalg.vector_norm(expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    objects = tuple(kernel.device_caches[device][0].values())
    assert len(objects) == 1
    resources = _assert_d64_code_object_scratch_free("two_stage_m192", objects[0])
    assert resources["lds_bytes"] == 33536


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_odd_frontier_accuracy_gfx950():
    shape = (8, 64, 8, 1088, 1152, 64)
    batch, hq, hkv, sq, skv, _head_dim = shape
    dispatch = _select_d64_dispatch(
        (batch, hq, sq, 64),
        (batch, hkv, skv, 64),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    assert dispatch.dkdv_lifetime == _D64_GQA_INTERLEAVED_D32
    assert dispatch.gqa_grid_mode == _D64_GQA_XCD
    start_m_blk, _masked = _d64_causal_physical_frontier(128, sq, skv, 64, 128)
    assert start_m_blk == 1

    case = _make_d64_aten_case(shape, seed=293, causal=True)
    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    compiled_dispatch, obj = _compile_d64_causal_gqa8_producer_variant("odd_frontier", shape)
    assert compiled_dispatch == dispatch
    resources = _assert_d64_code_object_scratch_free("odd_frontier", obj)
    assert resources["lds_bytes"] == 33792
    assert not re.search(r"\b\w*atomic\w*\b", obj.asm["amdgcn"])


_D64_GQA8_COMPILED_VARIANTS = {}


def _compile_d64_causal_gqa8_producer_variant(name, shape, *, lifetime_mode=None):
    cache_key = (name, tuple(shape), lifetime_mode)
    cached = _D64_GQA8_COMPILED_VARIANTS.get(cache_key)
    if cached is not None:
        return cached
    batch, hq, hkv, sq, skv, head_dim = shape
    q = torch.empty((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    do = torch.empty_like(q)
    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(q, k)
    delta = torch.empty_like(lse_term)
    dispatch = _select_d64_dispatch(
        tuple(q.shape),
        tuple(k.shape),
        True,
        arch="gfx950:sramecc+:xnack-",
        cu_count=256,
        sm_scale=0.125,
        bases_aligned_16=True,
    )
    assert dispatch.family == "causal_scheduled_gqa8"
    if lifetime_mode is not None:
        dispatch = dataclasses.replace(dispatch, dkdv_lifetime=lifetime_mode)
    _attn_bwd_dkdv_d64_causal_gqa8_kernel.device_caches.clear()
    _launch_bwd_d64_causal_gqa8_dkdv(
        q,
        k,
        v,
        do,
        lse_term,
        delta,
        dk_part,
        dv_part,
        0.125,
        dispatch,
    )
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    objects = tuple(_attn_bwd_dkdv_d64_causal_gqa8_kernel.device_caches[device][0].values())
    assert len(objects) == 1, (name, len(objects))
    result = dispatch, objects[0]
    _D64_GQA8_COMPILED_VARIANTS[cache_key] = result
    return result


def test_d64_causal_gqa8_compiled_variant_cache_identity(monkeypatch):
    compiled_objects = []

    class FakeProducerKernel:

        def __init__(self):
            self.device_caches = {}

        def __getitem__(self, _grid):

            def launch(*_args, **_kwargs):
                compiled = object()
                compiled_objects.append(compiled)
                self.device_caches[0] = ({"variant": compiled}, )

            return launch

    real_empty = torch.empty

    def meta_empty(*args, **kwargs):
        kwargs["device"] = "meta"
        return real_empty(*args, **kwargs)

    monkeypatch.setitem(globals(), "_D64_GQA8_COMPILED_VARIANTS", {})
    monkeypatch.setitem(
        globals(),
        "_attn_bwd_dkdv_d64_causal_gqa8_kernel",
        FakeProducerKernel(),
    )
    monkeypatch.setattr(torch, "empty", meta_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    first_shape = (4, 48, 6, 4096, 4096, 64)
    second_shape = (4, 48, 6, 4096, 8192, 64)
    first = _compile_d64_causal_gqa8_producer_variant("shared", first_shape)
    repeated = _compile_d64_causal_gqa8_producer_variant("shared", first_shape)
    different_shape = _compile_d64_causal_gqa8_producer_variant("shared", second_shape)
    different_lifetime = _compile_d64_causal_gqa8_producer_variant(
        "shared",
        first_shape,
        lifetime_mode=_D64_GQA_INTERLEAVED_D32,
    )

    assert repeated is first
    assert different_shape is not first
    assert different_lifetime is not first
    assert different_lifetime is not different_shape
    assert len(compiled_objects) == 3


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_lifetime_codegen_gfx950():
    assert _attn_bwd_dkdv_d64_causal_gqa8_kernel is not None
    same_shape = (4, 48, 6, 4096, 4096, 64)
    d32_objects = {}
    for lifetime in (
            _D64_GQA_INDEPENDENT_D32,
            _D64_GQA_INTERLEAVED_D32,
    ):
        dispatch, obj = _compile_d64_causal_gqa8_producer_variant(
            f"same_shape_{lifetime}",
            same_shape,
            lifetime_mode=lifetime,
        )
        assert dispatch.dkdv_lifetime == lifetime
        assert dispatch.gqa_grid_mode == _D64_GQA_XCD
        assert not dispatch.cyclic_query_split
        ttgir = obj.asm.get("ttgir", "")
        assert ttgir, lifetime
        assert "tensor<128x32xf32" in ttgir, lifetime
        resource = _assert_d64_code_object_scratch_free(lifetime, obj)
        assert resource["lds_bytes"] == 33792
        d32_objects[lifetime] = obj
    independent = d32_objects[_D64_GQA_INDEPENDENT_D32]
    interleaved = d32_objects[_D64_GQA_INTERLEAVED_D32]
    assert independent.asm["ttgir"] != interleaved.asm["ttgir"]
    assert independent.asm["amdgcn"] != interleaved.asm["amdgcn"]

    direct_dispatch, direct = _compile_d64_causal_gqa8_producer_variant(
        "gqa8_square_16k_direct_d64",
        (2, 32, 4, 16384, 16384, 64),
    )
    assert direct_dispatch.dkdv_lifetime == _D64_GQA_DIRECT_D64
    assert _d64_causal_gqa8_batch_stats4(16384, 16384, direct_dispatch)
    assert "tensor<128x32xf32" not in direct.asm.get("ttgir", "")
    direct_resource = _assert_d64_code_object_scratch_free("direct", direct)
    assert direct_resource["lds_bytes"] == 70656

    source = _attn_bwd_dkdv_d64_causal_gqa8_kernel.src
    assert "_d64_gqa8_direct_d64_impl" in source
    assert "_d64_gqa8_d32_impl" in source
    assert "LIFETIME_MODE == _D64_GQA_DIRECT_D64_JIT" in source
    assert "INTERLEAVED_D32" in source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_reducer_order_left_associated_fp32_gfx950():
    dk_values_cpu = torch.tensor([0.5, 1.0, 2**24, -(2**24)], dtype=torch.bfloat16)
    dv_values_cpu = torch.tensor([0.5, 2**24, -1.0, -(2**24)], dtype=torch.bfloat16)
    assert dk_values_cpu[3].item() != 0
    assert dv_values_cpu[3].item() != 0

    def fp32_add(left, right):
        return torch.add(left.to(torch.float32), right.to(torch.float32))

    def reference_orders(values):
        left = fp32_add(
            fp32_add(fp32_add(values[0], values[1]), values[2]),
            values[3],
        )
        balanced = fp32_add(
            fp32_add(values[0], values[1]),
            fp32_add(values[2], values[3]),
        )
        alternate_pairs = fp32_add(
            fp32_add(values[0], values[2]),
            fp32_add(values[1], values[3]),
        )
        reordered = fp32_add(
            fp32_add(fp32_add(values[0], values[2]), values[1]),
            values[3],
        )
        reverse = fp32_add(
            fp32_add(fp32_add(values[3], values[2]), values[1]),
            values[0],
        )
        return left, (balanced, alternate_pairs, reordered, reverse)

    dk_expected, dk_forbidden = reference_orders(dk_values_cpu)
    dv_expected, dv_forbidden = reference_orders(dv_values_cpu)
    dk_expected = dk_expected.to(torch.bfloat16)
    dv_expected = dv_expected.to(torch.bfloat16)
    assert dk_expected.item() == 2.0
    assert dv_expected.item() == -1.0
    assert all(not torch.equal(dk_expected, result.to(torch.bfloat16)) for result in dk_forbidden)
    assert all(not torch.equal(dv_expected, result.to(torch.bfloat16)) for result in dv_forbidden)

    dk_values = dk_values_cpu.to(device="cuda")
    dv_values = dv_values_cpu.to(device="cuda")
    dk_part = (dk_values.view(1, 1, 4, 1, 1).expand(1, 1, 4, 128, 64).contiguous())
    dv_part = (dv_values.view(1, 1, 4, 1, 1).expand(1, 1, 4, 128, 64).contiguous())
    dk = torch.full((1, 1, 128, 64), 7, device="cuda", dtype=torch.bfloat16)
    dv = torch.full_like(dk, -7)
    _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
    torch.cuda.synchronize()

    assert dk.dtype == dv.dtype == torch.bfloat16
    assert torch.equal(dk, torch.full_like(dk, dk_expected.item()))
    assert torch.equal(dv, torch.full_like(dv, dv_expected.item()))

    reducer_source = _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.src
    reducer_ast = ast.parse(reducer_source).body[0]

    def accumulator_chain(name):
        return [
            ast.unparse(statement.value)
            for statement in reducer_ast.body
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == name
        ]

    assert accumulator_chain("dk_acc") == [
        "tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)",
    ]
    assert accumulator_chain("dv_acc") == [
        "tlx.zeros((BLOCK_N, D), tl.float32, layout=out_layout)",
    ]
    reduction_loops = [statement for statement in reducer_ast.body if isinstance(statement, ast.For)]
    assert len(reduction_loops) == 1
    reduction_loop = reduction_loops[0]
    assert ast.unparse(reduction_loop.target) == "split"
    assert ast.unparse(reduction_loop.iter) == "tl.static_range(0, KV_SPLITS)"
    accumulator_updates = [(ast.unparse(statement.target), ast.unparse(statement.value))
                           for statement in reduction_loop.body
                           if isinstance(statement, ast.AugAssign) and isinstance(statement.op, ast.Add)]
    assert accumulator_updates == [
        ("dk_acc", "dk_part.to(tl.float32)"),
        ("dv_acc", "dv_part.to(tl.float32)"),
    ]
    assert reducer_source.count("dk_acc.to(tl.bfloat16)") == 1
    assert reducer_source.count("dv_acc.to(tl.bfloat16)") == 1
    for split in range(4):
        assert f"dk_split{split}.to(tl.bfloat16)" not in reducer_source
        assert f"dv_split{split}.to(tl.bfloat16)" not in reducer_source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_reducer_determinism_gfx950():
    assert _launch_bwd_d64_causal_gqa8_reduce is not None
    generator = torch.Generator(device="cuda")
    generator.manual_seed(271)
    dk_part = torch.randn(
        (1, 2, 4, 256, 64),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dv_part = torch.randn(
        (1, 2, 4, 256, 64),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    baseline = None
    for run in range(20):
        dk = torch.full(
            (1, 2, 256, 64),
            run + 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        dv = torch.full_like(dk, -(run + 1))
        _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
        torch.cuda.synchronize()
        current = (dk.clone(), dv.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run
    assert torch.isfinite(baseline[0]).all()
    assert torch.isfinite(baseline[1]).all()


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_end_to_end_determinism_gfx950():
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    case = _make_d64_aten_case((4, 48, 6, 4096, 4096, 64), seed=277, causal=True)
    baseline = None
    for run in range(5):
        _dq, dk, dv = fa_backward(*case.kernel_args)
        torch.cuda.synchronize()
        current = (dk.clone(), dv.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 32, 4, 16384, 16384, 64), id="square"),
        pytest.param((4, 48, 6, 4096, 8192, 64), id="rectangular"),
    ],
)
def test_d64_causal_gqa8_stats4_producer_determinism_gfx950(shape):
    assert _launch_bwd_d64_causal_gqa8_dkdv is not None
    case = _make_d64_aten_case(shape, seed=281, causal=True)
    dispatch = _select_d64_dispatch_for_device(*case.kernel_args[:-2], case.sm_scale, case.causal)
    assert _d64_causal_gqa8_batch_stats4(case.q.shape[2], case.k.shape[2], dispatch)

    lse_term, dk_part, dv_part = _allocate_bwd_d64_causal_gqa8_workspaces(case.q, case.k)
    delta = torch.empty_like(lse_term)
    dq = torch.empty_like(case.q)
    _launch_bwd_d64_causal_dq(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        delta,
        lse_term,
        dq,
        case.sm_scale,
        dispatch,
    )

    baseline = None
    for run in range(3):
        _launch_bwd_d64_causal_gqa8_dkdv(
            case.q,
            case.k,
            case.v,
            case.do,
            lse_term,
            delta,
            dk_part,
            dv_part,
            case.sm_scale,
            dispatch,
        )
        torch.cuda.synchronize()
        current = (dk_part.clone(), dv_part.clone())
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(current[0], baseline[0]), run
            assert torch.equal(current[1], baseline[1]), run


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_codegen_gfx950():
    assert _attn_bwd_dkdv_d64_causal_gqa8_kernel is not None
    assert _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel is not None
    producer_variants = {
        "square_xcd": (
            (2, 32, 4, 16384, 16384, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "short_square_xcd": (
            (4, 48, 6, 4096, 4096, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "rectangle_xcd": (
            (4, 48, 6, 4096, 16384, 64),
            _D64_GQA_XCD,
            False,
            _D64_GQA_DIRECT_D64,
        ),
        "cyclic": (
            (8, 64, 8, 16384, 16384, 64),
            _D64_GQA_XCD_N_FAST,
            True,
            _D64_GQA_DIRECT_D64,
        ),
    }
    resources = {}
    for name, (shape, expected_mode, expected_cyclic, expected_lifetime) in producer_variants.items():
        dispatch, obj = _compile_d64_causal_gqa8_producer_variant(name, shape)
        resources[name] = _assert_d64_code_object_scratch_free(name, obj)
        if _d64_causal_gqa8_batch_stats4(shape[3], shape[4], dispatch):
            expected_lds_bytes = 70656
        elif expected_cyclic:
            expected_lds_bytes = 36864
        else:
            expected_lds_bytes = 33792
        assert resources[name]["lds_bytes"] == expected_lds_bytes, (name, resources[name])
        if expected_lifetime == _D64_GQA_DIRECT_D64:
            assert resources[name]["agpr_count"] == 0, (name, resources[name])
        assert dispatch.gqa_grid_mode == expected_mode, name
        assert dispatch.cyclic_query_split is expected_cyclic, name
        assert dispatch.dkdv_lifetime == expected_lifetime, name
        amdgcn = obj.asm["amdgcn"]
        assert not re.search(r"\b\w*atomic\w*\b", amdgcn), name
        assert re.search(r"\bbuffer_store_dwordx4\b", amdgcn), name

    dk_part = torch.empty((1, 1, 4, 256, 64), device="cuda", dtype=torch.bfloat16)
    dv_part = torch.empty_like(dk_part)
    dk = torch.empty((1, 1, 256, 64), device="cuda", dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.device_caches.clear()
    _launch_bwd_d64_causal_gqa8_reduce(dk_part, dv_part, dk, dv)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    reducer_objects = tuple(_attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.device_caches[device][0].values())
    assert len(reducer_objects) == 1
    reducer = reducer_objects[0]
    resources["reducer"] = _assert_d64_code_object_scratch_free("reducer", reducer)
    assert resources["reducer"]["lds_bytes"] == 0
    reducer_asm = reducer.asm["amdgcn"]
    assert not re.search(r"\b\w*atomic\w*\b", reducer_asm)
    assert re.search(r"\bbuffer_store_dwordx4\b", reducer_asm)

    producer_source = _attn_bwd_dkdv_d64_causal_gqa8_kernel.src
    consume_source = _d64_gqa8_d32_consume.src
    assert "tlx.buffer_load(K" in producer_source
    assert "tlx.buffer_load(V" in producer_source
    assert "qdo_stages: tl.constexpr = 4 if BATCH_STATS4 else 2" in producer_source
    assert "tlx.local_alloc((BLOCK_M, D), tl.bfloat16, qdo_stages" in producer_source
    assert "n0 + BLOCK_N - 1 > m_blk * BLOCK_M + (SKV - SQ)" in consume_source
    assert "LSE_MODE == _D64_LSE_NEG_LOG2E_JIT" in producer_source
    assert "DELTA_MODE == _D64_DELTA_NEGATED_JIT" in producer_source
    assert "atomic" not in producer_source.lower()

    reducer_source = _attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel.src
    assert "for split in tl.static_range(0, KV_SPLITS)" in reducer_source
    assert "dk_acc += dk_part.to(tl.float32)" in reducer_source
    assert "dv_acc += dv_part.to(tl.float32)" in reducer_source
    assert "dk_acc.to(tl.bfloat16)" in reducer_source
    assert "dv_acc.to(tl.bfloat16)" in reducer_source


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_n256_gqa8_correctness_and_partials_gfx950(monkeypatch):
    case = _make_d64_aten_case((1, 16, 2, 4096, 4096, 64), seed=53)
    launches = []
    original_producer = _launch_bwd_d64_fused_n256
    original_convert = _launch_bwd_d64_fused_dq_convert
    original_reduce = _launch_bwd_d64_kv_reduce

    def record_producer(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_acc,
        dk_owner,
        dv_owner,
        sm_scale,
        dispatch,
    ):
        launches.append((
            "producer",
            tuple(dk_owner.shape),
            tuple(dv_owner.shape),
            dk_owner.dtype,
            dv_owner.dtype,
            dispatch.kv_splits,
        ))
        return original_producer(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_acc,
            dk_owner,
            dv_owner,
            sm_scale,
            dispatch,
        )

    def record_convert(dq_acc, dq):
        launches.append(("convert", dq_acc.dtype, dq.dtype))
        return original_convert(dq_acc, dq)

    def record_reduce(dk_part, dv_part, dk, dv, dispatch):
        launches.append((
            "reduce",
            tuple(dk_part.shape),
            tuple(dv_part.shape),
            dk_part.dtype,
            dv_part.dtype,
            dispatch.kv_splits,
        ))
        return original_reduce(dk_part, dv_part, dk, dv, dispatch)

    monkeypatch.setitem(globals(), "_launch_bwd_d64_fused_n256", record_producer)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_fused_dq_convert", record_convert)
    monkeypatch.setitem(globals(), "_launch_bwd_d64_kv_reduce", record_reduce)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_d64_fused_n256_kernel.device_caches.clear()
    _attn_bwd_d64_fused_dq_convert_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    actual = fa_backward(*case.kernel_args)

    partial_shape = (1, 2, 8, 4096, 64)
    assert launches == [
        (
            "producer",
            partial_shape,
            partial_shape,
            torch.bfloat16,
            torch.bfloat16,
            8,
        ),
        ("convert", torch.float32, torch.bfloat16),
        (
            "reduce",
            partial_shape,
            partial_shape,
            torch.bfloat16,
            torch.bfloat16,
            8,
        ),
    ]
    producer_source = _attn_bwd_d64_fused_n256_kernel.src
    assert "pid_split = pid_hq % group_size" in producer_source
    assert ("((pid_b * HKV + pid_hkv) * KV_SPLITS + pid_split)" in producer_source)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    assert (device not in _attn_bwd_dq_d64_direct_kernel.device_caches
            or not _attn_bwd_dq_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_direct_kernel.device_caches
            or not _attn_bwd_dkdv_d64_direct_kernel.device_caches[device][0])
    code_objects = {
        "producer": tuple(_attn_bwd_d64_fused_n256_kernel.device_caches[device][0].values()),
        "convert": tuple(_attn_bwd_d64_fused_dq_convert_kernel.device_caches[device][0].values()),
        "reduce": tuple(_attn_bwd_dkdv_d64_reduce_kernel.device_caches[device][0].values()),
    }
    assert all(len(objects) == 1 for objects in code_objects.values())
    for name, objects in code_objects.items():
        _assert_d64_code_object_scratch_free(name, objects[0])
    producer_asm = code_objects["producer"][0].asm["amdgcn"]
    assert re.search(r"\bbuffer_atomic_add_f32\b", producer_asm)
    assert "buffer_atomic_pk_add_bf16" not in producer_asm


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_fused_n256_mha_correctness_and_codegen_gfx950():
    case = _make_d64_aten_case((1, 16, 16, 4096, 4096, 64), seed=47)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_d64_fused_n256_kernel.device_caches.clear()
    _attn_bwd_d64_fused_dq_convert_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    actual = fa_backward(*case.kernel_args)

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    device = torch.cuda.current_device()
    assert (device not in _attn_bwd_dq_d64_direct_kernel.device_caches
            or not _attn_bwd_dq_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_direct_kernel.device_caches
            or not _attn_bwd_dkdv_d64_direct_kernel.device_caches[device][0])
    assert (device not in _attn_bwd_dkdv_d64_reduce_kernel.device_caches
            or not _attn_bwd_dkdv_d64_reduce_kernel.device_caches[device][0])
    producer = tuple(_attn_bwd_d64_fused_n256_kernel.device_caches[device][0].values())
    convert = tuple(_attn_bwd_d64_fused_dq_convert_kernel.device_caches[device][0].values())
    assert len(producer) == len(convert) == 1
    for name, obj in (("producer", producer[0]), ("convert", convert[0])):
        _assert_d64_code_object_scratch_free(name, obj)
    producer_asm = producer[0].asm["amdgcn"]
    assert re.search(r"\bbuffer_atomic_add_f32\b", producer_asm)
    assert "buffer_atomic_pk_add_bf16" not in producer_asm


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal", "expected"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False, 1),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), False, 8),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True, 4),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True, 4),
    ],
)
def test_d64_gqa_kv_split_policy(q_shape, k_shape, causal, expected):
    assert _select_d64_dispatch(q_shape, k_shape, causal).kv_splits == expected


@pytest.mark.parametrize(
    ("owner_start", "owner_rows", "sq", "skv", "block_n", "expected"),
    [
        (0, 192, 4096, 4096, 32, 6),
        (192, 192, 4096, 4096, 32, 12),
        (0, 192, 4096, 16384, 64, 195),
        (3840, 192, 4096, 4096, 32, 126),
    ],
)
def test_d64_causal_dq_compact_key_frontier(owner_start, owner_rows, sq, skv, block_n, expected):
    assert _d64_causal_dq_key_blocks(owner_start, owner_rows, sq, skv, block_n) == expected


@pytest.mark.parametrize(
    ("key_start", "sq", "skv", "block_m", "expected"),
    [
        (0, 4096, 4096, 64, 0),
        (256, 4096, 4096, 64, 4),
        (12288, 4096, 16384, 64, 0),
        (16320, 4096, 16384, 64, 63),
    ],
)
def test_d64_causal_dkdv_compact_query_frontier(key_start, sq, skv, block_m, expected):
    assert _d64_causal_dkdv_first_query_block(key_start, sq, skv, block_m) == expected


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal"),
    [
        ((2, 32, 16384, 64), (2, 32, 16384, 64), False),
        ((2, 32, 16384, 64), (2, 4, 16384, 64), True),
        ((4, 48, 4096, 64), (4, 6, 4096, 64), True),
        ((4, 48, 4096, 64), (4, 6, 16384, 64), True),
    ],
)
def test_d64_launch_uses_structural_dq_owner(monkeypatch, q_shape, k_shape, causal):

    class LaunchRecorder:

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):

            def record(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return record

    dq_launch = LaunchRecorder()
    dkdv_launch = LaunchRecorder()
    reduce_launch = LaunchRecorder()
    monkeypatch.setitem(globals(), "_attn_bwd_dq_d64_direct_kernel", dq_launch)
    monkeypatch.setitem(globals(), "_attn_bwd_dkdv_d64_direct_kernel", dkdv_launch)
    monkeypatch.setitem(globals(), "_attn_bwd_dkdv_d64_reduce_kernel", reduce_launch)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    k = torch.empty(k_shape, device="meta", dtype=torch.bfloat16)
    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)

    _run_bwd_d64_direct(q, k, k, q, object(), object(), q, k, k, 0.125, causal, dispatch)

    assert len(dq_launch.calls) == 1
    grid, _args, kwargs = dq_launch.calls[0]
    assert grid == (triton.cdiv(q_shape[2], dispatch.owner_rows), q_shape[1], q_shape[0])
    assert kwargs["OWNER_ROWS"] == dispatch.owner_rows
    assert kwargs["BLOCK_N"] == dispatch.key_rows
    assert len(dkdv_launch.calls) == 1
    dkdv_grid, _args, dkdv_kwargs = dkdv_launch.calls[0]
    assert dkdv_grid == (
        triton.cdiv(k_shape[2], 64),
        k_shape[1] * dispatch.kv_splits,
        k_shape[0],
    )
    assert dkdv_kwargs["KV_SPLITS"] == dispatch.kv_splits
    assert dkdv_kwargs["BLOCK_N"] == 64
    assert len(reduce_launch.calls) == (1 if dispatch.kv_splits > 1 else 0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("q_shape", "k_shape", "causal"),
    [
        pytest.param((1, 1, 256, 64), (1, 1, 256, 64), False, id="noncausal-n256-mha"),
        pytest.param((1, 8, 256, 64), (1, 1, 256, 64), False, id="noncausal-n256-gqa8"),
        pytest.param((1, 1, 256, 64), (1, 1, 256, 64), True, id="causal-m192-square"),
        pytest.param((1, 8, 256, 64), (1, 1, 256, 64), True, id="causal-m192-square-gqa8"),
        pytest.param((1, 8, 256, 64), (1, 1, 512, 64), True, id="causal-m192-rect-gqa8"),
        pytest.param((1, 1, 16384, 64), (1, 1, 16384, 64), True, id="causal-m256-deep"),
        pytest.param((1, 8, 16384, 64), (1, 1, 16384, 64), True, id="causal-m256-deep-gqa8"),
    ],
)
def test_d64_retained_specializations_are_scratch_free_gfx950(q_shape, k_shape, causal):
    q = torch.zeros(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(k_shape, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(q_shape[:-1], device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    _attn_bwd_dq_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_direct_kernel.device_caches.clear()
    _attn_bwd_dkdv_d64_reduce_kernel.device_caches.clear()

    dispatch = _select_d64_dispatch(q_shape, k_shape, causal)
    _run_bwd_d64_direct(q, k, k, q, lse, delta, dq, dk, dv, 0.125, causal, dispatch)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    kernels = [
        ("dq", _attn_bwd_dq_d64_direct_kernel),
        ("dkdv", _attn_bwd_dkdv_d64_direct_kernel),
    ]
    if dispatch.kv_splits > 1:
        kernels.append(("reduce", _attn_bwd_dkdv_d64_reduce_kernel))
    for name, kernel in kernels:
        compiled = tuple(kernel.device_caches[device][0].values())
        assert len(compiled) == 1, (name, len(compiled))
        _assert_d64_code_object_scratch_free(name, compiled[0])
