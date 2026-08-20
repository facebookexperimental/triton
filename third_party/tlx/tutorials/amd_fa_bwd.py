"""BF16 Flash-Attention backward kernel families for AMD gfx950.

The public ``fa_backward`` wrapper supports dense contiguous BF16 D64 tensors
with matching positive batches, Hkv dividing Hq, SQ/SKV at least 256 and
aligned to 64, SQ no larger than 8,388,544, and SKV no larger than 16,777,152.
Bottom-right causal D64 additionally requires SQ no larger than SKV. It also
supports the two validated equal-head configurations in ``SUPPORTED_SHAPES``
and the causal/non-causal HipKittens GQA contract: positive B/Hq/Hkv, Hkv dividing
Hq, N a positive multiple of 256, and D=128. ``GQA_BENCHMARK_SHAPES`` records
the published performance series; it is not an allow-list. Other
configurations are not part of this submission's public contract yet. Focused
D64 tests live in ``python/test/unit/language/test_tlx_amd_fa_bwd.py``.

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

import dataclasses
import math
import os

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
        lse_view = tlx.local_slice(lse_all, [local_head, 0], [1, BLOCK_M])
        delta_view = tlx.local_slice(delta_all, [local_head, 0], [1, BLOCK_M])
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
