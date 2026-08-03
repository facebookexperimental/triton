"""BF16 Flash-Attention backward kernel families for AMD gfx950.

The public ``fa_backward`` wrapper supports the two validated equal-head
configurations in ``SUPPORTED_SHAPES`` and the non-causal HipKittens GQA
contract: positive B/Hq/Hkv, Hkv dividing Hq, N a positive multiple of 256,
and D=128. ``GQA_BENCHMARK_SHAPES`` records the published performance series;
it is not an allow-list. Other configurations are not part of this
submission's public contract yet. Run this file with pytest for correctness.

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


def _make_gqa_smoke_case(shape=(1, 8, 1, 512, 128), seed=0):
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
    sm_scale = head_dim**-0.5
    group_size = hq // hk

    for batch_idx in range(batch):
        for query_head in range(hq):
            kv_head = query_head // group_size
            q_ref = q[batch_idx, query_head].float().requires_grad_(True)
            k_ref = k[batch_idx, kv_head].float().requires_grad_(True)
            v_ref = v[batch_idx, kv_head].float().requires_grad_(True)
            scores = torch.matmul(q_ref, k_ref.transpose(0, 1)) * sm_scale
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

    return ReferenceCase(
        q,
        k,
        v,
        o,
        do,
        lse,
        sm_scale,
        False,
        (dq, dk, dv),
    )


@triton.jit
def _attn_bwd_preprocess_kernel(O, DO, Delta, N: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
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


def _run_bwd_preprocess(o, do, delta):
    batch, heads, n_ctx, head_dim = o.shape
    block_m = 64
    grid = (triton.cdiv(n_ctx, block_m), batch * heads)
    _attn_bwd_preprocess_kernel[grid](o, do, delta, N=n_ctx, D=head_dim, BLOCK_M=block_m, num_warps=4)


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
    off_z,
    off_h_kv,
    step,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUTER_M: tl.constexpr,
    ASYNC_LAYOUT: tl.constexpr,
    STATS_ASYNC_LAYOUT: tl.constexpr,
):
    """Issue one valid 64-row Q/dO/stats tile as two async groups."""
    group_size: tl.constexpr = HQ // HK
    n_outer_blocks: tl.constexpr = N // OUTER_M
    total_outer_steps: tl.constexpr = group_size * n_outer_blocks
    refill_step = step % total_outer_steps
    group_idx = refill_step // n_outer_blocks
    outer_block = refill_step % n_outer_blocks
    off_h_q = off_h_kv * group_size + group_idx

    outer_slice = tl.arange(0, OUTER_M // BLOCK_M)
    inner_m = tlx.amd_rematerialized_range(0, BLOCK_M, 10, placement=step)
    offs_d = tlx.amd_rematerialized_range(0, D, 11, placement=step)
    offs_m = (outer_block * OUTER_M + outer_slice[:, None] * BLOCK_M + inner_m[None, :])
    q_base = (off_z * HQ + off_h_q) * N * D
    qdo_offsets = (q_base + offs_m[:, :, None] * D + offs_d[None, None, :]).to(tl.int32)
    qdo_offsets = tlx.require_layout(qdo_offsets, ASYNC_LAYOUT, pin=False)
    stats_base = (off_z * HQ + off_h_q) * N
    stats_m = tl.arange(0, OUTER_M)
    stats_offsets = tlx.require_layout(
        (stats_base + outer_block * OUTER_M + stats_m).to(tl.int32),
        STATS_ASYNC_LAYOUT,
        pin=False,
    )
    q_token = tlx.buffer_load_to_local(q_outer, Q, qdo_offsets)
    tlx.async_load_commit_group([q_token])
    lse_token = tlx.buffer_load_to_local(lse_outer, LSE, stats_offsets)
    delta_token = tlx.buffer_load_to_local(delta_outer, Delta, stats_offsets)
    do_token = tlx.buffer_load_to_local(do_outer, DO, qdo_offsets)
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
    lhs0 = tlx.amd_extract_slice(dv_lhs, [128, 16], [0, 0])
    lhs1 = tlx.amd_extract_slice(dv_lhs, [128, 16], [128, 0])
    rhs0 = tlx.amd_extract_slice(dv_rhs, [16, 32], [0, 0])
    rhs1 = tlx.amd_extract_slice(dv_rhs, [16, 32], [0, 32])
    rhs2 = tlx.amd_extract_slice(dv_rhs, [16, 32], [0, 64])
    rhs3 = tlx.amd_extract_slice(dv_rhs, [16, 32], [0, 96])

    c00 = tlx.amd_extract_slice(dv, [128, 32], [0, 0])
    c10 = tlx.amd_extract_slice(dv, [128, 32], [128, 0])
    c01 = tlx.amd_extract_slice(dv, [128, 32], [0, 32])
    c11 = tlx.amd_extract_slice(dv, [128, 32], [128, 32])
    c02 = tlx.amd_extract_slice(dv, [128, 32], [0, 64])
    c12 = tlx.amd_extract_slice(dv, [128, 32], [128, 64])
    c03 = tlx.amd_extract_slice(dv, [128, 32], [0, 96])
    c13 = tlx.amd_extract_slice(dv, [128, 32], [128, 96])

    if FIRST_HALF:
        c00 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs0,
            c00,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c10 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs0,
            c10,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c01 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs1,
            c01,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c11 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs1,
            c11,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
    else:
        c02 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs2,
            c02,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c12 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs2,
            c12,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c03 = tlx.amd_scheduled_mfma(
            lhs0,
            rhs3,
            c03,
            accumulator="persistent",
            accumulator_register_class="vgpr",
        )
        c13 = tlx.amd_scheduled_mfma(
            lhs1,
            rhs3,
            c13,
            accumulator="persistent",
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
    off_z,
    off_h_kv,
    pid_n,
    step,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MMA_NM: tl.constexpr,
    MMA_ND: tl.constexpr,
    K_NM_LAYOUT: tl.constexpr,
    QT_LAYOUT: tl.constexpr,
    P_ND_LAYOUT: tl.constexpr,
    Q_OUT_LAYOUT: tl.constexpr,
):
    """Compute score, dP, dS and dV, publishing current dS to LDS."""
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
    scores = tlx.amd_mfma(
        k_nm,
        q_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    scores = tlx.amd_register_resident(scores, register_class="vgpr", registers_per_group=16)
    do_t = tlx.local_load(
        tlx.local_trans(do_slice),
        layout=QT_LAYOUT,
        relaxed=True,
    )
    dp = tlx.amd_mfma(
        v_operand,
        do_t,
        tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=MMA_NM),
    )
    dp = tlx.amd_register_resident(dp, register_class="vgpr", registers_per_group=16)

    q_out = tlx.local_load(q_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    q_out = tlx.amd_register_resident(q_out, register_class="vgpr", registers_per_group=4)
    do_out = tlx.local_load(do_slice, layout=Q_OUT_LAYOUT, relaxed=True)
    # Keep LSE scaling on an independent scalar VALU chain.  Broadcasting the
    # multiply lets LLVM pair an LSE lane with a score fragment in a packed
    # multiply and creates a false cross-fragment dependency.
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
    p = tl.math.exp2(scores * scale_full - lse_full)
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
    off_z,
    off_h_kv,
    step,
    SM_SCALE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_MD: tl.constexpr,
):
    """Accumulate one dQ partial in the native MFMA ownership."""
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
    local_m = tlx.amd_rematerialized_range(0, BLOCK_M, 14, placement=step)
    offs_d = tlx.amd_rematerialized_range(0, D, 13, placement=step)
    d_swizzled = ((offs_d & 1)
                  | ((offs_d & 2) << 6)
                  | ((offs_d & 12) << 3)
                  | ((offs_d & 48) << 5)
                  | ((offs_d & 64) << 2))
    swizzled = (start_m * D + ((local_m[:, None] << 1) | d_swizzled[None, :])).to(tl.int32)
    swizzled = tl.max_contiguous(swizzled, [1, 2])
    swizzled = tlx.require_layout(swizzled, MMA_MD, pin=False)
    dq_base = (off_z * HQ + off_h_q) * N * D
    tlx.buffer_atomic_add(
        DQ_ACC + dq_base,
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
    )
    k7 = tlx.local_load(
        tlx.local_slice(k_buffer, [224, 0], [32, 128]),
        layout=K_MD_LAYOUT,
        relaxed=True,
    )
    k00 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [0, 0])
    k01 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [0, 64])
    k10 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [32, 0])
    k11 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [32, 64])
    dq0 = tlx.amd_scheduled_mfma(
        ds0,
        k00,
        dq0,
        resident_operand=1,
        accumulator="transient",
        initialize=True,
    )
    dq1 = tlx.amd_scheduled_mfma(
        ds0,
        k01,
        dq1,
        resident_operand=1,
        accumulator="transient",
        initialize=True,
    )

    ds3 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 96], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    dq0 = tlx.amd_scheduled_mfma(ds1, k10, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds1, k11, dq1, resident_operand=1, accumulator="transient")
    k20 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [64, 0])
    k21 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [64, 64])
    dq0 = tlx.amd_scheduled_mfma(ds2, k20, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds2, k21, dq1, resident_operand=1, accumulator="transient")

    ds4 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 128], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    k30 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [96, 0])
    k31 = tlx.amd_extract_slice(k_resident_lo, [32, 64], [96, 64])
    dq0 = tlx.amd_scheduled_mfma(ds3, k30, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds3, k31, dq1, resident_operand=1, accumulator="transient")

    ds5 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 160], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    k40 = tlx.amd_extract_slice(k_resident_mid, [32, 64], [0, 0])
    k41 = tlx.amd_extract_slice(k_resident_mid, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds4, k40, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds4, k41, dq1, resident_operand=1, accumulator="transient")

    ds6 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 192], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    k50 = tlx.amd_extract_slice(k_resident_mid, [32, 64], [32, 0])
    k51 = tlx.amd_extract_slice(k_resident_mid, [32, 64], [32, 64])
    dq0 = tlx.amd_scheduled_mfma(ds5, k50, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds5, k51, dq1, resident_operand=1, accumulator="transient")

    ds7 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 224], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    k60 = tlx.amd_extract_slice(k_resident_band6, [32, 64], [0, 0])
    k61 = tlx.amd_extract_slice(k_resident_band6, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds6, k60, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds6, k61, dq1, resident_operand=1, accumulator="transient")
    k70 = tlx.amd_extract_slice(k7, [32, 64], [0, 0])
    k71 = tlx.amd_extract_slice(k7, [32, 64], [0, 64])
    dq0 = tlx.amd_scheduled_mfma(ds7, k70, dq0, resident_operand=1, accumulator="transient")
    dq1 = tlx.amd_scheduled_mfma(ds7, k71, dq1, resident_operand=1, accumulator="transient")
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
):
    """Drain entry when no dK work is available to hide the first dS reads."""
    ds0 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 0], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    ds1 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 32], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
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
    lhs0 = tlx.amd_extract_slice(dk_lhs, [128, 16], [0, 0])
    lhs1 = tlx.amd_extract_slice(dk_lhs, [128, 16], [128, 0])
    rhs0 = tlx.amd_extract_slice(dk_rhs, [16, 32], [0, 0])
    rhs1 = tlx.amd_extract_slice(dk_rhs, [16, 32], [0, 32])
    rhs2 = tlx.amd_extract_slice(dk_rhs, [16, 32], [0, 64])
    rhs3 = tlx.amd_extract_slice(dk_rhs, [16, 32], [0, 96])

    c00 = tlx.amd_extract_slice(dk, [128, 32], [0, 0])
    c10 = tlx.amd_extract_slice(dk, [128, 32], [128, 0])
    c01 = tlx.amd_extract_slice(dk, [128, 32], [0, 32])
    c11 = tlx.amd_extract_slice(dk, [128, 32], [128, 32])
    c02 = tlx.amd_extract_slice(dk, [128, 32], [0, 64])
    c12 = tlx.amd_extract_slice(dk, [128, 32], [128, 64])
    c03 = tlx.amd_extract_slice(dk, [128, 32], [0, 96])
    c13 = tlx.amd_extract_slice(dk, [128, 32], [128, 96])

    c00 = tlx.amd_scheduled_mfma(lhs0, rhs0, c00, accumulator="persistent")
    c10 = tlx.amd_scheduled_mfma(lhs1, rhs0, c10, accumulator="persistent")
    ds0 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 0], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    c01 = tlx.amd_scheduled_mfma(lhs0, rhs1, c01, accumulator="persistent")
    c11 = tlx.amd_scheduled_mfma(lhs1, rhs1, c11, accumulator="persistent")
    c02 = tlx.amd_scheduled_mfma(lhs0, rhs2, c02, accumulator="persistent")
    c12 = tlx.amd_scheduled_mfma(lhs1, rhs2, c12, accumulator="persistent")
    ds1 = tlx.local_load(
        tlx.local_slice(prev_ds, [0, 32], [16, 32]),
        layout=DS_MD_LAYOUT,
        relaxed=True,
    )
    c03 = tlx.amd_scheduled_mfma(lhs0, rhs3, c03, accumulator="persistent")
    c13 = tlx.amd_scheduled_mfma(lhs1, rhs3, c13, accumulator="persistent")

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
    off_z,
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
):
    """Run one front/dV phase and either direct dK or the lagged bridge.

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
    lse_values = tlx.local_load(
        tlx.local_view(lse_tiles, phase),
        relaxed=True,
        rematerialize_coordinates_group=20,
    )
    delta_values = tlx.local_load(
        tlx.local_view(delta_tiles, phase),
        relaxed=True,
        rematerialize_coordinates_group=20,
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
        off_z,
        off_h_kv,
        pid_n,
        step,
        SM_SCALE,
        HQ,
        HK,
        N,
        D,
        BLOCK_M,
        BLOCK_N,
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
        dk = tlx.amd_mfma(dk_lhs, dk_rhs, dk)
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
            dq_step = tl.maximum(step - 1, 0)
        else:
            dq_step = step - 1
        _attn_bwd_gqa_store_dq_native(
            dq,
            DQ_ACC,
            off_z,
            off_h_kv,
            dq_step,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            MMA_MD,
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
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Four-wave outer64 GQA backward bridge."""
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
    # global tensor to this program's batch with 64-bit pointer arithmetic so
    # all buffer offsets remain batch-relative. Without this, the 16K HK
    # stress case silently reads/stores zero beginning at batch eight, exactly
    # where flattened BF16 Q/dO/dQ addresses reach 2 GiB.
    q_batch_base = off_z.to(tl.int64) * HQ * N * D
    kv_batch_base = off_z.to(tl.int64) * HK * N * D
    stats_batch_base = off_z.to(tl.int64) * HQ * N
    Q = Q + q_batch_base
    DO = DO + q_batch_base
    DQ_ACC = DQ_ACC + q_batch_base
    K = K + kv_batch_base
    V = V + kv_batch_base
    DK = DK + kv_batch_base
    DV = DV + kv_batch_base
    LSE = LSE + stats_batch_base
    Delta = Delta + stats_batch_base
    off_z = 0

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
    kv_base = (off_z * HK + off_h_kv) * N * D
    k_offsets = (kv_base + (pid_n * BLOCK_N + k_n) * D + k_d_base + raw_v[None, None, :])
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
        off_z,
        off_h_kv,
        0,
        HQ,
        HK,
        N,
        D,
        BLOCK_M,
        OUTER_M,
        qdo_async_layout,
        stats_async_layout,
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
        off_z,
        off_h_kv,
        1,
        HQ,
        HK,
        N,
        D,
        BLOCK_M,
        OUTER_M,
        qdo_async_layout,
        stats_async_layout,
    )
    initial_wait = tlx.async_load_wait_group(2)
    tl.debug_barrier()

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    v_offsets = kv_base + offs_n[:, None] * D + offs_d[None, :]
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
    total_outer_steps: tl.constexpr = (HQ // HK) * (N // OUTER_M)
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
            off_z,
            off_h_kv,
            pid_n,
            outer_step,
            phase0,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            BLOCK_N,
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
                off_z,
                off_h_kv,
                pid_n,
                outer_step,
                phase,
                SM_SCALE,
                HQ,
                HK,
                N,
                D,
                BLOCK_M,
                BLOCK_N,
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
            off_z,
            off_h_kv,
            outer_step + 2,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            OUTER_M,
            qdo_async_layout,
            stats_async_layout,
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
                off_z,
                off_h_kv,
                outer_step * 4 + 3,
                SM_SCALE,
                HQ,
                HK,
                N,
                D,
                BLOCK_M,
                mma_md,
            )
        tlx.async_load_wait_group(2)
        tl.debug_barrier()

    if continuous_bridge:
        # Only the last MHA dS has no following dK with which to braid dQ.
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
            off_z,
            off_h_kv,
            total_outer_steps * 4 - 1,
            SM_SCALE,
            HQ,
            HK,
            N,
            D,
            BLOCK_M,
            mma_md,
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
    store_n = pid_n * BLOCK_N + tlx.amd_rematerialized_range(0, BLOCK_N, 30)
    store_d = tlx.amd_rematerialized_range(0, D, 31)
    key_offsets = kv_base + store_n[:, None] * D + store_d[None, :]
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
    tensor_base = batch_head.to(tl.int64) * N * D
    DQ_ACC = DQ_ACC + tensor_base
    DQ = DQ + tensor_base
    native_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
    native_offsets = tlx.require_layout(native_offsets, native_layout)
    values = tlx.buffer_load(DQ_ACC, native_offsets, contiguity=4)
    values = tlx.require_layout(values, store_layout)

    store_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    store_d = tl.arange(0, D)
    store_offsets = store_m[:, None] * D + store_d[None, :]
    store_offsets = tlx.require_layout(store_offsets.to(tl.int32), store_layout, pin=False)
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
            q_nd_0 = tlx.amd_extract_slice(q_nd, [BLOCK_M, 32], [0, 0])
            q_nd_1 = tlx.amd_extract_slice(q_nd, [BLOCK_M, 32], [0, 32])
            q_nd_2 = tlx.amd_extract_slice(q_nd, [BLOCK_M, 32], [0, 64])
            q_nd_3 = tlx.amd_extract_slice(q_nd, [BLOCK_M, 32], [0, 96])

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
                ds_nd_lo = tlx.amd_extract_slice(
                    ds_nd,
                    [BLOCK_N // 2, BLOCK_M],
                    [0, 0],
                )
                ds_nd_hi = tlx.amd_extract_slice(
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
                previous_ds_0 = tlx.amd_extract_slice(
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
                    accumulator="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_0_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_0,
                    dk_0_hi,
                    accumulator="persistent",
                    initialize=bridge_phase == 0,
                )
            else:
                dk_0 = tl.dot(ds_nd, q_nd_0, acc=dk_0, out_dtype=dk_0.dtype)

            if bridge_phase == 1:
                previous_ds_1 = tlx.amd_extract_slice(
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
                        accumulator="transient",
                        initialize=True,
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_0,
                        k_b_0,
                        dq_b,
                        resident_operand=1,
                        accumulator="transient",
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
                previous_ds_2 = tlx.amd_extract_slice(
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
                    accumulator="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_1_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_1,
                    dk_1_hi,
                    accumulator="persistent",
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
                        accumulator="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_1,
                        k_b_1,
                        dq_b,
                        resident_operand=1,
                        accumulator="transient",
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
                previous_ds_3 = tlx.amd_extract_slice(
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
                    accumulator="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_2_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_2,
                    dk_2_hi,
                    accumulator="persistent",
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
                        accumulator="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_2,
                        k_b_2,
                        dq_b,
                        resident_operand=1,
                        accumulator="transient",
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
                previous_ds_4 = tlx.amd_extract_slice(
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
                    accumulator="persistent",
                    initialize=bridge_phase == 0,
                )
                dk_3_hi = tlx.amd_scheduled_mfma(
                    ds_nd_hi,
                    q_nd_3,
                    dk_3_hi,
                    accumulator="persistent",
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
                        accumulator="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_3,
                        k_b_3,
                        dq_b,
                        resident_operand=1,
                        accumulator="transient",
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
                        accumulator="transient",
                    )
                    dq_b = tlx.amd_scheduled_mfma(
                        previous_ds_4,
                        k_b_4,
                        dq_b,
                        resident_operand=1,
                        accumulator="transient",
                    )
                else:
                    dq_a = tl.dot(previous_ds_4, k_a_4, acc=dq_a, out_dtype=dq_a.dtype)
                    dq_b = tl.dot(previous_ds_4, k_b_4, acc=dq_b, out_dtype=dq_b.dtype)
                for dq_band in tl.static_range(5, num_dq_bands):
                    previous_ds_band = tlx.amd_extract_slice(
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
                            accumulator="transient",
                        )
                        dq_b = tlx.amd_scheduled_mfma(
                            previous_ds_band,
                            k_b_band,
                            dq_b,
                            resident_operand=1,
                            accumulator="transient",
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
        last_ds_band = tlx.amd_extract_slice(
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
        dk_offs_n = tlx.amd_rematerialized_range(0, BLOCK_N, 0)
        dk_offs_d = tlx.amd_rematerialized_range(0, D, 1)
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
        dv_offs_n = tlx.amd_rematerialized_range(0, BLOCK_N, 2)
        dv_offs_d = tlx.amd_rematerialized_range(0, D, 3)
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
    if not (mha_shape or gqa_shape):
        supported = sorted(SUPPORTED_SHAPES)
        raise ValueError(f"supported MHA shapes are {supported}; supported GQA shapes "
                         f"satisfy {_GQA_SHAPE_CONSTRAINT}; got q={tuple(q.shape)}, "
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
    gqa_signature = (
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[3],
    )
    if _is_supported_gqa_shape(gqa_signature):
        if causal:
            raise ValueError("the GQA TLX specialization is non-causal")
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
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 128), id="group1"),
        pytest.param((1, 1, 1, 512, 128), id="group1-two-kv-tiles"),
        pytest.param((1, 2, 1, 256, 128), id="group2"),
        pytest.param((2, 3, 1, 256, 128), id="group3-batch2"),
        pytest.param((1, 4, 1, 256, 128), id="group4"),
        pytest.param((1, 8, 1, 512, 128), id="group8-two-kv-tiles"),
    ],
)
def test_gqa_supported_shapes_end_to_end_gfx950(shape):
    case = _make_gqa_smoke_case(shape, seed=17)
    actual_grads = fa_backward(*case.kernel_args)
    for actual, expected in zip(actual_grads, case.grads):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_gqa_rejects_causal_gfx950():
    q = torch.empty((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="non-causal"):
        fa_backward(q, k, k, q, q, lse, 0.5, True)


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
