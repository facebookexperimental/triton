"""BF16 Flash-Attention backward kernel families for AMD gfx950.

The implementation contains separate D128 and D256 kernel families, but the
public ``fa_backward`` wrapper intentionally supports only the two validated
production tuples listed in ``SUPPORTED_SHAPES``.  Other D128/D256 shapes are
not part of this submission's public contract yet.  Run this file with pytest
for correctness.

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
import triton.language.core as tl_core
import triton.language.extra.tlx as tlx

# Public correctness contract for this submission.  The kernels
# have D128 and D256 families internally, but these are the only two complete
# BF16 tuples validated end-to-end on gfx950.
SUPPORTED_SHAPES = {
    (16, 27, 200, 128),
    (32, 1, 2600, 256),
}
# Gluon pins CDNA4's 16x16x32 MFMA for these BF16 tiles.  Leaving Triton to
# infer the instruction shape selects 32x32x16 on this checkout, which doubles
# register pressure for the D-sliced kernels and prevents the accumulator from
# using AGPRs on gfx950.
_CDNA4_MATRIX_INSTR_NONKDIM = 16


@tl_core.builtin
def _require_layout_soft(x, layout, _semantic=None):
    """Attach an explicit register layout without making it a hard anchor.

    Upstream ``tlx.require_layout`` pins user-authored epilogue ownership so
    layout optimization cannot rewrite it.  These FA kernels instead use soft
    requirements for direct-to-LDS offsets, MFMA operands, and intermediate
    arithmetic; those requirements must remain eligible for TLX fixup and
    placeholder resolution.
    """
    x = _semantic.to_tensor(x)
    layout = tl_core._unwrap_if_constexpr(layout)
    encoding = layout.to_ir(_semantic.builder, x.shape, x.dtype)
    handle = _semantic.builder.create_require_layout(x.handle, encoding, pin=False)
    return tl_core.tensor(handle, x.type)


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


@triton.jit
def _attn_bwd_preprocess_kernel(O, DO, Delta, N: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr):
    batch_head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    offsets = batch_head * N * D + rows[:, None] * D + cols[None, :]
    mask = rows[:, None] < N
    o = tl.load(O + offsets, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(Delta + batch_head * N + rows, tl.sum(o * do, axis=1), mask=rows < N)


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
        kv_offsets = _require_layout_soft(kv_offsets, qdo_async_layout)
        kv_load_mask = tl.broadcast_to(row_mask, kv_offsets.shape)
        kv_load_mask = _require_layout_soft(kv_load_mask, qdo_async_layout)
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
            qdo_offsets = _require_layout_soft(qdo_offsets, qdo_async_layout)
            qdo_load_mask = tl.broadcast_to(qdo_mask, qdo_offsets.shape)
            qdo_load_mask = _require_layout_soft(qdo_load_mask, qdo_async_layout)
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
    first_offsets = _require_layout_soft(first_offsets, qdo_async_layout)
    first_load_mask = tl.broadcast_to(first_mask, first_offsets.shape)
    first_load_mask = _require_layout_soft(first_load_mask, qdo_async_layout)
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
        next_offsets = _require_layout_soft(next_offsets, qdo_async_layout)
        next_load_mask = tl.broadcast_to(next_mask, next_offsets.shape)
        next_load_mask = _require_layout_soft(next_load_mask, qdo_async_layout)
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
    tl.static_assert(not IS_CAUSAL or IS_CAUSAL)
    batch_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
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
    # Match Gluon's causal producer schedule: Q is three-stage while dO stays
    # two-stage, so the score load can remain two tiles ahead of consumption.
    Q_STAGES: tl.constexpr = 3 if IS_CAUSAL else 2
    Q_LOOKAHEAD: tl.constexpr = 2 if IS_CAUSAL else 1
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
    v_buffer = tlx.local_alloc((BLOCK_N, D), tl.bfloat16, 1, layout=kv_smem_layout)
    q_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, Q_STAGES, layout=qdo_smem_layout)
    do_buffers = tlx.local_alloc((BLOCK_M, D), tl.bfloat16, 2, layout=qdo_smem_layout)
    ds_buffer = tlx.local_alloc((BLOCK_N, BLOCK_M), tl.bfloat16, 1, layout=ds_smem_layout)

    # Masked direct-to-LDS copies leave OOB rows untouched.  Clear only the
    # aligned K suffix needed by the N=200 tile; this preserves the valid K
    # rows while preventing undefined LDS values from entering dQ MFMA.  V is
    # still cleared as a complete tile because its score-side dP consumer reads
    # the full resident image before the probability mask is applied.
    # The current TLX AMD buffer-op conversion cannot lower a rank-3
    # memdesc_subslice in the same pipeline as a direct-to-LDS copy.  Keep the
    # physical image initialized with the same lane ownership for now; the
    # valid K rows are overwritten by the async copy below and the aligned tail
    # remains zero without relying on masked-copy fallback semantics.
    tlx.local_store(
        tlx.local_view(k_raw_buffer, 0),
        tlx.zeros((BLOCK_N, D // 8, 8), tl.bfloat16, layout=k_raw_async_layout),
    )
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
    k_offsets = _require_layout_soft(k_offsets.to(tl.int32), k_raw_async_layout)
    k_load_mask = _require_layout_soft(tl.broadcast_to(k_n < N, k_offsets.shape), k_raw_async_layout)
    key_offsets = _require_layout_soft(key_ptrs.to(tl.int32), kv_async_layout)
    key_load_mask = _require_layout_soft(tl.broadcast_to(key_mask, key_offsets.shape), kv_async_layout)
    k_token = tlx.buffer_load_to_local(
        tlx.local_view(k_raw_buffer, 0),
        K,
        k_offsets,
        mask=k_load_mask,
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
    v_nm = tlx.local_load(tlx.local_view(v_buffer, 0), token=kv_wait, layout=v_op0_nm)

    dk = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    dv = tlx.zeros((BLOCK_N, D), tl.float32, layout=mma_nd)
    num_m_blocks: tl.constexpr = tl.cdiv(N, BLOCK_M)
    log2e: tl.constexpr = 1.4426950408889634

    first_m = tl.arange(0, BLOCK_M)
    first_ptrs = tensor_base + first_m[:, None] * D + offs_d[None, :]
    first_mask = first_m[:, None] < N
    first_offsets = _require_layout_soft(first_ptrs.to(tl.int32), qdo_async_layout)
    first_load_mask = _require_layout_soft(tl.broadcast_to(first_mask, first_offsets.shape), qdo_async_layout)
    first_q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, 0), Q, first_offsets, mask=first_load_mask)
    first_do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, 0), DO, first_offsets, mask=first_load_mask)
    if IS_CAUSAL:
        # Commit Q[0], dO[0], and Q[1] together. Q[1] is intentionally loaded
        # without a separate wait so the first score tile can start as soon as
        # the current pair is ready.
        first_q1 = first_m + BLOCK_M
        first_q1_ptrs = tensor_base + first_q1[:, None] * D + offs_d[None, :]
        first_q1_mask = first_q1[:, None] < N
        first_q1_offsets = _require_layout_soft(first_q1_ptrs.to(tl.int32), qdo_async_layout)
        first_q1_load_mask = _require_layout_soft(tl.broadcast_to(first_q1_mask, first_q1_offsets.shape),
                                                  qdo_async_layout)
        first_q1_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, 1), Q, first_q1_offsets,
                                                  mask=first_q1_load_mask)
        tlx.async_load_commit_group([first_q_token, first_do_token, first_q1_token])
    else:
        tlx.async_load_commit_group([first_q_token, first_do_token])

    for m_block in range(0, num_m_blocks):
        if IS_CAUSAL:
            current_slot = m_block % Q_STAGES
            current_do_slot = m_block % 2
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
            next_q_offsets = _require_layout_soft(next_q_ptrs.to(tl.int32), qdo_async_layout)
            next_q_load_mask = _require_layout_soft(tl.broadcast_to(next_q_mask, next_q_offsets.shape),
                                                    qdo_async_layout)
            next_do_offsets = _require_layout_soft(next_do_ptrs.to(tl.int32), qdo_async_layout)
            next_do_load_mask = _require_layout_soft(tl.broadcast_to(next_do_mask, next_do_offsets.shape),
                                                     qdo_async_layout)
            next_q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, next_slot), Q, next_q_offsets,
                                                    mask=next_q_load_mask)
            next_do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, next_do_slot), DO, next_do_offsets,
                                                     mask=next_do_load_mask)
            tlx.async_load_commit_group([next_q_token, next_do_token])
            qdo_wait = tlx.async_load_wait_group(1)
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
            next_offsets = _require_layout_soft(next_ptrs.to(tl.int32), qdo_async_layout)
            next_load_mask = _require_layout_soft(tl.broadcast_to(next_mask, next_offsets.shape), qdo_async_layout)
            next_q_token = tlx.buffer_load_to_local(tlx.local_view(q_buffers, next_slot), Q, next_offsets,
                                                    mask=next_load_mask)
            next_do_token = tlx.buffer_load_to_local(tlx.local_view(do_buffers, next_slot), DO, next_offsets,
                                                     mask=next_load_mask)
            tlx.async_load_commit_group([next_q_token, next_do_token])
            qdo_wait = tlx.async_load_wait_group(1)
            do_slot = current_slot

        q_view = tlx.local_view(q_buffers, current_slot)
        do_view = tlx.local_view(do_buffers, do_slot)
        q_t = tlx.local_load(tlx.local_trans(q_view), token=qdo_wait, layout=qt_op1_nm)
        do_t = tlx.local_load(tlx.local_trans(do_view), token=qdo_wait, layout=dot_op1_nm)
        q_nd = tlx.local_load(q_view, token=qdo_wait, layout=q_op1_nd)
        do_nd = tlx.local_load(do_view, token=qdo_wait, layout=do_op1_nd)

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        lse = tl.load(LSE + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + batch_head * N + offs_m, mask=offs_m < N, other=0.0)
        score_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
        score_acc = tlx.mfma(k_nm, q_t, score_acc)
        lse_full = tl.broadcast_to(lse[None, :] * log2e, (BLOCK_N, BLOCK_M))
        lse_full = _require_layout_soft(lse_full, mma_nm)
        qk_scale_full = _require_layout_soft(
            tl.full((BLOCK_N, BLOCK_M), SM_SCALE * log2e, dtype=tl.float32),
            mma_nm,
        )
        scores_t = score_acc * qk_scale_full - lse_full
        valid = key_mask & (offs_m[None, :] < N)
        if IS_CAUSAL:
            valid = valid & (offs_n[:, None] <= offs_m[None, :])
        valid = _require_layout_soft(valid, mma_nm)
        neg_inf = _require_layout_soft(
            tl.full((BLOCK_N, BLOCK_M), float("-inf"), dtype=tl.float32),
            mma_nm,
        )
        scores_t = tl.where(valid, scores_t, neg_inf)
        p_t = _require_layout_soft(tl.math.exp2(scores_t), mma_nm)

        dpt_acc = tlx.zeros((BLOCK_N, BLOCK_M), tl.float32, layout=mma_nm)
        dpt_acc = tlx.mfma(v_nm, do_t, dpt_acc)
        delta_full = tl.broadcast_to(delta[None, :], (BLOCK_N, BLOCK_M))
        delta_full = _require_layout_soft(delta_full, mma_nm)
        ds_t = p_t * (dpt_acc - delta_full)
        # The MFMA score result has an explicit register ownership.  Release it
        # at the LDS handoff before narrowing; a direct cast would produce a
        # null-layout BF16 tensor while retaining an MFMA-encoded operand, which
        # the Triton cast verifier correctly rejects.
        ds_bf16 = tlx.release_layout(ds_t).to(tl.bfloat16)

        # Publish dS^T once, then consume it in both dQ and dK ownership.  The
        # barrier is required because the same LDS tile is overwritten next
        # iteration and relaxed local loads are intentionally used below.
        tlx.local_store(tlx.local_view(ds_buffer, 0), ds_bf16)
        tl.debug_barrier()
        ds_md = tlx.local_load(tlx.local_trans(tlx.local_view(ds_buffer, 0)), layout=ds_op0_md, relaxed=True)
        k_md = tlx.local_load(k_buffer, token=kv_wait, layout=k_op1_md, relaxed=True)
        dq_part = tlx.zeros((BLOCK_M, D), tl.float32, layout=mma_md)
        dq_part = tlx.mfma(ds_md, k_md, dq_part)
        dq_scale_full = _require_layout_soft(tl.full((BLOCK_M, D), SM_SCALE, dtype=tl.float32), mma_md)
        dq_part = dq_part * dq_scale_full
        q_ptrs = tensor_base + offs_m[:, None] * D + offs_d[None, :]
        q_mask = offs_m[:, None] < N
        tl.store(
            DQ + q_ptrs,
            tlx.release_layout(dq_part).to(tl.bfloat16),
            mask=q_mask,
        )

        # Reuse the score probabilities in the dK/dV operand ownership.  This
        # is a representation change, so release the score MFMA layout before
        # narrowing and pin the resulting BF16 tile to the dK/dV operand view.
        pt_nd = _require_layout_soft(tlx.release_layout(p_t).to(tl.bfloat16), pt_op0_nd)
        ds_nd = tlx.local_load(tlx.local_view(ds_buffer, 0), layout=dst_op0_nd, relaxed=True)
        dv = tlx.mfma(pt_nd, do_nd, dv)
        dk = tlx.mfma(ds_nd, q_nd, dk)

    tlx.async_load_wait_group(0)
    dk = tlx.release_layout(dk)
    dv = tlx.release_layout(dv)
    dk *= SM_SCALE
    # Causal and non-causal modes use Gluon's whole-tile epilogue: narrow while
    # retaining native MFMA ownership, then anchor the final store to eight
    # contiguous BF16 values per lane. ``cast_layout`` is required to preserve
    # the MFMA encoding; an ordinary ``to(bfloat16)`` drops it and fails
    # verification. The hard ``require_layout`` store anchor is provided by
    # #2290, so Coalesce and AMD OptimizeEpilogue retain the coalesced store
    # ownership without pinning the intermediate MFMA arithmetic.
    # Gluon's newer full-attention D64-half epilogue raised TLX from 496 to 503
    # VGPR for N=200, so the lower-resource whole-tile epilogue remains used.
    dk_mma = _require_layout_soft(dk, mma_nd)
    dk_bf16 = tlx.cast_layout(dk_mma, tl.bfloat16)
    dk_vec = tlx.require_layout(dk_bf16, kv_async_layout)
    tl.store(DK + key_ptrs, dk_vec, mask=key_mask)
    # Keep dV's physical conversion explicit. Pinning both output stores makes
    # the causal N=200 build cross the gfx950 register-spill threshold; the
    # dK-only anchor preserves the coalesced dwordx4 stores without that spill.
    dv_mma = _require_layout_soft(dv, mma_nd)
    dv_bf16 = tlx.cast_layout(dv_mma, tl.bfloat16)
    dv_vec = tlx.convert_layout(dv_bf16, kv_async_layout)
    tl.store(DV + key_ptrs, tlx.release_layout(dv_vec), mask=key_mask)


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
            num_warps=dispatch.num_warps,
            num_stages=1,
            matrix_instr_nonkdim=_matrix_instr_nonkdim(),
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

        q_lo_view = tlx.local_view(q_lo_buffer, 0)
        q_hi_view = tlx.local_view(q_hi_buffer, 0)
        do_lo_view = tlx.local_view(do_lo_buffer, 0)
        do_hi_view = tlx.local_view(do_hi_buffer, 0)
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
    tensors = {"q": q, "k": k, "v": v, "o": o, "do": do}
    if tuple(q.shape) not in SUPPORTED_SHAPES:
        raise ValueError(f"supported shapes are {sorted(SUPPORTED_SHAPES)}, got {tuple(q.shape)}")
    for name, tensor in tensors.items():
        if tensor.device != q.device or tensor.shape != q.shape:
            raise ValueError(f"{name} must match q shape and device")
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
    with pytest.raises(ValueError, match="supported shapes"):
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
