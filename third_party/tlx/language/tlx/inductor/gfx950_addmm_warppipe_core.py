"""Shared gfx950 addmm warp-pipe compute cores."""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def gfx950_addmm_warppipe_compute_async(
    a_ptr,
    b_ptr,
    smem_a,
    smem_b,
    a_base,
    b_base,
    offs_k,
    k_lo,
    n_iters,
    k_tail,
    tail_width,
    stride_ak,
    stride_bk,
    ALLOW_TF32: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """Compute one output tile with the shared async-load warp pipeline."""
    # gfx950 cannot lower partial-K async copies into this padded LDS layout.
    # Callers pass only full tiles here and handle the remainder synchronously.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = (k_lo + i) * BLOCK_K
        a_offs = a_base + (k_start + offs_k[None, :]) * stride_ak
        b_offs = b_base[:, None] + (k_start + offs_k[None, :]) * stride_bk
        tok_a = tlx.async_load(
            a_ptr + a_offs.to(tl.int32),
            tlx.local_view(smem_a, i),
        )
        tok_b = tlx.async_load(
            b_ptr + b_offs.to(tl.int32),
            tlx.local_view(smem_b, i),
        )
        # One combined group per K tile lets wait counts correspond to tiles.
        tlx.async_load_commit_group([tok_a, tok_b])

    tlx.async_load_wait_group(NUM_BUFFERS - 2)
    a_tile = tlx.local_load(tlx.local_view(smem_a, 0))
    b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, 0)))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for tile_id in tl.range(0, n_iters - NUM_BUFFERS):
        # local_view requires int32 buffer indices for symbolic K.
        prefetch_buf = (tile_id % NUM_BUFFERS).to(tl.int32)
        next_buf = ((tile_id + 1) % NUM_BUFFERS).to(tl.int32)
        k_prefetch = (k_lo + tile_id + NUM_BUFFERS) * BLOCK_K

        # The loop-tail wait prevents the leading wave from refilling LDS while
        # another wave still consumes the current buffer.
        tlx.async_load_wait_group(NUM_BUFFERS - 2)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(
                a_tile,
                b_tile,
                acc,
                allow_tf32=ALLOW_TF32,
                out_dtype=ACC_TYPE,
            )

        with tlx.warp_pipeline_stage("mem", priority=1):
            a_offs = a_base + (k_prefetch + offs_k[None, :]) * stride_ak
            b_offs = b_base[:, None] + (k_prefetch + offs_k[None, :]) * stride_bk
            tok_a = tlx.async_load(
                a_ptr + a_offs.to(tl.int32),
                tlx.local_view(smem_a, prefetch_buf),
            )
            tok_b = tlx.async_load(
                b_ptr + b_offs.to(tl.int32),
                tlx.local_view(smem_b, prefetch_buf),
            )
            tlx.async_load_commit_group([tok_a, tok_b])
            a_tile = tlx.local_load(tlx.local_view(smem_a, next_buf))
            b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, next_buf)))

    acc = tl.dot(
        a_tile,
        b_tile,
        acc,
        allow_tf32=ALLOW_TF32,
        out_dtype=ACC_TYPE,
    )
    tlx.async_load_wait_group(0)
    for i in tl.range(
        0,
        NUM_BUFFERS - 1,
        loop_unroll_factor=NUM_BUFFERS - 1,
    ):
        buf = ((n_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS).to(tl.int32)
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b, buf)))
        acc = tl.dot(
            a_tile,
            b_tile,
            acc,
            allow_tf32=ALLOW_TF32,
            out_dtype=ACC_TYPE,
        )

    # Fold the partial-K remainder into the accumulator with register loads.
    a_offs = a_base + (k_tail + offs_k[None, :]) * stride_ak
    a_tail = tl.load(
        a_ptr + a_offs.to(tl.int32),
        mask=offs_k[None, :] < tail_width,
        other=0.0,
    )
    b_offs = (k_tail + offs_k[:, None]) * stride_bk + b_base[None, :]
    b_tail = tl.load(
        b_ptr + b_offs.to(tl.int32),
        mask=offs_k[:, None] < tail_width,
        other=0.0,
    )
    return tl.dot(
        a_tail,
        b_tail,
        acc,
        allow_tf32=ALLOW_TF32,
        out_dtype=ACC_TYPE,
    )


@triton.jit
def gfx950_addmm_compute_register(
    a_ptr,
    b_ptr,
    a_base,
    b_base,
    offs_k,
    k,
    stride_ak,
    stride_bk,
    ALLOW_TF32: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one output tile with synchronous register loads."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_start in range(0, k, BLOCK_K):
        a_offs = a_base + (k_start + offs_k[None, :]) * stride_ak
        a = tl.load(
            a_ptr + a_offs.to(tl.int32),
            mask=offs_k[None, :] < k - k_start,
            other=0.0,
        )
        b_offs = (k_start + offs_k[:, None]) * stride_bk + b_base[None, :]
        b = tl.load(
            b_ptr + b_offs.to(tl.int32),
            mask=offs_k[:, None] < k - k_start,
            other=0.0,
        )
        acc = tl.dot(
            a,
            b,
            acc,
            allow_tf32=ALLOW_TF32,
            out_dtype=ACC_TYPE,
        )
    return acc
