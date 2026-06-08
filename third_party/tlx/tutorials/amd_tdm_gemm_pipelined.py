"""
TDM-pipelined GEMM for AMD gfx1250.

Demonstrates the AMD-specific TLX TDM (Tensor Data Movement) API on a
pipelined matmul: ``tl.make_tensor_descriptor`` +
``tlx.async_amd_descriptor_load`` for async global→LDS copies,
``tlx.amd_descriptor_prefetch_tensor`` for look-ahead L2 prefetch, and
``tlx.async_amd_descriptor_wait`` for tensorcnt-based synchronization.

``tlx.local_alloc`` without an explicit ``layout=`` — the compiler's
``tlx-insert-require-layout`` + ``tlx-propagate-layout`` passes rewrite
the alloc's encoding from the descriptor automatically.

Two-buffer software pipeline: prefetch tile k+1 while consuming tile k.
Output stored via pointer-based ``tl.store`` (TDM store requires
``alignTDMDescriptorEncodings`` which is not yet ported to main).
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def matmul_tdm_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B with TDM async loads and a 2-buffer software pipeline."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, tl.constexpr(1)],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, tl.constexpr(1)],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    NUM_BUFFERS: tl.constexpr = 2
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    b_buf = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, 0), [off_m, 0])
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, 0), [0, off_n])
    prefetch_pred = BLOCK_K < K
    tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, BLOCK_K], pred=prefetch_pred)
    tlx.amd_descriptor_prefetch_tensor(b_desc, [BLOCK_K, off_n], pred=prefetch_pred)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K_ITERS - 1):
        next_k = k + 1
        next_slot = next_k % NUM_BUFFERS
        tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, next_slot), [off_m, next_k * BLOCK_K])
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, next_slot), [next_k * BLOCK_K, off_n])

        prefetch_k = next_k + 1
        prefetch_pred = prefetch_k < K_ITERS
        tlx.amd_descriptor_prefetch_tensor(a_desc, [off_m, prefetch_k * BLOCK_K], pred=prefetch_pred)
        tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_k * BLOCK_K, off_n], pred=prefetch_pred)

        tlx.async_amd_descriptor_wait(2)

        cur_slot = k % NUM_BUFFERS
        a_reg = tlx.local_load(tlx.local_view(a_buf, cur_slot))
        b_reg = tlx.local_load(tlx.local_view(b_buf, cur_slot))
        acc = tl.dot(a_reg, b_reg, acc)

    tlx.async_amd_descriptor_wait(0)
    last_slot = (K_ITERS - 1) % NUM_BUFFERS
    a_reg = tlx.local_load(tlx.local_view(a_buf, last_slot))
    b_reg = tlx.local_load(tlx.local_view(b_buf, last_slot))
    acc = tl.dot(a_reg, b_reg, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, config=None) -> torch.Tensor:
    """C = A @ B using a TDM-pipelined kernel on AMD gfx1250."""
    if config is None:
        config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}
    assert a.is_contiguous() and b.is_contiguous(), "A and B must be contiguous"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, \
        "M, N, K must be multiples of their block sizes"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_tdm_pipelined_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
