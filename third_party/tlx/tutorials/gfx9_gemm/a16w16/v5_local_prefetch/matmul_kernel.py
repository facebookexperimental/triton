"""v5_local_prefetch — Manual 3-stage pipeline with local prefetch.

Prologue: issue buf 0 and buf 1, wait buf 0, local_load A0 B0
Loop:     DOT with prefetched data, wait next, local_load next, issue next async_copy
Epilogue: DOT last two iterations
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def v5_local_prefetch(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                      stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    # The bank-conflict-avoiding padded shared layout (shown explicitly in v3)
    # is now inferred by the compiler from how these buffers feed tl.dot.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 2)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_off = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_off = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    iterMax: tl.constexpr = K // BLOCK_K

    # Prologue: issue A0,B0 → buf 0 and A1,B1 → buf 1
    tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b[0], b_ptr, b_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    tlx.buffer_load_to_local(smem_a[1], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b[1], b_ptr, b_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    # Wait for buf 0, prefetch A0, B0 into registers
    tlx.async_load_wait_group(1)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b = tlx.local_load(smem_b[0], relaxed=True)

    # Main loop
    for k in tl.range(0, iterMax - 1, num_stages=1):
        g_idx = k % 2
        l_idx = 1 - g_idx

        acc = tl.dot(a, b, acc)

        tlx.async_load_wait_group(1)

        tlx.buffer_load_to_local(smem_a[g_idx], a_ptr, a_off + a_k, mask=(k != (iterMax - 2)))
        tlx.buffer_load_to_local(smem_b[g_idx], b_ptr, b_off + b_k, mask=(k != (iterMax - 2)))
        tlx.async_load_commit_group()

        a = tlx.local_load(smem_a[l_idx], relaxed=True)
        b = tlx.local_load(smem_b[l_idx], relaxed=True)

        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

    # Epilogue: last DOT
    acc = tl.dot(a, b, acc)

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    v5_local_prefetch[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                            c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=1,
                            matrix_instr_nonkdim=16)
    return c
