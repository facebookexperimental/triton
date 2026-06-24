"""v2_async_copy — Direct-to-LDS via buffer_load_to_local, bypassing registers."""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def v2_async_copy(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    # Plain (unpadded) swizzled layouts — this step uses async copy but NOT yet
    # the bank-conflict-avoiding padded layout (that's v3). Specify them
    # explicitly so the compiler's padded-layout inference doesn't pad here.
    layout_a: tl.constexpr = tlx.swizzled_shared_layout_encoding(vectorSize=1, perPhase=1, maxPhase=1, order=[1, 0],
                                                                 numCTAs=[1, 1], numCTAsPerCGA=[1, 1],
                                                                 numCTASplit=[1, 1], numCTAOrder=[1, 1])
    layout_b: tl.constexpr = tlx.swizzled_shared_layout_encoding(vectorSize=1, perPhase=1, maxPhase=1, order=[0, 1],
                                                                 numCTAs=[1, 1], numCTAsPerCGA=[1, 1],
                                                                 numCTASplit=[1, 1], numCTAOrder=[1, 1])
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 1, layout=layout_a)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 1, layout=layout_b)

    a_off = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_off = tl.arange(0, BLOCK_K)[:, None] * stride_bk + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_bn
    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=1):
        tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
        tlx.buffer_load_to_local(smem_b[0], b_ptr, b_off + b_k)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        a = tlx.local_load(smem_a[0], relaxed=True)
        b = tlx.local_load(smem_b[0], relaxed=True)

        acc = tl.dot(a, b, acc)
        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

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
    v2_async_copy[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=1,
                        matrix_instr_nonkdim=16)
    return c
