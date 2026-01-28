import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from typing import Optional

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device='cuda')


@triton.jit
def tcgen5_dot_kernel2cta_tma(a_ptr, stride_am, stride_ak, b_ptr, stride_bk, stride_bn, c_ptr, stride_cm, stride_cn,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                              OUT_DTYPE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # assuming CTA pairs along M dim
    cluster_cta_rank = tlx.cluster_cta_rank()  # 2cta specific
    pred_leader_cta = cluster_cta_rank % 2 == 0

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N + (cluster_cta_rank % 2) * (BLOCK_N // 2)  # 2cta specific

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    desc_a = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    desc_b = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N // 2],
    )

    # async load a and b into SMEM
    buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
    buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tl.float16, tl.constexpr(1))  # 2cta specific
    a_smem = tlx.local_view(buf_alloc_a, 0)
    b_smem = tlx.local_view(buf_alloc_b, 0)

    bars = tlx.alloc_barriers(tl.constexpr(3))
    bar_a = tlx.local_view(bars, 0)
    bar_b = tlx.local_view(bars, 1)

    # 2cta specific
    bar_cta = tlx.alloc_barriers(1, arrive_count=2)  # CTA0 waits for CTA1's data before mma
    bar_leader_cta = tlx.local_view(bar_cta, 0)

    buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
    acc_tmem = tlx.local_view(buffers, 0)

    acc_init = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tlx.local_store(acc_tmem, acc_init)

    dot_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    phase = 0
    num_iter = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_iter):
        offs_k = k * BLOCK_K

        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * 2)
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * (BLOCK_N // 2) * 2)  # 2cta specific

        tlx.async_descriptor_load(desc_a, a_smem, [offs_am, offs_k], bar_a)
        tlx.async_descriptor_load(desc_b, b_smem, [offs_k, offs_bn], bar_b)

        tlx.barrier_wait(bar_a, phase)
        tlx.barrier_wait(bar_b, phase)

        # CTA0 needs to know CTA1 is done loading data before issuing MMA
        tlx.barrier_arrive(bar_leader_cta, 1, remote_cta_rank=cluster_cta_rank & ~1)
        tlx.barrier_wait(bar_leader_cta, phase=k % 2, pred=pred_leader_cta)

        # 2cta specific
        tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=True, mBarriers=[dot_bars[0]], two_ctas=True,
                      out_dtype=OUT_DTYPE)

        tlx.barrier_wait(dot_bars[0], phase)
        phase = phase ^ 1

    result = tlx.local_load(acc_tmem)

    c = result.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    triton.set_allocator(alloc_fn)

    BLOCK_M, BLOCK_N, BLOCK_K = (128, 128, 128)

    kern_kwargs = {
        'BLOCK_M': BLOCK_M, 'BLOCK_K': BLOCK_K, 'BLOCK_N': BLOCK_N, 'OUT_DTYPE': tl.float32, 'M': M, 'N': N, 'K': K,
        'num_stages': 0, 'ctas_per_cga': (4, 2, 1)
    }
    _ = tcgen5_dot_kernel2cta_tma[(M // BLOCK_M, N // BLOCK_N)](a, a.stride(0), a.stride(1), b, b.stride(0),
                                                                b.stride(1), c, c.stride(0), c.stride(1), **kern_kwargs)

    return c
