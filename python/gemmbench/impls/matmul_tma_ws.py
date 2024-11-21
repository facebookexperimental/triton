import torch

import triton
import triton.language as tl
import triton.tools.experimental_descriptor


@triton.autotune(
    configs=[
        # fmt: off
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4, num_buffers_warp_spec=3, num_consumer_groups=1),
        # fmt: on
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_tma_ws_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # ffs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        with tl.async_task([0]):
            a = tl._experimental_descriptor_load(
                a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16
            )
            b = tl._experimental_descriptor_load(
                b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16
            )
        accumulator = tl.dot(a, b, accumulator)
        # a_ptrs += BLOCK_SIZE_K * stride_ak
        # b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = accumulator.to(tl.float16)

    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, c, mask=c_mask)
    with tl.async_task([1]):
        tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])


# matmul_tma_ws_kernel_ttgir = triton.compiler.compile("matmul_tma_ws_kernel.ttgir")
USE_TTGIR = False


def matmul_tma_ws(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    configs = {
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
        },
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
        },
    }
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        a.data_ptr(),
        M,
        K,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_K"],
        a.element_size(),
    )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        b.data_ptr(),
        K,
        N,
        configs[dtype]["BLOCK_SIZE_K"],
        configs[dtype]["BLOCK_SIZE_N"],
        b.element_size(),
    )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        c.data_ptr(),
        M,
        N,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_N"],
        c.element_size(),
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if USE_TTGIR:
        # matmul_tma_ws_kernel_ttgir._init_handles()
        # matmul_tma_ws_kernel_ttgir.run(
        #    triton.cdiv(M, configs[dtype]["BLOCK_SIZE_M"]) * triton.cdiv(N, configs[dtype]["BLOCK_SIZE_N"]),
        #    1,
        #    1,
        matmul_tma_ws_kernel_ttgir[
            triton.cdiv(M, configs[dtype]["BLOCK_SIZE_M"])
            * triton.cdiv(N, configs[dtype]["BLOCK_SIZE_N"]),
            1,
            1,
        ](
            # 0,
            # matmul_tma_ws_kernel_ttgir.function,
            # (8, 1, 133120, 1, 1, 1),
            # None,
            # None,
            # None,
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            a.stride(0),
            # a.stride(1),
            b.stride(0),
            # b.stride(1),
            c.stride(0),
            # c.stride(1),
        )
    else:
        matmul_tma_ws_kernel[grid](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c
