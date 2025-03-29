import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_ws_cooperative_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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

    offs_am1 = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)) % M
    offs_am2 = (pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs1 = a_ptr + (offs_am1[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_ptrs2 = a_ptr + (offs_am2[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator1 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        with tl.async_task([0]):
            a1 = tl.load(
                a_ptrs1, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            a2 = tl.load(
                a_ptrs2, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )

        accumulator1 = tl.dot(a1, b, accumulator1)
        accumulator2 = tl.dot(a2, b, accumulator2)
        a_ptrs1 += BLOCK_SIZE_K * stride_ak
        a_ptrs2 += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c1 = accumulator1.to(tl.float16)
    c2 = accumulator2.to(tl.float16)

    offs_cm1 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)
    offs_cm2 = pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs1 = c_ptr + stride_cm * offs_cm1[:, None] + stride_cn * offs_cn[None, :]
    c_ptrs2 = c_ptr + stride_cm * offs_cm2[:, None] + stride_cn * offs_cn[None, :]
    c_mask1 = (offs_cm1[:, None] < M) & (offs_cn[None, :] < N)
    c_mask2 = (offs_cm2[:, None] < M) & (offs_cn[None, :] < N)
    with tl.async_task([1]):
        tl.store(c_ptrs1, c1, mask=c_mask1)
    with tl.async_task([2]):
        tl.store(c_ptrs2, c2, mask=c_mask2)


def matmul_ws_cooperative(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_ws_cooperative_kernel[grid](
        a,
        b,
        c,
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
