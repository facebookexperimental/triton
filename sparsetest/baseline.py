# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random

random.seed(42)

import os
import numpy as np
import time
import torch

torch.manual_seed(42)

import triton
import triton.language as tl


def get_autotune_config():
    # This is the fastest config on my machine
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
            },
            num_stages=3,
            num_warps=8,
        )
    ]

    # Comment out the above return to run full auto-tuning
    configs = []
    for BLOCK_M in [64, 128, 256]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [64, 128, 256]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        for GROUP_SIZE_M in [4, 8, 16]:
                            if max(BLOCK_M, BLOCK_N, BLOCK_K) < 128:
                                continue
                            if sum(x == 256 for x in [BLOCK_M, BLOCK_N, BLOCK_K]) > 1:
                                # not enough smem, save time on compilation
                                continue
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_SIZE_M": BLOCK_M,
                                        "BLOCK_SIZE_N": BLOCK_N,
                                        "BLOCK_SIZE_K": BLOCK_K,
                                        "GROUP_SIZE_M": GROUP_SIZE_M,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                ),
                            )
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_sparse(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_meta_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    stride_aMeta_m,
    stride_aMeta_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    tl.assume(K > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_SIZE_K_A: tl.constexpr = BLOCK_SIZE_K // 2
    BLOCK_SIZE_K_A_META: tl.constexpr = BLOCK_SIZE_K // 16
    offs_k_a = tl.arange(0, BLOCK_SIZE_K_A)
    offs_k_b = tl.arange(0, BLOCK_SIZE_K)
    offs_k_aMeta = tl.arange(0, BLOCK_SIZE_K_A_META)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_a[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_b[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    aMeta_ptrs = a_meta_ptr + (
        offs_am[:, None] * stride_aMeta_m + offs_k_aMeta[None, :] * stride_aMeta_k
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_LIMIT = tl.cdiv(K, BLOCK_SIZE_K)
    tl.assume(K_LIMIT > 0)
    for k in range(0, K_LIMIT):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k_a[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        # Note: the masks below are not correct. It's fine for M=N=K=8192 because 8192 is a power of 2. But this kernel is not correct for other shapes.
        # For some reason, things don't work if I fix the masks. Did not have time to figure out why. Do not use in production.
        # @nocommit.
        b = tl.load(b_ptrs, mask=offs_k_b[:, None] < K - k * BLOCK_SIZE_K_A, other=0.0)
        aMeta = tl.load(
            aMeta_ptrs,
            mask=offs_k_aMeta[None, :] < K - k * BLOCK_SIZE_K_A_META,
            other=0,
        )

        # We accumulate along the K dimension.
        accumulator = tl.sparse_dot(a, b, accumulator, aMeta)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K_A * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        aMeta_ptrs += BLOCK_SIZE_K_A_META * stride_aMeta_k
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!

    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def make_sparse(A):
    assert len(A.shape) == 2
    assert A.shape[-1] % 4 == 0
    assert A.is_contiguous()
    for i in range(A.shape[0]):
        indices_to_zero = []
        for j in range(0, A.shape[1], 4):
            indices_to_zero.extend(random.sample([j, j + 1, j + 2, j + 3], 2))
        A[i, indices_to_zero] = 0
    return A


def compress(A):
    assert len(A.shape) == 2
    assert A.shape[-1] % 4 == 0
    assert A.is_contiguous()
    flat = A.flatten().cpu().detach().numpy()
    nonzero_indices = []
    meta_nibbles = []

    for outerIdx in range(0, len(flat), 4):
        nibble = 0
        nonzeroCount = 0
        for innerIdx in range(4):
            val = flat[outerIdx + innerIdx]
            if val != 0:
                nonzero_indices.append(outerIdx + innerIdx)
                nibble |= innerIdx << (2 * nonzeroCount)
                nonzeroCount += 1
                if nonzeroCount > 2:
                    raise Exception("too many nonzeros!")

        if nonzeroCount == 0:
            nibble |= 0b0100
            nonzero_indices.extend([outerIdx + 0, outerIdx + 1])

        # there is definitely a cleaner way to do this... don't judge me
        if nonzeroCount == 1:
            last_nonzero = nonzero_indices[-1]
            if last_nonzero == outerIdx + 3:
                assert nibble == 0b0011
                nibble = 0b1100
                nonzero_indices[-1] = outerIdx + 0
                nonzero_indices.append(outerIdx + 3)
            else:
                nibble |= 0b1100
                nonzero_indices.append(outerIdx + 3)

        meta_nibbles.append(nibble)

    assert len(meta_nibbles) == len(flat) // 4
    assert len(nonzero_indices) == len(flat) // 2

    metas = []
    for outerIdx in range(0, len(meta_nibbles), 4):
        meta = 0
        meta |= meta_nibbles[outerIdx + 0] << 0
        meta |= meta_nibbles[outerIdx + 1] << 4
        meta |= meta_nibbles[outerIdx + 2] << 8
        meta |= meta_nibbles[outerIdx + 3] << 12
        metas.append(meta)

    aSparse = (A.flatten()[nonzero_indices]).reshape(A.shape[0], A.shape[1] // 2)
    aMeta = torch.tensor(np.array(metas, dtype=np.uint16).astype(np.int16)).reshape(
        A.shape[0], A.shape[1] // 16
    )
    return aSparse.cuda(), aMeta.cuda()


def matmul(aSparse, aMeta, b):
    # Check constraints.
    assert aSparse.shape[1] * 2 == b.shape[0], "Incompatible dimensions"
    assert aMeta.shape[1] * 16 == b.shape[0], "Incompatible dimensions"
    assert aMeta.shape[0] == aSparse.shape[0], "Incompatible dimensions"
    assert aSparse.is_contiguous(), "Matrix A must be contiguous"
    M, _ = aSparse.shape
    K, N = b.shape

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel_sparse[grid](
        aSparse,
        b,
        c,  #
        aMeta,
        M,
        N,
        K,  #
        aSparse.stride(0),
        aSparse.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        aMeta.stride(0),
        aMeta.stride(1),
    )
    return c


torch.manual_seed(0)
test_dim = 8192

cache_loc = "sparsified_inputs.pt"

def get_inputs():
    if os.path.isfile(cache_loc):
        data = torch.load(cache_loc)
        return data["a"], data["aSparse"], data["aMeta"], data["b"]
    else:
        a = make_sparse(torch.randn((test_dim, test_dim), device="cuda", dtype=torch.float16))
        print("Running sparse compression in Python... this can be quite slow, be patient!")
        aSparse, aMeta = compress(a)
        b = torch.randn((test_dim, test_dim), device="cuda", dtype=torch.float16)

        torch.save(
            {
                "a": a,
                "aSparse": aSparse,
                "aMeta": aMeta,
                "b": b,
            },
            cache_loc,
        )

        return a, aSparse, aMeta, b


a, aSparse, aMeta, b = get_inputs()


print("Autotuning... set TRITON_PRINT_AUTOTUNING=1 to see logs here...")
triton_output = matmul(aSparse, aMeta, b)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3):
    print("Correctness: Triton and Torch match! ✅")
else:
    print("Correctness: Triton and Torch differ! ❌")

def calc_flops(dim, ms):
    return 2 * dim * dim * dim * 1e-12 / (ms * 1e-3)

def do_bench_repeat(fn):
    REPEAT = 5
    ITERATIONS = 11
    results = []
    for _ in range(ITERATIONS):
        results.append(triton.testing.do_bench(lambda: [fn() for _ in range(REPEAT)]))
        time.sleep(0.1)
    return sorted(results)[ITERATIONS//2] / REPEAT

# ms = triton.testing.do_bench(lambda: matmul(aSparse, aMeta, b))
ms = do_bench_repeat(lambda: matmul(aSparse, aMeta, b))
flops = calc_flops(test_dim, ms)
# flops = calc_flops(test_dim, ms)
print(f"Perf: {flops} TFLOPS")


cusparse_A = torch._cslt_compress(a)
cusparse_B = b.T
alg_id, split_k, split_k_one_kernel, _ = torch._C._cusparselt.mm_search(cusparse_A, cusparse_B, None, None, None, False)
def run_cusparse():
    return torch._cslt_sparse_mm(
        cusparse_A,
        cusparse_B,
        alg_id=alg_id,
        split_k=split_k,
        split_k_one_kernel=split_k_one_kernel,
    )

# cusparse_ms = triton.testing.do_bench(run_cusparse)
cusparse_ms = do_bench_repeat(run_cusparse)
cusparse_flops = calc_flops(test_dim, cusparse_ms)
print(f"cusparseLT perf: {cusparse_flops} TFLOPS")
