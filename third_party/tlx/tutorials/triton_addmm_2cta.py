"""
End-to-end test for matmul with cluster_dims=(2,1,1) using tl.dot() + TMA descriptors.

Tests that:
  1. Compilation succeeds with cluster_dims=(2,1,1) and num_ctas=2
  2. The Insert2CTASync pass runs without crashing
  3. The kernel produces correct results on a Blackwell GPU

NOTE: Whether canUseTwoCTAs() triggers depends on PlanCTA's CTASplitNum
assignment. With the current PlanCTA logic, cluster_dims=(2,1,1) produces
CTASplitNum=[1,2] (N-split), not [2,1] (M-split) required for 2-CTA MMA.
The pass logic itself is verified by the MLIR lit test
(test/Hopper/TwoCTA/insert_2cta_sync.mlir).

Usage:
  CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 \\
    buck2 run @fbcode//mode/opt -m ovr_config//triton:beta \\
      -c fbcode.enable_gpu_sections=true -c fbcode.platform010_cuda_version=12.8 \\
      --no-remote-cache \\
      '//third-party/triton/beta/triton:py_2cta_sync_blackwell_test' \\
      -- <path-to-this-file>
"""

import sys
import torch
import triton
import triton.language as tl
from typing import Optional

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    return torch.empty(size, dtype=torch.int8, device="cuda")


@triton.jit
def matmul_2cta_kernel(
    a_ptr,
    stride_am,
    stride_ak,
    b_ptr,
    stride_bk,
    stride_bn,
    c_ptr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    desc_a = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                       block_shape=[BLOCK_M, BLOCK_K])
    desc_b = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                       block_shape=[BLOCK_K, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K
        a = desc_a.load([offs_am, offs_k])
        b = desc_b.load([offs_k, offs_bn])
        acc = tl.dot(a, b, acc, two_ctas=True)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def matmul_2cta(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    triton.set_allocator(alloc_fn)
    grid = (M // 128, N // 128)
    h = matmul_2cta_kernel[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=64,
        M=M,
        N=N,
        K=K,
        num_stages=2,
        num_warps=4,
        num_ctas=2,
        cluster_dims=(2, 1, 1),
    )
    return c, h


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}, SM {cap[0]}.{cap[1]}")
    if cap[0] < 10:
        print("SKIPPED: Requires Blackwell (SM100+) GPU")
        sys.exit(0)

    M, N, K = 256, 256, 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    print("Compiling kernel with num_ctas=2, cluster_dims=(2,1,1)...")
    c, h = matmul_2cta(a, b)
    print("Compilation + execution OK")

    ttgir = h.asm["ttgir"]
    print("\n=== IR Check ===")
    print(f"  tc_gen5_mma: {'OK' if 'tc_gen5_mma' in ttgir else 'MISSING'}")
    print(f"  two_ctas: {'OK' if 'two_ctas' in ttgir else 'N/A (PlanCTA chose N-split, not M-split)'}")
    if "two_ctas" in ttgir:
        for kw in ["cluster_id", "map_to_remote_buffer", "arrive_barrier"]:
            print(f"  {kw}: {'OK' if kw in ttgir else 'MISSING'}")

    ref = torch.matmul(a, b)
    max_diff = (c.float() - ref.float()).abs().max().item()
    print("\n=== Correctness ===")
    print(f"  max_diff: {max_diff:.6f}")
    ok = torch.allclose(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"  result: {'PASSED' if ok else 'FAILED'}")

    if ok:
        print("\n=== TEST PASSED ===")
    else:
        print("\n=== TEST FAILED ===")
        sys.exit(1)
