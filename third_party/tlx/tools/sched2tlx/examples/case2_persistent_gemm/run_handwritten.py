"""Run the hand-written persistent GEMM target on B200."""

from __future__ import annotations

import sys

import handwritten
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor


NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    # Use 128×128 blocks to avoid the TMEM blockM=128 encoding limit on
    # single-MMA-group kernels. The persistent structure is what we're
    # testing here, not the exact tile shape.
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NUM_SMEM_BUFFERS = 2

    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 1024, 16384),
    ]
    failed = 0
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)

        # Build descriptors: a as [M, K] block [BM, BK]; b as [N, K] block [BN, BK]
        # (note B is laid out as N×K so TMA can load by [pid_n, k] coords).
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
        # B as [N, K] requires we view b's transpose contiguously — simpler:
        # let TMA load [BN, BK] from b's transposed view.
        b_t = b.t().contiguous()  # [N, K]
        b_desc = TensorDescriptor.from_tensor(b_t, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])

        grid = (NUM_SMS,)
        handwritten.matmul_kernel[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            NUM_SMS=NUM_SMS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

        ref = torch.matmul(a, b)
        err = (c.float() - ref.float()).abs().max().item()
        rel = err / ref.float().abs().max().item()
        ok = rel < 5e-3
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] M={M} N={N} K={K}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
