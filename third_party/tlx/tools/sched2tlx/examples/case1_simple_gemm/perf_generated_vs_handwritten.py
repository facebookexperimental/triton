"""case1 simple GEMM: modulo-generated vs hand-written TLX-WS. B200, TFLOP/s.

Apples-to-apples: both kernels take raw pointers and build TMA descriptors
on-device (the emitter cannot receive host-built descriptors), and both use the
same SMEM ring depth (2, matching the generated kernel), so the comparison
isolates schedule/partition quality rather than the descriptor-build site.
"""

from __future__ import annotations
import sys
import generated
import handwritten
import torch
import triton

BM, BN, BK = 128, 128, 64


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def run(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    flops = 2 * M * N * K

    def gen():
        generated.gemm_kernel[grid](
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
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

    def hw():
        handwritten.gemm_kernel[grid](
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
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            NUM_SMEM_BUFFERS=2,
            NUM_TMEM_BUFFERS=1,
        )

    tg = flops / (triton.testing.do_bench(gen, warmup=25, rep=100) * 1e-3) / 1e12
    th = flops / (triton.testing.do_bench(hw, warmup=25, rep=100) * 1e-3) / 1e12
    print(f"({M},{N},{K})  hw {th:7.1f} | gen {tg:7.1f} TF  gen/hw={tg/th:.2f}")


def main():
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    for M, N, K in [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        run(M, N, K)
    return 0


if __name__ == "__main__":
    sys.exit(main())
