"""Run emitter-generated case1 GEMM on GPU and verify vs torch."""

from __future__ import annotations

import sys

import generated  # local file produced by `python -m sched2tlx`
import torch
import triton


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)

    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # K not a multiple of 64 — exercises modulo schedule on non-clean trip count.
        (1024, 1024, 1024 - 64),  # K=960
        (1024, 1024, 1024 + 128),  # K=1152
    ]
    failed = 0
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
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

        nan = torch.isnan(c).sum().item()
        ref_fp16 = torch.matmul(a, b)
        ref_fp32 = torch.matmul(a.float(), b.float())
        err_vs_torch = (c.float() - ref_fp16.float()).abs().max().item()
        err_vs_fp32 = (c.float() - ref_fp32).abs().max().item()
        rel = err_vs_fp32 / ref_fp32.abs().max().item()
        ok = nan == 0 and rel < 5e-3
        marker = "PASS" if ok else "FAIL"
        print(
            f"[{marker}] M={M} N={N} K={K}  nan={nan}  "
            f"vs_torch_fp16={err_vs_torch:.3e}  vs_fp32={err_vs_fp32:.3e} (rel={rel:.3e})"
        )
        if not ok:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
