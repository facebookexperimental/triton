"""Run emitter-generated case2 persistent GEMM on B200 vs torch."""

from __future__ import annotations

import sys

import generated
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor


NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    shapes = [
        # K-heavy square shapes (original 6) — exercise the K-loop pipeline.
        (256, 256, 128),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (1024, 1024, 16384),
        # Epilogue-bound shapes (small K, large M·N) — perf-relevant for A.7
        # subtiling; baseline ~219/492/247 TF, subtile S=4 ~289/593/306 TF
        # (1.19-1.32× on B200). See perf_subtile_epi_bound.py.
        (8192, 8192, 64),
        (8192, 8192, 256),
        (16384, 16384, 64),
    ]
    failed = 0
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float16)

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
        # Generated uses [offs_bn, offs_k] for B → load order is [N, K], so
        # transpose to [K, N] view first then descriptor on [N, K].
        b_t = b.t().contiguous()  # [N, K]
        b_desc = TensorDescriptor.from_tensor(b_t, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])

        grid = (NUM_SMS,)
        # The kernel signature has 18 args from the deduped TensorDescriptor
        # flattening. We pass each TensorDescriptor as a single object — Triton
        # reflattens automatically into the 5 underlying fields per descriptor.
        generated.matmul_kernel_tma_persistent_simple[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
        )

        ref = torch.matmul(a, b)
        nan = torch.isnan(c).sum().item()
        err = (c.float() - ref.float()).abs().max().item()
        rel = err / max(ref.float().abs().max().item(), 1e-9)
        ok = nan == 0 and rel < 5e-3
        marker = "PASS" if ok else "FAIL"
        print(
            f"[{marker}] M={M} N={N} K={K}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}"
        )
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
