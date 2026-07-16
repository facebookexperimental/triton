"""Run emitter-generated case2 persistent GEMM on B200 vs torch."""

from __future__ import annotations

import importlib
import sys

import torch
import triton

try:
    import generated
except ModuleNotFoundError:  # buck par: module lives under the dotted package
    generated = importlib.import_module(
        (__package__ + ".generated") if __package__ else "generated"
    )

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)

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

        # The kernel builds its own TMA descriptors internally; pass raw
        # pointers + leading strides with the persistent (NUM_SMS,) grid.
        generated._gemm_persistent[(NUM_SMS,)](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            b.stride(0),
            c.stride(0),
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
        print(f"[{marker}] M={M} N={N} K={K}  nan={nan}  max abs={err:.3e}  rel={rel:.3e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
