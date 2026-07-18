"""Correctness + do_bench TFLOPS harness for the 8-wave TLX GEMM (inter_wave port)."""

import argparse
import torch
import triton

from matmul_kernel import matmul, MIN_K, KERNEL_NAME

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_x_vals():
    return [
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
    ]


def main():
    parser = argparse.ArgumentParser(description="8-wave TLX GEMM benchmark")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--shape", type=int, nargs=3, action="append", metavar=("M", "N", "K"),
                        help="Custom M N K shape (repeatable). Overrides default sizes.")
    args = parser.parse_args()

    sizes = [tuple(s) for s in args.shape] if args.shape else get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]

    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    # Correctness
    for M, N, K in sizes:
        if K < MIN_K:
            print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: SKIPPED (K < {MIN_K})")
            continue
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
        ref = torch.matmul(a, b)
        c = matmul(a, b)
        ok = torch.allclose(c, ref, atol=1e-1, rtol=0)
        print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: {'OK' if ok else 'FAIL'}")

    # Performance
    print(f"\n{KERNEL_NAME}:")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'rocBLAS':>8s}  {'TLX':>8s}")
    for M, N, K in sizes:
        if K < MIN_K:
            continue
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
        ms_ref = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=200)
        ms_tlx = triton.testing.do_bench(lambda: matmul(a, b), rep=200)
        print(f"{M:6d} {N:6d} {K:6d}  {tflops(ms_ref,M,N,K):7.1f}T  {tflops(ms_tlx,M,N,K):7.1f}T")


if __name__ == "__main__":
    main()
