"""Correctness + do_bench TFLOPS harness for the 8-wave TLX GEMM (inter_wave port)."""

import argparse
import os
import torch
import triton

from matmul_kernel import matmul, MIN_K, KERNEL_NAME

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


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
    parser.add_argument("--rep", type=positive_int, default=200,
                        help="timed duration in milliseconds; default: 200")
    parser.add_argument("--warmup", type=nonnegative_int, default=25,
                        help="warmup duration in milliseconds; default: 25")
    parser.add_argument("--seed", type=nonnegative_int, default=0,
                        help="deterministic input seed; default: 0")
    args = parser.parse_args()

    sizes = [tuple(s) for s in args.shape] if args.shape else get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]

    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    measurements = []
    for M, N, K in sizes:
        if K < MIN_K:
            print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: SKIPPED (K < {MIN_K})")
            continue
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(args.seed)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16, generator=generator)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16, generator=generator).T
        ref = torch.matmul(a, b)
        c = matmul(a, b)
        ok = torch.allclose(c, ref, atol=1e-1, rtol=0)
        print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: {'OK' if ok else 'FAIL'}")
        if not ok:
            continue
        ms_ref = triton.testing.do_bench(
            lambda: torch.matmul(a, b),
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
        ms_tlx = triton.testing.do_bench(
            lambda: matmul(a, b),
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
        measurements.append((M, N, K, ms_ref, ms_tlx))

    backend = "Wave" if os.environ.get("TRITON_DEFAULT_BACKEND") == "tlx_wave" else "LLVM"
    print(f"\n{KERNEL_NAME} ({backend}, seed={args.seed}; "
          f"triton median, {args.warmup}ms warmup/{args.rep}ms timed):")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'rocBLAS':>8s}  {backend:>8s}")
    for M, N, K, ms_ref, ms_tlx in measurements:
        print(f"{M:6d} {N:6d} {K:6d}  {tflops(ms_ref,M,N,K):7.1f}T  {tflops(ms_tlx,M,N,K):7.1f}T")


if __name__ == "__main__":
    main()
