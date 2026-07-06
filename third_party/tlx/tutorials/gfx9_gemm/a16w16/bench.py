"""Benchmark harness matching Gluon's bench.py format."""

import argparse
import importlib
import torch
import triton

VERSION_MAP = {
    0: "v0_naive",
    1: "v1_buffer_load",
    2: "v2_async_copy",
    3: "v3_lds",
    4: "v4_global_prefetch",
    5: "v5_local_prefetch",
    6: "v6_loop_unroll",
    7: "v7_slice",
    8: "v8_warp_pipeline",
    9: "v9_beyond_hotloop",
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_x_vals():
    return [
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
    ]


def main():
    parser = argparse.ArgumentParser(description="TLX GEMM benchmark")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--version", type=int, default=0, choices=range(0, 10))
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    version_dir = VERSION_MAP[args.version]
    module = importlib.import_module(f"{version_dir}.matmul_kernel")
    matmul = module.matmul

    sizes = get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]

    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    # Correctness
    failed = False
    for M, N, K in sizes:
        torch.manual_seed(0)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
        ref = torch.matmul(a, b)
        c = matmul(a, b)
        max_err = (c - ref).abs().max().item()
        ok = torch.allclose(c, ref, atol=args.atol, rtol=args.rtol)
        failed |= not ok
        print(f"[{version_dir}] M={M} N={N} K={K}: {'OK' if ok else 'FAIL'} max_err={max_err:.4f}")

    # Performance
    if args.no_bench:
        if failed:
            raise SystemExit(1)
        return

    print(f"\n{version_dir}:")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'rocBLAS':>8s}  {'TLX':>8s}")
    for M, N, K in sizes:
        torch.manual_seed(0)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T
        ms_ref = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=args.warmup, rep=args.rep)
        ms_tlx = triton.testing.do_bench(lambda: matmul(a, b), warmup=args.warmup, rep=args.rep)
        print(f"{M:6d} {N:6d} {K:6d}  {tflops(ms_ref,M,N,K):7.1f}T  {tflops(ms_tlx,M,N,K):7.1f}T")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
