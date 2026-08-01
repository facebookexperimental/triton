"""Benchmark the four addmm + GLU reference kernels vs PyTorch (unfused) and rocBLAS.

Runs V1-V4 at K in {256, 512, 1024} on the blog's shape (M=1024, N=21568, fp16),
checks each for correctness against a PyTorch reference, and prints throughput
(TFLOP/s over the GEMM FLOPs, 2*M*N*K) plus speedups vs:
  * an eager PyTorch unfused path (`torch.addmm` then the gate as a separate op), and
  * a rocBLAS matmul (the projection A @ B only -- no bias, no gate).

Numbers are intentionally NOT committed; run this yourself. See the README for the
compiler caveat (the current main compiler regresses these kernels vs the numbers
reported in the blog).

Run:
    python bench.py                                  # library column = rocBLAS
    TORCH_BLAS_PREFER_HIPBLASLT=1 python bench.py    # library column = hipBLASLt

DO NOT MERGE - reference code accompanying the blog.
"""
import os
# Force the rocBLAS backend for the library-GEMM column (must be set before torch import).
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import statistics
import torch
import triton

import v1_register_staged as v1
import v2_direct_to_lds as v2
import v3_deep_pipeline_persistent as v3
import v4_fused_epilogue as v4

M, N = v1.M, v1.N
K_VALUES = (256, 512, 1024)
VERSIONS = [("V1", v1.run), ("V2", v2.run), ("V3", v3.run), ("V4", v4.run)]


def reference(bias, a, b, y):
    """Correctness reference: out = (A@B + bias) + (A@B + bias) * Y, fp32 accum."""
    x = torch.matmul(a, b).to(torch.float32) + bias.to(torch.float32)[None, :]
    return (x + x * y.to(torch.float32)).to(torch.float16)


def pytorch_unfused(bias, a, b, y):
    """Eager unfused path: addmm (matmul + bias), then the gate as a separate op."""
    x = torch.addmm(bias, a, b)
    return x + x * y


def tflops(ms, K):
    return (2 * M * N * K) / (ms * 1e-3) / 1e12


def median_ms(fn):
    return statistics.median(triton.testing.do_bench(fn, warmup=50, rep=200) for _ in range(3))


def main():
    lib = "rocBLAS" if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0" else "hipBLASLt"
    print(f"M={M} N={N} fp16  |  library column = {lib}  (matmul only)\n")
    for K in K_VALUES:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.randn(M, N, device="cuda", dtype=torch.float16)
        ref = reference(bias, a, b, y)

        pt_tf = tflops(median_ms(lambda: pytorch_unfused(bias, a, b, y)), K)
        lib_tf = tflops(median_ms(lambda: torch.matmul(a, b)), K)

        print(f"=== K={K} ===")
        print(f"  {'kernel':<24}{'TFLOP/s':>9}{'vs PyTorch':>12}{('vs ' + lib):>12}  correct")
        print(f"  {'PyTorch unfused':<24}{pt_tf:>9.0f}{'1.00x':>12}{'-':>12}")
        print(f"  {(lib + ' (matmul only)'):<24}{lib_tf:>9.0f}{(lib_tf / pt_tf):>11.2f}x{'1.00x':>12}")
        for name, run in VERSIONS:
            out = run(a, b, bias, y)
            ok = torch.allclose(out.float(), ref.float(), atol=2e-1, rtol=2e-2)
            tf = tflops(median_ms(lambda: run(a, b, bias, y)), K)
            print(f"  {name:<24}{tf:>9.0f}{(tf / pt_tf):>11.2f}x{(tf / lib_tf):>11.2f}x  {'OK' if ok else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
