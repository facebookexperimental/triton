"""Benchmark the four addmm + GLU reference kernels vs PyTorch (torch.compile) and rocBLAS.

Runs V1-V4 at K in {256, 512, 1024} on the blog's shape (M=1024, N=21568, fp16),
checks each for correctness against a PyTorch reference, and prints throughput
(TFLOPS over the GEMM FLOPs, 2*M*N*K) plus speedups vs:
  * the addmm + gate path compiled with torch.compile(mode="max-autotune") -- the
    strongest PyTorch baseline (it fuses the gate into one pointwise kernel; the
    projection stays the hipBLASLt GEMM), and
  * a rocBLAS matmul (the projection A @ B only -- no bias, no gate).

Run:
    python bench.py                                  # library column = rocBLAS
    TORCH_BLAS_PREFER_HIPBLASLT=1 python bench.py    # library column = hipBLASLt
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


def addmm_glu(a, b, bias, y):
    """The addmm + gate path (X = A@B + bias, out = X + X*Y); compiled in main()."""
    x = torch.addmm(bias, a, b)
    return x + x * y


def tflops(ms, K):
    return (2 * M * N * K) / (ms * 1e-3) / 1e12


def median_ms(fn):
    return statistics.median(triton.testing.do_bench(fn, warmup=50, rep=200) for _ in range(3))


def main():
    lib = "rocBLAS" if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0" else "hipBLASLt"
    # Strongest PyTorch baseline: the addmm + gate path compiled with max-autotune.
    pytorch = torch.compile(addmm_glu, mode="max-autotune")
    print(f"M={M} N={N} fp16  |  PyTorch = torch.compile(max-autotune)  |  library = {lib} (matmul only)\n")
    for K in K_VALUES:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        y = torch.randn(M, N, device="cuda", dtype=torch.float16)
        ref = reference(bias, a, b, y)

        # Trigger compilation / autotuning for this shape before timing.
        for _ in range(6):
            pt_out = pytorch(a, b, bias, y)
        torch.cuda.synchronize()
        pt_ok = torch.allclose(pt_out.float(), ref.float(), atol=2e-1, rtol=2e-2)

        pt_tf = tflops(median_ms(lambda: pytorch(a, b, bias, y)), K)
        lib_tf = tflops(median_ms(lambda: torch.matmul(a, b)), K)

        print(f"=== K={K} ===")
        print(f"  {'kernel':<26}{'TFLOP/s':>9}{'vs PyTorch':>12}{('vs ' + lib):>12}  correct")
        print(f"  {'PyTorch (torch.compile)':<26}{pt_tf:>9.0f}{'1.00x':>12}{'-':>12}  {'OK' if pt_ok else 'FAIL'}")
        print(f"  {(lib + ' (matmul only)'):<26}{lib_tf:>9.0f}{(lib_tf / pt_tf):>11.2f}x{'1.00x':>12}")
        for name, run in VERSIONS:
            out = run(a, b, bias, y)
            ok = torch.allclose(out.float(), ref.float(), atol=2e-1, rtol=2e-2)
            tf = tflops(median_ms(lambda: run(a, b, bias, y)), K)
            print(f"  {name:<26}{tf:>9.0f}{(tf / pt_tf):>11.2f}x{(tf / lib_tf):>11.2f}x  {'OK' if ok else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
