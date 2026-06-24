import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.amd_mxfp_gemm_tdm_pipelined import (
    matmul as _amd_mxfp_gemm_tdm_pipelined,
    pack_scale,
)
from triton.tools.mxfp import MXScaleTensor

from triton._internal_testing import is_hip_gfx1250

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Benchmarks for the TLX MXFP TDM-pipelined GEMM kernel on AMD gfx1250.
# Run with:
#   third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_mxfp_gemm_perf.py
#   third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_mxfp_gemm_perf.py --transpose-b
#
# Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.

SCALE_BLOCK = 32

# Registry of available matmul implementations.
MATMUL_METHODS = {
    "tdm_pipelined": _amd_mxfp_gemm_tdm_pipelined,
}


def _init_fp8_e5m2(rows, cols):
    return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e5m2)


def create_benchmark(versions, transpose_b=False):
    suffix = "-bT" if transpose_b else ""

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2048, 4096, 8192],
            line_arg="provider",
            line_vals=versions,
            line_names=versions,
            ylabel="TFLOPS",
            plot_name=f"mxfp-matmul-performance-e5m2{suffix}",
            args={},
        ))
    def benchmark(M, N, K, provider):
        a = _init_fp8_e5m2(M, K).to(DEVICE)
        b_rows, b_cols = (N, K) if transpose_b else (K, N)
        b = _init_fp8_e5m2(b_rows, b_cols).to(DEVICE)
        a_scale = pack_scale(MXScaleTensor(size=(M, triton.cdiv(K, SCALE_BLOCK))).random(high=32.0).data).to(DEVICE)
        b_scale = pack_scale(MXScaleTensor(size=(N, triton.cdiv(K, SCALE_BLOCK))).random(high=32.0).data).to(DEVICE)

        quantiles = [0.5, 0.2, 0.8]
        matmul = MATMUL_METHODS[provider]
        config = {"TRANSPOSE_B": transpose_b}
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, a_scale, b_scale, config=config),
            quantiles=quantiles,
            warmup=500,
            rep=500,
        )

        def tflops(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD MXFP GEMM implementations (gfx1250)")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(MATMUL_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(MATMUL_METHODS.keys())}",
    )
    parser.add_argument("--transpose-b", action="store_true", help="Benchmark the transposed-B layout")
    args = parser.parse_args()

    if is_hip_gfx1250():
        versions = args.version if args.version else list(MATMUL_METHODS.keys())
        print(f"Running benchmarks for: {versions} (transpose_b={args.transpose_b})")
        benchmark = create_benchmark(versions, transpose_b=args.transpose_b)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no gfx1250 GPU found.")
