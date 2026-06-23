import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.amd_gemm_warp_pipeline import (
    matmul as _amd_gemm_warp_pipeline, )
from triton.language.extra.tlx.tutorials.amd_gemm_pipelined import (
    matmul as _amd_gemm_pipelined, )

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Known-good config for the warp-pipeline kernel (mirrors test_correctness.py).
# The pipelined kernel is autotuned, so it takes no config.
_WARP_PIPELINE_CONFIG = {
    "BLOCK_M": 256,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "NUM_BUFFERS": 3,
    "num_warps": 8,
}

# Registry of available matmul implementations.
MATMUL_METHODS = {
    "warp_pipeline": lambda a, b: _amd_gemm_warp_pipeline(a, b, config=_WARP_PIPELINE_CONFIG),
    "pipelined": lambda a, b: _amd_gemm_pipelined(a, b),
}

ref_lib = "rocBLAS"
"""
This script is used for benchmarking the performance of TLX AMD GEMM tutorial kernels.
It's recommended to run with `third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_gemm_perf.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def create_benchmark(versions, dtype=torch.float16):
    line_vals = [ref_lib.lower()] + versions
    line_names = [ref_lib] + versions
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2048, 4096, 8192],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name=f"amd-matmul-performance-{dtype_name}",
            args={},
        ))
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=dtype)
        b = torch.randn((K, N), device=DEVICE, dtype=dtype)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, warmup=500,
                                                         rep=500)
        elif provider in MATMUL_METHODS:
            matmul = MATMUL_METHODS[provider]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles, warmup=500, rep=500)

        def tflops(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD GEMM implementations")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(MATMUL_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(MATMUL_METHODS.keys())}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type for the benchmark (default: fp16)",
    )
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if is_hip():
        versions = args.version if args.version else list(MATMUL_METHODS.keys())
        print(f"Running benchmarks for: {versions} (dtype={args.dtype})")
        benchmark = create_benchmark(versions, dtype=dtype)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
