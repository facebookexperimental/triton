import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as _tlx_matmul_ws, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul as _tlx_matmul_clc, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul as _tlx_matmul_pipelined, )
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import (
    matmul as _tlx_matmul_2cta, )

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Registry of available matmul implementations
MATMUL_METHODS = {
    "ws": _tlx_matmul_ws,
    "clc": _tlx_matmul_clc,
    "pipelined": _tlx_matmul_pipelined,
    "2cta": _tlx_matmul_2cta,
}

ref_lib = "cuBLAS"
"""
This script is used for benchmarking the performance of TLX tutorial kernels.
It's recommended to run with `third_party/tlx/denoise.sh third_party/tlx/tutorials/blackwell_gemm_perf_test.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def create_benchmark(versions):
    line_vals = [ref_lib.lower()] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2048, 4096, 8192],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name="matmul-performance-fp16",
            args={},
        ))
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, warmup=2000,
                                                         rep=2000)
        elif provider in MATMUL_METHODS:
            matmul = MATMUL_METHODS[provider]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles, warmup=2000,
                                                         rep=2000)

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX Blackwell GEMM implementations")
    parser.add_argument(
        "--version",
        type=str,
        choices=list(MATMUL_METHODS.keys()),
        help=f"Run only the specified version. Choices: {list(MATMUL_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_blackwell():
        versions = [args.version] if args.version else list(MATMUL_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
