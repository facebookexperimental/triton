import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.amd_addmm_glu import (
    KERNEL_REGISTRY as _ADDMM_GLU_KERNELS,
    pytorch_baseline as _pytorch_baseline,
    M as _M,
    N as _N,
)

from triton._internal_testing import is_hip_cdna4

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Benchmarks for the TLX AMD fused addmm + GLU kernels on gfx950 (out = x + x*y,
# x = A@B + bias). Fixed M=1024, N=21568; K swept. Run with:
#   third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_addmm_glu_perf.py
#
# Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
#
# Reference gfx950/MI350 baselines (fp16, M=1024, N=21568), for regression context:
#   K=256  : tlx_baseline 237, tlx_simple_async 169, tlx_optimized_async 181, tlx_optimized 280, tlx_persistent 298
#   K=512  : tlx_baseline 355, tlx_simple_async 269, tlx_optimized_async 279, tlx_optimized 442, tlx_persistent 449
#   K=1024 : tlx_baseline 459, tlx_simple_async 349, tlx_optimized_async 406, tlx_optimized 597, tlx_persistent 613

ref_lib = "torch"  # naive matmul + add, then GLU

# Registry of available kernels (tlx_baseline, tlx_simple_async,
# tlx_optimized_async, tlx_optimized, tlx_persistent).
ADDMM_GLU_METHODS = dict(_ADDMM_GLU_KERNELS)


def create_benchmark(versions):
    line_vals = [ref_lib] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[256, 512, 1024],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name="amd-addmm-glu-performance-fp16",
            args={"M": _M, "N": _N},
        ))
    def benchmark(M, N, K, provider):
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib:
            fn = lambda: _pytorch_baseline(bias, a, b, y)
        else:
            kernel = ADDMM_GLU_METHODS[provider]
            fn = lambda: kernel(a, b, bias, y)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=25, rep=100)

        def tflops(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD addmm + GLU implementations (gfx950)")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(ADDMM_GLU_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(ADDMM_GLU_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_hip_cdna4():
        versions = args.version if args.version else list(ADDMM_GLU_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no gfx950 (CDNA4) GPU found.")
