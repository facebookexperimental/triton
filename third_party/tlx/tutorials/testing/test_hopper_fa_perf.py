import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined_pingpong_persistent import (
    attention as _attention_ws_pipelined_pingpong_persistent, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined_pingpong import (
    attention as _attention_ws_pipelined_pingpong, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws_pipelined import (
    attention as _attention_ws_pipelined, )
from triton.language.extra.tlx.tutorials.hopper_fa_ws import (
    attention as _attention_ws, )

from triton._internal_testing import is_hopper

DEVICE = triton.runtime.driver.active.get_active_torch_device()

ATTENTION_METHODS = {
    "ws_pipelined_pingpong_persistent": _attention_ws_pipelined_pingpong_persistent,
    "ws_pipelined_pingpong": _attention_ws_pipelined_pingpong,
    "ws_pipelined": _attention_ws_pipelined,
    "ws": _attention_ws,
}

ref_lib = "SDPA"
"""
This script is used for benchmarking the performance of TLX tutorial kernels.
It's recommended to run with `third_party/tlx/denoise.sh third_party/tlx/tutorials/hopper_fa_perf_test.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def create_benchmark(versions):
    line_vals = [ref_lib.lower()] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name="flash-attention-performance-fp16",
            args={"BATCH": 4, "H": 32, "HEAD_DIM": 128},
        ))
    def benchmark(BATCH, H, N_CTX, HEAD_DIM, provider):
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        sm_scale = 1.3
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False),
                quantiles=quantiles,
                warmup=500,
                rep=500,
            )
        elif provider in ATTENTION_METHODS:
            attention = ATTENTION_METHODS[provider]
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: attention(q, k, v, sm_scale),
                quantiles=quantiles,
                warmup=500,
                rep=500,
            )

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX Hopper Flash Attention implementations")
    parser.add_argument(
        "--version",
        type=str,
        choices=list(ATTENTION_METHODS.keys()),
        help=f"Run only the specified version. Choices: {list(ATTENTION_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_hopper():
        versions = [args.version] if args.version else list(ATTENTION_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Hopper GPU found.")
