import torch

import triton

from triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent_mxfp8 import (
    attention as _attention_ws_pipelined_persistent_mxfp8,
    generate_attention_inputs,
)

from triton._internal_testing import is_blackwell

DEVICE = triton.runtime.driver.active.get_active_torch_device()

ref_lib = "SDPA"
"""
This script is used for benchmarking the performance of the TLX MXFP8 flash attention kernel.
It's recommended to run with `third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_mxfp8_perf.py`

Facebook: If you are developing in fbsource, use tritonbench instead to collect perf numbers.
"""


def create_benchmark():

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=["ws_pipelined_persistent_mxfp8"],
            line_names=["ws_pipelined_persistent_mxfp8"],
            ylabel="TFLOPS",
            plot_name="flash-attention-performance-mxfp8",
            args={"BATCH": 4, "H": 32, "HEAD_DIM": 64, "causal": False},
        ))
    def benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, provider):
        shape = (BATCH, H, N_CTX, HEAD_DIM)
        sm_scale = 1.3
        quantiles = [0.5, 0.2, 0.8]
        dtype = torch.float8_e4m3fn
        (q, q_scale, _), (k, k_scale, _), (v, v_scale, _) = generate_attention_inputs(shape, DEVICE, dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _attention_ws_pipelined_persistent_mxfp8(q, k, v, q_scale, k_scale, v_scale, sm_scale, causal),
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
    if is_blackwell():
        print("Running MXFP8 flash attention benchmarks")
        benchmark = create_benchmark()
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no Blackwell GPU found.")
