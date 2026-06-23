import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.amd_fa_pipelined import (
    attention as _amd_fa_pipelined, )
from triton.language.extra.tlx.tutorials.amd_fa_persistent import (
    attention as _amd_fa_persistent, )

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Reference gfx950/MI350 baselines (bf16, H=64, TFLOPS), for regression context.
# key: (kernel, B, D, N, causal) -> TFLOPS
#   N=4096  (B=1): simple   64/128 F 614/646  T 344/341 ; prefetch 64/128 F 677/782 T 334/447 ;
#                  persistent 64/128 F 695/828 T 620/697
#   N=8192  (B=2): simple   64/128 F 668/704  T 380/372 ; prefetch 64/128 F 713/833 T 358/472 ;
#                  persistent 64/128 F 726/879 T 694/794
#   N=16384 (B=1): simple   64/128 F 668/724  T 432/475 ; prefetch 64/128 F 732/861 T 439/544 ;
#                  persistent 64/128 F 744/894 T 741/837

ATTENTION_METHODS = {
    "simple":
    lambda q, k, v, sm_scale, causal: _amd_fa_pipelined(q, k, v, sm_scale, causal),
    "prefetch":
    lambda q, k, v, sm_scale, causal: _amd_fa_pipelined(
        q, k, v, sm_scale, causal, config={
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "num_warps": 8,
            "PREFETCH": True,
        }),
    "persistent":
    lambda q, k, v, sm_scale, causal: _amd_fa_persistent(q, k, v, sm_scale, causal),
}

ref_lib = "SDPA"


def create_benchmark(versions):
    line_vals = [ref_lib.lower()] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N_CTX", "HEAD_DIM"],
            x_vals=[(1024, 64), (4096, 64), (8192, 64), (1024, 128), (4096, 128), (8192, 128)],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name="flash-attention-performance-fp16",
            args={"BATCH": 1, "H": 64, "causal": False},
        ))
    def benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, provider):
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=DEVICE, dtype=torch.float16).requires_grad_()
        sm_scale = 1.3
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
        elif provider in ATTENTION_METHODS:
            attention = ATTENTION_METHODS[provider]
            fn = lambda: attention(q, k, v, sm_scale, causal)

        ms, min_ms, max_ms = triton.testing.do_bench(
            fn,
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
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD Flash Attention implementations")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(ATTENTION_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(ATTENTION_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_hip():
        versions = args.version if args.version else list(ATTENTION_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
