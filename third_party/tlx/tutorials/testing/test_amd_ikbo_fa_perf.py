import argparse
import random

import torch

import triton

from triton.language.extra.tlx.tutorials.ikbo.ikbo_fa_triton import (
    create_inputs as _create_inputs,
    ikbo_fa as _ikbo_fa,
)

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

FA_METHODS = {
    "ikbo_triton": _ikbo_fa,
}

ref_lib = "broadcast_sdpa"


def create_benchmark(versions):
    line_vals = [ref_lib] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["max_seq_len"],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TFLOPS",
            plot_name="ikbo-fa-performance-fp16",
            args={"B": 512, "n_seed": 64, "num_heads": 1, "d_head": 128},
        ))
    def benchmark(B, n_seed, num_heads, d_head, max_seq_len, provider):
        random.seed(0)
        torch.manual_seed(0)
        query, key, value, cand_to_user_index, cand_grid = _create_inputs(
            B,
            n_seed,
            num_heads,
            d_head,
            max_seq_len,
            device=DEVICE,
        )
        quantiles = [0.5, 0.2, 0.8]

        if provider == ref_lib:
            Bu = key.shape[0] // max_seq_len
            query_sdpa = query.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
            key_sdpa = key.view(Bu, max_seq_len, num_heads, d_head)
            key_broadcast = torch.index_select(
                key_sdpa,
                dim=0,
                index=cand_to_user_index,
            ).permute(0, 2, 1, 3)
            value_sdpa = value.view(Bu, max_seq_len, num_heads, d_head)
            value_broadcast = torch.index_select(
                value_sdpa,
                dim=0,
                index=cand_to_user_index,
            ).permute(0, 2, 1, 3)
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                query_sdpa,
                key_broadcast,
                value_broadcast,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        elif provider in FA_METHODS:
            fa = FA_METHODS[provider]
            fn = lambda: fa(
                query,
                key,
                value,
                cand_to_user_index,
                cand_grid,
                n_seed,
                num_heads,
                d_head,
                max_seq_len,
            )
        else:
            return 0, 0, 0

        ms, min_ms, max_ms = triton.testing.do_bench(
            fn,
            quantiles=quantiles,
            warmup=500,
            rep=500,
        )

        flops_per_matmul = 2.0 * B * num_heads * n_seed * max_seq_len * d_head
        total_flops = 2 * flops_per_matmul
        perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IKBO Flash Attention on AMD")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(FA_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(FA_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_hip():
        versions = args.version if args.version else list(FA_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
