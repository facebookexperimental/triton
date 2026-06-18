import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.ikbo.ikbo_lce_triton import (
    create_inputs as _create_inputs,
    ikbo_lce as _ikbo_lce,
    lce_reference as _lce_reference,
)

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M, N, K_USER, K_CAND = 433, 256, 1178, 866
CAND_TO_USER_RATIO = 70

LCE_METHODS = {
    "ikbo_triton": _ikbo_lce,
}

ref_lib = "torch_decomposed"


def create_benchmark(versions):
    line_vals = [ref_lib] + versions
    line_names = [ref_lib] + versions

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B"],
            x_vals=[512 * i for i in range(1, 7)],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="Latency (ms)",
            plot_name="ikbo-lce-performance-fp16",
            args={},
        ))
    def benchmark(B, provider):
        torch.manual_seed(0)
        cw_c, cw_u, e_c, e_u, idx = _create_inputs(
            B,
            M,
            N,
            K_USER,
            K_CAND,
            CAND_TO_USER_RATIO,
            device=DEVICE,
        )
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib:
            fn = lambda: _lce_reference(cw_c, cw_u, e_c, e_u, idx)
        elif provider in LCE_METHODS:
            lce = LCE_METHODS[provider]
            fn = lambda: lce(cw_c, cw_u, e_c, e_u, idx)
        else:
            return 0, 0, 0

        ms, min_ms, max_ms = triton.testing.do_bench(
            fn,
            quantiles=quantiles,
            warmup=500,
            rep=500,
        )
        return ms, max_ms, min_ms

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IKBO LCE on AMD")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(LCE_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(LCE_METHODS.keys())}",
    )
    args = parser.parse_args()

    if is_hip():
        versions = args.version if args.version else list(LCE_METHODS.keys())
        print(f"Running benchmarks for: {versions}")
        benchmark = create_benchmark(versions)
        benchmark.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
