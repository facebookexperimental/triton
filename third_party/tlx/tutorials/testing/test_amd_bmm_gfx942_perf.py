"""Perf benchmark for the TLX MI300X (gfx942 / CDNA3) batched GEMM tutorial.

Compares ``amd_bmm_gfx942`` against **aten** (``torch.bmm`` -> hipBLASLt).

Inputs are SHARED-A (one (M, K) broadcast over the batch, ``a.stride(0) == 0``),
matching ``amd_bmm.py``'s convention: hipBLASLt reads A once and keeps it
L2-resident, so benchmarking distinct-A would flatter TLX.

The kernel autotunes, so the first call per shape pays for the search. Set
``TRITON_PRINT_AUTOTUNING=1`` to see the winning tile and ring depth.

Recommended:
    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_amd_bmm_gfx942_perf.py

``denoise.sh`` pins NUMA; its clock lock is nvidia-smi based and is skipped on
AMD, so pick an idle GPU with ``rocm-smi`` and pin via ``HIP_VISIBLE_DEVICES``.

Facebook: If you are developing in fbsource, use tritonbench instead to collect
perf numbers.
"""

import argparse
import os
import statistics
import time

# Keep the LLVM post-RA machine scheduler from re-ordering the hand-written
# load / MFMA / local_store order in the hot loop. Set before triton is imported;
# setdefault() lets an explicit env value win, so you can A/B it.
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

# Silence the per-launch "no C dispatcher, falling back to Python launch" warning.
# TRITON_USE_C_DISPATCHER defaults to on, but the C dispatcher needs the NVIDIA
# backend's asm["launch_metadata"] key, which the AMD backend never emits -- so on
# gfx942 it can never be built and every launch warns. Turning the knob off skips
# a branch that could not have succeeded; the launch path and the compiled kernel
# are unchanged.
os.environ.setdefault("TRITON_USE_C_DISPATCHER", "0")

import torch  # noqa: E402

import triton  # noqa: E402

from triton.language.extra.tlx.tutorials.amd_bmm_gfx942 import (  # noqa: E402
    bmm as _amd_bmm_gfx942, make_bmm_inputs as _make_bmm_inputs,
)

from triton._internal_testing import is_hip_cdna3  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()

REF = "aten"

METHODS = {
    "tlx_gfx942": lambda a, b: _amd_bmm_gfx942(a, b),
}

# (M, N, K, batch). Small per-matrix tiles with a large batch is the regime BMM
# actually shows up in, so the batch carries the parallelism rather than M/N.
SHAPES = [
    (256, 256, 256, 64),
    (512, 512, 512, 32),
    (1024, 1024, 1024, 16),
    (2048, 2048, 2048, 8),
    (128, 128, 4096, 64),
]


def measure(fn):
    """Measure in bursts to avoid throttling."""
    _ = triton.testing.do_bench(fn, warmup=0, rep=10)
    times = []
    for _ in range(10):
        time.sleep(1)
        times.append(triton.testing.do_bench(fn, rep=10))
    return statistics.median(times)


def tflops(M, N, K, B, ms):
    return 2 * B * M * N * K * 1e-12 / (ms * 1e-3)


def make_inputs(M, N, K, B, dtype):
    return _make_bmm_inputs(B, M, N, K, DEVICE, dtype=dtype)


def create_benchmark(versions, dtype):
    line_vals = [REF] + versions
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K", "B"],
            x_vals=SHAPES,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            ylabel="TFLOPS",
            plot_name=f"amd-gfx942-bmm-performance-{dtype_name}",
            args={},
        ))
    def benchmark(M, N, K, B, provider):
        a, b = make_inputs(M, N, K, B, dtype)
        if provider == REF:
            ms = measure(lambda: torch.bmm(a, b))
        else:
            fn = METHODS[provider]
            ms = measure(lambda: fn(a, b))
        return tflops(M, N, K, B, ms)

    return benchmark


def run_speedup_table(versions, dtype):
    """Print TFLOPS and the speedup of each TLX version over aten."""
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    header = f"{'M':>6} {'N':>6} {'K':>6} {'B':>5} {REF + ' TFLOPS':>14}"
    for v in versions:
        header += f" {v + ' TFLOPS':>16} {'vs ' + REF:>10}"
    print(f"\n=== amd_bmm_gfx942 vs {REF} ({dtype_name}) ===")
    print(header)
    for M, N, K, B in SHAPES:
        a, b = make_inputs(M, N, K, B, dtype)
        ref_ms = measure(lambda: torch.bmm(a, b))
        row = f"{M:>6} {N:>6} {K:>6} {B:>5} {tflops(M, N, K, B, ref_ms):>14.1f}"
        for v in versions:
            fn = METHODS[v]
            ms = measure(lambda: fn(a, b))
            row += f" {tflops(M, N, K, B, ms):>16.1f} {ref_ms / ms:>9.2f}x"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the TLX gfx942 BMM tutorial against aten")
    parser.add_argument("--version", type=str, nargs="+", choices=list(METHODS.keys()),
                        help=f"Run only the specified version(s). Choices: {list(METHODS.keys())}")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--table", action="store_true", help="Print a TFLOPS + speedup-vs-aten table")
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if not is_hip_cdna3():
        print("Skipping benchmarks: this script targets AMD gfx942 (MI300X / CDNA3).")
    else:
        versions = args.version if args.version else list(METHODS.keys())
        print(f"Running benchmarks for: {versions} (dtype={args.dtype})")
        if args.table:
            run_speedup_table(versions, dtype)
        else:
            create_benchmark(versions, dtype).run(print_data=True)
