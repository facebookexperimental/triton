"""Perf benchmark for the TLX MI300X (gfx942 / CDNA3) GEMM tutorial.

Compares ``amd_gemm_gfx942`` against **aten** (``torch.matmul``, which dispatches
to hipBLASLt / rocBLAS on ROCm) and, for context, the arch-generic TLX
``amd_gemm_pipelined`` kernel, and against the same kernel with the XCD remap
disabled.

Recommended:
    third_party/tlx/denoise.sh \
        python third_party/tlx/tutorials/testing/test_amd_gemm_gfx942_perf.py

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

import torch  # noqa: E402

import triton  # noqa: E402

from triton.language.extra.tlx.tutorials.amd_gemm_gfx942 import (  # noqa: E402
    matmul as _amd_gemm_gfx942, CONFIG_LARGE, pick_config, lds_bytes, CDNA3_LDS_BYTES,
)
from triton.language.extra.tlx.tutorials.amd_gemm_pipelined import (  # noqa: E402
    matmul as _amd_gemm_pipelined, )

from triton._internal_testing import is_hip_cdna3  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()

REF = "aten"

MATMUL_METHODS = {
    # Shipped behaviour: pick_config() selects the tile from the output shape.
    "gfx942": lambda a, b: _amd_gemm_gfx942(a, b),
    # Same tile, no XCD remap -- isolates what the chiplet remap buys.
    "gfx942_noxcd": lambda a, b: _amd_gemm_gfx942(a, b, config={**pick_config(a.shape[0], b.shape[1]), "NUM_XCDS": 1}),
    # Pinned to the big tile -- shows what the shape-aware default is worth.
    "gfx942_large": lambda a, b: _amd_gemm_gfx942(a, b, config=CONFIG_LARGE),
    # The arch-generic autotuned TLX kernel, for context.
    "pipelined": lambda a, b: _amd_gemm_pipelined(a, b),
}

# Square shapes plus two skinny/fat cases that stress the grid remap differently.
SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1024, 8192, 8192),
    (8192, 1024, 8192),
]

# Candidate tiles for --sweep. All fit the 64 KB CDNA3 LDS budget; the launcher
# asserts that, and configs that do not fit are filtered out below anyway.
SWEEP_CONFIGS = [{"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": g, "NUM_BUFFERS": nb, "num_warps": w}
                 for m, n in [(64, 64), (64, 128), (128, 64), (128, 128), (256, 128), (128, 256), (256, 256)]
                 for k in [32, 64]
                 for g in [1, 4, 8]
                 for nb in [2, 3]
                 for w in [4, 8]]


def measure(fn):
    """Measure in bursts to avoid throttling."""
    _ = triton.testing.do_bench(fn, warmup=0, rep=10)
    times = []
    for _ in range(10):
        time.sleep(1)
        times.append(triton.testing.do_bench(fn, rep=10))
    return statistics.median(times)


def tflops(M, N, K, ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def make_inputs(M, N, K, dtype):
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    return a, b


def create_benchmark(versions, dtype):
    line_vals = [REF] + versions
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=SHAPES,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            ylabel="TFLOPS",
            plot_name=f"amd-gfx942-matmul-performance-{dtype_name}",
            args={},
        ))
    def benchmark(M, N, K, provider):
        a, b = make_inputs(M, N, K, dtype)
        if provider == REF:
            ms = measure(lambda: torch.matmul(a, b))
        else:
            matmul = MATMUL_METHODS[provider]
            ms = measure(lambda: matmul(a, b))
        return tflops(M, N, K, ms)

    return benchmark


def run_speedup_table(versions, dtype):
    """Print TFLOPS and the speedup of each TLX version over aten."""
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    header = f"{'M':>6} {'N':>6} {'K':>6} {REF + ' TFLOPS':>14}"
    for v in versions:
        header += f" {v + ' TFLOPS':>16} {'vs ' + REF:>10}"
    print(f"\n=== amd_gemm_gfx942 vs {REF} ({dtype_name}) ===")
    print(header)
    for M, N, K in SHAPES:
        a, b = make_inputs(M, N, K, dtype)
        ref_ms = measure(lambda: torch.matmul(a, b))
        row = f"{M:>6} {N:>6} {K:>6} {tflops(M, N, K, ref_ms):>14.1f}"
        for v in versions:
            matmul = MATMUL_METHODS[v]
            ms = measure(lambda: matmul(a, b))
            row += f" {tflops(M, N, K, ms):>16.1f} {ref_ms / ms:>9.2f}x"
        print(row)


def run_sweep(dtype, shape):
    """Sweep SWEEP_CONFIGS on one shape and print the ranking against aten."""
    M, N, K = shape
    a, b = make_inputs(M, N, K, dtype)
    ref_ms = measure(lambda: torch.matmul(a, b))
    ref_tflops = tflops(M, N, K, ref_ms)
    print(f"\n=== config sweep, M={M} N={N} K={K}, {REF} = {ref_tflops:.1f} TFLOPS ===")

    results = []
    for cfg in SWEEP_CONFIGS:
        used = lds_bytes(cfg, a.element_size())
        if used > CDNA3_LDS_BYTES:
            continue
        if triton.cdiv(K, cfg["BLOCK_K"]) < cfg["NUM_BUFFERS"]:
            continue
        try:
            out = _amd_gemm_gfx942(a, b, config=cfg)
            torch.testing.assert_close(out, torch.matmul(a, b), atol=1e-2, rtol=1e-2)
            ms = measure(lambda: _amd_gemm_gfx942(a, b, config=cfg))
        except Exception as exc:  # noqa: BLE001 - a bad tile should not stop the sweep
            print(f"  skip {cfg}: {type(exc).__name__}: {exc}")
            continue
        results.append((tflops(M, N, K, ms), used, cfg))

    results.sort(key=lambda r: -r[0])
    print(f"{'TFLOPS':>8} {'vs ' + REF:>10} {'LDS KB':>7}  config")
    for tf, used, cfg in results:
        summary = (f"{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']} "
                   f"G{cfg['GROUP_M']} B{cfg['NUM_BUFFERS']} W{cfg['num_warps']}")
        print(f"{tf:>8.1f} {tf / ref_tflops:>9.2f}x {used / 1024:>7.1f}  {summary}")
    if results:
        print(f"\nbest:         {results[0][2]}")
        print(f"pick_config:  {pick_config(M, N)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the TLX gfx942 GEMM tutorial against aten")
    parser.add_argument("--version", type=str, nargs="+", choices=list(MATMUL_METHODS.keys()),
                        help=f"Run only the specified version(s). Choices: {list(MATMUL_METHODS.keys())}")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--sweep", action="store_true", help="Sweep SWEEP_CONFIGS on one shape instead of benchmarking")
    parser.add_argument("--sweep-shape", type=int, nargs=3, default=[4096, 4096, 4096], metavar=("M", "N", "K"))
    parser.add_argument("--table", action="store_true", help="Print a TFLOPS + speedup-vs-aten table")
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if not is_hip_cdna3():
        print("Skipping benchmarks: this script targets AMD gfx942 (MI300X / CDNA3).")
    elif args.sweep:
        run_sweep(dtype, tuple(args.sweep_shape))
    else:
        versions = args.version if args.version else list(MATMUL_METHODS.keys())
        print(f"Running benchmarks for: {versions} (dtype={args.dtype})")
        if args.table:
            run_speedup_table(versions, dtype)
        else:
            create_benchmark(versions, dtype).run(print_data=True)
