import argparse

import torch

import triton

from triton.language.extra.tlx.tutorials.amd_pa_decode import (
    pa_decode_tlx as _pa_decode_tlx,
    build_inputs as _build_inputs,
)

from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Fixed decode problem geometry (GQA, bf16 KV), matching the paged-decode
# correctness case. The sweep varies batch x context x query_length.
NUM_KV_HEADS = 8
QUERY_GROUP_SIZE = 8
NUM_Q_HEADS = NUM_KV_HEADS * QUERY_GROUP_SIZE
HEAD_DIM = 128
PAGE_SIZE = 64

DECODE_METHODS = {
    "tlx": lambda out, q, kc, vc, ctx, bt, sm, qlen: _pa_decode_tlx(out, q, kc, vc, ctx, bt, sm, query_length=qlen),
}
DEFAULT_DECODE_VERSIONS = ["tlx"]


def create_benchmark(versions, qlen):
    line_vals = list(versions)
    line_names = list(versions)
    # (BATCH, N_CTX)
    x_vals = [
        (1, 8192),
        (8, 8192),
        (32, 8192),
        (128, 8192),
        (1, 32768),
        (8, 32768),
        (32, 32768),
        (8, 131072),
    ]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["BATCH", "N_CTX"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel="TB/s (effective HBM read)",
            plot_name=f"paged-decode-performance-bf16-qlen{qlen}",
            args={},
        ))
    def benchmark(BATCH, N_CTX, provider):
        sm_scale = 1.0 / (HEAD_DIM**0.5)
        # Bound physical KV memory for large sweeps via a shared page pool.
        pool = 4 * ((N_CTX + PAGE_SIZE - 1) // PAGE_SIZE) + 16
        q, kc, vc, ctx, bt = _build_inputs(BATCH, [N_CTX] * BATCH, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
                                           query_length=qlen, device=DEVICE, pool_pages=pool)
        out = torch.empty_like(q)
        fn = DECODE_METHODS[provider]
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(out, q, kc, vc, ctx, bt, sm_scale, qlen),
                                                     quantiles=quantiles, warmup=100, rep=200)

        # Decode reads the whole KV cache once (K + V, bf16): report effective
        # HBM read bandwidth, the meaningful metric for this memory-bound op.
        kv_bytes = 2 * BATCH * NUM_KV_HEADS * N_CTX * HEAD_DIM * 2
        tbps = lambda ms: kv_bytes * 1e-12 / (ms * 1e-3)
        return tbps(ms), tbps(max_ms), tbps(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD paged-attention decode")
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        choices=list(DECODE_METHODS.keys()),
        help=f"Run only the specified version(s). Choices: {list(DECODE_METHODS.keys())}",
    )
    parser.add_argument(
        "--qlens",
        type=int,
        nargs="+",
        default=[1],
        help="Query lengths to sweep (multi-token prediction: 1-4).",
    )
    args = parser.parse_args()

    if is_hip():
        versions = args.version if args.version else DEFAULT_DECODE_VERSIONS
        print(f"Running paged-decode benchmarks for: {versions}, qlens={args.qlens}")
        for qlen in args.qlens:
            print(f"\n=== query_length = {qlen} ===")
            create_benchmark(versions, qlen).run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
