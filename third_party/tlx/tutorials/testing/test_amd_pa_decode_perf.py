import argparse
import math

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
HEAD_DIM = 64
PAGE_SIZE = 16

DECODE_METHODS = ("tlx", "aiter")
DEFAULT_DECODE_VERSIONS = list(DECODE_METHODS)


def _pack_aiter_kv_cache(key_cache, value_cache):
    """Convert [block, head, page, dim] caches to AITER's packed layouts."""
    x = 16 // key_cache.element_size()
    num_blocks, num_kv_heads, page_size, head_dim = key_cache.shape
    assert head_dim % x == 0
    assert page_size % x == 0

    key_cache = key_cache.view(num_blocks, num_kv_heads, page_size, head_dim // x, x)
    key_cache = key_cache.permute(0, 1, 3, 2, 4).contiguous()
    value_cache = value_cache.view(num_blocks, num_kv_heads, page_size // x, x, head_dim)
    value_cache = value_cache.permute(0, 1, 2, 4, 3).contiguous()
    return key_cache, value_cache


def _make_decode_fn(provider, out, q, kc, vc, ctx, bt, sm_scale, qlen):
    if provider == "tlx":
        return lambda: _pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=qlen)

    from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon

    kc, vc = _pack_aiter_kv_cache(kc, vc)
    num_seqs = q.shape[0] // qlen
    equivalent_group_size = qlen * QUERY_GROUP_SIZE
    context_partition_size = 256
    max_context_partition_num = math.ceil(int(ctx.max().item()) / context_partition_size)
    workspace_shape = (num_seqs, NUM_KV_HEADS, max_context_partition_num, equivalent_group_size)
    exp_sums = torch.empty(workspace_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty((*workspace_shape, HEAD_DIM), dtype=q.dtype, device=q.device)

    return lambda: pa_decode_gluon(
        output=out,
        query=q,
        key_cache=kc,
        value_cache=vc,
        context_lengths=ctx,
        block_tables=bt,
        softmax_scale=sm_scale,
        query_length=qlen,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=q.dtype,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sliding_window=0,
        ps=False,
    )


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
        fn = _make_decode_fn(provider, out, q, kc, vc, ctx, bt, sm_scale, qlen)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=100, rep=200)

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
        choices=list(DECODE_METHODS),
        help=f"Run only the specified version(s). Choices: {list(DECODE_METHODS)}",
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
            report = create_benchmark(versions, qlen)
            if "tlx" in versions and "aiter" in versions:
                df = report.run(return_df=True)
                ylabel = "TB/s (effective HBM read)"
                df["aiter/tlx speedup"] = df[f"aiter ({ylabel})"] / df[f"tlx ({ylabel})"]
                print(f"paged-decode-performance-bf16-qlen{qlen}:")
                print(df.to_string())
            else:
                report.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
