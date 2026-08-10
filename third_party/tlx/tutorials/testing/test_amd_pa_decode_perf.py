import argparse
import math
from types import SimpleNamespace

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

DECODE_METHODS = (
    "aiter_common",
    "sglang",
    "aiter_gluon",
    "tlx_5d",
    "tlx_5d_streaming",
    "tlx",
)
# PR #2306 target comparison: standalone AITER common/HIP, SGLang's actual
# vectorized-5D wrapper (which calls AITER Gluon), and native TLX 5D.
DEFAULT_DECODE_VERSIONS = ["aiter_common", "sglang", "tlx_5d"]


def _make_decode_fn(
    provider,
    out,
    q,
    kc,
    vc,
    ctx,
    bt,
    sm_scale,
    qlen,
    max_context_len,
    num_kv_heads,
    query_group_size,
    head_dim,
):
    expected_ndim = 4 if provider == "tlx" else 5
    assert kc.ndim == vc.ndim == expected_ndim
    if provider in ("tlx", "tlx_5d", "tlx_5d_streaming"):

        def _run_tlx():
            return _pa_decode_tlx(
                out,
                q,
                kc,
                vc,
                ctx,
                bt,
                sm_scale,
                query_length=qlen,
                max_context_len=max_context_len,
                streaming_kv=True if provider == "tlx_5d_streaming" else None,
            )

        return _run_tlx

    num_seqs = q.shape[0] // qlen
    context_partition_size = 256
    max_context_partition_num = math.ceil(int(ctx.max().item()) / context_partition_size)

    if provider == "sglang":
        if qlen != 1:
            raise ValueError("SGLang's vectorized_5d decode wrapper only supports query_length=1")

        from sglang.srt.layers.attention.aiter_utils import forward_decode_vectorized_5d

        # Use the real SGLang wrapper, including its recommended split count,
        # ps=True kernel selection, metadata handling, and per-call workspaces.
        backend = SimpleNamespace(
            input_dtype=q.dtype,
            kv_cache_dtype=kc.dtype,
            k_scale=None,
            v_scale=None,
            forward_metadata=SimpleNamespace(kv_indices=bt, swa_page_table=None),
        )
        layer = SimpleNamespace(
            tp_k_head_num=num_kv_heads,
            tp_q_head_num=q.shape[1],
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            scaling=sm_scale,
            sliding_window_size=None,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(batch_size=num_seqs, seq_lens=ctx)

        return lambda: forward_decode_vectorized_5d(backend, q, layer, forward_batch, kc, vc, out, None)

    if provider == "aiter_gluon":
        from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon

        equivalent_group_size = qlen * query_group_size
        workspace_shape = (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            equivalent_group_size,
        )
        exp_sums = torch.empty(workspace_shape, dtype=torch.float32, device=q.device)
        max_logits = torch.empty_like(exp_sums)
        temporary_output = torch.empty((*workspace_shape, head_dim), dtype=q.dtype, device=q.device)

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

    # This is the path used by vLLM when shuffled KV cache is enabled. AITER
    # dispatches to its HIP kernel for D=64 and may select hand-written ASM for
    # D=128 when the launched head count is large enough.
    from aiter import paged_attention_common

    num_q_heads = q.shape[1]
    tmp_out = torch.empty(
        (num_seqs, num_q_heads, max_context_partition_num, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    exp_sums = torch.empty(
        (num_seqs, num_q_heads, max_context_partition_num),
        dtype=torch.float32,
        device=q.device,
    )
    max_logits = torch.empty_like(exp_sums)
    scale = torch.ones(1, dtype=torch.float32, device=q.device)

    return lambda: paged_attention_common(
        Q=q,
        K=kc,
        V=vc,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        block_tables=bt,
        context_lens=ctx,
        block_tables_stride0=bt.stride(0),
        scale=sm_scale,
        max_qlen=qlen,
        max_seq_len=max_context_len,
        K_QScale_hip=scale,
        V_QScale_hip=scale,
        K_QScale_asm=scale,
        V_QScale_asm=scale,
        out_=out,
        kv_cache_dtype="auto",
    )


def create_benchmark(
    versions,
    qlen,
    head_dim,
    page_size,
    warmup_ms=100,
    rep_ms=200,
    shared_page_pool=False,
):
    if qlen != 1 and "sglang" in versions:
        raise ValueError("SGLang's vectorized_5d decode wrapper only supports query_length=1")
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
            plot_name=(f"paged-decode-performance-bf16-d{head_dim}-p{page_size}-qlen{qlen}" +
                       ("-shared-pool" if shared_page_pool else "")),
            args={},
        ))
    def benchmark(BATCH, N_CTX, provider):
        sm_scale = 1.0 / (head_dim**0.5)
        num_q_heads = NUM_KV_HEADS * QUERY_GROUP_SIZE
        # Give each sequence distinct physical pages. A small shared page pool
        # unrealistically turns a long-context decode benchmark into an L2 test.
        pool_pages = None
        if shared_page_pool:
            # Reproduce PR #2306 exactly: enough physical pages for roughly
            # four sequences, reused modulo the pool at larger batch sizes.
            pool_pages = 4 * ((N_CTX + page_size - 1) // page_size) + 16
        q, kc, vc, ctx, bt = _build_inputs(
            BATCH,
            [N_CTX] * BATCH,
            num_q_heads,
            NUM_KV_HEADS,
            head_dim,
            page_size,
            query_length=qlen,
            device=DEVICE,
            pool_pages=pool_pages,
            cache_layout="4d" if provider == "tlx" else "5d",
        )
        out = torch.empty_like(q)
        fn = _make_decode_fn(
            provider,
            out,
            q,
            kc,
            vc,
            ctx,
            bt,
            sm_scale,
            qlen,
            N_CTX,
            NUM_KV_HEADS,
            QUERY_GROUP_SIZE,
            head_dim,
        )
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup_ms, rep=rep_ms)

        # Decode reads the whole KV cache once (K + V, bf16): report effective
        # HBM read bandwidth, the meaningful metric for this memory-bound op.
        kv_bytes = 2 * BATCH * NUM_KV_HEADS * N_CTX * head_dim * 2
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
    parser.add_argument("--warmup-ms", type=int, default=100, help="Warmup time per benchmark point.")
    parser.add_argument("--rep-ms", type=int, default=200, help="Measurement time per benchmark point.")
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--shared-page-pool",
        action="store_true",
        help="Reproduce PR #2306 page reuse; large-batch results measure substantial L2 reuse.",
    )
    args = parser.parse_args()

    if is_hip():
        versions = args.version if args.version else DEFAULT_DECODE_VERSIONS
        print(f"Running paged-decode benchmarks for: {versions}, qlens={args.qlens}, "
              f"head_dims={args.head_dims}, page_sizes={args.page_sizes}")
        for head_dim in args.head_dims:
            for page_size in args.page_sizes:
                for qlen in args.qlens:
                    print(f"\n=== head_dim={head_dim}, page_size={page_size}, "
                          f"query_length={qlen} ===")
                    report = create_benchmark(
                        versions,
                        qlen,
                        head_dim,
                        page_size,
                        args.warmup_ms,
                        args.rep_ms,
                        args.shared_page_pool,
                    )
                    if len(versions) > 1:
                        df = report.run(return_df=True)
                        ylabel = "TB/s (effective HBM read)"
                        baseline = f"tlx_5d ({ylabel})"
                        if "tlx" in versions and "tlx_5d" in versions:
                            df["tlx_5d/tlx speedup"] = (df[baseline] / df[f"tlx ({ylabel})"])
                        aiter_col = f"aiter_common ({ylabel})"
                        if "aiter_common" in versions and "tlx_5d" in versions:
                            df["tlx_5d/aiter_common speedup"] = (df[baseline] / df[aiter_col])
                        if "aiter_common" in versions and "sglang" in versions:
                            df["sglang/aiter_common speedup"] = (df[f"sglang ({ylabel})"] / df[aiter_col])
                        print(f"paged-decode-performance-bf16-d{head_dim}-"
                              f"p{page_size}-qlen{qlen}:")
                        print(df.to_string())
                    else:
                        report.run(print_data=True)
    else:
        print("Skipping benchmarks, no AMD GPU found.")
