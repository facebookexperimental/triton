import argparse
import os
import sys

import torch

import triton

from triton._internal_testing import is_hip_cdna4
from triton.language.extra.tlx.tutorials import amd_hstu_attn as _hstu


def _load_hstu_bwd():
    """Import the gfx950 TLX ragged HSTU kernel (fwd+bwd).

    It lives in the sibling ``hstu_self_attn`` directory, which is a set of
    standalone scripts rather than a package (they import each other by bare
    name), so the directory has to go on ``sys.path`` first.
    """
    from triton.language.extra.tlx import tutorials as _tut

    kernel_dir = os.path.join(list(_tut.__path__)[0], "hstu_self_attn")
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    import tlx_gfx950_ragged_hstu_attention as _mod

    return _mod


def _bwd_inputs(batch_size, max_seq_len, sparsity, heads, attn_dim, hidden_dim, dtype):
    """Jagged HSTU inputs with grad-requiring leaves. Deterministic: the length
    generator seeds itself, so buck and OSS runs see identical shapes."""
    device = torch.device("cuda")
    lengths = _hstu.generate_sparse_seq_len(size=batch_size, max_seq_len=max_seq_len, sparsity=sparsity, device=device)
    lengths = _hstu.apply_SL(lengths, 2.0, max_seq_len=max_seq_len)
    num_targets = torch.randint(1, 21, (batch_size, ), device=device, dtype=lengths.dtype)
    num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty((L, heads, attn_dim * 2 + hidden_dim), dtype=dtype, device=device).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    return q, k, v, seq_offsets, num_targets, L


def run_bwd_benchmark(args):
    """Backward sweep across the selectable bwd schedules.

    One column per variant, mirroring the provider table of the hammer
    benchmark. Note the variants split across two code paths: the
    ``fa_schedule`` ones launch ``_tlx_gfx950_hstu_fa_schedule_bwd_kernel`` with
    hard-coded launch parameters, the rest go through the autotuned
    ``_tlx_gfx950_ragged_hstu_attn_bwd``.
    """
    hstu = _load_hstu_bwd()
    variants = sorted(hstu.BWD_VARIANTS) if args.variant == "all" else [args.variant]
    for v in variants:
        if v not in hstu.BWD_VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; expected one of {sorted(hstu.BWD_VARIANTS)}")

    x_names = ["batch_size", "max_seq_len", "sparsity", "heads", "attn_dim", "hidden_dim"]
    x_val_list = [(args.b, n, args.sparsity, args.heads, args.head_dim, args.hidden_dim)
                  for n in args.bwd_seq_lens] if args.bwd_seq_lens else [(args.b, args.max_seq_len, args.sparsity,
                                                                          args.heads, args.head_dim, args.hidden_dim)]

    ylabel = "Time (ms)" if args.metric == "time" else "Throughput (TFLOPS)"
    configs = [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_val_list,
            line_arg="variant",
            line_vals=variants,
            line_names=variants,
            styles=[("green", "-")] * len(variants),
            ylabel=ylabel,
            plot_name="hstu_attention bwd perf numbers",
            args={"metric": args.metric},
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_hstu_bwd(batch_size, max_seq_len, sparsity, heads, attn_dim, hidden_dim, variant, metric, **kwargs):
        dtype = _hstu.str_to_torch_dtype[args.dtype]
        q, k, v, seq_offsets, num_targets, L = _bwd_inputs(batch_size, max_seq_len, sparsity, heads, attn_dim,
                                                           hidden_dim, dtype)
        # attn_scale=None -> the kernel folds 1/max_seq_len into the activation,
        # which the fa_schedule variants require (they reject an explicit scale).
        out = hstu.BWD_VARIANTS[variant].apply(max_seq_len, 1.0 / attn_dim, q, k, v, seq_offsets, "lower_triangular",
                                               num_targets, None, None, None, 0, 0, False, 0, 0)
        dout = torch.randn_like(out)
        # Trigger JIT compilation and (for the non-fa_schedule variants)
        # autotuning outside the timed region: do_bench calibrates its repeat
        # count from the first call, and a multi-second autotune there skews
        # every subsequent measurement.
        out.backward(dout, retain_graph=True)
        torch.cuda.synchronize()
        # bind as defaults so the `del` below cannot unbind the closure
        ms = triton.testing.do_bench(lambda o=out, d=dout: o.backward(d, retain_graph=True), warmup=25, rep=args.rep)
        # Free before the next variant: holding several retained graphs of this
        # size alive perturbs the later measurements badly at long sequences.
        del out, dout, q, k, v
        torch.cuda.empty_cache()
        if metric == "time":
            return ms
        # HSTU bwd computes (QK^T), dQ, dK, dV and d(QK^T): 3*f1 + 2*f2 where
        # f1/f2 are the attn_dim/hidden_dim halves of the forward's flops.
        off = seq_offsets.cpu().numpy()
        flops = sum(int(off[i + 1] - off[i])**2 * heads * (3 * attn_dim + 2 * hidden_dim) for i in range(len(off) - 1))
        return flops / ms * 1e-9

    bench_hstu_bwd.run(save_path="." if args.o else None, print_data=True)


# benchmark
def run_benchmark(args):

    x_names = [
        "batch_size",
        "max_seq_len",
        "sparsity",
        "heads",
        "attn_dim",
        "hidden_dim",
    ]
    if args.user_input:
        x_val_list = [(
            args.b,
            args.max_seq_len,
            args.sparsity,
            args.heads,
            args.head_dim,
            args.hidden_dim,
        )]
    else:
        x_val_list = _hstu.get_inputs()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GBs)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    evaluation_metric_to_unit = {
        "throughput": "TFLOPS", "time": "Time_(ms)", "bandwidth":
        "Bandwidth_(GB/s)",  # spaces break prettytable parsing
    }
    line_names = [evaluation_metric_to_unit[args.metric]]
    line_vals = line_names

    configs = []
    metric = args.metric
    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_val_list,
            line_arg="unit",
            line_vals=line_vals,
            line_names=line_names,
            styles=[("green", "-")],
            ylabel=ylabel,
            plot_name='hstu_attention perf numbers',
            args={"metric": metric, "mode": 'fwd'},
        ))

    @triton.testing.perf_report(configs)
    def bench_hstu_attn(
        batch_size,
        max_seq_len,
        sparsity,
        heads,
        attn_dim,
        hidden_dim,
        metric,
        mode,
        **kwargs,
    ):
        type_str = args.dtype
        assert type_str in [
            "fp16",
            "bf16",
        ], "only fp16 or bf16 data types are supported!"
        dropout_pr = 0.0
        target_size: int = 20
        sl_alpha: float = 2.0
        dtype = _hstu.str_to_torch_dtype[type_str]

        invalid_attn_mask_type = "lower_triangular"
        causal = True
        alpha = 1.0 / attn_dim * 10000

        # generate inputs
        torch.manual_seed(1001)  # for reproducibility
        lengths = _hstu.generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_seq_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
        lengths = _hstu.apply_SL(lengths, sl_alpha, max_seq_len=max_seq_len)
        num_targets = torch.randint(
            1,
            target_size + 1,
            (batch_size, ),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)
        seq_offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=torch.device("cuda"))
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        L = int(seq_offsets[-1].item())
        x = torch.empty(
            (L, heads, attn_dim * 2 + hidden_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.01, 0.01)
        q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)

        q = _hstu.switch_to_contiguous_if_needed(q)
        k = _hstu.switch_to_contiguous_if_needed(k)
        v = _hstu.switch_to_contiguous_if_needed(v)

        _hstu.sanity_check_attention(
            max_seq_len=max_seq_len,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            dropout_pr=dropout_pr,
            attn_bias=None,
            max_attn_len=None,
            contextual_seq_len=0,
        )

        def attn_fwd():
            return _hstu.triton_hstu_attention_fwd(max_seq_len, alpha, q, k, v, seq_offsets, causal, num_targets,
                                                   0,  # max_attn_len,
                                                   0,  # contextual_seq_len
                                                   True,  # sort_by_length,
                                                   )

        ms = triton.testing.do_bench(
            attn_fwd,
            warmup=25,
            rep=100,
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            flops = _hstu.get_flops(seq_offsets.cpu().numpy(), heads, attn_dim, hidden_dim)
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            elem_size = q.element_size()
            bytes = _hstu.get_bytes(seq_offsets.cpu().numpy(), heads, attn_dim, hidden_dim, elem_size)
            bandwidth = bytes / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_hstu_attn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark HSTU Attention",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--b",
        type=int,
        default=512,
        help="Batch dim of input sequences",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="max sequence length",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="sparsity of input sequence lengths",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="number of heads",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="head dimension",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="hidden dimension",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="data type, default (bfloat16)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )

    parser.add_argument(
        "--user_input",
        action="store_true",
        default=False,
        help="Run user input info",
    )

    parser.add_argument("-o", action="store_true", help="Write performance results to CSV file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["fwd", "bwd"],
        default="fwd",
        help="fwd benchmarks amd_hstu_attn; bwd sweeps the gfx950 TLX bwd schedules",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        help="bwd schedule to run, or 'all' to sweep every one",
    )
    parser.add_argument(
        "--bwd_seq_lens",
        type=int,
        nargs="+",
        default=None,
        help="bwd sequence lengths to sweep (default: just --max_seq_len)",
    )
    parser.add_argument("--rep", type=int, default=500, help="do_bench rep")

    args = parser.parse_args()
    return args


def main():
    if not is_hip_cdna4():
        print("Skipping benchmarks, requires gfx950 hardware (CDNA4).")
        return
    args = parse_args()
    if args.mode == "bwd":
        run_bwd_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
