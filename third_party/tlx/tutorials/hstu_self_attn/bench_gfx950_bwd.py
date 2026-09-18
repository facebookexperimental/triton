"""Benchmark the experimental gfx950 HSTU backward schedules.

The default sweep matches the production-shaped validation workload:
B512/H4/D128 BF16, jagged lengths at density 0.95, target counts
sampled uniformly from 1 through 20, and maximum sequence lengths 1K--8K.

Run from the repository root, for example:

  HIP_VISIBLE_DEVICES=3 python \
    third_party/tlx/tutorials/hstu_self_attn/bench_gfx950_bwd.py

An external saved-input fixture can be replayed with:

  HIP_VISIBLE_DEVICES=3 python \
    third_party/tlx/tutorials/hstu_self_attn/bench_gfx950_bwd.py \
    --input-fixture /path/to/no_bias_bwd_inputs.pt

The reported TFLOP/s uses the target-aware causal mask and the five backward
matrix products.  It is therefore semantic work, not padded tile work.
"""

import argparse
import gc
import hashlib
import statistics
from pathlib import Path
import subprocess
import sys

import torch
import triton

KERNEL_DIR = Path(__file__).resolve().parent
if str(KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_DIR))

import tlx_gfx950_ragged_hstu_attention as hstu  # noqa: E402

DEFAULT_VARIANTS = (
    "kv_parallel",
    "kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
    "kv_parallel_fa_schedule_bn256_direct_qdo_g2l",
)


def _make_workload(max_seq_len, batch, heads, head_dim, sparsity, max_targets, seed):
    device = torch.device("cuda")
    if not 0.5 <= sparsity <= 1.0:
        raise ValueError("this benchmark expects sparsity in [0.5, 1.0]")

    length_gen = torch.Generator(device=device).manual_seed(seed)
    low = max(1, int((2 * sparsity - 1.0) * max_seq_len))
    if low == max_seq_len:
        lengths = torch.full((batch, ), max_seq_len, device=device, dtype=torch.int64)
    else:
        lengths = torch.randint(
            low,
            max_seq_len,
            (batch, ),
            device=device,
            dtype=torch.int64,
            generator=length_gen,
        )

    target_gen = torch.Generator(device=device).manual_seed(seed + 1)
    targets = torch.randint(
        1,
        max_targets + 1,
        (batch, ),
        device=device,
        dtype=torch.int64,
        generator=target_gen,
    )
    targets = torch.minimum(targets, lengths)

    offsets = torch.zeros((batch + 1, ), device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total_tokens = int(offsets[-1])

    value_gen = torch.Generator(device=device).manual_seed(seed + 2)

    def tensor():
        return torch.empty(
            (total_tokens, heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        ).uniform_(-0.01, 0.01, generator=value_gen).requires_grad_()

    q, k, v = tensor(), tensor(), tensor()
    lengths_cpu = lengths.cpu()
    targets_cpu = targets.cpu()
    history = lengths_cpu - targets_cpu
    valid_pairs = (history * (history + 1) // 2 + targets_cpu * (history + 1)).sum().item()
    flops = int(valid_pairs) * heads * (6 * head_dim + 4 * head_dim)
    digest = hashlib.sha256(lengths_cpu.numpy().tobytes() + targets_cpu.numpy().tobytes()).hexdigest()[:12]
    return q, k, v, offsets, targets, lengths_cpu, targets_cpu, int(valid_pairs), flops, digest


def _make_input_fixture_workload(path):
    """Reconstruct a no-bias workload saved by the issue #2005 harness."""
    inputs = torch.load(path, map_location="cpu", weights_only=True)
    if inputs.get("timestamps") is not None:
        raise ValueError("the optimized gfx950 schedule does not support relative attention bias")
    required = {
        "N",
        "alpha",
        "q_shape",
        "seq_offsets",
        "invalid_attn_mask_type",
        "num_targets",
        "attn_bias",
        "seq2_offsets",
        "max_attn_len",
        "contextual_seq_len",
        "sort_by_length",
    }
    missing = sorted(required - inputs.keys())
    if missing:
        raise ValueError(f"input fixture is missing keys: {missing}")
    if inputs["attn_bias"] is not None:
        raise ValueError("the optimized gfx950 schedule does not support explicit attention bias")
    if inputs["seq2_offsets"] is not None:
        raise ValueError("the optimized gfx950 schedule does not support cross-attention offsets")
    if inputs["invalid_attn_mask_type"] != "lower_triangular":
        raise ValueError("the optimized gfx950 schedule requires lower-triangular masking")
    if inputs["max_attn_len"] not in (None, 0) or inputs["contextual_seq_len"] != 0:
        raise ValueError("the optimized gfx950 schedule does not support attention windows or contextual prefixes")
    if inputs["sort_by_length"]:
        raise ValueError("the optimized gfx950 schedule does not support length sorting")

    total_tokens, heads, head_dim = map(int, inputs["q_shape"])
    if head_dim != 128:
        raise ValueError(f"the specialized gfx950 backward requires D=128, got {head_dim}")
    offsets = inputs["seq_offsets"].to(device="cuda", dtype=torch.int64)
    targets = inputs["num_targets"].to(device="cuda", dtype=torch.int64)
    if offsets.ndim != 1 or targets.ndim != 1 or offsets.numel() != targets.numel() + 1:
        raise ValueError("seq_offsets and num_targets must describe the same one-dimensional batch")
    lengths = offsets[1:] - offsets[:-1]
    if int(offsets[0]) != 0 or bool(torch.any(lengths <= 0)):
        raise ValueError("seq_offsets must start at zero and be strictly increasing")
    if bool(torch.any(targets < 0)) or bool(torch.any(targets > lengths)):
        raise ValueError("num_targets must be between zero and the corresponding sequence length")
    if int(offsets[-1]) != total_tokens:
        raise ValueError(f"q_shape has {total_tokens} tokens but offsets end at {int(offsets[-1])}")

    # Match the linked reproducer: Q/K/V are disjoint views of one [T, 4H, D]
    # allocation, so their token stride is 4 * H * D rather than H * D.
    value_gen = torch.Generator(device="cuda").manual_seed(1)
    backing = torch.empty(
        (total_tokens, 4 * heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-2.0, 2.0, generator=value_gen)
    q, k, v, _ = torch.split(backing, [heads, heads, heads, heads], dim=1)
    q, k, v = (tensor.detach().requires_grad_() for tensor in (q, k, v))

    lengths_cpu = lengths.cpu()
    targets_cpu = targets.cpu()
    history = lengths_cpu - targets_cpu
    valid_pairs = (history * (history + 1) // 2 + targets_cpu * (history + 1)).sum().item()
    flops = int(valid_pairs) * heads * (6 * head_dim + 4 * head_dim)
    digest = hashlib.sha256(lengths_cpu.numpy().tobytes() + targets_cpu.numpy().tobytes()).hexdigest()[:12]
    workload = q, k, v, offsets, targets, lengths_cpu, targets_cpu, int(valid_pairs), flops, digest
    return int(inputs["N"]), float(inputs["alpha"]), workload


def _time_variant(args, variant, max_seq_len, alpha, q, k, v, offsets, targets, dout):
    out = hstu.tlx_gfx950_hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        offsets,
        invalid_attn_mask_type="lower_triangular",
        num_targets=targets,
        attn_scale=None,
        max_attn_len=0,
        contextual_seq_len=0,
        sort_by_length=False,
        full_attn_size=0,
        num_softmax_heads=0,
        bwd_variant=variant,
    )

    def backward():
        # Measure the custom backward itself, not PyTorch's subsequent
        # accumulation into already-populated leaf .grad tensors.
        q.grad = k.grad = v.grad = None
        out.backward(dout, retain_graph=True)

    backward()
    torch.cuda.synchronize()
    medians = [
        triton.testing.do_bench(backward, warmup=args.warmup, rep=args.rep, return_mode="median")
        for _ in range(args.samples)
    ]
    return statistics.median(medians), min(medians), max(medians), out


def _git_revision():
    repo_root = Path(__file__).resolve().parents[4]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
        )
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.95,
                        help="length density: 0.95 samples lengths uniformly from [0.9*N, N)")
    parser.add_argument("--max-targets", type=int, default=20,
                        help="sample each sequence's target count uniformly from [1, max-targets]")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                        help="exact BWD_VARIANTS names, or a single 'all'")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--warmup", type=int, default=25, help="do_bench warmup window in milliseconds")
    parser.add_argument(
        "--rep",
        type=int,
        default=1000,
        help="do_bench measurement window in milliseconds; 1000 gives multiple timed launches at N=8192",
    )
    parser.add_argument("--samples", type=int, default=3, help="independent do_bench medians per table cell")
    parser.add_argument(
        "--input-fixture",
        type=Path,
        help="optional no-bias .pt input fixture, such as the reproducer linked from issue #2005",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("gfx950 GPU required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":", 1)[0]
    if arch != "gfx950":
        raise SystemExit("gfx950 GPU required")
    if args.input_fixture is None and args.head_dim != 128:
        raise SystemExit("the specialized gfx950 backward requires D=128")
    if args.input_fixture is None and args.max_targets < 1:
        raise SystemExit("--max-targets must be positive")

    variants = sorted(hstu.BWD_VARIANTS) if args.variants == ["all"] else args.variants
    unknown = [variant for variant in variants if variant not in hstu.BWD_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; available: {sorted(hstu.BWD_VARIANTS)}")

    if args.input_fixture is None:
        workloads = ((
            max_seq_len,
            1.0 / args.head_dim,
            _make_workload(
                max_seq_len,
                args.batch,
                args.heads,
                args.head_dim,
                args.sparsity,
                args.max_targets,
                args.seed,
            ),
        ) for max_seq_len in args.seq_lens)
        workload_description = (f"B={args.batch} H={args.heads} D={args.head_dim} sparsity={args.sparsity} "
                                f"max_targets={args.max_targets} seed={args.seed} alpha=1/D")
    else:
        fixture_n, fixture_alpha, fixture_workload = _make_input_fixture_workload(args.input_fixture)
        workloads = [(fixture_n, fixture_alpha, fixture_workload)]
        fixture_q = fixture_workload[0]
        workload_description = (
            f"fixture={args.input_fixture} B={fixture_workload[3].numel() - 1} H={fixture_q.shape[1]} "
            f"D={fixture_q.shape[2]} qkv_stride={fixture_q.stride()} alpha={fixture_alpha:.12g}")

    print(
        f"commit={_git_revision()} gpu={torch.cuda.get_device_name()} dtype=bf16 "
        f"{workload_description} warmup_ms={args.warmup} rep_ms={args.rep} samples={args.samples}",
        flush=True,
    )
    print("| N | length range / mean | target range / mean | variant | P50 ms | semantic TFLOP/s | sample range |")
    print("|---:|---:|---:|---|---:|---:|---:|")

    for max_seq_len, alpha, workload in workloads:
        q, k, v, offsets, targets, lengths_cpu, targets_cpu, valid_pairs, flops, digest = workload
        dout_gen = torch.Generator(device=q.device).manual_seed(args.seed + 3)
        dout = torch.empty_like(q).normal_(generator=dout_gen)
        length_stats = f"{int(lengths_cpu.min())}-{int(lengths_cpu.max())} / {lengths_cpu.float().mean():.1f}"
        target_stats = f"{int(targets_cpu.min())}-{int(targets_cpu.max())} / {targets_cpu.float().mean():.1f}"
        print(
            f"workload N={max_seq_len} tokens={int(offsets[-1])} valid_pairs={valid_pairs} "
            f"semantic_tflop={flops / 1e12:.6f} hash={digest}",
            flush=True,
        )

        for variant in variants:
            median_ms, low_ms, high_ms, out = _time_variant(args, variant, max_seq_len, alpha, q, k, v, offsets,
                                                            targets, dout)
            tflops = flops / median_ms * 1e-9
            print(
                f"| {max_seq_len} | {length_stats} | {target_stats} | `{variant}` | "
                f"{median_ms:.4f} | {tflops:.2f} | {low_ms:.4f}-{high_ms:.4f} |",
                flush=True,
            )
            q.grad = k.grad = v.grad = None
            del out
            gc.collect()
            torch.cuda.empty_cache()

        del q, k, v, offsets, targets, dout
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
