#!/usr/bin/env python3
"""Compare equivalent gfx950 FlashAttention forward implementations.

The benchmark gives every implementation the same deterministic BF16 tensors
and computes non-causal square attention with ``D=128``.  These constraints are
the intersection of the public adaptive, async, persistent, and cluster
contracts.  Each result is checked against PyTorch SDPA before timing.

Timing uses alternating forward/reverse variant order so that temperature and
clock drift do not consistently favor one implementation.  The reported time
is the median of the per-round medians.  Public launchers are measured, so the
small Python dispatch and cached-output-allocation costs are included equally.

Select a GPU before starting Python, for example:

    ROCR_VISIBLE_DEVICES=7 python third_party/tlx/tutorials/amd_fa_compare.py

Use ``--output results.csv`` to retain the complete measurements.
"""

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# This benchmark only uses the in-tree AMD backend.  Ignoring external backend
# entry points also keeps a stale optional plugin installation from changing
# which checkout is measured.
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_USE_C_DISPATCHER", "0")

import torch
import torch.nn.functional as F

import triton

import amd_fa_adaptive
import amd_fa_cluster
import amd_fa_persistent
import amd_fa_pipelined

DEFAULT_SHAPES = ((1, 64, 4096, 128), (2, 64, 8192, 128), (1, 64, 16384, 128))
DEFAULT_VARIANTS = (
    "adaptive",
    "adaptive_bounded",
    "async_simple",
    "async_prefetch",
    "persistent",
    "cluster",
    "cluster_persistent",
    "torch_sdpa",
)


@dataclass(frozen=True)
class Measurement:
    batch: int
    heads: int
    sequence: int
    head_dim: int
    variant: str
    correct: bool
    max_abs: float
    mean_abs: float
    median_ms: float
    tflops: float
    range_ms: float
    samples_ms: tuple[float, ...]


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def _shape(text):
    try:
        shape = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected B,H,N,D") from exc
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected four positive integers B,H,N,D")
    batch, heads, sequence, head_dim = shape
    if head_dim != 128:
        raise argparse.ArgumentTypeError("equivalent adaptive comparisons require D=128")
    if sequence < 256 or sequence % 256:
        raise argparse.ArgumentTypeError("equivalent adaptive comparisons require N >= 256 and divisible by 256")
    return batch, heads, sequence, head_dim


def _bounded_inputs(shape, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def bounded_tensor():
        values = torch.randint(-8, 9, shape, dtype=torch.int8, device=device, generator=generator)
        return (values * 0.125).to(torch.bfloat16)

    q = bounded_tensor()
    k = bounded_tensor()
    v = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    v = v.mul_(2.0).sub_(1.0).to(torch.bfloat16)
    return q, k, v


def _variant_launchers(q, k, v, scale):
    return {
        "adaptive":
        lambda: amd_fa_adaptive.attention(q, k, v, sm_scale=scale),
        "adaptive_bounded":
        lambda: amd_fa_adaptive.attention(q, k, v, sm_scale=scale, qk_max_abs=1.0),
        "async_simple":
        lambda: amd_fa_pipelined.attention(
            q,
            k,
            v,
            scale,
            False,
            config={"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4},
        ),
        "async_prefetch":
        lambda: amd_fa_pipelined.attention(
            q,
            k,
            v,
            scale,
            False,
            config={"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 8, "PREFETCH": True},
        ),
        "persistent":
        lambda: amd_fa_persistent.attention(q, k, v, scale, False),
        "cluster":
        lambda: amd_fa_cluster.attention(q, k, v, scale, False),
        "cluster_persistent":
        lambda: amd_fa_cluster.persistent_attention(q, k, v, scale, False),
        "torch_sdpa":
        lambda: F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False),
    }


def _reference(q, k, v, scale):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)
    torch.cuda.synchronize()
    return result


def _measure_shape(shape, variants, device, seed, warmup, rep, rounds):
    batch, heads, sequence, head_dim = shape
    q, k, v = _bounded_inputs(shape, device, seed)
    scale = 1.0 / math.sqrt(head_dim)
    all_launchers = _variant_launchers(q, k, v, scale)
    launchers = {name: all_launchers[name] for name in variants}
    reference = _reference(q, k, v, scale)

    correctness = {}
    for name, launch in launchers.items():
        output = launch()
        torch.cuda.synchronize()
        difference = (output.float() - reference.float()).abs()
        correct = bool(torch.isfinite(output).all()) and torch.allclose(output, reference, atol=2e-2, rtol=2e-2)
        correctness[name] = (correct, difference.max().item(), difference.mean().item())
        print(
            f"correctness shape={shape} variant={name} pass={correct} "
            f"max_abs={correctness[name][1]:.6g} mean_abs={correctness[name][2]:.6g}",
            file=sys.stderr,
            flush=True,
        )
        if not correct:
            raise RuntimeError(f"correctness failed for {name} at shape {shape}")

    orders = []
    forward = list(launchers)
    reverse = list(reversed(forward))
    for round_index in range(rounds):
        orders.append(forward if round_index % 2 == 0 else reverse)

    samples = {name: [] for name in launchers}
    for round_index, order in enumerate(orders, 1):
        for name in order:
            milliseconds = float(triton.testing.do_bench(launchers[name], warmup=warmup, rep=rep, return_mode="median"))
            samples[name].append(milliseconds)
            print(
                f"timing shape={shape} round={round_index}/{rounds} "
                f"variant={name} ms={milliseconds:.6f}",
                file=sys.stderr,
                flush=True,
            )

    operations = 4 * batch * heads * sequence * sequence * head_dim
    measurements = []
    for name in launchers:
        median_ms = statistics.median(samples[name])
        tflops = operations / (median_ms * 1e-3) / 1e12
        correct, max_abs, mean_abs = correctness[name]
        measurements.append(
            Measurement(
                batch,
                heads,
                sequence,
                head_dim,
                name,
                correct,
                max_abs,
                mean_abs,
                median_ms,
                tflops,
                max(samples[name]) - min(samples[name]),
                tuple(samples[name]),
            ))

    del all_launchers, launchers, reference, q, k, v, output, difference
    gc.collect()
    torch.cuda.empty_cache()
    return measurements


def _write_csv(measurements, stream):
    fields = tuple(Measurement.__dataclass_fields__)
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for measurement in measurements:
        row = dict(measurement.__dict__)
        row["samples_ms"] = json.dumps(row["samples_ms"])
        writer.writerow(row)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        type=_shape,
        help="B,H,N,D; repeat for multiple shapes (default: production comparison sweep)",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=DEFAULT_VARIANTS,
        help="variant to run; repeat as needed (default: all)",
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=50)
    parser.add_argument("--rep", type=_positive_int, default=300)
    parser.add_argument("--rounds", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    return parser.parse_args()


def main():
    args = _parse_args()
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx950":
        raise RuntimeError(f"AMD FA comparison requires gfx950, got {target}")

    shapes = tuple(args.shapes) if args.shapes else DEFAULT_SHAPES
    variants = tuple(args.variants) if args.variants else DEFAULT_VARIANTS
    device = triton.runtime.driver.active.get_active_torch_device()
    print(
        f"target={target} device={device} shapes={shapes} variants={variants} "
        f"warmup={args.warmup} rep={args.rep} rounds={args.rounds} seed={args.seed}",
        file=sys.stderr,
        flush=True,
    )

    measurements = []
    for shape in shapes:
        measurements.extend(_measure_shape(
            shape,
            variants,
            device,
            args.seed,
            args.warmup,
            args.rep,
            args.rounds,
        ))

    _write_csv(measurements, sys.stdout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            _write_csv(measurements, stream)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
