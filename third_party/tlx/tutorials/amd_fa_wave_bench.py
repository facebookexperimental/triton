#!/usr/bin/env python3
"""Correctness and performance driver for the separate eight-wave FA kernel."""

import argparse
import math

import torch
import torch.nn.functional as F

import triton

import amd_fa_wave

CORRECTNESS_SHAPE = (1, 1, 512, 128)
PERFORMANCE_SHAPE = (2, 64, 8192, 128)
WAVE_PERFORMANCE_FLOOR_TFLOPS = 1000.0


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def bounded_inputs(shape, device, *, seed):
    """Match the standalone runner's bounded Q/K distribution."""
    torch.manual_seed(seed)

    def bounded_tensor():
        values = torch.randint(
            -8,
            9,
            shape,
            device=device,
        )
        return (values * 0.125).to(torch.bfloat16)

    q = bounded_tensor()
    k = bounded_tensor()
    v = (torch.rand_like(q, dtype=torch.float32).mul_(2.0).sub_(1.0).to(torch.bfloat16))
    return q, k, v


def check_correctness(device, *, qk_max_abs):
    q, k, v = bounded_inputs(CORRECTNESS_SHAPE, device, seed=0)
    output = amd_fa_wave.attention(q, k, v, qk_max_abs=qk_max_abs)
    reference = F.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(CORRECTNESS_SHAPE[-1]),
    )
    difference = (output.float() - reference.float()).abs()
    maximum = difference.max().item()
    mean = difference.mean().item()
    passed = bool(torch.isfinite(output).all()) and torch.allclose(
        output,
        reference,
        atol=2e-2,
        rtol=2e-2,
    )
    print(f"  [{'PASS' if passed else 'FAIL'}] correctness "
          f"B=1 H=1 D=128 N=512 max={maximum:.6f} mean={mean:.6f}")
    return passed


def measure_performance(device, *, warmup, rep, qk_max_abs):
    batch, heads, sequence, head_dim = PERFORMANCE_SHAPE
    q, k, v = bounded_inputs(PERFORMANCE_SHAPE, device, seed=0)
    output = torch.empty_like(q)
    amd_fa_wave.attention(q, k, v, qk_max_abs=qk_max_abs, out=output, warmup=True)
    launch = lambda: amd_fa_wave.attention(q, k, v, qk_max_abs=qk_max_abs, out=output)
    launch()
    torch.cuda.synchronize()
    milliseconds = triton.testing.do_bench(
        launch,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    operations = 4 * batch * heads * sequence * sequence * head_dim
    return milliseconds, operations / milliseconds * 1e-9


def parse_args():
    parser = argparse.ArgumentParser(prog="AMD TLX eight-wave FA")
    parser.add_argument("--warmup", type=nonnegative_int, default=50)
    parser.add_argument("--rep", type=positive_int, default=500)
    parser.add_argument(
        "--min-tflops",
        type=float,
        help="required performance; defaults to 1000 for TLX Wave and 0 for LLVM",
    )
    parser.add_argument(
        "--qk-max-abs",
        type=float,
        help="explicit Q/K magnitude bound; generated inputs require a value >= 1",
    )
    args = parser.parse_args()
    invalid_bound = args.qk_max_abs is not None and (
        not math.isfinite(args.qk_max_abs) or args.qk_max_abs < 1.0
    )
    if invalid_bound:
        parser.error("--qk-max-abs must be finite and at least 1")
    return args


def main():
    args = parse_args()
    device = triton.runtime.driver.active.get_active_torch_device()
    backend = triton.runtime.driver.active.get_current_target().backend
    minimum = args.min_tflops
    if minimum is None:
        minimum = WAVE_PERFORMANCE_FLOOR_TFLOPS if backend == "tlx_wave" else 0.0

    mode = "adaptive" if args.qk_max_abs is None else f"bounded (|Q|, |K| <= {args.qk_max_abs:g})"
    print(f"Eight-wave {mode} FlashAttention ({backend})")
    correctness_ok = check_correctness(device, qk_max_abs=args.qk_max_abs)
    milliseconds, tflops = measure_performance(
        device,
        warmup=args.warmup,
        rep=args.rep,
        qk_max_abs=args.qk_max_abs,
    )
    performance_ok = tflops >= minimum
    print(f"  [{'PASS' if performance_ok else 'FAIL'}] performance "
          f"B=2 H=64 D=128 N=8192 {milliseconds:.6f} ms "
          f"{tflops:.1f} TFLOPS ({tflops / 1000.0:.3f} PFLOPS, floor {minimum:.1f})")
    passed = correctness_ok and performance_ok
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
