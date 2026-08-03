#!/usr/bin/env python3
"""Correctness and performance driver for the separate eight-wave FA kernel."""

import argparse
import math

import torch
import torch.nn.functional as F

import triton

import amd_fa_wave

ADVERTISED_SHAPE = (2, 64, 8192, 128)
QUERY_TILE_SIZE = amd_fa_wave.BLOCK_M.value
KV_TILE_SIZE = amd_fa_wave.BLOCK_N.value
XCD_COUNT = amd_fa_wave.XCDS.value
FORCED_REBASE_K_STEP = 64.0
FORCED_REBASE_LOG2_HEADROOM = amd_fa_wave.SOFTMAX_REFERENCE_HEADROOM_LOG2.value
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
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def bounded_tensor():
        values = torch.randint(
            -8,
            9,
            shape,
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        return (values * 0.125).to(torch.bfloat16)

    q = bounded_tensor()
    k = bounded_tensor()
    v = (torch.rand(shape, dtype=torch.float32, device=device,
                    generator=generator).mul_(2.0).sub_(1.0).to(torch.bfloat16))
    return q, k, v


def forced_rebase_inputs(shape, device, *, seed):
    """Build an adaptive input whose score maximum advances every K/V tile."""
    _, _, sequence, head_dim = shape
    if sequence % KV_TILE_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {KV_TILE_SIZE}")

    q = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    q[..., 0] = 1.0
    tile_indices = torch.arange(sequence // KV_TILE_SIZE, dtype=torch.int32, device=device)
    tile_keys = (tile_indices * FORCED_REBASE_K_STEP).to(torch.bfloat16)
    k = torch.zeros_like(q)
    k[..., 0] = tile_keys.repeat_interleave(KV_TILE_SIZE)
    # Keep the factory interface aligned with bounded_inputs; this pattern is
    # intentionally deterministic because tile parity is part of the oracle.
    del seed
    tile_values = torch.where(tile_indices % 2 == 0, 64.0, 0.0).to(torch.bfloat16)
    v = tile_values.repeat_interleave(KV_TILE_SIZE)[None, None, :, None].expand(shape).contiguous()

    tile_log2_scores = tile_keys.float() * (math.log2(math.e) / math.sqrt(head_dim))
    advances = tile_log2_scores.diff()
    if not bool(torch.all(advances > FORCED_REBASE_LOG2_HEADROOM)):
        raise ValueError("forced-rebase score construction does not exceed the adaptive headroom")
    return q, k, v, advances.numel(), advances.min().item()


def check_correctness(q, k, v, *, qk_max_abs, case):
    output = amd_fa_wave.attention(q, k, v, qk_max_abs=qk_max_abs)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        reference = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(ADVERTISED_SHAPE[-1]),
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
    batch, heads, sequence, head_dim = ADVERTISED_SHAPE
    print(f"  [{'PASS' if passed else 'FAIL'}] correctness {case} "
          f"B={batch} H={heads} D={head_dim} N={sequence} max={maximum:.6f} mean={mean:.6f}")
    return passed


def measure_performance(q, k, v, *, warmup, rep, qk_max_abs):
    batch, heads, sequence, head_dim = ADVERTISED_SHAPE
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
    invalid_bound = args.qk_max_abs is not None and (not math.isfinite(args.qk_max_abs) or args.qk_max_abs < 1.0)
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
    batch, heads, sequence, _ = ADVERTISED_SHAPE
    program_count = batch * heads * (sequence // QUERY_TILE_SIZE)
    if program_count != 4096 or program_count % XCD_COUNT != 0:
        raise RuntimeError("advertised FA shape must exercise all 4096 programs across the eight-XCD swizzle")
    print(f"Eight-wave {mode} FlashAttention ({backend}, programs={program_count}, XCDs={XCD_COUNT})")
    correctness_ok = True
    performance_ok = True
    performance_cases = [False, True] if args.qk_max_abs is None else [False]
    for force_rebases in performance_cases:
        if force_rebases:
            q, k, v, rebase_count, minimum_advance = forced_rebase_inputs(ADVERTISED_SHAPE, device, seed=17)
            case = f"forced-rebase count={rebase_count} min-log2-advance={minimum_advance:.6f}"
        else:
            q, k, v = bounded_inputs(ADVERTISED_SHAPE, device, seed=0)
            case = "bounded-distribution"
        case_correct = check_correctness(q, k, v, qk_max_abs=args.qk_max_abs, case=case)
        correctness_ok = correctness_ok and case_correct
        milliseconds, tflops = measure_performance(
            q,
            k,
            v,
            warmup=args.warmup,
            rep=args.rep,
            qk_max_abs=args.qk_max_abs,
        )
        case_ok = tflops >= minimum
        performance_ok = performance_ok and case_ok
        print(f"  [{'PASS' if case_ok else 'FAIL'}] performance {case} "
              f"B=2 H=64 D=128 N=8192 {milliseconds:.6f} ms "
              f"{tflops:.1f} TFLOPS ({tflops / 1000.0:.3f} PFLOPS, floor {minimum:.1f})")
    passed = correctness_ok and performance_ok
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
