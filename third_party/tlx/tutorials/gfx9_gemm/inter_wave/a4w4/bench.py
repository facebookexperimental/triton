#!/usr/bin/env python3
"""Correctness and timing harness for the 8-wave inter-wave MXFP4 GEMM."""

import argparse
import os

import torch
import triton
from triton import knobs
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a4w4.bench import (
    DEFAULT_SIZES as INTRA_WAVE_DEFAULT_SIZES,
    DEFAULT_TIMED_LAUNCHES,
    DEFAULT_TIMING_REPEATS,
    DEFAULT_WARMUP_LAUNCHES,
    TIMING_MODES,
    generate_mxfp4_inputs,
    measure_matmul,
    nonnegative_int,
    parse_shape,
    positive_int,
    shape_cache_dir,
    tflops,
    torch_reference,
)

try:
    from .matmul_kernel import (
        BLOCK_K,
        BLOCK_M,
        BLOCK_N,
        GROUP_SIZE_M,
        KERNEL_NAME,
        MIN_K,
        NUM_WARPS,
        NUM_XCDS,
        _a4w4_8wave_kernel,
    )
except ImportError:
    from matmul_kernel import (
        BLOCK_K,
        BLOCK_M,
        BLOCK_N,
        GROUP_SIZE_M,
        KERNEL_NAME,
        MIN_K,
        NUM_WARPS,
        NUM_XCDS,
        _a4w4_8wave_kernel,
    )

DEFAULT_SIZES = tuple(shape for shape in INTRA_WAVE_DEFAULT_SIZES if shape[2] >= MIN_K)


def parse_inter_wave_shape(text):
    shape = parse_shape(text)
    if shape[2] < MIN_K:
        raise argparse.ArgumentTypeError(f"8-wave inter-wave MXFP requires K >= {MIN_K}")
    return shape


def launch_matmul(a, b, a_scales, b_scales, out=None):
    M = a.shape[0]
    K = a.shape[1] * 2
    N = b.shape[0]
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _a4w4_8wave_kernel[(grid_mn, )](
        a,
        b,
        c,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        SPLIT_K=1,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=32,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    return c


def main():
    parser = argparse.ArgumentParser(description="8-wave inter-wave TLX gfx950 MXFP4 GEMM benchmark")
    parser.add_argument("--shape", action="append", type=parse_inter_wave_shape, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default="batched",
        help="timing methodology; default: batched",
    )
    parser.add_argument("--rep", type=positive_int, default=200)
    parser.add_argument("--warmup", type=nonnegative_int, default=25)
    parser.add_argument(
        "--warmup-launches",
        type=nonnegative_int,
        default=DEFAULT_WARMUP_LAUNCHES,
    )
    parser.add_argument(
        "--timed-launches",
        type=positive_int,
        default=DEFAULT_TIMED_LAUNCHES,
    )
    parser.add_argument(
        "--timing-repeats",
        type=positive_int,
        default=DEFAULT_TIMING_REPEATS,
    )
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    sizes = args.shape or DEFAULT_SIZES
    if args.K is not None:
        sizes = tuple((m, n, k) for m, n, k in sizes if k == args.K)
    if not sizes:
        raise SystemExit("no shapes selected")

    if args.timing_mode == "triton":
        timing_summary = f"triton median, {args.warmup}ms warmup/{args.rep}ms timed"
    else:
        timing_summary = (f"batched median, {args.timing_repeats}x{args.timed_launches} timed launches, "
                          f"{args.warmup_launches} warmups/repeat")
    backend = "Wave" if os.environ.get("TRITON_DEFAULT_BACKEND") == "tlx_wave" else "LLVM"
    print(f"\n{KERNEL_NAME} {backend} gfx950 ({timing_summary}):")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'status':>10s}  {'max_err':>10s}  {backend:>17s}")
    for M, N, K in sizes:
        a, b, a_scales, b_scales = generate_mxfp4_inputs(M, N, K)
        cache_dir = shape_cache_dir(args.cache_dir, M, N, K)
        with knobs.cache.scope():
            if cache_dir is not None:
                knobs.cache.dir = str(cache_dir)
            c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
            launch_matmul(a, b, a_scales, b_scales, out=c)
            torch.cuda.synchronize()
            ref = torch_reference(a, b, a_scales, b_scales)
            max_err = (c - ref).abs().max().item()
            ok = torch.allclose(c, ref, atol=args.atol, rtol=args.rtol)
            if ok and not args.no_bench:
                ms = measure_matmul(
                    args,
                    lambda: launch_matmul(a, b, a_scales, b_scales, out=c),
                )
                perf = f"{tflops(ms, M, N, K):8.1f}T/{ms:6.3f}ms"
            else:
                perf = "-"
        status = "ok" if ok else "FAIL"
        print(f"{M:6d} {N:6d} {K:6d}  {status:>10s}  {max_err:10.4f}  {perf:>17s}", flush=True)


if __name__ == "__main__":
    main()
