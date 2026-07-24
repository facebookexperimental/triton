"""Correctness + do_bench TFLOPS harness for the 8-wave TLX GEMM (inter_wave port)."""

import argparse
import importlib.util
from pathlib import Path

import torch
import triton

from matmul_kernel import matmul, MIN_K, KERNEL_NAME


def _load_f16_inputs():
    path = Path(__file__).resolve().parents[2] / "f16_inputs.py"
    spec = importlib.util.spec_from_file_location("_tlx_gfx9_inter_wave_f16_inputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import f16 input helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_f16_inputs = _load_f16_inputs()
INPUT_MODES = _f16_inputs.INPUT_MODES
DEFAULT_INPUT_SEED = _f16_inputs.DEFAULT_INPUT_SEED
make_inputs = _f16_inputs.make_inputs


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


def get_x_vals():
    return [
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
    ]


def main():
    parser = argparse.ArgumentParser(description="8-wave TLX GEMM benchmark")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--shape", type=int, nargs=3, action="append", metavar=("M", "N", "K"),
                        help="Custom M N K shape (repeatable). Overrides default sizes.")
    parser.add_argument("--rep", type=positive_int, default=200, help="timed duration in milliseconds; default: 200")
    parser.add_argument("--warmup", type=nonnegative_int, default=25,
                        help="warmup duration in milliseconds; default: 25")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="normal",
        help=("input distribution; hpl and rand-int reproduce hipBLASLt seed-zero data, "
              "and rand-int applies its alternating sign to B"),
    )
    parser.add_argument("--seed", type=nonnegative_int, default=DEFAULT_INPUT_SEED,
                        help=f"deterministic input seed; default: {DEFAULT_INPUT_SEED}")
    args = parser.parse_args()

    sizes = [tuple(s) for s in args.shape] if args.shape else get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]

    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    device = triton.runtime.driver.active.get_active_torch_device()

    measurements = []
    for M, N, K in sizes:
        if K < MIN_K:
            print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: SKIPPED (K < {MIN_K})")
            continue
        a, b = make_inputs(
            M,
            N,
            K,
            device,
            "transposed",
            input_mode=args.input_mode,
            seed=args.seed,
        )
        ref = torch.matmul(a, b)
        c = matmul(a, b)
        ok = torch.allclose(c, ref, atol=1e-1, rtol=0)
        print(f"[{KERNEL_NAME}] M={M} N={N} K={K}: {'OK' if ok else 'FAIL'}")
        if not ok:
            continue
        ms_ref = triton.testing.do_bench(
            lambda: torch.matmul(a, b),
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
        ms_tlx = triton.testing.do_bench(
            lambda: matmul(a, b),
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
        measurements.append((M, N, K, ms_ref, ms_tlx))

    print(f"\n{KERNEL_NAME} (LLVM, input={args.input_mode}, seed={args.seed}; "
          f"triton median, {args.warmup}ms warmup/{args.rep}ms timed):")
    print(f"{'M':>6s} {'N':>6s} {'K':>6s}  {'rocBLAS':>8s}  {'TLX':>8s}")
    for M, N, K, ms_ref, ms_tlx in measurements:
        print(f"{M:6d} {N:6d} {K:6d}  {tflops(ms_ref,M,N,K):7.1f}T  {tflops(ms_tlx,M,N,K):7.1f}T")


if __name__ == "__main__":
    main()
