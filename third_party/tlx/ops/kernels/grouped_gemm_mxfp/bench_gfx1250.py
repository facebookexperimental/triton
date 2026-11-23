"""Benchmark the gfx1250 grouped MXFP GEMM (``grouped_gemm_mxfp``) shape sweep.

Each case is ``(G, M_per_group, N, K)`` with equal groups; the default sweep
is the FP16 grouped GEMM benchmark's
(``tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py``) plus a weight-bound
case of 128 groups of 64 rows. Each run reports TFLOPS and the bandwidth of
the bytes a call must move (X, W, their scales and the output). A8W8 and A8W4 run for
each shape by default (``--variant``; W4A4 on request). Each run uses a
separate process (``bench_gfx1250.py --single``) so its large allocations are
released before the next run starts.

By default the timed call takes preshuffled weight scales and a tile map built
once by ``prepare_grouped_gemm_mxfp``, as a serving loop would reuse them;
``--natural-w-scale`` and ``--per-call-tile-map`` include that preparation in
every call. ``--benchmark-mode graph`` (default) times the call with
``triton.testing.do_bench_cudagraph`` (no host launch overhead; L2 is not
cleared between replays), ``eager`` with ``triton.testing.do_bench`` (L2
cleared before each call); both report the median.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CASES = (
    (32, 32768, 8192, 4096),
    (32, 32768, 4096, 4096),
    (32, 65536, 8192, 4096),
    (32, 65536, 4096, 4096),
    (8, 32768, 8192, 4096),
    (8, 32768, 4096, 4096),
    (8, 65536, 8192, 4096),
    (8, 65536, 4096, 4096),
    (1, 4096, 4096, 4096),
    (1, 8192, 8192, 8192),
    (1, 16384, 16384, 16384),
    (16, 4096, 4096, 4096),
    # Weight-bound: each weight byte feeds 64 rows.
    (128, 64, 8192, 8192),
)

# (x_dtype, w_dtype) per variant.
VARIANTS = {"a8w8": ("e4m3", "e4m3"), "a8w4": ("e4m3", "e2m1"), "w4a4": ("e2m1", "e2m1")}
DEFAULT_VARIANTS = ("a8w8", "a8w4")

RESULT_RE = re.compile(r"execution time:\s*([0-9.eE+-]+)\s*ms,\s*([0-9.eE+-]+)\s*TFLOPS,\s*([0-9.eE+-]+)\s*TB/s")


@dataclass(frozen=True)
class BenchCase:
    groups: int
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.groups}x{self.m}x{self.n}x{self.k}"

    def tensor_bytes(self, variant: str) -> float:
        # X, W, their scales, and the BF16 output: the bytes one call must move.
        x_bits, w_bits = (8 if dtype == "e4m3" else 4 for dtype in VARIANTS[variant])
        gm = self.groups * self.m
        size = gm * self.k * x_bits / 8 + self.groups * self.n * self.k * w_bits / 8
        return size + (gm + self.groups * self.n) * self.k / 32 + gm * self.n * 2

    def tensor_gib(self, variant: str) -> float:
        return self.tensor_bytes(variant) / 1024**3


@dataclass
class BenchResult:
    case: BenchCase
    variant: str
    returncode: int
    ms: float | None
    tflops: float | None
    tbps: float | None

    @property
    def status(self) -> str:
        if self.returncode != 0:
            return f"exit {self.returncode}"
        if self.ms is None or self.tflops is None:
            return "no timing"
        return "ok"


def _parse_case(value: str) -> BenchCase:
    fields = [field for field in re.split(r"[xX,]", value) if field]
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("case must be G,M,N,K or GxMxNxK")
    groups, m, n, k = (int(field) for field in fields)
    if min(groups, m, n, k) <= 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    return BenchCase(groups, m, n, k)


def _command(args: argparse.Namespace, case: BenchCase, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()), "--single", "--variant", variant, "--case", case.label, "--seed",
        str(args.seed), "--benchmark-mode", args.benchmark_mode
    ]
    if args.natural_w_scale:
        command.append("--natural-w-scale")
    if args.per_call_tile_map:
        command.append("--per-call-tile-map")
    if args.check:
        command.append("--check")
    return command


def _run_case(args: argparse.Namespace, case: BenchCase, variant: str, index: int, total: int) -> BenchResult:
    command = _command(args, case, variant)
    print()
    print(f"[{index}/{total}] {variant} {case.label}  estimated tensors: {case.tensor_gib(variant):.1f} GiB")
    print(f"$ {shlex.join(command)}", flush=True)
    if args.dry_run:
        return BenchResult(case, variant, 0, None, None, None)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    ms = tflops = tbps = None
    for line in process.stdout:
        print(line, end="")
        match = RESULT_RE.search(line)
        if match:
            ms, tflops, tbps = (float(match.group(i)) for i in (1, 2, 3))
    return BenchResult(case, variant, process.wait(), ms, tflops, tbps)


def _print_summary(results: list[BenchResult], benchmark_mode: str) -> None:
    print()
    print(f"Summary ({benchmark_mode} timing)")
    print(f"{'variant':>7} {'G':>4} {'M/group':>9} {'N':>7} {'K':>7} {'ms':>12} {'TFLOPS':>10} {'TB/s':>8}  status")
    for result in results:
        case = result.case
        ms = "-" if result.ms is None else f"{result.ms:.4f}"
        tflops = "-" if result.tflops is None else f"{result.tflops:.2f}"
        tbps = "-" if result.tbps is None else f"{result.tbps:.2f}"
        print(f"{result.variant:>7} {case.groups:4d} {case.m:9d} {case.n:7d} {case.k:7d} {ms:>12} {tflops:>10} "
              f"{tbps:>8}  {result.status}")


def _write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=("variant", "groups", "m_per_group", "n", "k", "ms", "tflops", "tbps", "status"))
        writer.writeheader()
        for result in results:
            case = result.case
            writer.writerow({
                "variant": result.variant, "groups": case.groups, "m_per_group": case.m, "n": case.n, "k": case.k, "ms":
                "" if result.ms is None else result.ms, "tflops": "" if result.tflops is None else result.tflops,
                "tbps": "" if result.tbps is None else result.tbps, "status": result.status
            })


def _inputs(case: BenchCase, variant: str, seed: int):
    import torch

    x_dtype, w_dtype = VARIANTS[variant]
    gm, k = case.groups * case.m, case.k
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def payload(shape, dtype):
        if dtype == "e4m3":
            # Small positive E4M3 values keep the BF16 output finite.
            values = torch.randint(20, 40, shape, dtype=torch.uint8, device="cuda", generator=generator)
            return values.view(torch.float8_e4m3fn)
        return torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device="cuda",
                             generator=generator)

    x = payload((gm, k), x_dtype)
    w = payload((case.groups, case.n, k), w_dtype)
    x_scale = torch.randint(124, 130, (gm, k // 32), dtype=torch.uint8, device="cuda", generator=generator)
    w_scale = torch.randint(124, 130, (case.groups, case.n, k // 32), dtype=torch.uint8, device="cuda",
                            generator=generator)
    split_sizes = torch.full((case.groups, ), case.m, dtype=torch.int32, device="cuda")
    return x, x_scale, w, w_scale, split_sizes


def _dequantize(values, scale, dtype):
    import torch

    if dtype == "e4m3":
        data = values.float()
    else:
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           device=values.device)
        data = torch.stack([lut[(values & 15).long()], lut[(values >> 4).long()]], -1).flatten(-2)
    return data * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def _check(out, x, x_scale, w, w_scale, case: BenchCase, variant: str) -> float:
    # Compare up to 64 rows at the start and end of the first and last group.
    x_dtype, w_dtype = VARIANTS[variant]
    worst = 0.0
    for group in sorted({0, case.groups - 1}):
        weights = _dequantize(w[group], w_scale[group], w_dtype)
        for first in sorted({0, max(case.m - 64, 0)}):
            rows = slice(group * case.m + first, group * case.m + min(first + 64, case.m))
            ref = _dequantize(x[rows], x_scale[rows], x_dtype) @ weights.T
            worst = max(worst, ((out[rows].float() - ref).abs() / ref.abs().clamp_min(1.0)).max().item())
    return worst


def run_single(args: argparse.Namespace) -> int:
    import torch
    import triton

    from triton.tlx.ops.kernels.grouped_gemm_mxfp import gfx1250 as grouped

    case, variant = args.cases[0], args.variant[0]
    x_dtype, w_dtype = VARIANTS[variant]
    x, x_scale, w, w_scale, split_sizes = _inputs(case, variant, args.seed)
    gm = case.groups * case.m
    out = torch.empty((gm, case.n), dtype=torch.bfloat16, device="cuda")
    if not args.natural_w_scale:
        w_scale_arg = grouped.preshuffle_grouped_weight_scale(w_scale)
    else:
        w_scale_arg = w_scale
    tile_map = None if args.per_call_tile_map else grouped.prepare_grouped_gemm_mxfp(split_sizes, gm, x_dtype, w_dtype)

    def call():
        grouped.grouped_gemm_mxfp(x, x_scale, w, w_scale_arg, split_sizes, x_dtype=x_dtype, w_dtype=w_dtype, out=out,
                                  tile_map=tile_map, w_scale_preshuffled=not args.natural_w_scale)

    call()
    torch.cuda.synchronize()
    if args.check:
        err = _check(out, x, x_scale, w, w_scale, case, variant)
        print(f"check max_rel={err:.3g} {'OK' if err < 1e-2 else 'FAIL'}", flush=True)
        if err >= 1e-2:
            return 1
    if args.benchmark_mode == "graph":
        ms = triton.testing.do_bench_cudagraph(call, return_mode="median")
    else:
        ms = triton.testing.do_bench(call, return_mode="median")
    tflops = 2 * gm * case.n * case.k / (ms * 1e-3) / 1e12
    tbps = case.tensor_bytes(variant) / (ms * 1e-3) / 1e12
    print(f"{variant} {case.label}: execution time: {ms:.4f} ms, {tflops:.2f} TFLOPS, {tbps:.2f} TB/s", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the gfx1250 grouped MXFP GEMM shape sweep")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS),
                        help=f"repeatable; default: {', '.join(DEFAULT_VARIANTS)}")
    parser.add_argument("--case", action="append", type=_parse_case, dest="cases",
                        help="override defaults with a repeatable G,M,N,K or GxMxNxK case")
    parser.add_argument("--natural-w-scale", action="store_true",
                        help="pass natural weight scales, preshuffled in every call")
    parser.add_argument("--per-call-tile-map", action="store_true", help="build the tile map in every call")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False,
                        help="compare sampled rows with a dequantized reference before timing")
    parser.add_argument(
        "--benchmark-mode", choices=("graph", "eager"), default="graph",
        help="graph (default): CUDA graph replay without host launch overhead or L2 clearing; "
        "eager: one call per repetition after clearing L2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None, help="optional summary CSV path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.single:
        if len(args.cases or ()) != 1 or len(args.variant or ()) != 1:
            parser.error("--single takes exactly one --case and one --variant")
        return run_single(args)

    cases = args.cases or [BenchCase(*case) for case in DEFAULT_CASES]
    variants = args.variant or DEFAULT_VARIANTS
    runs = [(case, variant) for case in cases for variant in variants]
    results = []
    for index, (case, variant) in enumerate(runs, start=1):
        result = _run_case(args, case, variant, index, len(runs))
        results.append(result)
        if args.fail_fast and not args.dry_run and result.status != "ok":
            break
    _print_summary(results, args.benchmark_mode)
    if args.csv is not None:
        _write_csv(args.csv, results)
        print(f"\nWrote {args.csv}")
    if args.dry_run:
        return 0
    return int(any(result.status != "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
