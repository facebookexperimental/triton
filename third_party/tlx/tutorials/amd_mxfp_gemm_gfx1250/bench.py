"""Benchmark the persistent gfx1250 MXFP8 x MXFP8 and MXFP8 x MXFP4 GEMM.

Each case runs in a separate process through amd_mxfp_gemm_tdm_pipelined.py.
By default both variants run at 8192x8192x4096, 8192x8192x8192,
16384x16384x16384, and 32768x32768x32768, first with FP32 and then with FP16
output.

Both variants use 256x256 output tiles, four waves, partial TDM fusion, 256
persistent programs, group-M 8, 16-workgroup clusters in a 4x4 tile layout
with multicast inputs and a cluster barrier every eight K blocks, and chunked
XCD remapping of whole clusters (eight XCDs, chunk two).

- mx8xmx8 (FP8 E4M3 B) uses BK128 with cross-tile prefetch and overlaps each
  next tile's first K step with the C epilogue. Below K=16384 it uses three
  input stages and three dedicated C slots (six with FP16 output), and from
  K=8192 unrolls the steady loop by the ring depth. From K=16384 it uses four
  input stages and two C slots.
- mx8xmx4 (FP4 E2M1 B) uses BK256 with three input stages, stages C in
  retired A stages, and uses split cluster barriers and early C stores.

Timing repeats eager batches of --benchmark-num-iters milliseconds: batches
run for --warmup-seconds, then the median batch over --duration-seconds is
reported. Use --dry-run to print the commands.
"""

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_CASES = ((8192, 8192, 4096), (8192, 8192, 8192), (16384, 16384, 16384), (32768, 32768, 32768))
VARIANT_DTYPES_B = {"mx8xmx8": "float8_e4m3", "mx8xmx4": "float4"}
OUT_DTYPES = ("fp32", "fp16")
GRID_CLUSTER_LAYOUTS = {"2x2": (2, 2), "4x2": (4, 2), "2x4": (2, 4), "4x4": (4, 4), "8x2": (8, 2), "2x8": (2, 8)}
# MX8xMX8 defaults by K: static slot unroll from STATIC_SLOT_UNROLL_MIN_K,
# and a four-stage ring with two C slots from FOUR_BUFFER_MIN_K.
STATIC_SLOT_UNROLL_MIN_K = 8192
FOUR_BUFFER_MIN_K = 16384
DEFAULT_OUTPUT_C_SLOTS = {"fp32": 3, "fp16": 6}
RESULT_RE = re.compile(r"execution time:\s*([0-9.eE+-]+)\s*ms,\s*([0-9.eE+-]+)\s*TFLOPS")


def _parse_case(value):
    try:
        dims = tuple(int(field) for field in re.split("[xX,]", value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be M,N,K or MxNxK") from exc
    if len(dims) != 3 or min(dims) <= 0:
        raise argparse.ArgumentTypeError("case must contain three positive dimensions")
    return dims


def _variant_args(args, dtype_b, out_dtype):
    """Resolve per-variant defaults; per-case defaults are applied by _case_args."""
    resolved = argparse.Namespace(**vars(args))
    resolved.out_dtype = out_dtype
    fp4 = dtype_b == "float4"
    resolved.block_k = 256 if fp4 else 128
    if resolved.cross_tile_prefetch is None:
        resolved.cross_tile_prefetch = not fp4
    cluster_barriers = resolved.cluster_size > 1 and resolved.cluster_barrier_interval > 0
    if resolved.split_cluster_barrier is None:
        resolved.split_cluster_barrier = fp4 and cluster_barriers
    if resolved.early_c_stores is None:
        resolved.early_c_stores = fp4
    return resolved


def _case_args(args, case):
    """Resolve the input ring, C slots, static unroll, and overlap peel for one case."""
    resolved = argparse.Namespace(**vars(args))
    k = case[2]
    if args.block_k == 256:
        resolved.num_buffers = args.num_buffers or 3
        resolved.static_slot_unroll = bool(args.static_slot_unroll)
        resolved.overlap_peel = bool(args.overlap_peel)
        return resolved
    resolved.num_buffers = args.num_buffers or (4 if k >= FOUR_BUFFER_MIN_K else 3)
    three_stage = resolved.num_buffers == 3
    if args.output_c_slots is None and three_stage:
        resolved.output_c_slots = DEFAULT_OUTPUT_C_SLOTS[args.out_dtype]
    if args.static_slot_unroll is None:
        resolved.static_slot_unroll = three_stage and k >= STATIC_SLOT_UNROLL_MIN_K
    if args.overlap_peel is None:
        resolved.overlap_peel = (three_stage and args.cross_tile_prefetch and k // args.block_k > resolved.num_buffers
                                 and not (args.split_cluster_barrier or args.early_c_stores))
    return resolved


def _command(args, case, dtype_b):
    m, n, k = case
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("amd_mxfp_gemm_tdm_pipelined.py")),
        "-M",
        str(m),
        "-N",
        str(n),
        "-K",
        str(k),
        "-BM",
        "256",
        "-BN",
        "256",
        "-BK",
        str(args.block_k),
        "--dtype_a",
        "float8_e4m3",
        "--dtype_b",
        dtype_b,
        "--num_buffers",
        str(args.num_buffers),
        "--group_size_m",
        str(args.group_m),
        "--schedule",
        "sliceMNK",
        "--tdm_fusion",
        "partial",
        "--transpose_b",
        "--scale_preshuffled",
        "--with_a_scale",
        "--persistent",
        "--benchmark_mode",
        args.benchmark_mode,
        "--benchmark_num_iters",
        str(args.benchmark_num_iters),
        "--benchmark_warmup_seconds",
        str(args.warmup_seconds),
        "--benchmark_duration_seconds",
        str(args.duration_seconds),
        "--seed",
        str(args.seed),
        "--xcd_remap",
        args.xcd_remap,
        "--num_xcds",
        str(args.num_xcds),
        "--xcd_chunk",
        str(args.xcd_chunk),
        "--cluster_size",
        str(args.cluster_size),
        "--cluster_layout",
        args.cluster_layout,
        "--cluster_barrier_interval",
        str(args.cluster_barrier_interval),
    ]
    if args.static_slot_unroll:
        command.append("--static_slot_unroll")
    if args.overlap_peel:
        command.append("--overlap_peel")
    if args.output_c_slots is not None:
        command.extend(["--output_c_slots", str(args.output_c_slots)])
    if args.out_dtype != "fp32":
        command.extend(["--out_dtype", args.out_dtype])
    if args.split_cluster_barrier:
        command.append("--split_cluster_barrier")
    if args.early_c_stores:
        command.append("--early_c_stores")
    if not args.cluster_multicast:
        command.append("--no-cluster_multicast")
    if not args.cross_tile_prefetch:
        command.append("--no-cross_tile_prefetch")
    if args.num_programs is not None:
        command.extend(["--num_programs", str(args.num_programs)])
    return command


def parse_benchmark_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_DTYPES_B),
                        help="repeatable variant selection; default: both")
    parser.add_argument("--case", action="append", type=_parse_case, help="repeatable M,N,K or MxNxK override")
    parser.add_argument("-M", type=int)
    parser.add_argument("-N", type=int)
    parser.add_argument("-K", type=int)
    parser.add_argument("--out-dtype", choices=OUT_DTYPES, action="append",
                        help="repeatable C element type; default: fp32 and fp16")
    parser.add_argument("--num-buffers", type=int, choices=(2, 3, 4), default=None,
                        help="input ring depth (default: MX8xMX8 3, or 4 for K >= 16384; MX8xMX4 3)")
    parser.add_argument("--output-c-slots", type=int, default=None,
                        help="dedicated MX8xMX8 C slots (default: 3, 6 with FP16, 2 with four stages)")
    parser.add_argument("--static-slot-unroll", action=argparse.BooleanOptionalAction, default=None,
                        help=f"unroll the three-stage MX8xMX8 steady loop (default: K >= {STATIC_SLOT_UNROLL_MIN_K})")
    parser.add_argument(
        "--overlap-peel", action=argparse.BooleanOptionalAction, default=None,
        help="overlap each next tile's first K step with the C epilogue (default: three-stage "
        "MX8xMX8)")
    parser.add_argument("--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="prefetch next-tile inputs during the K tail (default: MX8xMX8)")
    parser.add_argument("--split-cluster-barrier", action=argparse.BooleanOptionalAction, default=None,
                        help="split the periodic cluster barrier (default: MX8xMX4)")
    parser.add_argument("--early-c-stores", action=argparse.BooleanOptionalAction, default=None,
                        help="interleave the final quadrant dots with C stores (default: MX8xMX4)")
    parser.add_argument("--group-m", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-programs", type=int, default=256, help="persistent workgroups, capped by the tile count")
    parser.add_argument("--cluster-size", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--cluster-layout", choices=("linear", ) + tuple(GRID_CLUSTER_LAYOUTS), default="4x4")
    parser.add_argument("--cluster-barrier-interval", type=int, default=8,
                        help="cluster barrier every N input K blocks; 0 disables cluster barriers")
    parser.add_argument("--cluster-multicast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xcd-remap", choices=("none", "balanced", "chunked"), default="chunked")
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", type=int, default=2)
    parser.add_argument("--benchmark-mode", choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-num-iters", type=int, default=256,
                        help="timing batch length in milliseconds (not an iteration count)")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="write results to this CSV file")
    parser.add_argument("--output-dir", type=Path, help="run each case in its own subdirectory")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)

    dims = (args.M, args.N, args.K)
    if any(dim is not None for dim in dims):
        if args.case or any(dim is None for dim in dims):
            parser.error("provide all of -M/-N/-K together, or use --case")
        cases = [dims]
    else:
        cases = args.case or list(DEFAULT_CASES)
    if args.num_programs <= 0:
        parser.error("--num-programs must be positive")
    if args.warmup_seconds < 0 or args.duration_seconds < 0 or args.benchmark_num_iters <= 0:
        parser.error("timing durations must be nonnegative and --benchmark-num-iters positive")
    if args.cluster_barrier_interval < 0:
        parser.error("--cluster-barrier-interval must be nonnegative")
    if args.cluster_layout in GRID_CLUSTER_LAYOUTS:
        rows, cols = GRID_CLUSTER_LAYOUTS[args.cluster_layout]
        if args.cluster_size != rows * cols:
            parser.error(f"--cluster-layout {args.cluster_layout} requires --cluster-size {rows * cols}")
    if args.output_c_slots is not None and args.output_c_slots < 2:
        parser.error("--output-c-slots must be at least 2")
    out_dtypes = list(dict.fromkeys(args.out_dtype or OUT_DTYPES))
    variants = [(variant, VARIANT_DTYPES_B[variant], _variant_args(args, VARIANT_DTYPES_B[variant], out_dtype))
                for variant in dict.fromkeys(args.variant or VARIANT_DTYPES_B)
                for out_dtype in out_dtypes]
    return args, cases, variants


def _amd_codegen_revision():
    try:
        from triton.backends.amd.compiler import get_amd_codegen_revision
        return get_amd_codegen_revision()
    except Exception as error:  # Report rather than hide an unusable code generator.
        return f"unknown ({error})"


def main(argv=None):
    args, cases, variants = parse_benchmark_args(argv)
    codegen = None if args.dry_run else _amd_codegen_revision()
    if codegen is not None:
        print(f"AMD code generator: {codegen}", flush=True)
    results = []
    runs = [(case, variant, dtype_b, run_args) for case in cases for variant, dtype_b, run_args in variants]
    for index, (case, variant, dtype_b, run_args) in enumerate(runs, 1):
        run_args = _case_args(run_args, case)
        label = "x".join(map(str, case))
        command = _command(run_args, case, dtype_b)
        print(f"\n[{index}/{len(runs)}] {variant} {label} (float8_e4m3 x {dtype_b} -> {run_args.out_dtype})",
              flush=True)
        print(f"$ {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        cwd = None
        if args.output_dir is not None:
            cwd = args.output_dir.resolve() / f"{index:02d}-{variant}-{run_args.out_dtype}-{label}"
            cwd.mkdir(parents=True, exist_ok=False)
        ms = tflops = None
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = RESULT_RE.search(line)
                if match:
                    ms, tflops = map(float, match.groups())
            returncode = process.wait()
        status = "ok" if returncode == 0 else f"exit {returncode}"
        if status == "ok" and args.benchmark_mode != "none" and ms is None:
            status = "missing timing"
        results.append(
            dict(variant=variant, dtype_a="float8_e4m3", dtype_b=dtype_b, out_dtype=run_args.out_dtype, M=case[0],
                 N=case[1], K=case[2], block_k=run_args.block_k, num_buffers=run_args.num_buffers,
                 output_c_slots=run_args.output_c_slots, static_slot_unroll=run_args.static_slot_unroll,
                 overlap_peel=run_args.overlap_peel, cross_tile_prefetch=run_args.cross_tile_prefetch,
                 split_cluster_barrier=run_args.split_cluster_barrier, early_c_stores=run_args.early_c_stores,
                 group_m=run_args.group_m, cluster_size=run_args.cluster_size, cluster_layout=run_args.cluster_layout,
                 cluster_barrier_interval=run_args.cluster_barrier_interval,
                 cluster_multicast=run_args.cluster_multicast, xcd_remap=run_args.xcd_remap, num_xcds=run_args.num_xcds,
                 xcd_chunk=run_args.xcd_chunk, num_programs=run_args.num_programs,
                 benchmark_mode=run_args.benchmark_mode, benchmark_ms=run_args.benchmark_num_iters,
                 warmup_seconds=run_args.warmup_seconds, duration_seconds=run_args.duration_seconds, ms=ms,
                 tflops=tflops, status=status, amd_codegen=codegen, command=shlex.join(command)))

    if args.dry_run:
        return 0
    print(f"\n{'variant':>9} {'dtype B':>12} {'out':>5} {'M':>7} {'N':>7} {'K':>7} {'BK':>4} {'bufs':>4} "
          f"{'ms':>12} {'TFLOPS':>12}  status")
    for row in results:
        ms = "-" if row["ms"] is None else f"{row['ms']:.4f}"
        tflops = "-" if row["tflops"] is None else f"{row['tflops']:.2f}"
        print(f"{row['variant']:>9} {row['dtype_b']:>12} {row['out_dtype']:>5} {row['M']:7d} {row['N']:7d} "
              f"{row['K']:7d} {row['block_k']:4d} {row['num_buffers']:4d} {ms:>12} {tflops:>12}  {row['status']}")
    if args.csv is not None and results:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=tuple(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
