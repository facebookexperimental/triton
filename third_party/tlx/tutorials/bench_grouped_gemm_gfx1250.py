"""Run the regular gfx1250 grouped GEMM benchmark shape sweep.

Each shape is ``(G, M_per_group, N, K)``. Every group has the same M.
Each case runs in a separate process so its large GPU allocations are released
before the next case starts.
"""

from __future__ import annotations

import argparse
import csv
import os
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
    (16, 4096, 4096, 4096),
)

RESULT_RE = re.compile(r"execution time:\s*([0-9.eE+-]+)\s*ms,\s*([0-9.eE+-]+)\s*TFLOPS", )


@dataclass(frozen=True)
class BenchCase:
    groups: int
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.groups}x{self.m}x{self.n}x{self.k}"

    @property
    def m_list(self) -> str:
        return ",".join([str(self.m)] * self.groups)

    @property
    def tensor_gib(self) -> float:
        # Packed A, transposed/K-contiguous B, and C. This excludes allocator
        # overhead and any temporary memory used by the runtime.
        elements = self.groups * self.m * self.k
        elements += self.groups * self.n * self.k
        elements += self.groups * self.m * self.n
        return elements * 2 / (1024**3)


@dataclass
class BenchResult:
    case: BenchCase
    tensor_gib: float
    returncode: int
    ms: float | None
    tflops: float | None

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


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    local_paths = [str(repo_root / "python"), str(repo_root)]
    if env.get("PYTHONPATH"):
        local_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(local_paths)
    return env


def _build_command(args: argparse.Namespace, case: BenchCase) -> list[str]:
    kernel = Path(__file__).with_name("amd_grouped_gemm_gfx1250_test.py")
    command = [
        sys.executable,
        str(kernel),
        "--m_list",
        case.m_list,
        "-N",
        str(case.n),
        "-K",
        str(case.k),
        "-BM",
        str(args.block_m),
        "-BN",
        str(args.block_n),
        "-BK",
        str(args.block_k),
        "--group_m",
        str(args.group_m),
        "--tdm_pipeline_depth",
        str(args.tdm_pipeline_depth),
        "--l2_prefetch_distance",
        str(args.l2_prefetch_distance),
        "--benchmark_mode",
        args.benchmark_mode,
        "--benchmark_num_iters",
        str(args.benchmark_num_iters),
        "--xcd_remap",
        args.xcd_remap,
        "--num_xcds",
        str(args.num_xcds),
        "--xcd_chunk",
        str(args.xcd_chunk),
        "--seed",
        str(args.seed),
    ]
    if args.num_programs is not None:
        command.extend(["--num_programs", str(args.num_programs)])
    if args.dedicated_c_buffer:
        command.append("--dedicated_c_buffer")
    if args.cross_tile_prefetch:
        command.append("--cross_tile_prefetch")
    if args.auto_config:
        command.append("--auto_config")
    if args.check:
        command.append("--check")
    return command


def _run_case(args: argparse.Namespace, case: BenchCase, index: int, total: int) -> BenchResult:
    command = _build_command(args, case)
    print()
    print(f"[{index}/{total}] {case.label}  estimated tensors: {case.tensor_gib:.1f} GiB")
    print(f"$ {shlex.join(command)}")
    if args.dry_run:
        return BenchResult(case, case.tensor_gib, 0, None, None)

    process = subprocess.Popen(
        command,
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    ms = None
    tflops = None
    for line in process.stdout:
        print(line, end="")
        match = RESULT_RE.search(line)
        if match:
            ms = float(match.group(1))
            tflops = float(match.group(2))
    returncode = process.wait()
    return BenchResult(case, case.tensor_gib, returncode, ms, tflops)


def _print_summary(results: list[BenchResult]) -> None:
    print()
    print("Summary")
    print(f"{'G':>4} {'M/group':>9} {'N':>7} {'K':>7} {'GiB':>7} {'ms':>12} {'TFLOPS':>12}  status")
    for result in results:
        ms = "-" if result.ms is None else f"{result.ms:.4f}"
        tflops = "-" if result.tflops is None else f"{result.tflops:.2f}"
        case = result.case
        print(f"{case.groups:4d} {case.m:9d} {case.n:7d} {case.k:7d} "
              f"{result.tensor_gib:7.1f} {ms:>12} {tflops:>12}  {result.status}")


def _write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("groups", "m_per_group", "n", "k", "tensor_gib", "ms", "tflops", "status"),
        )
        writer.writeheader()
        for result in results:
            case = result.case
            writer.writerow({
                "groups": case.groups,
                "m_per_group": case.m,
                "n": case.n,
                "k": case.k,
                "tensor_gib": f"{result.tensor_gib:.3f}",
                "ms": "" if result.ms is None else result.ms,
                "tflops": "" if result.tflops is None else result.tflops,
                "status": result.status,
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the gfx1250 grouped GEMM shape sweep")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        dest="cases",
        help="override defaults with a repeatable G,M,N,K or GxMxNxK case",
    )
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--group-m", type=int, default=4)
    parser.add_argument("--tdm-pipeline-depth", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    parser.add_argument("--num-programs", type=int, default=None)
    parser.add_argument("--dedicated-c-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--xcd-remap", choices=("none", "balanced", "chunked"), default="none")
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", type=int, default=2)
    parser.add_argument("--benchmark-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--benchmark-num-iters", type=int, default=32)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None, help="optional summary CSV path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    cases = args.cases or [BenchCase(*case) for case in DEFAULT_CASES]
    results = []
    for index, case in enumerate(cases, start=1):
        result = _run_case(args, case, index, len(cases))
        results.append(result)
        if args.fail_fast and not args.dry_run and result.status != "ok":
            break

    _print_summary(results)
    if args.csv is not None:
        _write_csv(args.csv, results)
        print(f"\nWrote {args.csv}")

    if args.dry_run:
        return 0
    return int(any(result.status != "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
