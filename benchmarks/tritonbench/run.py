"""Run TritonBench suites from a TritonBench checkout."""

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", required=True, help="Comma-separated TritonBench suites")
    parser.add_argument("--benchmark-arguments", default="", help="Additional command-line arguments for each suite")
    parser.add_argument("--tritonbench-dir", type=Path, default=Path.cwd(), help="Path to the TritonBench checkout")
    args = parser.parse_args()

    suites = [suite.strip() for suite in args.benchmarks.split(",")]
    if not any(suites):
        parser.error("No benchmark suites specified")
    try:
        arguments = shlex.split(args.benchmark_arguments)
    except ValueError as error:
        parser.error(f"Invalid benchmark arguments: {error}")

    status = 0
    for suite in suites:
        if not re.fullmatch(r"[a-zA-Z0-9_]+",
                            suite) or not (args.tritonbench_dir / "benchmarks" / suite / "run.py").is_file():
            print(f"Unknown benchmark suite: {suite}", file=sys.stderr)
            status = 1
            continue
        print(f"Running {suite} benchmark", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", f"benchmarks.{suite}.run", "--ci", *arguments],
            cwd=args.tritonbench_dir,
        )
        if result.returncode:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
