"""Run the tlx.ops perf suite (python/test/tlx_benchmark/test_ops_perf.py), forwarding pytest arguments."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

TRITON_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = TRITON_ROOT / "python" / "test" / "tlx_benchmark" / "test_ops_perf.py"
# The harness writes one JSON per op here (see _harness/driver.py:default_json).
HARNESS_JSON_DIR = Path("/tmp/tlx_benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd() / ".benchmarks" / "tlx_ops_benchmark",
        help="Where to collect the junit XML and per-op JSON (default: ./.benchmarks/tlx_ops_benchmark)")
    parser.add_argument("--include-torchtlx", action="store_true",
                        help="Also run the torchtlx ops, which need Meta's Inductor patches")
    args, arguments = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # --import-mode=importlib: otherwise python/conftest.py puts python/ on sys.path and the
    # source tree shadows an installed Triton wheel.
    command = [
        sys.executable, "-m", "pytest",
        str(TEST_FILE), "--import-mode=importlib", "-v", "-s", f"--junitxml={args.output_dir / 'junit.xml'}"
    ]
    # Forwarded arguments come last, so a caller's own -k overrides this one.
    if not args.include_torchtlx:
        command += ["-k", "not torchtlx"]
    command += arguments

    print("Running tlx_ops_benchmark benchmark", flush=True)
    start = time.time()
    result = subprocess.run(command, cwd=TRITON_ROOT)
    # The harness JSON path is fixed and may hold stale files from earlier runs.
    for path in HARNESS_JSON_DIR.glob("*.json"):
        if path.stat().st_mtime >= start:
            shutil.copy2(path, args.output_dir / path.name)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
