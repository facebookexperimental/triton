#!/usr/bin/env python3
"""Choose the tests a CUTracer sanitizer run checks.

A manual run checks the test it names. A pull request checks:

    TLX kernel changes   the B200 test files (``*sm100*``) that the repo's
                         kernel_test_selection.py maps the changes to,
                         ignoring kernels written for other GPUs
    CUTracer CI changes  SELF_TEST, to show the pipeline still works
    anything else        nothing; running every test under the sanitizer
                         would take hours

The list goes to ``$GITHUB_OUTPUT`` as ``tests=<json>``; an empty list means the
run has nothing to check.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SELECTOR = Path(__file__).resolve().parents[1] / "kernel_test_selection.py"
SELF_TEST = "python/test/unit/tlx_ops/test_flash_attn_sm100.py::test_flash_attn_fwd[1-1-256-64-False-fp16]"
# Kernel files for other GPUs. The selector maps a kernel file to every test of
# its op, so without this a gfx950 change would check the sm100 test.
OTHER_GPU_MARKERS = ("gfx", "hip", "sm90", "hopper")
CUTRACER_CI_PATHS = (
    ".github/workflows/cutracer.yml",
    ".github/scripts/cutracer_ci/*",
    ".github/actions/setup-cutracer/*",
)


def kernel_tests(changed: list[str]) -> list[str]:
    """B200 test files for a kernel-only change, else none."""
    with tempfile.TemporaryDirectory() as tmp:
        changed_file, affected, output = (Path(tmp) / name for name in ("changed", "affected", "output"))
        changed_file.write_text("\n".join(changed) + "\n")
        env = dict(os.environ, GITHUB_OUTPUT=str(output))
        subprocess.run(
            [sys.executable,
             str(SELECTOR), "--changed-files",
             str(changed_file), "--affected-out",
             str(affected)], check=True, env=env)
        if "kernel_only=true" not in output.read_text().splitlines():
            return []
        files = affected.read_text().split()
    return [path for path in files if "sm100" in os.path.basename(path)]


def select(changed: list[str]) -> list[str]:
    ours = [path for path in changed if any(fnmatch.fnmatch(path, pattern) for pattern in CUTRACER_CI_PATHS)]
    others = [path for path in changed if path not in ours]
    tests = [SELF_TEST] if ours else []
    if others and not all(any(marker in path for marker in OTHER_GPU_MARKERS) for path in others):
        tests += kernel_tests(others)
    return tests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test", help="The test a manual run names.")
    source.add_argument("--changed-files", type=Path, help="Files a pull request changes, one per line.")
    args = parser.parse_args(argv)

    tests = [args.test] if args.test else select(args.changed_files.read_text().split())
    print(f"tests={json.dumps(tests)}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            handle.write(f"tests={json.dumps(tests)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
