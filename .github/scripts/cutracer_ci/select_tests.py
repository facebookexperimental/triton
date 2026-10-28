#!/usr/bin/env python3
"""Choose the tests a CUTracer sanitizer run checks on each NVIDIA GPU.

A manual run checks the test it names on the GPU it names. A pull request
checks:

    TLX kernel changes   on each GPU a changed file targets, that GPU's test
                         files that the repo's kernel_test_selection.py maps
                         the changes to
    CUTracer CI changes  each GPU's SELF_TESTS entry, to show the pipeline
                         still works there
    anything else        nothing; running every test under the sanitizer
                         would take hours

Kernel files name the GPU they target: ``sm100`` or ``blackwell`` for B200,
``sm90`` or ``hopper`` for H100. An unnamed one (``__init__.py``,
``_shapes.py``) targets both; AMD files (``gfx``, ``hip``) target neither, since
CUTracer instruments through NVBit. A test file runs on B200 when its name has
``sm100``, on H100 when it has ``sm90``, and on both with ``sm90up``; other
test files, such as benchmarks and the tutorial correctness suite, do not run.

Each GPU's list goes to ``$GITHUB_OUTPUT`` as ``<gpu>=<json>``; an empty list
means that GPU has nothing to check.
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
GPUS = ("b200", "h100")
SELF_TESTS = {
    "b200": "python/test/unit/tlx_ops/test_flash_attn_sm100.py::test_flash_attn_fwd[1-1-256-64-False-fp16]",
    "h100": "python/test/unit/tlx_ops/test_flash_attn_sm90.py::test_flash_attn_fwd[1-1-256-64-False-fp16]",
}
KERNEL_MARKERS = {"b200": ("sm100", "blackwell"), "h100": ("sm90", "hopper")}
TEST_MARKERS = {"b200": ("sm100", ), "h100": ("sm90", )}
AMD_MARKERS = ("gfx", "hip")
CUTRACER_CI_PATHS = (
    ".github/workflows/cutracer.yml",
    ".github/workflows/cutracer-gpu.yml",
    ".github/scripts/cutracer_ci/*",
    ".github/actions/setup-cutracer/*",
)


def named_gpus(name: str, markers: dict[str, tuple[str, ...]]) -> set[str]:
    """The NVIDIA GPUs ``name`` targets; ``sm90up`` means Hopper and newer."""
    if "sm90up" in name:
        return set(GPUS)
    return {gpu for gpu, gpu_markers in markers.items() if any(marker in name for marker in gpu_markers)}


def targeted_gpus(path: str) -> set[str]:
    """The NVIDIA GPUs a change to ``path`` can affect."""
    if any(marker in path for marker in AMD_MARKERS):
        return set()
    return named_gpus(path, KERNEL_MARKERS) or set(GPUS)


def kernel_tests(changed: list[str]) -> list[str]:
    """The test files a kernel-only change affects, else none."""
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
    return files


def select(changed: list[str]) -> dict[str, list[str]]:
    ours = [path for path in changed if any(fnmatch.fnmatch(path, pattern) for pattern in CUTRACER_CI_PATHS)]
    others = [path for path in changed if path not in ours]
    tests = {gpu: [SELF_TESTS[gpu]] if ours else [] for gpu in GPUS}
    targeted = set().union(*map(targeted_gpus, others))
    if targeted:
        files = kernel_tests(others)
        for gpu in targeted:
            tests[gpu] += [path for path in files if gpu in named_gpus(os.path.basename(path), TEST_MARKERS)]
    return tests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test", help="The test a manual run names.")
    source.add_argument("--changed-files", type=Path, help="Files a pull request changes, one per line.")
    parser.add_argument("--gpu", choices=GPUS, default="b200", help="The GPU a manual run uses.")
    args = parser.parse_args(argv)

    if args.test:
        tests = {gpu: [args.test] if gpu == args.gpu else [] for gpu in GPUS}
    else:
        tests = select(args.changed_files.read_text().split())
    lines = [f"{gpu}={json.dumps(tests[gpu])}" for gpu in GPUS]
    print("\n".join(lines))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
