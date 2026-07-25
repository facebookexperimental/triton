"""Correctness test for the Blackwell symmetric-memory all-gather tutorial."""

import os
import subprocess
import sys
import tempfile

import pytest
import torch


def _is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10


def test_dist_all_gather():
    world_size = 2
    if not _is_blackwell():
        pytest.skip("requires Blackwell GPUs")
    if torch.cuda.device_count() < world_size:
        pytest.skip("requires 2 GPUs")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    tutorial = os.path.join(
        repo_root,
        "third_party",
        "tlx",
        "tutorials",
        "blackwell_dist_all_gather.py",
    )

    # A tiny launcher file makes torch.distributed.run independent of pytest's
    # process state and matches how users invoke the tutorial.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as launcher:
        launcher.write("import runpy\n"
                       f"runpy.run_path({tutorial!r}, run_name='__main__')\n")
        launcher_path = launcher.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world_size}",
                launcher_path,
                "--mode",
                "correctness",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(world_size)),
                "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
            },
        )
        if result.returncode != 0:
            pytest.fail(f"worker failed (rc={result.returncode})\n"
                        f"--- stdout ---\n{result.stdout}\n"
                        f"--- stderr ---\n{result.stderr}\n")
        assert f"correctness PASS: ranks={world_size}" in result.stdout
    finally:
        os.unlink(launcher_path)
