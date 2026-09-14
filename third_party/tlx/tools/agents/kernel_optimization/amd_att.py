from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_ATT_COUNTERS: tuple[str, ...] = ()


def find_fb_att(environment: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environment is None else environment
    configured = env.get("TLX_FB_ATT")
    if configured:
        path = Path(configured).resolve()
        return path if path.is_file() and os.access(path, os.X_OK) else None

    discovered = shutil.which("fb_att", path=env.get("PATH"))
    if discovered:
        return Path(discovered).resolve()
    default = Path("/usr/local/bin/fb_att")
    return (
        default.resolve() if default.is_file() and os.access(default, os.X_OK) else None
    )


def collect_fb_att(
    workload: Sequence[str],
    artifacts_dir: Path,
    *,
    kernel_filter: str,
    iteration: int = 4,
    counters: Sequence[str] = DEFAULT_ATT_COUNTERS,
    perfcounter_control: int = 3,
    buffer_size: str | None = None,
    timeout_seconds: float = 900.0,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not workload:
        raise ValueError("fb_att workload must not be empty")
    if not kernel_filter.strip():
        raise ValueError("fb_att kernel_filter must not be empty")
    if iteration <= 0:
        raise ValueError("fb_att iteration must be positive")

    executable = find_fb_att(environment)
    if executable is None:
        return {"error": "fb_att was not found; set TLX_FB_ATT or PATH"}

    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="fb-att-", dir=artifacts_dir))
    capture_dir = run_dir / "capture"
    iteration_range = f"[{iteration}-{iteration}]"
    command: list[str] = [
        str(executable),
        "--fb-specify-kernel",
        kernel_filter,
        "--kernel-iteration-range",
        iteration_range,
        "--fb-output-directory",
        str(capture_dir),
    ]
    if counters:
        command.extend(
            (
                "--att-perfcounter-ctrl",
                str(perfcounter_control),
                "--att-perfcounters",
                " ".join(counters),
            )
        )
    if buffer_size:
        command.extend(("--att-buffer-size", buffer_size))
    # fb_att inserts rocprofv3's `--` separator itself.
    command.extend(str(argument) for argument in workload)

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(os.environ if environment is None else environment),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        completed = subprocess.CompletedProcess(command, 1, "", str(error))

    _write_process_artifacts(run_dir, command, completed)
    ui_directories = (
        sorted(path for path in capture_dir.rglob("*_ui") if path.is_dir())
        if capture_dir.exists()
        else []
    )
    wstates = sorted(capture_dir.rglob("wstates*.json")) if capture_dir.exists() else []
    result: dict[str, Any] = {
        "tool": "fb_att",
        "tool_path": str(executable),
        "schema_version": 1,
        "valid": completed.returncode == 0 and bool(wstates),
        "kernel_filter": kernel_filter,
        "iteration_range": iteration_range,
        "counters": list(counters),
        "perfcounter_control": perfcounter_control,
        "artifacts": {
            "directory": str(run_dir),
            "capture_directory": str(capture_dir),
            "ui_directories": [str(path) for path in ui_directories],
            "wstates": [str(path) for path in wstates],
            "command": str(run_dir / "command.json"),
            "stdout": str(run_dir / "stdout.txt"),
            "stderr": str(run_dir / "stderr.txt"),
        },
    }
    if completed.returncode != 0:
        result["error"] = f"fb_att failed with code {completed.returncode}"
    elif not wstates:
        result["error"] = "fb_att produced no wstates JSON"
    return result


def _write_process_artifacts(
    run_dir: Path,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": list(command),
                "returncode": completed.returncode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run_dir / "stdout.txt").write_text(completed.stdout or "")
    (run_dir / "stderr.txt").write_text(completed.stderr or "")
