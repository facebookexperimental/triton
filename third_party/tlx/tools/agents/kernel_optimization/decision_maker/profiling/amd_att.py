from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import tempfile
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .rocm_profiler import find_rocprofv3

_DECODER_PATTERNS = ("librocprof-trace-decoder.so*", "libatt_decoder_trace.so")
_ATT_IDENTIFIER_COLUMNS = frozenset(
    {"agent", "codeobj", "dispatch", "pc", "se", "simd", "vaddr", "wave"}
)


def find_amd_att(
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path] | None:
    """Find a compatible ATT-capable rocprofv3 and decoder directory."""
    env = os.environ if environment is None else environment
    profiler = find_rocprofv3(env)
    if profiler is None:
        return None
    try:
        completed = subprocess.run(
            [str(profiler), "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=60.0,
            env=dict(env),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    help_text = (completed.stdout or "") + (completed.stderr or "")
    if "--att" not in help_text and "--advanced-thread-trace" not in help_text:
        return None

    candidates: list[Path] = []
    configured = env.get("ROCPROF_ATT_LIBRARY_PATH")
    if configured:
        candidates.extend(Path(item) for item in configured.split(":") if item)
    candidates.append(profiler.parent.parent / "lib")
    candidates.extend((Path("/opt/rocm/lib"), Path("/opt/rocm/lib/rocprofiler")))
    for candidate in candidates:
        if any(
            path.is_file()
            for pattern in _DECODER_PATTERNS
            for path in candidate.glob(pattern)
        ):
            return profiler.resolve(), candidate.resolve()
    return None


def collect_amd_att(
    workload: Sequence[str],
    artifacts_dir: Path,
    *,
    kernel_filter: str,
    iteration: int = 4,
    timeout_seconds: float = 900.0,
    environment: Mapping[str, str] | None = None,
    target_cus: Sequence[int] = tuple(range(8)),
) -> dict[str, Any]:
    """Collect and validate one rocprofv3 Advanced Thread Trace dispatch."""
    if not workload:
        raise ValueError("AMD ATT workload must not be empty")
    if not kernel_filter.strip():
        raise ValueError("AMD ATT kernel_filter must not be empty")
    if iteration <= 0:
        raise ValueError("AMD ATT iteration must be positive")
    if not target_cus or any(target_cu < 0 for target_cu in target_cus):
        raise ValueError("AMD ATT target_cus must contain nonnegative CU indices")

    capability = find_amd_att(environment)
    if capability is None:
        return {
            "error": "no ATT-capable rocprofv3 and compatible decoder were found"
        }
    profiler, decoder_dir = capability
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="amd-att-", dir=artifacts_dir))
    env = dict(os.environ if environment is None else environment)
    env["ROCPROF_ATT_LIBRARY_PATH"] = str(decoder_dir)
    attempts: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"valid": False, "errors": []}
    capture_dir = run_dir / f"capture-cu{target_cus[0]}"
    selected_target_cu = target_cus[0]
    for target_cu in target_cus:
        capture_dir = run_dir / f"capture-cu{target_cu}"
        command = _att_command(
            profiler,
            decoder_dir,
            capture_dir,
            workload,
            kernel_filter=kernel_filter,
            iteration=iteration,
            target_cu=target_cu,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            completed = subprocess.CompletedProcess(command, 1, "", str(error))
        attempt_dir = run_dir / f"attempt-cu{target_cu}"
        _write_process_artifacts(attempt_dir, command, completed)
        validation = (
            _validate_bundle(capture_dir)
            if completed.returncode == 0
            else {
                "valid": False,
                "errors": [
                    f"rocprofv3 ATT failed with code {completed.returncode}"
                ],
            }
        )
        attempts.append({
            "target_cu": target_cu,
            "returncode": completed.returncode,
            "valid": bool(validation["valid"]),
            "errors": validation.get("errors", []),
            "artifacts": str(attempt_dir),
        })
        selected_target_cu = target_cu
        if validation["valid"]:
            break
    result: dict[str, Any] = {
        "tool": "rocprofv3_att",
        "tool_path": str(profiler),
        "decoder_directory": str(decoder_dir),
        "schema_version": 1,
        "valid": bool(validation["valid"]),
        "kernel_filter": kernel_filter,
        "iteration_range": f"[{iteration}]",
        "target_cu": selected_target_cu,
        "attempts": attempts,
        "validation": validation,
        "artifacts": {
            "directory": str(run_dir),
            "capture_directory": str(capture_dir),
            "ui_directories": validation.get("ui_directories", []),
            "attempt_directory": str(run_dir / f"attempt-cu{selected_target_cu}"),
        },
    }
    if not result["valid"]:
        result["error"] = "; ".join(validation.get("errors", ()))
    return result


def _att_command(
    profiler: Path,
    decoder_dir: Path,
    capture_dir: Path,
    workload: Sequence[str],
    *,
    kernel_filter: str,
    iteration: int,
    target_cu: int,
) -> list[str]:
    return [
        str(profiler),
        "-d",
        str(capture_dir),
        "-o",
        "tlx_att",
        "--att",
        "--att-library-path",
        str(decoder_dir),
        "--kernel-trace",
        "--att-activity",
        "8",
        "--att-target-cu",
        str(target_cu),
        "--att-shader-engine-mask",
        "0x1",
        "--att-simd-select",
        "0xF",
        "--kernel-include-regex",
        kernel_filter,
        "--kernel-iteration-range",
        f"[{iteration}]",
        "--",
        *(str(argument) for argument in workload),
    ]


def compare_amd_att_profiles(
    tlx_profile: Mapping[str, Any], aten_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a compact, prompt-safe comparison from two validated ATT bundles."""
    tlx_stats = _att_numeric_stats(tlx_profile)
    aten_stats = _att_numeric_stats(aten_profile)
    common = sorted(tlx_stats.keys() & aten_stats.keys())
    ratios = {
        name: tlx_stats[name] / aten_stats[name]
        for name in common
        if aten_stats[name] != 0.0
    }
    return {
        "valid": bool(tlx_profile.get("valid")) and bool(aten_profile.get("valid")),
        "tlx_stats": tlx_stats,
        "aten_stats": aten_stats,
        "tlx_over_aten": ratios,
        "diagnostics": (
            []
            if common
            else ["validated ATT bundles had no common numeric stats columns"]
        ),
    }


def _att_numeric_stats(profile: Mapping[str, Any]) -> dict[str, float]:
    validation = profile.get("validation", {})
    if not isinstance(validation, Mapping):
        return {}
    paths = validation.get("stats_files", ())
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        return {}
    values: dict[str, list[float]] = {}
    for raw_path in paths:
        try:
            with Path(str(raw_path)).open(newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    for name, raw_value in row.items():
                        if str(name).lower().replace("_", "") in _ATT_IDENTIFIER_COLUMNS:
                            continue
                        try:
                            value = float(str(raw_value).replace(",", ""))
                        except (TypeError, ValueError):
                            continue
                        values.setdefault(str(name), []).append(value)
        except (OSError, csv.Error):
            continue
    return {
        name: statistics.median(samples)
        for name, samples in sorted(values.items())
        if samples
    }


def _validate_bundle(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    databases = _nonempty(root.rglob("*_results.db")) if root.is_dir() else []
    raw_traces = _nonempty(root.rglob("*.att")) if root.is_dir() else []
    code_objects = _nonempty(root.rglob("*.out")) if root.is_dir() else []
    stats_files = (
        _nonempty(root.rglob("stats_ui_output_agent_*_dispatch_*.csv"))
        if root.is_dir()
        else []
    )
    ui_dirs = (
        sorted(
            path
            for path in root.rglob("ui_output_agent_*_dispatch_*")
            if path.is_dir()
        )
        if root.is_dir()
        else []
    )
    for label, values in (
        ("results database", databases),
        ("raw ATT trace", raw_traces),
        ("code object", code_objects),
        ("decoded stats CSV", stats_files),
        ("decoded UI directory", ui_dirs),
    ):
        if not values:
            errors.append(f"missing non-empty {label}")
    for database in databases:
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            connection.close()
            if integrity != ("ok",):
                errors.append(f"SQLite integrity check failed for {database}")
        except sqlite3.Error as error:
            errors.append(f"could not read {database}: {error}")
    for stats_file in stats_files:
        try:
            with stats_file.open(newline="") as stream:
                if sum(1 for _ in csv.reader(stream)) < 2:
                    errors.append(f"stats CSV has no data rows: {stats_file}")
        except (OSError, csv.Error) as error:
            errors.append(f"could not read {stats_file}: {error}")
    ui_paths: list[str] = []
    for ui_dir in ui_dirs:
        ui_paths.append(str(ui_dir))
        for filename in ("code.json", "filenames.json", "occupancy.json"):
            if not _nonempty((ui_dir / filename,)):
                errors.append(f"missing non-empty {filename} in {ui_dir}")
        if not _nonempty(ui_dir.glob("wstates*.json")):
            errors.append(f"missing wstates JSON in {ui_dir}")
        if not _nonempty(ui_dir.glob("se*_sm*_sl*_wv*.json")):
            errors.append(f"missing per-wave JSON in {ui_dir}")
        for json_file in _nonempty(ui_dir.glob("*.json")):
            try:
                json.loads(json_file.read_text())
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"invalid JSON {json_file}: {error}")
    return {
        "valid": not errors,
        "root": str(root),
        "results_databases": [str(path) for path in databases],
        "raw_traces": [str(path) for path in raw_traces],
        "code_objects": [str(path) for path in code_objects],
        "stats_files": [str(path) for path in stats_files],
        "ui_directories": ui_paths,
        "errors": errors,
    }


def _nonempty(paths) -> list[Path]:
    return sorted(
        path for path in paths if path.is_file() and path.stat().st_size > 0
    )


def _write_process_artifacts(
    run_dir: Path,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(
            {"argv": list(command), "returncode": completed.returncode},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run_dir / "stdout.txt").write_text(completed.stdout or "")
    (run_dir / "stderr.txt").write_text(completed.stderr or "")
