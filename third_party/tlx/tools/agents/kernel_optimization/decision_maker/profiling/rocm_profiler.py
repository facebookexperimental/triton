from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SUMMARY_COUNTER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("MfmaUtil", "VALUBusy", "MemUnitStalled"),
)

DEEP_COUNTER_GROUPS: tuple[tuple[str, ...], ...] = (
    *SUMMARY_COUNTER_GROUPS,
    ("FETCH_SIZE",),
    ("SQ_LDS_BANK_CONFLICT", "SQ_LDS_ADDR_CONFLICT"),
)

_PROFILER_ENV_TO_CLEAR = (
    "ROCP_TOOL_LIBRARIES",
    "HSA_TOOLS_LIB",
    "ROCPROFILER_LIBRARY_CTOR",
)

_PROFILER_PRELOAD_MARKERS = (
    "fbrocrof",
    "rocprof",
    "roctracer",
)


def find_rocprofv3(environment: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environment is None else environment
    configured = env.get("TLX_ROCPROFV3") or env.get("ROCPROFV3")
    if configured:
        path = Path(configured).resolve()
        return path if path.is_file() and os.access(path, os.X_OK) else None

    discovered = shutil.which("rocprofv3", path=env.get("PATH"))
    if discovered:
        return Path(discovered).resolve()

    candidates = [Path("/opt/rocm/bin/rocprofv3"), Path("/usr/bin/rocprofv3")]
    internal_root = Path("/usr/local/fbcode/platform010/lib")
    if internal_root.is_dir():
        candidates.extend(
            sorted(
                internal_root.glob("rocm-*/bin/rocprofv3"),
                key=_rocm_version_key,
                reverse=True,
            )
        )
    return next(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )


def parse_rocprof_kernel_trace(
    csv_text: str, *, sample_count: int = 3
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return {"summary": {}, "kernels": [], "diagnostics": ["empty kernel trace"]}
    headers = {_normalize_header(name): name for name in reader.fieldnames}
    name_key = _header(headers, "kernel_name", "name")
    start_key = _header(headers, "start_timestamp", "start_timestamp_ns")
    end_key = _header(headers, "end_timestamp", "end_timestamp_ns")
    if name_key is None or start_key is None or end_key is None:
        return {
            "summary": {},
            "kernels": [],
            "diagnostics": ["kernel trace is missing name/start/end columns"],
        }

    timings: dict[str, list[float]] = {}
    resources: dict[str, dict[str, float]] = {}
    for row in reader:
        name = str(row.get(name_key, "")).strip()
        start = _float(row.get(start_key))
        end = _float(row.get(end_key))
        if not name or start is None or end is None or end <= start:
            continue
        timings.setdefault(name, []).append((end - start) / 1000.0)
        resources.setdefault(name, _resource_values(row, headers))

    if not timings:
        return {
            "summary": {},
            "kernels": [],
            "diagnostics": ["kernel trace contains no valid dispatches"],
        }

    eligible = {
        name: values for name, values in timings.items() if len(values) >= sample_count
    } or timings
    kernels = [
        _kernel_summary(name, values, resources.get(name, {}), sample_count)
        for name, values in eligible.items()
    ]
    kernels.sort(key=lambda item: float(item["total_sampled_us"]), reverse=True)
    dominant = kernels[0]
    return {
        "summary": {
            "duration_us": dominant["median_us"],
            "dominant_kernel": dominant["name"],
            "scope": "dominant_kernel",
            "kernel_count": len(kernels),
        },
        "kernels": kernels,
        "diagnostics": [],
    }


def parse_rocprof_counter_collection(
    csv_text: str,
) -> dict[str, dict[str, float]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return {}
    headers = {_normalize_header(name): name for name in reader.fieldnames}
    kernel_key = _header(headers, "kernel_name", "name")
    counter_key = _header(headers, "counter_name", "metric_name")
    value_key = _header(headers, "counter_value", "metric_value", "value")
    if kernel_key is None or counter_key is None or value_key is None:
        return {}

    values: dict[str, dict[str, list[float]]] = {}
    for row in reader:
        kernel = str(row.get(kernel_key, "")).strip()
        counter = str(row.get(counter_key, "")).strip()
        value = _float(row.get(value_key))
        if not kernel or not counter or value is None:
            continue
        values.setdefault(kernel, {}).setdefault(counter, []).append(value)
    return {
        kernel: {
            counter: statistics.median(samples) for counter, samples in counters.items()
        }
        for kernel, counters in values.items()
    }


def filter_extreme_timing_outliers(samples: Sequence[float]) -> list[float]:
    values = list(samples)
    if len(values) < 4:
        return values
    lower_quartile, _, upper_quartile = statistics.quantiles(
        values, n=4, method="inclusive"
    )
    interquartile_range = upper_quartile - lower_quartile
    if interquartile_range <= 0:
        return values
    lower_bound = lower_quartile - 3 * interquartile_range
    upper_bound = upper_quartile + 3 * interquartile_range
    filtered = [value for value in values if lower_bound <= value <= upper_bound]
    return filtered if len(filtered) >= 3 else values


def collect_rocprofv3(
    workload: Sequence[str],
    artifacts_dir: Path,
    *,
    level: str = "summary",
    sample_count: int = 3,
    timeout_seconds: float = 900.0,
    environment: Mapping[str, str] | None = None,
    counter_workload: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not workload:
        raise ValueError("rocprofv3 workload must not be empty")
    if level not in {"timing", "summary", "deep"}:
        raise ValueError("rocprofv3 level must be 'timing', 'summary', or 'deep'")
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profiler = find_rocprofv3(environment)
    if profiler is None:
        return {"error": "rocprofv3 was not found; set TLX_ROCPROFV3 or PATH"}

    run_dir = Path(tempfile.mkdtemp(prefix="rocprofv3-", dir=artifacts_dir))
    env = _profiler_environment(profiler, environment)
    diagnostics: list[str] = []
    trace_dir = run_dir / "trace"
    completed = _run_pass(
        profiler,
        workload,
        trace_dir,
        "trace",
        (),
        env,
        timeout_seconds,
    )
    if completed.returncode != 0:
        return {
            "error": f"rocprofv3 kernel trace failed with code {completed.returncode}",
            "artifacts": _pass_artifacts(run_dir, trace_dir),
        }

    trace_files = sorted(trace_dir.rglob("*kernel_trace.csv"))
    if not trace_files:
        return {
            "error": "rocprofv3 produced no kernel trace CSV",
            "artifacts": _pass_artifacts(run_dir, trace_dir),
        }
    try:
        profile = parse_rocprof_kernel_trace(
            trace_files[0].read_text(), sample_count=sample_count
        )
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return {
            "error": (
                "failed to read rocprofv3 kernel trace: "
                f"{type(error).__name__}: {error}"
            ),
            "artifacts": _pass_artifacts(run_dir, trace_dir),
        }

    counter_groups = (
        ()
        if level == "timing"
        else SUMMARY_COUNTER_GROUPS
        if level == "summary"
        else DEEP_COUNTER_GROUPS
    )
    counter_workload = counter_workload or workload
    counter_values: dict[str, dict[str, float]] = {}
    counter_files: list[Path] = []
    for index, counters in enumerate(counter_groups):
        pass_dir = run_dir / f"counters-{index}"
        completed = _run_pass(
            profiler,
            counter_workload,
            pass_dir,
            f"counters-{index}",
            counters,
            env,
            timeout_seconds,
        )
        if completed.returncode != 0:
            diagnostics.append(
                f"counter group {index} failed with code {completed.returncode}"
            )
            continue
        files = sorted(pass_dir.rglob("*counter_collection*.csv"))
        if not files:
            diagnostics.append(f"counter group {index} produced no CSV")
            continue
        counter_files.extend(files)
        for path in files:
            _merge_counters(
                counter_values,
                parse_rocprof_counter_collection(path.read_text()),
            )

    dominant = profile.get("summary", {}).get("dominant_kernel")
    profile.update(
        {
            "tool": "rocprofv3",
            "tool_path": str(profiler),
            "schema_version": 1,
            "level": level,
            "counters": counter_values.get(str(dominant), {}),
            "diagnostics": [*profile.get("diagnostics", []), *diagnostics],
            "artifacts": {
                "directory": str(run_dir),
                "kernel_trace_csv": str(trace_files[0]),
                "counter_collection_csv": [str(path) for path in counter_files],
            },
        }
    )
    return profile


def _run_pass(
    profiler: Path,
    workload: Sequence[str],
    output_dir: Path,
    output_file: str,
    counters: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(profiler),
        "--mangled-kernels",
        "true",
        "--output-format",
        "csv",
        "--output-directory",
        str(output_dir),
        "--output-file",
        output_file,
    ]
    rocm_root = profiler.parent.parent
    if (rocm_root / "share" / "rocprofiler-sdk").is_dir():
        command.extend(("--rocm-root", str(rocm_root)))
    if counters:
        command.extend(("--pmc", ",".join(counters)))
    else:
        command.extend(("--kernel-trace", "true"))
    command.extend(("--", *[str(argument) for argument in workload]))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(environment),
            cwd=output_dir,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        completed = subprocess.CompletedProcess(command, 1, "", str(error))
    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": command,
                "returncode": completed.returncode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "stdout.txt").write_text(completed.stdout or "")
    (output_dir / "stderr.txt").write_text(completed.stderr or "")
    return completed


def _profiler_environment(
    profiler: Path, environment: Mapping[str, str] | None
) -> dict[str, str]:
    env = dict(os.environ if environment is None else environment)
    for key in _PROFILER_ENV_TO_CLEAR:
        env.pop(key, None)
    preload = _sanitize_ld_preload(env.get("LD_PRELOAD", ""))
    if preload:
        env["LD_PRELOAD"] = preload
    else:
        env.pop("LD_PRELOAD", None)
    rocm_root = profiler.parent.parent
    metrics = rocm_root / "share" / "rocprofiler-sdk"
    if metrics.is_dir():
        env.setdefault("ROCPROFILER_METRICS_PATH", str(metrics))
    return env


def _sanitize_ld_preload(value: str) -> str:
    libraries = (entry for entry in re.split(r"[:\s]+", value) if entry)
    return ":".join(
        library
        for library in libraries
        if not any(marker in library.lower() for marker in _PROFILER_PRELOAD_MARKERS)
    )


def _rocm_version_key(profiler: Path) -> tuple[int, ...]:
    match = re.fullmatch(r"rocm-(\d+(?:\.\d+)*)", profiler.parent.parent.name)
    return (
        tuple(int(component) for component in match.group(1).split("."))
        if match
        else ()
    )


def _pass_artifacts(run_dir: Path, pass_dir: Path) -> dict[str, Any]:
    return {
        "directory": str(run_dir),
        "command": str(pass_dir / "command.json"),
        "stdout": str(pass_dir / "stdout.txt"),
        "stderr": str(pass_dir / "stderr.txt"),
    }


def _merge_counters(
    destination: dict[str, dict[str, float]],
    source: Mapping[str, Mapping[str, float]],
) -> None:
    for kernel, counters in source.items():
        destination.setdefault(kernel, {}).update(counters)


def _kernel_summary(
    name: str,
    durations: Sequence[float],
    resources: Mapping[str, float],
    sample_count: int,
) -> dict[str, Any]:
    sampled = list(durations[-sample_count:])
    return {
        "name": name,
        "dispatches": len(durations),
        "sampled_dispatches": len(sampled),
        "median_us": statistics.median(sampled),
        "total_sampled_us": sum(sampled),
        "samples_us": sampled,
        "resources": dict(resources),
    }


def _resource_values(
    row: Mapping[str, Any], headers: Mapping[str, str]
) -> dict[str, float]:
    aliases = {
        "workgroup_size": "workgroup_size",
        "lds_bytes": "lds_block_size",
        "vgpr_count": "vgpr_count",
        "accum_vgpr_count": "accum_vgpr_count",
        "sgpr_count": "sgpr_count",
    }
    result: dict[str, float] = {}
    for output_name, header_name in aliases.items():
        key = headers.get(header_name)
        value = _float(row.get(key)) if key else None
        if value is not None:
            result[output_name] = value
    return result


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def _header(headers: Mapping[str, str], *names: str) -> str | None:
    return next((headers[name] for name in names if name in headers), None)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
