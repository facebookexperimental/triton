from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_INLINE_PROFILE_LIMIT_BYTES = 1_000_000
_PROFILE_LEVELS = frozenset({"summary", "deep"})
_RAW_PROFILE_KEYS = frozenset(
    {
        "blob",
        "content",
        "contents",
        "data",
        "events",
        "raw",
        "raw_metrics",
        "raw_profile",
        "rows",
        "trace_events",
    }
)

SUMMARY_NCU_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "duration_us": (
        "duration_us",
        "gpu__time_duration.sum",
        "gpu__time_duration.avg",
        "Duration",
    ),
    "sm_throughput_pct": (
        "sm_throughput_pct",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_throughput_pct": (
        "dram_throughput_pct",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
}

DEEP_NCU_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    **SUMMARY_NCU_METRIC_ALIASES,
    "barrier_pct": ("smsp__warp_issue_stalled_barrier_per_warp_active.pct",),
    "async_wait_pct": ("smsp__warp_issue_stalled_wait_per_warp_active.pct",),
    "long_scoreboard_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    ),
    "short_scoreboard_pct": (
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    ),
    "mio_throttle_pct": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    ),
    "dependency_pct": ("smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",),
    "registers_per_thread": ("launch__registers_per_thread",),
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "theoretical_occupancy_pct": ("launch__occupancy_limit_registers",),
    "local_load_bytes": ("memory_l1_wavefronts_shared",),
    "local_store_bytes": ("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",),
    "spill_loads": ("launch__sass_reg_spill_loads",),
    "spill_stores": ("launch__sass_reg_spill_stores",),
    "l1_bytes": ("l1tex__t_bytes.sum",),
    "l1_hit_rate_pct": ("l1tex__t_sector_hit_rate.pct",),
    "shared_bank_conflicts": ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",),
    "l2_read_bytes": ("lts__t_bytes_op_read.sum",),
    "l2_write_bytes": ("lts__t_bytes_op_write.sum",),
    "l2_hit_rate_pct": ("lts__t_sector_hit_rate.pct",),
    "dram_read_bytes": ("dram__bytes_read.sum",),
    "dram_write_bytes": ("dram__bytes_write.sum",),
    "tensor_activity_pct": ("sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",),
    "issue_active_pct": ("smsp__issue_active.avg.pct_of_peak_sustained_active",),
    "eligible_warps_per_cycle": ("smsp__warps_eligible.avg.per_cycle_active",),
    "active_warps_per_cycle": ("smsp__warps_active.avg.per_cycle_active",),
}

_NCU_FIELD_GROUPS: Mapping[str, str] = {
    "duration_us": "summary",
    "sm_throughput_pct": "summary",
    "dram_throughput_pct": "summary",
    "barrier_pct": "stalls",
    "async_wait_pct": "stalls",
    "long_scoreboard_pct": "stalls",
    "short_scoreboard_pct": "stalls",
    "mio_throttle_pct": "stalls",
    "dependency_pct": "stalls",
    "registers_per_thread": "registers",
    "achieved_occupancy_pct": "registers",
    "theoretical_occupancy_pct": "registers",
    "occupancy_limiters": "registers",
    "local_load_bytes": "registers",
    "local_store_bytes": "registers",
    "spill_loads": "registers",
    "spill_stores": "registers",
    "l1_bytes": "memory",
    "l1_hit_rate_pct": "memory",
    "shared_bank_conflicts": "memory",
    "l2_read_bytes": "memory",
    "l2_write_bytes": "memory",
    "l2_hit_rate_pct": "memory",
    "dram_read_bytes": "memory",
    "dram_write_bytes": "memory",
    "tensor_activity_pct": "compute",
    "issue_active_pct": "compute",
    "eligible_warps_per_cycle": "compute",
    "active_warps_per_cycle": "compute",
}


@dataclass(frozen=True)
class ProfileRequest:
    level: str = "summary"
    tools: tuple[str, ...] = ()
    experiment_id: str = ""
    artifacts_dir: Path | None = None
    reason: str = ""
    diagnostic_only: bool = False
    granularity: str | None = None

    def __post_init__(self) -> None:
        if self.level not in _PROFILE_LEVELS:
            raise ValueError("profile level must be 'summary' or 'deep'")
        object.__setattr__(self, "tools", tuple(str(tool) for tool in self.tools))
        if self.artifacts_dir is not None:
            artifacts_dir = Path(self.artifacts_dir)
            if not artifacts_dir.is_absolute():
                raise ValueError("profile artifacts_dir must be absolute")
            object.__setattr__(self, "artifacts_dir", artifacts_dir)
        if "proton_intra_kernel" in self.tools:
            if not self.diagnostic_only:
                raise ValueError("proton_intra_kernel requires diagnostic_only=True")
            if self.granularity is not None and self.granularity != "warp":
                raise ValueError("proton_intra_kernel only supports warp granularity")

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "tools": list(self.tools),
            "experiment_id": self.experiment_id,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
            "reason": self.reason,
            "diagnostic_only": self.diagnostic_only,
            "granularity": self.granularity,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProfileRequest:
        tools = payload.get("tools", ())
        if isinstance(tools, str):
            tools = (tools,)
        if not isinstance(tools, Iterable):
            raise TypeError("profile tools must be a string or iterable")
        artifacts_dir_raw = payload.get("artifacts_dir")
        artifacts_dir = Path(str(artifacts_dir_raw)) if artifacts_dir_raw else None
        return cls(
            level=str(payload.get("level", "summary")),
            tools=tuple(str(tool) for tool in tools),
            experiment_id=str(payload.get("experiment_id", "")),
            artifacts_dir=artifacts_dir,
            reason=str(payload.get("reason", "")),
            diagnostic_only=bool(payload.get("diagnostic_only", False)),
            granularity=(
                str(payload["granularity"]) if payload.get("granularity") is not None else None
            ),
        )


def normalize_profile_request(profile: bool | ProfileRequest | Mapping[str, Any]) -> ProfileRequest | None:
    if profile is False or profile is None:
        return None
    if profile is True:
        return ProfileRequest()
    if isinstance(profile, ProfileRequest):
        return profile
    if isinstance(profile, Mapping):
        return ProfileRequest.from_mapping(profile)
    raise TypeError("profile must be bool, ProfileRequest, or mapping")


def profile_request_to_json(profile: bool | ProfileRequest | Mapping[str, Any]) -> dict[str, Any] | bool:
    request = normalize_profile_request(profile)
    return request.to_json() if request is not None else False


def safe_case_id(case_id: object) -> str:
    raw = str(case_id)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("._-") or "case"
    if safe == raw and len(safe) <= 96:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:80].rstrip('._-') or 'case'}-{digest}"


def per_case_profile_request(
    request_payload: Mapping[str, Any] | bool | None, case_id: object
) -> dict[str, Any] | None:
    if not request_payload:
        return None
    request = normalize_profile_request(request_payload)
    if request is None:
        return None
    if request.artifacts_dir is None:
        return request.to_json()
    case_dir = request.artifacts_dir / safe_case_id(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    return replace(request, artifacts_dir=case_dir).to_json()


def resolve_profile_tools(
    tools: Iterable[object],
    *,
    native_profiler: str | None = None,
    default: Iterable[object] = (),
) -> tuple[str, ...]:
    """Resolve target-neutral profiler names while preserving request order."""
    requested = tuple(str(tool) for tool in tools) or tuple(str(tool) for tool in default)
    resolved: list[str] = []
    for tool in requested:
        if tool == "native_profiler" and native_profiler is not None:
            tool = native_profiler
        if tool not in resolved:
            resolved.append(tool)
    return tuple(resolved)


def profile_accepts_request(profile_fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(profile_fn)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(object(), {}, {})
    except TypeError:
        return False
    return True


def invoke_profile(
    profile_fn: Callable[..., Any],
    artifact: object,
    case: Mapping[str, Any],
    request_payload: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if profile_accepts_request(profile_fn):
        raw_profile = profile_fn(artifact, case, request_payload)
    else:
        raw_profile = profile_fn(artifact, case)
    if not isinstance(raw_profile, Mapping):
        raise TypeError("profile() must return a mapping")
    return raw_profile


def compact_profile_output(
    raw_profile: Mapping[str, Any], request_payload: Mapping[str, Any] | None
) -> dict[str, Any]:
    serialized = json.dumps(dict(raw_profile))
    if len(serialized.encode("utf-8")) <= _INLINE_PROFILE_LIMIT_BYTES:
        return dict(raw_profile)
    artifacts_dir_raw = request_payload.get("artifacts_dir") if request_payload else None
    if artifacts_dir_raw:
        artifacts_dir = Path(str(artifacts_dir_raw))
        if artifacts_dir.is_absolute():
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / "raw_profile.json"
            artifact_path.write_text(serialized + "\n")
            return {
                "artifact": str(artifact_path),
                "size_bytes": len(serialized.encode("utf-8")),
                "truncated": True,
            }
    return {
        "error": "profile payload exceeded inline limit (1MB)",
        "size_bytes": len(serialized.encode("utf-8")),
        "truncated_keys": list(raw_profile.keys()),
    }


def parse_proton_launch_attribution(payload: Any) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    for node in _iter_nodes(payload):
        children = _node_children(node)
        if children:
            continue
        time_us = _node_time_us(node)
        if time_us is None:
            continue
        name = _node_name(node)
        count = _node_count(node)
        leaves.append({"name": name, "time_us": time_us, "count": count})

    totals = {
        "wrapper_us": 0.0,
        "main_kernel_us": 0.0,
        "non_main_kernel_us": 0.0,
        "leaf_time_us": 0.0,
        "count": 0,
    }
    for leaf in leaves:
        time_us = float(leaf["time_us"] or 0.0)
        totals["leaf_time_us"] += time_us
        totals["count"] += int(leaf["count"] or 0)
        kind = _leaf_kind(leaf["name"])
        if kind == "main_kernel":
            totals["main_kernel_us"] += time_us
        elif kind == "non_main_kernel":
            totals["non_main_kernel_us"] += time_us
        else:
            totals["wrapper_us"] += time_us
    return {
        "schema": "launch_attribution_only",
        "leaves": leaves,
        "totals": totals,
    }


def parse_ncu_csv(csv_text: str) -> dict[str, dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return {}
    headers = {_normalize_header(header): header for header in reader.fieldnames}
    name_key = _first_header(headers, ("metric_name", "name", "metric"))
    value_key = _first_header(headers, ("metric_value", "value"))
    unit_key = _first_header(headers, ("metric_unit", "unit"))
    if name_key is None or value_key is None:
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for row in reader:
        name = str(row.get(name_key, "")).strip()
        if not name:
            continue
        value_raw = str(row.get(value_key, "")).strip()
        unit = str(row.get(unit_key, "")).strip() if unit_key else ""
        metrics[name] = {"value": _parse_metric_value(value_raw), "unit": unit}
    return metrics


def parse_ncu_query_metrics(query_text: str) -> set[str]:
    reader = csv.DictReader(io.StringIO(query_text))
    if reader.fieldnames is not None:
        headers = {_normalize_header(header): header for header in reader.fieldnames}
        name_key = _first_header(
            headers,
            ("metric_name", "name", "metric", "identifier", "metric_identifier"),
        )
        if name_key is not None:
            metrics = {
                str(row.get(name_key, "")).strip()
                for row in reader
                if str(row.get(name_key, "")).strip()
            }
            if metrics:
                return metrics
    metrics: set[str] = set()
    for line in query_text.splitlines():
        token = line.strip().split(",", 1)[0].strip().strip('"')
        if token and "__" in token:
            metrics.add(token)
    return metrics


def select_ncu_metric_names(
    supported_names: Iterable[str], level: str = "summary"
) -> dict[str, Any]:
    aliases = _ncu_aliases_for_level(level)
    supported = {str(name) for name in supported_names}
    selected: dict[str, str | None] = {}
    diagnostics: list[str] = []
    for semantic_name, candidates in aliases.items():
        match = next((candidate for candidate in candidates if candidate in supported), None)
        selected[semantic_name] = match
        if match is None:
            diagnostics.append(f"missing NCU metric for {semantic_name}")
    return {"metrics": selected, "diagnostics": diagnostics}


def normalize_ncu_metrics(raw_metrics: Mapping[str, Any], level: str = "summary") -> dict[str, Any]:
    selected = select_ncu_metric_names(raw_metrics.keys(), level)
    result: dict[str, Any] = {
        "level": level,
        "summary": {},
        "stalls": {},
        "registers": {},
        "memory": {},
        "compute": {},
        "raw_metrics": {},
        "diagnostics": list(selected["diagnostics"]),
    }
    for semantic_name, metric_name in selected["metrics"].items():
        group = _NCU_FIELD_GROUPS.get(semantic_name, "summary")
        if metric_name is None:
            result[group][semantic_name] = None
            continue
        record = raw_metrics[metric_name]
        value = _metric_record_value(record)
        if semantic_name == "duration_us":
            value = _duration_to_us(value, _metric_record_unit(record))
        result[group][semantic_name] = value
        result["raw_metrics"][metric_name] = record
    return result


def extract_ncu_duration_us(profile: Mapping[str, Any]) -> float | None:
    ncu = profile.get("ncu", profile)
    if not isinstance(ncu, Mapping):
        return None
    summary = ncu.get("summary")
    if isinstance(summary, Mapping):
        duration = _coerce_float(summary.get("duration_us"))
        if duration is not None:
            return duration
    for key in SUMMARY_NCU_METRIC_ALIASES["duration_us"]:
        if key in ncu:
            value = ncu[key]
            return _duration_to_us(_metric_record_value(value), _metric_record_unit(value))
    raw_metrics = ncu.get("raw_metrics")
    if isinstance(raw_metrics, Mapping):
        for key in SUMMARY_NCU_METRIC_ALIASES["duration_us"]:
            if key in raw_metrics:
                value = raw_metrics[key]
                return _duration_to_us(_metric_record_value(value), _metric_record_unit(value))
    return None


def ncu_regression_diagnostic(
    baseline_profile: Mapping[str, Any], candidate_profile: Mapping[str, Any]
) -> str:
    baseline_us = extract_ncu_duration_us(baseline_profile)
    candidate_us = extract_ncu_duration_us(candidate_profile)
    if baseline_us is None or candidate_us is None:
        return ""
    if candidate_us > baseline_us * 1.01:
        return (
            "NCU duration regressed: "
            f"candidate {candidate_us:.3f}us > baseline {baseline_us:.3f}us by >1%"
        )
    return ""


def compact_profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    preferred = (
        "level",
        "tools",
        "scope",
        "summary",
        "proton",
        "ncu",
        "native_profiler",
        "diagnostics",
        "diagnostic_proton_intra_kernel",
        "artifacts",
        "artifact",
        "size_bytes",
        "truncated",
        "error",
    )
    compact = {key: _compact_value(profile[key]) for key in preferred if key in profile}
    if compact:
        return compact
    return _compact_value(profile)


def _ncu_aliases_for_level(level: str) -> Mapping[str, tuple[str, ...]]:
    if level == "summary":
        return SUMMARY_NCU_METRIC_ALIASES
    if level == "deep":
        return DEEP_NCU_METRIC_ALIASES
    raise ValueError("profile level must be 'summary' or 'deep'")


def _iter_nodes(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        yield payload
        for child in _node_children(payload):
            yield from _iter_nodes(child)
        for key in ("result", "tree", "list"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    yield from _iter_nodes(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_nodes(item)


def _node_children(node: Mapping[str, Any]) -> list[Any]:
    for key in ("children", "child", "nodes"):
        value = node.get(key)
        if isinstance(value, list):
            return value
    return []


def _node_name(node: Mapping[str, Any]) -> str:
    for key in ("name", "label", "scope", "function", "kernel_name"):
        value = node.get(key)
        if value:
            return str(value)
    return "unknown"


def _node_time_us(node: Mapping[str, Any]) -> float | None:
    for key, multiplier in (
        ("time_us", 1.0),
        ("duration_us", 1.0),
        ("total_time_us", 1.0),
        ("time_ns", 0.001),
        ("duration_ns", 0.001),
        ("time_ms", 1000.0),
        ("duration_ms", 1000.0),
        ("time", 1.0),
        ("duration", 1.0),
    ):
        value = _coerce_float(node.get(key))
        if value is not None:
            return value * multiplier
    metrics = node.get("metrics")
    if isinstance(metrics, Mapping):
        return _node_time_us(metrics)
    return None


def _node_count(node: Mapping[str, Any]) -> int:
    for key in ("count", "calls", "num_calls", "instances"):
        value = _coerce_float(node.get(key))
        if value is not None:
            return int(value)
    return 1


def _leaf_kind(name: object) -> str:
    lowered = str(name).lower()
    if "main" in lowered and "kernel" in lowered:
        return "main_kernel"
    if "kernel" in lowered or "launch" in lowered:
        return "non_main_kernel"
    return "wrapper"


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def _first_header(headers: Mapping[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in headers:
            return headers[candidate]
    return None


def _parse_metric_value(value: str) -> float | str | None:
    parsed = _coerce_float(value)
    return parsed if parsed is not None else value or None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _metric_record_value(record: Any) -> float | str | None:
    if isinstance(record, Mapping):
        return _parse_metric_value(str(record.get("value", "")))
    return _parse_metric_value(str(record))


def _metric_record_unit(record: Any) -> str:
    if isinstance(record, Mapping):
        return str(record.get("unit", ""))
    return ""


def _duration_to_us(value: float | str | None, unit: str) -> float | None:
    duration = _coerce_float(value)
    if duration is None:
        return None
    normalized = unit.strip().lower()
    if normalized in {"ns", "nsecond", "nseconds", "nanosecond", "nanoseconds"}:
        return duration / 1000.0
    if normalized in {"ms", "msecond", "mseconds", "millisecond", "milliseconds"}:
        return duration * 1000.0
    if normalized in {"s", "sec", "second", "seconds"}:
        return duration * 1_000_000.0
    return duration


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in _RAW_PROFILE_KEYS:
                continue
            compact[str(key)] = _compact_value(item)
        return compact
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:50]]
    if isinstance(value, tuple):
        return [_compact_value(item) for item in value[:50]]
    if isinstance(value, (str, bytes)) and len(value) > 4096:
        return f"<omitted {len(value)} bytes>"
    return value
