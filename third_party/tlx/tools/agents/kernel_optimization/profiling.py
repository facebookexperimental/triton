from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_INLINE_PROFILE_LIMIT_BYTES = 1_000_000
_PROFILE_LEVELS = frozenset({"summary", "deep"})
_INTRA_KERNEL_PROFILE_KEY = "diagnostic_proton_intra_kernel"
_PROTON_MAX_TRACE_BYTES = 64 * 1024 * 1024
_PROTON_MAX_EVENTS = 100_000
_PROTON_MAX_TASKS = 32
_PROTON_MAX_WARPS = 128
_PROTON_MAX_SCOPES = 128
_PROTON_MAX_WAITS = 10
_PROTON_MAX_OVERLAPS = 128
_PROTON_MAX_DIAGNOSTICS = 50
_PROTON_MAX_TEXT = 256
_PROTON_PID_RE = re.compile(r"^(?P<kernel>.+?) Core(?P<core>\d+) CTA(?P<cta>\d+)$")
_PROTON_TID_RE = re.compile(r"^warp (?P<warp>\d+) \(line (?P<line>\d+)\)$")
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
        "gpu__time_duration",
        "gpu__time_duration_measured_user",
        "gpu__time_duration_measured_wallclock",
        "Duration",
    ),
    "sm_throughput_pct": (
        "sm_throughput_pct",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput",
    ),
    "dram_throughput_pct": (
        "dram_throughput_pct",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput",
        "gpu__dram_throughput",
    ),
}

DEEP_NCU_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    **SUMMARY_NCU_METRIC_ALIASES,
    "barrier_pct": (
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active",
    ),
    "async_wait_pct": (
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active",
    ),
    "long_scoreboard_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    ),
    "short_scoreboard_pct": (
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active",
    ),
    "mio_throttle_pct": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active",
    ),
    "dependency_pct": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active",
    ),
    "registers_per_thread": ("launch__registers_per_thread",),
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "theoretical_occupancy_pct": ("launch__occupancy_limit_registers",),
    "local_load_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_local_op_ld",
    ),
    "local_store_bytes": (
        "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_bytes_pipe_lsu_mem_local_op_st",
    ),
    "spill_loads": ("launch__sass_reg_spill_loads",),
    "spill_stores": ("launch__sass_reg_spill_stores",),
    "l1_bytes": ("l1tex__t_bytes.sum", "l1tex__t_bytes"),
    "l1_hit_rate_pct": (
        "l1tex__t_sector_hit_rate.pct",
        "l1tex__t_sector_hit_rate",
    ),
    "shared_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared",
    ),
    "l2_read_bytes": ("lts__t_bytes_op_read.sum",),
    "l2_write_bytes": ("lts__t_bytes_op_write.sum",),
    "l2_hit_rate_pct": ("lts__t_sector_hit_rate.pct",),
    "dram_read_bytes": ("dram__bytes_read.sum", "dram__bytes_read"),
    "dram_write_bytes": ("dram__bytes_write.sum", "dram__bytes_write"),
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
class _ProtonEvent:
    name: str
    kernel: str
    core: int
    cta: int
    warp: int
    start_us: float
    end_us: float
    duration_us: float


@dataclass(frozen=True)
class _ProtonTask:
    name: str
    scope: str
    warps: tuple[int, ...]


@dataclass(frozen=True)
class ProfileRequest:
    level: str = "summary"
    tools: tuple[str, ...] = ()
    passes: tuple[str, ...] = ()
    experiment_id: str = ""
    artifacts_dir: Path | None = None
    reason: str = ""
    source_digest: str = ""
    policy_reason: str = ""
    diagnostic_only: bool = False
    granularity: str | None = None

    def __post_init__(self) -> None:
        if self.level not in _PROFILE_LEVELS:
            raise ValueError("profile level must be 'summary' or 'deep'")
        object.__setattr__(self, "tools", tuple(str(tool) for tool in self.tools))
        object.__setattr__(
            self, "passes", tuple(str(pass_name) for pass_name in self.passes)
        )
        if self.passes and "proton_intra_kernel" not in self.tools:
            raise ValueError("profile passes require proton_intra_kernel")
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
            "passes": list(self.passes),
            "experiment_id": self.experiment_id,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
            "reason": self.reason,
            "source_digest": self.source_digest,
            "policy_reason": self.policy_reason,
            "diagnostic_only": self.diagnostic_only,
            "granularity": self.granularity,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProfileRequest:
        tools = _string_tuple(payload.get("tools", ()), "tools")
        passes = _string_tuple(payload.get("passes", ()), "passes")
        artifacts_dir_raw = payload.get("artifacts_dir")
        artifacts_dir = Path(str(artifacts_dir_raw)) if artifacts_dir_raw else None
        return cls(
            level=str(payload.get("level", "summary")),
            tools=tools,
            passes=passes,
            experiment_id=str(payload.get("experiment_id", "")),
            artifacts_dir=artifacts_dir,
            reason=str(payload.get("reason", "")),
            source_digest=str(payload.get("source_digest", "")),
            policy_reason=str(payload.get("policy_reason", "")),
            diagnostic_only=bool(payload.get("diagnostic_only", False)),
            granularity=(
                str(payload["granularity"])
                if payload.get("granularity") is not None
                else None
            ),
        )


def normalize_profile_request(
    profile: bool | ProfileRequest | Mapping[str, Any],
) -> ProfileRequest | None:
    if profile is False or profile is None:
        return None
    if profile is True:
        return ProfileRequest()
    if isinstance(profile, ProfileRequest):
        return profile
    if isinstance(profile, Mapping):
        return ProfileRequest.from_mapping(profile)
    raise TypeError("profile must be bool, ProfileRequest, or mapping")


def profile_request_to_json(
    profile: bool | ProfileRequest | Mapping[str, Any],
) -> dict[str, Any] | bool:
    request = normalize_profile_request(profile)
    return request.to_json() if request is not None else False


def annotate_profile(
    profile: Mapping[str, Any],
    request: ProfileRequest | Mapping[str, Any],
    case_id: object,
) -> dict[str, Any]:
    request_obj = normalize_profile_request(request)
    if request_obj is None:
        raise TypeError("profile metadata requires a profile request")
    annotated = dict(profile)
    annotated["profile_metadata"] = {
        "source_digest": request_obj.source_digest,
        "case_id": str(case_id),
        "experiment_id": request_obj.experiment_id,
        "tools": list(request_obj.tools),
        "passes": list(request_obj.passes),
        "reason": request_obj.reason,
        "policy_reason": request_obj.policy_reason,
        "diagnostic_only": request_obj.diagnostic_only,
        "granularity": request_obj.granularity,
    }
    return annotated


def is_profile_fresh(
    profile: Mapping[str, Any],
    source_digest: str,
    *,
    allow_legacy: bool = True,
) -> bool:
    metadata = profile.get("profile_metadata")
    if not isinstance(metadata, Mapping):
        return allow_legacy
    return str(metadata.get("source_digest", "")) == source_digest


def is_valid_intra_kernel_evidence(profile: Mapping[str, Any]) -> bool:
    layers = _intra_kernel_diagnostic_layers(profile)
    if not layers:
        return False
    for layer in layers:
        if "valid" in layer and layer.get("valid") is not True:
            return False
    diagnostic = layers[-1]
    if diagnostic.get("valid") is not True:
        return False
    validation = diagnostic.get("validation")
    if not isinstance(validation, Mapping):
        return False
    if _validation_has_errors(validation):
        return False
    passes = validation.get("passes")
    if not isinstance(passes, Mapping):
        return False
    for record in passes.values():
        if not isinstance(record, Mapping):
            return False
        if record.get("valid") is not True:
            return False
        if _dropped_event_count(record) > 0:
            return False
        if _dropped_event_count(record.get("drop_warnings")) > 0:
            return False
        if _dropped_event_count(record.get("validation")) > 0:
            return False
    return True


def diagnostic_capabilities(profile: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    diagnostic = _intra_kernel_diagnostic(profile)
    if diagnostic is None:
        return {}
    validation = diagnostic.get("validation")
    if not isinstance(validation, Mapping):
        return {}
    passes = validation.get("passes")
    if not isinstance(passes, Mapping):
        return {}
    capabilities: dict[str, tuple[str, ...]] = {}
    for pass_name, record in passes.items():
        if not isinstance(record, Mapping):
            continue
        expected_regions = record.get("expected_regions", ())
        capabilities[str(pass_name)] = _string_tuple(
            expected_regions,
            "expected_regions",
        )
    return capabilities


def affected_region_upper_bound(
    profile: Mapping[str, Any],
    regions: Iterable[str] | str,
) -> float | None:
    diagnostic = _intra_kernel_diagnostic(profile)
    if diagnostic is None:
        return None
    requested = (
        {str(regions)}
        if isinstance(regions, str)
        else {str(region) for region in regions}
    )
    if not requested:
        return None
    cycles_by_region = _dominant_region_cycles(diagnostic)
    consumer_loop_cycles = cycles_by_region.get("consumer_loop")
    if consumer_loop_cycles is None or consumer_loop_cycles <= 0:
        return None
    matched = requested & cycles_by_region.keys()
    if not matched:
        return None
    ratio = sum(cycles_by_region[region] for region in matched) / consumer_loop_cycles
    return min(1.0, max(0.0, ratio))


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


def resolve_profile_request_for_target(
    request_payload: Mapping[str, Any] | bool | None,
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
    request = normalize_profile_request(request_payload)
    if request is None:
        return None
    backend = str(target.get("backend", "")).strip().lower()
    native_profiler = "ncu" if backend in {"cuda", "nvidia"} else None
    return replace(
        request,
        tools=resolve_profile_tools(
            request.tools,
            native_profiler=native_profiler,
        ),
    ).to_json()


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


def parse_proton_launch_attribution(
    payload: Any, *, main_scope: str | None = None
) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    leaf_kinds: list[str] = []
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
        leaf_kinds.append(_leaf_kind(name, node=node, main_scope=main_scope))

    totals = {
        "wrapper_us": 0.0,
        "main_kernel_us": 0.0,
        "non_main_kernel_us": 0.0,
        "leaf_time_us": 0.0,
        "count": 0,
    }
    for leaf, kind in zip(leaves, leaf_kinds):
        time_us = float(leaf["time_us"] or 0.0)
        totals["leaf_time_us"] += time_us
        totals["count"] += int(leaf["count"] or 0)
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


def parse_proton_intra_kernel_trace(
    trace_path: str | Path, mapping: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize one representative CTA from a Proton warp-granularity trace."""
    path = _validated_proton_trace_path(trace_path)
    result = _empty_proton_intra_kernel_result(path)
    try:
        kernel_pattern, selected_cta, tasks, required_scopes, scope_kinds = (
            _parse_proton_mapping(mapping)
        )
    except (TypeError, ValueError, re.error) as error:
        _proton_diagnostic(result, f"invalid mapping: {error}")
        return result

    if path.stat().st_size > _PROTON_MAX_TRACE_BYTES:
        _proton_diagnostic(
            result,
            f"trace exceeds {_PROTON_MAX_TRACE_BYTES} byte processing limit",
        )
        return result
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _proton_diagnostic(result, f"could not parse Chrome trace: {error}")
        return result
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("traceEvents"), list
    ):
        _proton_diagnostic(result, "Chrome trace must contain a traceEvents list")
        return result
    raw_events = payload["traceEvents"]
    if len(raw_events) > _PROTON_MAX_EVENTS:
        _proton_diagnostic(
            result,
            f"trace contains {len(raw_events)} events; limit is {_PROTON_MAX_EVENTS}",
        )
        return result

    events = _parse_proton_events(raw_events, kernel_pattern)
    groups = _group_proton_events(events)
    selected = _select_proton_cta(groups, tasks, selected_cta, result)
    if selected is None:
        return result
    (selected_kernel, selected_cta_value), selected_events = selected
    result["selected_kernel"] = selected_kernel
    result["selected_cta"] = selected_cta_value
    result["task_mappings"] = {
        task.name: {"scope": task.scope, "warps": list(task.warps)} for task in tasks
    }

    errors = _validate_proton_ownership(selected_events, tasks, required_scopes)
    for error in errors:
        _proton_diagnostic(result, error)
    spans, intervals = _proton_task_spans(selected_events, tasks)
    result["task_spans"] = spans
    result["scope_stats"] = _proton_scope_stats(
        selected_events, tasks, scope_kinds, result
    )
    result["dominant_waits"] = _proton_dominant_waits(result["scope_stats"])
    overlaps = _proton_task_overlaps(tasks, intervals)
    if len(overlaps) > _PROTON_MAX_OVERLAPS:
        _proton_diagnostic(
            result,
            f"pairwise task overlaps truncated to {_PROTON_MAX_OVERLAPS}",
        )
    result["task_overlaps"] = overlaps[:_PROTON_MAX_OVERLAPS]
    result["key_overlaps"] = result["task_overlaps"]
    result["valid"] = not errors and len(overlaps) <= _PROTON_MAX_OVERLAPS
    return result


def _validated_proton_trace_path(trace_path: str | Path) -> Path:
    path = Path(trace_path)
    if not path.is_absolute():
        raise ValueError("Proton trace path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(f"Proton trace does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Proton trace is empty: {path}")
    return path


def _empty_proton_intra_kernel_result(path: Path) -> dict[str, Any]:
    return {
        "schema": "proton_intra_kernel_v1",
        "diagnostic_only": True,
        "valid": False,
        "selected_kernel": None,
        "selected_cta": None,
        "task_mappings": {},
        "task_spans": {},
        "scope_stats": {},
        "dominant_waits": [],
        "task_overlaps": [],
        "key_overlaps": [],
        "diagnostics": [],
        "trace_path": str(path),
    }


def _proton_diagnostic(result: dict[str, Any], message: str) -> None:
    diagnostics = result["diagnostics"]
    if len(diagnostics) < _PROTON_MAX_DIAGNOSTICS:
        diagnostics.append(message[:_PROTON_MAX_TEXT])


def _parse_proton_mapping(
    mapping: Mapping[str, Any],
) -> tuple[re.Pattern[str], int | None, tuple[_ProtonTask, ...], tuple[str, ...], dict[str, str]]:
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be a mapping")
    expected_kernel = mapping.get("expected_kernel")
    if not isinstance(expected_kernel, str) or not expected_kernel.strip():
        raise ValueError("expected_kernel must be a nonempty regex string")
    if len(expected_kernel) > _PROTON_MAX_TEXT:
        raise ValueError(f"expected_kernel exceeds {_PROTON_MAX_TEXT} characters")
    kernel_pattern = re.compile(expected_kernel)
    selected_cta = mapping.get("selected_cta")
    if isinstance(selected_cta, bool) or (
        selected_cta is not None and not isinstance(selected_cta, int)
    ):
        raise TypeError("selected_cta must be an integer")
    if selected_cta is not None and selected_cta < 0:
        raise ValueError("selected_cta must be nonnegative")
    tasks = _parse_proton_tasks(mapping.get("tasks"))
    required_scopes = _parse_proton_names(
        mapping.get("required_scopes"), "required_scopes"
    )
    scope_kinds = _parse_proton_scope_kinds(mapping)
    return kernel_pattern, selected_cta, tasks, required_scopes, scope_kinds


def _parse_proton_tasks(value: Any) -> tuple[_ProtonTask, ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("tasks must be a nonempty mapping")
    if len(value) > _PROTON_MAX_TASKS:
        raise ValueError(f"tasks exceeds {_PROTON_MAX_TASKS} entries")
    tasks: list[_ProtonTask] = []
    owned_warps: set[int] = set()
    owned_scopes: set[str] = set()
    for raw_name in sorted(value, key=str):
        raw_task = value[raw_name]
        name = _bounded_proton_name(raw_name, "task name")
        if not isinstance(raw_task, Mapping):
            raise TypeError(f"task {name!r} must be a mapping")
        scope = _bounded_proton_name(raw_task.get("scope"), f"task {name!r} scope")
        warps = _parse_proton_warps(raw_task.get("warps"), name)
        overlap = owned_warps.intersection(warps)
        if overlap:
            raise ValueError(f"task {name!r} reuses owned warps {sorted(overlap)}")
        if scope in owned_scopes:
            raise ValueError(f"task scope {scope!r} has more than one owner")
        owned_warps.update(warps)
        owned_scopes.add(scope)
        tasks.append(_ProtonTask(name=name, scope=scope, warps=warps))
    return tuple(tasks)


def _parse_proton_warps(value: Any, task_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"task {task_name!r} warps must be a nonempty list")
    warps: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"task {task_name!r} has a non-integer warp")
        if isinstance(item, int):
            start = end = item
        elif isinstance(item, str) and re.fullmatch(r"\d+-\d+", item):
            start, end = (int(part) for part in item.split("-", 1))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        elif isinstance(item, Mapping) and set(item) >= {"start", "end"}:
            start, end = item["start"], item["end"]
        else:
            raise ValueError(f"task {task_name!r} has an invalid warp or range")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(f"task {task_name!r} has an invalid warp range")
        if end - start + 1 > _PROTON_MAX_WARPS:
            raise ValueError(f"task {task_name!r} warp range is too large")
        warps.update(range(start, end + 1))
    if len(warps) > _PROTON_MAX_WARPS:
        raise ValueError(f"task {task_name!r} exceeds {_PROTON_MAX_WARPS} warps")
    return tuple(sorted(warps))


def _parse_proton_names(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    names = tuple(_bounded_proton_name(item, field) for item in value)
    if len(names) > _PROTON_MAX_SCOPES:
        raise ValueError(f"{field} exceeds {_PROTON_MAX_SCOPES} entries")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains duplicates")
    return names


def _parse_proton_scope_kinds(mapping: Mapping[str, Any]) -> dict[str, str]:
    raw_kinds = mapping.get("scope_kinds")
    raw_waits = mapping.get("wait_scopes")
    if raw_kinds is None and raw_waits is None:
        raise ValueError("scope_kinds or wait_scopes is required")
    if raw_kinds is not None and raw_waits is not None:
        raise ValueError("provide scope_kinds or wait_scopes, not both")
    if raw_waits is not None:
        return {name: "wait" for name in _parse_proton_names(raw_waits, "wait_scopes")}
    if not isinstance(raw_kinds, Mapping):
        raise TypeError("scope_kinds must be a mapping")
    kinds: dict[str, str] = {}
    for raw_scope, raw_kind in raw_kinds.items():
        scope = _bounded_proton_name(raw_scope, "scope_kinds scope")
        kind = _bounded_proton_name(raw_kind, f"scope kind for {scope!r}")
        kinds[scope] = kind
    if len(kinds) > _PROTON_MAX_SCOPES:
        raise ValueError(f"scope_kinds exceeds {_PROTON_MAX_SCOPES} entries")
    return kinds


def _bounded_proton_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if len(value) > _PROTON_MAX_TEXT:
        raise ValueError(f"{field} exceeds {_PROTON_MAX_TEXT} characters")
    return value


def _parse_proton_events(
    raw_events: list[Any], kernel_pattern: re.Pattern[str]
) -> tuple[_ProtonEvent, ...]:
    events: list[_ProtonEvent] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping) or raw.get("ph") != "X":
            continue
        pid_match = _PROTON_PID_RE.fullmatch(str(raw.get("pid", "")))
        tid_match = _PROTON_TID_RE.fullmatch(str(raw.get("tid", "")))
        if pid_match is None or tid_match is None:
            continue
        kernel = pid_match.group("kernel")
        if kernel_pattern.search(kernel) is None:
            continue
        name = raw.get("name")
        start_us = _coerce_float(raw.get("ts"))
        duration_us = _coerce_float(raw.get("dur"))
        if (
            not isinstance(name, str)
            or not name
            or len(name) > _PROTON_MAX_TEXT
            or start_us is None
            or duration_us is None
            or not math.isfinite(start_us)
            or not math.isfinite(duration_us)
            or duration_us < 0
        ):
            continue
        events.append(
            _ProtonEvent(
                name=name,
                kernel=kernel,
                core=int(pid_match.group("core")),
                cta=int(pid_match.group("cta")),
                warp=int(tid_match.group("warp")),
                start_us=start_us,
                end_us=start_us + duration_us,
                duration_us=duration_us,
            )
        )
    return tuple(events)


def _group_proton_events(
    events: tuple[_ProtonEvent, ...],
) -> dict[tuple[str, int], tuple[_ProtonEvent, ...]]:
    grouped: dict[tuple[str, int], list[_ProtonEvent]] = {}
    for event in events:
        grouped.setdefault((event.kernel, event.cta), []).append(event)
    return {
        key: tuple(sorted(value, key=lambda event: (event.start_us, event.warp, event.name)))
        for key, value in grouped.items()
    }


def _select_proton_cta(
    groups: Mapping[tuple[str, int], tuple[_ProtonEvent, ...]],
    tasks: tuple[_ProtonTask, ...],
    requested_cta: int | None,
    result: dict[str, Any],
) -> tuple[tuple[str, int], tuple[_ProtonEvent, ...]] | None:
    candidates = [
        (key, events)
        for key, events in groups.items()
        if requested_cta is None or key[1] == requested_cta
    ]
    if not candidates:
        detail = f" CTA {requested_cta}" if requested_cta is not None else ""
        _proton_diagnostic(result, f"no matching complete events found for{detail}")
        return None
    candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    complete = [item for item in candidates if _complete_proton_task_count(item[1], tasks) == len(tasks)]
    if complete:
        return complete[0]
    selected = max(
        candidates,
        key=lambda item: (
            _complete_proton_task_count(item[1], tasks),
            -item[0][1],
            item[0][0],
        ),
    )
    missing = _missing_proton_task_scopes(selected[1], tasks)
    _proton_diagnostic(
        result,
        "no CTA has complete task scopes; selected best partial CTA with missing scopes: "
        + ", ".join(missing),
    )
    return selected


def _complete_proton_task_count(
    events: tuple[_ProtonEvent, ...], tasks: tuple[_ProtonTask, ...]
) -> int:
    return len(tasks) - len(_missing_proton_task_scopes(events, tasks))


def _missing_proton_task_scopes(
    events: tuple[_ProtonEvent, ...], tasks: tuple[_ProtonTask, ...]
) -> list[str]:
    return [
        task.scope
        for task in tasks
        if not any(event.name == task.scope and event.warp in task.warps for event in events)
    ]


def _validate_proton_ownership(
    events: tuple[_ProtonEvent, ...],
    tasks: tuple[_ProtonTask, ...],
    required_scopes: tuple[str, ...],
) -> list[str]:
    warp_owner = {warp: task.name for task in tasks for warp in task.warps}
    scope_owner = {task.scope: task.name for task in tasks}
    errors = [
        f"missing task scope {scope!r}"
        for scope in _missing_proton_task_scopes(events, tasks)
    ]
    for event in events:
        expected_owner = scope_owner.get(event.name)
        if expected_owner is None:
            continue
        actual_owner = warp_owner.get(event.warp)
        if actual_owner != expected_owner:
            errors.append(
                f"scope {event.name!r} on warp {event.warp} belongs to "
                f"{actual_owner or 'no mapped task'}, expected {expected_owner}"
            )
    owned_scope_names = {
        event.name for event in events if event.warp in warp_owner
    }
    errors.extend(
        f"missing required scope {scope!r}"
        for scope in required_scopes
        if scope not in owned_scope_names
    )
    return sorted(set(errors))


def _proton_task_spans(
    events: tuple[_ProtonEvent, ...], tasks: tuple[_ProtonTask, ...]
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    spans: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for task in tasks:
        matching = [
            event
            for event in events
            if event.name == task.scope and event.warp in task.warps
        ]
        if not matching:
            continue
        start = min(event.start_us for event in matching)
        end = max(event.end_us for event in matching)
        intervals[task.name] = (start, end)
        spans[task.name] = end - start
    return spans, intervals


def _proton_scope_stats(
    events: tuple[_ProtonEvent, ...],
    tasks: tuple[_ProtonTask, ...],
    scope_kinds: Mapping[str, str],
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    owned_warps = {warp for task in tasks for warp in task.warps}
    stats: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.warp not in owned_warps:
            continue
        entry = stats.setdefault(
            event.name,
            {
                "kind": scope_kinds.get(event.name, "other"),
                "count": 0,
                "total_us": 0.0,
                "max_us": 0.0,
            },
        )
        entry["count"] += 1
        entry["total_us"] += event.duration_us
        entry["max_us"] = max(entry["max_us"], event.duration_us)
    ordered = {name: stats[name] for name in sorted(stats)}
    if len(ordered) > _PROTON_MAX_SCOPES:
        _proton_diagnostic(result, f"scope stats truncated to {_PROTON_MAX_SCOPES}")
        result["valid"] = False
    return dict(list(ordered.items())[:_PROTON_MAX_SCOPES])


def _proton_dominant_waits(
    scope_stats: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    waits = [
        {
            "name": name,
            "duration_us": stats["total_us"],
            "count": stats["count"],
            "total_us": stats["total_us"],
            "max_us": stats["max_us"],
        }
        for name, stats in scope_stats.items()
        if str(stats.get("kind", "")).lower() == "wait"
    ]
    waits.sort(key=lambda item: (-item["total_us"], -item["max_us"], item["name"]))
    return waits[:_PROTON_MAX_WAITS]


def _proton_task_overlaps(
    tasks: tuple[_ProtonTask, ...], intervals: Mapping[str, tuple[float, float]]
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(tasks):
        if left.name not in intervals:
            continue
        left_start, left_end = intervals[left.name]
        left_duration = left_end - left_start
        for right in tasks[left_index + 1 :]:
            if right.name not in intervals:
                continue
            right_start, right_end = intervals[right.name]
            right_duration = right_end - right_start
            overlap_us = max(0.0, min(left_end, right_end) - max(left_start, right_start))
            overlaps.append(
                {
                    "tasks": [left.name, right.name],
                    "overlap_us": overlap_us,
                    "left_ratio": overlap_us / left_duration if left_duration else 0.0,
                    "right_ratio": overlap_us / right_duration if right_duration else 0.0,
                }
            )
    return overlaps


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


def export_ncu_report_details(
    report_path: str | Path,
    *,
    artifacts_dir: str | Path,
    level: str = "summary",
    ncu_binary: str = "ncu",
    timeout_s: float | None = 120.0,
) -> dict[str, Any]:
    _ncu_aliases_for_level(level)
    report = Path(report_path)
    output_dir = Path(artifacts_dir)
    if not report.is_absolute():
        raise ValueError("NCU report path must be absolute")
    if not output_dir.is_absolute():
        raise ValueError("NCU artifacts directory must be absolute")

    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "import_command.json"
    csv_path = output_dir / "details.csv"
    stderr_path = output_dir / "import_stderr.txt"
    command = [
        ncu_binary,
        "--import",
        str(report),
        "--page",
        "details",
        "--csv",
    ]
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    csv_path.write_text("")
    stderr_path.write_text("")
    artifacts = {
        "ncu_report": str(report),
        "ncu_import_command": str(command_path),
        "ncu_details_csv": str(csv_path),
        "ncu_import_stderr": str(stderr_path),
    }
    diagnostics: list[str] = []
    stdout = ""
    stderr = ""
    return_code: int | None = None

    if not report.is_file():
        diagnostics.append(f"NCU report does not exist: {report}")
    else:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_s,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            if return_code != 0:
                diagnostics.append(
                    f"NCU report import failed with exit code {return_code}"
                )
        except subprocess.TimeoutExpired as error:
            stdout = _subprocess_output_text(error.stdout)
            stderr = _subprocess_output_text(error.stderr)
            diagnostics.append(
                f"NCU report import timed out after {timeout_s} seconds"
            )
        except OSError as error:
            diagnostics.append(
                f"NCU report import failed: {type(error).__name__}: {error}"
            )

    csv_path.write_text(stdout)
    stderr_path.write_text(stderr)
    raw_metrics = parse_ncu_csv(stdout)
    if return_code == 0 and not raw_metrics:
        diagnostics.append("NCU report import produced no metric rows")
    normalized = normalize_ncu_metrics(raw_metrics, level)
    return {
        "success": return_code == 0 and bool(raw_metrics),
        "ncu": normalized,
        "artifacts": artifacts,
        "diagnostics": diagnostics + list(normalized["diagnostics"]),
    }


def parse_ncu_query_metrics(query_text: str) -> set[str]:
    lines = query_text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("$"):
            continue
        header_reader = csv.reader((line,))
        header = next(header_reader, ())
        normalized = {_normalize_header(column) for column in header}
        name_headers = {
            "metric_name",
            "name",
            "metric",
            "identifier",
            "metric_identifier",
        }
        if not normalized.intersection(name_headers):
            continue
        reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
        if reader.fieldnames is None:
            break
        headers = {
            _normalize_header(column): column for column in reader.fieldnames
        }
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
        break
    metrics: set[str] = set()
    for line in lines:
        token = line.strip().split(",", 1)[0].strip().strip('"')
        if token and "__" in token:
            metrics.add(token)
    return metrics


def select_ncu_metric_names(
    supported_names: Iterable[str], level: str = "summary"
) -> dict[str, Any]:
    aliases = _ncu_aliases_for_level(level)
    supported = {str(name) for name in supported_names}
    canonical_supported: dict[str, list[str]] = {}
    supported_base_metrics: set[str] = set()
    for name in sorted(supported, key=lambda value: ("." in value, value)):
        canonical = _canonical_ncu_metric_name(name)
        canonical_supported.setdefault(canonical, []).append(name)
        if canonical == _ncu_metric_family(canonical):
            supported_base_metrics.add(canonical)
    selected: dict[str, str | None] = {}
    diagnostics: list[str] = []
    for semantic_name, candidates in aliases.items():
        match = _select_ncu_metric(candidates, supported, canonical_supported)
        if match is None:
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if "__" in candidate
                    and _ncu_metric_family(candidate) in supported_base_metrics
                ),
                None,
            )
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
        raw_metric_name = _find_raw_ncu_metric_name(raw_metrics, metric_name)
        if raw_metric_name is None:
            result[group][semantic_name] = None
            result["diagnostics"].append(
                "missing collected NCU metric for "
                f"{semantic_name}: {metric_name}"
            )
            continue
        record = raw_metrics[raw_metric_name]
        value = _metric_record_value(record)
        unit = _metric_record_unit(record)
        if semantic_name == "duration_us":
            value = _duration_to_us(value, unit)
        elif semantic_name.endswith("_bytes"):
            value = _bytes_to_bytes(value, unit)
        result[group][semantic_name] = value
        result["raw_metrics"][raw_metric_name] = record
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


def format_intra_kernel_evidence(profile: Mapping[str, Any]) -> str:
    """Format bounded, diagnostic-only Proton evidence for an agent prompt."""
    diagnostic = profile.get("diagnostic_proton_intra_kernel", profile)
    if not isinstance(diagnostic, Mapping):
        return "Unavailable"
    nested = diagnostic.get("diagnostic_proton_intra_kernel")
    if isinstance(nested, Mapping):
        diagnostic = nested

    lines = [
        "Instrumented Proton cycles are diagnostic-only and cannot justify promotion."
    ]
    valid = diagnostic.get("valid")
    lines.append(f"valid={valid if isinstance(valid, bool) else 'unknown'}")
    role_mapping = diagnostic.get("role_mapping")
    if isinstance(role_mapping, Mapping):
        lines.append(
            f"warp_roles={json.dumps(_compact_value(role_mapping), sort_keys=True)}"
        )

    for label, key in (
        ("dominant_phases", "dominant_phases"),
        ("dominant_waits", "dominant_waits"),
    ):
        values = diagnostic.get(key)
        if isinstance(values, list):
            lines.append(
                f"{label}={json.dumps(_compact_value(values[:5]), sort_keys=True)}"
            )

    overlap = diagnostic.get("overlap")
    if isinstance(overlap, Mapping):
        lines.append(
            f"overlap={json.dumps(_compact_value(overlap), sort_keys=True)}"
        )
    validation = diagnostic.get("validation")
    if isinstance(validation, Mapping):
        lines.append(
            f"validation={json.dumps(_compact_value(validation), sort_keys=True)}"
        )
    artifacts = diagnostic.get("artifacts")
    if isinstance(artifacts, Mapping):
        lines.append(
            f"artifacts={json.dumps(_compact_value(artifacts), sort_keys=True)}"
        )
    return "\n".join(lines)[:8000]


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
        "profile_metadata",
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


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise TypeError(f"profile {field_name} must be a string or iterable")
    return tuple(str(item) for item in value)


def _intra_kernel_diagnostic_layers(profile: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    first = profile.get(_INTRA_KERNEL_PROFILE_KEY, profile)
    if not isinstance(first, Mapping):
        return ()
    nested = first.get(_INTRA_KERNEL_PROFILE_KEY)
    if isinstance(nested, Mapping):
        return (first, nested)
    return (first,)


def _intra_kernel_diagnostic(profile: Mapping[str, Any]) -> Mapping[str, Any] | None:
    layers = _intra_kernel_diagnostic_layers(profile)
    if not layers:
        return None
    return layers[-1]


def _validation_has_errors(validation: Mapping[str, Any]) -> bool:
    errors = validation.get("errors", ())
    if isinstance(errors, str):
        return bool(errors)
    if isinstance(errors, Iterable):
        return any(bool(error) for error in errors)
    return bool(errors)


def _dropped_event_count(record: object) -> int:
    if not isinstance(record, Mapping):
        return 0
    count = _coerce_float(record.get("dropped_events"))
    if count is not None:
        return int(count)
    total = 0
    details = record.get("details")
    if isinstance(details, Iterable) and not isinstance(details, (str, bytes)):
        for detail in details:
            if isinstance(detail, Mapping):
                dropped = _coerce_float(detail.get("dropped"))
                if dropped is not None:
                    total += int(dropped)
    return total


def _dominant_region_cycles(diagnostic: Mapping[str, Any]) -> dict[str, float]:
    cycles_by_region: dict[str, float] = {}
    for key in ("dominant_phases", "dominant_waits"):
        records = diagnostic.get(key)
        if not isinstance(records, Iterable) or isinstance(records, (str, bytes)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            name = record.get("name")
            mean_cycles = _coerce_float(record.get("mean_cycles"))
            if name is None or mean_cycles is None:
                continue
            region = str(name)
            cycles_by_region[region] = max(
                cycles_by_region.get(region, 0.0), mean_cycles
            )
    return cycles_by_region


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
    frame = node.get("frame")
    if isinstance(frame, Mapping):
        value = frame.get("name")
        if value:
            return str(value)
    elif frame:
        return str(frame)
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
        value = _coerce_float(_mapping_value(node, key))
        if value is not None:
            return value * multiplier
    metrics = node.get("metrics")
    if isinstance(metrics, Mapping):
        return _node_time_us(metrics)
    return None


def _node_count(node: Mapping[str, Any]) -> int:
    for key in ("count", "calls", "num_calls", "instances"):
        value = _coerce_float(_mapping_value(node, key))
        if value is not None:
            return int(value)
    metrics = node.get("metrics")
    if isinstance(metrics, Mapping):
        return _node_count(metrics)
    return 1


def _leaf_kind(
    name: object,
    *,
    node: Mapping[str, Any] | None = None,
    main_scope: str | None = None,
) -> str:
    if main_scope is not None:
        if str(name).strip() == main_scope.strip():
            return "main_kernel"
        metrics = node.get("metrics") if node is not None else None
        if isinstance(metrics, Mapping):
            device_type = _mapping_value(metrics, "device_type")
            if str(device_type or "").strip().lower() in {"cuda", "gpu"}:
                return "non_main_kernel"
        return "wrapper"
    lowered = str(name).lower()
    if "main" in lowered and "kernel" in lowered:
        return "main_kernel"
    if "kernel" in lowered or "launch" in lowered:
        return "non_main_kernel"
    return "wrapper"


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    normalized_key = _normalize_header(key)
    for candidate, value in mapping.items():
        if _normalize_header(str(candidate)) == normalized_key:
            return value
    return None


def _canonical_ncu_metric_name(name: str) -> str:
    parts = name.split(".")
    for index, part in enumerate(parts):
        if "__" in part:
            return ".".join(parts[index:])
    return name


def _ncu_metric_family(name: str) -> str:
    return _canonical_ncu_metric_name(name).split(".", 1)[0]


def _select_ncu_metric(
    candidates: tuple[str, ...],
    supported: set[str],
    canonical_supported: Mapping[str, list[str]],
) -> str | None:
    for candidate in candidates:
        if candidate in supported:
            if "__" in candidate and "." not in candidate:
                return _preferred_derived_ncu_metric(candidates, candidate)
            return candidate
    for candidate in candidates:
        canonical = _canonical_ncu_metric_name(candidate)
        matches = canonical_supported.get(canonical, ())
        if matches:
            match = matches[0]
            if "." not in _canonical_ncu_metric_name(match):
                return _preferred_derived_ncu_metric(candidates, match)
            return match
    return None


def _preferred_derived_ncu_metric(
    candidates: tuple[str, ...], base_name: str
) -> str:
    family = _ncu_metric_family(base_name)
    return next(
        (
            candidate
            for candidate in candidates
            if _ncu_metric_family(candidate) == family
            and "." in _canonical_ncu_metric_name(candidate)
        ),
        base_name,
    )


def _find_raw_ncu_metric_name(
    raw_metrics: Mapping[str, Any], requested_name: str
) -> str | None:
    if requested_name in raw_metrics:
        return requested_name
    requested_canonical = _canonical_ncu_metric_name(requested_name)
    for name in raw_metrics:
        if _canonical_ncu_metric_name(name) == requested_canonical:
            return name
    requested_family = _ncu_metric_family(requested_name)
    base_matches = [
        name
        for name in raw_metrics
        if _canonical_ncu_metric_name(name) == requested_family
    ]
    if len(base_matches) == 1:
        return base_matches[0]
    return None


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


def _subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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


def _bytes_to_bytes(value: float | str | None, unit: str) -> float | None:
    byte_count = _coerce_float(value)
    if byte_count is None:
        return None
    multiplier = {
        "byte": 1.0,
        "bytes": 1.0,
        "kbyte": 1_000.0,
        "mbyte": 1_000_000.0,
        "gbyte": 1_000_000_000.0,
        "tbyte": 1_000_000_000_000.0,
    }.get(unit.strip().lower(), 1.0)
    return byte_count * multiplier


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
