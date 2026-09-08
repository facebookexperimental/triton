from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from .profiling import compact_profile_output, invoke_profile, per_case_profile_request
except ImportError:  # pragma: no cover - subprocess script execution path
    from profiling import compact_profile_output, invoke_profile, per_case_profile_request


def _load_harness(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tlx_kernel_agent_user_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load harness from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_build(result: Any) -> tuple[bool, Any, str]:
    if isinstance(result, Mapping) and "success" in result:
        return (
            bool(result["success"]),
            result.get("artifact"),
            str(result.get("diagnostics", "")),
        )
    return True, result, ""


def _normalize_verification(result: Any) -> dict[str, Any]:
    if isinstance(result, bool):
        return {"passed": result, "diagnostics": "", "metrics": {}}
    if not isinstance(result, Mapping):
        raise TypeError("verify() must return bool or a mapping")
    return {
        "passed": bool(result.get("passed", False)),
        "diagnostics": str(result.get("diagnostics", "")),
        "metrics": dict(result.get("metrics", {})),
    }


def _normalize_timing(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        samples = result.get("samples_us")
        warmup_count = int(result.get("warmup_count", 0))
        cache_policy = str(result.get("cache_policy", "unspecified"))
    else:
        samples = result
        warmup_count = 0
        cache_policy = "unspecified"
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("benchmark() must return timing samples or a mapping")
    return {
        "samples_us": [float(sample) for sample in samples],
        "warmup_count": warmup_count,
        "cache_policy": cache_policy,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    harness = _load_harness(args.harness)
    target = request["target"]
    build_result = harness.build(request["kernel_source"], target)
    success, artifact, diagnostics = _normalize_build(build_result)
    response: dict[str, Any] = {
        "build": {"success": success, "diagnostics": diagnostics},
        "cases": [],
    }
    if not success:
        args.response.write_text(json.dumps(response))
        return 0

    repetitions = int(request["benchmark_repetitions"])
    for case in request["cases"]:
        verification = _normalize_verification(harness.verify(artifact, case))
        case_result: dict[str, Any] = {
            "case_id": case["case_id"],
            "verification": verification,
            "timing": None,
            "profile": {},
        }
        if verification["passed"]:
            case_result["timing"] = _normalize_timing(
                harness.benchmark(artifact, case, repetitions)
            )
            if request.get("profile") and hasattr(harness, "profile"):
                try:
                    profile_request = per_case_profile_request(
                        request.get("profile"), case["case_id"]
                    )
                    raw_profile = invoke_profile(
                        harness.profile, artifact, case, profile_request
                    )
                    case_result["profile"] = compact_profile_output(
                        raw_profile, profile_request
                    )
                except Exception as error:  # noqa: BLE001
                    case_result["profile"] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
        response["cases"].append(case_result)
    args.response.write_text(json.dumps(response))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
