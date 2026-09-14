from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

_AGENT_DIR = Path(__file__).resolve().parents[4]
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from amd_att import collect_fb_att  # noqa: E402
from rocm_profiler import (  # noqa: E402
    collect_rocprofv3,
    filter_extreme_timing_outliers,
)

_ATT_ITERATION = 4
_ATT_LAUNCHES = 3
_ATT_WARMUP = 2
_COUNTER_BURN_SECONDS = 0.0
_PROFILE_LAUNCHES = 50
_STEADY_STATE_SECONDS = 20.0


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    if str(target.get("backend", "")).lower() not in {"amd", "hip", "rocm"}:
        return {"success": False, "diagnostics": "gfx950 harness requires HIP"}
    if not getattr(torch.version, "hip", None):
        return {
            "success": False,
            "diagnostics": "gfx950 harness requires a ROCm-enabled PyTorch build",
        }
    directory = tempfile.TemporaryDirectory(prefix="tlx-agent-gfx950-gemm-")
    source_path = Path(directory.name) / "candidate.py"
    source_path.write_text(kernel_source)
    try:
        module = _load_module(source_path, "tlx_agent_gfx950_candidate")
    except Exception as error:  # noqa: BLE001
        directory.cleanup()
        return {
            "success": False,
            "diagnostics": f"candidate import failed: {type(error).__name__}: {error}",
        }
    if not callable(getattr(module, "matmul", None)):
        directory.cleanup()
        return {"success": False, "diagnostics": "candidate must define matmul(a, b)"}
    return {
        "success": True,
        "artifact": (
            directory,
            module,
            source_path,
            str(target.get("device") or "cuda"),
        ),
    }


def verify(
    artifact: tuple[Any, ModuleType, Path, str], case: dict[str, Any]
) -> dict[str, Any]:
    _, module, _, device = artifact
    a, b = _inputs(case, device)
    try:
        actual = module.matmul(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    except Exception as error:  # noqa: BLE001
        return {"passed": False, "diagnostics": f"{type(error).__name__}: {error}"}
    return {"passed": True}


def benchmark(
    artifact: tuple[Any, ModuleType, Path, str],
    case: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    _, _, source_path, device = artifact
    launches = max(repetitions, _PROFILE_LAUNCHES)
    burn_seconds = _steady_state_seconds()
    with tempfile.TemporaryDirectory(prefix="tlx-agent-gfx950-timing-") as directory:
        case_path = Path(directory) / "case.json"
        case_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
        profile_result = collect_rocprofv3(
            _profile_workload_command(
                source_path,
                case_path,
                device,
                burn_seconds=burn_seconds,
                launches=launches,
            ),
            Path(directory),
            level="timing",
            sample_count=launches,
            timeout_seconds=max(120.0, burn_seconds + 60.0),
        )
    if error := profile_result.get("error"):
        raise RuntimeError(str(error))
    kernels = profile_result.get("kernels", [])
    raw_samples = kernels[0].get("samples_us", []) if kernels else []
    if not raw_samples:
        raise RuntimeError(
            f"rocprofv3 produced no kernel timing samples: {profile_result}"
        )
    samples = filter_extreme_timing_outliers(raw_samples)
    return {
        "samples_us": samples,
        "warmup_count": 0,
        "cache_policy": f"steady_state_{burn_seconds:g}s_rocprofv3_iqr3",
    }


def profile(
    artifact: tuple[Any, ModuleType, Path, str],
    case: dict[str, Any],
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _, _, source_path, device = artifact
    request = request or {}
    tools = [str(tool) for tool in request.get("tools", ())]
    result: dict[str, Any] = {
        "level": str(request.get("level", "summary")),
        "tools": tools,
        "scope": "dominant_kernel",
    }
    if not bool(case.get("parameters", {}).get("profile", True)):
        result["diagnostics"] = ["rocprofv3 skipped for this case"]
        return result
    if not {"rocprofv3", "fb_att"}.intersection(tools):
        result["diagnostics"] = ["no supported AMD profiler was requested"]
        return result
    artifacts_dir_raw = request.get("artifacts_dir")
    if not artifacts_dir_raw:
        result["rocprofv3"] = {"error": "profile request has no artifacts_dir"}
        return result

    artifacts_dir = Path(str(artifacts_dir_raw))
    case_path = artifacts_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
    burn_seconds = _steady_state_seconds()
    if "rocprofv3" in tools:
        result["rocprofv3"] = collect_rocprofv3(
            _profile_workload_command(
                source_path,
                case_path,
                device,
                burn_seconds=burn_seconds,
                launches=_PROFILE_LAUNCHES,
            ),
            artifacts_dir,
            level=str(result["level"]),
            sample_count=_PROFILE_LAUNCHES,
            timeout_seconds=max(120.0, burn_seconds + 60.0),
            counter_workload=_profile_workload_command(
                source_path,
                case_path,
                device,
                burn_seconds=_COUNTER_BURN_SECONDS,
                launches=10,
            ),
        )
    if "fb_att" in tools:
        rocprof = result.get("rocprofv3", {})
        summary = rocprof.get("summary", {}) if isinstance(rocprof, Mapping) else {}
        kernel_filter = str(
            case.get("parameters", {}).get("profile_kernel", "")
            or summary.get("dominant_kernel", "")
        ).removesuffix(".kd")
        if kernel_filter:
            result["fb_att"] = collect_fb_att(
                _profile_workload_command(
                    source_path,
                    case_path,
                    device,
                    burn_seconds=0.0,
                    warmup=_ATT_WARMUP,
                    launches=_ATT_LAUNCHES,
                ),
                artifacts_dir,
                kernel_filter=kernel_filter,
                iteration=_ATT_ITERATION,
                timeout_seconds=180.0,
            )
        else:
            result["fb_att"] = {
                "error": "fb_att requires a dominant kernel from rocprofv3"
            }
    return result


def _profile_workload_command(
    source_path: Path,
    case_path: Path,
    device: str,
    *,
    burn_seconds: float,
    launches: int,
    warmup: int = 0,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-workload",
        "--candidate",
        str(source_path),
        "--case-json",
        str(case_path),
        "--device",
        device,
        "--burn-seconds",
        str(burn_seconds),
        "--warmup",
        str(warmup),
        "--launches",
        str(launches),
    ]


def _steady_state_seconds() -> float:
    value = float(os.environ.get("TLX_AMD_STEADY_STATE_SECONDS", _STEADY_STATE_SECONDS))
    if value < 0:
        raise ValueError("TLX_AMD_STEADY_STATE_SECONDS must be non-negative")
    return value


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(case: Mapping[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = case["parameters"]
    m = int(parameters["m"])
    n = int(parameters["n"])
    k = int(parameters["k"])
    dtype = getattr(torch, str(parameters.get("dtype", "float16")))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(parameters.get("seed", 0)))
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)
    return a, b


def _profile_workload(args: argparse.Namespace) -> None:
    case = json.loads(args.case_json.read_text())
    module = _load_module(args.candidate, "tlx_agent_gfx950_profile_candidate")
    a, b = _inputs(case, args.device)
    # Compile before warmup so ATT iteration 4 means the first measured launch
    # after this launch plus two explicit warmups.
    module.matmul(a, b)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        module.matmul(a, b)
    torch.cuda.synchronize()
    deadline = time.monotonic() + args.burn_seconds
    while time.monotonic() < deadline:
        module.matmul(a, b)
    torch.cuda.synchronize()
    for _ in range(args.launches):
        module.matmul(a, b)
    torch.cuda.synchronize()


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-workload", action="store_true")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--burn-seconds", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--launches", type=int, default=_PROFILE_LAUNCHES)
    args = parser.parse_args()
    if not args.profile_workload:
        parser.error("this entry point is only for rocprofv3 workload execution")
    _profile_workload(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
