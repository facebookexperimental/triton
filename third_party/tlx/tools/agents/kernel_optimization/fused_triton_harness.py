from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


_RUNNER = Path(__file__).resolve().parent / "fused_triton_runner.py"


def _environment(target: Mapping[str, Any], name: str) -> str:
    environment = target.get("environment", {})
    if not isinstance(environment, Mapping) or not environment.get(name):
        raise ValueError(f"target environment is missing {name}")
    return str(environment[name])


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    """Stage one generated wrapper for a direct local-Triton evaluation."""
    if target.get("backend") != "cuda":
        return {"success": False, "diagnostics": "fused Triton harness requires CUDA"}
    try:
        tree = ast.parse(kernel_source)
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "call"
            for node in tree.body
        ):
            raise ValueError(
                "candidate must preserve the generated call(args) entry point"
            )

        case_dir = Path(_environment(target, "FUSED_TRITON_CASE_DIR"))
        fbsource_root = Path(_environment(target, "FUSED_TRITON_FBSOURCE_ROOT"))
        temporary = tempfile.TemporaryDirectory(prefix="fused-triton-agent-")
        root = Path(temporary.name)
        shutil.copyfile(case_dir / "manifest.json", root / "manifest.json")
        shutil.copyfile(case_dir / "bench_vs_geo.py", root / "bench_vs_geo.py")
        original_path = root / "output_code.py"
        shutil.copyfile(case_dir / "output_code.py", original_path)
        candidate_path = root / "output_code_fused.py"
        candidate_path.write_text(kernel_source)
        return {
            "success": True,
            "artifact": {
                "temporary": temporary,
                "root": root,
                "benchmark_path": root / "bench_vs_geo.py",
                "original_path": original_path,
                "candidate_path": candidate_path,
                "fbsource_root": fbsource_root,
                "triton_dir": Path(_environment(target, "FUSED_TRITON_DIR")),
                "python": Path(_environment(target, "FUSED_TRITON_PYTHON")),
                "source": kernel_source,
                "target_environment": dict(target.get("environment", {})),
                "evaluation": None,
            },
        }
    except Exception as error:  # noqa: BLE001
        return {
            "success": False,
            "diagnostics": f"{type(error).__name__}: {error}",
        }


def _run_evaluation(
    artifact: dict[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    if artifact["evaluation"] is not None:
        return artifact["evaluation"]

    parameters = case.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError("case parameters must be a mapping")
    warmup_ms = int(parameters.get("warmup_ms", 500))
    benchmark_ms = int(parameters.get("benchmark_ms", 2000))

    def command(
        result_path: Path, only: str = "both", extra: tuple[str, ...] = ()
    ) -> list[str]:
        result = [
            str(artifact["python"]),
            str(_RUNNER),
            "--benchmark",
            str(artifact["benchmark_path"]),
            "--triton-dir",
            str(artifact["triton_dir"]),
            "--fbsource-root",
            str(artifact["fbsource_root"]),
            "--fused",
            str(artifact["candidate_path"]),
            "--original",
            str(artifact["original_path"]),
            "--json",
            str(result_path),
            "--warmup-ms",
            str(warmup_ms),
            "--benchmark-ms",
            str(benchmark_ms),
        ]
        reference_config = artifact["target_environment"].get(
            "FUSED_TRITON_REFERENCE_CONFIG"
        )
        if reference_config:
            result.extend(("--reference-config-json", str(reference_config)))
        result.extend(("--only", only))
        result.extend(extra)
        return result

    environment = dict(os.environ)
    environment.update(
        {str(key): str(value) for key, value in artifact["target_environment"].items()}
    )
    if environment.get("FUSED_TRITON_AUTOWS_IMPLEMENTATION") == "upstream":
        environment.pop("TRITON_USE_META_WS", None)
        environment.pop("TRITON_DISABLE_WSBARRIER_REORDER", None)
    python_path = str(artifact["triton_dir"] / "python")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path

    def invoke(
        result_path: Path,
        only: str,
        run_environment: dict[str, str],
        extra: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        completed = subprocess.run(
            command(result_path, only, extra),
            cwd=artifact["root"],
            env=run_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not result_path.is_file():
            output = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            raise RuntimeError(
                f"local fused Triton {only} evaluation failed with code "
                f"{completed.returncode}: {output[-6000:]}"
            )
        return json.loads(result_path.read_text())

    reference_environment_overrides: dict[str, str] = {}
    reference_environment_json = artifact["target_environment"].get(
        "FUSED_TRITON_REFERENCE_ENV"
    )
    if reference_environment_json:
        parsed = json.loads(reference_environment_json)
        if not isinstance(parsed, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in parsed.items()
        ):
            raise TypeError("FUSED_TRITON_REFERENCE_ENV must encode string pairs")
        reference_environment_overrides = parsed

    if (
        artifact["target_environment"].get("FUSED_TRITON_AUTOWS") == "1"
        or reference_environment_overrides
    ):
        reference_output = artifact["root"] / "reference_output.pt"
        reference_environment = dict(environment)
        reference_environment.pop("TRITON_USE_META_WS", None)
        reference_environment.pop("TRITON_DISABLE_WSBARRIER_REORDER", None)
        reference_environment.update(reference_environment_overrides)
        reference = invoke(
            artifact["root"] / "reference_result.json",
            "reference",
            reference_environment,
            ("--input-seed", "0", "--save-output", str(reference_output)),
        )
        try:
            fused = invoke(
                artifact["root"] / "fused_result.json",
                "fused",
                environment,
                ("--input-seed", "0", "--compare-output", str(reference_output)),
            )
        finally:
            reference_output.unlink(missing_ok=True)
        result = dict(fused)
        result["reference"] = reference["reference"]
    else:
        result = invoke(artifact["root"] / "benchmark_result.json", "both", environment)
    artifact["evaluation"] = result
    return result


def _leg(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    payload = result.get(name, {})
    return payload if isinstance(payload, Mapping) else {}


def verify(artifact: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    try:
        result = _run_evaluation(artifact, case)
        candidate = _leg(result, "fused")
        reference = _leg(result, "reference")
        passed = (
            candidate.get("accuracy") == "PASS"
            and reference.get("accuracy") == "PASS"
            and bool(candidate.get("samples_us"))
            and bool(reference.get("samples_us"))
        )
        metrics = {
            "candidate_median_us": candidate.get("latency_us"),
            "tlx_median_us": reference.get("latency_us"),
        }
        precision = result.get("tlx_precision_comparison")
        if isinstance(precision, Mapping):
            metrics["tlx_precision_comparison"] = dict(precision)
        return {
            "passed": passed,
            "diagnostics": "" if passed else "fused or TLX correctness failed",
            "metrics": metrics,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "passed": False,
            "diagnostics": f"{type(error).__name__}: {error}",
        }


def benchmark(
    artifact: dict[str, Any], case: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    del repetitions
    result = _run_evaluation(artifact, case)
    samples = _leg(result, "fused").get("samples_us")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("fused Triton evaluation did not report timing samples")
    parameters = case.get("parameters", {})
    warmup_ms = int(parameters.get("warmup_ms", 500))
    benchmark_ms = int(parameters.get("benchmark_ms", 2000))
    return {
        "samples_us": samples,
        "warmup_count": 0,
        "cache_policy": (
            "L2-cold per execution; triton.testing.do_bench; "
            f"warmup_ms={warmup_ms}; benchmark_ms={benchmark_ms}; "
            "selected local Triton checkout"
        ),
    }


def profile(
    artifact: dict[str, Any],
    case: dict[str, Any],
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _run_evaluation(artifact, case)
    candidate = _leg(result, "fused")
    reference = _leg(result, "reference")
    candidate_us = float(candidate["latency_us"])
    reference_us = float(reference["latency_us"])
    source = str(artifact["source"])
    call_match = re.search(r"(?ms)^def call\(args\):.*?(?=^def |\Z)", source)
    call_source = call_match.group(0) if call_match else source
    direct_launches = len(re.findall(r"(?m)^\s*[A-Za-z_]\w*\[.*\]\(", call_source))
    return {
        "level": (request or {}).get("level", "summary"),
        "tools": [],
        "scope": "complete generated output_code.call",
        "summary": {
            "candidate_us": candidate_us,
            "tlx_us": reference_us,
            "candidate_to_tlx_ratio": candidate_us / reference_us,
            "triton_package": result.get("triton_package"),
            "wrapper_launch_sites": len(re.findall(r"\.run\(", call_source))
            + direct_launches,
            "embedded_triton_kernels": source.count("async_compile.triton(")
            + len(re.findall(r"(?m)^@triton\.(?:jit|autotune)", source)),
            "materialized_input_cast_launches": call_source.count(
                "triton_poi_fused__to_copy"
            ),
            "uses_ieee_dot": 'input_precision="ieee"' in source,
        },
        "diagnostics": (
            "Profiler-specific counters are not collected by this adapter; "
            "latencies are correctness-gated CUDA-event measurements from the "
            "selected local Triton checkout."
        ),
    }
