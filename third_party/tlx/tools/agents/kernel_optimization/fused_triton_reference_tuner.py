#!/usr/bin/env python3
"""Correctness-gated TLX reference sweep for a generated fused Triton case."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_RUNNER = Path(__file__).resolve().parent / "fused_triton_runner.py"


def _cv(samples: list[float]) -> float:
    mean = statistics.fmean(samples)
    return statistics.pstdev(samples) / mean if mean else float("inf")


def tune_reference(
    *,
    benchmark: Path,
    fused: Path,
    original: Path,
    triton_dir: Path,
    fbsource_root: Path,
    python_executable: Path,
    candidates_path: Path,
    output_path: Path,
    artifacts_dir: Path,
    warmup_ms: int,
    benchmark_ms: int,
    gpu_id: int,
    max_cv: float,
    environment_overrides: dict[str, str],
) -> dict[str, Any]:
    """Sweep explicit TLX configs and persist the fastest correct stable one."""
    candidate_document = json.loads(candidates_path.read_text())
    candidates = (
        candidate_document.get("configs")
        if isinstance(candidate_document, dict)
        else candidate_document
    )
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("TLX tuning candidates must be a non-empty JSON list")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(environment_overrides)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    python_path = str(triton_dir / "python")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path
    environment.pop("TRITON_USE_META_WS", None)
    environment.pop("TRITON_DISABLE_WSBARRIER_REORDER", None)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fused-triton-tlx-sweep-") as directory:
        temporary = Path(directory)
        for index, raw_candidate in enumerate(candidates):
            if not isinstance(raw_candidate, dict):
                raise TypeError(f"TLX tuning candidate {index} must be an object")
            name = str(raw_candidate.get("name", f"config-{index:03d}"))
            candidate = dict(raw_candidate)
            candidate.pop("name", None)
            config = dict(candidate)
            config.update(
                {
                    "schema_version": 1,
                    "tuned": False,
                    "kernel_config": candidate.get("kernel_config"),
                    "module_overrides": candidate.get("module_overrides", []),
                }
            )
            config_path = temporary / f"config-{index:03d}.json"
            result_path = artifacts_dir / f"result-{index:03d}.json"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            command = [
                str(python_executable),
                str(_RUNNER),
                "--benchmark",
                str(benchmark),
                "--triton-dir",
                str(triton_dir),
                "--fbsource-root",
                str(fbsource_root),
                "--fused",
                str(fused),
                "--original",
                str(original),
                "--json",
                str(result_path),
                "--warmup-ms",
                str(warmup_ms),
                "--benchmark-ms",
                str(benchmark_ms),
                "--reference-config-json",
                str(config_path),
                "--input-seed",
                "0",
                "--only",
                "reference",
            ]
            completed = subprocess.run(
                command,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            record: dict[str, Any] = {
                "index": index,
                "name": name,
                "config": config,
                "returncode": completed.returncode,
            }
            if completed.returncode == 0 and result_path.is_file():
                result = json.loads(result_path.read_text())
                reference = result.get("reference", {})
                samples = reference.get("samples_us", [])
                if reference.get("accuracy") == "PASS" and samples:
                    numeric_samples = [float(sample) for sample in samples]
                    record.update(
                        {
                            "passed": True,
                            "median_us": statistics.median(numeric_samples),
                            "cv": _cv(numeric_samples),
                        }
                    )
                else:
                    record.update(
                        {
                            "passed": False,
                            "diagnostics": reference.get("mismatch", "accuracy failed"),
                        }
                    )
            else:
                diagnostics = "\n".join(
                    part.strip()
                    for part in (completed.stdout, completed.stderr)
                    if part.strip()
                )
                record.update(
                    {
                        "passed": False,
                        "diagnostics": diagnostics[-4000:],
                    }
                )
            results.append(record)
            print(
                f"[tlx-reference-tuner] {name}: "
                + (
                    f"median={record['median_us']:.3f}us cv={record['cv']:.4f}"
                    if record.get("passed")
                    else "rejected: "
                    + str(record.get("diagnostics", "unknown failure")).splitlines()[-1]
                ),
                flush=True,
            )

    eligible = [
        record
        for record in results
        if record.get("passed") and float(record["cv"]) <= max_cv
    ]
    if not eligible:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tuned": False,
                    "tuning": {"results": results},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise RuntimeError("no TLX tuning candidate was both correct and stable")
    winner = min(eligible, key=lambda record: float(record["median_us"]))
    output = dict(winner["config"])
    output["tuned"] = True
    output["tuning"] = {
        "candidate_name": winner["name"],
        "median_us": winner["median_us"],
        "cv": winner["cv"],
        "warmup_ms": warmup_ms,
        "benchmark_ms": benchmark_ms,
        "candidate_count": len(candidates),
        "correct_stable_candidate_count": len(eligible),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"[tlx-reference-tuner] winner={winner['name']} "
        f"median={winner['median_us']:.3f}us output={output_path}",
        flush=True,
    )
    return output


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--triton-dir", type=Path, required=True)
    parser.add_argument("--fbsource-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--benchmark-ms", type=int, default=2000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--reference-env", action="append", default=[])
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    overrides: dict[str, str] = {}
    for assignment in args.reference_env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise SystemExit(f"invalid --reference-env {assignment!r}")
        overrides[name] = value
    tune_reference(
        benchmark=args.benchmark.resolve(),
        fused=args.fused.resolve(),
        original=args.original.resolve(),
        triton_dir=args.triton_dir.resolve(),
        fbsource_root=args.fbsource_root.resolve(),
        python_executable=args.python_executable.resolve(),
        candidates_path=args.candidates_json.resolve(),
        output_path=args.output.resolve(),
        artifacts_dir=args.artifacts_dir.resolve(),
        warmup_ms=args.warmup_ms,
        benchmark_ms=args.benchmark_ms,
        gpu_id=args.gpu_id,
        max_cv=args.max_cv,
        environment_overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
