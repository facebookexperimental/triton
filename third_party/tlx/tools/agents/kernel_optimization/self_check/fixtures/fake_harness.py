from __future__ import annotations

import re
from typing import Any


def build(kernel_source: str, target: dict[str, Any]) -> dict[str, Any]:
    del target
    print("synthetic compiler log")
    latency_match = re.search(r"LATENCY_US\s*=\s*([0-9.]+)", kernel_source)
    correct_match = re.search(r"CORRECT\s*=\s*(True|False)", kernel_source)
    if latency_match is None or correct_match is None:
        return {"success": False, "diagnostics": "missing fake kernel controls"}
    return {
        "success": True,
        "artifact": {
            "latency_us": float(latency_match.group(1)),
            "correct": correct_match.group(1) == "True",
        },
    }


def build_experiment(
    kernel_source: str,
    target: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    payload = experiment.get("payload")
    if not isinstance(payload, dict) or not payload:
        return {"success": False, "diagnostics": "missing experiment payload"}
    result = build(kernel_source, target)
    artifact = result.get("artifact")
    if result.get("success") and isinstance(artifact, dict):
        if "latency_us" in payload:
            artifact["latency_us"] = float(payload["latency_us"])
        if "correct" in payload:
            artifact["correct"] = bool(payload["correct"])
    return result


def verify(artifact: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    del case
    return {
        "passed": artifact["correct"],
        "diagnostics": "" if artifact["correct"] else "intentional mismatch",
    }


def benchmark(
    artifact: dict[str, Any], case: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    scale = float(case["parameters"].get("scale", 1.0))
    return {
        "samples_us": [artifact["latency_us"] * scale] * repetitions,
        "warmup_count": 2,
        "cache_policy": "warm",
    }


def profile(artifact: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    del artifact, case
    return {"bottleneck": "synthetic"}
