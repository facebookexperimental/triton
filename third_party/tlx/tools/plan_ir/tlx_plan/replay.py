"""Exact baseline replay verification and layered PlanBundle comparison."""

from __future__ import annotations

from typing import Any

from .model import PlanBundle
from .ttgir import extract_plan, normalize_ttgir


def compare_plans(expected: PlanBundle, actual: PlanBundle) -> dict[str, Any]:
    expected.refresh_hashes()
    actual.refresh_hashes()
    layers = sorted(set(expected.layer_hashes) | set(actual.layer_hashes))
    comparisons = {
        layer: {
            "match": expected.layer_hashes.get(layer) == actual.layer_hashes.get(layer),
            "expected": expected.layer_hashes.get(layer),
            "actual": actual.layer_hashes.get(layer),
        }
        for layer in layers
    }
    return {
        "exact": expected.normalized_ir_hash == actual.normalized_ir_hash,
        "semantic_match": all(value["match"] for value in comparisons.values()),
        "kernel_match": expected.kernel == actual.kernel,
        "layers": comparisons,
        "expected_diagnostics": expected.diagnostics,
        "actual_diagnostics": actual.diagnostics,
    }


def verify_replay(
    text: str,
    expected: PlanBundle,
    *,
    native_value_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-extract final TTGIR and verify an exact baseline PlanBundle replay."""
    actual = extract_plan(
        text,
        source_name=expected.provenance.get("source_name", ""),
        native_value_graph=native_value_graph,
    )
    # Provenance and case are external contracts, not properties recoverable
    # from TTGIR. Preserve them before comparing the executable plan layers.
    actual.provenance = expected.provenance
    actual.case = expected.case
    return compare_plans(expected, actual)


def replay_normalized(
    text: str,
    expected: PlanBundle,
    *,
    native_value_graph: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return deterministic IR plus proof that it represents ``expected``.

    M1.3 deliberately supports exact replay only. Applying mutated schedules or
    storage placements is a later lowering milestone and must not be confused
    with verification of a captured baseline.
    """
    return normalize_ttgir(text), verify_replay(
        text, expected, native_value_graph=native_value_graph
    )
