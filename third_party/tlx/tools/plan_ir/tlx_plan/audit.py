"""Strict M1.4d audit for a captured PlanBundle."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .model import PlanBundle


def audit_plan(plan: PlanBundle) -> dict[str, Any]:
    """Return a deterministic, machine-readable M1.4d acceptance report."""

    plan.validate()
    dependency_kinds = Counter(edge.get("kind", "unknown") for edge in plan.dependency_edges)
    open_important = [
        fact
        for fact in plan.unresolved_facts
        if fact.get("importance") == "important" and fact.get("status") == "open"
    ]
    error_diagnostics = [
        diagnostic for diagnostic in plan.diagnostics if diagnostic.get("severity") == "error"
    ]
    accepted_fallbacks = [
        fact
        for fact in plan.unresolved_facts
        if fact.get("status") == "accepted_deterministic_fallback"
    ]
    reuse_hazards = list(plan.lds_reuse_hazards)

    checks = [
        {
            "name": "referential_integrity_and_layer_hashes",
            "passed": True,
            "details": "PlanBundle.validate completed successfully",
        },
        {
            "name": "no_error_diagnostics",
            "passed": not error_diagnostics,
            "count": len(error_diagnostics),
        },
        {
            "name": "no_open_important_facts",
            "passed": not open_important,
            "count": len(open_important),
        },
        {
            "name": "no_lds_reuse_hazards",
            "passed": not reuse_hazards,
            "count": len(reuse_hazards),
        },
        {
            "name": "physical_resource_claims_are_separated",
            "passed": (
                not plan.resource_summary.get("logical_tensor_bytes_are_per_wave_vgpr_bytes", False)
                and plan.resource_summary.get("physical_vgpr_peak") is None
                and plan.resource_summary.get("physical_lds_bytes") is None
            ),
            "details": "logical TTGIR overlap is not reported as physical allocation",
        },
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": "plan-audit/0.1",
        "passed": passed,
        "kernel": plan.kernel,
        "value_graph_fingerprint": plan.value_graph_fingerprint,
        "checks": checks,
        "summary": {
            "operations": len(plan.operations),
            "values": len(plan.values),
            "dependencies": len(plan.dependency_edges),
            "dependency_kinds": dict(sorted(dependency_kinds.items())),
            "blocks": len(plan.blocks),
            "live_segments": len(plan.live_segments),
            "lds_allocations": len(plan.lds_allocations),
            "async_transactions": len(plan.async_transactions),
            "async_groups": len(plan.async_groups),
            "async_waits": len(plan.async_waits),
            "lds_reuse_hazards": len(reuse_hazards),
            "accepted_identity_fallbacks": len(accepted_fallbacks),
            "open_important_facts": len(open_important),
            "resource_summary": plan.resource_summary,
        },
        "open_important_facts": open_important,
        "error_diagnostics": error_diagnostics,
        "lds_reuse_hazards": reuse_hazards,
    }


def audit_markdown(report: dict[str, Any]) -> str:
    """Render the compact human-readable companion to an audit report."""

    status = "PASS" if report["passed"] else "FAIL"
    summary = report["summary"]
    resources = summary["resource_summary"]
    lines = [
        f"# M1.4d plan audit: {status}",
        "",
        f"Kernel: `{report['kernel']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = check.get("details", f"count={check.get('count', 0)}")
        lines.append(f"- **{marker}** `{check['name']}` — {detail}")
    lines.extend(
        [
            "",
            "## Static TTGIR summary",
            "",
            f"- Operations / values: {summary['operations']} / {summary['values']}",
            f"- Dependencies: {summary['dependencies']} ({summary['dependency_kinds']})",
            f"- Live segments: {summary['live_segments']}",
            f"- LDS allocations: {summary['lds_allocations']}",
            (
                "- Async transactions / groups / waits: "
                f"{summary['async_transactions']} / {summary['async_groups']} / {summary['async_waits']}"
            ),
            f"- Peak logical tensor bytes: {resources.get('peak_logical_tensor_bytes', 0)}",
            f"- Peak logical LDS bytes: {resources.get('peak_logical_lds_bytes', 0)}",
            f"- Maximum logical slot depth: {resources.get('max_logical_slot_depth', 1)}",
            "",
            (
                "All intervals and dependency distances are final structured-TTGIR program order and "
                "iteration distances. Logical tensor bytes are not physical per-wave VGPR bytes; "
                "logical LDS bytes are not assigned physical offsets."
            ),
            "",
        ]
    )
    return "\n".join(lines)
