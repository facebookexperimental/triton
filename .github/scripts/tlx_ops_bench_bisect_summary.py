#!/usr/bin/env python3
"""Render measured bisect or recovery results as a GitHub Actions job summary."""
import argparse
import json
import os
from pathlib import Path

from tlx_ops_bench_bisect import valid_run


def measurement_rows(measurements, commits, repo):
    rows = ["| Wheel | Commit | Measurement | TFLOP/s | Accepted CV |", "| --- | --- | --- | ---: | --- |"]
    for role, sha in commits.items():
        if not sha:
            continue
        m = measurements.get(role) or {}
        status = m.get("status", "not measured")
        value = f"{m['tflops']:.2f}" if status == "ok" else "not available"
        cvs = ", ".join(f"{r['cv'] * 100:.2f}%" for r in m.get("runs", []) if valid_run(r)) or "not available"
        link = f"[{sha[:8]}](https://github.com/{repo}/commit/{sha})"
        rows.append(f"| {role} | {link} | {status} | {value} | {cvs} |")
    return rows


def bisect_summary(prepared, state, measurements, repo):
    lines = ["# Throughput bisection", ""]
    results = state.get("results", [])
    for result in results:
        lines.extend([
            f"## Issue [{result['issue_number']}](https://github.com/{repo}/issues/{result['issue_number']})", "",
            f"Result: **{result['status']}**.", "", f"Case: `{result['key']}`.", ""
        ])
        if result.get("status") == "conclusive" and result.get("verified") and result.get("first_bad_sha"):
            sha = result["first_bad_sha"]
            lines.extend([f"First bad commit: [{sha}](https://github.com/{repo}/commit/{sha}).", ""])
        lines.extend([result.get("reason", ""), ""])
        endpoints = result.get("endpoints", {})
        lines.extend(measurement_rows(endpoints, {"good": result["good_sha"], "bad": result["bad_sha"]}, repo))
        if endpoints.get("good", {}).get("status") == endpoints.get("bad", {}).get("status") == "ok":
            loss = 100 * (1 - endpoints["bad"]["tflops"] / endpoints["good"]["tflops"])
            lines.extend(["", f"Measured throughput loss: **{loss:.1f}%**."])
        lines.append("")
        if result.get("gpu_uuid"):
            lines.extend([f"GPU UUID: `{result['gpu_uuid']}`.", ""])
    if state["index"] < len(state["specs"]):
        spec = state["specs"][state["index"]]
        lines.extend([
            "## Bisection continues", "", f"Case: `{spec['key']}`.", "",
            "No first-bad commit has been established for this case.", ""
        ])
        if prepared.get("spec") == spec:
            lines.extend(
                measurement_rows(
                    measurements,
                    {"good": spec["good_sha"], "bad": spec["bad_sha"], "candidate": prepared.get("candidate_sha")},
                    repo))
        if state.get("log"):
            codes = {0: "good", 1: "bad", 125: "skipped"}
            entry = state["log"][-1]
            if entry.get("candidate"):
                lines.extend(["", f"This round classified `{entry['candidate'][:8]}` as **{codes[entry['code']]}**."])
    elif not results:
        lines.append("No cases were selected for bisection.")
    lines.extend([
        "", "Artifacts: `perf-bisect-state` contains the decisions and measurements; "
        "`perf-bisect-measurements` contains the per-wheel measurements."
    ])
    return "\n".join(lines) + "\n"


def recovery_summary(spec, measurements, repo, baseline):
    before, after = measurements.get("good", {}), measurements.get("bad", {})
    paired = (before.get("status") == after.get("status") == "ok" and before.get("gpu_uuid")
              and before["gpu_uuid"] == after.get("gpu_uuid"))
    recovered = paired and before["tflops"] < 0.9 * baseline <= after["tflops"]
    verdict = "recovery confirmed" if recovered else "recovery not established"
    lines = ["# H100 recovery verification", "", f"Result: **{verdict}**.", "", f"Case: `{spec['key']}`.", ""]
    rows = measurement_rows(measurements, {"good": spec["good_sha"], "bad": spec["bad_sha"]}, repo)
    lines.extend(row.replace("| good |", "| before |", 1).replace("| bad |", "| after |", 1) for row in rows)
    if paired:
        gain = 100 * (after["tflops"] / before["tflops"] - 1)
        lines.extend(["", f"Measured throughput change: **{gain:+.1f}%**.", "", "Both wheels ran on the same GPU."])
    lines.extend([
        "", f"[Recovery bundle PR #4639](https://github.com/{repo}/pull/4639). "
        "This comparison does not isolate an individual upstream patch.", "",
        f"[Issue #4887](https://github.com/{repo}/issues/4887). "
        "The `recovery-measurements` artifact contains both measurements."
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--recovery-baseline", type=float)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton"))
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    measurements = json.loads(args.measurements.read_text())
    if args.state:
        text = bisect_summary(prepared, json.loads(args.state.read_text()), measurements, args.repo)
    elif args.recovery_baseline:
        text = recovery_summary(prepared["spec"], measurements, args.repo, args.recovery_baseline)
    else:
        parser.error("--state or --recovery-baseline is required")
    print(text)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(text)


if __name__ == "__main__":
    main()
