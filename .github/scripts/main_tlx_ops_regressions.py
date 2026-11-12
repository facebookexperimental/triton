#!/usr/bin/env python3
"""Compare a main-push benchmark with nightly history and report on relevant PRs."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess

from failure_issues import gh
from main_tlx_ops_bench import Catalog, changed_paths
from tlx_ops_bench_dashboard import read_summaries
from tlx_ops_bench_regressions import BASELINE_SAMPLES, MAX_AGE_DAYS, day, detect_all, regime, valid

ROOT = Path(__file__).resolve().parents[2]


def detect_push(history, current):
    prior = [night for night in history if night["date"] < current["date"]]
    alerts = [a for a in detect_all(prior + [current]) if a["sha"] == current["sha"] and a["date"] == current["date"]]
    compared = collections.Counter()
    for op, result in current["ops"].items():
        for case in result["cases"]:
            if not valid(case):
                continue
            baseline = [
                previous for previous in prior if regime(previous) == regime(current) and 0 <
                (day(current["date"]) - day(previous["date"])).days <= MAX_AGE_DAYS and any(
                    old["key"] == case["key"] and old["direction"] == case["direction"] and valid(old)
                    for old in previous["ops"].get(op, {}).get("cases", []))
            ]
            if len(baseline) >= BASELINE_SAMPLES:
                compared[op] += 1
    return alerts, dict(compared)


def compatible_history(history, current, repo, plan):
    retained = []
    common = [
        "python/test/tlx_benchmark/_harness/", "python/test/tlx_benchmark/conftest.py",
        "python/test/tlx_benchmark/test_ops_perf.py"
    ]
    for night in history:
        if night["date"] >= current["date"]:
            continue
        ancestor = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", night["sha"], current["sha"]], capture_output=True)
        if ancestor.returncode:
            continue
        ops = {}
        for op, result in night["ops"].items():
            modules = [name for name, measured_op in plan["bench_ops"].items() if measured_op == op]
            if not modules:
                continue
            paths = common + [f"python/test/tlx_benchmark/{module}.py" for module in modules]
            paths.extend(path for module in modules for path in plan["instruments"][module])
            changed = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--name-only", night["sha"], current["sha"], "--", *paths], text=True)
            if not changed:
                ops[op] = result
        retained.append({**night, "ops": ops})
    return retained


def push_prs(repo, before, after, catalog, platform, api=gh):
    commits = subprocess.check_output(
        ["git", "-C", str(catalog.repo), "rev-list", "--reverse", before + ".." + after],
        text=True,
    ).splitlines()
    found = {}
    for commit in commits:
        paths = changed_paths(catalog.repo, commit + "^", commit)
        plans = [p for p in catalog.plan(paths) if p["platform"] == platform]
        if not plans:
            continue
        ops = {op for plan in plans for op in plan["ops"]}
        candidates = api("GET", f"repos/{repo}/commits/{commit}/pulls?per_page=100", paginate=True)
        if not candidates:
            subject = subprocess.check_output(
                ["git", "-C", str(catalog.repo), "show", "-s", "--format=%s", commit], text=True).strip()
            match = re.search(r"\(#(\d+)\)\s*$", subject)
            if match:
                candidates = [api("GET", f"repos/{repo}/pulls/{match[1]}")]
        for pr in candidates:
            if not pr.get("merged_at") or pr["base"]["ref"] != "main" or pr.get("merge_commit_sha") not in commits:
                continue
            found.setdefault(pr["number"], set()).update(ops)
    return found


def comment_body(alerts, run_url, sha, marker):
    lines = [
        "## TLX ops performance regression",
        "",
        f"Detected in the [main-push benchmark]({run_url}) at `{sha[:12]}`.",
        "The shared nightly detector found throughput loss against earlier nightly measurements. "
        "This is not a paired before/after experiment and does not establish this PR as the cause.",
        "",
    ]
    for alert in alerts:
        lines.append(f"### {alert['platform'].upper()} {alert['op']} {alert['direction']}")
        lines.append("Affected public ops: " + ", ".join(alert.get("public_ops", [alert["op"]])))
        lines.append("Current environment: `" + json.dumps(alert["environment"], sort_keys=True) + "`")
        lines.append("Baseline environment: `" + json.dumps(alert["cases"][0]["previous_environment"], sort_keys=True) +
                     "`")
        lines.extend("- Benchmark signal: " + reason for reason in alert["reasons"])
        lines.extend([
            "", "| Input | Baseline TFLOP/s | Current TFLOP/s | Loss | Baseline dates |",
            "| --- | ---: | ---: | ---: | --- |"
        ])
        for case in alert["cases"][:50]:
            label = ", ".join(f"{k}={v}" for k, v in case["params"].items()).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {case['dtype']} {label} | {case['baseline']:.2f} | {case['current']:.2f} | "
                         f"{100 * (1 - case['ratio']):.1f}% | {', '.join(case['baseline_dates'])} |")
        lines.append("")
    lines.append(marker)
    return "\n".join(lines)


def case_op(alert, case):
    if alert["op"] != "norm_fwd":
        return alert["op"]
    parts = case["params"].get("input", "").split(":")
    if len(parts) < 4:
        return None
    family, norm, _, variant = parts[:4]
    direction = alert["direction"]
    if family == "plain":
        return "norm_" + direction
    if family == "softmax":
        return "softmax_" + direction
    if family == "mxfp8":
        return ("layernorm" if norm == "layer" else "rmsnorm") + "_mxfp8_quantize"
    if family == "saved":
        return ("layernorm" if norm == "layer" else "rmsnorm") + "_backward" + ("_from_x_norm"
                                                                                if variant != "raw" else "")
    if family == "dx_quant":
        suffix = "_from_output_recompute_q_precomputed" if variant == "from_output" else "_bf16_precomputed" if variant == "bf16" else "_precomputed"
        return "rmsnorm_backward_quantize_dx_mxfp8" + suffix
    if family == "mxfp4":
        return ("layernorm" if norm == "layer" else "rmsnorm") + "_mxfp4"
    return {
        "preserving": "layernorm_mxfp8_quantize_numerics_preserving",
        "two_pass": "rmsnorm_mxfp8_quantize_3232_2pass",
        "autograd": "layernorm_mxfp8_with_backward",
        "gdpa": "layernorm_mxfp8_gdpa_with_backward",
        "gdpa_precomputed": "layernorm_mxfp8_gdpa_precomputed_no_normalized",
        "nvfp4": "layernorm_nvfp4_quantize",
        "stacked": "layernorm_nvfp4_quantize_q0_jagged_transpose",
        "nvfp4_backward": "layernorm_nvfp4_backward_from_x_norm_padded_dy",
    }.get(family)


def comment_regressions(alerts, prs, repo, run_url, sha, api=gh):
    posted = []
    for number, ops in sorted(prs.items()):
        relevant = []
        for alert in alerts:
            cases = [case for case in alert["cases"] if case_op(alert, case) in ops]
            if cases:
                relevant.append(
                    {**alert, "cases": cases, "public_ops": sorted({case_op(alert, case)
                                                                    for case in cases})})
        if not relevant:
            continue
        platform = relevant[0]["platform"]
        marker = f"<!-- tlx-main-perf:{platform}:{sha} -->"
        comments = api("GET", f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True)
        if any(marker in (comment.get("body") or "") for comment in comments):
            continue
        comment = api("POST", f"repos/{repo}/issues/{number}/comments", {
            "body": comment_body(relevant, run_url, sha, marker),
        })
        posted.append(comment["html_url"])
    return posted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("history", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comment", action="store_true")
    args = parser.parse_args()
    current = json.loads(args.summary.read_text())
    history = read_summaries(args.history).get(current["platform"], [])
    plan = json.loads(os.environ["BENCH_PLAN"])
    history = compatible_history(history, current, ROOT, plan)
    alerts, compared = detect_push(history, current)
    result = {"alerts": alerts, "compared_cases": compared, "comments": []}
    args.out.write_text(json.dumps(result, indent=2))
    if args.comment:
        if os.environ["GITHUB_EVENT_NAME"] != "push" or os.environ["GITHUB_REF"] != "refs/heads/main":
            raise SystemExit("PR comments require a main push")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if current["sha"] != event["after"]:
            raise SystemExit("Benchmark source does not match the main push")
        if alerts:
            repo = os.environ["GITHUB_REPOSITORY"]
            catalog = Catalog(ROOT)
            prs = push_prs(repo, event["before"], event["after"], catalog, current["platform"])
            run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            result["comments"] = comment_regressions(alerts, prs, repo, run_url, current["sha"])
            args.out.write_text(json.dumps(result, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    text = (f"{current['platform']}: {len(alerts)} regression cohort(s), "
            f"{sum(compared.values())} cases with a recent historical baseline. "
            "Missing or incompatible historical cases are not passes.\n")
    print(text)
    if summary:
        with open(summary, "a") as output:
            output.write(text)


if __name__ == "__main__":
    main()
