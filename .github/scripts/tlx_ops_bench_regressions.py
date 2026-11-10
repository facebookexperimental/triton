#!/usr/bin/env python3
"""Detect measured nightly throughput regressions; --file-issues enables GitHub writes."""
import argparse
import collections
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import statistics

from failure_issues import gh
from tlx_ops_bench_dashboard import read_summaries

BASELINE_SAMPLES = 3
MAX_AGE_DAYS = 14
CASE_RATIO = 0.50
COHORT_RATIO = 0.90
MIN_COHORT_CASES = 3
CV_NOTE = re.compile(r"\bCV\b.*\bover\b.*\blimit\b")


def day(date):
    return dt.datetime.strptime(date, "%Y%m%d").date()


def valid(case):
    value = case.get("tlx_tflops")
    return case["status"] in ("ok", "pip", "noisy") and isinstance(value,
                                                                   (int, float)) and math.isfinite(value) and value > 0


def reliable(case):
    return valid(case) and case["status"] != "noisy" and not any(CV_NOTE.search(n) for n in case.get("notes", []))


def regime(night):
    timing = night.get("timing") or {}
    return timing.get("latency_mode"), timing.get("replicates")


def comparison(case, history, night, trusted):
    now = day(night["date"])
    baseline = [
        h for h in history if h["regime"] == regime(night) and 0 < (now - day(h["date"])).days <= MAX_AGE_DAYS and (
            not trusted or h["reliable"])
    ][-BASELINE_SAMPLES:]
    if len(baseline) < BASELINE_SAMPLES:
        return None
    value = statistics.median(h["case"]["tlx_tflops"] for h in baseline)
    good = next(h for h in reversed(baseline) if h["case"]["tlx_tflops"] >= value)
    return {
        "key": case["key"], "params": case["params"], "dtype": case["dtype"], "baseline": value, "current":
        case["tlx_tflops"], "ratio": case["tlx_tflops"] / value, "baseline_dates": [h["date"] for h in baseline],
        "baseline_sha": baseline[0]["sha"], "good_sha": good["sha"], "good_date": good["date"], "reference_before":
        baseline[-1]["case"].get("ref_tflops"), "reference_now": case.get("ref_tflops"), "previous_environment":
        baseline[-1]["environment"]
    }


def detect(nights):
    history = collections.defaultdict(list)
    previous_cohort = {}
    previous_case = {}
    alerts = []
    for night in sorted(nights, key=lambda n: n["date"]):
        for op, result in sorted(night["ops"].items()):
            cohorts = collections.defaultdict(list)
            severe = collections.defaultdict(list)
            noisy = collections.defaultdict(list)
            for case in result["cases"]:
                if not valid(case):
                    continue
                identity = op, case["direction"], case["key"]
                trusted = reliable(case)
                broad = comparison(case, history[identity], night, False)
                clean = comparison(case, history[identity], night, True) if trusted else None
                direction = case["direction"]
                if clean:
                    cohorts[direction].append(clean)
                    if clean["ratio"] <= CASE_RATIO:
                        severe[direction].append(clean)
                prev = previous_case.get(identity)
                if broad and not trusted and broad["ratio"] <= CASE_RATIO and prev:
                    if prev["regime"] == regime(night) and (day(night["date"]) - day(
                            prev["date"])).days <= MAX_AGE_DAYS and prev["row"]["ratio"] <= CASE_RATIO:
                        noisy[direction].append(broad)
                if broad:
                    previous_case[identity] = {"date": night["date"], "regime": regime(night), "row": broad}
                else:
                    previous_case.pop(identity, None)
                history[identity].append({
                    "date": night["date"], "sha": night["sha"], "regime": regime(night), "reliable": trusted, "case":
                    case, "environment":
                    {k: night.get(k)
                     for k in ("gpu", "driver", "torch", "runtime", "governed", "timing")}
                })
            for direction in sorted(cohorts.keys() | severe.keys() | noisy.keys()):
                rows = cohorts[direction]
                median = statistics.median(r["ratio"] for r in rows) if rows else None
                cohort = op, direction
                prev = previous_cohort.get(cohort)
                previous = {r["key"]: r for r in prev["rows"]} if prev else {}
                common = [r for r in rows if r["key"] in previous]
                confirmed = (len(common) >= MIN_COHORT_CASES and statistics.median(r["ratio"]
                                                                                   for r in common) <= COHORT_RATIO
                             and statistics.median(previous[r["key"]]["ratio"]
                                                   for r in common) <= COHORT_RATIO and prev["regime"] == regime(night)
                             and (day(night["date"]) - day(prev["date"])).days <= MAX_AGE_DAYS)
                reasons = []
                affected = {}
                if severe[direction]:
                    reasons.append("at least 50% throughput loss in a reliable case")
                    affected.update((r["key"], r) for r in severe[direction])
                if confirmed:
                    reasons.append("at least 10% median throughput loss on two evaluated nights")
                    affected.update((r["key"], r) for r in common if r["ratio"] <= COHORT_RATIO)
                if noisy[direction]:
                    reasons.append("at least 50% throughput loss on two measurements, latest flagged noisy")
                    affected.update((r["key"], r) for r in noisy[direction])
                if reasons:
                    alerts.append({
                        "platform": night["platform"], "op": op, "direction": direction, "date": night["date"], "sha":
                        night["sha"], "reasons": reasons, "median_ratio": median, "compared": len(rows),
                        "reported_cases": sum(c["direction"] == direction
                                              for c in result["cases"]), "space": result.get("space"), "required_ratio":
                        CASE_RATIO if severe[direction] or noisy[direction] else COHORT_RATIO, "confirmation_date":
                        prev["date"] if confirmed else None, "environment":
                        {k: night.get(k)
                         for k in ("gpu", "driver", "torch", "runtime", "governed", "timing")
                         }, "cases": sorted(affected.values(), key=lambda r: r["ratio"])
                    })
                if len(rows) >= MIN_COHORT_CASES:
                    previous_cohort[cohort] = {"date": night["date"], "regime": regime(night), "rows": rows}
    return alerts


def issue_title(alert):
    return f"[perf-regression] {alert['platform']} / {alert['op']} / {alert['direction']}"


def bisect_spec(alert):
    case = min(alert["cases"], key=lambda c: c["ratio"])
    return {
        "op": alert["op"], "direction": alert["direction"], "key": case["key"], "platform": alert["platform"],
        "good_sha": case["good_sha"], "bad_sha": alert["sha"], "required_ratio": alert["required_ratio"], "space":
        alert["space"]
    }


def issue_body(alert, repo, run_url):
    date = alert["date"]
    base = f"https://github.com/{repo}/releases/download/tlx-ops-bench"
    dates = sorted({d for c in alert["cases"] for d in c["baseline_dates"]})
    losses = [100 * (1 - case["ratio"]) for case in alert["cases"]]
    loss = f"{min(losses):.1f}% to {max(losses):.1f}%" if len(losses) > 1 else f"{losses[0]:.1f}%"
    direction = {"fwd": "forward", "bwd": "backward"}.get(alert["direction"], alert["direction"])
    lines = [
        "## Regression", "",
        f"{alert['platform'].upper()} `{alert['op']}` {direction} throughput fell {loss} across {len(losses)} cases on {day(date).isoformat()}.",
        "", "Detection: " + "; ".join(alert["reasons"]) + ".",
        f"Compared {alert['compared']} reliable cases out of {alert['reported_cases']} reported. The cause is not established.",
        "", "## Affected cases", "", "Throughput is in TFLOP/s.", "",
        "| Input | TLX baseline | TLX now | Loss | Reference before | Reference now |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    fmt = lambda v: f"{v:.2f}" if v is not None else "not available"
    labels = [" · ".join([c["dtype"], *(f"{k}={v}" for k, v in c["params"].items())]) for c in alert["cases"]]
    counts = collections.Counter(labels)
    for c, inputs in zip(alert["cases"][:50], labels[:50]):
        if counts[inputs] > 1:
            inputs += f" · case={c['key'].split('/', 3)[-1]}"
        inputs = inputs.replace("|", "\\|").replace("`", "")
        lines.append(
            f"| {inputs} | {c['baseline']:.2f} | {c['current']:.2f} | {(1 - c['ratio']) * 100:.1f}% | {fmt(c['reference_before'])} | {fmt(c['reference_now'])} |"
        )
    if len(alert["cases"]) > 50:
        lines.append(f"\n{len(alert['cases']) - 50} additional affected cases are in the source summary.")
    if alert["confirmation_date"]:
        lines.append(f"\nThe operation/direction also crossed the median threshold on {alert['confirmation_date']}.")
    lines.extend(["", "## Environment changes", ""])
    before = alert["cases"][0]["previous_environment"]

    def environment_value(value):
        if isinstance(value, list):
            return ", ".join(map(str, value)) or "none"
        if isinstance(value, dict):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        return str(value) if value is not None else "not recorded"

    changes = []
    for key, value in alert["environment"].items():
        previous = before.get(key)
        if previous == value:
            continue
        if isinstance(previous, list) and isinstance(value, list):
            removed = [x for x in previous if x not in value]
            added = [x for x in value if x not in previous]
            if removed:
                changes.append(f"- {key}: removed {environment_value(removed)}.")
            if added:
                changes.append(f"- {key}: added {environment_value(added)}.")
        else:
            changes.append(f"- {key}: before {environment_value(previous)}; now {environment_value(value)}.")
    lines.extend(changes or ["- No recorded environment changes."])
    spec = bisect_spec(alert)
    lines.extend([
        "", f"[Suspect commit range](https://github.com/{repo}/compare/{spec['good_sha']}...{spec['bad_sha']})", "",
        "## Performance bisection", "",
        f"Reproducer: `{spec['key']}`. One bisect job checks the worst affected case for this operation and direction.",
        "The job remeasures both endpoints on the same GPU before bisecting. Missing cases, build errors, incorrect output and unreliable measurements are inconclusive, not performance failures.",
        "",
        "This is a historical throughput regression, separate from the existing absolute 0.9x reference floor. Root cause and recovery require investigation.",
        "", "<details>", "<summary>Measurement evidence</summary>", "",
        f"[Current results]({base}/{date}.{alert['platform']}.json) · [Benchmark run]({run_url}) · [Measured commit](https://github.com/{repo}/commit/{alert['sha']})",
        "", "Baseline results: " + " · ".join(f"[{day(d).isoformat()}]({base}/{d}.{alert['platform']}.json)"
                                              for d in dates), "",
        "The baseline is each case's median from three prior reliable measurements within 14 days, using the same timing mode and replicate count. Errors and missing measurements are excluded. Noisy cases use valid measurements and require two severe observations.",
        "", "</details>"
    ])
    return "\n".join(lines)


def file_issues(alerts, repo, run_url, api=gh):
    if not alerts:
        return []
    open_issues = api("GET", f"repos/{repo}/issues?state=open&per_page=100", paginate=True)
    titles = {i["title"] for i in open_issues if "pull_request" not in i}
    created = []
    for alert in alerts:
        title = issue_title(alert)
        if title in titles:
            print(f"Already tracked: {title}")
            continue
        issue = api(
            "POST", f"repos/{repo}/issues",
            {"title": title, "body": issue_body(alert, repo, run_url), "labels": ["nightly", "perf-regression"]})
        titles.add(title)
        alert["bisect_issue"] = issue["number"]
        alert["bisect"] = bisect_spec(alert)
        created.append(issue["html_url"])
        print(f"Filed {issue['html_url']}")
    return created


def queue_bisects(alerts, repo, run_id, artifact_name, api=gh):
    platforms = sorted({alert["platform"] for alert in alerts if alert.get("bisect_issue")})
    for platform in platforms:
        inputs = {"source_run_id": str(run_id), "artifact_name": artifact_name, "platform": platform}
        api("POST", f"repos/{repo}/actions/workflows/bisect-tlx-ops-regression.yml/dispatches",
            {"ref": "main", "inputs": inputs})
        print(f"Queued performance bisect batch for {platform}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--platform")
    parser.add_argument("--date")
    parser.add_argument("--out")
    parser.add_argument("--file-issues", action="store_true")
    parser.add_argument("--queue-bisects", action="store_true")
    parser.add_argument("--artifact-name")
    args = parser.parse_args()
    if args.queue_bisects:
        if not args.artifact_name:
            parser.error("--artifact-name is required for bisection")
        queue_bisects(json.loads(Path(args.directory).read_text()), os.environ["GITHUB_REPOSITORY"],
                      os.environ["GITHUB_RUN_ID"], args.artifact_name)
        return
    if not (args.platform and args.date and args.out):
        parser.error("--platform, --date and --out are required for detection")
    nights = read_summaries(args.directory).get(args.platform, [])
    if not any(n["date"] == args.date for n in nights):
        raise SystemExit(f"Missing current summary: {args.date}.{args.platform}.json")
    alerts = [a for a in detect(nights) if a["date"] == args.date]
    if args.date != max(n["date"] for n in nights):
        print("Historical dispatch; no current alerts.")
        alerts = []
    Path(args.out).write_text(json.dumps(alerts, indent=2))
    print(f"{args.platform} {args.date}: {len(alerts)} regression cohort(s)")
    if args.file_issues and alerts:
        repo = os.environ["GITHUB_REPOSITORY"]
        run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        try:
            file_issues(alerts, repo, run_url)
        finally:
            Path(args.out).write_text(json.dumps(alerts, indent=2))


if __name__ == "__main__":
    main()
