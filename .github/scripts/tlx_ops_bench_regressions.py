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
from tlx_ops_bench_dashboard import commits, read_summaries, suspects

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
    return {
        "key": case["key"], "params": case["params"], "dtype": case["dtype"], "baseline": value, "current":
        case["tlx_tflops"], "ratio": case["tlx_tflops"] / value, "baseline_dates": [h["date"] for h in baseline],
        "baseline_sha": baseline[0]["sha"], "reference_before": baseline[-1]["case"].get("ref_tflops"), "reference_now":
        case.get("ref_tflops"), "previous_environment": baseline[-1]["environment"]
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
                        "reported_cases": sum(c["direction"] == direction for c in result["cases"]),
                        "confirmation_date": prev["date"] if confirmed else None, "environment":
                        {k: night.get(k)
                         for k in ("gpu", "driver", "torch", "runtime", "governed", "timing")
                         }, "cases": sorted(affected.values(), key=lambda r: r["ratio"])
                    })
                if len(rows) >= MIN_COHORT_CASES:
                    previous_cohort[cohort] = {"date": night["date"], "regime": regime(night), "rows": rows}
    return alerts


def issue_title(alert):
    return f"[perf-regression] {alert['platform']} / {alert['op']} / {alert['direction']}"


def add_suspects(alert):
    oldest = min(alert["cases"], key=lambda c: c["baseline_dates"][0])
    sha = oldest["baseline_sha"]
    found = commits(sha, alert["sha"])
    return {
        **alert, "suspect_base": sha, "history_available": found is not None, "suspect_commits":
        suspects(found or [], alert["op"], alert["platform"], None)
    }


def issue_body(alert, repo, run_url):
    if "suspect_commits" not in alert:
        alert = add_suspects(alert)
    date = alert["date"]
    base = f"https://github.com/{repo}/releases/download/tlx-ops-bench"
    dates = sorted({d for c in alert["cases"] for d in c["baseline_dates"]})
    lines = [
        f"Measured TLX throughput regressed on {alert['platform'].upper()} for `{alert['op']}` {alert['direction']} in {date}.",
        "", "\n".join(f"- {reason}." for reason in alert["reasons"]), "",
        f"{alert['compared']} reliable cases compared out of {alert['reported_cases']} reported cases.",
        "Baseline: per-case median of the last three reliable measurements within 14 days, using the same timing mode and replicate count. Errors and missing measurements are excluded. The noisy fallback uses three valid measurements, including noisy samples, and requires two severe observations.",
        "", f"Current summary: {base}/{date}.{alert['platform']}.json", f"Workflow run: {run_url}",
        f"Measured commit: https://github.com/{repo}/commit/{alert['sha']}", "", "Baseline summaries:",
        *[f"- {base}/{d}.{alert['platform']}.json" for d in dates], "",
        "| Case | Baseline TLX TF/s | Current TLX TF/s | Change | Previous reference TF/s | Current reference TF/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    fmt = lambda v: f"{v:.2f}" if v is not None else "not available"
    for c in alert["cases"][:50]:
        key = c["key"].replace("|", "\\|").replace("`", "")
        lines.append(
            f"| `{key}` | {c['baseline']:.2f} | {c['current']:.2f} | {(c['ratio'] - 1) * 100:.1f}% | {fmt(c['reference_before'])} | {fmt(c['reference_now'])} |"
        )
    if len(alert["cases"]) > 50:
        lines.append(f"\n{len(alert['cases']) - 50} additional affected cases are in the source summary.")
    if alert["confirmation_date"]:
        lines.append(f"\nThe operation/direction also crossed the median threshold on {alert['confirmation_date']}.")
    lines.extend(["", "Environment changes, not established causes:"])
    before = alert["cases"][0]["previous_environment"]
    changes = [
        f"- {key}: `{json.dumps(before.get(key))}` -> `{json.dumps(value)}`"
        for key, value in alert["environment"].items()
        if before.get(key) != value
    ]
    lines.extend(changes or ["- No recorded environment changes."])
    sha = alert["suspect_base"]
    lines.extend([
        "", "## Suspect commits", "",
        "Candidates ranked by affected code area, then newest first. This is not confirmed blame. The range starts at the oldest measurement used in the affected cases' median baselines.",
        ""
    ])
    candidates = alert["suspect_commits"]
    if not alert["history_available"]:
        lines.append("Commit history is unavailable; use the comparison link below to investigate.")
    elif not candidates:
        lines.append(
            "No commits in this range touch the operation, its compiler or its benchmark tooling according to file-based ranking."
        )
    else:
        for candidate in candidates[:10]:
            title = candidate["title"].replace("[", "\\[").replace("]", "\\]")
            link = f"https://github.com/{repo}/commit/{candidate['sha']}"
            pr = f" · [PR {candidate['pr']}](https://github.com/{repo}/pull/{candidate['pr']})" if candidate[
                "pr"] else ""
            lines.append(f"- [{candidate['sha']}: {title}]({link}){pr} · {candidate['area']}")
        if len(candidates) > 10:
            lines.append(
                f"\n{len(candidates) - 10} additional ranked candidates are in the regression evidence artifact; the comparison below contains the full commit range."
            )
    lines.extend([
        "", f"Candidate commit range, not confirmed blame: https://github.com/{repo}/compare/{sha}...{alert['sha']}",
        "",
        "This is a historical throughput regression, separate from the existing absolute 0.9x reference floor. Root cause and recovery require investigation."
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
        created.append(issue["html_url"])
        print(f"Filed {issue['html_url']}")
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--file-issues", action="store_true")
    args = parser.parse_args()
    nights = read_summaries(args.directory).get(args.platform, [])
    if not any(n["date"] == args.date for n in nights):
        raise SystemExit(f"Missing current summary: {args.date}.{args.platform}.json")
    alerts = [a for a in detect(nights) if a["date"] == args.date]
    if args.date != max(n["date"] for n in nights):
        print("Historical dispatch; no current alerts.")
        alerts = []
    alerts = [add_suspects(alert) for alert in alerts]
    Path(args.out).write_text(json.dumps(alerts, indent=2))
    print(f"{args.platform} {args.date}: {len(alerts)} regression cohort(s)")
    if args.file_issues and alerts:
        repo = os.environ["GITHUB_REPOSITORY"]
        run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        file_issues(alerts, repo, run_url)


if __name__ == "__main__":
    main()
