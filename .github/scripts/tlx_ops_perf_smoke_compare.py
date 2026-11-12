#!/usr/bin/env python3
"""Compare a PR tlx.ops smoke summary against the nearest nightly baseline.

Usage:
    tlx_ops_perf_smoke_compare.py PR_JSON HISTORY_DIR --platform PLATFORM \
        --base-date YYYYMMDD --head SHA --out VERDICT_MD \
        [--min-ratio 0.80] [--alarm-ratio 0.60]

Picks the newest baseline summary in HISTORY_DIR (YYYYMMDD.<platform>.json)
dated on or before --base-date (YYYY-MM-DD is accepted too), joins PR cases to
baseline cases by (key, direction)
on tlx_tflops, and writes a markdown verdict section. Advisory thresholds
(starting point for calibration, deliberately looser than the nightly
detector's CASE_RATIO=0.70 / COHORT_RATIO=0.90 to absorb baseline skew and
runner noise): flag a case below --min-ratio of baseline; INVESTIGATE when 2+
cases flag or any case falls below --alarm-ratio. Always exits 0.
"""

import argparse
import glob
import json
import os
import re
import sys


def date_re(platform):
    return re.compile(r"(\d{8})\." + re.escape(platform) + r"\.json$")


def load_cases(summary):
    rows = {}
    for op, payload in (summary.get("ops") or {}).items():
        for case in payload.get("cases") or []:
            tflops = case.get("tlx_tflops")
            if not isinstance(tflops, (int, float)) or tflops <= 0:
                continue
            rows[(case.get("key"), case.get("direction"))] = {
                "op": op,
                "tflops": float(tflops),
            }
    return rows


def pick_baseline(history_dir, base_date, platform):
    pattern = date_re(platform)
    candidates = []
    for path in glob.glob(os.path.join(history_dir, f"*.{platform}.json")):
        match = pattern.search(os.path.basename(path))
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        return None, None
    candidates.sort()
    on_or_before = [c for c in candidates if c[0] <= base_date]
    if on_or_before:
        return on_or_before[-1]
    oldest = candidates[0]
    return oldest[0], oldest[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_json")
    parser.add_argument("history_dir")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--base-date", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-ratio", type=float, default=0.80)
    parser.add_argument("--alarm-ratio", type=float, default=0.60)
    parser.add_argument("--min-baseline-tflops", type=float, default=1.0,
                        help="ignore baseline cases below this (tiny shapes are noise-dominated)")
    args = parser.parse_args()

    with open(args.pr_json) as handle:
        pr = json.load(handle)
    pr_cases = load_cases(pr)
    base_date = args.base_date.replace("-", "")

    lines = [
        f"### {args.platform}",
        "",
        f"Head `{args.head[:12]}` vs nightly baseline. ",
    ]
    verdict = "PASS"
    baseline_date, baseline_path = pick_baseline(args.history_dir, base_date, args.platform)
    if baseline_path is None:
        verdict = "NO SIGNAL"
        lines.append(f"No nightly baseline found — no signal. (Nightly release has no `*.{args.platform}.json` yet.)")
    elif not pr_cases:
        verdict = "NO SIGNAL"
        lines.append("No PR bench cases produced — no signal. (Check the smoke job log; the bench may have failed.)")
    else:
        with open(baseline_path) as handle:
            base_cases = load_cases(json.load(handle))
        stale_note = ""
        if baseline_date > base_date:
            stale_note = (f" (oldest available; base {base_date} predates baseline history — "
                          "treat ratios with extra skepticism)")
        lines[2] += f"Baseline `{baseline_date}.{args.platform}.json`{stale_note}."
        rows = []
        missing = 0
        tiny = 0
        for key, case in pr_cases.items():
            base = base_cases.get(key)
            if base is None:
                missing += 1
                continue
            if base["tflops"] < args.min_baseline_tflops:
                tiny += 1
                continue
            rows.append({
                "key": key[0],
                "pr": case["tflops"],
                "base": base["tflops"],
                "ratio": case["tflops"] / base["tflops"],
            })
        rows.sort(key=lambda row: row["ratio"])
        flagged = [row for row in rows if row["ratio"] < args.min_ratio]
        alarmed = [row for row in rows if row["ratio"] < args.alarm_ratio]
        if not rows:
            verdict = "NO SIGNAL"
        elif len(flagged) >= 2 or alarmed:
            verdict = "INVESTIGATE"
        lines += [
            "",
            f"Compared {len(rows)} cases ({missing} PR cases have no baseline, "
            f"{tiny} below {args.min_baseline_tflops} baseline TFLOPs). "
            f"Verdict: **{verdict}**.",
            "",
            "| case | PR TFLOPs | base TFLOPs | ratio |",
            "|---|---|---|---|",
        ]
        for row in rows[:10]:
            lines.append(f"| `{row['key']}` | {row['pr']:.1f} | {row['base']:.1f} | {row['ratio']:.3f} |")
        if verdict == "INVESTIGATE":
            lines += [
                "",
                "This is advisory, not a gate failure. Before believing it: re-run the smoke "
                "via `workflow_dispatch` (runner noise is 5–10%), and on a repeated signal run "
                "a same-runner base-vs-head A/B of the smoke subset.",
            ]
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"smoke verdict [{args.platform}]: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
