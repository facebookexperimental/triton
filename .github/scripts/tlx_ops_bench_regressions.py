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
from urllib.parse import urlencode

from failure_issues import gh
from tlx_ops_bench_dashboard import read_summaries

BASELINE_SAMPLES = 3
MAX_AGE_DAYS = 14
CASE_RATIO = 0.70
NOISY_CASE_RATIO = 0.50
COHORT_RATIO = 0.90
MIN_COHORT_CASES = 3
BEST_RATIO = 0.85
RECOVERY_RATIO = 0.90
GATE_SAMPLES = 10
SWING = 1.15
BEST_REASON = "at least 15% throughput loss against the historical best on two reliable measurements"
SIGNALS = {"recent": "Recent baseline", "historical-best": "Historical best"}
MARKER = re.compile(r"<!-- tlx-perf-episode: (\S+) -->")
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
        "key":
        case["key"], "params":
        case["params"], "dtype":
        case["dtype"], "baseline":
        value, "current":
        case["tlx_tflops"], "ratio":
        case["tlx_tflops"] / value, "baseline_stable":
        max(h["case"]["tlx_tflops"] for h in baseline) <= SWING * min(h["case"]["tlx_tflops"] for h in baseline),
        "baseline_dates": [h["date"] for h in baseline], "baseline_sha":
        baseline[0]["sha"], "good_sha":
        good["sha"], "good_date":
        good["date"], "reference_before":
        baseline[-1]["case"].get("ref_tflops"), "reference_now":
        case.get("ref_tflops"), "previous_environment":
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
                prev = previous_case.get(identity)
                if broad and broad["ratio"] <= CASE_RATIO:
                    carried = prev and prev["regime"] == regime(night) and (day(night["date"]) - day(
                        prev["date"])).days <= MAX_AGE_DAYS and prev["row"]["ratio"] <= CASE_RATIO
                    broad["recent_onset"] = prev["row"]["recent_onset"] if carried else night["date"]
                if clean:
                    cohorts[direction].append(clean)
                    if clean["ratio"] <= CASE_RATIO and (clean["ratio"] <= NOISY_CASE_RATIO
                                                         or clean["baseline_stable"]):
                        severe[direction].append(clean)
                if broad and not trusted and broad["ratio"] <= NOISY_CASE_RATIO and prev:
                    if prev["regime"] == regime(night) and (day(night["date"]) - day(
                            prev["date"])).days <= MAX_AGE_DAYS and prev["row"]["ratio"] <= NOISY_CASE_RATIO:
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
                    reasons.append(
                        "at least 30% throughput loss in a reliable case; losses below 50% require a stable baseline")
                    affected.update((r["key"], r) for r in severe[direction])
                if confirmed:
                    reasons.append("at least 10% median throughput loss on two evaluated nights")
                    affected.update((r["key"], r) for r in common if r["ratio"] <= COHORT_RATIO)
                if noisy[direction]:
                    reasons.append("at least 50% throughput loss on two measurements, latest flagged noisy")
                    affected.update((r["key"], r) for r in noisy[direction])
                if reasons:
                    alerts.append({
                        "platform":
                        night["platform"], "op":
                        op, "direction":
                        direction, "date":
                        night["date"], "sha":
                        night["sha"], "reasons":
                        reasons, "median_ratio":
                        median, "compared":
                        len(rows), "reported_cases":
                        sum(c["direction"] == direction for c in result["cases"]), "space":
                        result.get("space"), "required_ratio":
                        CASE_RATIO if severe[direction] else NOISY_CASE_RATIO if noisy[direction] else COHORT_RATIO,
                        "confirmation_date":
                        prev["date"] if confirmed else None, "environment":
                        {k: night.get(k)
                         for k in ("gpu", "driver", "torch", "runtime", "governed", "timing")}, "cases":
                        sorted(affected.values(), key=lambda r: r["ratio"])
                    })
                if len(rows) >= MIN_COHORT_CASES:
                    previous_cohort[cohort] = {"date": night["date"], "regime": regime(night), "rows": rows}
    return alerts


def environment(night):
    return {k: night.get(k) for k in ("gpu", "driver", "torch", "runtime", "governed", "timing")}


def reverses(values):
    """True if two consecutive changes larger than SWING go in opposite directions."""
    moves = [b > a for a, b in zip(values, values[1:]) if max(a, b) / min(a, b) > SWING]
    return any(a != b for a, b in zip(moves, moves[1:]))


def track_best(nights):
    """Compare every reliable case with its historical best in the same timing regime.

    The best is the largest median of three consecutive reliable samples spanning at most MAX_AGE_DAYS, taken
    before tonight and kept indefinitely. Missing nights and unreliable rows are skipped without resetting it.
    Returns {(op, direction, key): {date: comparison}}.
    """
    valid_values = collections.defaultdict(list)
    valid_history = collections.defaultdict(list)
    samples = collections.defaultdict(list)
    best = {}
    onsets = {}
    last_mode = {}
    out = collections.defaultdict(dict)
    for night in sorted(nights, key=lambda n: n["date"]):
        mode = regime(night)
        for op, result in sorted(night["ops"].items()):
            for case in result["cases"]:
                if not valid(case):
                    continue
                identity = op, case["direction"], case["key"]
                prior_values = valid_values[(identity, mode)][-GATE_SAMPLES:]
                valid_values[(identity, mode)].append(case["tlx_tflops"])
                peak = best.get((identity, mode))
                if last_mode.get(identity) != mode:
                    onsets.pop((identity, mode), None)
                    valid_history[(identity, mode)].clear()
                last_mode[identity] = mode
                valid_history[(identity, mode)].append((night["date"], case["tlx_tflops"], reliable(case)))
                if peak:
                    if case["tlx_tflops"] <= RECOVERY_RATIO * peak["value"]:
                        onsets.setdefault((identity, mode), night["date"])
                    elif reliable(case):
                        onsets.pop((identity, mode), None)
                if not reliable(case):
                    continue
                same = [s for s in samples[identity] if s["regime"] == mode]
                current = {
                    "date": night["date"], "sha": night["sha"], "regime": mode, "case": case, "environment":
                    environment(night)
                }
                if peak:
                    ratio = case["tlx_tflops"] / peak["value"]
                    prev = same[-1] if same and "ratio" in same[-1] else None
                    low = ratio <= BEST_RATIO
                    onset = onsets.get((identity, mode))
                    continuing = low and prev and prev["ratio"] <= BEST_RATIO and prev["onset"] == onset
                    stable = prev["stable"] if continuing else not (low and reverses(prior_values))
                    good = next(
                        (s for s in reversed(same)
                         if onset and s["date"] < onset and s["case"]["tlx_tflops"] >= RECOVERY_RATIO * peak["value"]),
                        peak["top"])
                    current.update(ratio=ratio, onset=onset, stable=stable)
                    out[identity][night["date"]] = {
                        "best": peak["value"], "best_dates": peak["dates"], "best_sha": peak["end"]["sha"], "ratio":
                        ratio, "previous_ratio": prev["ratio"] if prev else None, "onset": onset, "deficit":
                        bool(continuing) and stable, "good": good
                    }
                window = (same + [current])[-BASELINE_SAMPLES:]
                span = (day(night["date"]) - day(window[0]["date"])).days
                if len(window) == BASELINE_SAMPLES and span <= MAX_AGE_DAYS:
                    value = statistics.median(s["case"]["tlx_tflops"] for s in window)
                    if not peak or value > peak["value"]:
                        if not peak:
                            for date, observed, trusted in reversed(valid_history[(identity, mode)]):
                                if trusted and observed > RECOVERY_RATIO * value:
                                    break
                                if observed <= RECOVERY_RATIO * value:
                                    onsets[(identity, mode)] = date
                        best[(identity, mode)] = {
                            "value": value, "dates": [s["date"] for s in window], "end": current, "top":
                            max(window, key=lambda s: s["case"]["tlx_tflops"])
                        }
                samples[identity].append(current)
    return out


def detect_best(nights, tracked=None):
    """Alert when a stable case stays at or below BEST_RATIO of its historical best on two reliable nights."""
    tracked = track_best(nights) if tracked is None else tracked
    alerts = []
    for night in sorted(nights, key=lambda n: n["date"]):
        for op, result in sorted(night["ops"].items()):
            compared = collections.defaultdict(list)
            for case in result["cases"]:
                found = tracked.get((op, case["direction"], case["key"]), {}).get(night["date"])
                if found and valid(case) and reliable(case):
                    compared[case["direction"]].append((case, found))
            for direction, rows in sorted(compared.items()):
                low = [(c, f) for c, f in rows if f["deficit"]]
                if not low:
                    continue
                cases = [{
                    "key": c["key"], "params": c["params"], "dtype": c["dtype"], "baseline": f["best"], "current":
                    c["tlx_tflops"], "ratio": f["ratio"], "previous_ratio": f["previous_ratio"], "baseline_dates":
                    f["best_dates"], "baseline_sha": f["best_sha"], "good_sha": f["good"]["sha"], "good_date":
                    f["good"]["date"], "reference_before": f["good"]["case"].get("ref_tflops"), "reference_now":
                    c.get("ref_tflops"), "previous_environment": f["good"]["environment"], "detector":
                    "historical-best", "onset": f["onset"], "required_ratio":
                    BEST_RATIO * f["best"] / f["good"]["case"]["tlx_tflops"]
                } for c, f in low]
                median = statistics.median(f["ratio"] for _, f in rows)
                alerts.append({
                    "platform": night["platform"], "op": op, "direction": direction, "date": night["date"], "sha":
                    night["sha"], "reasons": [BEST_REASON], "median_ratio": median, "compared": len(rows),
                    "reported_cases": sum(c["direction"] == direction for c in result["cases"]), "space":
                    result.get("space"), "required_ratio": BEST_RATIO, "confirmation_date": None, "environment":
                    environment(night), "cases": sorted(cases, key=lambda r: r["ratio"])
                })
    return alerts


def episode(alert, case):
    """Hidden marker entry shared by both detectors: platform, case, timing regime and loss onset."""
    mode, replicates = regime({"timing": alert["environment"].get("timing")})
    onset = case.get("onset") or alert.get("confirmation_date") or alert["date"]
    return "|".join([alert["platform"], alert["op"], alert["direction"], case["key"], f"{mode},{replicates}", onset])


def detect_all(nights):
    """Recent-baseline and historical-best alerts merged per platform, op, direction and night."""
    tracked = track_best(nights)
    merged = {}
    for detector, alerts in (("recent", detect(nights)), ("historical-best", detect_best(nights, tracked))):
        for alert in alerts:
            for case in alert["cases"]:
                case.setdefault("detector", detector)
                if "onset" not in case:
                    found = tracked.get((alert["op"], alert["direction"], case["key"]), {}).get(alert["date"])
                    case["onset"] = (found["onset"] if found else None) or case.get("recent_onset")
            evidence = {k: alert[k] for k in ("reasons", "median_ratio", "compared", "confirmation_date")}
            slot = alert["platform"], alert["op"], alert["direction"], alert["date"]
            if slot not in merged:
                merged[slot] = {**alert, "detectors": [detector], "evidence": {detector: evidence}}
                continue
            target = merged[slot]
            target["detectors"].append(detector)
            target["evidence"][detector] = evidence
            target["reasons"] = target["reasons"] + alert["reasons"]
            target["cases"] = sorted(target["cases"] + alert["cases"], key=lambda r: r["ratio"])
    for alert in merged.values():
        alert["episodes"] = sorted({episode(alert, case) for case in alert["cases"]})
    return sorted(merged.values(), key=lambda a: (a["date"], a["op"], a["direction"]))


def issue_title(alert):
    return f"[perf-regression] {alert['platform']} / {alert['op']} / {alert['direction']}"


def bisect_spec(alert):
    case = min(alert["cases"], key=lambda c: c["ratio"])
    return {
        "op": alert["op"], "direction": alert["direction"], "key": case["key"], "platform": alert["platform"],
        "good_sha": case["good_sha"], "bad_sha": alert["sha"], "required_ratio":
        case.get("required_ratio", alert["required_ratio"]), "space": alert["space"]
    }


def dashboard_url(alert, case=None):
    filters = {"tab": "trending", "platform": alert["platform"], "op": alert["op"], "dir": alert["direction"]}
    if case is not None:
        filters["dtype"] = case["dtype"]
        filters.update({f"in.{k}": v for k, v in case["params"].items()})
    return f"https://facebookexperimental.github.io/triton/bench/#{urlencode(filters)}"


def issue_body(alert, repo, run_url, commit_dates):
    date = alert["date"]
    case = min(alert["cases"], key=lambda c: c["ratio"])
    spec = bisect_spec(alert)

    def summary_link(date):
        url = f"https://github.com/{repo}/releases/download/tlx-ops-bench/{date}.{alert['platform']}.json"
        return f"[{day(date).isoformat()}]({url})"

    dashboard = dashboard_url(alert)
    losses = [100 * (1 - case["ratio"]) for case in alert["cases"]]
    loss = f"{min(losses):.1f}% to {max(losses):.1f}%" if len(losses) > 1 else f"{losses[0]:.1f}%"
    direction = {"fwd": "forward", "bwd": "backward"}.get(alert["direction"], alert["direction"])
    evidence = alert.get("evidence") or {"recent": alert}
    affected = len({c["key"] for c in alert["cases"]})
    lines = [
        "## Detection status", "",
        "Nightly throughput loss detected. These measurements do not establish a code cause or verify recovery.", "",
        f"[Source nightly run]({run_url}) · [Benchmark dashboard]({dashboard})", "", "## Regression", "",
        f"{alert['platform'].upper()} `{alert['op']}` {direction} throughput fell {loss} across {affected} {'case' if affected == 1 else 'cases'} on {day(date).isoformat()}.",
        ""
    ]
    for detector, found in evidence.items():
        lines.append(f"- {SIGNALS[detector]}: " + "; ".join(found["reasons"]) +
                     f". Compared {found['compared']} reliable cases out of {alert['reported_cases']} reported.")
    lines.extend([
        "", "## Affected cases", "",
        "Throughput is in TFLOP/s. Each baseline is a three-night median; historical-best rows use the best median in the same timing mode.",
        "Reference values are context, not the historical regression gate. A missing value means no reference measurement was recorded.",
        "", "| Signal | Input | Baseline nights | TLX baseline | TLX now | Loss | Reference before | Reference now |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    ])
    fmt = lambda v: f"{v:.2f}" if v is not None else "not available"
    shown = sorted(alert["cases"], key=lambda c: (list(SIGNALS).index(c.get("detector", "recent")), c["ratio"]))
    labels = [" · ".join([c["dtype"], *(f"{k}={v}" for k, v in c["params"].items())]) for c in shown]
    counts = collections.Counter(labels)
    for c, inputs in zip(shown[:50], labels[:50]):
        if counts[inputs] > 1:
            inputs += f" · case={c['key'].split('/', 3)[-1]}"
        inputs = inputs.replace("|", "\\|").replace("`", "").replace("[", "\\[").replace("]", "\\]")
        inputs = f"[{inputs}]({dashboard_url(alert, c)})"
        lines.append(
            f"| {SIGNALS[c.get('detector', 'recent')]} | {inputs} | {', '.join(summary_link(d) for d in c['baseline_dates'])} | {c['baseline']:.2f} | {c['current']:.2f} | {(1 - c['ratio']) * 100:.1f}% | {fmt(c['reference_before'])} | {fmt(c['reference_now'])} |"
        )
    if len(alert["cases"]) > 50:
        lines.append(
            f"\n{len(alert['cases']) - 50} additional affected cases can be viewed in the benchmark dashboard.")
    if alert["confirmation_date"]:
        lines.append(
            f"\nThe operation/direction also crossed the median threshold on {day(alert['confirmation_date']).isoformat()}."
        )
    lines.extend(["", "## Measurement context", ""])
    before = case["previous_environment"]

    def environment_value(value):
        if isinstance(value, list):
            return ", ".join(map(str, value)) or "none"
        if isinstance(value, dict):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        return str(value) if value is not None else "not recorded"

    lines.extend([
        "Recorded settings for the representative case's endpoint nights. Matching settings do not prove the same physical GPU or execution history.",
        "", "| Setting | Good benchmark night | Bad benchmark night |", "| --- | --- | --- |"
    ])
    for key in sorted(before.keys() | alert["environment"].keys()):
        lines.append(
            f"| {key} | {environment_value(before.get(key))} | {environment_value(alert['environment'].get(key))} |")
    lines.extend(["", f"Recorded search space: `{alert['space']}`."])
    lines.extend([
        "", "## Performance bisection", "",
        "Only one representative case is selected for bisection. Its result does not establish the cause or recovery of every affected case.",
        "", f"Case: `{spec['key']}`.", "",
        f"[Suspect commit range](https://github.com/{repo}/compare/{spec['good_sha']}...{spec['bad_sha']})", "",
        "| Endpoint | Commit | Commit date (UTC) | Benchmark date |", "| --- | --- | --- | --- |",
        f"| Good | [{spec['good_sha'][:8]}](https://github.com/{repo}/commit/{spec['good_sha']}) | {commit_dates[spec['good_sha']]} | {summary_link(case['good_date'])} |",
        f"| Bad | [{spec['bad_sha'][:8]}](https://github.com/{repo}/commit/{spec['bad_sha']}) | {commit_dates[spec['bad_sha']]} | {summary_link(date)} |",
        "", f"Endpoint validation requires bad/good throughput <= {spec['required_ratio']:.6f} on the same GPU.",
        "Missing cases, build errors, incorrect output and unreliable measurements are inconclusive, not performance failures.",
        "The job posts its result in the issue comments. A first-bad bundle does not identify an individual upstream patch.",
        "Issues on one platform may share a workflow batch; each case has independent endpoints and candidate observations.",
        "", "## One-case reproduction", "",
        "From the repository root, activate the source run's Torch/runtime environment with a separately installed endpoint wheel. The command preserves the bad endpoint's benchmark snapshot and measures only the case above; it does not build or install a wheel.",
        "Repeat with each endpoint wheel. Set `GPU_UUID` to the same UUID for both measurements; if unset, the first measurement selects a GPU and reports its UUID.",
        "An isolated measurement can differ from the full nightly sequence. Failure to reproduce alone is not evidence of recovery.",
        "", "```bash", 'mkdir -p "$HOME/ai_works/triton/perf-repro"',
        'export REPRO_DIR="$(mktemp -d "$HOME/ai_works/triton/perf-repro/run.XXXXXX")"',
        f'git archive {spec["bad_sha"]} python/test/tlx_benchmark | tar -x -C "$REPRO_DIR"',
        'TRITON_HOME="$REPRO_DIR/triton-home" python - <<\'PY\'', "import json, os, sys", "from pathlib import Path",
        'sys.path.insert(0, ".github/scripts")', "from tlx_ops_bench_bisect import bench_modules, measure_case",
        'harness = Path(os.environ["REPRO_DIR"]) / "python/test/tlx_benchmark"',
        f"step = { {k: spec[k] for k in ('op', 'direction', 'key', 'space')}!r}",
        'step.update(source=str(Path.cwd()), harness=str(harness),',
        '            modules=bench_modules(harness).get(step["op"], []), gpu_uuid=os.environ.get("GPU_UUID"))',
        "print(json.dumps(measure_case(step), indent=2))", "PY", "```"
    ])
    if alert.get("episodes"):
        lines.extend(["", *(f"<!-- tlx-perf-episode: {e} -->" for e in alert["episodes"])])
    return "\n".join(lines)


def file_issues(alerts, repo, run_url, api=gh):
    """File one issue per untracked episode. Never comments on, edits or closes existing issues.

    An open issue with the same title, or any open or closed perf-regression issue whose hidden markers share one
    of the alert's case/regime/onset entries, tracks the alert. A partly seen alert is skipped too: its new cases
    belong to an episode a person already holds.
    """
    if not alerts:
        return []
    open_issues = api("GET", f"repos/{repo}/issues?state=open&per_page=100", paginate=True)
    titles = {i["title"] for i in open_issues if "pull_request" not in i}
    seen = None
    created = []
    for alert in alerts:
        title = issue_title(alert)
        if title in titles:
            print(f"Already tracked: {title}")
            continue
        if seen is None:
            labelled = api("GET", f"repos/{repo}/issues?state=all&labels=perf-regression&per_page=100", paginate=True)
            seen = {
                e: i["number"]
                for i in labelled if "pull_request" not in i for e in MARKER.findall(i.get("body") or "")
            }
        known = [e for e in alert.get("episodes", []) if e in seen]
        if known:
            alert["tracked_by"] = sorted({seen[e] for e in known})
            alert["untracked_episodes"] = [e for e in alert["episodes"] if e not in seen]
            print(f"Already tracked by #{', #'.join(map(str, alert['tracked_by']))}: {title}")
            continue
        spec = bisect_spec(alert)
        commit_dates = {
            sha: api("GET", f"repos/{repo}/commits/{sha}")["commit"]["committer"]["date"]
            for sha in (spec["good_sha"], spec["bad_sha"])
        }
        issue = api(
            "POST", f"repos/{repo}/issues", {
                "title": title, "body": issue_body(alert, repo, run_url, commit_dates), "labels":
                ["nightly", "perf-regression"]
            })
        titles.add(title)
        seen.update((e, issue["number"]) for e in alert.get("episodes", []))
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
    alerts = [a for a in detect_all(nights) if a["date"] == args.date]
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
