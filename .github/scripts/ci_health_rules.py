"""Rules behind every number on the /ci/ dashboard.

Pure functions over stored run records (see ci_health.py for the record shape):
which test group a step belongs to, why a run failed, what each test group did in
a run, and page_data(), which builds the data.json the page renders. No I/O.
"""
import collections
import datetime as dt
import re

WORKFLOWS = {"mi350.yml": "MI350", "b200.yml": "B200", "h100.yml": "H100"}
EVENTS = {"pull_request": "PR", "push": "push", "schedule": "nightly"}
#: Finished with a verdict. Running and cancelled runs are never the "latest" check.
DONE = ("success", "failure", "startup_failure")
RUNNING = ("in_progress", "queued", "waiting", "pending", "requested")

#: Step name -> test group.
GROUPS = [
    (r"^Run TLX core tests$", "TLX core"),
    (r"^Run TLX tutorial correctness tests$", "Tutorial correctness"),
    (r"^Run device-agnostic TLX codegen tests$", "TLX codegen"),
    (r"^Run TritonBench", "TritonBench"),
    (r"^Run Gluon tests", "Gluon"),
    (r"^Run runtime unit tests$", "Runtime"),
    (r"^Run host-default TLX L2 perf$", "TLX perf"),
    (r"^Run TorchTLX", "TorchTLX"),
]
GROUP_ORDER = [g for _, g in GROUPS if g != "TorchTLX"]
#: Job name -> job group, for runner time. *-meta-triton-test was renamed *-tritonbench-test in #3943.
JOB_GROUPS = [
    (r"-build", "Build"),
    (r"-tlx-test$", "TLX"),
    (r"-(tritonbench|meta-triton)-test$", "TritonBench"),
    (r"-gluon-test$", "Gluon"),
    (r"-runtime-test$", "Runtime"),
    (r"-torchtlx-test$", "TorchTLX"),
    (r"^(report|reconcile)-", "Reporting"),
]
#: A failed step matching this is an infrastructure failure, not a test result.
INFRA_STEP = re.compile(
    r"set up job|checkout|download prebuilt|install triton|upload wheel|build wheel|setup tritonbench|"
    r"authenticate|azure|aws|restore ccache|tune nvidia|fail if the build job|select kernel-affected", re.I)
CAUSES = ("test", "infra", "perf", "config", "unknown")


def group_of(step):
    for pat, g in GROUPS:
        if re.search(pat, step):
            return g
    return None


def job_group(job):
    for pat, g in JOB_GROUPS:
        if re.search(pat, job):
            return g
    return "Other"


def t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def mins(a, b):
    return (t(b) - t(a)).total_seconds() / 60 if a and b else 0.0


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def week(ts):
    d = t(ts).date()
    return str(d - dt.timedelta(days=d.weekday()))


def gpu_job(j):
    return bool(j.get("labels")) and not any("ubuntu" in label for label in j["labels"])


def classify(run):
    """Why a failed run failed: config > test > infra > perf; anything else is "unknown"."""
    if run["conclusion"] == "startup_failure":
        return "config", []
    failed = [(j["name"], [s["name"]
                           for s in j["steps"]
                           if s["conclusion"] == "failure"])
              for j in run["jobs"] or []
              if j["conclusion"] == "failure"]
    if not failed:  # the run failed but no job did: its jobs were never created
        return "config", []
    groups = {group_of(s) for _, fs in failed for s in fs}
    if groups - {None, "TLX perf"}:
        return "test", failed
    if any(not fs for _, fs in failed) or any(INFRA_STEP.search(s) for _, fs in failed for s in fs):
        return "infra", failed
    if any(job_group(n) == "Reporting" for n, _ in failed):
        return "infra", failed
    if "TLX perf" in groups:
        return "perf", failed
    return "unknown", failed


def group_results(run):
    """What each test group did in a run: pass, fail, infra, notrun (blocked by an earlier failure),
    na (skipped by its condition), cancelled or running."""
    out = {}
    for j in run["jobs"] or []:
        steps, first_fail = j["steps"], None
        for s in steps:
            g, c = group_of(s["name"]), s["conclusion"]
            if g is None:
                if c == "failure" and first_fail is None:
                    first_fail = s["name"]
                continue
            if c == "success":
                res = "pass"
            elif c == "failure":
                res = "fail"
            elif c == "skipped":
                res = ("infra" if INFRA_STEP.search(first_fail) else "notrun") if first_fail else "na"
            elif c is None:
                res = "running"
            else:
                res = c
            if c == "failure" and first_fail is None:
                first_fail = s["name"]
            out[g] = res
        if j["conclusion"] == "failure" and not any(s["conclusion"] == "failure" for s in steps):
            for s in steps:  # the job died with no failed step: runner lost
                g = group_of(s["name"])
                if g and out.get(g) in (None, "na", "running"):
                    out[g] = "infra"
    return out


def _latest(runs, wf, event):
    done = [r for r in runs if r["wf"] == wf and r["event"] == event and r["conclusion"] in DONE]
    return max(done, key=lambda r: r["created_at"]) if done else None


def page_data(runs, asof, changelog):
    """The data.json the /ci/ page renders, computed from stored run records."""
    now = t(asof)
    runs = [r for r in runs if r["wf"] in WORKFLOWS and r["created_at"] <= asof]
    cause = {r["id"]: classify(r) for r in runs if r["conclusion"] in ("failure", "startup_failure")}
    days = sorted({r["created_at"][:10] for r in runs if r["created_at"] >= iso(now - dt.timedelta(days=30))})
    strip_from = iso(now - dt.timedelta(days=14))
    pr_from = iso(now - dt.timedelta(hours=24))

    # Pass rate and causes per workflow x trigger x day.
    agg = {}
    for r in runs:
        ev = EVENTS.get(r["event"])
        if not ev or r["created_at"][:10] not in days:
            continue
        a = agg.setdefault(r["wf"],
                           {}).setdefault(ev, {}).setdefault(r["created_at"][:10],
                                                             dict.fromkeys(("pass", "fail", "cancelled") + CAUSES, 0))
        if r["conclusion"] == "success":
            a["pass"] += 1
        elif r["conclusion"] == "cancelled":
            a["cancelled"] += 1
        elif r["id"] in cause:
            a["fail"] += 1
            a[cause[r["id"]][0]] += 1

    # Every finished nightly in the last 14 days, by test group.
    strip = {}
    for wf, label in WORKFLOWS.items():
        night = sorted((r for r in runs if r["wf"] == wf and r["event"] == "schedule" and r["created_at"] >= strip_from
                        and r["conclusion"] not in RUNNING and r["status"] == "completed"),
                       key=lambda r: r["created_at"])
        res = [(r, group_results(r)) for r in night]
        groups = [g for g in GROUP_ORDER if any(g in x for _, x in res)]
        strip[label] = {
            g: [{
                "t": r["created_at"][5:16].replace("T", " "), "c": x.get(g)
                or ("cancelled" if r["conclusion"] == "cancelled" else "nodata")
            }
                for r, x in res]
            for g in groups
        }

    pr24, latest_push, latest_nightly, by_purpose = {}, {}, {}, {}
    for wf, label in WORKFLOWS.items():
        recent = [r for r in runs if r["wf"] == wf and r["event"] == "pull_request" and r["created_at"] >= pr_from]
        o = dict.fromkeys(("passed", "failed", "infra", "test", "other", "cancelled", "running"), 0)
        o["last_infra"] = None
        pr = {g: [0, 0] for g in GROUP_ORDER}
        for r in recent:
            c = r["conclusion"]
            if c == "success":
                o["passed"] += 1
            elif c == "cancelled":
                o["cancelled"] += 1
            elif r["id"] in cause:
                o["failed"] += 1
                k = cause[r["id"]][0]
                k = k if k in ("infra", "test") else "other"
                o[k] += 1
                if k == "infra" and (o["last_infra"] is None or r["created_at"] > o["last_infra"]):
                    o["last_infra"] = r["created_at"]
            else:
                o["running"] += 1
            for g, x in group_results(r).items():
                if g in pr and x in ("pass", "fail"):
                    pr[g][0 if x == "pass" else 1] += 1
        pr24[wf] = o

        push, night = _latest(runs, wf, "push"), _latest(runs, wf, "schedule")
        if push:
            latest_push[wf] = {
                "sha": (push.get("head_sha") or "")[:9], "title": push.get("title", ""), "url": push.get("url", ""),
                "c": push["conclusion"], "t": push["created_at"], "failed":
                [f"{n}: {', '.join(fs) or 'no failed step'}" for n, fs in cause.get(push["id"], (None, []))[1]]
            }
        if night:
            latest_nightly[label] = night["created_at"][5:16].replace("T", " ")
        pg, ng = (group_results(push) if push else {}), (group_results(night) if night else {})
        by_purpose[wf] = [{"group": g, "pr": pr[g], "push": pg.get(g), "nightly": ng.get(g)}
                          for g in GROUP_ORDER
                          if sum(pr[g]) or g in pg or g in ng]

    # GPU hours per day by how the run ended; jobs on CPU runners (labels with "ubuntu") are left out.
    gpu_hours = {}
    for wf in WORKFLOWS:
        rows = {d: {"day": d, "success": 0.0, "failed": 0.0, "cancelled": 0.0} for d in days}
        for r in runs:
            c = r["conclusion"]
            k = "success" if c == "success" else "cancelled" if c == "cancelled" else "failed" if c in (
                "failure", "startup_failure") else None
            if r["wf"] != wf or not k or r["created_at"][:10] not in rows:
                continue
            rows[r["created_at"][:10]][k] += sum(
                mins(j["started_at"], j["completed_at"]) for j in r["jobs"] or [] if gpu_job(j)) / 60
        gpu_hours[wf] = [{**v, **{x: round(v[x], 1)
                                  for x in ("success", "failed", "cancelled")}}
                         for v in rows.values()]

    # Runner minutes per successful PR run, by job group, per ISO week (Monday start).
    runner_min = {}
    for wf, label in WORKFLOWS.items():
        by = collections.defaultdict(list)
        for r in runs:
            if r["wf"] == wf and r["event"] == "pull_request" and r["conclusion"] == "success" and r[
                    "created_at"][:10] in days:
                by[week(r["created_at"])].append(r)
        runner_min[label] = []
        for wk in sorted(by):
            acc, tots = collections.Counter(), []
            for r in by[wk]:
                per = collections.Counter()
                for j in r["jobs"] or []:
                    if j["conclusion"] in ("success", "failure") and job_group(j["name"]) != "Reporting":
                        per[job_group(j["name"])] += mins(j["started_at"], j["completed_at"])
                acc.update(per)
                tots.append(sum(per.values()))
            n = len(by[wk])
            tots.sort()
            runner_min[label].append({
                "week": wk, "n": n, "avg": round(sum(tots) / n, 1), "median": round(tots[n // 2], 1), "jobs":
                {k: round(v / n, 1)
                 for k, v in acc.items()}
            })

    return {
        "asof": asof, "days": days, "agg": agg, "strip": strip, "pr24": pr24, "latest_push": latest_push,
        "latest_nightly": latest_nightly, "by_purpose": by_purpose, "gpu_hours": gpu_hours, "runner_min": runner_min,
        "changelog": changelog
    }
