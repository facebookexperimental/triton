"""Rules behind every number on the /ci/ dashboard.

Pure functions over stored run records (see ci_health.py for the record shape).
page_data() builds the data.json the page renders, one section per tab:

  main     did a push to main break something new? Per test step (GPU, job,
           step), each failing test is new (it passed on the previous push
           that ran the step) or known (it failed there too, and keeps the push
           it first failed at). A skipped or setup-failed step has no result
           and is invisible to the comparison.
  lit      the same new/known question for Compiler LIT Tests, per lit test,
           over compiler.yml's push runs.
  nightly  the nightly release and the bench dashboard's perf events.
  pr       the latest run of each PR in the last 24 h, tagged by whose it is.
  ops      GPU hours, job durations and setup failures.
  bump     is the branch's Meta LLVM pin current, what a newer fork release
           would add, the open bump issues, and the pin's recent changes.
No I/O.
"""
import collections
import datetime as dt
import re

#: The reusable workflows gpu-tests.yml calls, one per GPU. Their step names define the test steps.
GPU_WORKFLOWS = ("mi350.yml", "b200.yml", "h100.yml")
#: The summary job of gpu-tests.yml: fails whenever any suite does, so it is ignored everywhere.
SUMMARY_JOB = "GPU tests"

#: Branches the page tracks. Every tab shows one of them at a time.
TRACKED_BRANCHES = ("main", "release/3.8.x")
#: Meta LLVM bump branches: scratch/meta-llvm-bump/<target with "/" as "-">. Their runs count for their target.
BUMP_PREFIX = "scratch/meta-llvm-bump/"

#: Step name -> test group. A step with a group is a test step; any other step is setup.
GROUPS = [
    (r"^Run TLX (core|unit) tests$", "TLX core"),  # MI350 called it "unit" until its rename
    (r"^Run TLX tutorial correctness tests$", "Tutorial correctness"),
    (r"^Run device-agnostic TLX codegen tests$", "TLX codegen"),
    (r"^Run TritonBench", "TritonBench"),
    (r"^Run Gluon tests", "Gluon"),
    (r"^Run runtime unit tests$", "Runtime"),
    (r"^Run host-default TLX L2 perf$", "TLX perf"),
    (r"^Run TorchTLX", "TorchTLX"),
    (r"^Run tlx\.ops perf suite$", "TLX ops bench"),
]
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
    r"authenticate|azure|aws|restore ccache|tune nvidia|fail if the build job|select kernel-affected|"
    r"set up test environment|install prebuilt triton|resolve tritonbench ref|install tritonbench dependencies", re.I)


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


def gpu_job(j):
    return bool(j.get("labels")) and not any("ubuntu" in label for label in j["labels"])


GPU_PREFIX = {"h100": "H100", "b200": "B200", "mi350": "MI350"}
GPUS = ("H100", "B200", "MI350")
#: Shared jobs of gpu-tests.yml that every GPU job needs: their failure is a build failure, not a setup failure.
BUILD_JOBS = ("build", "select")
#: How many pushes the Main tab's grid and setup counts look back.
PUSH_GRID = 12
SETUP_WINDOW = 10
RAN = ("success", "failure", "cancelled")
#: Triggers the Operations tab's GPU minutes per run is split by, with their labels.
PER_RUN_EVENTS = {"push": "push", "pull_request": "PR"}
#: How many days the Operations tab's daily charts look back.
OPS_DAYS = 30
#: How many hours its hourly charts (the 3-day range) look back.
OPS_HOURS = 72
#: The Operations tab's time ranges in days. Job durations and the slowest tests are computed for each.
OPS_RANGES = (3, 7, 14, 30)
#: How many of the slowest tests to keep per GPU and range.
SLOW_TOP = 10
#: Run outcomes on the Operations tab, in stacking order.
OUTCOMES = ("passed", "test", "build", "setup", "cancelled")
#: A job that ran on no push for this long is retired (disabled or removed), and its failures no longer count.
RETIRE_DAYS = 7

#: Workflows whose runs hold GPU jobs.
#: The TLX ops perf bench, run on each GPU after every nightly release ("b200 / bench").
BENCH_WF = "nightly-tlx-ops-bench.yml"
GPU_RUN_WORKFLOWS = ("gpu-tests.yml", "gpu-tests-nightly.yml", BENCH_WF)
#: Compiler LIT Tests: one job builds Triton and runs lit; it triggers on pushes to main only.
LIT_WF = "compiler.yml"
LIT_JOB = "LIT Tests"


def gpu_of(run, job):
    """The GPU a job ran on, from the calling job's name: "h100 / ..." in gpu-tests.yml, which calls one workflow per
    GPU, or a matrix value as in "tritonbench (h100) / ..." in gpu-tests-nightly.yml."""
    if run.get("wf") not in GPU_RUN_WORKFLOWS:
        return None
    prefix, sep, _ = job["name"].partition(" / ")
    if not sep:
        return None
    m = re.fullmatch(r"[\w-]+ \((\w+)\)", prefix)
    return GPU_PREFIX.get(m.group(1) if m else prefix)


def job_key(name):
    return name.split(" / ", 1)[-1]


def _gpu_jobs(run):
    return [(gpu_of(run, j), j) for j in run.get("jobs") or [] if gpu_of(run, j) and gpu_job(j)]


def gpu_minutes(run):
    """GPU runner time of a run's jobs that ran. A skipped job used no runner, and GitHub can give it a started_at
    after its completed_at."""
    return sum(mins(j["started_at"], j["completed_at"]) for _, j in _gpu_jobs(run) if j["conclusion"] != "skipped")


def step_result(job, step):
    """"pass" or "fail" for a test step that ran to a verdict; None when it has no result (skipped, cancelled,
    not a test step, or a failed step whose log gave no failures)."""
    if group_of(step["name"]) is None:
        return None
    if step["conclusion"] == "success":
        return "pass"
    if step["conclusion"] == "failure" and "failures" in step:
        return "fail"
    return None


def setup_step(job):
    """The step a setup-failed job died in, or None if the job has a test failure or did not fail."""
    if job["conclusion"] != "failure":
        return None
    failed = [s for s in job["steps"] if s["conclusion"] == "failure"]
    if any(group_of(s["name"]) and "failures" in s for s in failed):
        return None
    return failed[0]["name"] if failed else "runner lost (no failed step)"


def _build_failure(run):
    for j in run.get("jobs") or []:
        if j["name"].split(" / ", 1)[0] in BUILD_JOBS and j["conclusion"] == "failure":
            failed = [s["name"] for s in j["steps"] if s["conclusion"] == "failure"]
            return failed[0] if failed else j["name"]
    return None


def run_outcome(run):
    """passed / test / setup / build / cancelled, or None while the run is not finished."""
    if run.get("status") != "completed":
        return None
    if run["conclusion"] == "cancelled":
        return "cancelled"
    if run["conclusion"] == "success":
        return "passed"
    if _build_failure(run):
        return "build"
    for _, j in _gpu_jobs(run):
        if any(group_of(s["name"]) and s["conclusion"] == "failure" for s in j["steps"]):
            return "test"
    return "setup"


def gpu_outcome(run, gpu):
    """run_outcome for one GPU's jobs, or None when that GPU ran nothing. A run where B200 passed and MI350 failed
    passed for B200. A build failure counts for the GPUs that ran jobs or that the select job picked."""
    if run.get("status") != "completed":
        return None
    jobs = [j for g, j in _gpu_jobs(run) if g == gpu and j["conclusion"] != "skipped"]
    if run["conclusion"] == "cancelled":
        return "cancelled" if jobs else None
    if _build_failure(run):
        return "build" if jobs or (run.get("select") or {}).get(gpu.lower()) else None
    if not jobs:
        return None
    if all(j["conclusion"] == "success" for j in jobs):
        return "passed"
    if any(group_of(s["name"]) and s["conclusion"] == "failure" for j in jobs for s in j["steps"]):
        return "test"
    return "setup"


def slow_test_run(run):
    """Runs whose GPU job logs give the slowest tests: main push runs of gpu-tests.yml, which run every test of the
    GPUs they select, so their timings compare."""
    return run.get("wf") == "gpu-tests.yml" and run["event"] == "push" and run.get("branch") == "main"


def tracked_branch(run):
    """The tracked branch a run counts for: a PR's base, a bump branch's target, else the branch it ran on."""
    if run["event"] == "pull_request":
        return run.get("base") or "main"
    branch = run.get("branch") or "main"
    if branch.startswith(BUMP_PREFIX):
        line = branch[len(BUMP_PREFIX):]
        return next((b for b in TRACKED_BRANCHES if b.replace("/", "-") == line), branch)
    return branch


def _pushes(runs, branch):
    return sorted((r for r in runs if r.get("wf") == "gpu-tests.yml" and r["event"] == "push" and (
        r.get("branch") or "main") == branch and r.get("jobs") is not None and r.get("status") == "completed"),
                  key=lambda r: r["created_at"])


def _running(runs, wf, branch, asof):
    return sorted((r for r in runs if r.get("wf") == wf and r["event"] == "push" and (
        r.get("branch") or "main") == branch and r.get("status") != "completed" and r["created_at"] <= asof),
                  key=lambda r: r["created_at"])


def _commit(run):
    return {"sha": run.get("head_sha") or "", "pr": run.get("pr"), "author": run.get("author"), "t": run["created_at"]}


def _history(pushes):
    """Walk pushes oldest first. A unit is (gpu, job, test group), so a renamed step keeps its history. Returns, per
    unit: its current failing tests, the run index of its last result and its latest step name; per run index:
    {gpu: {"new", "known"}} statuses seen there; per run index: {unit: "new" | "known"} for failing units; and per run
    index: {unit: {index of the push each known failure first failed at}}."""
    state, last, names, seen, units = {}, {}, {}, collections.defaultdict(dict), collections.defaultdict(dict)
    blame = collections.defaultdict(lambda: collections.defaultdict(set))
    for i, run in enumerate(pushes):
        for gpu, j in _gpu_jobs(run):
            for s in j["steps"]:
                res = step_result(j, s)
                if res is None:
                    continue
                unit = (gpu, job_key(j["name"]), group_of(s["name"]))
                names[unit] = s["name"]
                prev, cur = state.get(unit, {}), {}
                for f in s.get("failures") or [] if res == "fail" else []:
                    if f["id"] in prev:
                        cur[f["id"]] = {**prev[f["id"]], "streak": prev[f["id"]]["streak"] + 1}
                    else:
                        base = pushes[last[unit]] if unit in last else None
                        cur[f["id"]] = {
                            "id": f["id"], "msg": f["msg"], "first": i, "prev_sha": (base or {}).get("head_sha"),
                            "streak": 1
                        }
                    status = "new" if cur[f["id"]]["streak"] == 1 else "known"
                    if status == "known":
                        blame[i][unit].add(cur[f["id"]]["first"])
                    seen[i].setdefault(gpu, set()).add(status)
                    if units[i].get(unit) != "new":
                        units[i][unit] = status
                state[unit], last[unit] = cur, i
    return state, last, names, seen, units, blame


def _sched_result(runs, sha, unit, test_id):
    gpu, key, group = unit
    found = "none"
    for run in runs:
        if run.get("wf") != "gpu-tests.yml" or run["event"] != "schedule" or run.get("head_sha") != sha:
            continue
        for g, j in _gpu_jobs(run):
            if g != gpu or job_key(j["name"]) != key:
                continue
            for s in j["steps"]:
                if group_of(s["name"]) == group and step_result(j, s):
                    if any(f["id"] == test_id for f in s.get("failures") or []):
                        return "reproduced"
                    found = "not reproduced"
    return found


def main_view(runs, asof, branch="main"):
    """New and known failures over the push runs of one branch. A bump branch's pushes are not the branch's."""
    pushes = [r for r in _pushes(runs, branch) if r["created_at"] <= asof]
    state, last, names, seen, units, blame = _history(pushes)

    def since(idx):
        """The push(es) a known cell's failures started at, newest first."""
        return [{**_commit(pushes[k]), "title": pushes[k].get("title", "")} for k in sorted(idx, reverse=True)]

    # A unit counts only while its job ran on a push in the last RETIRE_DAYS and its test group still exists in
    # the latest push that ran the job.
    retire = iso(t(asof) - dt.timedelta(days=RETIRE_DAYS))
    latest_steps = {}
    for run in pushes:
        for gpu, j in _gpu_jobs(run):
            if j["conclusion"] in RAN and run["created_at"] >= retire:
                latest_steps[(gpu, job_key(j["name"]))] = {group_of(s["name"]) for s in j["steps"]} - {None}
    gpus = {}
    for gpu in GPUS:
        ran = [r for r in pushes if any(g == gpu and j["conclusion"] in RAN for g, j in _gpu_jobs(r))]
        out = {"verdict": "none", "new": [], "known": [], **(_commit(ran[-1]) if ran else {})}
        for unit, fails in state.items():
            if unit[0] != gpu or unit[2] not in latest_steps.get(unit[:2], ()):
                continue
            for f in fails.values():
                first = pushes[f["first"]]
                item = {
                    "gpu": gpu, "job": unit[1], "step": names[unit], "id": f["id"], "msg": f["msg"], "first_sha":
                    first.get("head_sha") or "", "first_pr": first.get("pr"), "first_author": first.get("author"),
                    "first_t": first["created_at"], "first_url": first.get("url"), "prev_sha": f["prev_sha"], "streak":
                    f["streak"]
                }
                if f["streak"] == 1:
                    item["sched"] = _sched_result(runs, item["first_sha"], unit, f["id"])
                    out["new"].append(item)
                else:
                    out["known"].append(item)
        if ran:
            out["verdict"] = "new" if out["new"] else "known" if out["known"] else "passing"
        for k in ("new", "known"):
            out[k].sort(key=lambda x: (x["first_t"], x["job"], x["id"]))
        gpus[gpu] = out

    grid = []
    for i in range(max(0, len(pushes) - PUSH_GRID), len(pushes)):
        run, cells = pushes[i], {}
        broken = _build_failure(run)
        for gpu in GPUS:
            jobs = [j for g, j in _gpu_jobs(run) if g == gpu and j["conclusion"] in RAN]
            st = seen[i].get(gpu, set())
            if broken or not jobs:
                # skip: path selection did not pick this GPU, so nothing it tests changed.
                cells[gpu] = "skip" if not broken and (run.get("select") or {}).get(gpu.lower()) is False else "na"
            elif "new" in st:
                cells[gpu] = "new"
            elif "known" in st:
                cells[gpu] = "known"
            elif any(setup_step(j) for j in jobs):
                cells[gpu] = "setup"
            elif not any(step_result(j, s) for j in jobs for s in j["steps"]):
                cells[gpu] = "na"  # e.g. cancelled before any test step finished: no result, not a pass
            else:
                cells[gpu] = "pass"
        known_since = {
            gpu: since(set().union(*(v
                                     for u, v in blame[i].items()
                                     if u[0] == gpu)))
            for gpu in GPUS
            if cells[gpu] == "known"
        }
        grid.append({
            **_commit(run), "title": run.get("title", ""), "url": run.get("url"), "cells": cells, "since": known_since
        })

    # Push runs still in progress have no result yet; they show as running so a missing cell is never ambiguous.
    grid += [{**_commit(r), "title": r.get("title", ""), "url": r.get("url"), "cells": {g: "running"
                                                                                        for g in GPUS}}
             for r in _running(runs, "gpu-tests.yml", branch, asof)]
    steps = _step_grid(pushes, max(0, len(pushes) - PUSH_GRID), units, blame, since)
    setup = [{
        "sha": r.get("head_sha") or "", "t": r["created_at"], "gpu": g, "job": job_key(j["name"]), "step": setup_step(j)
    } for r in pushes[-SETUP_WINDOW:] for g, j in _gpu_jobs(r) if setup_step(j)]
    build = None
    if pushes and _build_failure(pushes[-1]):
        build = {**_commit(pushes[-1]), "step": _build_failure(pushes[-1])}
    return {"gpus": gpus, "build": build, "pushes": grid, "steps": steps, "setup": setup}


def lit_result(run):
    """(result, failures) of a LIT push run: pass, fail (lit reported failing tests), build (the job failed with no
    lit result, e.g. the build broke), unknown (the job failed but its log could not be read) or na (cancelled or
    skipped)."""
    job = next((j for j in run.get("jobs") or [] if j["name"] == LIT_JOB), None)
    if job is None or job["conclusion"] not in ("success", "failure"):
        return "na", []
    if job["conclusion"] == "failure" and job.get("log_missing"):
        return "unknown", []
    if job["conclusion"] == "success":
        return "pass", []
    fails = [f for s in job["steps"] for f in s.get("failures") or []]
    return ("fail", fails) if fails else ("build", [])


def lit_view(runs, asof, branch="main"):
    """New and known lit failures over the LIT push runs of one branch, or None when it has none. A test is new when
    it fails on a push and passed on the previous push with a lit result; a build failure has no result and does
    not count."""
    pushes = sorted(
        (r for r in runs if r.get("wf") == LIT_WF and r["event"] == "push" and (r.get("branch") or "main") == branch
         and r.get("jobs") is not None and r.get("status") == "completed" and r["created_at"] <= asof),
        key=lambda r: r["created_at"])
    running = _running(runs, LIT_WF, branch, asof)
    if not pushes and not running:
        return None
    state, last, cells, starts = {}, None, [], {}
    for i, run in enumerate(pushes):
        res, fails = lit_result(run)
        if res not in ("pass", "fail"):
            cells.append(res)
            continue
        cur = {}
        for f in fails:
            if f["id"] in state:
                cur[f["id"]] = {**state[f["id"]], "streak": state[f["id"]]["streak"] + 1}
            else:
                first = pushes[i]
                cur[f["id"]] = {
                    "id": f["id"], "msg": f["msg"], "first_sha": first.get("head_sha") or "", "first_title":
                    first.get("title", ""), "first_t": first["created_at"], "first_url": first.get("url"), "prev_sha":
                    pushes[last].get("head_sha") if last is not None else None, "streak": 1
                }
        cells.append("new" if any(f["streak"] == 1 for f in cur.values()) else "known" if cur else "pass")
        starts[i] = [{"sha": sha, "t": t, "title": title} for t, sha, title in sorted(
            {(f["first_t"], f["first_sha"], f["first_title"])
             for f in cur.values()
             if f["streak"] > 1}, reverse=True)]
        state, last = cur, i
    fails = sorted(state.values(), key=lambda f: (f["first_t"], f["id"]))
    new, known = [f for f in fails if f["streak"] == 1], [f for f in fails if f["streak"] > 1]
    out = {
        "verdict":
        "new" if new else "known" if known else "passing" if last is not None else "none", "new":
        new, "known":
        known, "build":
        None, "unknown":
        None, "pushes": [{
            **_commit(r), "title": r.get("title", ""), "url": r.get("url"), "cell": c, "since":
            starts.get(k, []) if c == "known" else []
        } for k, (r, c) in list(enumerate(zip(pushes, cells)))[-PUSH_GRID:]]
    }
    out["pushes"] += [{**_commit(r), "title": r.get("title", ""), "url": r.get("url"), "cell": "running"}
                      for r in running]
    if last is not None:
        out.update(_commit(pushes[last]))
    if cells and cells[-1] in ("build", "unknown"):
        out[cells[-1]] = {**_commit(pushes[-1]), "url": pushes[-1].get("url")}
    return out


#: Worst first, for a cell where a job has more than one step in the same test group.
STEP_RANK = ("new", "known", "setup", "blocked", "pass", "na")
GROUP_ORDER = [g for _, g in GROUPS]


def _step_grid(pushes, start, units, blame, since):
    """Per GPU, one row per test group over the grid's pushes (oldest first): pass / new / known, or why the
    group has no result there: setup (the job failed in setup), blocked (skipped after an earlier step failed),
    na (the job or the step did not run). Groups that never ran in the grid get no row. since maps a known cell's
    column to the push(es) its failures started at."""
    out = {}
    for gpu in GPUS:
        rows, names, starts = {}, {}, {}
        for col, i in enumerate(range(start, len(pushes))):
            run = pushes[i]
            if _build_failure(run):
                continue
            for g, j in _gpu_jobs(run):
                if g != gpu or j["conclusion"] not in RAN:
                    continue
                key, setup, failed_before = job_key(j["name"]), setup_step(j), False
                for s in j["steps"]:
                    failed_before, earlier = failed_before or s["conclusion"] == "failure", failed_before
                    group = group_of(s["name"])
                    if group is None:
                        continue
                    res = step_result(j, s)
                    if res == "fail":
                        cell = units[i].get((gpu, key, group), "known")
                        starts.setdefault(group, {}).setdefault(col, set()).update(blame[i].get((gpu, key, group), ()))
                    else:
                        cell = res or ("setup" if setup else "blocked" if earlier else "na")
                    cells = rows.setdefault(group, ["na"] * (len(pushes) - start))
                    if STEP_RANK.index(cell) < STEP_RANK.index(cells[col]):
                        cells[col] = cell
                    names[group] = s["name"]
        out[gpu] = [{
            "group": g, "step": names[g], "cells": rows[g], "since":
            {col: since(idx)
             for col, idx in starts.get(g, {}).items()
             if rows[g][col] == "known"}
        }
                    for g in GROUP_ORDER
                    if g in rows and any(c != "na" for c in rows[g])]
    return out


def pr_view(runs, asof, main, branch="main"):
    """PR runs, with their failures compared with main: the GPU tests view of the PRs' base branch."""
    since = iso(t(asof) - dt.timedelta(hours=24))
    on_main = {(i["gpu"], i["job"], i["id"]) for g in main["gpus"].values() for i in g["new"] + g["known"]}
    latest = {}
    for run in runs:
        if run.get("wf") != "gpu-tests.yml" or run["event"] != "pull_request" or not (since <= run["created_at"] <=
                                                                                      asof):
            continue
        key = run.get("pr") or ("branch", run.get("branch"))
        if key not in latest or run["created_at"] > latest[key]["created_at"]:
            latest[key] = run
    out = []
    for run in sorted(latest.values(), key=lambda r: r["created_at"], reverse=True):
        jobs = _gpu_jobs(run)
        fails = {(g, job_key(j["name"]), f["id"]) for g, j in jobs for s in j["steps"] for f in s.get("failures") or []}
        outcome = run_outcome(run)
        row = {
            "pr": run.get("pr"), "title": run.get("title", ""), "author": run.get("author"), "url": run.get("url"),
            "branch": run.get("branch"), "t": run["created_at"], "gpus":
            sorted({g
                    for g, j in jobs
                    if j["conclusion"] in RAN}, key=GPUS.index), "n_failed": len(fails), "whose": None, "note": ""
        }
        if outcome is None:
            row["result"] = "running"
        elif fails:
            row["result"] = "tests"
            if fails <= on_main:
                row["whose"], row["note"] = "main", f"all {len(fails)} also fail on {branch}"
            else:
                row["whose"] = "author"
        elif outcome in ("setup", "build"):
            row["result"] = outcome if outcome == "build" else "setup"
            row["whose"] = "author" if outcome == "build" else "ci"
            row["note"] = _build_failure(run) or next((setup_step(j) for _, j in jobs if setup_step(j)), "")
        else:
            row["result"] = outcome
        out.append(row)
    return out


def nightly_view(runs, asof, bench):
    nights = sorted(
        (r for r in runs
         if r.get("wf") == "nightly-fbtriton.yml" and r.get("status") == "completed" and r["created_at"] <= asof),
        key=lambda r: r["created_at"])
    week_ago = iso(t(asof) - dt.timedelta(days=7))

    def job_result(run, name):
        for j in run.get("jobs") or []:
            if j["name"] == name or j["name"].startswith(name + " / "):
                return j["conclusion"]
        return None

    release = None
    if nights:
        n = nights[-1]
        release = {
            "conclusion": n["conclusion"], "build": job_result(n, "build"), "publish": job_result(n, "publish"), "t":
            n["created_at"], "url": n.get("url"), "published_7":
            sum(job_result(r, "publish") == "success" for r in nights if r["created_at"] >= week_ago), "nights_7":
            sum(r["created_at"] >= week_ago for r in nights)
        }
    perf = []
    for platform, events in sorted((bench or {}).items()):
        for night in events[-2:]:  # the latest night is unconfirmed by construction, so keep the one before too
            for op, e in sorted(night.get("perf", {}).items()):
                perf.append({
                    "platform": platform, "date": night["date"], "op": op, "n_slower": e["n_slower"], "n_faster":
                    e["n_faster"], "signature": e.get("signature"), "unconfirmed": e.get("unconfirmed"), "range":
                    night.get("range"), "compare": night.get("compare"), "suspects": e.get("suspects", [])[:3]
                })
    return {
        "release": release, "perf": perf, "gpu": gpu_nightly_view(runs, asof), "bench":
        gpu_nightly_view(runs, asof, BENCH_WF)
    }


def gpu_nightly_view(runs, asof, wf="gpu-tests-nightly.yml"):
    """The latest run of a daily GPU workflow per GPU: result and failing tests; and passes over the last 7 days."""
    nights = sorted((r for r in runs if r.get("wf") == wf and r.get("status") == "completed"
                     and r.get("jobs") is not None and r["created_at"] <= asof), key=lambda r: r["created_at"])
    week_ago = iso(t(asof) - dt.timedelta(days=7))
    out = {}
    for gpu in GPUS:
        ran = [(r, j) for r in nights for g, j in _gpu_jobs(r) if g == gpu and j["conclusion"] in RAN]
        if not ran:
            continue
        run, j = ran[-1]
        fails = [{"id": f["id"], "msg": f["msg"], "step": s["name"]}
                 for s in j["steps"]
                 for f in s.get("failures") or []]
        result = ("passed" if j["conclusion"] == "success" else
                  "cancelled" if j["conclusion"] == "cancelled" else "setup" if setup_step(j) else "test")
        week = [jj for r, jj in ran if r["created_at"] >= week_ago]
        out[gpu] = {
            "job": job_key(j["name"]), "result": result, "setup": setup_step(j), "fails": fails, "sha":
            run.get("head_sha") or "", "t": run["created_at"], "url": run.get("url"), "passed_7":
            sum(jj["conclusion"] == "success" for jj in week), "runs_7": len(week)
        }
    return out


def _pct(xs, p, digits=1):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], digits) if xs else None


def ops_days(now):
    """The Operations tab's daily-chart days, oldest first, ending on now's day."""
    return [str((now - dt.timedelta(days=d)).date()) for d in range(OPS_DAYS - 1, -1, -1)]


def ops_hours(now):
    """The Operations tab's hourly-chart hours, oldest first, ending on now's hour."""
    top = now.replace(minute=0, second=0, microsecond=0)
    return [(top - dt.timedelta(hours=h)).strftime("%Y-%m-%d %H:00") for h in range(OPS_HOURS - 1, -1, -1)]


def _hour(ts):
    return ts[:10] + " " + ts[11:13] + ":00"


def _outcome_days(gpu_runs, days, outcome, minutes):
    """GPU hours and run counts per day by outcome, and GPU minutes per run by trigger, for one outcome function
    (the whole run, or one GPU's jobs) and its GPU minutes."""
    hours = {d: dict.fromkeys(OUTCOMES, 0.0) for d in days}
    counts = {d: dict.fromkeys(OUTCOMES, 0) for d in days}
    per_run = {d: {e: [] for e in PER_RUN_EVENTS} for d in days}
    for r in gpu_runs:
        k, d = outcome(r), r["created_at"][:10]
        if k not in OUTCOMES or d not in hours:
            continue
        h = minutes(r) / 60
        hours[d][k] += h
        # A run that never reached a GPU is counted only when its build failed: the rest ran nothing.
        if h > 0 or k == "build":
            counts[d][k] += 1
        # Only runs whose tests ran to the end: cancelled, setup- and build-failed runs stopped early, so their GPU
        # time understates what a run costs.
        if h > 0 and k in ("passed", "test") and r["event"] in PER_RUN_EVENTS:
            per_run[d][r["event"]].append(h * 60)
    return {
        "gpu_hours": [{"day": d, **{k: round(v, 1)
                                    for k, v in hours[d].items()}}
                      for d in days],
        "runs": [{"day": d, **counts[d]} for d in days],
        "per_run": [{
            "day": d, **{
                PER_RUN_EVENTS[e]: {f"p{q}": _pct(v, q / 100, 0)
                                    for q in (25, 50, 75)} | {"n": len(v)}
                for e, v in per_run[d].items()
            }
        }
                    for d in days],
    }


def _ran(j):
    """A GPU job that held a runner: it started and ran steps (a skipped job has none)."""
    return j["conclusion"] != "skipped" and j.get("steps") and j.get("started_at") and j.get(
        "completed_at") and j["started_at"] < j["completed_at"]


def _occupancy(spans, lo, hi):
    """(peak, time-weighted mean) of how many of the (start, end) spans overlap within [lo, hi)."""
    ev = []
    for a, b in spans:
        a, b = max(a, lo), min(b, hi)
        if a < b:
            ev += [(a, 1), (b, -1)]
    ev.sort()  # an end sorts before a start at the same instant, so back-to-back jobs do not overlap
    cur = peak = 0
    area, prev = 0.0, lo
    for x, d in ev:
        area += cur * (x - prev).total_seconds()
        prev, cur = x, cur + d
        peak = max(peak, cur)
    return peak, round(area / (hi - lo).total_seconds(), 2)


def concurrency_view(runs, asof):
    """GPU jobs running and waiting for a runner at once, per GPU type, per day (OPS_DAYS) and per hour
    (OPS_HOURS): the peak and the time-weighted mean. All tracked branches and triggers count, since they share
    the runners. Jobs of a run are stored once it completes, so the counts end where the oldest recent run without
    jobs began: after that they would read low, and are left out."""
    now = t(asof)
    running, waiting = {g: [] for g in GPUS}, {g: [] for g in GPUS}
    recent = iso(now - dt.timedelta(hours=OPS_HOURS))
    horizon = min([
        t(r["created_at"])
        for r in runs
        if r.get("wf") in GPU_RUN_WORKFLOWS and r.get("jobs") is None and recent <= r["created_at"] <= asof
    ] + [now])
    for r in runs:
        if r.get("wf") not in GPU_RUN_WORKFLOWS or r["created_at"] > asof:
            continue
        for g, j in _gpu_jobs(r):
            if _ran(j):
                running[g].append((t(j["started_at"]), t(j["completed_at"])))
                if j.get("created_at"):
                    waiting[g].append((t(j["created_at"]), t(j["started_at"])))

    def buckets(labels, width):
        out = []
        for label in labels:
            lo = t(label.replace(" ", "T") + (":00Z" if " " in label else "T00:00:00Z"))
            hi = min(lo + width, horizon)
            row = {"day": label}
            for g in GPUS:
                if hi <= lo:
                    row[g] = {"peak": None, "mean": None, "wait_peak": None}
                    continue
                (peak, mean), (wpeak, _) = _occupancy(running[g], lo, hi), _occupancy(waiting[g], lo, hi)
                row[g] = {"peak": peak, "mean": mean, "wait_peak": wpeak}
            out.append(row)
        return out

    return {
        "days": buckets(ops_days(now), dt.timedelta(days=1)), "hours": buckets(ops_hours(now), dt.timedelta(hours=1))
    }


def job_durations(gpu_runs, now, days):
    """Per GPU and job, the duration of jobs that ran to a result over the last `days`, the slowest 10% over the
    `days` before, and the job's slowest step."""
    lo, lo_prev = iso(now - dt.timedelta(days=days)), iso(now - dt.timedelta(days=2 * days))
    durations, prev, step_t = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in gpu_runs:
        for gpu, j in _gpu_jobs(r):
            if j["conclusion"] not in ("success", "failure") or r["created_at"] < lo_prev:
                continue
            name = job_key(j["name"])
            key = (gpu, name.split("-", 1)[-1] if name.split("-", 1)[0] in GPU_PREFIX else name)
            m = mins(j["started_at"], j["completed_at"])
            if r["created_at"] >= lo:
                durations[key].append(m)
                for s in j["steps"]:
                    if s.get("started_at") and s.get("completed_at"):
                        step_t[key][s["name"]].append(mins(s["started_at"], s["completed_at"]))
            else:
                prev[key].append(m)
    jobs = []
    for key in sorted(durations):
        slow = max(step_t[key].items(), key=lambda kv: _pct(kv[1], 0.5), default=(None, []))
        jobs.append({
            "gpu": key[0], "job": key[1], "n": len(durations[key]), "p50": _pct(durations[key], 0.5), "p90":
            _pct(durations[key], 0.9), "p90_prev": _pct(prev[key], 0.9), "slowest_step": slow[0], "slowest_p50":
            _pct(slow[1], 0.5), "slowest_p90": _pct(slow[1], 0.9)
        })
    jobs.sort(key=lambda x: -(x["p90"] or 0))
    return jobs


def _slow_runs(runs, gpu, lo, hi):
    """[{test key: seconds}] for one GPU's main push runs created in [lo, hi] that recorded durations. A test key is
    (step group, test id)."""
    out = []
    for r in runs:
        if not slow_test_run(r) or not lo <= r["created_at"] <= hi:
            continue
        found = {}
        for g, j in _gpu_jobs(r):
            if g == gpu:
                for s in j["steps"]:
                    for e in s.get("slowest") or []:
                        found[(group_of(s["name"]) or s["name"], e["id"])] = e["s"]
        if found:
            out.append(found)
    return out


def slow_tests_view(runs, asof, days):
    """Per GPU, the SLOW_TOP tests with the highest median time over the main push runs of the last `days` that
    recorded durations. Each run lists only its slowest tests (pytest --durations), so n says how often a test was
    among them."""
    lo = iso(t(asof) - dt.timedelta(days=days))
    out = {}
    for gpu in GPUS:
        per_run = _slow_runs(runs, gpu, lo, asof)
        times = collections.defaultdict(list)
        for found in per_run:
            for k, sec in found.items():
                times[k].append(sec)
        rows = [{"step": k[0], "id": k[1], "n": len(v), "p50": _pct(v, 0.5), "max": round(max(v), 1)}
                for k, v in times.items()]
        rows.sort(key=lambda x: -x["p50"])
        out[gpu] = {"runs": len(per_run), "tests": rows[:SLOW_TOP]}
    return out


def ops_view(runs, asof):
    now = t(asof)
    week_ago, two_weeks = iso(now - dt.timedelta(days=7)), iso(now - dt.timedelta(days=14))
    gpu_runs = [r for r in runs if r.get("wf") in GPU_RUN_WORKFLOWS and r["created_at"] <= asof and r.get("jobs")]
    days = ops_days(now)
    daily = _outcome_days(gpu_runs, days, run_outcome, gpu_minutes)
    by_gpu = {
        g:
        _outcome_days(
            gpu_runs, days, lambda r, g=g: gpu_outcome(r, g), lambda r, g=g: sum(
                mins(j["started_at"], j["completed_at"])
                for gg, j in _gpu_jobs(r)
                if gg == g and j["conclusion"] != "skipped"))
        for g in GPUS
    }

    def gpu_h(lo, hi):
        return round(sum(gpu_minutes(r) for r in gpu_runs if lo <= r["created_at"] < hi) / 60, 1)

    recent = [r for r in gpu_runs if r["created_at"] >= week_ago]
    finished = [r for r in recent if run_outcome(r) in ("passed", "test", "setup", "build")]
    setups = [r for r in finished if run_outcome(r) == "setup"]
    queue = queue_view(gpu_runs, now)
    prs = [r for r in recent if r["event"] == "pull_request"]

    steps = collections.defaultdict(lambda: {"n": 0, "triggers": collections.Counter()})
    for r in gpu_runs:
        if r["created_at"] < week_ago:
            continue
        for gpu, j in _gpu_jobs(r):
            name = setup_step(j)
            if name:
                e = steps[(name, gpu)]
                e["n"] += 1
                e["triggers"][r["event"]] += 1
    setup_steps = sorted(({"step": k[0], "gpu": k[1], "n": v["n"], "triggers": dict(v["triggers"])}
                          for k, v in steps.items()), key=lambda x: -x["n"])
    return {
        "tiles": {
            "gpu_h_7d": gpu_h(week_ago, asof + "~"), "gpu_h_prev_7d": gpu_h(two_weeks,
                                                                            week_ago), "setup_n": len(setups), "runs_n":
            len(finished), "setup_rate": round(len(setups) / len(finished), 3) if finished else None, "setup_mi350":
            sum(any(g == "MI350" and setup_step(j)
                    for g, j in _gpu_jobs(r))
                for r in setups), "queue_p50": queue["pr_p50"], "queue_p90": queue["pr_p90"], "pr_n": len(prs),
            "cancelled_pr": sum(r["conclusion"] == "cancelled" for r in prs), "cancelled_gpu_h":
            round(sum(gpu_minutes(r) for r in prs if r["conclusion"] == "cancelled") / 60, 1)
        }, "select": select_view(runs, asof), **daily, "by_gpu": by_gpu, "setup_steps": setup_steps, "queue": queue,
        # Keyed by range in days, as strings for JSON.
        "jobs": {str(n): job_durations(gpu_runs, now, n)
                 for n in OPS_RANGES}, "slow_tests": {str(n): slow_tests_view(runs, asof, n)
                                                      for n in OPS_RANGES}
    }


def queue_view(gpu_runs, now):
    """Queue time (job created to started) of GPU jobs that ran: per GPU pool per day and per hour (OPS_HOURS), and
    per pool over the last 7 days with how often its job was the last to finish on a PR run. That job decides when the
    PR gets its result, so its queue time is what a PR author waits (pr_p50/pr_p90, last 7 days)."""
    days, hours = ops_days(now), ops_hours(now)
    week_ago = iso(now - dt.timedelta(days=7))
    by_day = {d: {g: [] for g in GPUS} for d in days}
    by_hour = {h: {g: [] for g in GPUS} for h in hours}
    week, last, pr_wait = {g: [] for g in GPUS}, collections.Counter(), []
    for r in gpu_runs:
        d = r["created_at"][:10]
        jobs = [(g, j) for g, j in _gpu_jobs(r) if j["conclusion"] in ("success", "failure") and j.get("created_at")]
        for g, j in jobs:
            q = mins(j["created_at"], j["started_at"])
            if d in by_day:
                by_day[d][g].append(q)
            if _hour(j["created_at"]) in by_hour:
                by_hour[_hour(j["created_at"])][g].append(q)
            if r["created_at"] >= week_ago:
                week[g].append(q)
        if jobs and r["event"] == "pull_request" and r["created_at"] >= week_ago and run_outcome(r) != "cancelled":
            g, j = max(jobs, key=lambda gj: gj[1]["completed_at"])
            last[g] += 1
            pr_wait.append(mins(j["created_at"], j["started_at"]))
    n_pr = sum(last.values())
    return {
        "days": [{
            "day": d,
            **{g: {f"p{q}": _pct(v, q / 100)
                   for q in (25, 50, 75)} | {"n": len(v)}
               for g, v in by_day[d].items()}
        }
                 for d in days], "hours":
        [{
            "day": h,
            **{g: {f"p{q}": _pct(v, q / 100)
                   for q in (25, 50, 75)} | {"n": len(v)}
               for g, v in by_hour[h].items()}
        }
         for h in hours], "gpus": [{
             "gpu": g, "n": len(week[g]), "p50": _pct(week[g], 0.5), "p90": _pct(week[g], 0.9), "last_pct":
             round(last[g] / n_pr, 2) if n_pr else None
         } for g in GPUS], "pr_n":
        n_pr, "pr_p50":
        _pct(pr_wait, 0.5), "pr_p90":
        _pct(pr_wait, 0.9)
    }


def select_view(runs, asof):
    """The smart trigger (the select job of gpu-tests.yml) per day over OPS_DAYS: what it chose per trigger, and
    runs where it did not do its job (failed, fell back to every suite, or skipped a suite on a full run)."""
    days = ops_days(t(asof))
    recent = sorted(
        (r for r in runs
         if r.get("wf") == "gpu-tests.yml" and days[0] <= r["created_at"] <= asof and r.get("jobs") is not None),
        key=lambda r: r["created_at"])
    by_day, anomalies = {d: {} for d in days}, []
    for r in recent:
        sel, base = r.get("select"), {
            "event": r["event"], "sha": r.get("head_sha") or "", "t": r["created_at"], "url": r.get("url"), "pr":
            r.get("pr")
        }
        if any(j["name"] == "select" and j["conclusion"] == "failure" for j in r["jobs"]):
            anomalies.append({**base, "kind": "failed", "detail": "select job failed"})
            continue
        if not sel:
            continue
        gpus = [g for g in GPUS if sel.get(g.lower())]
        e = by_day[r["created_at"][:10]].setdefault(
            r["event"], {
                "runs": 0, "gpus": dict.fromkeys("0123", 0), "build_skipped": 0, "fallback": 0, "per_gpu":
                dict.fromkeys(GPUS, 0)
            })
        e["runs"] += 1
        e["gpus"][str(len(gpus))] += 1
        e["build_skipped"] += not sel.get("build")
        e["fallback"] += bool(sel.get("fallback"))
        for g in gpus:
            e["per_gpu"][g] += 1
        if sel.get("fallback"):
            anomalies.append({**base, "kind": "fallback", "detail": "changed paths unavailable, ran every suite"})
        elif r["event"] in ("schedule", "workflow_dispatch") and len(gpus) < len(GPUS):
            missing = ", ".join(g for g in GPUS if g not in gpus)
            anomalies.append({**base, "kind": "partial", "detail": f"full run skipped {missing}"})
    return {"days": [{"day": d, "triggers": by_day[d]} for d in days], "anomalies": anomalies}


def bump_view(line):
    """The Meta LLVM tab of one tracked branch, from its bump.json entry (None before the first snapshot).
    status: current (pins the newest fork release), new (a newer release is not pinned yet) or unknown."""
    if not line or not line.get("newest") or not line.get("pinned"):
        return {"status": "unknown", **(line or {})}
    return {"status": "current" if line["pinned"]["tag"] == line["newest"]["tag"] else "new", **line}


def page_data(runs, asof, bench, bump=None):
    """The data.json the /ci/ page renders, computed from stored run records, bench perf events and the Meta LLVM
    pin snapshot."""
    branches = {}
    concurrency = concurrency_view(runs, asof)
    for b in TRACKED_BRANCHES:
        rs = [r for r in runs if tracked_branch(r) == b]
        main = main_view(rs, asof, b)
        # The bench dashboard's perf events come from main's nightly bench.
        branches[b] = {
            "main": main, "lit": lit_view(rs, asof, b), "nightly": nightly_view(rs, asof, bench if b == "main" else {}),
            "pr": pr_view(rs, asof, main, b), "ops": {**ops_view(rs, asof), "concurrency": concurrency}, "bump":
            bump_view(((bump or {}).get("lines") or {}).get(b))
        }
    return {"asof": asof, "branches": branches}
