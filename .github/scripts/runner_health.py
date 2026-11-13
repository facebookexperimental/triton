#!/usr/bin/env python3
"""Open an issue when a GPU runner pool stops picking up jobs, and close it once the pool runs a job again.

A pool, named by its runner labels, is down when one of its GPU test jobs has been queued longer than STALL_MIN
and none of its jobs is running. A busy pool always has a running job, so a long queue alone never counts.

A test run left waiting only on a down pool is cancelled after QUEUE_TIMEOUT_MIN, so it reports instead of holding
its result until GitHub drops the job after 24 hours. timeout-minutes cannot do this: it starts when a runner picks
the job up.

GITHUB_REPOSITORY comes from the Actions environment.
"""
import collections
import datetime as dt
import os
import re
import subprocess

import ci_health_rules as rules
from failure_issues import gh

STALL_MIN = 15
QUEUE_TIMEOUT_MIN = 60
#: The CI test workflows whose stuck runs are cancelled. Their aggregate jobs still run and report the cancelled
#: platform.
CANCEL_WORKFLOWS = ("gpu-tests.yml", rules.FAST_WF)
LABELS = ["ci_issues"]
TITLE = re.compile(r"^\[runner\] (.+) is not picking up jobs$")


def title_of(pool):
    return f"[runner] {pool} is not picking up jobs"


def pool_of(job):
    return ",".join(sorted(job["labels"]))


def pool_states(jobs, now):
    """(down, running): each down pool mapped to its longest-queued GPU test job, and the pools running a job.
    Each job carries "wf", the workflow file of its run."""
    running = {pool_of(j) for j in jobs if j["status"] == "in_progress"}
    down = {}
    for j in sorted(jobs, key=lambda j: j["created_at"]):
        pool = pool_of(j)
        if (j["status"] == "queued" and j["wf"] in rules.GPU_RUN_WORKFLOWS and rules.gpu_job(j) and pool not in running
                and pool not in down and now - rules.t(j["created_at"]) > dt.timedelta(minutes=STALL_MIN)):
            down[pool] = j
    return down, running


def stuck_runs(jobs, down, now):
    """Run id -> its jobs waiting on a down pool, for each CANCEL_WORKFLOWS run with nothing left but those jobs and
    one of them queued longer than QUEUE_TIMEOUT_MIN. A run with any other job still running or queued is left to
    finish first."""
    runs = collections.defaultdict(list)
    for j in jobs:
        if j["status"] != "completed":
            runs[j["run_id"]].append(j)
    out = {}
    for run_id, unfinished in runs.items():
        if (unfinished[0]["wf"] in CANCEL_WORKFLOWS
                and all(j["status"] == "queued" and pool_of(j) in down for j in unfinished)
                and any(now - rules.t(j["created_at"]) > dt.timedelta(minutes=QUEUE_TIMEOUT_MIN) for j in unfinished)):
            out[run_id] = unfinished
    return out


def pages(path, key):
    """Every item of a paginated list endpoint whose items sit under key."""
    items, page = [], 1
    while True:
        batch = gh("GET", f"{path}&per_page=100&page={page}")[key]
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def active_jobs(repo):
    jobs = []
    for status in ("queued", "in_progress"):
        for run in pages(f"repos/{repo}/actions/runs?status={status}", "workflow_runs"):
            wf = os.path.basename(run["path"])
            for j in pages(f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=latest", "jobs"):
                jobs.append({**j, "wf": wf})
    return jobs


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    now = dt.datetime.now(dt.timezone.utc)
    jobs = active_jobs(repo)
    down, running = pool_states(jobs, now)
    issues = gh("GET", f"repos/{repo}/issues?state=open&labels={','.join(LABELS)}&per_page=100", paginate=True)
    open_pools = {m[1]: i["number"] for i in issues if "pull_request" not in i and (m := TITLE.match(i["title"]))}
    # One scan can catch a busy pool between two jobs, so only a pool also down at the previous scan cancels runs.
    still_down = {pool: job for pool, job in down.items() if pool in open_pools}
    for pool, job in down.items():
        if pool in open_pools:
            continue
        waited = int((now - rules.t(job["created_at"])).total_seconds() // 60)
        body = (f"No `{pool}` runner has picked up a job for {waited} minutes, and none of its jobs is running.\n\n"
                f"Longest-waiting job: {job['html_url']}\n\n"
                f"Opened by ci-health.yml; it closes this issue once a job runs on `{pool}` again.")
        number = gh("POST", f"repos/{repo}/issues", {"title": title_of(pool), "body": body, "labels": LABELS})["number"]
        open_pools[pool] = number
        print(f"{pool}: down, opened #{number}")
    cancelled = collections.defaultdict(list)
    for run_id, waiting in stuck_runs(jobs, still_down, now).items():
        try:
            gh("POST", f"repos/{repo}/actions/runs/{run_id}/cancel")
        except subprocess.CalledProcessError as exc:  # e.g. the run finished meanwhile
            print(f"run {run_id}: not cancelled: {exc.stderr.strip() if exc.stderr else exc}")
            continue
        for pool in {pool_of(j) for j in waiting}:
            cancelled[pool].append(run_id)
        print(f"run {run_id}: cancelled after waiting over {QUEUE_TIMEOUT_MIN} minutes for a runner")
    for pool, run_ids in cancelled.items():
        links = "\n".join(f"- https://github.com/{repo}/actions/runs/{i}" for i in sorted(run_ids))
        gh(
            "POST", f"repos/{repo}/issues/{open_pools[pool]}/comments", {
                "body":
                f"Cancelled {len(run_ids)} run(s) that waited over {QUEUE_TIMEOUT_MIN} minutes for `{pool}` with "
                f"nothing else left to run:\n{links}\n\nTo test the other platforms meanwhile, set the repository "
                "variable `GPU_CI_SKIP_PLATFORMS` to the platform (e.g. `b200`), then use \"Re-run all jobs\"; "
                "\"Re-run failed jobs\" keeps the old selection."
            })
    for pool, number in open_pools.items():
        if pool in running:
            gh("POST", f"repos/{repo}/issues/{number}/comments",
               {"body": f"✅ A job is running on `{pool}` again; closing."})
            gh("PATCH", f"repos/{repo}/issues/{number}", {"state": "closed", "state_reason": "completed"})
            print(f"{pool}: recovered, closed #{number}")


if __name__ == "__main__":
    main()
