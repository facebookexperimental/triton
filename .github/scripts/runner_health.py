#!/usr/bin/env python3
"""Open an issue when a GPU runner pool stops picking up jobs, and close it once the pool runs a job again.

A pool, named by its runner labels, is down when one of its GPU test jobs has been queued longer than STALL_MIN
and none of its jobs is running. A busy pool always has a running job, so a long queue alone never counts.

GITHUB_REPOSITORY comes from the Actions environment.
"""
import datetime as dt
import os
import re

import ci_health_rules as rules
from failure_issues import gh

STALL_MIN = 15
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


def active_jobs(repo):
    jobs = []
    for status in ("queued", "in_progress"):
        for run in gh("GET", f"repos/{repo}/actions/runs?status={status}&per_page=100")["workflow_runs"]:
            wf = os.path.basename(run["path"])
            for j in gh("GET", f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]:
                jobs.append({**j, "wf": wf})
    return jobs


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    now = dt.datetime.now(dt.timezone.utc)
    down, running = pool_states(active_jobs(repo), now)
    issues = gh("GET", f"repos/{repo}/issues?state=open&labels={','.join(LABELS)}&per_page=100", paginate=True)
    open_pools = {m[1]: i for i in issues if "pull_request" not in i and (m := TITLE.match(i["title"]))}
    for pool, job in down.items():
        if pool in open_pools:
            continue
        waited = int((now - rules.t(job["created_at"])).total_seconds() // 60)
        body = (f"No `{pool}` runner has picked up a job for {waited} minutes, and none of its jobs is running.\n\n"
                f"Longest-waiting job: {job['html_url']}\n\n"
                f"Opened by runner-health.yml; it closes this issue once a job runs on `{pool}` again.")
        number = gh("POST", f"repos/{repo}/issues", {"title": title_of(pool), "body": body, "labels": LABELS})["number"]
        print(f"{pool}: down, opened #{number}")
    for pool, issue in open_pools.items():
        if pool in running:
            gh("POST", f"repos/{repo}/issues/{issue['number']}/comments",
               {"body": f"✅ A job is running on `{pool}` again; closing."})
            gh("PATCH", f"repos/{repo}/issues/{issue['number']}", {"state": "closed", "state_reason": "completed"})
            print(f"{pool}: recovered, closed #{issue['number']}")


if __name__ == "__main__":
    main()
