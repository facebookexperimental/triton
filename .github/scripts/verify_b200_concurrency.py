#!/usr/bin/env python3
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import re
import urllib.request

BACKGROUND = re.compile(r"probe / background slot prototype-background-b200-([0-6]) copy ([12])")
FAST = "probe / fast without background group"
WORKFLOW = "B200 concurrency prototype smoke"


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def maximum_parallel(intervals):
    events = [(start, 1) for start, end in intervals] + [(end, -1) for start, end in intervals]
    running = maximum = 0
    for _, change in sorted(events):
        running += change
        maximum = max(maximum, running)
    return maximum


def audit(run_jobs):
    if len(run_jobs) < 2:
        raise ValueError("Use at least two overlapping smoke runs to test cross-run groups")
    by_slot = defaultdict(list)
    background_details = []
    fast = []
    seen = set()
    for run_id, jobs in run_jobs.items():
        background_count = fast_count = assignment_count = 0
        for job in jobs:
            if job["id"] in seen:
                raise ValueError(f"Duplicate job id: {job['id']}")
            seen.add(job["id"])
            if job.get("status") != "completed" or job.get("conclusion") != "success":
                raise ValueError(f"Run {run_id}: job did not complete successfully: {job['name']}")
            if not job.get("started_at") or not job.get("completed_at"):
                raise ValueError(f"Run {run_id}: missing execution timestamps: {job['name']}")
            interval = (timestamp(job["started_at"]), timestamp(job["completed_at"]))
            if interval[1] <= interval[0]:
                raise ValueError(f"Run {run_id}: invalid execution interval: {job['name']}")
            match = BACKGROUND.fullmatch(job["name"])
            if match:
                background_count += 1
                slot = int(match[1])
                if not job.get("created_at"):
                    raise ValueError(f"Run {run_id}: missing queue timestamp: {job['name']}")
                created = timestamp(job["created_at"])
                if created > interval[0]:
                    raise ValueError(f"Run {run_id}: invalid queue timestamp: {job['name']}")
                by_slot[slot].append(interval)
                background_details.append((run_id, slot, created, *interval))
            elif job["name"] == FAST:
                fast_count += 1
                fast.append(interval)
            elif job["name"] == "probe / assign":
                assignment_count += 1
            else:
                raise ValueError(f"Run {run_id}: unexpected smoke job: {job['name']}")
        if (background_count, fast_count, assignment_count) != (14, 1, 1):
            raise ValueError(f"Run {run_id}: expected 14 background jobs, one Fast probe and one assignment job")
    contention = any(run != other_run and slot == other_slot and max(created, other_start) < min(start, other_end)
                     for run, slot, created, start, end in background_details
                     for other_run, other_slot, other_created, other_start, other_end in background_details)
    if not contention:
        raise ValueError("INCONCLUSIVE: no cross-run wait on the same group was observed")
    peaks = {slot: maximum_parallel(by_slot[slot]) for slot in range(7)}
    if any(peak > 1 for peak in peaks.values()):
        raise ValueError(f"Same-group execution overlap: {peaks}")
    background = [interval for intervals in by_slot.values() for interval in intervals]
    peak = maximum_parallel(background)
    if peak != 7:
        raise ValueError(f"INCONCLUSIVE: background peak was {peak}, so the seven-job ceiling was not exercised")
    events = [(a, "background", 1) for a, b in background] + [(b, "background", -1) for a, b in background]
    events += [(a, "fast", 1) for a, b in fast] + [(b, "fast", -1) for a, b in fast]
    running = defaultdict(int)
    overlaps = False
    for _, kind, change in sorted(events, key=lambda event: (event[0], event[2])):
        running[kind] += change
        overlaps |= running["background"] == 7 and running["fast"] > 0
    if not overlaps:
        raise ValueError("INCONCLUSIVE: no Fast probe execution overlapped seven running background jobs")
    return {
        "runs": len(run_jobs), "background_jobs": len(background), "fast_probes": len(fast), "peak_by_slot": peaks,
        "max_parallel_background": peak, "fast_overlaps_seven_backgrounds": overlaps, "cross_run_group_contention":
        contention, "verdict": "PASS"
    }


def api_get(repo, path):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_jobs(repo, run_id):
    run = api_get(repo, f"actions/runs/{run_id}")
    if run.get("name") != WORKFLOW or run.get("status") != "completed":
        raise ValueError(f"Run {run_id} is not a completed {WORKFLOW} run")
    jobs = []
    page = 1
    while True:
        result = api_get(repo, f"actions/runs/{run_id}/jobs?filter=latest&per_page=100&page={page}")
        jobs.extend(result["jobs"])
        if len(jobs) >= result["total_count"]:
            return jobs
        if not result["jobs"]:
            raise ValueError(f"Run {run_id}: incomplete jobs pagination")
        page += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args()
    if len(set(args.run)) != len(args.run):
        parser.error("--run values must be distinct")
    result = audit({run_id: fetch_jobs(args.repo, run_id) for run_id in args.run})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
