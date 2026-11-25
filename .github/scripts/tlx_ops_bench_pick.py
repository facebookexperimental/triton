#!/usr/bin/env python3
"""Pick the main commit the nightly tlx.ops bench runs on, from gpu-tests.yml's main push runs.

Takes the newest completed run whose wheel built, preferring one in which every benched platform's TLX tests ran.
Their conclusions are recorded with the numbers instead of gating them.
"""
import datetime
import json
import os
import re
import subprocess
import urllib.parse

PLATFORMS = ("b200", "h100", "mi350")
VERDICTS = ("success", "failure", "timed_out")
REPO = os.environ.get("GH_REPO", "facebookexperimental/triton")


def pick(runs, jobs_of, platforms):
    fallback = None
    for run in runs:
        jobs = jobs_of(run)
        if jobs.get("build / build") != "success":
            continue
        ci = {p: jobs.get(f"{p} / {p}-tlx-test", "not run") for p in platforms}
        if all(s in VERDICTS for s in ci.values()):
            return run, ci
        fallback = fallback or (run, ci)
    return fallback or (None, {p: "not run" for p in platforms})


def gh(path, **query):
    url = f"repos/{REPO}/{path}?{urllib.parse.urlencode(query)}"
    return json.loads(subprocess.check_output(["gh", "api", url], text=True))


def push_runs(**query):
    return gh("actions/workflows/gpu-tests.yml/runs", branch="main", event="push", status="completed", per_page=50,
              **query)["workflow_runs"]


def jobs_of(run):
    jobs = gh(f"actions/runs/{run['id']}/jobs", filter="latest", per_page=100)["jobs"]
    return {j["name"]: j["conclusion"] for j in jobs}


def main():
    event = os.environ["EVENT_NAME"]
    platforms = [p for p in PLATFORMS if p not in os.environ.get("SKIP_PLATFORMS", "")]
    now = datetime.datetime.now(datetime.timezone.utc)
    date = os.environ.get("INPUT_DATE") or now.strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        raise SystemExit(f"not a YYYYMMDD date: {date}")
    if event == "schedule":
        assets = {a["name"] for a in gh("releases/tags/tlx-ops-bench")["assets"]}
        if all(f"{date}.{p}.json" in assets for p in platforms):
            print(f"{date} already has every platform's summary; nothing to bench.")
            return

    sha = os.environ.get("INPUT_SHA", "")
    if sha:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise SystemExit(f"not a full commit sha: {sha}")
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], check=True)
        run, ci = pick(push_runs(head_sha=sha), jobs_of, platforms)
    else:
        # A backfilled night sees only the runs main had by that night's Pacific midnight.
        cut = f"{date[:4]}-{date[4:6]}-{date[6:]}T08:00:00Z" if os.environ.get("INPUT_DATE") else now.strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        run, ci = pick(push_runs(created=f"<={cut}"), jobs_of, platforms)
        if run is None:
            raise SystemExit(f"no main push run with a built wheel before {cut}")
        sha = run["head_sha"]

    setup = subprocess.check_output(["git", "show", f"{sha}:setup.py"], text=True)
    version = re.search(r'^TRITON_VERSION = "([0-9.]+)"', setup, re.M).group(1) + f".dev{date}"
    run_url = run["html_url"] if run else ""
    note = ", ".join(f"{p} {s}" for p, s in ci.items())
    print(f"Benching {version} ({sha}); TLX tests on that commit: {note}. {run_url}")
    if any(s != "success" for s in ci.values()):
        print(f"::warning::Benching a commit whose TLX tests are not all green: {note}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        out.write(f"version={version}\nsha={sha}\nrun_url={run_url}\n")
        out.writelines(f"ci_{p}={s}\n" for p, s in ci.items())
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"Benching `{version}` at {sha}.\n\nTLX tests on that commit ([run]({run_url})): {note}\n")


if __name__ == "__main__":
    main()
