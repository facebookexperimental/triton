#!/usr/bin/env python3
"""Collect GPU workflow runs and render the /ci/ dashboard.

  collect DATA_DIR [--since ISO]
      Fetch runs of mi350/b200/h100.yml created since the saved cursor (or
      --since), plus the jobs and steps of every completed run, into
      DATA_DIR/runs-YYYY-MM.jsonl.gz and DATA_DIR/state.json. ci-health.yml
      keeps DATA_DIR on the 'ci-health' release; DATA_DIR/changed.txt lists
      the month files this run rewrote, for upload.
  render OUTDIR --from DATA_DIR
      Write OUTDIR/index.html (the page, no data) and OUTDIR/data.json (what
      the page shows, computed by ci_health_rules.page_data), for pages.yml
      to mount at /ci/. The page loads data.json asynchronously.

Stored run record, one JSON object per line in the month of created_at; a later
line for the same id replaces an earlier one:
  {id, wf, event, status, conclusion, created_at, head_sha, title, url, branch,
   jobs: [{name, conclusion, started_at, completed_at, labels,
           steps: [{name, conclusion}]}] | null until the run completes}
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import os
import re
import shutil
import subprocess

import ci_health_rules as rules

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ci_health_dashboard.html")
#: Stop fetching jobs below this many remaining API calls; the next scheduled run continues.
RATE_FLOOR = 200
#: First collection with no saved state goes back this far.
BACKFILL_DAYS = 30
#: A run still pending after this many days is treated as stuck.
STUCK_DAYS = 7
WORKFLOW_FILES = [f".github/workflows/{wf}" for wf in rules.WORKFLOWS] + [".github/workflows/build.yml"]


def gh_api(path):
    out = subprocess.run(["gh", "api", "--paginate", "--slurp", path], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def month_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "runs-*.jsonl.gz")))


def load_runs(data_dir):
    runs = {}
    for path in month_files(data_dir):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    runs[row["id"]] = row
    return runs


def _encode(rows):
    return "".join(
        json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["id"]))


def save_runs(data_dir, runs):
    """Write one file per month of created_at; rewrite only the files whose content changed."""
    os.makedirs(data_dir, exist_ok=True)
    by_month = {}
    for r in runs.values():
        by_month.setdefault(r["created_at"][:7], []).append(r)
    changed = []
    for month, rows in sorted(by_month.items()):
        path = os.path.join(data_dir, f"runs-{month}.jsonl.gz")
        body = _encode(rows)
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                if fh.read() == body:
                    continue
        with gzip.open(path, "wt") as fh:
            fh.write(body)
        changed.append(path)
    return changed


def _record(run, wf):
    return {
        "id": run["id"], "wf": wf, "event": run["event"], "status": run["status"], "conclusion": run["conclusion"],
        "created_at": run["created_at"], "head_sha": run.get("head_sha"), "title": run.get("display_title", ""), "url":
        run.get("html_url", ""), "branch": run.get("head_branch"), "jobs": None
    }


def _jobs(pages):
    return [{
        "name": j["name"], "conclusion": j["conclusion"], "started_at": j["started_at"], "completed_at":
        j["completed_at"], "labels": j.get("labels") or [], "steps":
        [{"name": s["name"], "conclusion": s["conclusion"]} for s in j.get("steps") or []]
    } for page in pages for j in page["jobs"]]


def _remaining(api):
    return api("rate_limit")[0]["resources"]["core"]["remaining"]


def collect(data_dir, since, now, api=gh_api):
    state_path = os.path.join(data_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if since is None:
        since = (rules.iso(rules.t(state["cursor"]) - dt.timedelta(hours=1))
                 if state.get("cursor") else rules.iso(rules.t(now) - dt.timedelta(days=BACKFILL_DAYS)))
    runs = load_runs(data_dir)
    for wf in rules.WORKFLOWS:
        for page in api(f"repos/{REPO}/actions/workflows/{wf}/runs?created=>={since}&per_page=100"):
            for run in page["workflow_runs"]:
                old = runs.get(run["id"])
                if old and old["status"] == "completed" and old["jobs"] is not None:
                    continue
                runs[run["id"]] = _record(run, wf)
    todo = sorted((r for r in runs.values() if r["status"] == "completed" and r["jobs"] is None),
                  key=lambda r: r["created_at"])
    unfetched = []
    for i, r in enumerate(todo):
        if i % 50 == 0 and _remaining(api) < RATE_FLOOR:
            unfetched = todo[i:]
            break
        r["jobs"] = _jobs(api(f"repos/{REPO}/actions/runs/{r['id']}/jobs?per_page=100&filter=latest"))
    # A run still not completed after STUCK_DAYS (a nightly stuck in "pending") no longer holds the cursor back.
    stuck = rules.iso(rules.t(now) - dt.timedelta(days=STUCK_DAYS))
    pending = [r["created_at"] for r in runs.values() if r["status"] != "completed" and r["created_at"] >= stuck]
    pending += [r["created_at"] for r in unfetched]
    state = {"cursor": min(pending) if pending else now, "collected_at": now}
    changed = save_runs(data_dir, runs)
    with open(state_path, "w") as fh:
        json.dump(state, fh)
    with open(os.path.join(data_dir, "changed.txt"), "w") as fh:
        fh.write(" ".join(changed))
    print(f"{len(runs)} runs stored, {len(todo) - len(unfetched)} fetched, {len(unfetched)} left for the next run; "
          f"cursor {state['cursor']}")
    return state


def workflow_changes(since):
    """Commits that changed the GPU workflow files, drawn as markers on the charts."""
    try:
        out = subprocess.run(["git", "log", f"--since={since}", "--format=%cI%x09%h%x09%s", "--"] + WORKFLOW_FILES,
                             capture_output=True, text=True, check=True, cwd=HERE).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    changes = []
    for line in out.splitlines():
        ts, sha, subject = line.split("\t", 2)
        m = re.search(r"\(#(\d+)\)\s*$", subject)
        changes.append({
            "t": rules.t(ts).astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "sha": sha, "pr":
            m.group(1) if m else None, "subject": subject
        })
    return changes


def render(data_dir, outdir, changelog=None):
    state_path = os.path.join(data_dir, "state.json")
    asof = json.load(open(state_path))["collected_at"] if os.path.exists(state_path) else None
    if asof:
        if changelog is None:
            changelog = workflow_changes(rules.iso(rules.t(asof) - dt.timedelta(days=30)))
        data = rules.page_data(list(load_runs(data_dir).values()), asof, changelog)
    else:
        data = {"asof": None}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "data.json"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    shutil.copyfile(TEMPLATE, os.path.join(outdir, "index.html"))
    print(f"rendered /ci/ as of {asof}: {os.path.getsize(os.path.join(outdir, 'data.json')) // 1024} KB of data")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("data_dir")
    c.add_argument("--since", help="re-collect runs created at or after this ISO time")
    r = sub.add_parser("render")
    r.add_argument("outdir")
    r.add_argument("--from", dest="data_dir", required=True)
    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.data_dir, args.since, rules.iso(dt.datetime.now(dt.timezone.utc)))
    else:
        render(args.data_dir, args.outdir)


if __name__ == "__main__":
    main()
