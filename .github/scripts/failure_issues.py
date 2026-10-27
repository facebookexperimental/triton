#!/usr/bin/env python3
"""Reopen and close CI failure issues filed by report-nightly-failure.yml.

  failure_issues.py reopen --title T --labels a,b --summary S --sha SHA --run-url U
      If the failure's last issue was auto-closed within a week and failed with the same error, reopen it instead
      of filing a new one. An issue a person closed is never reopened. After MAX_REOPENS reopens, the next
      recurrence labels it flaky. Writes reopened=true|false to GITHUB_OUTPUT; on false, or on any error, the
      report step files or updates the issue as before.

  failure_issues.py reconcile --prefix P --labels a,b --failures JSON --run-url U
      Close this job's open issues whose failure no longer appears. Leaves flaky issues for a person to close, and
      issues a newer run reported: pushes run concurrently, so an older run can finish after a newer one.

GITHUB_REPOSITORY and GITHUB_RUN_ID come from the Actions environment.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote

BOT = "github-actions[bot]"
WINDOW = dt.timedelta(days=7)
MAX_REOPENS = 2
FLAKY = "flaky"


def gh(method, path, fields=None, paginate=False):
    """One GitHub API call through gh, with fields as the JSON request body. paginate concatenates list pages."""
    cmd = ["gh", "api", "-X", method, path] + (["--paginate", "--slurp"] if paginate else [])
    cmd += ["--input", "-"] if fields is not None else []
    out = subprocess.run(cmd, input=json.dumps(fields) if fields is not None else None, capture_output=True, text=True,
                         check=True).stdout
    data = json.loads(out) if out.strip() else None
    return [x for page in data for x in page] if paginate else data


def same_error(a, b):
    """Whether two failure summaries describe the same error: first lines equal once numbers are masked."""
    norm = lambda s: re.sub(r"\d+(\.\d+)?", "N", (s or "").split("\n")[0].strip())
    return norm(a) == norm(b)


def summary_of(body):
    """The summary line of the last failure an issue body recorded."""
    marker = "### Summary\n```\n"
    i = (body or "").find(marker)
    return body[i + len(marker):].split("\n")[0] if i >= 0 else ""


def run_id_of(body):
    """The id of the run that last reported the failure, from the issue's status table; 0 if absent."""
    m = re.search(r"^\| Run \| \S*/actions/runs/(\d+)", body or "", re.M)
    return int(m[1]) if m else 0


def record_failure(body, sha, run_url):
    """The body with its status table pointing at this failing run."""
    body = re.sub(r"^\| SHA \| .*$", f"| SHA | `{sha}` |", body or "", flags=re.M)
    return re.sub(r"^\| Run \| .*$", f"| Run | {run_url} |", body, flags=re.M)


def decide(api, repo, title, labels, summary, now):
    """(issue, reopens, flaky) for the closed issue to reopen, or None to leave filing to the report step."""
    quoted = title.replace('"', '\\"')
    q = " ".join([f"repo:{repo}", "is:issue", *[f'label:"{l}"' for l in labels], f'in:title "{quoted}"'])
    found = api("GET", f"search/issues?sort=created&order=desc&per_page=100&q={quote(q, safe='')}")["items"]
    found = [i for i in found if i["title"] == title and "pull_request" not in i]
    # Search can lag a close by seconds, so confirm what it reports as open.
    issues = [api("GET", f"repos/{repo}/issues/{i['number']}") if i["state"] == "open" else i for i in found]
    if any(i["state"] == "open" for i in issues):
        return None  # the report step updates the open issue
    closed = [i for i in issues if i.get("closed_at")]
    if not closed:
        return None
    # Only the most recently closed issue counts: a person's close stands even if an older one was the bot's.
    issue = api("GET", f"repos/{repo}/issues/{max(closed, key=lambda i: i['closed_at'])['number']}")
    if (issue.get("closed_by") or {}).get("login") != BOT:
        return None
    if now - dt.datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00")) > WINDOW:
        return None
    if not same_error(summary_of(issue.get("body")), summary):
        return None
    events = api("GET", f"repos/{repo}/issues/{issue['number']}/events?per_page=100")
    reopens = sum(e["event"] == "reopened" and (e.get("actor") or {}).get("login") == BOT for e in events)
    return issue, reopens, reopens >= MAX_REOPENS


def reopen(api, repo, issue, reopens, flaky, sha, run_url, reopened):
    """Reopen the issue for this failing run. Calls reopened() once the issue is open again."""
    number = issue["number"]
    api("PATCH", f"repos/{repo}/issues/{number}",
        {"state": "open", "body": record_failure(issue["body"], sha, run_url)})
    reopened()
    if flaky:
        api("POST", f"repos/{repo}/issues/{number}/labels", {"labels": [FLAKY]})
        body = (f"🔁 Failed again in {run_url} after {reopens} reopens; labeled `{FLAKY}`. Passing runs no longer "
                "close this issue: close it by hand once the test is fixed.")
    else:
        body = f"🔁 Failed again in {run_url}; reopened ({reopens + 1} of {MAX_REOPENS})."
    api("POST", f"repos/{repo}/issues/{number}/comments", {"body": body})


def reconcile(api, repo, prefix, labels, failures_json, run_url, run_id):
    """Close this job's recovered issues; returns the numbers closed."""
    try:
        failures = json.loads((failures_json or "").strip() or "[]")
    except ValueError as e:
        print(f"::warning::Could not parse failures; closing nothing: {e}")
        return []
    # A job-level fallback item (compile error, timeout, missing JUnit XML) means this run has no per-test
    # signal, and closing now could resolve a real failure during an infra outage.
    if any(isinstance(f, dict) and f.get("fallback") is True for f in failures):
        print("Fallback failure present (no per-test signal); closing nothing.")
        return []
    still_failing = {f["issue_title"] for f in failures}
    closed = []
    for issue in api("GET", f"repos/{repo}/issues?state=open&labels={quote(','.join(labels))}&per_page=100",
                     paginate=True):
        if "pull_request" in issue or not issue["title"].startswith(prefix) or issue["title"] in still_failing:
            continue
        if any(l["name"] == FLAKY for l in issue["labels"]):
            continue  # a person closes flaky issues
        if run_id_of(issue["body"]) > run_id:
            continue  # a newer run reported it after this run started
        api(
            "POST", f"repos/{repo}/issues/{issue['number']}/comments", {
                "body":
                f"✅ Auto-closing: this failure identity no longer appears in the latest run ({run_url}). "
                "A new issue will be filed automatically if it recurs."
            })
        api("PATCH", f"repos/{repo}/issues/{issue['number']}", {"state": "closed", "state_reason": "completed"})
        closed.append(issue["number"])
    print(f'Reconcile complete for "{prefix}": closed {len(closed)} issue(s).')
    return closed


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reopen")
    r.add_argument("--title", required=True)
    r.add_argument("--summary", default="")
    r.add_argument("--sha", required=True)
    c = sub.add_parser("reconcile")
    c.add_argument("--prefix", required=True)
    c.add_argument("--failures", default="")
    for s in (r, c):
        s.add_argument("--labels", required=True)
        s.add_argument("--run-url", required=True)
    a = p.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    if a.cmd == "reconcile":
        reconcile(gh, repo, a.prefix, labels, a.failures, a.run_url, int(os.environ["GITHUB_RUN_ID"]))
        return

    def output(value):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"reopened={value}\n")

    try:
        found = decide(gh, repo, a.title, labels, a.summary, dt.datetime.now(dt.timezone.utc))
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"::warning::Reopen check failed; filing as usual: {e}")
        found = None
    if not found:
        output("false")
        return
    done = []
    try:
        reopen(gh, repo, *found, a.sha, a.run_url, reopened=lambda: (output("true"), done.append(True)))
        print(f"Reopened #{found[0]['number']}{' as flaky' if found[2] else ''}: {a.title}")
    except subprocess.CalledProcessError as e:
        # Before the issue reopened, let the report step file; after, it is open and filing would duplicate it.
        print(f"::warning::Reopen did not finish: {e}")
        if not done:
            output("false")


if __name__ == "__main__":
    sys.exit(main())
