#!/usr/bin/env python3
"""Reopen the last auto-closed issue of a recurring CI failure instead of filing a new one.

A failure reopens its issue only if reconcile closed it within the last week and the failure has the same exception
type. An issue a person closed is never reopened. After MAX_REOPENS reopens, the next recurrence labels the issue
flaky, and reconcile then leaves it open until a person closes it.

  reopen_failure_issue.py --repo OWNER/REPO --title T --labels a,b --summary S --run-url U

Writes reopened=true|false to GITHUB_OUTPUT. When false, report-nightly-failure.yml files or updates the issue as
before.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
from urllib.parse import quote

BOT = "github-actions[bot]"
WINDOW = dt.timedelta(days=7)
MAX_REOPENS = 2
FLAKY = "flaky"


def gh(method, path, fields=None):
    """One GitHub API call through gh, with fields as the JSON request body."""
    cmd = ["gh", "api", "-X", method, path] + (["--input", "-"] if fields is not None else [])
    out = subprocess.run(cmd, input=json.dumps(fields) if fields is not None else None, capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out) if out.strip() else None


def error_type(summary):
    """'AssertionError: Tensor-likes are not close!' -> 'AssertionError'."""
    return (summary or "").split("\n")[0].split(":")[0].strip()


def summary_of(body):
    """The summary line of the last failure an issue body recorded."""
    marker = "### Summary\n```\n"
    i = (body or "").find(marker)
    return body[i + len(marker):].split("\n")[0] if i >= 0 else ""


def search(api, repo, state, title, labels):
    quoted = title.replace('"', '\\"')
    q = " ".join([f"repo:{repo}", "is:issue", f"is:{state}", *[f'label:"{l}"' for l in labels], f'in:title "{quoted}"'])
    items = api("GET", "search/issues?per_page=10&q=" + quote(q, safe=""))["items"]
    return [i for i in items if i["title"] == title and "pull_request" not in i]


def decide(api, repo, title, labels, summary, now):
    """(issue, reopens, flaky) for the closed issue to reopen, or None to leave filing to the report step."""
    if search(api, repo, "open", title, labels):
        return None  # the report step updates the open issue
    closed = search(api, repo, "closed", title, labels)
    if not closed:
        return None
    # Only the most recently closed issue counts: a person's close stands even if an older one was the bot's.
    last = max(closed, key=lambda i: i["closed_at"])
    issue = api("GET", f"repos/{repo}/issues/{last['number']}")
    if (issue.get("closed_by") or {}).get("login") != BOT:
        return None
    closed_at = dt.datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00"))
    if now - closed_at > WINDOW:
        return None
    if error_type(summary_of(issue.get("body"))) != error_type(summary):
        return None
    events = api("GET", f"repos/{repo}/issues/{issue['number']}/events?per_page=100")
    reopens = sum(e["event"] == "reopened" and (e.get("actor") or {}).get("login") == BOT for e in events)
    return issue, reopens, reopens >= MAX_REOPENS


def reopen(api, repo, issue, reopens, flaky, run_url):
    number = issue["number"]
    api("PATCH", f"repos/{repo}/issues/{number}", {"state": "open"})
    if flaky:
        api("POST", f"repos/{repo}/issues/{number}/labels", {"labels": [FLAKY]})
        body = (f"🔁 Failed again in {run_url} after {reopens} reopens; labeled `{FLAKY}`. Passing runs no longer "
                "close this issue: close it by hand once the test is fixed.")
    else:
        body = f"🔁 Failed again in {run_url}; reopened ({reopens + 1} of {MAX_REOPENS})."
    api("POST", f"repos/{repo}/issues/{number}/comments", {"body": body})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--run-url", required=True)
    a = p.parse_args()
    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    found = decide(gh, a.repo, a.title, labels, a.summary, dt.datetime.now(dt.timezone.utc))
    if found:
        reopen(gh, a.repo, *found, a.run_url)
        print(f"Reopened #{found[0]['number']}{' as flaky' if found[2] else ''}: {a.title}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"reopened={'true' if found else 'false'}\n")


if __name__ == "__main__":
    main()
