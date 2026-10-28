#!/usr/bin/env python3
"""Reopen and close CI failure issues filed by report-nightly-failure.yml.

  failure_issues.py reopen --title T --labels a,b --summary S --sha SHA --run-url U
      If the failure's last issue was auto-closed within a week and failed with the same error, reopen it instead
      of filing a new one. An issue a person closed is never reopened. After MAX_REOPENS reopens, the next
      recurrence labels it flaky. Writes action=reopened|skip|file to GITHUB_OUTPUT: skip when the open issue
      already records a newer run, so an older run cannot wind its status back. On file, or on any error, the
      report step files or updates the issue as before.

  failure_issues.py reconcile --prefix P --labels a,b --failures JSON --run-url U
      Close this job's open issues whose failure no longer appears. Leaves flaky issues for a person to close, and
      issues a newer run reported: pushes run concurrently, so an older run can finish after a newer one.

  failure_issues.py group --prefix P --labels a,b --sha SHA --run-url U
      Group this job's issues under one parent per commit where failures were first seen, assigned to that
      commit's author. Issues this run filed join the parent for SHA; issues already open keep their parent. A
      parent reopens when one of its issues reopens, and closes when only flaky issues are left open.

GITHUB_REPOSITORY, GITHUB_RUN_ID and GITHUB_REF_NAME come from the Actions environment.
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
PARENT = "ci-parent"


def gh(method, path, fields=None, paginate=False):
    """One GitHub API call through gh, with fields as the JSON request body. paginate concatenates list pages."""
    cmd = ["gh", "api", "-X", method, path] + (["--paginate", "--slurp"] if paginate else [])
    cmd += ["--input", "-"] if fields is not None else []
    out = subprocess.run(cmd, input=json.dumps(fields) if fields is not None else None, capture_output=True, text=True,
                         check=True).stdout
    data = json.loads(out) if out.strip() else None
    return [x for page in data for x in page] if paginate else data


NUMBER = re.compile(r"0x[0-9a-fA-F]+|[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def same_error(a, b):
    """Whether two failure summaries describe the same error: equal once numbers are masked."""
    norm = lambda s: " ".join(NUMBER.sub("N", s or "").split())
    return norm(a) == norm(b)


def summary_of(body):
    """The summary of the last failure an issue body recorded."""
    m = re.search(r"^### Summary\n```\n(.*?)\n```", body or "", re.M | re.S)
    return m[1] if m else ""


def run_id_of(body):
    """The id of the run that last reported the failure, from the issue's status table; 0 if absent."""
    m = re.search(r"^\| Run \| \S*/actions/runs/(\d+)", body or "", re.M)
    return int(m[1]) if m else 0


def record_failure(body, sha, run_url):
    """The body with its status table pointing at this failing run."""
    body = re.sub(r"^\| SHA \| .*$", f"| SHA | `{sha}` |", body or "", flags=re.M)
    return re.sub(r"^\| Run \| .*$", f"| Run | {run_url} |", body, flags=re.M)


def decide(api, repo, title, labels, summary, now, run_id):
    """("reopen", (issue, reopens, flaky)), ("skip", None) when the open issue records a newer run, or
    ("file", None) to leave filing to the report step."""
    quoted = title.replace('"', '\\"')
    q = " ".join([f"repo:{repo}", "is:issue", *[f'label:"{l}"' for l in labels], f'in:title "{quoted}"'])
    found = api("GET", f"search/issues?per_page=100&q={quote(q, safe='')}")["items"]
    found = [i for i in found if i["title"] == title and "pull_request" not in i]
    # Search can lag a close by seconds, so confirm what it reports as open.
    issues = [api("GET", f"repos/{repo}/issues/{i['number']}") if i["state"] == "open" else i for i in found]
    open_issues = [i for i in issues if i["state"] == "open"]
    if open_issues:
        return ("skip" if any(run_id_of(i["body"]) > run_id for i in open_issues) else "file"), None
    closed = [i for i in issues if i.get("closed_at")]
    if not closed:
        return "file", None
    # Only the most recently closed issue counts: a person's close stands even if an older one was the bot's.
    issue = api("GET", f"repos/{repo}/issues/{max(closed, key=lambda i: i['closed_at'])['number']}")
    if (issue.get("closed_by") or {}).get("login") != BOT:
        return "file", None
    if now - dt.datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00")) > WINDOW:
        return "file", None
    if not same_error(summary_of(issue.get("body")), summary):
        return "file", None
    events = api("GET", f"repos/{repo}/issues/{issue['number']}/events?per_page=100")
    reopens = sum(e["event"] == "reopened" and (e.get("actor") or {}).get("login") == BOT for e in events)
    return "reopen", (issue, reopens, reopens >= MAX_REOPENS)


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


def first_seen_sha(body):
    """The commit whose run filed the issue, from the bisect metadata the report writes once."""
    m = re.search(r"^bad_sha=(\w+)", body or "", re.M)
    return m[1] if m else ""


def suite_name(job):
    """The test suite a job runs, for parent titles: mi350-tlx-test -> MI350 TLX, lit-tests -> LIT."""
    words = re.sub(r"-tests?$", "", job).split("-")
    names = {"tlx": "TLX", "torchtlx": "TorchTLX", "lit": "LIT", "mi350": "MI350", "h100": "H100", "b200": "B200"}
    return " ".join(names.get(w, w.capitalize()) for w in words)


def find_or_create_parent(api, repo, sha, suite, branch, run_url, open_parents):
    """The open parent for failures first seen at sha, with suite in its title. A new parent is assigned to the
    commit's author. Parents are found by a marker in the body, since their titles grow as suites join."""
    marker = f"<!-- ci-parent sha={sha} -->"
    parent = next((p for p in open_parents if marker in (p.get("body") or "")), None)
    if parent:
        head, suites = parent["title"].split(": ", 1)
        names, on, rest = suites.partition(" on ")
        if suite not in names.split(", "):
            title = f"{head}: {names}, {suite}{on}{rest}"
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"title": title})
            parent["title"] = title
        return parent
    commit = api("GET", f"repos/{repo}/commits/{sha}")
    author = (commit.get("author") or {}).get("login")
    pr = re.search(r"\(#(\d+)\)\s*$", commit["commit"]["message"].split("\n")[0])
    at = f"{sha[:7]}{f' (#{pr[1]})' if pr else ''}"
    title = f"Test failures at {at}: {suite}" + (f" on {branch}" if branch != "main" else "")
    body = (f"Tests started failing at {at}, first seen in {run_url}. Each failing test is a sub-issue; this "
            "issue closes when they are all fixed.\n\n"
            "Please comment within a working day: fixing it (link the PR), reverting it, or not yours (say why, "
            "and reassign). The first commit a failure is seen at is usually, not always, the cause; each "
            f"sub-issue lists the suspect range.\n\n{marker}")
    fields = {"title": title, "body": body, "labels": [PARENT]}
    try:
        parent = api("POST", f"repos/{repo}/issues", dict(fields, assignees=[author] if author else []))
    except subprocess.CalledProcessError:  # the author cannot be assigned: mention them instead
        parent = api("POST", f"repos/{repo}/issues", dict(fields, body=f"@{author} {body}" if author else body))
    open_parents.append(parent)
    return parent


def group(api, repo, prefix, labels, sha, branch, run_url):
    """Attach this run's new issues to their commit's parent, then reopen and close parents."""
    issues = [
        i for i in api("GET", f"repos/{repo}/issues?state=open&labels={quote(','.join(labels))}&per_page=100",
                       paginate=True) if i["title"].startswith(prefix) and "pull_request" not in i
    ]
    open_parents = api("GET", f"repos/{repo}/issues?state=open&labels={PARENT}&per_page=100", paginate=True)
    new = [i for i in issues if not i.get("parent_issue_url") and first_seen_sha(i["body"]) == sha]
    if new:
        parent = find_or_create_parent(api, repo, sha, suite_name(prefix.split(" / ")[1]), branch, run_url,
                                       open_parents)
        for i in new:
            api("POST", f"repos/{repo}/issues/{parent['number']}/sub_issues", {"sub_issue_id": i["id"]})
    # A sub-issue that reopened (failed again within the reopen window) reopens its parent.
    for url in {i["parent_issue_url"] for i in issues if i.get("parent_issue_url")}:
        parent = api("GET", "repos/" + url.split("/repos/", 1)[1])
        if parent["state"] == "closed":
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"state": "open"})
            api("POST", f"repos/{repo}/issues/{parent['number']}/comments",
                {"body": f"🔁 A sub-issue failed again in {run_url}; reopened."})
            open_parents.append(parent)
    for parent in open_parents:
        subs = api("GET", f"repos/{repo}/issues/{parent['number']}/sub_issues?per_page=100")
        left = [s for s in subs if s["state"] == "open"]
        if subs and all(any(l["name"] == FLAKY for l in s["labels"]) for s in left):
            note = "".join(f"\nStill open as flaky: #{s['number']}" for s in left)
            api("POST", f"repos/{repo}/issues/{parent['number']}/comments",
                {"body": f"✅ All sub-issues are fixed as of {run_url}.{note}"})
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"state": "closed", "state_reason": "completed"})


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
    g = sub.add_parser("group")
    g.add_argument("--prefix", required=True)
    g.add_argument("--sha", required=True)
    for s in (r, c, g):
        s.add_argument("--labels", required=True)
        s.add_argument("--run-url", required=True)
    a = p.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    if a.cmd == "group":
        group(gh, repo, a.prefix, labels, a.sha, os.environ.get("GITHUB_REF_NAME", "main"), a.run_url)
        return
    if a.cmd == "reconcile":
        reconcile(gh, repo, a.prefix, labels, a.failures, a.run_url, int(os.environ["GITHUB_RUN_ID"]))
        return

    def output(action):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"action={action}\n")

    try:
        action, found = decide(gh, repo, a.title, labels, a.summary, dt.datetime.now(dt.timezone.utc),
                               int(os.environ["GITHUB_RUN_ID"]))
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"::warning::Reopen check failed; filing as usual: {e}")
        action, found = "file", None
    if action != "reopen":
        output(action)
        return
    done = []
    try:
        reopen(gh, repo, *found, a.sha, a.run_url, reopened=lambda: (output("reopened"), done.append(True)))
        print(f"Reopened #{found[0]['number']}{' as flaky' if found[2] else ''}: {a.title}")
    except subprocess.CalledProcessError as e:
        # Before the issue reopened, let the report step file; after, it is open and filing would duplicate it.
        print(f"::warning::Reopen did not finish: {e}")
        if not done:
            output("file")


if __name__ == "__main__":
    sys.exit(main())
