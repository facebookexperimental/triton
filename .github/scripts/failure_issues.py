#!/usr/bin/env python3
"""Reopen and close CI failure issues filed by report-nightly-failure.yml.

  failure_issues.py reopen --title T --labels a,b --summary S --sha SHA --run-url U
      If the failure's last issue was auto-closed within a week and failed with the same error, reopen it instead
      of filing a new one. An issue a person closed is never reopened. After MAX_REOPENS reopens, the next
      recurrence labels it flaky. Writes action=reopened|skip|file to GITHUB_OUTPUT: skip when the issue
      already records a newer failing or passing run, so an older run cannot wind its status back. On file, or on
      any error, the report step files or updates the issue as before.

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
    """The newest run recorded as failing or resolving the issue; 0 if absent."""
    runs = re.findall(r"(?:^\| Run \| \S*/actions/runs/|<!-- ci-resolved run=)(\d+)", body or "", re.M)
    return max(map(int, runs), default=0)


def record_resolution(body, run_id):
    """Record the resolving run without changing the last failure's SHA, run or summary."""
    body = re.sub(r"\n?<!-- ci-resolved run=\d+ -->", "", body or "").rstrip()
    return f"{body}\n\n<!-- ci-resolved run={run_id} -->\n"


def record_failure(body, sha, run_url):
    """The body with its status table pointing at this failing run."""
    body = re.sub(r"^\| SHA \| .*$", f"| SHA | `{sha}` |", body or "", flags=re.M)
    return re.sub(r"^\| Run \| .*$", f"| Run | {run_url} |", body, flags=re.M)


def decide(api, repo, title, labels, summary, now, run_id):
    """("reopen", (issue, reopens, flaky)), ("skip", None) when the issue records a newer run, or
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
    if run_id_of(issue["body"]) > run_id:
        return "skip", None
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
        api("PATCH", f"repos/{repo}/issues/{issue['number']}",
            {"state": "closed", "state_reason": "completed", "body": record_resolution(issue["body"], run_id)})
        closed.append(issue["number"])
    print(f'Reconcile complete for "{prefix}": closed {len(closed)} issue(s).')
    return closed


def first_seen_sha(body):
    """The commit whose run filed the issue, from the bisect metadata the report writes once."""
    m = re.search(r"^bad_sha=(\w+)", body or "", re.M)
    return m[1] if m else ""


def find_or_create_parent(api, repo, sha, job, branch, run_url, open_parents):
    """The parent for failures first seen at sha, with job in its title, reopened if a passing run closed it. A new
    parent is assigned to the author of the commit's PR. Parents are found by a marker in the body, since their
    titles grow as jobs join."""
    marker = f"<!-- ci-parent sha={sha} -->"
    recent = api("GET", f"repos/{repo}/issues?state=closed&labels={PARENT}&sort=updated&direction=desc&per_page=100")
    parent = next((p for p in open_parents + recent if marker in (p.get("body") or "")), None)
    if parent:
        if parent["state"] == "closed":
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"state": "open"})
            api("POST", f"repos/{repo}/issues/{parent['number']}/comments",
                {"body": f"🔁 More tests failed at this commit in {run_url}; reopened."})
            parent["state"] = "open"
            open_parents.append(parent)
        head, _, jobs = parent["title"].partition(": ")
        names, on, rest = jobs.partition(" on ")
        if jobs and job not in names.split(", "):  # a person may have retitled it
            title = f"{head}: {names}, {job}{on}{rest}"
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"title": title})
            parent["title"] = title
        return parent
    commit = api("GET", f"repos/{repo}/commits/{sha}")
    pr = re.search(r"\(#(\d+)\)\s*$", commit["commit"]["message"].split("\n")[0])
    # The PR's author wrote the change; a commit's author can be unlinked to any GitHub user. A PR is also an
    # issue, so the issues permission is enough to read it.
    author = api("GET", f"repos/{repo}/issues/{pr[1]}")["user"]["login"] if pr else (commit.get("author")
                                                                                     or {}).get("login")
    at = f"{sha[:7]}{f' (#{pr[1]})' if pr else ''}"
    title = f"Test failures at {at}: {job}" + (f" on {branch}" if branch != "main" else "")
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
        parent = find_or_create_parent(api, repo, sha, prefix.split(" / ")[1], branch, run_url, open_parents)
        for i in new:
            try:
                api("POST", f"repos/{repo}/issues/{parent['number']}/sub_issues", {"sub_issue_id": i["id"]})
            except subprocess.CalledProcessError as e:  # a parent holds at most 100; the rest stay unparented
                print(f"::warning::Could not attach #{i['number']} to #{parent['number']}: {e}")
                break
    # A sub-issue that failed again (reopened within the reopen window) reopens its parent; a flaky one does not,
    # since a parent closes while flaky sub-issues stay open.
    flaky = lambda i: any(l["name"] == FLAKY for l in i["labels"])
    for url in {i["parent_issue_url"] for i in issues if i.get("parent_issue_url") and not flaky(i)}:
        parent = api("GET", "repos/" + url.split("/repos/", 1)[1])
        if parent["state"] == "closed":
            api("PATCH", f"repos/{repo}/issues/{parent['number']}", {"state": "open"})
            api("POST", f"repos/{repo}/issues/{parent['number']}/comments",
                {"body": f"🔁 A sub-issue failed again in {run_url}; reopened."})
            open_parents.append(parent)
    for parent in open_parents:
        subs = api("GET", f"repos/{repo}/issues/{parent['number']}/sub_issues?per_page=100")
        left = [s for s in subs if s["state"] == "open"]
        if subs and all(flaky(s) for s in left):
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
