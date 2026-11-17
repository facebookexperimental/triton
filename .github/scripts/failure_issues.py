#!/usr/bin/env python3
"""File one issue per CI failure episode and close it only after verified recovery."""
import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote

from test_results import decode_recovery

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


def run_id_of(body):
    """The newest run recorded as failing or resolving the issue; 0 if absent."""
    runs = re.findall(r"(?:^\| Run \| \S*/actions/runs/|<!-- ci-resolved run=)(\d+)", body or "", re.M)
    return max(map(int, runs), default=0)


def record_resolution(body, run_id):
    """Record the resolving run without changing the last failure's SHA, run or summary."""
    body = re.sub(r"\n?<!-- ci-resolved run=\d+ -->", "", body or "").rstrip()
    return f"{body}\n\n<!-- ci-resolved run={run_id} -->\n"


def decide(api, repo, title, labels, run_id):
    issues = api("GET", f"repos/{repo}/issues?labels={quote(','.join(labels))}&state=all&per_page=100", paginate=True)
    newer = any(i["title"] == title and "pull_request" not in i and run_id_of(i.get("body")) > run_id for i in issues)
    return "skip" if newer else "file"


def recovered(body, proof):
    if not proof["complete"]:
        return False
    raw = re.search(r"^### Raw failing test IDs\n```\n(.*?)\n```", body or "", re.M | re.S)
    ids = set(raw[1].splitlines()) if raw else set()
    if ids and ids <= set(proof["passed"]):
        return True
    norm = re.search(r"^\| Normalized failing identity \| `([^`]+)` \|", body or "", re.M)
    return bool(norm and norm[1] in proof["resolved"])


def reconcile(api, repo, prefix, labels, failures_json, run_url, run_id, recovery=""):
    """Close this job's recovered issues; returns the numbers closed."""
    try:
        failures = json.loads((failures_json or "").strip() or "[]")
        proof = decode_recovery(recovery)
    except (ValueError, KeyError) as e:
        print(f"::warning::Could not parse test results; closing nothing: {e}")
        return []
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
            continue
        if run_id_of(issue["body"]) > run_id or not recovered(issue["body"], proof):
            continue
        api(
            "POST", f"repos/{repo}/issues/{issue['number']}/comments", {
                "body":
                f"Auto-closing after the previously failing tests passed in {run_url}. "
                "A new failure after this recovery will get a new issue."
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
        if jobs and job not in names.split(", "):
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
    except subprocess.CalledProcessError:
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
            except subprocess.CalledProcessError as e:
                print(f"::warning::Could not attach #{i['number']} to #{parent['number']}: {e}")
                break
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
    check = sub.add_parser("check")
    check.add_argument("--title", required=True)
    c = sub.add_parser("reconcile")
    c.add_argument("--prefix", required=True)
    c.add_argument("--failures", default="")
    c.add_argument("--recovery", default="")
    g = sub.add_parser("group")
    g.add_argument("--prefix", required=True)
    g.add_argument("--sha", required=True)
    g.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME", "main"))
    for s in (check, c, g):
        s.add_argument("--labels", required=True)
        s.add_argument("--run-url", required=True)
    a = p.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    if a.cmd == "group":
        group(gh, repo, a.prefix, labels, a.sha, a.branch, a.run_url)
        return
    run_id = int(os.environ["GITHUB_RUN_ID"])
    if a.cmd == "reconcile":
        reconcile(gh, repo, a.prefix, labels, a.failures, a.run_url, run_id, a.recovery)
        return
    try:
        action = decide(gh, repo, a.title, labels, run_id)
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"::warning::Stale check failed; filing as usual: {e}")
        action = "file"
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"action={action}\n")


if __name__ == "__main__":
    sys.exit(main())
