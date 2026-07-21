#!/usr/bin/env python3
"""Nightly selection entrypoint.

Reads the boundary (last-shipped commit) from the rolling 'nightly' Release's
latest.json, enumerates new commits since then, walks them newest->oldest with
nightly_select, and emits GitHub Actions outputs: status (ship|noop|nogreen),
sha, version.
"""
import datetime
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nightly_select import select  # noqa: E402

REQUIRED = ["LIT Tests", "h100-tlx-test", "mi350-tlx-test", "b200-tlx-test"]
# Publish target (releases + latest.json): the repo running this workflow.
REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
# Where CI check-runs are read from. Defaults to REPO; override for fork testing
# (a fork's own GPU CI is guarded off, but it shares commit SHAs with upstream).
SIGNAL_REPO = os.environ.get("TRITON_SIGNAL_REPO") or REPO
CAP = 100


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def gh_json(path):
    out = sh("gh", "api", path)
    return json.loads(out) if out.strip() else None


def boundary_commit():
    """Full SHA of the last shipped nightly, from latest.json on the rolling release."""
    data = gh_json(f"repos/{REPO}/releases/tags/nightly")
    if not data:
        return None
    for asset in data.get("assets", []):
        if asset["name"] == "latest.json":
            raw = sh("gh", "api", asset["url"], "-H", "Accept: application/octet-stream")
            try:
                return json.loads(raw).get("commit")
            except json.JSONDecodeError:
                return None
    return None


def new_commits(boundary):
    rng = f"{boundary}..HEAD" if boundary else "HEAD"
    out = sh("git", "rev-list", "--max-count", str(CAP), rng)
    return [line for line in out.split() if line]


def fetch(sha):
    data = gh_json(f"repos/{SIGNAL_REPO}/commits/{sha}/check-runs?per_page=100") or {}
    by_check = {}
    for cr in data.get("check_runs", []):
        by_check.setdefault(cr["name"],
                            []).append({"completed_at": cr.get("completed_at"), "conclusion": cr.get("conclusion")})
    return by_check


def emit(**kv):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


def base_version():
    m = re.search(r'^TRITON_VERSION = "([0-9]+\.[0-9]+\.[0-9]+)"', open("setup.py").read(), re.M)
    return m.group(1) if m else "3.8.0"


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    cut = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date = now.strftime("%Y%m%d")
    shas = new_commits(boundary_commit())
    if not shas:
        print("no new commits since last nightly -> noop")
        emit(status="noop")
        return
    sha = select(shas, fetch, REQUIRED, cut, CAP)
    if not sha:
        print(f"no all-green commit in {len(shas)} candidates -> nogreen")
        emit(status="nogreen")
        return
    # The nightly version *scheme* is decided here (the caller); wheels_fb only
    # applies the provided version string.
    version = f"{base_version()}.dev{date}"
    print(f"selected {sha} -> ship {version}")
    emit(status="ship", sha=sha, version=version)


if __name__ == "__main__":
    main()
