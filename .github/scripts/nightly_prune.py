#!/usr/bin/env python3
"""Delete wheel assets on the rolling 'nightly' release beyond the last N builds
(grouped by dev-date). latest.json and non-dated assets are left untouched.
"""
import collections
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def main():
    rel = json.loads(sh("gh", "api", f"repos/{REPO}/releases/tags/nightly"))
    by_date = collections.defaultdict(list)
    for a in rel.get("assets", []):
        m = re.search(r"\.dev(\d{8})", a["name"])
        if m:
            by_date[m.group(1)].append(a)
    stale_dates = sorted(by_date)[:-KEEP] if len(by_date) > KEEP else []
    for date in stale_dates:
        for a in by_date[date]:
            sh("gh", "api", "-X", "DELETE", f"repos/{REPO}/releases/assets/{a['id']}")
            print("pruned", a["name"])
    print(f"kept {min(len(by_date), KEEP)} dev-date(s); pruned {len(stale_dates)}")


if __name__ == "__main__":
    main()
