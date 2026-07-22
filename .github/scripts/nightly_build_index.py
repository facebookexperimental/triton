#!/usr/bin/env python3
"""Fetch current 'nightly' release assets and write a PEP 503 simple index
(root + fbtriton/) under the given output directory, for GitHub Pages.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nightly_gen_simple_index import render_project_index, render_root_index  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def main(outdir):
    rel = json.loads(sh("gh", "api", f"repos/{REPO}/releases/tags/nightly"))
    files = [(a["name"], a["browser_download_url"], (a.get("digest") or "").replace("sha256:", ""))
             for a in rel.get("assets", [])
             if a["name"].endswith(".whl")]
    os.makedirs(os.path.join(outdir, "fbtriton"), exist_ok=True)
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(render_root_index())
    with open(os.path.join(outdir, "fbtriton", "index.html"), "w") as f:
        f.write(render_project_index(files))
    print(f"wrote index with {len(files)} wheels -> {outdir}")


if __name__ == "__main__":
    main(sys.argv[1])
