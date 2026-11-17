"""Decide whether a pull request runs the TLX codegen trace.

tlx-codegen-trace.yml uses it to gate the trace.
Usage: codegen_trace.py BASE HEAD (prints true or false).
"""
import os
import re
import subprocess
import sys

CODEGEN_PATHS = re.compile(
    r"^(lib/|include/|python/src/|python/triton/|third_party/(nvidia|tlx)/|cmake/|CMakeLists\.txt$"
    r"|setup\.py$|\.github/scripts/(tlx_codegen_snapshot|tlx_ops_bench_ir_diff)\.py$"
    r"|\.github/workflows/tlx-codegen-trace\.yml$)")


def needs_trace(base_ref, head_ref, paths):
    """Bundle (reactor) PRs into main always trace; other PRs into main trace compiler or kernel changes."""
    if base_ref != "main":
        return False
    return head_ref.startswith("reactor/sync-train/") or any(CODEGEN_PATHS.match(path) for path in paths)


def main():
    base, head = sys.argv[1:]
    paths = subprocess.check_output(["git", "diff", "--name-only", base, head], text=True).splitlines()
    print(str(needs_trace(os.environ.get("GITHUB_BASE_REF", ""), os.environ.get("GITHUB_HEAD_REF", ""), paths)).lower())


if __name__ == "__main__":
    main()
