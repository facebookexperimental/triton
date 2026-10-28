#!/usr/bin/env python3
"""Run tests under NVIDIA Compute Sanitizer through CUTracer and summarize them.

For each test file the command names, and each requested tool, runs
``cutracer compute-sanitizer --tool <tool>`` on the command narrowed to that
file, then classifies the run from the sanitizer log. Each file gets its own
process because a sanitizer error aborts the kernel and leaves a sticky CUDA
error that fails every later test in the process; a test case named on its own
also gets its own process, for cases too large to share one. A run is:

    findings  the sanitizer reported at least one error or hazard in device
              code
    none      the sanitizer finished and reported no device-code error
    infra     the sanitizer did not produce a summary (timeout, crash,
              missing binary), so the run says nothing about the kernel

Host CUDA API errors ("Program hit ..."), which memcheck reports for calls such
as Triton's launcher probing ``cuFuncSetAttribute``, and launches the sanitizer
could not track ("Internal Sanitizer Error", such as cuBLAS kernels), are
counted separately and do not make a run "findings".

"none" is not proof that the kernel is race-free: the sanitizer only checks
what the test executed, and racecheck only covers shared memory.

Results go to ``<out>/results.json`` and, as markdown, to
``$GITHUB_STEP_SUMMARY`` when set. Exit status: 0 normally, 1 with
``--fail-on-findings`` when any tool has findings, 2 when any tool is infra.

With ``--issue-title-prefix``, the run's findings also go to
``$GITHUB_OUTPUT`` as ``failures=<json>``, and the issue title prefixes the run
checked as ``reconcile_prefixes=<json>``, for the issue workflows (see
findings.py).

Usage:
    sanitize.py --tools racecheck,synccheck --kernel _attn_fwd_ws \\
        --out /tmp/sanitize -- python -m pytest path/to/test.py::test_x[id]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from findings import failure_list, reconcile_prefixes
from job_summary import render
from sanitizer_log import classify, device_reports, racecheck_findings, report_lines
from source_paths import compiled_kernels, resolve_paths

TOOLS = ("racecheck", "synccheck", "memcheck", "initcheck")


def is_test(arg: str) -> bool:
    return arg.split("::")[0].endswith(".py")


def split_by_file(cmd: list[str]) -> list[list[str]]:
    """``cmd`` once per test file it names, or once per test case named on its own."""
    tests = [arg for arg in cmd if is_test(arg)]
    groups: dict[str, list[str]] = {}
    for test in tests:
        groups.setdefault(test if "[" in test else test.split("::")[0], []).append(test)
    if len(groups) <= 1:
        return [cmd]
    first = cmd.index(tests[0])
    rest = [arg for arg in cmd[first:] if not is_test(arg)]
    return [cmd[:first] + group + rest for group in groups.values()]


def run_tool(tool: str, cmd: list[str], out: Path, timeout_s: int, triton_cache: Path) -> dict:
    tool_dir = out / tool
    argv = ["cutracer", "compute-sanitizer", "--tool", tool, "-o", str(tool_dir), "--", *cmd]
    print(f"::group::{tool}: {' '.join(argv)}", flush=True)
    # Its own session, so the tests it started are stopped with it: the next
    # run must not share the GPU. A sanitizer that crashes can leave its
    # target running, so this applies to every exit, not only a timeout.
    proc = subprocess.Popen(argv, start_new_session=True)
    try:
        returncode, timed_out = proc.wait(timeout=max(timeout_s, 1)), False
    except subprocess.TimeoutExpired:
        returncode, timed_out = None, True
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    print("::endgroup::", flush=True)
    log_path = tool_dir / f"{tool}.log"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    result = {"tool": tool, "cmd": cmd, "returncode": returncode, "timed_out": timed_out, "log": str(log_path)}
    result.update(classify(tool, log_text) if not timed_out else {"outcome": "infra", "count": None})
    result["report"] = report_lines(log_text)
    result["reports"] = device_reports(log_text) if result["outcome"] == "findings" else []
    resolve_paths(result["reports"], triton_cache)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # racecheck is opt-in: on mbarrier-synchronized warp-specialized kernels it
    # reports false-positive hazards and can run for over 30 minutes on a
    # seconds-long test.
    parser.add_argument("--tools", default="synccheck,memcheck")
    parser.add_argument("--kernel", default=None, help="Kernel name used to attribute racecheck findings.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Per tool and test file.")
    parser.add_argument("--deadline-minutes", type=int, default=None,
                        help="For all runs; files left when it passes are reported as not run.")
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--issue-title-prefix", default=None,
                        help="Write the findings to $GITHUB_OUTPUT as failures=, titled '<prefix> <kernel> in <file>'.")
    parser.add_argument("--issue-job", default="sanitize", help="Job name the issues report.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("missing the command to run after --")
    tools = [tool.strip() for tool in args.tools.split(",") if tool.strip()]
    unknown = sorted(set(tools) - set(TOOLS))
    if unknown:
        parser.error(f"unknown tools: {', '.join(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    # Stopping before the step's own limit leaves time to publish what ran.
    deadline = time.monotonic() + args.deadline_minutes * 60 if args.deadline_minutes else None
    results = []
    clean: list[str] = []
    unfinished: set[str] = set()
    for i, group in enumerate(split_by_file(cmd)):
        out = args.out if i == 0 else args.out / f"group-{i}"
        # A fresh cache holds exactly the kernels this group compiled, and so
        # launched under the sanitizer; their IR also maps a report's basename
        # to its path.
        triton_cache = out / "triton-cache"
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        group_results = []
        for tool in tools:
            timeout_s = args.timeout_minutes * 60
            if deadline is not None:
                timeout_s = min(timeout_s, int(deadline - time.monotonic()))
            if timeout_s <= 0:
                group_results.append({
                    "tool": tool, "cmd": group, "returncode": None, "timed_out": True, "outcome": "infra", "count":
                    None, "report": [], "reports": []
                })
                continue
            result = run_tool(tool, group, out, timeout_s, triton_cache)
            if tool == "racecheck" and result["outcome"] == "findings":
                result["findings"] = racecheck_findings(Path(result["log"]).read_text(errors="replace"), args.kernel)
            group_results.append(result)
        # A kernel counts as checked only if every tool ran the tests to the end,
        # in every group that compiled it: the cache is shared across tools, and
        # a sanitizer abort leaves a sticky CUDA error that can stop later tests
        # from launching their kernels.
        kernels = compiled_kernels(triton_cache)
        for result in group_results:
            result["kernels"] = kernels
        if all(result["returncode"] == 0 and result["outcome"] != "infra" for result in group_results):
            clean += [kernel for kernel in kernels if kernel not in clean]
        else:
            unfinished.update(kernels)
        results += group_results
    checked = [kernel for kernel in clean if kernel not in unfinished]

    (args.out / "results.json").write_text(json.dumps({"kernel": args.kernel, "cmd": cmd, "results": results},
                                                      indent=2))
    summary = render(results, args.kernel, cmd)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(summary)
    if args.issue_title_prefix:
        outputs = {
            "failures": failure_list(results, args.issue_title_prefix, args.issue_job),
            "reconcile_prefixes": reconcile_prefixes(checked, tools, args.issue_title_prefix),
        }
        for key, value in outputs.items():
            print(f"{key}={json.dumps(value)}")
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
                    handle.write(f"{key}={json.dumps(value)}\n")

    if any(result["outcome"] == "infra" for result in results):
        return 2
    if args.fail_on_findings and any(result["outcome"] == "findings" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
