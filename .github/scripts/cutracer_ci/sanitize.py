#!/usr/bin/env python3
"""Run a test under NVIDIA Compute Sanitizer through CUTracer and summarize it.

For each requested tool, runs ``cutracer compute-sanitizer --tool <tool>`` on
the given command, then classifies the run from the sanitizer log:

    findings  the sanitizer reported at least one error or hazard in device
              code
    none      the sanitizer finished and reported no device-code error
    infra     the sanitizer did not produce a summary (timeout, crash,
              missing binary), so the run says nothing about the kernel

Host CUDA API errors ("Program hit CUDA_ERROR_..."), which memcheck reports
for calls such as Triton's launcher probing ``cuFuncSetAttribute``, are
counted separately as ``api_errors`` and do not make a run "findings".

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
import subprocess
import sys
from pathlib import Path

from findings import failure_list, reconcile_prefixes
from job_summary import render
from sanitizer_log import classify, device_reports, racecheck_findings, report_lines
from source_paths import compiled_kernels, resolve_paths

TOOLS = ("racecheck", "synccheck", "memcheck", "initcheck")


def run_tool(tool: str, cmd: list[str], out: Path, timeout_s: int, triton_cache: Path) -> dict:
    tool_dir = out / tool
    argv = ["cutracer", "compute-sanitizer", "--tool", tool, "-o", str(tool_dir), "--", *cmd]
    print(f"::group::{tool}: {' '.join(argv)}", flush=True)
    try:
        returncode = subprocess.run(argv, timeout=timeout_s, check=False).returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode, timed_out = None, True
    print("::endgroup::", flush=True)
    log_path = tool_dir / f"{tool}.log"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    result = {"tool": tool, "returncode": returncode, "timed_out": timed_out, "log": str(log_path)}
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
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Per tool.")
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
    # A fresh cache holds exactly the kernels this run compiled, and so launched
    # under the sanitizer; their IR also maps a report's basename to its path.
    triton_cache = args.out / "triton-cache"
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    results = []
    for tool in tools:
        result = run_tool(tool, cmd, args.out, args.timeout_minutes * 60, triton_cache)
        if tool == "racecheck" and result["outcome"] == "findings":
            result["findings"] = racecheck_findings(Path(result["log"]).read_text(errors="replace"), args.kernel)
        results.append(result)

    (args.out / "results.json").write_text(json.dumps({"kernel": args.kernel, "cmd": cmd, "results": results},
                                                      indent=2))
    summary = render(results, args.kernel, cmd)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(summary)
    if args.issue_title_prefix:
        # A kernel counts as checked only if every tool ran the tests to the end:
        # the cache is shared across tools, and a sanitizer abort leaves a sticky
        # CUDA error that can stop later tests from launching their kernels.
        clean = all(result["returncode"] == 0 for result in results)
        checked = compiled_kernels(triton_cache) if clean else []
        outputs = {
            "failures": failure_list(results, cmd, args.issue_title_prefix, args.issue_job),
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
