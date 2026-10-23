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

Usage:
    cutracer_sanitize.py --tools racecheck,synccheck --kernel _attn_fwd_ws \\
        --out /tmp/sanitize -- python -m pytest path/to/test.py::test_x[id]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS = ("racecheck", "synccheck", "memcheck", "initcheck")
_ERROR_SUMMARY_RE = re.compile(r"^=+ ERROR SUMMARY: (\d+) errors?", re.MULTILINE)
_RACECHECK_SUMMARY_RE = re.compile(
    r"^=+ RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)",
    re.MULTILINE,
)
# First lines of each reported error, for the summary. Compute Sanitizer
# prefixes every report line with "========= "; continuation lines are
# indented further, report headers are not.
_REPORT_LINE_RE = re.compile(r"^=+\s+(?!ERROR SUMMARY|RACECHECK SUMMARY|COMPUTE-SANITIZER)(\S.*)$", re.MULTILINE)
_REPORT_HEADER_RE = re.compile(r"^=+ (?!ERROR SUMMARY|RACECHECK SUMMARY|COMPUTE-SANITIZER|Target application)(\S.*)$",
                               re.MULTILINE)
_API_ERROR_PREFIX = "Program hit CUDA_ERROR_"
MAX_REPORT_LINES = 20


def classify(tool: str, log_text: str) -> dict:
    """Return the outcome and counts that ``log_text`` supports for ``tool``."""
    if tool == "racecheck":
        match = _RACECHECK_SUMMARY_RE.search(log_text)
        if match:
            hazards, errors, warnings = (int(group) for group in match.groups())
            return {
                "outcome": "findings" if hazards else "none",
                "count": hazards,
                "errors": errors,
                "warnings": warnings,
            }
    match = _ERROR_SUMMARY_RE.search(log_text)
    if match is None:
        return {"outcome": "infra", "count": None}
    api_errors = sum(header.startswith(_API_ERROR_PREFIX) for header in _REPORT_HEADER_RE.findall(log_text))
    count = max(int(match.group(1)) - api_errors, 0)
    return {"outcome": "findings" if count else "none", "count": count, "api_errors": api_errors}


def report_lines(log_text: str, limit: int = MAX_REPORT_LINES) -> list[str]:
    return [match.group(1) for match in _REPORT_LINE_RE.finditer(log_text)][:limit]


def racecheck_findings(log_text: str, kernel: str | None) -> list[dict]:
    """Racecheck hazards from CUTracer's parser, flagged when in ``kernel``."""
    from cutracer.compute_sanitizer.parser import parse_racecheck_log

    findings = []
    for finding in parse_racecheck_log(log_text).findings:
        accesses = [{
            "role": access.role,
            "type": access.access_type,
            "function": access.function,
            "location": f"{access.file}:{access.line}",
        } for access in finding.accesses]
        findings.append({
            "severity": finding.severity.value,
            "in_kernel": bool(kernel) and any(kernel in access["function"] for access in accesses),
            "accesses": accesses,
        })
    return findings


def run_tool(tool: str, cmd: list[str], out: Path, timeout_s: int) -> dict:
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
    return result


def markdown(results: list[dict], kernel: str | None, cmd: list[str]) -> str:
    lines = [
        "## CUTracer compute-sanitizer",
        "",
        f"Kernel: `{kernel or 'all'}`  ",
        f"Command: `{' '.join(cmd)}`",
        "",
        "| Tool | Outcome | Device errors | Host API errors | Test exit |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        count = "-" if result["count"] is None else result["count"]
        exit_code = "timeout" if result["timed_out"] else result["returncode"]
        api_errors = result.get("api_errors", "-")
        lines.append(f"| {result['tool']} | {result['outcome']} | {count} | {api_errors} | {exit_code} |")
    lines += ["", "`none` means the sanitizer reported nothing for what the test ran; it is not proof of no race."]
    for result in results:
        if result["outcome"] != "findings":
            continue
        lines += ["", f"### {result['tool']}", "", "```"]
        lines += result["report"] or ["(no report lines parsed; see the uploaded log)"]
        lines += ["```"]
        for finding in result.get("findings", []):
            marker = "in kernel" if finding["in_kernel"] else "other kernel"
            sites = "; ".join(f"{a['role']} {a['type']} {a['function']} {a['location']}" for a in finding["accesses"])
            lines.append(f"- {finding['severity']} ({marker}): {sites}")
    return "\n".join(lines) + "\n"


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
    results = []
    for tool in tools:
        result = run_tool(tool, cmd, args.out, args.timeout_minutes * 60)
        if tool == "racecheck" and result["outcome"] == "findings":
            result["findings"] = racecheck_findings(Path(result["log"]).read_text(errors="replace"), args.kernel)
        results.append(result)

    (args.out / "results.json").write_text(json.dumps({"kernel": args.kernel, "cmd": cmd, "results": results},
                                                      indent=2))
    summary = markdown(results, args.kernel, cmd)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(summary)

    if any(result["outcome"] == "infra" for result in results):
        return 2
    if args.fail_on_findings and any(result["outcome"] == "findings" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
