"""The failure list report-nightly-failure.yml and reconcile-nightly-issues.yml take.

One entry per kernel and source file collects the reports of every tool, so
the issue title is the dedupe key, e.g.
``[CUTracer][B200] _attn_fwd_ws in triton/tlx/ops/kernels/flash_attn/sm100.py``.
A tool that did not finish adds a fallback entry, which tells reconcile the
list is incomplete and must not close anything. The entry format is
documented in parse_junit_failures.py.

reconcile closes every open issue under its title prefix that the list lacks,
so a run may only close issues for what it checked: one prefix per kernel it
checked under every tool, and one per tool it ran (for that tool's
"incomplete" issue).
"""

from __future__ import annotations

import shlex

# GitHub caps issue titles at 256 characters; C++ kernel names can be longer.
MAX_KERNEL_CHARS = 120
# An issue summary carries the sanitizer's own reports; keep it readable.
MAX_SUMMARY_CHARS = 4000


def kernel_label(kernel: str) -> str:
    return kernel[:MAX_KERNEL_CHARS] + "..." if len(kernel) > MAX_KERNEL_CHARS else kernel


def finding_id(report: dict) -> str:
    """The kernel and source file a report is against: the issue's dedupe key."""
    kernel = report.get("kernel")
    if not kernel:
        return "unattributed"
    file = report.get("path") or report.get("file")
    return f"{kernel_label(kernel)} in {file}" if file else kernel_label(kernel)


def describe(tool: str, report: dict) -> str:
    where = ""
    if report.get("file"):
        where = f" at {report['file']}:{report['line']}" if report.get("line") else f" in {report['file']}"
    elif report.get("kernel"):
        where = f" in {report['kernel']}"
    return f"{tool}: {report['header']}{where}"


def failure_entry(finding: str, title_prefix: str, job: str, summary: str, repro: str, fallback: bool) -> dict:
    return {
        "normalized_failure_id": finding,
        "raw_failure_ids": "",
        "issue_title": f"{title_prefix} {finding}",
        "job_name": job,
        "summary": summary,
        "repro": repro,
        "fallback": fallback,
        # A sanitizer finding is a property of the kernel, not of the runner.
        "external_dep": False,
        "external_dep_reason": "",
    }


def add_to_summary(entry: dict, tool: str, lines: list[str]) -> None:
    """Append one distinct report's text to the issue summary, within the cap."""
    block = "\n".join([f"[{tool}]"] + lines)
    if not entry["summary"]:
        entry["summary"] = block[:MAX_SUMMARY_CHARS]
        return
    summary = f"{entry['summary']}\n\n{block}"
    if len(summary) <= MAX_SUMMARY_CHARS:
        entry["summary"] = summary
    elif "(more reports in the run's logs)" not in entry["summary"]:
        entry["summary"] += "\n\n(more reports in the run's logs)"


def failure_list(results: list[dict], title_prefix: str, job: str) -> list[dict]:
    """The run's findings, one entry per kernel and source file, for the issue workflows."""
    entries: dict[str, dict] = {}
    for result in results:
        tool = result["tool"]
        repro = f"cutracer compute-sanitizer --tool {tool} -- {shlex.join(result['cmd'])}"
        if result["outcome"] == "infra":
            why = "timed out" if result["timed_out"] else "produced no summary"
            finding = f"{tool} incomplete"
            entries[finding] = failure_entry(finding, title_prefix, job,
                                             f"compute-sanitizer {tool} {why} (see run log).", repro, True)
            continue
        if result["outcome"] != "findings":
            continue
        for report in result["reports"] or [{"header": f"{result['count']} error(s)"}]:
            # A report in a kernel the run did not compile is in a library the
            # test called, such as cuDNN for a reference result, not in Triton.
            if report.get("kernel") and "kernels" in result and report["kernel"] not in result["kernels"]:
                continue
            finding, line = finding_id(report), describe(tool, report)
            entry = entries.setdefault(finding, failure_entry(finding, title_prefix, job, "", repro, False))
            raw = entry["raw_failure_ids"].split("\n") if entry["raw_failure_ids"] else []
            if line not in raw:
                entry["raw_failure_ids"] = "\n".join(raw + [line])
                add_to_summary(entry, tool, report.get("lines") or [line])
    return list(entries.values())


def reconcile_prefixes(kernels: list[str], tools: list[str], title_prefix: str) -> list[str]:
    """The issue title prefixes this run checked, for reconcile to close within."""
    return [f"{title_prefix} {kernel_label(kernel)} in "
            for kernel in kernels] + [f"{title_prefix} {tool} incomplete" for tool in tools]
