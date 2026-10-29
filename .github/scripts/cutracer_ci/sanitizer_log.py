"""Parse NVIDIA Compute Sanitizer logs into outcomes and per-report locations."""

from __future__ import annotations

import re

_ERROR_SUMMARY_RE = re.compile(r"^=+ ERROR SUMMARY: (\d+) errors?", re.MULTILINE)
# Errors past --print-limit (default 100): counted in ERROR SUMMARY, but
# their kind and kernel are unknown.
_NOT_PRINTED_RE = re.compile(r"^=+ ERROR SUMMARY: (\d+) errors? were not printed", re.MULTILINE)
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
# Errors outside the kernel under test, which Compute Sanitizer still counts
# in ERROR SUMMARY: a host CUDA API call that failed (driver "CUDA_ERROR_..."
# or runtime "cudaError...", such as out of memory under memcheck), and a
# launch the sanitizer could not track, such as cuBLAS's nvjet kernels.
_API_ERROR_PREFIX = "Program hit "
_SANITIZER_ERROR_PREFIX = "Internal Sanitizer Error"
# memcheck flags a TMA load that signals an mbarrier in the peer CTA, but the
# PTX ISA allows that under .cta_group::2 (cp.async.bulk.tensor). NVIDIA's own
# 2-CTA cuBLAS kernels get the same report.
_PEER_BARRIER = "is not located in destination CTA"
# A report line split into its "=" prefix, indent and text.
_LINE_RE = re.compile(r"^=+( *)(.*)$")
# Faulting device PC: "at <kernel>+0x<offset>", plus " in <file>:<line>" when
# the kernel was built with line info. "Device Frame:" lines have no "at".
_LOCATION_RE = re.compile(r"\bat (\S+?)\+0x[0-9a-fA-F]+(?: in (\S+?):(\d+))?")
# Host backtrace lines: long, and they say nothing about the kernel.
_HOST_LINE = ("Host Frame:", "Saved host backtrace")
_NOT_A_REPORT = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "Target application",
                 "Warning: Maximum number of hazards", _API_ERROR_PREFIX, _SANITIZER_ERROR_PREFIX)
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
    headers = _REPORT_HEADER_RE.findall(log_text)
    api_errors = sum(header.startswith(_API_ERROR_PREFIX) for header in headers)
    sanitizer_errors = sum(header.startswith(_SANITIZER_ERROR_PREFIX) for header in headers)
    peer_barrier = sum(_is_peer_barrier(report) for report in _reports(log_text))
    not_printed = sum(int(n) for n in _NOT_PRINTED_RE.findall(log_text))
    count = max(int(match.group(1)) - api_errors - sanitizer_errors - peer_barrier - not_printed, 0)
    return {
        "outcome": "findings" if count else "none",
        "count": count,
        "api_errors": api_errors,
        "sanitizer_errors": sanitizer_errors,
        "peer_barrier": peer_barrier,
        "not_printed": not_printed,
    }


def _is_peer_barrier(report: dict) -> bool:
    return any(_PEER_BARRIER in line for line in report["lines"])


def device_reports(log_text: str) -> list[dict]:
    """One entry per device-code report: its header, first faulting location and text.

    ``lines`` is the report as the sanitizer printed it, without the host
    backtrace. ``kernel``, ``file`` and ``line`` are None when the log does not
    give them; ``file`` is a basename, and source_paths.resolve_paths fills in
    ``path``. Host API errors, untracked launches, peer-CTA barrier signals and
    the sanitizer's own banners are not reports.
    """
    return [report for report in _reports(log_text) if not _is_peer_barrier(report)]


def _reports(log_text: str) -> list[dict]:
    reports = []
    current = None
    for raw in log_text.splitlines():
        match = _LINE_RE.match(raw)
        if match is None:
            continue  # target output interleaved with the sanitizer's
        indent, text = match.groups()
        if not text.strip():
            current = None
            continue
        if len(indent) <= 1:
            current = None
            if text.startswith(_NOT_A_REPORT):
                continue
            current = {"header": text.strip(), "kernel": None, "file": None, "line": None, "path": None, "lines": []}
            reports.append(current)
        if current is not None and not text.lstrip().startswith(_HOST_LINE):
            current["lines"].append(" " * max(len(indent) - 1, 0) + text.rstrip())
        if current is not None and current["kernel"] is None:
            location = _LOCATION_RE.search(text)
            if location:
                kernel, file, line = location.groups()
                current.update(kernel=kernel, file=file, line=int(line) if line else None)
    return reports


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
