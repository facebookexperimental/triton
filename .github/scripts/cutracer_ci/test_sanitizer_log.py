import pytest

import sanitizer_log
from sample_logs import (MEMCHECK_API_ONLY, MEMCHECK_MIXED, OUTSIDE_THE_KERNEL, PEER_BARRIER, RACECHECK_CLEAN,
                         RACECHECK_HAZARD, SYNCCHECK_CLEAN, SYNCCHECK_ERROR)
from sanitizer_log import classify, device_reports, report_lines


def test_racecheck_hazards_are_findings():
    assert classify("racecheck", RACECHECK_HAZARD) == {"outcome": "findings", "count": 7, "errors": 6, "warnings": 1}


def test_racecheck_clean_is_none():
    assert classify("racecheck", RACECHECK_CLEAN)["outcome"] == "none"


def _counts(outcome, count, api_errors, sanitizer_errors, peer_barrier=0, not_printed=0):
    return dict(outcome=outcome, count=count, api_errors=api_errors, sanitizer_errors=sanitizer_errors,
                peer_barrier=peer_barrier, not_printed=not_printed)


def test_error_summary_tools():
    assert classify("synccheck", SYNCCHECK_ERROR) == _counts("findings", 1, 0, 0)
    assert classify("memcheck", SYNCCHECK_CLEAN) == _counts("none", 0, 0, 0)


# A finding is an error in device code the test launched. Errors the sanitizer
# also counts but that lie outside that code -- a failed host API call, a launch
# it could not track -- must not become issues against a kernel.
def test_errors_outside_the_kernel_are_not_device_findings():
    assert classify("memcheck", MEMCHECK_API_ONLY) == _counts("none", 0, 2, 0)
    assert classify("memcheck", MEMCHECK_MIXED) == _counts("findings", 1, 2, 0)
    assert classify("synccheck", OUTSIDE_THE_KERNEL) == _counts("none", 0, 1, 1)
    assert device_reports(OUTSIDE_THE_KERNEL) == []


# The PTX ISA lets a .cta_group::2 TMA load signal an mbarrier in the peer CTA;
# memcheck reports it anyway. Errors past the print limit are not findings
# either: their kind is unknown (sanitize.py keeps their kernels unchecked).
def test_peer_cta_barrier_signals_are_not_findings():
    assert classify("memcheck", PEER_BARRIER) == _counts("none", 0, 0, 0, peer_barrier=1, not_printed=2369)
    assert device_reports(PEER_BARRIER) == []


@pytest.mark.parametrize("log", ["", "========= COMPUTE-SANITIZER\n", "Segmentation fault\n"])
def test_missing_summary_is_infra(log):
    assert classify("synccheck", log)["outcome"] == "infra"
    assert classify("racecheck", log)["outcome"] == "infra"


def test_report_lines_keep_reports_and_drop_summaries():
    lines = report_lines(RACECHECK_HAZARD)
    assert lines[0].startswith("Error: Race reported between Write access at kernel_a+0x100")
    assert "and Read access at kernel_a+0x200 in kernels.py:57 [3 hazards]" in lines
    assert not any("SUMMARY" in line or line.startswith("COMPUTE-SANITIZER") for line in lines)


def test_device_reports_locate_each_report():
    assert device_reports(SYNCCHECK_ERROR) == [{
        "header":
        "Barrier error detected. Missing wait.", "kernel":
        "_attn_fwd_ws", "file":
        "sm100.py", "line":
        812, "path":
        None, "lines": ["Barrier error detected. Missing wait.", "    at _attn_fwd_ws+0x1a40 in sm100.py:812"]
    }]
    # Racecheck puts the location on the header line; interleaved target output
    # and the hazard-limit warning are not reports.
    assert [(r["kernel"], r["file"], r["line"]) for r in device_reports(RACECHECK_HAZARD)] == [
        ("kernel_a", "kernels.py", 42),
        ("kernel_b", "other.py", 9),
    ]


def test_device_reports_skip_host_api_errors():
    assert [(r["header"], r["kernel"]) for r in device_reports(MEMCHECK_MIXED)] == [
        ("Invalid __global__ read of size 4 bytes", "kernel"),
    ]
    assert device_reports(MEMCHECK_API_ONLY) == []


def test_device_reports_without_line_info():
    log = "========= Barrier error detected. Missing init.\n=========     at kern+0x10\n========= ERROR SUMMARY: 1 error\n"
    assert device_reports(log) == [{
        "header": "Barrier error detected. Missing init.", "kernel": "kern", "file": None, "line": None, "path": None,
        "lines": ["Barrier error detected. Missing init.", "    at kern+0x10"]
    }]


def test_device_reports_keep_report_text_without_host_backtrace():
    # synccheck on the B200 runner, tlx.ops flash-attention forward (2-CTA config).
    log = """\
========= COMPUTE-SANITIZER
========= Barrier error detected. Missing init.
=========     at _attn_fwd_ws+0x6660 in sm100.py:1107
=========     by thread (384,0,0) in block (0,0,0)
=========     Barrier is located at shared address 0x1018720
=========         Device Frame: _attn_fwd_ws+0x240 in sm100.py:626
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: cuLaunchKernelEx [0x2d1a5] in libcuda.so.1
=========
========= ERROR SUMMARY: 1 error
"""
    (report, ) = device_reports(log)
    assert report["lines"] == [
        "Barrier error detected. Missing init.",
        "    at _attn_fwd_ws+0x6660 in sm100.py:1107",
        "    by thread (384,0,0) in block (0,0,0)",
        "    Barrier is located at shared address 0x1018720",
        "        Device Frame: _attn_fwd_ws+0x240 in sm100.py:626",
    ]


def test_racecheck_findings_attribute_kernel():
    pytest.importorskip("cutracer.compute_sanitizer.parser")
    findings = sanitizer_log.racecheck_findings(RACECHECK_HAZARD, "kernel_a")
    assert [finding["in_kernel"] for finding in findings] == [True, False]
    assert findings[0]["accesses"][0]["location"] == "kernels.py:42"
