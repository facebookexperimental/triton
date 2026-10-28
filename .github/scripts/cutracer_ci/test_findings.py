from findings import failure_list, reconcile_prefixes

FA_PATH = "triton/tlx/ops/kernels/flash_attn/sm100.py"

CMD = ["python", "-m", "pytest", "test_fa.py::test_fwd[1-fp16]"]


def _report(header, kernel="_attn_fwd_ws", file="sm100.py", line=1107, path=FA_PATH):
    lines = [header, f"    at {kernel}+0x10 in {file}:{line}", "    by thread (384,0,0) in block (0,0,0)"]
    return {"header": header, "kernel": kernel, "file": file, "line": line, "path": path, "lines": lines}


def _found(tool, *reports):
    return {"tool": tool, "outcome": "findings", "count": len(reports), "timed_out": False, "reports": list(reports)}


def _failures(*results):
    return failure_list([{"cmd": CMD, **result} for result in results], "[CUTracer][B200]", "sanitize-b200")


def test_reports_against_one_kernel_share_an_entry_across_tools():
    (entry, ) = _failures(
        _found("synccheck", _report("Barrier error detected. Missing init."),
               _report("Barrier error detected. Missing wait.", line=812),
               _report("Barrier error detected. Missing init.")),
        _found("memcheck", _report("Invalid __shared__ write of size 4 bytes", line=900)))
    assert entry["normalized_failure_id"] == f"_attn_fwd_ws in {FA_PATH}"
    assert entry["issue_title"] == f"[CUTracer][B200] _attn_fwd_ws in {FA_PATH}"
    assert entry["raw_failure_ids"].split("\n") == [
        "synccheck: Barrier error detected. Missing init. at sm100.py:1107",
        "synccheck: Barrier error detected. Missing wait. at sm100.py:812",
        "memcheck: Invalid __shared__ write of size 4 bytes at sm100.py:900",
    ]
    assert entry["summary"].split("\n\n")[0] == "\n".join([
        "[synccheck]",
        "Barrier error detected. Missing init.",
        "    at _attn_fwd_ws+0x10 in sm100.py:1107",
        "    by thread (384,0,0) in block (0,0,0)",
    ])
    assert entry["summary"].count("[memcheck]") == 1
    assert entry["repro"] == ("cutracer compute-sanitizer --tool synccheck -- "
                              "python -m pytest 'test_fa.py::test_fwd[1-fp16]'")
    assert entry["fallback"] is False and entry["external_dep"] is False


def test_kernel_and_directory_split_entries():
    mm = "triton/tlx/ops/kernels/mm/sm100.py"
    entries = _failures(
        _found("synccheck", _report("A"), _report("A", kernel="_attn_bwd"), _report("A", kernel="_gemm_ws", path=mm)))
    assert [e["normalized_failure_id"] for e in entries] == [
        f"_attn_fwd_ws in {FA_PATH}",
        f"_attn_bwd in {FA_PATH}",
        f"_gemm_ws in {mm}",
    ]


def test_unresolved_path_falls_back_to_basename_then_kernel_then_unattributed():
    entries = _failures(
        _found("synccheck", _report("A", path=None), _report("B", file=None, line=None, path=None),
               _report("C", kernel=None, path=None)))
    assert [e["normalized_failure_id"] for e in entries] == ["_attn_fwd_ws in sm100.py", "_attn_fwd_ws", "unattributed"]
    assert entries[1]["raw_failure_ids"] == "synccheck: B in _attn_fwd_ws"


# A sanitizer report in a kernel the run did not compile is in a library the
# test called (cuDNN computing a reference result), not in a Triton kernel.
def test_reports_in_kernels_the_run_did_not_compile_are_not_filed():
    result = _found("synccheck", _report("Divergent", kernel="cudnn_sdpa_bprop"), _report("Missing init"))
    result["kernels"] = ["_attn_fwd_ws"]
    assert [e["normalized_failure_id"] for e in _failures(result)] == ["_attn_fwd_ws in " + FA_PATH]


def test_summary_is_capped():
    (entry, ) = _failures(_found("racecheck", *[_report(f"Error {i}", line=i) for i in range(200)]))
    assert len(entry["summary"]) <= 4000 + 40
    assert entry["summary"].endswith("(more reports in the run's logs)")
    assert len(entry["raw_failure_ids"].split("\n")) == 200
    huge = _report("Error", line=1)
    huge["lines"] = ["x" * 100] * 100
    assert len(_failures(_found("racecheck", huge))[0]["summary"]) == 4000


def test_long_kernel_names_are_truncated():
    assert len(_failures(_found("synccheck", _report("A", kernel="k" * 300)))[0]["issue_title"]) < 256


def test_clean_and_incomplete_tools():
    clean = {"tool": "memcheck", "outcome": "none", "count": 0, "timed_out": False, "reports": []}
    timed_out = {"tool": "racecheck", "outcome": "infra", "count": None, "timed_out": True, "reports": []}
    assert _failures(clean) == []
    (entry, ) = _failures(clean, timed_out)
    assert entry["issue_title"] == "[CUTracer][B200] racecheck incomplete"
    assert entry["fallback"] is True
    assert "timed out" in entry["summary"]


def test_reconcile_prefixes_cover_checked_kernels_and_tools():
    assert reconcile_prefixes(["_attn_fwd_ws", "k" * 300], ["synccheck", "memcheck"], "[CUTracer][B200]") == [
        "[CUTracer][B200] _attn_fwd_ws in ",
        f"[CUTracer][B200] {'k' * 120}... in ",
        "[CUTracer][B200] synccheck incomplete",
        "[CUTracer][B200] memcheck incomplete",
    ]
    # Each prefix starts the title of the issues it may close.
    (entry, ) = _failures(_found("synccheck", _report("A", kernel="k" * 300)))
    assert entry["issue_title"].startswith(f"[CUTracer][B200] {'k' * 120}... in ")
