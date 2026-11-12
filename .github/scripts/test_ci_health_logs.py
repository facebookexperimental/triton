import ci_health_logs as lg
import ci_health_sanitizer as sanitizer
import io
import json
import zipfile
import pytest


def _sanitizer_report(pc, thread):
    return ("========= Uninitialized __global__ memory read of size 8 bytes\n"
            f"=========     at copy_kernel+0x{pc}\n"
            f"=========     by thread ({thread},0,0) in block (1,0,0)\n"
            f"=========     Address 0x{1000 + thread:x}\n========= \n")


def test_sanitizer_dedup_ignores_threads_but_keeps_faulting_sites():
    text = _sanitizer_report("d0", 0) + _sanitizer_report("d0", 1) + _sanitizer_report("e0", 2)
    text += "========= ERROR SUMMARY: 3 errors\n"
    result = sanitizer.read_log(text.splitlines())
    assert result["raw_count"] == 3
    assert [signal["count"] for signal in result["signals"]] == [2, 1]
    assert [signal["sites"] for signal in result["signals"]] == [["at copy_kernel+0xd0"], ["at copy_kernel+0xe0"]]


@pytest.mark.parametrize("text,code,timeout,complete", [
    ("========= COMPUTE-SANITIZER\n", 0, False, False),
    ("========= ERROR SUMMARY: 0 errors\n", 0, True, False),
    ("========= ERROR SUMMARY: 0 errors\n", 1, False, False),
    ("========= ERROR SUMMARY: 5 errors were not printed\n========= ERROR SUMMARY: 5 errors\n", 0, False, False),
    ("========= ERROR SUMMARY: 0 errors\n", 0, False, True),
])
def test_sanitizer_completion_requires_a_complete_log_and_test(text, code, timeout, complete):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "sanitize/results.json",
            json.dumps({
                "cmd": ["python", "repro.py"], "results": [{
                    "tool":
                    "memcheck",
                    "returncode":
                    code,
                    "timed_out":
                    timeout,
                    "log":
                    "/runner/cutracer/sanitize/group-0/memcheck/memcheck.log",
                }]
            }))
        archive.writestr("sanitize/group-0/memcheck/memcheck.log", text)
    row = sanitizer.archive_data(buffer.getvalue(), {"name": "cutracer-sanitizer-b200"})[0]
    assert (row["completion"] == "complete") == complete
    if "ERROR SUMMARY" not in text:
        assert row["raw_count"] is None


def test_sanitizer_missing_group_cannot_borrow_another_groups_log():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "results.json",
            json.dumps({
                "results": [{
                    "tool": "memcheck",
                    "returncode": 0,
                    "timed_out": False,
                    "log": "/runner/cutracer/group-99/memcheck/memcheck.log",
                }]
            }))
        archive.writestr("memcheck/memcheck.log", "========= ERROR SUMMARY: 0 errors\n")
    row = sanitizer.archive_data(buffer.getvalue(), {"name": "cutracer-sanitizer-b200"})[0]
    assert row["log"] is None
    assert row["raw_count"] is None
    assert row["completion"] == "incomplete"


def test_racecheck_dedup_preserves_access_sites_not_actor_or_hazard_counts():
    report = ("========= Error: Race reported between Write access at k+0x100 in k.py:4\n"
              "=========     and Read access at k+0x200 in k.py:5 [3 hazards]\n========= \n")
    second = report.replace("[3 hazards]", "[5 hazards]").replace("k.py:5", "k.py:5 by thread (1,0,0)")
    parsed = sanitizer.read_log(
        (report + second + "========= RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings)\n").splitlines())
    assert len(parsed["signals"]) == 1
    assert parsed["signals"][0]["count"] == 2
    assert parsed["signals"][0]["sites"] == [
        "Error: Race reported between Write access at k+0x100 in k.py:4",
        "and Read access at k+0x200 in k.py:5",
    ]


def test_racecheck_cutoff_keeps_omissions_unknown():
    parsed = sanitizer.read_log(
        ("========= Warning: Maximum number of hazards reached.\n"
         "========= RACECHECK SUMMARY: 7 hazards displayed (6 errors, 1 warning)\n").splitlines())
    assert parsed["truncated"] is True
    assert parsed["not_printed"] is None
    assert parsed["signals"] == []


def test_sanitizer_unsummarized_tail_cannot_reuse_an_earlier_summary():
    parsed = sanitizer.read_log(("========= ERROR SUMMARY: 0 errors\n" + _sanitizer_report("d0", 0)).splitlines())
    assert parsed["raw_count"] is None
    assert parsed["truncated"] is True
    assert len(parsed["signals"]) == 1


@pytest.mark.parametrize("tests,op", [
    (["python/test/unit/tlx_ops/test_mm_sm100.py::test_mm[256]"], "mm"),
    (["python/test/unit/tlx_ops/test_mm_sm100.py", "python/test/unit/tlx_ops/test_mm_sm90.py"], "mm"),
    (["python/test/unit/tlx_ops/test_mm_sm100.py", "python/test/unit/tlx_ops/test_addmm_sm100.py"], "Mixed op suites"),
    (["python/test/unit/language/test_autows_addmm.py"], "Unattributed / compiler tests"),
])
def test_sanitizer_op_group_follows_the_suite_without_guessing(tests, op):
    assert sanitizer.op_group(tests) == op


def test_api_dedup_preserves_the_call_and_module_not_return_offsets():
    report = ('========= Program hit CUDA_ERROR_INVALID_VALUE on CUDA API call to cuGetProcAddress_v2.\n'
              '=========     Host Frame:  [0x10db6] in driver.so\n========= \n')
    parsed = sanitizer.read_log(
        (report + report.replace('0x10db6', '0x10ddc') + '========= ERROR SUMMARY: 2 errors\n').splitlines())
    assert len(parsed['signals']) == 1
    assert parsed['signals'][0]['count'] == 2
    assert parsed['signals'][0]['kind'] == 'api'


def test_peer_barrier_pattern_is_not_presented_as_a_generic_device_fault():
    report = ('========= Invalid __remote_shared__ write of size 8 bytes\n'
              '=========     at kernel+0x100\n'
              '=========     Barrier at 0x4400 in CTA 0 is not located in destination CTA\n'
              '========= ERROR SUMMARY: 1 error\n')
    parsed = sanitizer.read_log(report.splitlines())
    assert parsed['raw_count'] == 1
    assert parsed['signals'][0]['kind'] == 'peer-barrier'


def test_sanitizer_unsummarized_middle_process_stays_incomplete():
    banner = "========= COMPUTE-SANITIZER\n"
    clean = "========= ERROR SUMMARY: 0 errors\n"
    parsed = sanitizer.read_log((banner + clean + banner + _sanitizer_report("d0", 0) + banner + clean).splitlines())
    assert parsed["raw_count"] is None
    assert parsed["truncated"] is True
    assert parsed["not_printed"] is None


def test_sanitizer_sums_only_fully_summarized_processes():
    banner = "========= COMPUTE-SANITIZER\n"
    parsed = sanitizer.read_log(
        (banner + "========= ERROR SUMMARY: 3 errors\n" + banner + "========= ERROR SUMMARY: 4 errors\n").splitlines())
    assert parsed["raw_count"] == 7
    assert parsed["truncated"] is False


def test_sanitizer_summary_retains_report_totals_and_incomplete_evidence():
    signal = {"kind": "device", "message": "Invalid read", "sites": ["at mm_kernel+0x100"], "count": 4}
    row = {
        "run_id": 1, "sha": "a", "gpu": "B200", "tool": "memcheck", "signals": [signal], "kernels": ["mm_kernel"],
        "log": "group-1/memcheck.log", "raw_count": 4, "not_printed": 0, "completion": "complete", "elapsed_s": 2.5
    }
    second = {
        **row, "signals": [{**signal, "count": 6}], "log": "group-2/memcheck.log", "raw_count": None, "not_printed":
        None, "completion": "incomplete", "timed_out": True, "elapsed_s": 90
    }
    summary = sanitizer.observation_summary([row, second])
    assert summary["kinds"]["device"] == 1
    assert summary["signatures"][0]["printed_reports"] == 10
    assert summary["signatures"][0]["observed_logs"] == 2
    assert summary["signatures"][0]["attribution"] == "compiled kernel match"
    tool = summary["tools"][0]
    assert tool["complete"] == 1 and tool["observations"] == 2
    assert tool["raw_known"] == 4 and tool["unknown_count"] == 1
    assert tool["elapsed_s"] == 92.5 and tool["timed_out"] == 1
    assert summary["tools"][1]["elapsed_s"] is None


@pytest.mark.parametrize("changed", [{"run_id": 2}, {"gpu": "H100"}, {"tool": "synccheck"}, {"sha": "b"}])
def test_sanitizer_summary_never_merges_distinct_execution_contexts(changed):
    signal = {"kind": "device", "message": "Invalid read", "sites": ["at mm_kernel+0x100"], "count": 1}
    row = {
        "run_id": 1, "sha": "a", "gpu": "B200", "tool": "memcheck", "signals": [signal], "log": "memcheck.log",
        "kernels": ["mm_kernel"], "raw_count": 1, "not_printed": 0, "completion": "complete"
    }
    assert len(sanitizer.observation_summary([row, {**row, **changed}])["signatures"]) == 2


def test_tracking_uses_reported_kernel_evidence_not_the_op_suite_name():
    signals = [{"kind": "device", "message": "Invalid read", "sites": [f"at {kernel}+0x100"], "count": 3}
               for kernel in ["mm_kernel", "void at::native::copy_kernel(int)", "nvjet_sm100"]]
    signals += [{"kind": "api", "message": "Program hit CUDA_ERROR_INVALID_VALUE", "sites": [], "count": 1}]
    row = {
        "run_id": 1, "sha": "a", "gpu": "B200", "tool": "memcheck", "signals": signals, "op": "mm", "log":
        "memcheck.log", "raw_count": 10, "not_printed": 0, "completion": "complete", "kernels": ["mm_kernel"]
    }
    summary = sanitizer.observation_summary([row])
    assert summary["kinds"]["device"] == 1
    assert summary["kinds"]["api"] == 0
    assert summary["signatures"][0]["sites"] == ["at mm_kernel+0x100"]
    assert summary["tracked_printed_reports"] == 3
    assert summary["excluded_signatures"] == 3
    assert summary["tools"][0]["raw_known"] == 10
    assert len(row["signals"]) == 4


@pytest.mark.parametrize("kernels", [None, []])
def test_unknown_or_empty_kernel_evidence_never_creates_triton_signal(kernels):
    signal = {"kind": "device", "message": "Invalid read", "sites": ["at mm_kernel+0x100"], "count": 1}
    row = {
        "run_id": 1, "sha": "a", "gpu": "B200", "tool": "memcheck", "signals": [signal], "kernels": kernels, "log":
        "memcheck.log", "raw_count": 1, "not_printed": 0, "completion": "complete"
    }
    summary = sanitizer.observation_summary([row])
    assert summary["signatures"] == []
    assert summary["unattributed_signatures"] == 1
    assert summary["tools"][0]["raw_known"] == 1


def test_cross_kernel_race_requires_attribution_for_both_accesses():
    signal = {
        "kind": "device", "message": "Race reported", "sites":
        ["Write access at mm_kernel+0x100", "Read access at cuda_kernel+0x200"], "count": 1
    }
    assert sanitizer.signal_attribution(signal, ["mm_kernel"]) == "mixed attribution"


def step(name, conclusion, start, end):
    return {"name": name, "conclusion": conclusion, "started_at": start, "completed_at": end}


@pytest.mark.parametrize('quoted', [False, True])
def test_checkout_identity_excludes_tooling_and_unscoped_hashes(quoted):
    q = "'" if quoted else ""
    job = {
        "steps": [
            step("Checkout build tooling", "success", "2026-10-16T00:00:00Z", "2026-10-16T00:01:00Z"),
            step("Checkout", "success", "2026-10-16T00:02:00Z", "2026-10-16T00:03:00Z")
        ]
    }
    log = (f"2026-10-16T00:00:30Z [command]/usr/bin/git log -1 --format={q}%H{q}\n"
           f"2026-10-16T00:00:30Z {q}{'a' * 40}{q}\n"
           f"2026-10-16T00:02:30Z [command]/usr/bin/git log -1 --format={q}%H{q}\n"
           f"2026-10-16T00:02:30.1234567Z {q}{'b' * 40}{q}\n"
           f"2026-10-16T00:04:00Z {'c' * 40}\n")
    assert lg.checkout_sha(job, log) == "b" * 40
    assert lg.checkout_sha(job, log.split("2026-10-16T00:02:30Z")[0]) is None


def test_exact_failure_evidence_keeps_parameters_and_verdict_kind():
    job = {
        "name": "mi350 / mi350-tlx-test", "steps":
        [step("Run TLX core tests", "failure", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    log = ("2026-10-16T00:01:00Z FAILED x.py::test_boundary_graph[False] - AssertionError: bitwise\n"
           "2026-10-16T00:01:01Z FAILED x.py::test_boundary_graph[True] - AssertionError: bitwise\n"
           "2026-10-16T00:01:02Z ERROR x.py::test_fixture\n")
    lg.attach_failures(job, log)
    evidence = job["steps"][0]["test_failures"]
    assert [item["id"] for item in evidence
            ] == ["x.py::test_boundary_graph[False]", "x.py::test_boundary_graph[True]", "x.py::test_fixture"]
    assert [item["kind"] for item in evidence] == ["failed", "failed", "error"]
    lg.attach_failures(job, "2026-10-16T00:01:00Z Timeout\n")
    assert job["steps"][0]["test_failures"] == []


def test_short_summary_lines_are_parsed_and_normalized():
    log = ("2026-10-17T09:46:11.5522510Z FAILED a.py::t[x] - E: boom\n"
           "2026-10-17T09:46:11.6000000Z ERROR b.py::u\n"
           "2026-10-17T09:40:00.0000000Z a.py::t[x] FAILED                    [ 26%]\n")
    assert lg.parse_lines(log) == [("2026-10-17T09:46:11.552251Z", "a.py::t", "E: boom"),
                                   ("2026-10-17T09:46:11.600000Z", "b.py::u", "")]


def test_ansi_color_codes_are_ignored():
    log = "2026-10-17T09:46:11.5522510Z \x1b[31mFAILED\x1b[0m a.py::t - E: boom\n"
    assert [i for _, i, _ in lg.parse_lines(log)] == ["a.py::t"]


def test_parametrized_variants_collapse_to_one_id():
    job = {
        "name": "b200 / b200-tlx-test", "steps":
        [step("Run TLX core tests", "failure", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    log = ("2026-10-16T00:01:56.2055972Z FAILED x.py::test_addmm[bf16] - AssertionError: one\n"
           "2026-10-16T00:01:56.2057707Z FAILED x.py::test_addmm[fp16] - AssertionError: two\n")
    lg.attach_failures(job, log)
    assert job["steps"][0]["failures"] == [{"id": "x.py::test_addmm", "msg": "AssertionError: one"}]


def test_bench_keeps_the_op_parameter():
    job = {
        "name": "b200 / bench", "steps":
        [step("Run tlx.ops perf suite", "failure", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    log = ("2026-10-16T00:01:56.2055972Z FAILED t.py::test_op_perf[bench_mm] - AssertionError: one\n"
           "2026-10-16T00:01:56.2057707Z FAILED t.py::test_op_perf[bench_kda] - AssertionError: two\n")
    lg.attach_failures(job, log, normalize_ids=False)
    assert [f["id"]
            for f in job["steps"][0]["failures"]] == ["t.py::test_op_perf[bench_mm]", "t.py::test_op_perf[bench_kda]"]


def test_lines_attach_to_the_step_by_timestamp():
    job = {
        "name":
        "mi350 / mi350-tlx-test", "steps": [
            step("Run TLX core tests", "failure", "2026-10-17T01:00:00Z", "2026-10-17T01:30:00Z"),
            step("Run TLX tutorial correctness tests", "failure", "2026-10-17T01:30:01Z", "2026-10-17T02:00:00Z"),
        ]
    }
    log = ("2026-10-17T01:59:00.0000000Z FAILED tut.py::t2 - E\n"
           "2026-10-17T01:10:00.0000000Z [gw0] [ 10%] FAILED core.py::t1[a]\n"
           "2026-10-17T01:29:00.0000000Z FAILED core.py::t1[a] - E\n")
    lg.attach_failures(job, log)
    assert [f["id"] for f in job["steps"][0]["failures"]] == ["core.py::t1"]
    assert [f["id"] for f in job["steps"][1]["failures"]] == ["tut.py::t2"]


def test_failed_step_without_lines_gets_job_step_id():
    job = {
        "name": "b200 / b200-tlx-test", "steps":
        [step("Run TLX core tests", "failure", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    lg.attach_failures(job, "2026-10-16T00:30:00.0000000Z Segmentation fault\n")
    assert job["steps"][0]["failures"] == [{"id": "b200-tlx-test/Run TLX core tests", "msg": ""}]


def test_failed_setup_step_gets_no_failures():
    job = {
        "name": "mi350 / mi350-tlx-test", "steps":
        [step("Install prebuilt Triton", "failure", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    lg.attach_failures(job, "")
    assert "failures" not in job["steps"][0]


def test_passing_steps_get_no_failures():
    job = {
        "name": "h100 / h100-tlx-test", "steps":
        [step("Run TLX core tests", "success", "2026-10-16T00:00:00Z", "2026-10-16T01:00:00Z")]
    }
    lg.attach_failures(job, "2026-10-16T00:30:00.0000000Z FAILED a.py::t - E\n")
    assert "failures" not in job["steps"][0]


def test_selection_line_is_parsed():
    log = ("2026-10-19T19:56:15.6894398Z GPU selection for e46f: "
           "{'h100': True, 'b200': False, 'mi350': True, 'build': True}\n")
    assert lg.parse_selection(log) == {"h100": True, "b200": False, "mi350": True, "build": True, "fallback": False}


def test_selection_fallback_is_flagged():
    log = (
        "2026-10-19T19:56:15.1Z ::warning::Cannot determine changed paths; running all GPU suites: boom\n"
        "2026-10-19T19:56:15.2Z GPU selection for e46f: {'h100': True, 'b200': True, 'mi350': True, 'build': True}\n")
    assert lg.parse_selection(log)["fallback"] is True


def test_selection_missing_is_none():
    assert lg.parse_selection("2026-10-19T19:56:15.1Z nothing here\n") is None


def lit_job():
    return {"name": "LIT Tests", "steps": [step("Build and run LIT tests", "failure", None, None)]}


def test_lit_failures_take_the_first_error_of_each_test():
    log = ("2026-10-21T22:22:47.7Z Testing:  0.. 10.. 20..\n"
           "2026-10-21T22:22:47.8Z FAIL: TRITON :: TritonGPU/a.mlir (401 of 646)\n"
           "2026-10-21T22:22:47.8Z ******************** TEST 'TRITON :: TritonGPU/a.mlir' FAILED ********************\n"
           "2026-10-21T22:22:47.8Z # | /w/a.mlir:19:10: remark: MMA version 5 acceleration not applied\n"
           "2026-10-21T22:22:47.8Z # | /w/a.mlir:32:50: error: undefined variable: MMA\n"
           "2026-10-21T22:22:47.8Z # | /w/a.mlir:40:1: error: a second error\n"
           "2026-10-21T22:22:47.9Z ********************\n"
           "2026-10-21T22:22:48.0Z XPASS: TRITON :: Conversion/b.mlir (402 of 646)\n"
           "2026-10-21T22:22:48.1Z XFAIL: TRITON :: Conversion/c.mlir (403 of 646)\n"
           "2026-10-21T22:22:49.0Z Failed Tests (1):\n"
           "2026-10-21T22:22:49.0Z   TRITON :: TritonGPU/a.mlir\n")
    job = lit_job()
    lg.attach_lit_failures(job, log)
    assert job["steps"][0]["failures"] == [{"id": "TritonGPU/a.mlir", "msg": "undefined variable: MMA"},
                                           {"id": "Conversion/b.mlir", "msg": "XPASS"}]


def test_lit_step_that_failed_before_lit_ran_gets_no_failures():
    job = lit_job()
    lg.attach_lit_failures(job, "2026-10-21T22:00:00.0Z FAILED: lib/x.cpp.o\n2026-10-21T22:00:00.1Z error: boom\n")
    assert "failures" not in job["steps"][0]


def test_durations_go_to_the_step_that_printed_them():
    job = {
        "name":
        "b200 / b200-tlx-test", "steps": [
            step("Run TLX core tests", "success", "2026-10-22T01:00:00Z", "2026-10-22T01:26:02Z"),
            step("Run TLX tutorial correctness tests", "success", "2026-10-22T01:26:02Z", "2026-10-22T01:27:00Z"),
        ]
    }
    log = (
        "2026-10-22T01:20:00.0000000Z x.py::t PASSED\n"
        "2026-10-22T01:26:02.7203967Z ============================= slowest 25 durations =============================\n"
        "2026-10-22T01:26:02.7204000Z 367.30s call     a.py::test_slow\n"
        "2026-10-22T01:26:02.7205000Z 2.00s setup    b.py::test_p[x]\n"
        "2026-10-22T01:26:02.7206000Z 10.00s call     b.py::test_p[x]\n"
        "2026-10-22T01:26:02.7207000Z \n"
        "2026-10-22T01:26:02.7208000Z (12 durations < 0.005s hidden.  Use -vv to show these durations.)\n"
        "2026-10-22T01:26:02.7216176Z ========= 1592 passed, 102 skipped in 1328.03s (0:22:08) ==========\n"
        "2026-10-22T01:26:40.0000000Z ============================= slowest 25 durations =============================\n"
        "2026-10-22T01:26:40.1000000Z 3.50s call     third_party/c.py::test_gemm\n")
    lg.attach_durations(job, log)
    core, corr = job["steps"]
    # The core block, printed in the second the core step ended and the correctness step began, stays with core.
    assert core["slowest"] == [{"id": "a.py::test_slow", "s": 367.3}, {"id": "b.py::test_p[x]", "s": 12.0}]
    assert corr["slowest"] == [{"id": "third_party/c.py::test_gemm", "s": 3.5}]


def test_no_durations_block_leaves_steps_alone():
    job = {
        "name": "b200 / b200-tlx-test", "steps":
        [step("Run TLX core tests", "success", "2026-10-22T01:00:00Z", "2026-10-22T01:30:00Z")]
    }
    lg.attach_durations(job, "2026-10-22T01:20:00.0000000Z 12.00s call     a.py::t\n")
    assert "slowest" not in job["steps"][0]
