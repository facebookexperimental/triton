import ci_health_logs as lg


def step(name, conclusion, start, end):
    return {"name": name, "conclusion": conclusion, "started_at": start, "completed_at": end}


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
