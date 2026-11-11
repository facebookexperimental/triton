import os
import subprocess
import sys
from pathlib import Path

import pytest

from random_delay import entries, races, read_records

DOT = "python/test/unit/language/test_tlx_dot_sm100.py::test_async_dot_scaled_2cta"
A, B = f"{DOT}[m64_per_cta_unit_scales]", f"{DOT}[m64_per_cta_n128_a_nonuniform]"
FAILED, PASSED, SKIPPED = ("failed", "AssertionError: Tensor-likes are not close!"), ("passed", ""), ("skipped", "")


@pytest.mark.parametrize(
    "plain, delay, confirm, expected",
    [
        ({A: PASSED}, {A: FAILED}, {A: FAILED}, [A]),
        # Failing once is not enough: the confirming run must fail too.
        ({A: PASSED}, {A: FAILED}, {A: PASSED}, []),
        # A confirming run that died before recording the test proves nothing.
        ({A: PASSED}, {A: FAILED}, {}, []),
        # Only a recorded pass counts: not a failure, a skip, or no record at all.
        ({A: FAILED}, {A: FAILED}, {A: FAILED}, []),
        ({A: SKIPPED}, {A: FAILED}, {A: FAILED}, []),
        ({}, {A: FAILED}, {A: FAILED}, []),
    ],
)
def test_races_need_a_recorded_pass_and_two_delay_failures(plain, delay, confirm, expected):
    assert races(plain, delay, confirm) == expected


def test_a_process_death_keeps_the_results_recorded_before_it(tmp_path):
    # CUTracer can _exit() mid-run; only the test running at that moment may be lost.
    (tmp_path / "test_death.py").write_text("import os\n\n"
                                            "def test_a():\n    assert False\n\n"
                                            "def test_b():\n    os._exit(1)\n\n"
                                            "def test_c():\n    pass\n")
    records = tmp_path / "records.jsonl"
    env = dict(os.environ, RECORD_OUTCOMES=str(records), PYTHONPATH=str(Path(__file__).resolve().parent))
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "record_outcomes", "test_death.py"],
        cwd=tmp_path, env=env, capture_output=True, check=False)
    run = read_records(records.read_text().splitlines())
    assert {test: outcome for test, (outcome, _) in run.outcomes.items()} == {"test_death.py::test_a": "failed"}
    assert run.crashed == "test_death.py::test_b"
    assert not run.finished


def test_a_cuda_error_poisons_every_later_result():
    # A sticky CUDA error fails every later test in the process; none of those failures is evidence.
    lines = [
        '{"event": "start", "nodeid": "t::a"}',
        '{"event": "teardown", "nodeid": "t::a", "outcome": "passed", "message": ""}',
        '{"event": "start", "nodeid": "t::b"}',
        '{"event": "call", "nodeid": "t::b", "outcome": "failed", "message": "RuntimeError: Triton Error [CUDA]: unspecified launch failure"}',
        '{"event": "teardown", "nodeid": "t::b", "outcome": "passed", "message": ""}',
        '{"event": "start", "nodeid": "t::c"}',
        '{"event": "call", "nodeid": "t::c", "outcome": "failed", "message": "torch.AcceleratorError: CUDA error: unspecified launch failure"}',
        '{"event": "teardown", "nodeid": "t::c", "outcome": "passed", "message": ""}', '{"event": "session_finished"}'
    ]
    run = read_records(lines)
    assert run.outcomes == {"t::a": ("passed", "")}
    assert run.crashed == "t::b"
    assert not run.finished


def test_one_issue_per_test_function():
    found = entries({A: FAILED[1], B: FAILED[1]}, dict.fromkeys([A, B], "repro"), "[CUTracer][B200]", "sanitize-b200")
    assert [entry["issue_title"] for entry in found] == [f"[CUTracer][B200] random delay: {DOT}"]
    assert found[0]["raw_failure_ids"].split("\n") == [B, A]
