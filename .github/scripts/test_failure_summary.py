"""Unit tests for failure_summary (stdlib-only; run with pytest)."""

import json
import subprocess
import sys
from pathlib import Path

import failure_summary as fs

SCRIPT = Path(__file__).with_name("failure_summary.py")


def item(norm, summary="AssertionError: boom", repro="", external=False, reason=""):
    return {
        "normalized_failure_id": norm, "summary": summary, "repro": repro, "external_dep": external,
        "external_dep_reason": reason
    }


def run(tmp_path, failures, job="mi350-tlx-test"):
    summary = tmp_path / "summary.md"
    env = {"JOB": job, "GITHUB_STEP_SUMMARY": str(summary)}
    out = subprocess.run([sys.executable, str(SCRIPT)], input=failures, env=env, capture_output=True, text=True,
                         check=True)
    return (summary.read_text() if summary.exists() else ""), out.stdout + out.stderr


def test_failures_become_a_table_with_repro_commands(tmp_path):
    failures = [
        item("test_a.py::test_x", repro="python -m pytest test_a.py::test_x -v"),
        item("test_b.py::test_y", "CUDA out of memory", external=True, reason="GPU out of memory")
    ]
    md, _ = run(tmp_path, json.dumps(failures))
    assert "### mi350-tlx-test: 2 failures" in md
    assert "| `test_a.py::test_x` | AssertionError: boom |  |" in md
    assert "| `test_b.py::test_y` | CUDA out of memory | GPU out of memory |" in md
    assert "python -m pytest test_a.py::test_x -v" in md


def test_nothing_reaches_the_job_log(tmp_path):
    # Internal CI signal cards pick their highlight from the job log.
    failures = [item("t::x", "Traceback (most recent call last):"), item("slow-step:core", "80 min of 90")]
    md, log = run(tmp_path, json.dumps(failures))
    assert md
    assert log == ""


def test_empty_list_writes_nothing(tmp_path):
    for failures in ("", "[]"):
        assert run(tmp_path, failures) == ("", "")


def test_slow_steps_are_listed_in_their_own_section():
    md = fs.render("h100-tlx-test", [item("slow-step:tlx-core", "tlx-core.xml: 80.0 min of a 90 min budget (89%).")])
    assert "failure" not in md
    assert "### h100-tlx-test: close to its time budget" in md


def test_table_cells_are_escaped():
    assert "| a \\| b second line |" in fs.render("j", [item("t::x", "a | b\nsecond line")])
