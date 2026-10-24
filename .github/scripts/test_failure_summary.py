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
    env = {"FAILURES": failures, "JOB": job, "GITHUB_STEP_SUMMARY": str(summary)}
    out = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    return (summary.read_text() if summary.exists() else ""), out.stdout


def test_failures_become_a_table_annotations_and_repro_commands(tmp_path):
    failures = [
        item("test_a.py::test_x", repro="python -m pytest test_a.py::test_x -v"),
        item("test_b.py::test_y", "CUDA out of memory", external=True, reason="GPU out of memory")
    ]
    md, out = run(tmp_path, json.dumps(failures))
    assert "### mi350-tlx-test: 2 failures" in md
    assert "| `test_a.py::test_x` | AssertionError: boom |  |" in md
    assert "| `test_b.py::test_y` | CUDA out of memory | GPU out of memory |" in md
    assert "python -m pytest test_a.py::test_x -v" in md
    assert out.splitlines() == [
        "::error title=mi350-tlx-test%3A test_a.py%3A%3Atest_x::AssertionError: boom",
        "::error title=mi350-tlx-test%3A test_b.py%3A%3Atest_y::CUDA out of memory",
    ]


def test_empty_list_writes_nothing(tmp_path):
    for failures in ("", "[]"):
        assert run(tmp_path, failures) == ("", "")


def test_slow_steps_are_warnings_in_their_own_section():
    items = [item("slow-step:tlx-core", "tlx-core.xml: 80.0 min of a 90 min budget (89%).")]
    md = fs.render("h100-tlx-test", items)
    assert "failure" not in md
    assert "### h100-tlx-test: close to its time budget" in md
    assert fs.annotations("h100-tlx-test", items)[0].startswith("::warning ")


def test_table_cells_and_annotations_are_escaped():
    items = [item("t::x", "a | b\nsecond line 100%")]
    assert "| a \\| b second line 100% |" in fs.render("j", items)
    assert fs.annotations("j", items) == ["::error title=j%3A t%3A%3Ax::a | b%0Asecond line 100%25"]


def test_annotations_are_capped():
    items = [item(f"t::{i}") for i in range(fs.MAX_ANNOTATIONS + 5)]
    assert len(fs.annotations("j", items)) == fs.MAX_ANNOTATIONS
