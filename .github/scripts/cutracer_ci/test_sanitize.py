import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from sample_logs import SYNCCHECK_CLEAN, SYNCCHECK_ERROR
from sanitize import main


def _fake_cutracer(tmp_path: Path, logs: dict[str, str], exits: dict[str, int] | None = None,
                   kernel: str | None = None) -> None:
    """Put a `cutracer` on PATH that writes a canned log for each --tool.

    It exits with ``exits[tool]`` (default 0) and, like a real run, compiles
    ``kernel`` into the Triton cache.
    """
    (tmp_path / "logs.json").write_text(json.dumps({"logs": logs, "exits": exits or {}, "kernel": kernel}))
    script = tmp_path / "bin" / "cutracer"
    script.parent.mkdir()
    script.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys
        args = sys.argv[1:]
        tool, out = args[args.index("--tool") + 1], args[args.index("-o") + 1]
        fake = json.load(open({str(tmp_path / "logs.json")!r}))
        if fake["kernel"]:
            ir = os.path.join(os.environ["TRITON_CACHE_DIR"], "K", fake["kernel"] + ".ttgir")
            os.makedirs(os.path.dirname(ir), exist_ok=True)
            open(ir, "w").write('#loc = loc("/x/sm100.py":1:0)')
        if tool in fake["logs"]:
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, tool + ".log"), "w").write(fake["logs"][tool])
        sys.exit(fake["exits"].get(tool, 0))
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return tmp_path


def _run(tmp_path, *extra):
    out = tmp_path / "out"
    code = main([
        "--tools", "synccheck,memcheck", "--kernel", "_attn_fwd_ws", "--out",
        str(out), *extra, "--", "python", "-c", "pass"
    ])
    return code, json.loads((out / "results.json").read_text())


def test_main_reports_findings_without_failing(fake_env):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_ERROR, "memcheck": SYNCCHECK_CLEAN})
    code, results = _run(fake_env)
    assert code == 0
    assert [r["outcome"] for r in results["results"]] == ["findings", "none"]
    assert results["cmd"] == ["python", "-c", "pass"]
    assert [r["kernel"] for r in results["results"][0]["reports"]] == ["_attn_fwd_ws"]
    assert os.environ["TRITON_CACHE_DIR"] == str(fake_env / "out" / "triton-cache")
    assert results["results"][1]["reports"] == []


def test_main_uses_a_fresh_triton_cache(fake_env, monkeypatch):
    # A cache shared with earlier runs would claim kernels this run never launched.
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_CLEAN, "memcheck": SYNCCHECK_CLEAN}, kernel="mm_kernel")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(fake_env / "shared-cache"))
    _, payload = _run(fake_env)
    assert os.environ["TRITON_CACHE_DIR"] == str(fake_env / "out" / "triton-cache")
    assert all(result["kernels"] == ["mm_kernel"] for result in payload["results"])
    assert all(result["elapsed_s"] >= 0 for result in payload["results"])


def test_main_fail_on_findings(fake_env):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_ERROR, "memcheck": SYNCCHECK_CLEAN})
    assert _run(fake_env, "--fail-on-findings")[0] == 1


def test_main_infra_when_no_log(fake_env):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_CLEAN})
    code, results = _run(fake_env)
    assert code == 2
    assert results["results"][1]["outcome"] == "infra"


def test_main_writes_step_summary(fake_env, monkeypatch):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_ERROR, "memcheck": SYNCCHECK_CLEAN})
    summary = fake_env / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _run(fake_env)
    text = summary.read_text()
    assert "| synccheck | findings | 1 | 0 |" in text
    assert "Barrier error detected. Missing wait." in text


def test_rejects_unknown_tool(fake_env):
    with pytest.raises(SystemExit):
        main(["--tools", "bogus", "--out", str(fake_env / "out"), "--", "true"])


def test_main_writes_the_failure_list_only_when_asked(fake_env, monkeypatch):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_ERROR, "memcheck": SYNCCHECK_CLEAN})
    output = fake_env / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    _run(fake_env)
    assert not output.exists()
    _run(fake_env, "--issue-title-prefix", "[CUTracer][B200]", "--issue-job", "sanitize-b200")
    values = dict(line.partition("=")[::2] for line in output.read_text().splitlines())
    assert [(f["issue_title"], f["job_name"]) for f in json.loads(values["failures"])] == [
        ("[CUTracer][B200] _attn_fwd_ws in sm100.py", "sanitize-b200"),
    ]
    # No kernel compiled in the fake run, so only the tools' prefixes.
    assert json.loads(values["reconcile_prefixes"]) == [
        "[CUTracer][B200] synccheck incomplete",
        "[CUTracer][B200] memcheck incomplete",
    ]


def test_main_reports_an_incomplete_tool_as_a_fallback(fake_env, monkeypatch):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_CLEAN})
    output = fake_env / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert _run(fake_env, "--issue-title-prefix", "[CUTracer][B200]")[0] == 2
    values = dict(line.partition("=")[::2] for line in output.read_text().splitlines())
    (entry, ) = json.loads(values["failures"])
    assert entry["issue_title"] == "[CUTracer][B200] memcheck incomplete"
    assert entry["fallback"] is True


@pytest.mark.parametrize("minutes,race_seconds,other_seconds", [(30, 300, 1800), (1, 60, 60)])
def test_racecheck_budget_does_not_extend_other_tool_limits(fake_env, minutes, race_seconds, other_seconds):
    _fake_cutracer(
        fake_env, {
            "racecheck": "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n", "synccheck":
            SYNCCHECK_CLEAN
        })
    out = fake_env / "out"
    assert main([
        "--tools", "racecheck,synccheck", "--timeout-minutes",
        str(minutes), "--out",
        str(out), "--", "python", "-c", "pass"
    ]) == 0
    results = json.loads((out / "results.json").read_text())["results"]
    assert [result["timeout_s"] for result in results] == [race_seconds, other_seconds]


def test_timed_out_racecheck_stops_its_worker_and_continues_as_incomplete(fake_env, monkeypatch):
    import sanitize

    _fake_cutracer(fake_env, {"racecheck": SYNCCHECK_CLEAN, "synccheck": SYNCCHECK_CLEAN})
    script = fake_env / "bin" / "cutracer"
    pid_file = fake_env / "worker.pid"
    script.write_text(script.read_text().replace(
        'sys.exit(fake["exits"].get(tool, 0))', f'''if tool == "racecheck":
    import subprocess, time
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open({str(pid_file)!r}, "w").write(str(worker.pid))
    time.sleep(60)
sys.exit(fake["exits"].get(tool, 0))'''))
    monkeypatch.setattr(sanitize, "RACECHECK_TIMEOUT_S", 1)
    out = fake_env / "out"
    assert main(["--tools", "racecheck,synccheck", "--out", str(out), "--", "python", "-c", "pass"]) == 2
    results = json.loads((out / "results.json").read_text())["results"]
    assert results[0]["timed_out"] is True
    assert results[0]["outcome"] == "infra" and results[0]["count"] is None
    assert results[1]["outcome"] == "none" and results[1]["returncode"] == 0
    assert (out / "racecheck" / "racecheck.log").read_text() == SYNCCHECK_CLEAN
    stat_file = Path(f"/proc/{int(pid_file.read_text())}/stat")
    assert not stat_file.exists() or stat_file.read_text().split(") ")[1].split()[0] == "Z"


def _prefixes(fake_env, monkeypatch):
    output = fake_env / "github_output"
    output.unlink(missing_ok=True)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    _run(fake_env, "--issue-title-prefix", "[CUTracer][B200]")
    values = dict(line.partition("=")[::2] for line in output.read_text().splitlines())
    return json.loads(values["reconcile_prefixes"])


def test_a_clean_run_counts_its_kernels_as_checked(fake_env, monkeypatch):
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_CLEAN, "memcheck": SYNCCHECK_CLEAN}, kernel="_attn_fwd_ws")
    assert _prefixes(fake_env, monkeypatch) == [
        "[CUTracer][B200] _attn_fwd_ws in ",
        "[CUTracer][B200] synccheck incomplete",
        "[CUTracer][B200] memcheck incomplete",
    ]


def test_a_tool_that_did_not_finish_the_tests_checks_no_kernel(fake_env, monkeypatch):
    # synccheck aborted a kernel (the test failed); memcheck ran clean. The
    # shared cache holds the kernel, but synccheck may not have reached it.
    _fake_cutracer(fake_env, {"synccheck": SYNCCHECK_ERROR, "memcheck": SYNCCHECK_CLEAN}, exits={"synccheck": 1},
                   kernel="_attn_fwd_ws")
    assert _prefixes(fake_env, monkeypatch) == [
        "[CUTracer][B200] synccheck incomplete",
        "[CUTracer][B200] memcheck incomplete",
    ]
