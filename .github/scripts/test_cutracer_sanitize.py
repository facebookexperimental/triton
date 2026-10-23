import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

import cutracer_sanitize
from cutracer_sanitize import classify, main, report_lines

# Real compute-sanitizer output shapes, taken from CUTracer's parser fixtures
# (python/tests/example_inputs/racecheck_*.txt in facebookexperimental/CUTracer).
RACECHECK_HAZARD = """\
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at kernel_a+0x100 in kernels.py:42
child process output may be interleaved
=========     and Read access at kernel_a+0x200 in kernels.py:57 [3 hazards]
=========     and Read access at kernel_a+0x240 in kernels.py:58 [5 hazards]
=========
========= Warning: Maximum number of hazards reached.
=========
========= Warning: Race reported between Read access at kernel_b+0x300 in other.py:9
=========     and Write access at kernel_b+0x340 in other.py:12 [2 hazards]
=========
1 failed, 1 warning in 12.34s
========= Target application returned an error
========= RACECHECK SUMMARY: 7 hazards displayed (6 errors, 1 warning)
"""
RACECHECK_CLEAN = """\
========= COMPUTE-SANITIZER
7 passed in 5.01s
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
"""
SYNCCHECK_ERROR = """\
========= COMPUTE-SANITIZER
========= Barrier error detected. Missing wait.
=========     at _attn_fwd_ws+0x1a40 in sm100.py:812
=========
========= ERROR SUMMARY: 1 error
"""
SYNCCHECK_CLEAN = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
# memcheck on the B200 runner (tlx.ops flash-attention forward): Triton's
# launcher probing cuFuncSetAttribute, which the test survives.
MEMCHECK_API_ONLY = """\
========= COMPUTE-SANITIZER
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuFuncSetAttribute.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: loadBinary [0x6f43] in cuda_utils.cpython-312-x86_64-linux-gnu.so
=========
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuFuncSetAttribute.
=========     Saved host backtrace up to driver entry point at error
=========
========= ERROR SUMMARY: 2 errors
"""
MEMCHECK_MIXED = MEMCHECK_API_ONLY.replace(
    "========= ERROR SUMMARY: 2 errors",
    "========= Invalid __global__ read of size 4 bytes\n"
    "=========     at kernel+0x10 in k.py:3\n"
    "========= ERROR SUMMARY: 3 errors",
)


def test_racecheck_hazards_are_findings():
    assert classify("racecheck", RACECHECK_HAZARD) == {"outcome": "findings", "count": 7, "errors": 6, "warnings": 1}


def test_racecheck_clean_is_none():
    assert classify("racecheck", RACECHECK_CLEAN)["outcome"] == "none"


def test_error_summary_tools():
    assert classify("synccheck", SYNCCHECK_ERROR) == {"outcome": "findings", "count": 1, "api_errors": 0}
    assert classify("memcheck", SYNCCHECK_CLEAN) == {"outcome": "none", "count": 0, "api_errors": 0}


def test_host_api_errors_are_not_device_findings():
    assert classify("memcheck", MEMCHECK_API_ONLY) == {"outcome": "none", "count": 0, "api_errors": 2}
    assert classify("memcheck", MEMCHECK_MIXED) == {"outcome": "findings", "count": 1, "api_errors": 2}


@pytest.mark.parametrize("log", ["", "========= COMPUTE-SANITIZER\n", "Segmentation fault\n"])
def test_missing_summary_is_infra(log):
    assert classify("synccheck", log)["outcome"] == "infra"
    assert classify("racecheck", log)["outcome"] == "infra"


def test_report_lines_keep_reports_and_drop_summaries():
    lines = report_lines(RACECHECK_HAZARD)
    assert lines[0].startswith("Error: Race reported between Write access at kernel_a+0x100")
    assert "and Read access at kernel_a+0x200 in kernels.py:57 [3 hazards]" in lines
    assert not any("SUMMARY" in line or line.startswith("COMPUTE-SANITIZER") for line in lines)


def test_racecheck_findings_attribute_kernel():
    pytest.importorskip("cutracer.compute_sanitizer.parser")
    findings = cutracer_sanitize.racecheck_findings(RACECHECK_HAZARD, "kernel_a")
    assert [finding["in_kernel"] for finding in findings] == [True, False]
    assert findings[0]["accesses"][0]["location"] == "kernels.py:42"


def _fake_cutracer(tmp_path: Path, logs: dict[str, str]) -> None:
    """Put a `cutracer` on PATH that writes a canned log for each --tool."""
    (tmp_path / "logs.json").write_text(json.dumps(logs))
    script = tmp_path / "bin" / "cutracer"
    script.parent.mkdir()
    script.write_text(
        textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys
        args = sys.argv[1:]
        tool, out = args[args.index("--tool") + 1], args[args.index("-o") + 1]
        logs = json.load(open({str(tmp_path / "logs.json")!r}))
        if tool in logs:
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, tool + ".log"), "w").write(logs[tool])
        sys.exit(0)
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
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
