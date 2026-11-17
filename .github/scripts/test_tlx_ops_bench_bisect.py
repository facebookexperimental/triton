import json
import os

import tlx_ops_bench_bisect as bisect


def run(tflops, cv=0.01, status="ok"):
    return {"status": status, "tflops": tflops, "cv": cv, "notes": []}


def ok(*values):
    return bisect.summarize_runs([run(v) for v in values])


def test_failed_or_unreliable_commits_are_skipped_never_bad():
    cutoff = 75.0
    unreliable = [
        {"status": "build_failed", "reason": "cmake"},
        {"status": "missing_case"},
        bisect.summarize_runs([run(10), run(None, None, "error")]),
        bisect.summarize_runs([run(10, cv=0.2), run(10, cv=0.04),
                               run(10, cv=0.5), run(10, cv=0.1)]),
        bisect.summarize_runs([run(10, cv=float("nan")),
                               run(10, cv=-0.01),
                               run(10, cv=None), run(10)]),
        bisect.summarize_runs([run(10)]),
        ok(50, 90),
    ]
    for measurement in unreliable:
        assert bisect.classify(measurement, cutoff) == bisect.SKIP
    assert bisect.classify(ok(50, 52), cutoff) == bisect.BAD
    assert bisect.classify(ok(99, 101), cutoff) == bisect.GOOD


def test_competitor_floor_is_ignored_only_raw_throughput_counts():
    measurement = bisect.summarize_runs([run(100, status="pip"), run(100, status="pip")])
    assert measurement["status"] == "ok" and bisect.classify(measurement, 75) == bisect.GOOD


def test_endpoints_must_reproduce_before_any_blame():
    cutoff, reason = bisect.check_endpoints(ok(100, 101), ok(40, 41), 0.5)
    assert reason is None and 70 < cutoff < 71
    assert bisect.check_endpoints(ok(100, 101), ok(60, 61), 0.5)[0] is None
    assert bisect.check_endpoints(ok(100, 101), ok(95, 96), 0.9)[0] is None
    assert bisect.check_endpoints({"status": "noisy"}, ok(40, 41), 0.5)[0] is None
    assert bisect.check_endpoints(ok(100, 101), {"status": "build_failed"}, 0.5)[0] is None


def test_only_a_conclusive_git_bisect_names_a_commit():
    sha = "a" * 40
    assert bisect.first_bad(f"{sha} is the first bad commit\ncommit {sha}\n", 0) == sha
    skipped = f"There are only 'skip'ped commits left to test.\nThe first bad commit could be any of:\n{sha}\n"
    assert bisect.first_bad(skipped, 2) is None


def test_a_crashing_step_skips(tmp_path):
    spec = tmp_path / "step.json"
    spec.write_text(json.dumps({"source": str(tmp_path / "missing"), "log": str(tmp_path / "log")}))
    assert bisect.step_main(str(spec)) == bisect.SKIP
    assert bisect.step_main(str(tmp_path / "absent.json")) == bisect.SKIP


def test_a_step_builds_without_the_repository_git_bisect_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(tmp_path / ".git" / "worktrees" / "source"))
    seen = {}
    monkeypatch.setattr(bisect, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(bisect, "evaluate", lambda step: seen.update(git_dir=os.environ.get("GIT_DIR")) or {})
    spec = tmp_path / "step.json"
    spec.write_text(
        json.dumps({"source": str(tmp_path), "gpu_uuid": "GPU-x", "cutoff": 1, "log": str(tmp_path / "log")}))
    assert bisect.step_main(str(spec)) == bisect.SKIP
    assert seen == {"git_dir": None}


def test_candidate_code_has_private_caches_and_no_action_credentials(tmp_path, monkeypatch):
    credentials = ("ACTIONS_RUNTIME_TOKEN", "ACTIONS_RUNTIME_URL", "ACTIONS_CACHE_URL", "ACTIONS_RESULTS_URL",
                   "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL", "GH_TOKEN", "GITHUB_TOKEN",
                   "GITHUB_ENV", "GITHUB_PATH", "GITHUB_OUTPUT", "GITHUB_STATE", "GITHUB_STEP_SUMMARY")
    for key in credentials:
        monkeypatch.setenv(key, "credential")
    monkeypatch.setenv("TRITON_HOME", "/shared/triton")
    monkeypatch.setenv("UV_CACHE_DIR", "/shared/uv")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    bisect.isolate_environment(tmp_path)
    assert all(key not in os.environ for key in credentials)
    assert os.environ["TRITON_HOME"] == str(tmp_path / "triton-home")
    assert os.environ["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")
    assert os.environ["TMPDIR"] == str(tmp_path / "tmp")
