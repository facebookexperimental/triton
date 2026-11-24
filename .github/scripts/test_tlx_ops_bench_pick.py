import datetime
from types import SimpleNamespace

import tlx_ops_bench_pick
from tlx_ops_bench_pick import pick


def run(sha, **jobs):
    return {
        "head_sha": sha, "jobs": {"build / build": "success", **{f"{p} / {p}-tlx-test": c
                                                                 for p, c in jobs.items()}}
    }


def test_red_or_untested_platform_never_blocks_the_bench():
    runs = [run("untested", h100="success"), run("red", h100="failure", mi350="success")]
    chosen, ci = pick(runs, lambda r: r["jobs"], ["h100", "mi350"])
    assert chosen["head_sha"] == "red" and ci == {"h100": "failure", "mi350": "success"}
    chosen, ci = pick(runs[:1], lambda r: r["jobs"], ["h100", "mi350"])
    assert chosen["head_sha"] == "untested" and ci == {"h100": "success", "mi350": "not run"}


def test_scheduled_run_benches_a_night_without_summaries_at_any_hour(monkeypatch, tmp_path):

    class At14Utc(datetime.datetime):

        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 20, 14, tzinfo=datetime.timezone.utc)

    sha = "a" * 40
    monkeypatch.setattr(tlx_ops_bench_pick, "datetime", SimpleNamespace(datetime=At14Utc, timezone=datetime.timezone))
    monkeypatch.setattr(tlx_ops_bench_pick, "gh", lambda path, **query: {"assets": []})
    monkeypatch.setattr(tlx_ops_bench_pick, "push_runs", lambda **query: [{"head_sha": sha, "html_url": "url"}])
    monkeypatch.setattr(tlx_ops_bench_pick, "jobs_of", lambda r: run(sha, h100="success")["jobs"])
    monkeypatch.setattr(tlx_ops_bench_pick.subprocess, "check_output", lambda *a, **k: 'TRITON_VERSION = "3.9.0"\n')
    output = tmp_path / "output"
    output.write_text("")
    for name, value in {
            "EVENT_NAME": "schedule", "SKIP_PLATFORMS": "b200 mi350", "INPUT_DATE": "", "INPUT_SHA": "",
            "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")
    }.items():
        monkeypatch.setenv(name, value)
    tlx_ops_bench_pick.main()
    assert f"sha={sha}\n" in output.read_text()
