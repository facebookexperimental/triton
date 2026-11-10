import copy
import json

import pytest

import tlx_ops_bench_regressions as regressions


def night(date, value=100, *, status="ok", keys=("a", "b", "c"), direction="bwd", mode="gpu_events", **meta):
    cases = [{
        "key": key, "params": {"N": key}, "dtype": "bf16", "direction": direction, "status": status, "tlx_tflops":
        value, "ref_tflops": 20, "notes": []
    } for key in keys]
    return {
        "schema": 2, "platform": "mi350", "date": date, "sha": date * 5, "timing":
        {"latency_mode": mode, "replicates": 1}, "ops": {"flash_attn_mxfp8": {"cases": cases}}, **meta
    }


def history():
    return [night("20261101"), night("20261102"), night("20261103")]


def test_severe_throughput_loss_crosses_empty_nights_without_using_the_reference():
    gap = night("20261106", keys=())
    current = night("20261109", 30)
    current["ops"]["flash_attn_mxfp8"]["cases"][0]["ref_tflops"] = 1
    alerts = regressions.detect(history() + [gap, current])
    assert len(alerts) == 1 and alerts[0]["date"] == "20261109"
    assert len(alerts[0]["cases"]) == 3
    assert all(c["baseline"] == 100 and c["ratio"] == 0.3 for c in alerts[0]["cases"])
    assert all(c["baseline_dates"] == ["20261101", "20261102", "20261103"] for c in alerts[0]["cases"])
    stable = night("20261109", 100)
    for c in stable["ops"]["flash_attn_mxfp8"]["cases"]:
        c["ref_tflops"] = 200
    assert regressions.detect(history() + [stable]) == []


def test_moderate_loss_needs_two_observations_and_is_not_diluted_by_forward():
    nights = history() + [night("20261104", 85), night("20261109", 85)]
    for n in nights:
        n["ops"]["flash_attn_mxfp8"]["cases"] += night(n["date"], 100,
                                                       direction="fwd")["ops"]["flash_attn_mxfp8"]["cases"]
    alerts = regressions.detect(nights)
    assert len(alerts) == 1 and alerts[0]["direction"] == "bwd"
    assert alerts[0]["confirmation_date"] == "20261104"
    assert regressions.detect(history() + [night("20261104", 85), night("20261109", 100)]) == []


def test_cohort_confirmation_requires_the_same_cases_on_both_nights():
    nights = history()
    for n in nights:
        n["ops"]["flash_attn_mxfp8"]["cases"] += night(n["date"],
                                                       keys=("d", "e", "f"))["ops"]["flash_attn_mxfp8"]["cases"]
    nights += [night("20261104", 85), night("20261105", 85, keys=("d", "e", "f"))]
    assert regressions.detect(nights) == []


@pytest.mark.parametrize("status,notes", [("noisy", []), ("pip", ["CV 5.0% is also over the 3% limit"])])
def test_noisy_cases_need_two_severe_measurements(status, notes):
    bad = night("20261104", 30, status=status)
    for c in bad["ops"]["flash_attn_mxfp8"]["cases"]:
        c["notes"] = notes
    assert regressions.detect(history() + [bad]) == []
    again = copy.deepcopy(bad)
    again["date"] = "20261105"
    alerts = regressions.detect(history() + [bad, again])
    assert len(alerts) == 1 and "flagged noisy" in alerts[0]["reasons"][0]


def test_new_cases_errors_and_old_or_different_timing_data_cannot_supply_a_baseline():
    assert regressions.detect(history()[:2] + [night("20261104", 30)]) == []
    assert regressions.detect(history() + [night("20261201", 30)]) == []
    assert regressions.detect(history() + [night("20261104", 30, mode="wallclock")]) == []
    assert regressions.detect(history() + [night("20261104", 30, status="error")]) == []
    assert regressions.detect(history() + [night("20261104", 30, keys=("new", ))]) == []
    changed_clocks = night("20261104", 30, governed=["unlocked"])
    assert len(regressions.detect(history() + [changed_clocks])) == 1


def test_issue_filing_lists_unindexed_issues_and_deduplicates_without_closing():
    alert = regressions.detect(history() + [night("20261104", 30)])[0]
    calls = []

    def api(method, path, fields=None, paginate=False):
        calls.append((method, path, fields, paginate))
        if method == "GET":
            return [{"title": regressions.issue_title(alert)}]
        raise AssertionError("An already tracked regression must not write")

    assert regressions.file_issues([alert], "o/r", "run", api) == []
    assert calls == [("GET", "repos/o/r/issues?state=open&per_page=100", None, True)]
    calls.clear()

    def create(method, path, fields=None, paginate=False):
        calls.append((method, path, fields))
        return [] if method == "GET" else {"html_url": "https://github.com/o/r/issues/1"}

    assert regressions.file_issues([alert, alert], "o/r", "run", create) == ["https://github.com/o/r/issues/1"]
    assert [c[0] for c in calls] == ["GET", "POST"]
    assert calls[1][2]["labels"] == ["nightly", "perf-regression"]
    assert "not confirmed blame" in calls[1][2]["body"]


def test_suspect_commits_include_relevant_dependency_changes_without_claiming_causation(monkeypatch):
    alert = regressions.detect(history() + [night("20261104", 30)])[0]
    alert["cases"][1]["baseline_dates"][0] = "20261031"
    alert["cases"][1]["baseline_sha"] = "oldest"
    found = [{"sha": "nvidia", "title": "NVIDIA change", "pr": 1, "files": ["third_party/nvidia/lib/foo.cpp"]},
             {"sha": "llvm", "title": "AMD codegen LLVM repin", "pr": 2, "files": ["cmake/meta-amd-llvm-info.json"]}, {
                 "sha": "kernel", "title": "MXFP8 backward change", "pr": 3, "files":
                 ["third_party/tlx/ops/kernels/flash_attn_mxfp8/gfx950_bwd_shared.py"]
             }, {"sha": "docs", "title": "Documentation", "pr": None, "files": ["README.md"]}]
    ranges = []

    def commits(base, head):
        ranges.append((base, head))
        return found

    monkeypatch.setattr(regressions, "commits", commits)
    enriched = regressions.add_suspects(alert)
    assert ranges == [("oldest", alert["sha"])]
    assert [c["sha"] for c in enriched["suspect_commits"]] == ["kernel", "llvm"]
    body = regressions.issue_body(enriched, "o/r", "run")
    assert "https://github.com/o/r/commit/kernel" in body
    assert "https://github.com/o/r/pull/2" in body and "AMD codegen LLVM repin" in body
    assert "This is not confirmed blame" in body
    monkeypatch.setattr(regressions, "commits", lambda base, head: None)
    assert "Commit history is unavailable" in regressions.issue_body(alert, "o/r", "run")


def test_local_replay_cannot_write_issues_without_explicit_opt_in(monkeypatch, tmp_path):
    for n in history() + [night("20261104", 30)]:
        (tmp_path / f"{n['date']}.mi350.json").write_text(json.dumps(n))
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["detector", str(tmp_path), "--platform", "mi350", "--date", "20261104", "--out",
         str(output)])
    monkeypatch.setattr(regressions, "file_issues", lambda *args: pytest.fail("Unasked shared write"))
    regressions.main()
    assert len(json.loads(output.read_text())) == 1
    monkeypatch.setattr(
        "sys.argv",
        ["detector",
         str(tmp_path), "--platform", "mi350", "--date", "20261103", "--out",
         str(output), "--file-issues"])
    regressions.main()
    assert json.loads(output.read_text()) == []
