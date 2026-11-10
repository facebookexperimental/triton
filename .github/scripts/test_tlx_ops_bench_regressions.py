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


@pytest.mark.parametrize("value,status,notes,expected", [
    (70, "ok", [], 1),
    (70, "pip", ["speedup below the reference floor"], 1),
    (70.01, "ok", [], 0),
    (60, "noisy", [], 0),
    (60, "pip", ["CV 5.0% is also over the 3% limit"], 0),
])
def test_one_stable_reliable_case_losing_thirty_percent_alerts_without_cohort_confirmation(
        value, status, notes, expected):
    current = night("20261104", keys=("a", "b", "c", "d", "e"))
    baseline = [night(f"202611{day:02d}", keys=("a", "b", "c", "d", "e")) for day in range(1, 4)]
    current["ops"]["flash_attn_mxfp8"]["cases"][0].update(tlx_tflops=value, status=status, notes=notes)
    alerts = regressions.detect(baseline + [current])
    assert len(alerts) == expected
    if expected:
        assert [c["key"] for c in alerts[0]["cases"]] == ["a"]
        assert alerts[0]["confirmation_date"] is None
        assert regressions.bisect_spec(alerts[0])["required_ratio"] == 0.7
    else:
        again = copy.deepcopy(current)
        again["date"] = "20261105"
        assert regressions.detect(baseline + [current, again]) == []


def test_an_unstable_baseline_cannot_trigger_a_thirty_percent_single_night_alert():
    baseline = [night(f"202611{day:02d}", value) for day, value in enumerate((100, 160, 100), 1)]
    assert regressions.detect(baseline + [night("20261104", 70)]) == []
    assert len(regressions.detect(baseline + [night("20261104", 50)])) == 1


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
        return [] if method == "GET" else {"html_url": "https://github.com/o/r/issues/1", "number": 1}

    assert regressions.file_issues([alert, alert], "o/r", "run", create) == ["https://github.com/o/r/issues/1"]
    assert [c[:2] for c in calls[:2]] == [("GET", "repos/o/r/issues?state=open&per_page=100"),
                                          ("GET", "repos/o/r/issues?state=all&labels=perf-regression&per_page=100")]
    assert [c[0] for c in calls] == ["GET", "GET", "POST"]
    assert calls[2][2]["labels"] == ["nightly", "perf-regression"]
    assert "Suspect commit range" in calls[2][2]["body"]
    assert alert["bisect_issue"] == 1


def test_bisect_uses_a_good_baseline_not_the_already_regressed_previous_night():
    alert = regressions.detect(history() + [night("20261104", 85), night("20261105", 85)])[0]
    spec = regressions.bisect_spec(alert)
    assert spec["good_sha"] == "20261103" * 5
    assert spec["bad_sha"] == "20261105" * 5
    assert spec["required_ratio"] == regressions.COHORT_RATIO
    assert "ranked" not in regressions.issue_body(alert, "o/r", "run")


def test_only_new_regression_issues_queue_one_representative_bisect_job():
    alert = regressions.detect(history() + [night("20261104", 30)])[0]
    calls = []
    api = lambda method, path, fields=None: calls.append((method, path, fields))
    regressions.queue_bisects([alert], "o/r", 20, "evidence", api)
    assert calls == []
    alert.update(bisect_issue=1, bisect=regressions.bisect_spec(alert))
    other = {**alert, "bisect_issue": 2}
    regressions.queue_bisects([alert, other], "o/r", 20, "evidence", api)
    assert len(calls) == 1
    assert calls[0] == ("POST", "repos/o/r/actions/workflows/bisect-tlx-ops-regression.yml/dispatches", {
        "ref": "main", "inputs": {"source_run_id": "20", "artifact_name": "evidence", "platform": "mi350"}
    })


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


def nights_of(values, mode="gpu_events", start=1):
    """Consecutive nights of one fwd mm cohort; each entry maps case key to TFLOP/s."""
    out = []
    for i, by_key in enumerate(values, start):
        date = f"202610{i:02d}"
        cases = [{
            "key": key, "params": {"N": key}, "dtype": "bf16", "direction": "fwd", "status": "ok", "tlx_tflops": value,
            "ref_tflops": 20, "notes": []
        } for key, value in by_key.items()]
        out.append({
            "schema": 2, "platform": "mi350", "date": date, "sha": date * 5, "timing":
            {"latency_mode": mode, "replicates": 1}, "ops": {"mm": {"cases": cases}}
        })
    return out


@pytest.mark.parametrize("prior_status,statuses,detectors", [
    ("ok", ("ok", "ok"), ["recent", "historical-best"]),
    ("ok", ("ok", "noisy"), ["recent"]),
    ("ok", ("noisy", "noisy", "ok"), ["recent"]),
    ("noisy", ("noisy", "noisy", "ok", "ok"), ["recent"]),
])
def test_one_episode_is_filed_once_across_detectors_and_after_a_person_closes_it(prior_status, statuses, detectors):
    baseline = history()
    for case in baseline[-1]["ops"]["flash_attn_mxfp8"]["cases"]:
        case["status"] = prior_status
    alerts = regressions.detect_all(
        baseline + [night(f"202611{day:02d}", 30, status=status) for day, status in enumerate(statuses, 4)])
    first, second = alerts[0], alerts[-1]
    assert second["detectors"] == detectors
    assert second["episodes"] == first["episodes"] and all(
        e.endswith("|gpu_events,1|20261104") for e in first["episodes"])
    posts = []

    def api(method, path, fields=None, paginate=False):
        if method == "POST":
            posts.append(fields)
            return {"html_url": "https://github.com/o/r/issues/1", "number": 1}
        if "state=all" in path and posts:
            return [{"number": 1, "state": "closed", "title": posts[0]["title"], "body": posts[0]["body"]}]
        return []

    regressions.file_issues([first], "o/r", "run", api)
    regressions.file_issues([second], "o/r", "run", api)
    assert len(posts) == 1 and second["tracked_by"] == [1]


def test_historical_best_reports_a_lasting_loss_in_one_stable_case_but_not_in_a_flapping_case():
    steady = {k: 100 for k in "bcde"}
    flapping = [100, 130, 100, 130, 100, 80, 80]
    stable = [100, 100, 100, 100, 100, 80, 80]
    alerts = regressions.detect_all(nights_of([{"a": a, "f": f, **steady} for a, f in zip(stable, flapping)]))
    assert [(a["date"], a["detectors"], [c["key"]
                                         for c in a["cases"]])
            for a in alerts] == [("20261007", ["historical-best"], ["a"])]


def test_historical_best_skips_missing_nights_but_never_crosses_a_timing_mode():
    gaps = [night("20261104", status="error"), night("20261105", keys=())]
    low = [night("20261106", 80), night("20261107", 80)]
    assert [a["date"] for a in regressions.detect_best(history() + gaps + low)] == ["20261107"]
    other_mode = [night("20261106", 80, mode="wallclock"), night("20261107", 80, mode="wallclock")]
    assert regressions.detect_best(history() + gaps + other_mode) == []
    old_mode = [night(f"202610{d:02d}", v, mode="wallclock") for d, v in zip(range(20, 25), [100, 130, 100, 130, 100])]
    assert [a["date"] for a in regressions.detect_best(old_mode + history() + low)] == ["20261107"]


def test_one_fast_night_cannot_raise_the_historical_best():
    tracked = regressions.track_best(nights_of([{"a": v} for v in [100, 100, 200, 100, 100, 88]]))
    assert tracked[("mm", "fwd", "a")]["20261006"]["best"] == 100


def test_bisect_from_the_nearest_good_night_can_still_reproduce_the_loss():
    alert = regressions.detect_best(nights_of([{"a": v} for v in [100, 100, 100, 92, 92, 84, 84]]))[0]
    spec = regressions.bisect_spec(alert)
    assert (spec["good_sha"], spec["bad_sha"]) == ("20261005" * 5, "20261007" * 5)
    assert 84 / 92 <= spec["required_ratio"] < 1
