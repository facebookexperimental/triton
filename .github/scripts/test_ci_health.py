import json

import ci_health as ch


def fake_api(runs_by_wf, jobs_by_id, remaining=5000):
    calls = []

    def api(path):
        calls.append(path)
        if path.startswith("rate_limit"):
            return [{"resources": {"core": {"remaining": remaining}}}]
        if "/jobs" in path:
            rid = int(path.split("/runs/")[1].split("/")[0])
            return [{"jobs": jobs_by_id[rid]}]
        wf = path.split("/workflows/")[1].split("/")[0]
        return [{"workflow_runs": runs_by_wf.get(wf, [])}]

    api.calls = calls
    return api


def gh_run(rid, created, status="completed", conclusion="success"):
    return {
        "id": rid, "event": "pull_request", "status": status, "conclusion": conclusion, "created_at": created,
        "head_sha": "a" * 40, "display_title": "t", "html_url": "u", "head_branch": "b", "path":
        ".github/workflows/h100.yml"
    }


JOBS = {
    1: [{
        "name": "h100-tlx-test", "conclusion": "success", "started_at": "2026-10-12T01:00:00Z", "completed_at":
        "2026-10-12T01:05:00Z", "labels": ["linux-gcp-h100"], "runner_name": "r1", "steps":
        [{"name": "Run TLX core tests", "conclusion": "success", "number": 1}]
    }]
}


def test_collect_stores_completed_runs_with_jobs_and_keeps_cursor_at_running(tmp_path):
    runs = {"h100.yml": [gh_run(1, "2026-10-12T01:00:00Z"), gh_run(2, "2026-10-12T02:00:00Z", "in_progress", None)]}
    state = ch.collect(str(tmp_path), "2026-10-12T00:00:00Z", "2026-10-12T03:00:00Z", api=fake_api(runs, JOBS))
    stored = ch.load_runs(str(tmp_path))
    assert stored[1]["jobs"][0]["steps"] == [{"name": "Run TLX core tests", "conclusion": "success"}]
    assert "runner_name" not in stored[1]["jobs"][0]
    assert stored[2]["jobs"] is None and state["cursor"] == "2026-10-12T02:00:00Z"
    assert json.loads((tmp_path / "state.json").read_text())["collected_at"] == "2026-10-12T03:00:00Z"
    assert (tmp_path / "changed.txt").read_text().split() == [str(tmp_path / "runs-2026-10.jsonl.gz")]


def test_collect_does_not_refetch_jobs_of_stored_completed_runs(tmp_path):
    runs = {"h100.yml": [gh_run(1, "2026-10-12T01:00:00Z")]}
    ch.collect(str(tmp_path), "2026-10-12T00:00:00Z", "2026-10-12T03:00:00Z", api=fake_api(runs, JOBS))
    api = fake_api(runs, JOBS)
    ch.collect(str(tmp_path), None, "2026-10-12T04:00:00Z", api=api)
    assert not [c for c in api.calls if "/jobs" in c]
    assert (tmp_path / "changed.txt").read_text().split() == []


def test_collect_stops_before_the_rate_limit(tmp_path):
    runs = {"h100.yml": [gh_run(1, "2026-10-12T01:00:00Z")]}
    state = ch.collect(str(tmp_path), "2026-10-12T00:00:00Z", "2026-10-12T03:00:00Z", api=fake_api(runs, JOBS, 50))
    assert ch.load_runs(str(tmp_path))[1]["jobs"] is None and state["cursor"] == "2026-10-12T01:00:00Z"


def test_later_line_wins_and_only_changed_months_are_rewritten(tmp_path):
    a = {"id": 1, "created_at": "2026-09-30T00:00:00Z", "conclusion": None}
    b = {"id": 2, "created_at": "2026-10-01T00:00:00Z", "conclusion": "success"}
    ch.save_runs(str(tmp_path), {1: a, 2: b})
    changed = ch.save_runs(str(tmp_path), {1: {**a, "conclusion": "success"}, 2: b})
    assert [p.rsplit("/", 1)[1] for p in changed] == ["runs-2026-09.jsonl.gz"]
    assert ch.load_runs(str(tmp_path))[1]["conclusion"] == "success"


def test_render_writes_shell_and_data(tmp_path):
    ch.save_runs(
        str(tmp_path), {
            1: {
                "id": 1, "wf": "h100.yml", "event": "schedule", "status": "completed", "conclusion": "success",
                "created_at": "2026-10-12T01:00:00Z", "head_sha": "a" * 40, "title": "t", "url": "u", "branch": "main",
                "jobs": []
            }
        })
    (tmp_path / "state.json").write_text(json.dumps({"cursor": "x", "collected_at": "2026-10-12T02:00:00Z"}))
    out = tmp_path / "site"
    ch.render(str(tmp_path), str(out), changelog=[])
    page = (out / "index.html").read_text()
    data = json.loads((out / "data.json").read_text())
    assert 'fetch("data.json"' in page and "%DATA%" not in page
    assert data["asof"] == "2026-10-12T02:00:00Z" and {"strip", "pr24", "runner_min"} <= set(data)


def test_render_without_data_still_writes_a_page(tmp_path):
    out = tmp_path / "site"
    ch.render(str(tmp_path / "missing"), str(out), changelog=[])
    assert json.loads((out / "data.json").read_text())["asof"] is None


def test_a_run_stuck_in_pending_does_not_pin_the_cursor(tmp_path):
    runs = {"h100.yml": [gh_run(1, "2026-10-01T01:00:00Z", "pending", None)]}
    state = ch.collect(str(tmp_path), "2026-09-30T00:00:00Z", "2026-10-12T03:00:00Z", api=fake_api(runs, JOBS))
    assert state["cursor"] == "2026-10-12T03:00:00Z"
