import json

import ci_health as ch

import subprocess


def fake_api(runs_by_wf, jobs_by_id, remaining=5000, pulls=None):
    calls = []
    pulls = pulls or {}

    def api(path):
        calls.append(path)
        if path.startswith("rate_limit"):
            return [{"resources": {"core": {"remaining": remaining}}}]
        if "/jobs" in path:
            rid = int(path.split("/runs/")[1].split("/")[0])
            return [{"jobs": jobs_by_id[rid]}]
        if "/pulls" in path:
            return [pulls[path.split("/pulls", 1)[1]]]
        wf = path.split("/workflows/")[1].split("/")[0]
        return [{"workflow_runs": runs_by_wf.get(wf, [])}]

    api.calls = calls
    return api


def fake_text(logs):
    calls = []

    def text(path):
        calls.append(path)
        jid = int(path.split("/jobs/")[1].split("/")[0])
        if jid not in logs:
            raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 410: logs expired")
        return logs[jid]

    text.calls = calls
    return text


def gh_run(rid, created, status="completed", conclusion="success", event="pull_request", title="t", **kw):
    return {
        "id": rid, "event": event, "status": status, "conclusion": conclusion, "created_at": created, "head_sha":
        "a" * 40, "display_title": title, "html_url": "u", "head_branch": "b", "path":
        ".github/workflows/gpu-tests.yml", "actor": {"login": "alice"}, "pull_requests": [], "head_repository":
        {"owner": {"login": "alice"}}, **kw
    }


def gh_job(jid, name, conclusion="success", steps=None):
    return {
        "id":
        jid, "name":
        name, "conclusion":
        conclusion, "created_at":
        "2026-10-12T00:59:00Z", "started_at":
        "2026-10-12T01:00:00Z", "completed_at":
        "2026-10-12T01:05:00Z", "labels": ["linux-gcp-h100"], "runner_name":
        "r1", "steps":
        steps or [{
            "name": "Run TLX core tests", "conclusion": conclusion, "number": 1, "started_at": "2026-10-12T01:01:00Z",
            "completed_at": "2026-10-12T01:04:00Z"
        }]
    }


JOBS = {1: [gh_job(11, "h100 / h100-tlx-test")]}
NOW = "2026-10-12T03:00:00Z"


def collect(tmp_path, runs, jobs=JOBS, since="2026-10-12T00:00:00Z", now=NOW, logs=None, pulls=None, remaining=5000):
    api, text = fake_api(runs, jobs, remaining, pulls), fake_text(logs or {})
    state = ch.collect(str(tmp_path), since, now, api=api, text=text)
    return state, api, text


def test_collect_stores_completed_runs_with_jobs_and_keeps_cursor_at_running(tmp_path):
    runs = {
        "gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z"),
                          gh_run(2, "2026-10-12T02:00:00Z", "in_progress", None)]
    }
    state, _, _ = collect(tmp_path, runs)
    stored = ch.load_runs(str(tmp_path))
    job = stored[1]["jobs"][0]
    assert job["steps"] == [{
        "name": "Run TLX core tests", "conclusion": "success", "started_at": "2026-10-12T01:01:00Z", "completed_at":
        "2026-10-12T01:04:00Z"
    }]
    assert job["created_at"] == "2026-10-12T00:59:00Z" and "runner_name" not in job
    assert stored[2]["jobs"] is None and state["cursor"] == "2026-10-12T02:00:00Z"
    assert json.loads((tmp_path / "state.json").read_text())["collected_at"] == NOW
    assert (tmp_path / "changed.txt").read_text().split() == [str(tmp_path / "runs-2026-10.jsonl.gz")]


def test_collect_lists_gpu_and_nightly_runs_only(tmp_path):
    _, api, _ = collect(tmp_path, {})
    listed = [c for c in api.calls if "/workflows/" in c]
    assert {c.split("/workflows/")[1].split("/")[0]
            for c in listed
            } == {"gpu-tests.yml", "gpu-tests-nightly.yml", "nightly-tlx-ops-bench.yml", "nightly-fbtriton.yml"}


def test_select_log_gives_the_runs_selection(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z")]}
    jobs = {1: [gh_job(10, "select"), gh_job(11, "h100 / h100-tlx-test")]}
    sel = "2026-10-12T01:00:30.1Z GPU selection for a: {'h100': True, 'b200': False, 'mi350': False, 'build': True}\n"
    collect(tmp_path, runs, jobs, logs={10: sel})
    assert ch.load_runs(
        str(tmp_path))[1]["select"] == {"h100": True, "b200": False, "mi350": False, "build": True, "fallback": False}


def test_collect_does_not_refetch_jobs_of_stored_completed_runs(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z")]}
    collect(tmp_path, runs)
    _, api, _ = collect(tmp_path, runs, since=None, now="2026-10-12T04:00:00Z")
    assert not [c for c in api.calls if "/jobs" in c]
    assert (tmp_path / "changed.txt").read_text().split() == []


def test_failed_job_log_is_fetched_once_and_failures_stored(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", event="push")]}
    jobs = {
        1: [
            gh_job(11, "b200 / b200-tlx-test", "failure"),
            gh_job(12, "b200 / b200-gluon-test"),
            gh_job(13, "GPU tests", "failure")
        ]
    }
    log = "2026-10-12T01:02:00.1234567Z FAILED python/test/unit/language/test_tlx_dot_sm100.py::test_async_dot_scaled_2cta[m64] - AssertionError: x\n"
    _, _, text = collect(tmp_path, runs, jobs, logs={11: log})
    step = ch.load_runs(str(tmp_path))[1]["jobs"][0]["steps"][0]
    assert step["failures"] == [{
        "id": "python/test/unit/language/test_tlx_dot_sm100.py::test_async_dot_scaled_2cta", "msg": "AssertionError: x"
    }]
    assert text.calls == ["repos/facebookexperimental/triton/actions/jobs/11/logs"]


def test_expired_log_leaves_failures_absent_and_run_complete(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure")]}
    jobs = {1: [gh_job(11, "b200 / b200-tlx-test", "failure")]}
    collect(tmp_path, runs, jobs)
    stored = ch.load_runs(str(tmp_path))[1]
    assert stored["jobs"] is not None and "failures" not in stored["jobs"][0]["steps"][0]
    _, _, text = collect(tmp_path, runs, jobs, since=None, now="2026-10-12T04:00:00Z")
    assert text.calls == []


def test_push_author_comes_from_the_pr_in_the_title(tmp_path):
    runs = {
        "gpu-tests.yml": [
            gh_run(1, "2026-10-12T01:00:00Z", event="push", title="[TLX] fix (#3972)",
                   actor={"login": "meta-codesync[bot]"})
        ]
    }
    collect(tmp_path, runs, pulls={"/3972": {"number": 3972, "user": {"login": "bob"}}})
    stored = ch.load_runs(str(tmp_path))[1]
    assert (stored["pr"], stored["author"]) == (3972, "bob")


def test_pr_run_without_pull_requests_looks_up_by_head(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z", head_branch="feat")]}
    collect(tmp_path, runs, pulls={"?head=alice:feat&state=all": [{"number": 4150, "user": {"login": "alice"}}]})
    stored = ch.load_runs(str(tmp_path))[1]
    assert (stored["pr"], stored["author"]) == (4150, "alice")


def test_collect_stops_before_the_rate_limit(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z")]}
    state, _, _ = collect(tmp_path, runs, remaining=50)
    assert ch.load_runs(str(tmp_path))[1]["jobs"] is None and state["cursor"] == "2026-10-12T01:00:00Z"


def test_later_line_wins_and_only_changed_months_are_rewritten(tmp_path):
    a = {"id": 1, "created_at": "2026-09-30T00:00:00Z", "conclusion": None}
    b = {"id": 2, "created_at": "2026-10-01T00:00:00Z", "conclusion": "success"}
    ch.save_runs(str(tmp_path), {1: a, 2: b})
    changed = ch.save_runs(str(tmp_path), {1: {**a, "conclusion": "success"}, 2: b})
    assert [p.rsplit("/", 1)[1] for p in changed] == ["runs-2026-09.jsonl.gz"]
    assert ch.load_runs(str(tmp_path))[1]["conclusion"] == "success"


def _store_one_push(tmp_path):
    ch.save_runs(
        str(tmp_path), {
            1: {
                "id": 1, "wf": "gpu-tests.yml", "event": "push", "status": "completed", "conclusion": "success",
                "created_at": "2026-10-12T01:00:00Z", "head_sha": "a" * 40, "title": "t", "url": "u", "branch": "main",
                "pr": None, "author": None, "jobs": []
            }
        })
    (tmp_path / "state.json").write_text(json.dumps({"cursor": "x", "collected_at": "2026-10-12T02:00:00Z"}))


def test_render_writes_every_section(tmp_path):
    _store_one_push(tmp_path)
    (tmp_path / "bench").mkdir()
    out = tmp_path / "site"
    ch.render(str(tmp_path), str(out), bench_dir=str(tmp_path / "bench"))
    page = (out / "index.html").read_text()
    data = json.loads((out / "data.json").read_text())
    assert 'fetch("data.json"' in page and 'data-tab="ops"' in page
    assert data["asof"] == "2026-10-12T02:00:00Z" and set(data) == {"asof", "branches"}


def test_render_survives_bench_failure(tmp_path, monkeypatch):
    _store_one_push(tmp_path)

    def boom():
        raise RuntimeError("no gh")

    monkeypatch.setattr(ch.bench, "load_summaries", boom)
    out = tmp_path / "site"
    ch.render(str(tmp_path), str(out))
    assert json.loads((out / "data.json").read_text())["branches"]["main"]["nightly"]["perf"] == []


def test_render_without_data_still_writes_a_page(tmp_path):
    out = tmp_path / "site"
    ch.render(str(tmp_path / "missing"), str(out), bench_dir=str(tmp_path))
    assert json.loads((out / "data.json").read_text())["asof"] is None


def test_a_run_stuck_in_pending_does_not_pin_the_cursor(tmp_path):
    runs = {"gpu-tests.yml": [gh_run(1, "2026-10-01T01:00:00Z", "pending", None)]}
    state, _, _ = collect(tmp_path, runs, since="2026-09-30T00:00:00Z")
    assert state["cursor"] == "2026-10-12T03:00:00Z"
