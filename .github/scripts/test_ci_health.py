import base64
import json
import re
from pathlib import Path

import pytest
import yaml

import ci_health as ch

import subprocess


def test_sanitizer_collection_cannot_file_or_reconcile_issues():
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/cutracer.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    assert jobs["b200"]["with"]["file_issues"] is False
    assert jobs["h100"]["with"]["file_issues"] is False


def test_sanitizer_transient_artifact_failure_retries_with_cached_jobs(tmp_path):
    created = "2026-10-12T01:00:00Z"
    run = gh_run(1, created, event="workflow_dispatch", head_branch="main")
    api = fake_api({"cutracer.yml": [run]}, {1: [gh_job(11, "b200 / sanitize")]})
    artifact_calls = []

    def fetch(path):
        if "/artifacts" in path:
            artifact_calls.append(path)
            raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 503")
        return api(path)

    ch.collect(str(tmp_path), created, NOW, api=fetch, get=no_get)
    assert ch.load_runs(str(tmp_path))[1]["sanitizer"]["retry"] is True
    ch.collect(str(tmp_path), None, NOW, api=fetch, get=no_get)
    assert len(artifact_calls) == 2
    assert sum("/jobs" in call for call in api.calls) == 1


def fake_api(runs_by_wf, jobs_by_id, remaining=5000, pulls=None, action_runs=None):
    calls = []
    pulls = pulls or {}
    action_runs = action_runs or {}

    def api(path):
        calls.append(path)
        if path.startswith("rate_limit"):
            return [{"resources": {"core": {"remaining": remaining}}}]
        if "/jobs" in path:
            rid = int(path.split("/runs/")[1].split("/")[0])
            return [{"jobs": jobs_by_id[rid]}]
        if "/actions/runs/" in path and "/attempts/" in path:
            rid = int(path.split("/runs/")[1].split("/")[0])
            attempt = int(path.rsplit("/", 1)[1])
            return [action_runs[(rid, attempt)]]
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


def no_get(path):
    raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 404")


def collect(tmp_path, runs, jobs=JOBS, since="2026-10-12T00:00:00Z", now=NOW, logs=None, pulls=None, remaining=5000,
            get=no_get, action_runs=None):
    api, text = fake_api(runs, jobs, remaining, pulls, action_runs), fake_text(logs or {})
    state = ch.collect(str(tmp_path), since, now, api=api, text=text, get=get)
    return state, api, text


def workflow_test_jobs(workflow="gpu-tests.yml"):
    root = Path(__file__).resolve().parents[2]
    parent = yaml.load((root / ".github/workflows" / workflow).read_text(), Loader=yaml.BaseLoader)
    found = []
    for key, job in parent["jobs"].items():
        if "uses" in job:
            children = yaml.load((root / job["uses"]).read_text(), Loader=yaml.BaseLoader)["jobs"]
            platforms = job.get("strategy", {}).get("matrix", {}).get("platform", [None])
            candidates = []
            for platform in platforms:
                prefix = f"{key} ({platform})" if platform else key
                for name, child in children.items():
                    name = child.get("name", name)
                    if platform:
                        name = name.replace("${{ inputs.platform }}", platform)
                    candidates.append((f"{prefix} / {name}", child))
        else:
            candidates = [(job.get("name", key), job)]
        for name, child in candidates:
            if child.get("if") in ("${{ false }}", "false"):
                continue
            tests = [
                step["name"]
                for step in child.get("steps", [])
                if step.get("name", "").startswith("Run ") or step.get("uses") == "./.github/actions/run-selected-tests"
                or "run_selected_tests.py" in step.get("run", "") or "check-triton-lit-tests" in step.get("run", "")
                or re.search(r"(?m)^\s*(?:python\S*\s+-m\s+)?pytest\b", step.get("run", ""))
            ]
            if tests:
                found.append((name, child["runs-on"], tests))
    return found


@pytest.mark.parametrize("name,label,steps", workflow_test_jobs())
def test_workflow_test_jobs_reach_dashboard_health(tmp_path, name, label, steps):
    rules = ch.rules
    platform = rules.test_platform_of({"wf": "gpu-tests.yml"}, {"name": name})
    assert platform is not None, f"Dashboard does not recognize test job {name}"
    for step in steps:
        assert rules.group_of(step) is not None, f"Dashboard does not recognize test step {name}: {step}"
    test_steps = [{
        "name": step, "conclusion": "failure", "number": i + 1, "started_at": f"2026-10-12T01:01:{i * 10:02d}Z",
        "completed_at": f"2026-10-12T01:01:{i * 10 + 9:02d}Z"
    } for i, step in enumerate(steps)]
    j = gh_job(20, name, "failure", test_steps)
    j["labels"] = [label]
    text = "".join(
        f"2026-10-12T01:01:{i * 10 + 5:02d}Z FAILED test_contract.py::test_signal_{i} - AssertionError: signal\n"
        for i in range(len(steps)))
    collect(
        tmp_path,
        {"gpu-tests.yml": [gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", event="push", head_branch="main")]},
        {1: [j]}, logs={20: text})
    stored = ch.load_runs(str(tmp_path))[1]
    assert rules.run_outcome(stored) == "test"
    view = rules.main_view([stored], NOW)
    assert len(view["gpus"][platform]["new"]) == len(steps)
    assert view["pushes"][0]["cells"][platform] == "new"
    assert len(view["steps"][platform]) == len(steps)
    assert all(row["cells"] == ["new"] for row in view["steps"][platform])
    assert {item["step"] for item in view["gpus"][platform]["new"]} == set(steps)
    assert rules.gpu_minutes(stored) == (0 if platform == "CPU" else 5)
    pr = rules.pr_view([{**stored, "event": "pull_request", "pr": 1}], NOW, rules.main_view([], NOW))[0]
    assert pr["result"] == "tests" and pr["n_failed"] == len(steps)
    assert platform in pr["gpus"]
    out = tmp_path / "site"
    ch.render(str(tmp_path), str(out), bench_dir=str(tmp_path / "bench"))
    data = json.loads((out / "data.json").read_text())
    assert data["branches"]["main"]["main"]["gpus"][platform]["new"]
    assert "CPU compiler tests" in (out / "index.html").read_text()


@pytest.mark.parametrize("workflow", ["gpu-tests-nightly.yml", "nightly-tlx-ops-bench.yml"])
def test_nightly_workflow_jobs_are_visible_to_dashboard(workflow):
    jobs = workflow_test_jobs(workflow)
    assert jobs
    for name, _, steps in jobs:
        platform = ch.rules.test_platform_of({"wf": workflow}, {"name": name})
        assert platform in ch.rules.GPUS, f"Dashboard does not recognize {workflow}: {name}"
        assert all(ch.rules.group_of(step) for step in steps), (workflow, name, steps)


def test_compiler_workflow_job_reaches_lit_dashboard():
    jobs = workflow_test_jobs(ch.rules.LIT_WF)
    assert len(jobs) == 1
    name, _, steps = jobs[0]
    assert name == ch.rules.LIT_JOB and steps
    j = gh_job(1, name, "failure", [{"name": steps[0], "conclusion": "failure", "failures": [{"id": "lit-test"}]}])
    assert ch.rules.lit_result({"jobs": [j]}) == ("fail", [{"id": "lit-test"}])


def test_lit_cache_preserves_compiler_and_full_test_coverage():
    import shlex
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/compiler.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["lit-tests"]
    steps = {step["name"]: step for step in job["steps"]}
    run = steps["Build and run LIT tests"]["run"]
    configure = shlex.split(run)
    assert "-DTRITON_CODEGEN_BACKENDS=nvidia;amd;cpu" in configure
    assert "-DTRITON_BUILD_UT=OFF" in configure
    assert "-DLLVM_EXTERNAL_LIT=$(which lit)" in configure
    assert not any(arg.startswith(("-DCMAKE_C_COMPILER=", "-DCMAKE_CXX_COMPILER=")) for arg in configure)
    assert not {"CC", "CXX"} & job.get("env", {}).keys()
    assert "ninja -C build check-triton-lit-tests" in run
    assert "set -o pipefail" in run
    assert job["env"]["CCACHE_DIR"].endswith("/triton-lit-ccache")
    assert steps["Restore LIT ccache"]["with"]["path"] == steps["Save LIT ccache"]["with"][
        "path"] == "${{ env.CCACHE_DIR }}"


def test_fast_workflow_preserves_dashboard_and_gpu_usage_contract(tmp_path):
    workflow = ch.rules.FAST_WF
    workers = [job for job in workflow_test_jobs(workflow) if not job[0].startswith('full / ')]
    assert len(workers) == 4
    for name, label, steps in workers:
        platform = ch.rules.test_platform_of({"wf": workflow}, {"name": name})
        assert platform in ch.rules.TEST_PLATFORMS
        assert all(ch.rules.group_of(step) for step in steps)
        item = gh_job(1, name, "failure", [{
            "name": steps[0], "conclusion": "failure", "number": 1, "started_at": "2026-10-12T01:01:00Z",
            "completed_at": "2026-10-12T01:04:00Z"
        }])
        item["labels"] = ["ubuntu-latest" if platform == "CPU" else platform]
        stored = ch._record(gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", event="push", head_branch="main"),
                            workflow)
        stored["jobs"] = ch._jobs([{"jobs": [item]}])
        ch._attach_failures(stored["jobs"],
                            fake_text({1: "2026-10-12T01:02:00Z FAILED smoke.py::test_required - AssertionError: x"}),
                            workflow)
        assert ch.rules.run_outcome(stored) == "test"
        assert ch.rules.gpu_minutes(stored) == (0 if platform == "CPU" else 5)
        assert ch.rules.main_view([stored], NOW) == ch.rules.main_view([], NOW)
        assert ch.rules.pr_view([stored], NOW, ch.rules.main_view([], NOW)) == []


def test_dashboard_contract_runs_when_its_inputs_change():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/ci-health.yml").read_text(), Loader=yaml.BaseLoader)
    paths = workflow["on"]["pull_request"]["paths"]
    import fnmatch
    inputs = [
        ".github/workflows/gpu-tests.yml", ".github/scripts/gpu_test_selection.py", ".github/gpu-test-suites.json",
        ".github/actions/run-selected-tests/action.yml", ".github/scripts/test_gpu_ci.py",
        ".github/scripts/test_kernel_test_selection.py", ".github/scripts/wheel_cache.py"
    ]
    inputs += [
        ".github/workflows/" + name for name in (*ch.rules.GPU_WORKFLOWS, *ch.rules.GPU_RUN_WORKFLOWS, ch.rules.LIT_WF,
                                                 "tritonbench.yml", "_tlx-ops-bench.yml")
    ]
    for path in inputs:
        assert any(fnmatch.fnmatch(path, pattern) for pattern in paths), f"CI does not test dashboard changes to {path}"


def test_selector_output_remains_readable_by_dashboard(tmp_path):
    import os
    import sys
    from gpu_test_selection import REPO_ROOT
    event, output = tmp_path / "event.json", tmp_path / "outputs"
    event.write_text("{}")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github/scripts/gpu_test_selection.py")], cwd=REPO_ROOT, env=os.environ | {
            "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_EVENT_PATH": str(event), "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")
        }, capture_output=True, text=True, check=True)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    selected = ch.logs.parse_selection(proc.stdout)
    assert selected is not None, "Dashboard cannot parse the selector's real output"
    for key in ("h100", "b200", "mi350", "build", "cpu"):
        assert selected[key] == (outputs[key] == "true")
    assert selected["fallback"] is False


def source_log(sha):
    return ("2026-10-12T01:01:30Z [command]/usr/bin/git log -1 --format=%H\n"
            f"2026-10-12T01:01:30Z {sha}\n")


def checkout_job(jid, name):
    return gh_job(
        jid, name, steps=[{
            "name": "Checkout", "conclusion": "success", "started_at": "2026-10-12T01:01:00Z", "completed_at":
            "2026-10-12T01:02:00Z"
        }])


def followup_log(**changes):
    source = {
        "status": "ok", "merge": "2" * 40, "base": "3" * 40, "pr": 4855, "pr_head": "1" * 40, "base_ref": "main",
        "head_ref": "feature", "head_repo": "alice/triton", "fork": True, "producer_run_id": 7, "producer_attempt": 2,
        "wheel_reason": ""
    } | changes
    return "\n".join(f"2026-10-12T01:01:30Z {line}" for line in json.dumps(source, indent=2).splitlines())


def followup_pr(head_repo="alice/triton", head_ref="feature", base_repo=ch.REPO, base_ref="main"):
    return {
        "number": 4855, "user": {"login":
                                 "carol"}, "head": {"sha": "9" * 40, "ref": head_ref, "repo": {"full_name": head_repo}},
        "base": {"ref": base_ref, "repo": {"full_name": base_repo}}
    }


def followup_pulls(pr=None):
    return {
        "/4855": pr or followup_pr(),
        "?head=alice:feature&state=all": [{"number": 4855}, {"number": 4999}],
    }


def followup_collection_fixture(selected_sha="2" * 40, producer_head="1" * 40):
    producer = gh_run(7, "2026-10-12T00:30:00Z", event="pull_request", run_attempt=2, head_sha=producer_head,
                      head_branch="feature", pull_requests=[], path=".github/workflows/fast-gpu-tests.yml",
                      repository={"full_name":
                                  ch.REPO}, head_repository={"full_name": "alice/triton", "owner": {"login": "alice"}})
    full = gh_run(8, "2026-10-12T01:00:00Z", event="workflow_run", head_sha="d" * 40, head_branch="main",
                  title=f"GPU tests for Fast run 999 attempt 9 (PR head {'9' * 40})")
    verify = gh_job(
        80, "followup / verify", steps=[{
            "name": "Verify the producer", "conclusion": "success", "started_at": "2026-10-12T01:01:00Z",
            "completed_at": "2026-10-12T01:02:00Z"
        }])
    verify["labels"] = ["ubuntu-latest"]
    select = checkout_job(81, "select")
    select["steps"][0]["name"] = "Checkout the PR's test merge commit"
    worker = gh_job(82, "b200 / b200-tlx-test")
    aggregate = gh_job(83, "GPU tests")
    aggregate["labels"] = ["ubuntu-latest"]
    return full, producer, [verify, select, worker, aggregate], {80: followup_log(), 81: source_log(selected_sha)}


def test_collect_correlates_independent_full_run_from_verified_producer_and_selected_merge(tmp_path):
    full, producer, jobs, log = followup_collection_fixture()
    _, api, text = collect(tmp_path, {"gpu-tests.yml": [full]}, {8: jobs}, logs=log, pulls=followup_pulls(),
                           action_runs={(7, 2): producer})
    stored = ch.load_runs(str(tmp_path))[8]
    assert stored["event"] == "workflow_run" and stored["head_sha"] == "d" * 40
    assert (stored["followup_producer_run_id"], stored["followup_producer_attempt"]) == (7, 2)
    assert (stored["pr"], stored["author"], stored["pr_head_sha"], stored["pr_branch"]) == (4855, "carol", "1" * 40,
                                                                                            "feature")
    assert stored["tested_sha"] == stored["followup_merge"] == "2" * 40
    assert [item["source"] for item in stored["tested_sha_provenance"]
            ] == ["API-verified Fast producer and test merge", "selected test merge checkout log"]
    assert any("/actions/runs/7/attempts/2" in call for call in api.calls)
    assert any("/pulls/4855" in call for call in api.calls) and not any("/pulls?head=" in call for call in api.calls)
    assert text.calls.count("repos/facebookexperimental/triton/actions/jobs/80/logs") == 1


@pytest.mark.parametrize("producer_head,selected_sha", [("4" * 40, "2" * 40), ("1" * 40, "4" * 40)])
def test_collect_does_not_verify_a_wrong_producer_or_selected_merge(tmp_path, producer_head, selected_sha):
    full, producer, jobs, log = followup_collection_fixture(selected_sha, producer_head)
    collect(tmp_path, {"gpu-tests.yml": [full]}, {8: jobs}, logs=log, pulls=followup_pulls(), action_runs={(7, 2):
                                                                                                           producer})
    stored = ch.load_runs(str(tmp_path))[8]
    assert stored["tested_sha"] is None
    if producer_head != "1" * 40:
        assert stored.get("followup_status") is None and stored.get("pr") is None
    else:
        assert stored["followup_status"] == "ok" and {item["sha"]
                                                      for item in stored["tested_sha_provenance"]
                                                      } == {"2" * 40, "4" * 40}


@pytest.mark.parametrize("pr", [followup_pr(head_repo="other/triton"), followup_pr(head_ref="reused-feature")])
def test_collect_rejects_exact_pr_with_wrong_verified_repository_or_ref(tmp_path, pr):
    full, producer, jobs, log = followup_collection_fixture()
    collect(tmp_path, {"gpu-tests.yml": [full]}, {8: jobs}, logs=log, pulls=followup_pulls(pr), action_runs={(7, 2):
                                                                                                             producer})
    stored = ch.load_runs(str(tmp_path))[8]
    assert stored.get("followup_status") is None and stored.get("pr") is None and stored["tested_sha"] is None


def test_collect_hydrates_running_fast_parent_and_preserves_source_evidence(tmp_path):
    sha = 'ab' * 20
    raw = gh_run(1, '2026-10-12T01:00:00Z', status='in_progress', conclusion=None, pull_requests=[{'number': 4855}])
    jobs = [
        checkout_job(10, 'select'),
        checkout_job(11, 'build / build'),
        checkout_job(12, 'full / select'),
        gh_job(13, 'Fast correctness'),
        gh_job(14, 'full / mi350 / mi350-tlx-test', conclusion=None)
    ]
    jobs[3]['labels'] = ['ubuntu-latest']
    jobs[4]['completed_at'] = None
    _, api, _ = collect(tmp_path, {ch.rules.FAST_WF: [raw]}, {1: jobs}, logs={i: source_log(sha) for i in (10, 11, 12)})
    record = ch.load_runs(str(tmp_path))[1]
    assert record['status'] == 'in_progress' and len(record['jobs']) == 5
    assert record['tested_sha'] == sha and len(record['tested_sha_provenance']) == 2
    assert record['stage_provenance']['gpu-tests.yml']['sha'] == sha
    assert any('/runs/1/jobs?' in call for call in api.calls)


def test_ordered_stage_source_identity_does_not_borrow_missing_full_logs():
    sha = 'ab' * 20
    record = ch._record(gh_run(1, '2026-10-12T01:00:00Z'), ch.rules.FAST_WF)
    jobs = ch._jobs(
        [{'jobs': [checkout_job(10, 'select'),
                   checkout_job(11, 'build / build'),
                   checkout_job(12, 'full / select')]}])
    record['jobs'] = jobs
    ch._tested_revision(record, jobs, fake_text({10: source_log(sha), 11: source_log(sha)}))
    assert record['tested_sha'] == sha
    assert ch.rules.stage_runs(record)[1]['tested_sha'] is None
    assert ch.rules.stage_runs(record)[0]['tested_sha'] == sha


def test_collect_uses_same_attempt_checkout_evidence_for_both_workflows(tmp_path):
    sha = "a46d2568f456948b74a7e8da5823cca07c3ac370"
    raw = gh_run(1, "2026-10-12T01:00:00Z", pull_requests=[{"number": 4855}], run_attempt=1)
    fast = {**raw, "id": 2}
    jobs = {
        1: [checkout_job(10, "select"), checkout_job(11, "build / build")], 2:
        [checkout_job(20, "select"), checkout_job(21, "build / build")]
    }
    _, api, text = collect(tmp_path, {"gpu-tests.yml": [raw], ch.rules.FAST_WF: [fast]}, jobs,
                           logs={jid: source_log(sha)
                                 for jid in [10, 11, 20, 21]})
    records = ch.load_runs(str(tmp_path))
    for item in records.values():
        assert item["pr"] == 4855 and item["tested_sha"] == sha and item["run_attempt"] == 1
        assert {e["job"] for e in item["tested_sha_provenance"]} == {"select", "build / build"}
    assert len(text.calls) == len(set(text.calls)) == 4
    assert all("filter=latest" in call for call in api.calls if "/jobs?" in call)


@pytest.mark.parametrize("only_failed", [True, False])
def test_collect_refreshes_changed_attempt_without_erasing_history(tmp_path, only_failed):
    sha = "ab" * 20
    first = gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", run_attempt=1, pull_requests=[{"number": 4855}])
    jobs = {
        1:
        [checkout_job(10, "select"),
         checkout_job(11, "build / build"),
         gh_job(12, "b200 / b200-tlx-test", "failure")]
    }
    logs = {
        10: source_log(sha), 11: source_log(sha), 12:
        "2026-10-12T01:02:00Z FAILED boundary.py::test_bitwise[False] - AssertionError: bitwise"
    }
    collect(tmp_path, {"gpu-tests.yml": [first]}, jobs, logs=logs)
    retry = {**first, "run_attempt": 2, "conclusion": "success", "updated_at": "2026-10-12T04:00:00Z"}
    prior_jobs = jobs[1]
    jobs[1] = [checkout_job(20, "select"), checkout_job(21, "build / build"), gh_job(22, "b200 / b200-tlx-test")]
    for item in jobs[1]:
        item["run_attempt"] = 2
    jobs[1][2]["steps"].insert(0, checkout_job(22, "worker")["steps"][0])
    jobs[1] = prior_jobs + (jobs[1][2:] if only_failed else jobs[1])
    _, api, _ = collect(
        tmp_path, {"gpu-tests.yml": [retry]}, jobs, since=None, now="2026-10-12T05:00:00Z",
        logs={10: source_log(sha), 11: source_log(sha), 20: source_log(sha), 21: source_log(sha), 22: source_log(sha)})
    stored = ch.load_runs(str(tmp_path))[1]
    assert stored["run_attempt"] == 2 and stored["conclusion"] == "success" and stored["tested_sha"] == sha
    assert stored["attempts"][0]["run_attempt"] == 1 and stored["attempts"][0]["conclusion"] == "failure"
    assert stored["attempts"][0]["jobs"][2]["steps"][0]["test_failures"][0]["id"].endswith("[False]")
    assert any("filter=all" in call for call in api.calls if "/jobs?" in call)
    assert len(stored["jobs"]) == 3 and stored["jobs"][2]["id"] == 22
    assert any("created=>=2026-10-05T05:00:00Z" in call for call in api.calls if "/gpu-tests.yml/" in call)
    _, api, _ = collect(tmp_path, {"gpu-tests.yml": [retry]}, jobs, since=None, now="2026-10-12T06:00:00Z")
    assert not any("/jobs?" in call for call in api.calls)
    assert len(ch.load_runs(str(tmp_path))[1]["attempts"]) == 1


@pytest.mark.parametrize("wf", ["gpu-tests.yml", ch.rules.LIT_WF, ch.rules.BENCH_WF])
def test_collect_preserves_completed_push_history_on_rerun(tmp_path, wf):
    first = gh_run(1, "2026-10-12T01:00:00Z", event="push", run_attempt=1, head_branch="main")
    collect(tmp_path, {wf: [first]})
    original = ch.load_runs(str(tmp_path))[1]
    retry = {**first, "run_attempt": 2, "conclusion": "failure"}
    _, api, text = collect(tmp_path, {wf: [retry]}, since=None)
    assert ch.load_runs(str(tmp_path))[1] == original
    assert not any("/jobs?" in call for call in api.calls) and text.calls == []


@pytest.mark.parametrize("readable", [True, False])
def test_collect_enriches_recent_legacy_pr_once_with_cached_jobs(tmp_path, readable):
    sha = "ab" * 20
    raw = gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", pull_requests=[{"number": 4855}])
    jobs = {
        1: [
            checkout_job(10, "select"),
            checkout_job(11, "build / build"),
            gh_job(12, "mi350 / mi350-tlx-test", "failure")
        ]
    }
    logs = {
        10: source_log(sha), 11: source_log(sha), 12:
        "2026-10-12T01:02:00Z FAILED x.py::test_bitwise[False] - AssertionError: bitwise"
    }
    collect(tmp_path, {"gpu-tests.yml": [raw]}, jobs, logs=logs)
    records = ch.load_runs(str(tmp_path))
    legacy = records[1]
    for key in ["tested_sha", "tested_sha_provenance", "provenance_version", "run_attempt"]:
        legacy.pop(key, None)
    for job in legacy["jobs"]:
        job.pop("run_attempt", None)
        for step in job["steps"]:
            step.pop("test_failures", None)
    prior = ch.rules.pr_view([legacy], NOW, ch.rules.main_view([], NOW))
    ch.save_runs(str(tmp_path), records)
    _, api, text = collect(tmp_path, {"gpu-tests.yml": [raw]}, jobs, since=None, logs=logs if readable else {})
    updated = ch.load_runs(str(tmp_path))[1]
    assert updated["provenance_version"] == 1
    assert updated["tested_sha"] == (sha if readable else None)
    assert ch.rules.pr_view([updated], NOW, ch.rules.main_view([], NOW)) == prior
    assert not any("/jobs?" in call for call in api.calls)
    assert len(text.calls) == 3
    _, _, text = collect(tmp_path, {"gpu-tests.yml": [raw]}, jobs, since=None, logs=logs)
    assert text.calls == []


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


def test_collect_lists_gpu_nightly_and_lit_push_runs_only(tmp_path):
    _, api, _ = collect(tmp_path, {})
    listed = [c for c in api.calls if "/workflows/" in c]
    assert {c.split("/workflows/")[1].split("/")[0]
            for c in listed} == {
                "gpu-tests.yml", "gpu-tests-nightly.yml", "nightly-tlx-ops-bench.yml", "nightly-fbtriton.yml",
                "compiler.yml", "fast-gpu-tests.yml", "cutracer.yml"
            }
    assert [c for c in listed if "/compiler.yml/" in c][0].endswith("&event=push&per_page=100")


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
    assert data["asof"] == "2026-10-12T02:00:00Z" and set(data) == {"asof", "branches", "occupancy"}


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


def fake_get(responses):

    def get(path):
        for prefix, value in responses.items():
            if path.startswith(prefix):
                return value
        raise subprocess.CalledProcessError(1, "gh", stderr=f"HTTP 404: {path}")

    return get


def meta_file(tag, llvm_hash):
    body = json.dumps({"llvm_hash": llvm_hash, "release_base_url": f"https://x/download/{tag}"})
    return {"content": base64.b64encode(body.encode()).decode()}


def fork_commit(sha, title):
    return {"sha": sha, "commit": {"message": title + "\n\nbody"}}


T, F = "repos/facebookexperimental/triton", "repos/facebookexperimental/llvm-project"
A, B, C = "a" * 40, "b" * 40, "c" * 40


def test_snapshot_line_reads_pin_pending_changes_issues_and_history():
    get = fake_get({
        f"{T}/contents/cmake/meta-llvm-info.json?ref=main":
        meta_file("llvm-bbbbbbbb-1", B),
        f"{T}/contents/cmake/meta-llvm-info.json?ref={'2' * 40}":
        meta_file("llvm-bbbbbbbb-1", B),
        f"{T}/contents/cmake/meta-llvm-info.json?ref={'1' * 40}":
        meta_file("llvm-aaaaaaaa-1", A),
        f"{F}/commits?sha=triton_beta": [{"sha": C, "commit": {"tree": {"sha": "t"}}}],
        f"{F}/compare/{B}...{C}": {
            "commits": [
                fork_commit("s" * 40, "fbshipit-source-id: 48d598b00a9b"),
                fork_commit("d" * 40, "[triton/beta] Backport VOPD fix (#18)")
            ]
        },
        f"{T}/pulls?head=":
        [{"number": 4282, "title": "[Meta LLVM] Bump main", "html_url": "p", "user": {"login": "me"}}],
        f"{T}/commits?sha=main&path=cmake/meta-llvm-info.json": [
            {
                "sha": "2" * 40, "commit": {
                    "message": "[Meta LLVM] Bump main to llvm-bbbbbbbb-1 (#4300)", "committer":
                    {"date": "2026-10-20T00:00:00Z"}
                }
            },
            {
                "sha": "1" * 40, "commit":
                {"message": "Build against the fork (#4122)", "committer": {"date": "2026-10-16T00:00:00Z"}}
            },
        ],
    })
    releases = [{"tag_name": "llvm-cccccccc-1", "target_commitish": C, "published_at": "2026-10-21T00:00:00Z"}]
    issues = [
        {
            "number": 7, "title": "[Meta LLVM] Bump ready: main", "state": "open", "html_url": "i7", "created_at":
            "2026-10-21T02:00:00Z", "body": "<!-- meta-llvm-bump main -->"
        },
        {
            "number": 8, "title": "[Meta LLVM] Bump ready: release/3.8.x", "state": "open", "html_url": "i8",
            "created_at": "2026-10-21T02:00:00Z", "body": "<!-- meta-llvm-bump release/3.8.x -->"
        },
    ]
    line = ch.snapshot_line(get, "main", "triton_beta", releases, issues)
    assert line["pinned"] == {"tag": "llvm-bbbbbbbb-1", "llvm_hash": B, "base": None}
    assert line["newest"]["tag"] == "llvm-cccccccc-1"
    assert line["pending"] == [{"sha": "d" * 40, "title": "[triton/beta] Backport VOPD fix", "pr": 18}]
    assert line["pr"]["number"] == 4282 and [i["number"] for i in line["issues"]] == [7]
    assert [(h["pr"], h["tag"]) for h in line["history"]] == [(4300, "llvm-bbbbbbbb-1"), (4122, "llvm-aaaaaaaa-1")]
    assert [(h["prev_hash"], h["llvm_hash"]) for h in line["history"]] == [(A, B), (None, A)]


def test_failed_snapshot_keeps_the_previous_bump_json(tmp_path):
    (tmp_path / "bump.json").write_text('{"lines": {}}')
    collect(tmp_path, {})
    assert (tmp_path / "bump.json").read_text() == '{"lines": {}}'
    assert "bump.json" not in (tmp_path / "changed.txt").read_text()


def test_failed_lit_job_log_gives_the_failing_tests(tmp_path):
    runs = {"compiler.yml": [gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", event="push")]}
    lit_step = {
        "name": "Build and run LIT tests", "conclusion": "failure", "number": 1, "started_at": "2026-10-12T01:01:00Z",
        "completed_at": "2026-10-12T01:04:00Z"
    }
    jobs = {1: [gh_job(11, "LIT Tests", "failure", [lit_step]), gh_job(12, "report-compiler-lit")]}
    log = ("2026-10-12T01:02:00.1Z FAIL: TRITON :: TritonGPU/a.mlir (401 of 646)\n"
           "2026-10-12T01:02:00.2Z ******************** TEST 'TRITON :: TritonGPU/a.mlir' FAILED ********************\n"
           "2026-10-12T01:02:00.3Z # | /w/test/TritonGPU/a.mlir:32:50: error: undefined variable: MMA\n")
    _, _, text = collect(tmp_path, runs, jobs, logs={11: log})
    run = ch.load_runs(str(tmp_path))[1]
    assert run["wf"] == "compiler.yml"
    assert run["jobs"][0]["steps"][0]["failures"] == [{"id": "TritonGPU/a.mlir", "msg": "undefined variable: MMA"}]
    assert text.calls == ["repos/facebookexperimental/triton/actions/jobs/11/logs"]


def test_unreadable_lit_log_marks_the_job_log_missing(tmp_path):
    runs = {"compiler.yml": [gh_run(1, "2026-10-12T01:00:00Z", conclusion="failure", event="push")]}
    jobs = {1: [gh_job(11, "LIT Tests", "failure")]}
    collect(tmp_path, runs, jobs, logs={})
    j = ch.load_runs(str(tmp_path))[1]["jobs"][0]
    assert j["log_missing"] is True and "failures" not in j["steps"][0]
