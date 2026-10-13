import ci_health_rules as r


def run(conclusion="failure", event="pull_request", jobs=()):
    return {
        "id": 1, "wf": "h100.yml", "event": event, "status": "completed", "conclusion": conclusion, "created_at":
        "2026-10-12T00:00:00Z", "jobs": list(jobs)
    }


def job(name, conclusion, *steps):
    return {
        "name": name, "conclusion": conclusion, "started_at": "2026-10-12T00:00:00Z", "completed_at":
        "2026-10-12T00:10:00Z", "labels": ["linux-gcp-h100"], "steps": [{"name": n, "conclusion": c} for n, c in steps]
    }


def stamp(h):
    return f"2026-10-12T{h:02d}:00:00Z"


def full(ev, conclusion, h, core="success", perf=None, wf="h100.yml"):
    steps = [("Run TLX core tests", core)] + ([("Run host-default TLX L2 perf", perf)] if perf else [])
    jc = "failure" if "failure" in (core, perf) else "success"
    return {
        **run(conclusion, ev, [job(wf.split(".")[0] + "-tlx-test", jc, *steps)]), "id": h, "wf": wf, "created_at":
        stamp(h), "head_sha": "abcdef0123456789", "title": "t", "url": "u"
    }


def test_group_mapping_covers_renamed_tritonbench_and_perf():
    assert r.group_of("Run TLX core tests") == "TLX core"
    assert r.group_of("Run TritonBench tests on H100 GPU") == "TritonBench"
    assert r.group_of("Run host-default TLX L2 perf") == "TLX perf"
    assert r.group_of("Download prebuilt Triton wheel from S3") is None
    assert r.job_group("h100-meta-triton-test") == r.job_group("h100-tritonbench-test") == "TritonBench"
    assert r.job_group("report-h100-tlx (x) / file-issue") == "Reporting"


def test_cause_priority_is_config_test_infra_perf():
    assert r.classify(run("startup_failure"))[0] == "config"
    assert r.classify(run(jobs=[job("a", "skipped")]))[0] == "config"  # failed, but no job failed
    both = job("h100-tlx-test", "failure", ("Run TLX core tests", "failure"),
               ("Run host-default TLX L2 perf", "failure"))
    assert r.classify(run(jobs=[both]))[0] == "test"
    assert r.classify(run(jobs=[job("h100-tlx-test", "failure", ("Install Triton", "failure"))]))[0] == "infra"
    lost = job("h100-tritonbench-test", "failure", ("Set up job", "success"))
    assert r.classify(run(jobs=[lost]))[0] == "infra"  # runner lost: job failed with no failed step
    perf = job("h100-tlx-test", "failure", ("Run host-default TLX L2 perf", "failure"))
    assert r.classify(run(jobs=[perf]))[0] == "perf"


def test_group_results_distinguish_blocked_gated_and_infra():
    g = r.group_results(
        run(jobs=[
            job("h100-tlx-test", "failure", ("Run TLX core tests",
                                             "failure"), ("Run TLX tutorial correctness tests",
                                                          "skipped"), ("Run host-default TLX L2 perf", "skipped"))
        ]))
    assert g == {"TLX core": "fail", "Tutorial correctness": "notrun", "TLX perf": "notrun"}
    g = r.group_results(
        run(jobs=[
            job("h100-tlx-test", "failure", ("Download prebuilt Triton wheel from S3",
                                             "failure"), ("Run TLX core tests", "skipped"))
        ]))
    assert g == {"TLX core": "infra"}
    g = r.group_results(
        run(
            "success", jobs=[
                job("h100-tlx-test", "success", ("Run TLX core tests", "success"),
                    ("Run host-default TLX L2 perf", "skipped"))
            ]))
    assert g == {"TLX core": "pass", "TLX perf": "na"}


def test_latest_ignores_running_and_cancelled():
    runs = [
        full("push", "success", 1),
        full("push", "cancelled", 2), {**full("push", "success", 3), "status": "in_progress", "conclusion": None}
    ]
    d = r.page_data(runs, stamp(4), [])
    assert d["latest_push"]["h100.yml"]["t"] == stamp(1)
    assert d["latest_push"]["h100.yml"]["sha"] == "abcdef012"


def test_pr24_counts_any_failure_and_splits_causes():
    runs = [full("pull_request", "success", 1), full("pull_request", "failure", 2, core="failure")]
    p = r.page_data(runs, stamp(5), [])["pr24"]["h100.yml"]
    assert (p["passed"], p["failed"], p["test"], p["infra"]) == (1, 1, 1, 0)


def test_strip_and_purpose_use_finished_nightlies():
    runs = [full("schedule", "failure", 1, perf="failure"), full("schedule", "success", 2, perf="success")]
    d = r.page_data(runs, stamp(3), [])
    assert [c["c"] for c in d["strip"]["H100"]["TLX perf"]] == ["fail", "pass"]
    assert d["latest_nightly"]["H100"] == "10-12 02:00"
    row = {x["group"]: x for x in d["by_purpose"]["h100.yml"]}
    assert row["TLX perf"]["nightly"] == "pass"


def test_gpu_hours_skip_cpu_runner_jobs():
    a = full("pull_request", "success", 1)
    a["jobs"].append({**job("h100-build / build", "success"), "labels": ["32-core-ubuntu"]})
    gh = r.page_data([a], stamp(2), [])["gpu_hours"]["h100.yml"]
    assert sum(x["success"] for x in gh) == round(10 / 60, 1)


def test_runner_minutes_are_weekly_over_successful_pr_runs():
    a, b = full("pull_request", "success", 1), full("pull_request", "failure", 2, core="failure")
    rm = r.page_data([a, b], stamp(3), [])["runner_min"]["H100"]
    assert rm == [{"week": "2026-10-12", "n": 1, "avg": 10.0, "median": 10.0, "jobs": {"TLX": 10.0}}]


def test_every_test_step_in_the_gpu_workflows_maps_to_a_group():
    """Catches a renamed step on main (and typos in GROUPS): each `Run ...` step of a test job needs a group."""
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workflows")
    unmapped = []
    for wf in r.WORKFLOWS:
        job = None
        for line in open(os.path.join(root, wf)):
            m = re.match(r"^  ([\w-]+):\s*$", line)
            if m:
                job = m.group(1)
            m = re.match(r"^\s+- name: (Run .+?)\s*$", line)
            if m and job and r.job_group(job) in ("TLX", "TritonBench", "Gluon", "Runtime", "TorchTLX"):
                if r.group_of(m.group(1).strip("'\"")) is None:
                    unmapped.append((wf, job, m.group(1)))
    assert not unmapped
