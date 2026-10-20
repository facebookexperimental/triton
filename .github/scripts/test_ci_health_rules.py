import ci_health_rules as r

CORE, CODEGEN = "Run TLX core tests", "Run device-agnostic TLX codegen tests"


def stamp(h):
    return f"2026-10-12T{h:02d}:00:00Z"


def step(name, conclusion, failures=None):
    s = {"name": name, "conclusion": conclusion, "started_at": None, "completed_at": None}
    if failures is not None:
        s["failures"] = [{"id": i, "msg": f"msg {i}"} for i in failures]
    return s


def job(name, *steps, conclusion=None, minutes=10, h=0, label="linux-gcp-b200"):
    if conclusion is None:
        cs = [s["conclusion"] for s in steps]
        conclusion = "failure" if "failure" in cs else "success"
    return {
        "name": name, "conclusion": conclusion, "created_at": stamp(h), "started_at": stamp(h), "completed_at":
        f"2026-10-12T{h:02d}:{minutes:02d}:00Z", "labels": [label], "steps": list(steps)
    }


def tlx(*core, gpu="b200", codegen=None):
    """A *-tlx-test job: core fails with the given ids (passes when none); codegen skipped unless given."""
    c = step(CORE, "failure", list(core)) if core else step(CORE, "success")
    g = step(CODEGEN, "skipped") if codegen is None else (
        step(CODEGEN, "failure", codegen) if codegen else step(CODEGEN, "success"))
    return job(f"{gpu} / {gpu}-tlx-test", c, g)


def run(h, jobs, event="push", sha=None, conclusion=None, wf="gpu-tests.yml", pr=None, author=None, status="completed",
        branch="main"):
    if conclusion is None:
        conclusion = "failure" if any(j["conclusion"] == "failure" for j in jobs) else "success"
    return {
        "id": h, "wf": wf, "event": event, "status": status, "conclusion": conclusion, "created_at": stamp(h),
        "head_sha": sha or f"{h:02d}" * 20, "title": f"commit {h} (#{4000 + h})", "url": f"u{h}", "branch": branch,
        "pr": pr if pr is not None else 4000 + h, "author": author or f"dev{h}", "jobs": jobs
    }


def build_ok():
    return job("build", step("Build wheel", "success"), label="ubuntu-latest")


def push(h, *gpu_jobs, **kw):
    return run(h, [build_ok(), *gpu_jobs], **kw)


def main(runs, asof=stamp(23)):
    return r.main_view(runs, asof)


def items(view, gpu, kind):
    return {(i["id"], i["first_sha"][:2]) for i in view["gpus"][gpu][kind]}


def test_mi350_core_step_keeps_its_name_before_the_rename():
    assert r.group_of("Run TLX unit tests") == r.group_of("Run TLX core tests") == "TLX core"


def test_renamed_step_keeps_its_known_failures():
    old = job("mi350 / mi350-tlx-test", step("Run TLX unit tests", "failure", ["t1"]))
    v = main([push(1, old), push(2, tlx("t1", gpu="mi350"))])
    assert items(v, "MI350", "known") == {("t1", "01")} and v["gpus"]["MI350"]["known"][0]["step"] == CORE


def test_gpu_and_job_key_come_from_the_job_name_prefix():
    assert r.gpu_of({"wf": "gpu-tests.yml"}, {"name": "b200 / b200-tlx-test"}) == "B200"
    assert r.gpu_of({"wf": "gpu-tests.yml"}, {"name": "build"}) is None
    assert r.gpu_of({"wf": "gpu-tests.yml"}, {"name": "GPU tests"}) is None
    assert r.job_key("mi350 / mi350-gluon-test") == "mi350-gluon-test"


def test_failure_is_new_then_known_with_its_first_push():
    runs = [push(1, tlx()), push(2, tlx("t1")), push(3, tlx("t1"))]
    at2 = main(runs[:2])
    assert at2["gpus"]["B200"]["verdict"] == "new" and items(at2, "B200", "new") == {("t1", "02")}
    at3 = main(runs)
    g = at3["gpus"]["B200"]
    assert g["verdict"] == "known" and not g["new"]
    (k, ) = g["known"]
    assert (k["id"], k["first_sha"][:2], k["prev_sha"][:2], k["streak"]) == ("t1", "02", "01", 2)
    assert (k["first_pr"], k["first_author"], k["job"], k["step"]) == (4002, "dev2", "b200-tlx-test", CORE)


def test_skipped_step_is_unknown_not_passing():
    runs = [push(1, tlx(codegen=["t2"])), push(2, tlx("t1")), push(3, tlx(codegen=["t2"]))]
    g = main(runs)["gpus"]["B200"]
    known = {i["id"]: i for i in g["known"]}
    assert "t2" in known and known["t2"]["first_sha"][:2] == "01" and known["t2"]["streak"] == 2


def test_setup_failed_push_is_skipped_as_baseline():
    setup = job("b200 / b200-tlx-test", step("Install prebuilt Triton", "failure"), step(CORE, "skipped"))
    runs = [push(1, tlx("t1")), push(2, setup), push(3, tlx("t1"))]
    v = main(runs)
    assert items(v, "B200", "known") == {("t1", "01")}
    assert [p["cells"]["B200"] for p in v["pushes"]] == ["new", "setup", "known"]
    assert v["setup"] == [{
        "sha": "02" * 20, "t": stamp(2), "gpu": "B200", "job": "b200-tlx-test", "step": "Install prebuilt Triton"
    }]


def test_known_failure_does_not_hide_a_new_one_in_the_same_step():
    v = main([push(1, tlx("t1")), push(2, tlx("t1", "t3"))])
    assert items(v, "B200", "new") == {("t3", "02")} and items(v, "B200", "known") == {("t1", "01")}
    assert v["gpus"]["B200"]["verdict"] == "new"


def test_fixed_failure_drops_off():
    v = main([push(1, tlx("t1")), push(2, tlx())])
    assert v["gpus"]["B200"]["verdict"] == "passing" and not v["gpus"]["B200"]["known"]


def test_build_failure_banner_keeps_last_verdicts():
    broken = run(3, [
        job("build", step("Build wheel", "failure"), label="ubuntu-latest"),
        job("b200 / b200-tlx-test", conclusion="skipped")
    ])
    v = main([push(1, tlx()), push(2, tlx("t1")), broken])
    assert v["build"] == {"sha": "03" * 20, "pr": 4003, "author": "dev3", "t": stamp(3), "step": "Build wheel"}
    assert v["pushes"][-1]["cells"]["B200"] == "na"
    assert v["gpus"]["B200"]["verdict"] == "new" and v["gpus"]["B200"]["sha"][:2] == "02"
    assert not v["setup"]


def test_build_failure_matches_the_reusable_build_check_name():
    """gpu-tests.yml's build is a reusable workflow, so its check is named "build / build"."""
    broken = run(2, [job("build / build", step("Build wheel", "failure"), label="ubuntu-latest")])
    v = main([push(1, tlx()), broken])
    assert v["build"]["step"] == "Build wheel" and r.run_outcome(broken) == "build"


def test_build_failure_is_not_setup():
    broken = run(3, [job("build", step("Build wheel", "failure"), label="ubuntu-latest")])
    assert r.run_outcome(broken) == "build"


def test_scheduled_run_on_same_sha_marks_reproduced():
    sha = "ab" * 20
    base = [push(1, tlx()), push(2, tlx("t1"), sha=sha)]
    rep = run(3, [build_ok(), tlx("t1")], event="schedule", sha=sha)
    ok = run(3, [build_ok(), tlx()], event="schedule", sha=sha)
    assert main(base)["gpus"]["B200"]["new"][0]["sched"] == "none"
    assert main(base + [rep])["gpus"]["B200"]["new"][0]["sched"] == "reproduced"
    assert main(base + [ok])["gpus"]["B200"]["new"][0]["sched"] == "not reproduced"


def test_job_not_run_for_a_week_is_retired():
    """A disabled job (if: false) keeps its last failures forever unless retired."""
    torch = job("b200 / b200-torchtlx-test", step("Run TorchTLX template + fusion tests", "failure", ["t9"]))
    old = run(1, [build_ok(), torch, tlx()])
    old["created_at"] = "2026-10-01T00:00:00Z"
    v = main([old, push(2, tlx())])
    assert v["gpus"]["B200"]["verdict"] == "passing" and not v["gpus"]["B200"]["known"]


def test_job_skipped_by_path_filter_keeps_its_known_failures():
    gluon = job("b200 / b200-gluon-test", step("Run Gluon tests (python/test/gluon)", "failure", ["g1"]))
    v = main([push(1, gluon, tlx()), push(2, tlx())])
    assert items(v, "B200", "new") == {("g1", "01")}


def test_unit_whose_step_disappeared_is_dropped():
    renamed = job("b200 / b200-tlx-test", step("Run TLX smoke tests", "success"))
    v = main([push(1, tlx("t1")), push(2, renamed)])
    assert v["gpus"]["B200"]["verdict"] == "passing"


def test_pr_tags_main_ci_author():
    pushes = [push(1, tlx("t1"))]
    setup = job("b200 / b200-tlx-test", step("Install prebuilt Triton", "failure"))
    prs = [
        run(20, [build_ok(), tlx("t1")], event="pull_request", pr=1, branch="a"),
        run(21, [build_ok(), setup], event="pull_request", pr=2, branch="b"),
        run(22, [build_ok(), tlx("t9")], event="pull_request", pr=3, branch="c"),
        run(18, [build_ok(), tlx("t9")], event="pull_request", pr=4, branch="d"),
        run(19, [build_ok(), tlx()], event="pull_request", pr=4, branch="d"),
    ]
    view = r.pr_view(pushes + prs, stamp(23), main(pushes))
    got = {p["pr"]: (p["result"], p["whose"]) for p in view}
    assert got == {1: ("tests", "main"), 2: ("setup", "ci"), 3: ("tests", "author"), 4: ("passed", None)}
    assert [p["pr"] for p in view] == [3, 2, 1, 4]  # newest first
    assert next(p for p in view if p["pr"] == 3)["gpus"] == ["B200"]


def test_tracked_branch_of_pushes_bumps_and_prs():
    assert r.tracked_branch(run(1, [], branch="release/3.8.x")) == "release/3.8.x"
    assert r.tracked_branch(run(1, [], branch="scratch/meta-llvm-bump/release-3.8.x")) == "release/3.8.x"
    assert r.tracked_branch(run(1, [], branch="scratch/meta-llvm-bump/main")) == "main"
    assert r.tracked_branch({**run(1, [], event="pull_request", branch="a"), "base":
                             "release/3.8.x"}) == "release/3.8.x"
    assert r.tracked_branch(run(1, [], event="pull_request", branch="a")) == "main"  # older records lack a base


def test_each_tracked_branch_gets_its_own_tabs():
    rel = "release/3.8.x"
    runs = [
        push(1, tlx()),
        push(2, tlx("t1"), branch=rel),
        push(3, tlx()),
        push(4, tlx(), branch="scratch/meta-llvm-bump/main"),
        {**run(20, [build_ok(), tlx("t1")], event="pull_request", pr=1, branch="a"), "base": rel},
        run(21, [build_ok(), tlx("t1")], event="pull_request", pr=2, branch="b"),
    ]
    page = r.page_data(runs, stamp(23), {})["branches"]
    assert set(page) == set(r.TRACKED_BRANCHES)
    m, rv = page["main"]["main"], page[rel]["main"]
    # A bump branch's pushes count for its line in the other tabs, but are not pushes to the branch.
    assert m["gpus"]["B200"]["verdict"] == "passing" and [p["sha"][:2] for p in m["pushes"]] == ["01", "03"]
    assert items(rv, "B200", "new") == {("t1", "02")} and len(rv["pushes"]) == 1
    # Each PR is listed under its base, and judged against that branch's failures.
    assert [(p["pr"], p["whose"], p["note"]) for p in page[rel]["pr"]] == [(1, "main", f"all 1 also fail on {rel}")]
    assert [(p["pr"], p["whose"]) for p in page["main"]["pr"]] == [(2, "author")]


def tb(gpu, *failing, setup=False, label="linux-gcp-b200"):
    """A gpu-tests-nightly.yml TritonBench job, called from a job named after its GPU."""
    if setup:
        steps = [
            step("Install prebuilt Triton", "failure"),
            step(f"Run TritonBench tests on {gpu.upper()} GPU", "skipped")
        ]
    else:
        steps = [
            step(f"Run TritonBench tests on {gpu.upper()} GPU", "failure" if failing else "success",
                 list(failing) if failing else None)
        ]
    return job(f"{gpu} / {gpu}-tritonbench-test", *steps, label=label)


def nightly(h, *jobs):
    return run(h, [build_ok(), *jobs,
                   job("h100 / report / report", step("x", "success"), label="ubuntu-latest")], event="schedule",
               wf="gpu-tests-nightly.yml")


def test_gpu_of_reads_both_gpu_workflows():
    assert r.gpu_of({"wf": "gpu-tests.yml"}, {"name": "b200 / b200-tlx-test"}) == "B200"
    assert r.gpu_of({"wf": "gpu-tests-nightly.yml"}, {"name": "mi350 / mi350-tritonbench-test"}) == "MI350"
    assert r.gpu_of({"wf": "gpu-tests-nightly.yml"}, {"name": "build"}) is None
    assert r.gpu_of({"wf": "nightly-fbtriton.yml"}, {"name": "h100 / x"}) is None


def test_nightly_gpu_view_latest_result_and_week():
    runs = [nightly(1, tb("h100", "gemm")), nightly(2, tb("h100"), tb("b200", setup=True))]
    v = r.page_data(runs, stamp(23), {})["branches"]["main"]["nightly"]["gpu"]
    assert v["H100"]["result"] == "passed" and (v["H100"]["passed_7"], v["H100"]["runs_7"]) == (1, 2)
    assert v["B200"]["result"] == "setup" and v["B200"]["setup"] == "Install prebuilt Triton"
    assert "MI350" not in v
    first = r.gpu_nightly_view(runs[:1], stamp(23))["H100"]
    assert first["result"] == "test" and [f["id"] for f in first["fails"]] == ["gemm"]


def test_nightly_gpu_hours_count_in_ops():
    day = r.ops_view([nightly(1, tb("h100"))], stamp(23))["gpu_hours"][-1]
    assert day["passed"] == 0.2


def test_gpu_hours_split_by_outcome_and_skip_cpu_jobs():
    setup = job("b200 / b200-tlx-test", step("Install prebuilt Triton", "failure"), minutes=30)
    runs = [
        push(1, tlx()),
        push(2, tlx("t1")),
        push(3, setup),
        run(4, [build_ok(), tlx()], conclusion="cancelled"),
    ]
    day = r.ops_view(runs, stamp(23))["gpu_hours"][-1]
    assert day == {"day": "2026-10-12", "passed": 0.2, "test": 0.2, "setup": 0.5, "cancelled": 0.2}


def test_job_duration_p90_by_job_key_across_gpus():
    runs = [
        push(h, job(f"{g} / {g}-tlx-test", step(CORE, "success"), minutes=m))
        for h, (g, m) in enumerate([("b200", 10), ("h100", 20), ("b200", 30)], 1)
    ]
    jobs = {j["job"]: j for j in r.ops_view(runs, stamp(23))["jobs"]}
    assert jobs["tlx-test"]["p50"] == 20 and jobs["tlx-test"]["p90"] == 30


def test_setup_steps_counted_by_step_gpu_and_trigger():
    setup = job("mi350 / mi350-tlx-test", step("Install prebuilt Triton", "failure"))
    runs = [push(1, setup), run(2, [build_ok(), setup], event="pull_request")]
    (row, ) = r.ops_view(runs, stamp(23))["setup_steps"]
    assert row == {
        "step": "Install prebuilt Triton", "gpu": "MI350", "n": 2, "triggers": {"push": 1, "pull_request": 1}
    }


def test_mi350_environment_steps_are_setup():
    for name in ("Set up test environment", "Install prebuilt Triton", "Resolve TritonBench ref from PR body (X-link)",
                 "Install TritonBench dependencies"):
        assert r.run_outcome(push(1, job("mi350 / mi350-tlx-test", step(name, "failure")))) == "setup", name


def test_nightly_release_and_perf_events():
    nightly = run(5, [
        job("select", step("Pick commit", "success")),
        job("build / wheel", step("x", "success")),
        job("publish", step("Upload wheels + latest.json", "success"))
    ], event="schedule", wf="nightly-fbtriton.yml")
    bench = {
        "b200": [{"date": "20261011", "perf": {}}, {
            "date": "20261012", "range": "aaa..bbb", "compare": "https://x", "perf":
            {"mm": {"n_slower": 3, "n_faster": 0, "signature": "sig", "unconfirmed": True, "suspects": [{"sha": "c"}]}}
        }]
    }
    v = r.nightly_view([nightly], stamp(23), bench)
    assert v["release"]["conclusion"] == "success" and v["release"]["build"] == "success"
    assert v["release"]["publish"] == "success" and v["release"]["published_7"] == 1
    assert v["perf"] == [{
        "platform": "b200", "date": "20261012", "op": "mm", "n_slower": 3, "n_faster": 0, "signature": "sig",
        "unconfirmed": True, "range": "aaa..bbb", "compare": "https://x", "suspects": [{"sha": "c"}]
    }]


def test_records_from_before_gpu_tests_are_ignored():
    old = run(2, [job("b200 / b200-tlx-test", step(CORE, "failure", ["t1"]))], wf="b200.yml")
    v = r.page_data([push(1, tlx()), old], stamp(23), {})["branches"]["main"]
    assert sum(d["test"] for d in v["ops"]["gpu_hours"]) == 0 and len(v["main"]["pushes"]) == 1


def sel_run(h, event, h100=True, b200=True, mi350=True, build=True, fallback=False, select="success"):
    x = run(h, [job("select", step("Select affected GPU platforms", select), label="ubuntu-latest")], event=event)
    x["select"] = {"h100": h100, "b200": b200, "mi350": mi350, "build": build, "fallback": fallback}
    return x


def test_smart_trigger_counts_selection_per_trigger():
    runs = [
        sel_run(1, "pull_request", b200=False, mi350=False),
        sel_run(2, "pull_request", h100=False, b200=False, mi350=False, build=False),
        sel_run(3, "pull_request", fallback=True),
        sel_run(4, "push"),
        sel_run(5, "schedule", mi350=False),
    ]
    sel = r.ops_view(runs, stamp(23))["select"]
    pr = sel["triggers"]["pull_request"]
    assert pr == {
        "runs": 3, "gpus": {"0": 1, "1": 1, "2": 0, "3": 1}, "build_skipped": 1, "fallback": 1, "per_gpu":
        {"H100": 2, "B200": 1, "MI350": 1}
    }
    assert [a["kind"] for a in sel["anomalies"]] == ["fallback", "partial"]
    assert sel["anomalies"][1]["event"] == "schedule"


def test_failed_select_is_an_anomaly():
    sel = r.ops_view([sel_run(1, "push", select="failure")], stamp(23))["select"]
    assert [a["kind"] for a in sel["anomalies"]] == ["failed"]


def test_page_data_has_one_section_per_tab():
    data = r.page_data([push(1, tlx())], stamp(23), {})
    assert set(data) == {"asof", "branches"}
    assert all(set(v) == {"main", "nightly", "pr", "ops"} for v in data["branches"].values())


def test_every_test_step_in_the_gpu_workflows_maps_to_a_group():
    """Catches a renamed step on main (and typos in GROUPS): each `Run ...` step of a test job needs a group."""
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workflows")
    unmapped = []
    for wf in r.GPU_WORKFLOWS:
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


def steps_of(view, gpu):
    return {row["group"]: row["cells"] for row in view["steps"][gpu]}


def test_step_grid_marks_new_known_and_blocked():
    v = main([push(1, tlx(codegen=[])), push(2, tlx("t1")), push(3, tlx("t1", codegen=[]))])
    assert steps_of(v, "B200") == {"TLX core": ["pass", "new", "known"], "TLX codegen": ["pass", "blocked", "pass"]}


def test_step_grid_marks_setup_and_not_run():
    setup = job("b200 / b200-tlx-test", step("Install prebuilt Triton", "failure"), step(CORE, "skipped"))
    v = main([push(1, tlx()), push(2, setup), push(3, tlx(gpu="h100"))])
    assert steps_of(v, "B200")["TLX core"] == ["pass", "setup", "na"]
    assert steps_of(v, "H100") == {"TLX core": ["na", "na", "pass"]}


def test_step_grid_has_no_rows_for_a_gpu_without_results():
    v = main([push(1, tlx())])
    assert v["steps"]["MI350"] == []
