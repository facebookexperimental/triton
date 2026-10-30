"""GPU CI orchestration tests; no GPU, AWS account, or compiler build needed."""
import io
import json
from pathlib import Path
import subprocess

from botocore.exceptions import ClientError
import pytest
import yaml

from gpu_test_selection import PATH_FILTERS, bump_ref, changed_paths, passed_push_run, select_platforms
from wheel_cache import BUCKET, LEASE_SECONDS, RETENTION_SECONDS, WheelCache

ROOT = Path(__file__).resolve().parents[2]
FILTERS = json.loads(PATH_FILTERS.read_text())


@pytest.mark.parametrize("path,expected", [
    ("lib/Dialect/Triton/Transforms/Combine.cpp", {"h100", "b200", "mi350"}),
    ("third_party/nvidia/backend/compiler.py", {"h100", "b200"}),
    ("third_party/amd/backend/compiler.py", {"mi350"}),
    ("third_party/tlx/ops/kernels/mm/sm100.py", {"b200"}),
    ("third_party/tlx/ops/kernels/mm/sm90.py", {"h100"}),
    ("third_party/tlx/ops/kernels/mm/gfx950.py", {"mi350"}),
    ("python/test/unit/language/test_tlx_dot_hip_or_sm90.py", {"h100", "mi350"}),
    ("python/test/unit/language/test_tlx_codegen.py", {"b200"}),
    ("README.md", set()),
    ("docs/index.rst", set()),
    ("test/Triton/ops.mlir", set()),
    (".agents/skills/example/script.py", set()),
    (".github/workflows/build.yml", {"h100", "b200", "mi350"}),
])
def test_platform_paths(path, expected):
    selected = select_platforms([path], FILTERS)
    assert {platform for platform, enabled in selected.items() if enabled} == expected


def test_mixed_changes_and_empty_diff():
    assert all(select_platforms(["third_party/amd/new.cpp", "third_party/nvidia/new.cpp"], FILTERS).values())
    assert not any(select_platforms([], FILTERS).values())


def test_schedule_manual_and_new_branch_run_all():
    assert changed_paths("schedule", {}) is None
    assert changed_paths("workflow_dispatch", {}) is None
    assert changed_paths("push", {"before": "0" * 40}) is None


def test_bump_ref_push_and_same_repo_pr_only():
    assert bump_ref("push", {"ref": "refs/heads/scratch/meta-llvm-bump/main"}) == "scratch/meta-llvm-bump/main"
    assert bump_ref("push", {"ref": "refs/heads/main"}) is None
    pr = {
        "head": {"ref": "scratch/meta-llvm-bump/main", "repo": {"full_name": "o/r"}}, "base":
        {"repo": {"full_name": "o/r"}}
    }
    assert bump_ref("pull_request", {"pull_request": pr}) == "scratch/meta-llvm-bump/main"
    fork = {**pr, "head": {**pr["head"], "repo": {"full_name": "x/r"}}}
    assert bump_ref("pull_request", {"pull_request": fork}) is None
    assert bump_ref("schedule", {}) is None


def test_passed_push_run_needs_green_push_run_of_gpu_tests():
    run = {"event": "push", "path": ".github/workflows/gpu-tests.yml", "conclusion": "success", "html_url": "u"}
    assert passed_push_run([run]) == "u"
    assert passed_push_run([{**run, "conclusion": "failure"}]) is None
    assert passed_push_run([{**run, "event": "pull_request"}]) is None
    assert passed_push_run([{**run, "path": ".github/workflows/pre-commit.yml"}]) is None


def test_diff_uses_pr_merge_base_and_push_range(monkeypatch):
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, b"old.py\0new.py\0")

    monkeypatch.setattr(subprocess, "run", run)
    assert changed_paths("pull_request",
                         {"pull_request": {"base": {"sha": "base"}, "head": {"sha": "head"}}}) == ["old.py", "new.py"]
    assert commands[-1][-2:] == ["base...head", "--"]
    changed_paths("push", {"before": "before", "after": "after"})
    assert commands[-1][-3:] == ["before", "after", "--"]
    assert "--no-renames" in commands[-1]


class MemoryS3:
    """Model S3's atomic conditional writes, including competing lease holders."""

    def __init__(self):
        self.objects = {}
        self.version = 0
        self.before_put = None

    def fail(self, code):
        raise ClientError({"Error": {"Code": code}}, "test")

    def get_object(self, Bucket, Key):
        assert Bucket == BUCKET
        if Key not in self.objects:
            self.fail("NoSuchKey")
        body, etag = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.before_put:
            hook, self.before_put = self.before_put, None
            hook()
        current = self.objects.get(Key)
        if kwargs.get("IfNoneMatch") == "*" and current is not None:
            self.fail("PreconditionFailed")
        if "IfMatch" in kwargs and (current is None or current[1] != kwargs["IfMatch"]):
            self.fail("PreconditionFailed")
        self.version += 1
        etag = f'"{self.version}"'
        self.objects[Key] = Body, etag
        return {"ETag": etag}

    def head_object(self, Bucket, Key):
        obj = self.get_object(Bucket, Key)
        return {"ContentLength": len(obj["Body"].read())}


@pytest.fixture
def cache():
    s3 = MemoryS3()
    now = [1000]
    cache = WheelCache(s3, {"repository": "owner/repo", "sha": "a" * 40}, clock=lambda: now[0],
                       sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    return cache, s3, now


def upload(cache):
    key = "tritonci-artifacts/run_1/fbtriton.whl"
    cache.s3.put_object(Bucket=BUCKET, Key=key, Body=b"wheel")
    return {"key": key, "size": 5, "sha256": "a" * 64}


def test_repeated_requests_reuse_published_wheel(cache):
    producer, s3, now = cache
    state, lease = producer.acquire("first")
    assert state is None and lease
    manifest = producer.publish(lease, upload(producer))
    producer.release(lease)
    state, lease = producer.acquire("second")
    assert lease is None
    assert state == manifest


def test_simultaneous_misses_have_one_producer(cache):
    producer, s3, now = cache
    other = WheelCache(s3, producer.identity, clock=producer.clock, sleep=producer.sleep)
    winner = []
    # Both callers read a missing key. The other writer wins before our PUT.
    s3.before_put = lambda: winner.append(other.acquire("winner")[1])

    def finish_other(seconds):
        other.publish(winner[0], upload(other))
        now[0] += seconds

    producer.sleep = finish_other
    state, lease = producer.acquire("loser")
    assert state["status"] == "ready" and lease is None
    assert len(winner) == 1


def test_failed_build_releases_lease_without_publishing(cache):
    producer, s3, now = cache
    _, first = producer.acquire("first")
    producer.release(first)
    state, second = producer.acquire("retry")
    assert state is None and second["etag"] != first["etag"]


def test_expired_lease_cannot_publish_or_release_its_successor(cache):
    producer, s3, now = cache
    _, first = producer.acquire("lost-runner")
    now[0] += LEASE_SECONDS + 1
    _, second = producer.acquire("replacement")
    with pytest.raises(RuntimeError, match="Lost the wheel build lease"):
        producer.publish(first, upload(producer))
    producer.release(first)
    assert producer.read()[1] == second["etag"]
    producer.publish(second, upload(producer))


@pytest.mark.parametrize("damage", ["missing", "size", "expired"])
def test_invalid_or_expired_artifacts_rebuild(cache, damage):
    producer, s3, now = cache
    _, lease = producer.acquire("first")
    wheel = upload(producer)
    producer.publish(lease, wheel)
    if damage == "missing":
        del s3.objects[wheel["key"]]
    elif damage == "size":
        s3.put_object(Bucket=BUCKET, Key=wheel["key"], Body=b"broken")
    else:
        now[0] += RETENTION_SECONDS
    state, lease = producer.acquire("retry")
    assert state is None and lease


def test_cannot_publish_missing_upload(cache):
    producer, s3, now = cache
    _, lease = producer.acquire("first")
    with pytest.raises(RuntimeError, match="Uploaded wheel is missing"):
        producer.publish(lease, {"key": "missing.whl", "sha256": "a" * 64, "size": 1})
    assert producer.read()[0]["status"] == "building"


def test_different_builds_do_not_share_a_key(cache):
    producer, s3, now = cache
    for changes in ({"sha": "b" * 40}, {"repository": "fork/repo"}, {"recipe": "new"}, {"abi": "cp313"}):
        other = WheelCache(s3, producer.identity | changes)
        assert other.key != producer.key


def test_cache_access_denied_does_not_bypass_lock(cache, monkeypatch):
    producer, s3, now = cache

    def denied(**kwargs):
        s3.fail("AccessDenied")

    monkeypatch.setattr(s3, "get_object", denied)
    with pytest.raises(ClientError, match="AccessDenied"):
        producer.acquire("first")
    assert s3.objects == {}


@pytest.mark.parametrize("build_repository", [None, "triton-lang/triton"])
def test_cache_identity_uses_workflow_tooling_for_old_source_refs(tmp_path, monkeypatch, build_repository):
    import wheel_cache

    tooling = tmp_path / "tooling"
    (tooling / "scripts").mkdir(parents=True)
    (tooling / "workflows").mkdir()
    script = tooling / "scripts/wheel_cache.py"
    script.write_text("cache implementation")
    recipe = tooling / "workflows/build.yml"
    recipe.write_text("build recipe")
    source = tmp_path / "old-source"
    source.mkdir()  # This old source ref has no cache script or build workflow.
    monkeypatch.chdir(source)
    monkeypatch.setattr(wheel_cache, "__file__", str(script))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    for key, value in {"GITHUB_REPOSITORY": "owner/repo", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TRITON_BUILD_REPOSITORY", raising=False)
    if build_repository:
        monkeypatch.setenv("TRITON_BUILD_REPOSITORY", build_repository)
    before = wheel_cache.build_identity()
    assert before["sha"] == "a" * 40
    assert before["repository"] == (build_repository or "owner/repo")
    recipe.write_text("changed build recipe")
    assert wheel_cache.build_identity()["recipe"] != before["recipe"]


def test_build_source_checkout_after_sparse_tooling(tmp_path, monkeypatch):
    workflow = yaml.load((ROOT / ".github/workflows/build.yml").read_text(), Loader=yaml.BaseLoader)
    steps = {step["name"]: step for step in workflow["jobs"]["build"]["steps"]}
    source = tmp_path / "source"
    source.mkdir()

    def git(directory, *args):
        return subprocess.run(["git", *args], cwd=directory, check=True, text=True, capture_output=True).stdout.strip()

    # A real Git checkout is needed: sparse state persists across action calls.
    files = {
        ".github/scripts/wheel_cache.py": "# tooling\n",
        ".github/workflows/build.yml": "# recipe\n",
        "cmake/llvm-info.json": '{"llvm_hash": "test"}\n',
        "cmake/amd-llvm-info.json": '{"llvm_hash": "test-amd"}\n',
        "python/requirements.txt": "setuptools\n",
        "pyproject.toml": "[build-system]\n",
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(source, "init", "-q")
    git(source, "add", ".")
    git(source, "-c", "user.name=CI test", "-c", "user.email=ci@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-qm", "fixture")
    sha = git(source, "rev-parse", "HEAD")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("GITHUB_ENV", str(runner_temp / "env"))

    def checkout(step):
        settings = step["with"]
        directory = workspace / settings.get("path", ".")
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / ".git").exists():
            git(directory, "init", "-q")
            git(directory, "remote", "add", "origin", str(source))
        git(directory, "fetch", "--depth=1", "origin", sha)
        if "sparse-checkout" in settings:
            assert settings["sparse-checkout-cone-mode"] == "false"
            git(directory, "config", "core.sparseCheckout", "true")
            (directory / ".git/info/sparse-checkout").write_text(settings["sparse-checkout"])
        else:
            # actions/checkout@v6 disables sparse checkout, then unsets this
            # extension. Reusing a non-cone checkout reactivates its old config.
            git(directory, "sparse-checkout", "disable")
            git(directory, "config", "--local", "--unset-all", "extensions.worktreeConfig")
        git(directory, "checkout", "--force", sha)

    checkout(steps["Checkout build tooling"])
    subprocess.run(["bash", "-e", "-c", steps["Preserve build tooling outside the source checkout"]["run"]],
                   cwd=workspace, check=True)
    checkout(steps["Checkout"])
    for name, content in files.items():
        assert (workspace / name).read_text() == content
    assert (runner_temp / "wheel-ci/scripts/wheel_cache.py").read_text() == files[".github/scripts/wheel_cache.py"]
    assert (runner_temp / "wheel-ci/workflows/build.yml").read_text() == files[".github/workflows/build.yml"]


@pytest.mark.parametrize("results,selected,success", [
    ({"select": "success", "build": "skipped", "h100": "skipped"}, False, True),
    ({"select": "success", "build": "failure", "h100": "skipped"}, True, False),
    ({"select": "success", "build": "success", "h100": "failure"}, True, False),
    ({"select": "success", "build": "success", "h100": "success"}, True, True),
    ({"select": "failure", "build": "skipped", "h100": "skipped"}, False, False),
])
def test_aggregate_gate_rejects_failed_build_and_selected_tests(results, selected, success):
    import os
    import sys

    parent = yaml.load((ROOT / ".github/workflows/gpu-tests.yml").read_text(), Loader=yaml.BaseLoader)
    gate = parent["jobs"]["gpu-tests"]
    assert gate["if"].startswith("always()")
    script = gate["steps"][0]["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    needs = {name: {"result": result} for name, result in results.items()}
    needs["select"]["outputs"] = {
        "build": str(selected).lower(), "h100": str(selected).lower(), "b200": "false", "mi350": "false", "cpu": "false"
    }
    needs["cpu-tests"] = {"result": "skipped"}
    for platform in ("b200", "mi350"):
        needs[platform] = {"result": "skipped"}
    result = subprocess.run([sys.executable, "-c", script], env=os.environ | {"NEEDS": json.dumps(needs)},
                            capture_output=True)
    assert (result.returncode == 0) == success, result.stderr.decode()


def test_workflows_share_one_build_and_pin_all_checkouts():

    def workflow(name):
        return yaml.load((ROOT / ".github/workflows" / f"{name}.yml").read_text(), Loader=yaml.BaseLoader)

    parent = workflow("gpu-tests")
    assert sum(job.get("uses") == "./.github/workflows/build.yml" for job in parent["jobs"].values()) == 1
    for platform in FILTERS:
        caller = parent["jobs"][platform]
        assert caller["needs"] == ["select", "build"]
        assert caller["with"]["wheel_url"] == "${{ needs.build.outputs.wheel_url }}"
        child = workflow(platform)
        assert set(child["on"]) == {"workflow_call"}
        assert "concurrency" not in child
        for job in child["jobs"].values():
            assert job.get("uses") != "./.github/workflows/build.yml"
            for step in job.get("steps", []):
                if step.get("name") == "Checkout":
                    assert step["with"]["ref"] == "${{ inputs.ref }}"
                if step.get("name") == "Install Triton":
                    steps = job["steps"]
                    verify = steps[steps.index(step) - 1]
                    assert "sha256sum --check" in verify["run"]
                    assert verify["env"]["WHEEL_SHA256"] == "${{ inputs.wheel_sha256 }}"
                if step.get("uses") == "./.github/actions/install-prebuilt-triton":
                    assert step["with"]["wheel-sha256"] == "${{ inputs.wheel_sha256 }}"
                    action = yaml.load((ROOT / ".github/actions/install-prebuilt-triton/action.yml").read_text(),
                                       Loader=yaml.BaseLoader)
                    steps = action["runs"]["steps"]
                    install = next(i for i, item in enumerate(steps) if item.get("name") == "Install Triton")
                    assert "sha256sum --check" in steps[install - 1]["run"]
    assert LEASE_SECONDS > int(workflow("build")["jobs"]["build"]["timeout-minutes"]) * 60


@pytest.mark.parametrize("changed", [
    "python/test/conftest.py",
    "python/test/unit/tlx_ops/mm_test_utils.py",
    "python/triton/runtime/jit.py",
    "third_party/nvidia/backend/compiler.py",
    "third_party/amd/backend/driver.py",
    "third_party/tlx/ops/kernels/flash_attn/gfx950.py",
    "unknown.py",
])
def test_execution_dependencies_keep_full_suites_on_the_gpu(changed):
    from gpu_test_selection import SUITES, test_plan as plan_tests
    suites = json.loads(SUITES.read_text())
    plan = plan_tests(["lib/Analysis/Allocation.cpp", changed], ROOT, FILTERS, suites)
    assert not plan["cpu"]
    assert plan["gpu"]["b200"]["gluon"]["files"] == ["python/test/gluon"]
    assert plan["gpu"]["mi350"]["gluon"]["files"] == ["python/test/gluon"]
    assert plan["gpu"]["b200"]["codegen"]["files"]


def test_fixture_only_changes_do_not_skip_tests():
    from gpu_test_selection import SUITES, test_plan as plan_tests
    plan = plan_tests(["python/test/conftest.py"], ROOT, FILTERS, json.loads(SUITES.read_text()))
    assert not plan["cpu"]
    for platform in plan["gpu"].values():
        assert platform["core"]["files"] and platform["correctness"]["files"]
    assert plan["gpu"]["h100"]["runtime"]["files"]


def test_cpu_consumers_still_require_the_wheel():
    from gpu_test_selection import SUITES, test_plan as plan_tests
    plan = plan_tests(["lib/Analysis/Allocation.cpp"], ROOT, FILTERS, json.loads(SUITES.read_text()))
    assert plan["cpu"]["codegen"]["files"]
    assert plan["cpu"]["gluon"]["files"]
    assert plan["build"]
    assert not plan["gpu"]["b200"]["gluon"]["files"]
    assert not plan["gpu"]["mi350"]["gluon"]["files"]


def test_new_gluon_tests_and_reenabled_files_keep_gpu_coverage(tmp_path):
    from gpu_test_selection import SUITES, cpu_eligible
    spec = json.loads(SUITES.read_text())["cpu"]["gluon"]
    directory = tmp_path / "python/test/gluon"
    directory.mkdir(parents=True)
    for path in spec["files"]:
        (tmp_path / path).write_text("")
    for filename in spec["gluon_skip_files"]:
        (directory / filename).write_text("")
    original = (ROOT / "python/test/gluon/conftest.py").read_text()
    fixture = directory / "conftest.py"
    fixture.write_text(original)
    assert cpu_eligible(["lib/Analysis/Allocation.cpp"], tmp_path, spec)
    extra = directory / "new_test.py"
    extra.write_text("")
    assert not cpu_eligible(["lib/Analysis/Allocation.cpp"], tmp_path, spec)
    extra.unlink()
    fixture.write_text(original.replace("collect_ignore = sorted(_VERSION_SKEW_FILES)", "collect_ignore = []"))
    assert not cpu_eligible(["lib/Analysis/Allocation.cpp"], tmp_path, spec)


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_cpu_failure_reaches_the_final_check(result):
    import os
    import sys
    parent = yaml.load((ROOT / ".github/workflows/gpu-tests.yml").read_text(), Loader=yaml.BaseLoader)
    script = parent["jobs"]["gpu-tests"]["steps"][0]["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    needs = {name: {"result": "success"} for name in ("select", "build", "h100", "b200", "mi350")}
    needs["cpu-tests"] = {"result": result}
    needs["select"]["outputs"] = {name: "true" for name in ("build", "h100", "b200", "mi350", "cpu")}
    proc = subprocess.run([sys.executable, "-c", script], env=os.environ | {"NEEDS": json.dumps(needs)},
                          capture_output=True)
    assert proc.returncode != 0


def test_execution_preserves_the_selected_expression():
    from run_selected_tests import command
    plan = {"files": ["python/test/gluon/test_frontend.py"], "k": "test_a or test_b"}
    argv = command(plan, 4, "results.xml")
    assert argv[argv.index("-k") + 1] == plan["k"]
    assert "python/test/gluon/test_frontend.py" in argv
    with pytest.raises(ValueError):
        command({"files": [], "k": ""}, 1, "results.xml")


def test_cpu_routing_cannot_hide_new_gpu_only_skips(tmp_path):
    from run_selected_tests import check_cpu_skips
    report = tmp_path / "results.xml"
    report.write_text('<testsuite><testcase><skipped message="requires a CUDA GPU"/></testcase></testsuite>')
    with pytest.raises(ValueError, match="Unexpected skip"):
        check_cpu_skips({"allowed_skips": []}, report)
    report.write_text('<testsuite><testcase><skipped message="known target limitation"/></testcase></testsuite>')
    check_cpu_skips({"allowed_skips": ["known target limitation"]}, report)


def test_cpu_dependency_changes_remain_invalid_after_commit(tmp_path):
    from gpu_test_selection import reviewed_changes
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    helper = tmp_path / "helper.py"
    helper.write_text("value = 1\n")
    subprocess.run(["git", "add", "helper.py"], cwd=tmp_path, check=True)
    commit = ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid", "commit", "-qm", "test"]
    subprocess.run(commit, cwd=tmp_path, check=True)
    reviewed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert reviewed_changes(tmp_path, reviewed, ["helper.py"]) == []
    helper.write_text("value = 2\n")
    subprocess.run(["git", "add", "helper.py"], cwd=tmp_path, check=True)
    subprocess.run(commit, cwd=tmp_path, check=True)
    assert reviewed_changes(tmp_path, reviewed, ["helper.py"]) == ["helper.py"]


def test_cpu_eligibility_rejects_an_overridden_ignore_policy(tmp_path):
    from gpu_test_selection import SUITES, cpu_eligible
    spec = json.loads(SUITES.read_text())["cpu"]["gluon"]
    directory = tmp_path / "python/test/gluon"
    directory.mkdir(parents=True)
    for path in spec["files"]:
        (tmp_path / path).write_text("")
    for name in spec["gluon_skip_files"]:
        (directory / name).write_text("")
    source = (ROOT / "python/test/gluon/conftest.py").read_text()
    (directory / "conftest.py").write_text(source + "\ncollect_ignore = []\n")
    assert not cpu_eligible(["lib/Analysis/Allocation.cpp"], tmp_path, spec)


@pytest.mark.parametrize("paths,offload,placement,gpu_jobs,reason", [
    (["lib/Analysis/Allocation.cpp"], True, "CPU", 4, "match the CPU audit"),
    (["python/triton/runtime/jit.py"], True, "GPU", 6, "current diff changes Python"),
    (["README.md"], True, "Not selected", 0, "No work for this suite"),
    (["lib/Analysis/Allocation.cpp"], False, "GPU", 6, "CPU offload is disabled"),
])
def test_run_summary_explains_actual_placement(paths, offload, placement, gpu_jobs, reason):
    from gpu_test_selection import SUITES, selection_summary, test_plan as plan_tests
    suites = json.loads(SUITES.read_text())
    plan = plan_tests(paths, ROOT, FILTERS, suites, offload=offload)
    summary = selection_summary(plan, suites, "tested-sha", "Selection note")
    assert f"Planned GPU jobs: {gpu_jobs}." in summary
    assert f"| gluon | {placement} |" in summary
    assert reason in summary and "Selection note" in summary
    assert "`tested-sha`" in summary
    assert "not collected test cases" in summary
    for name in suites["cpu"]["gluon"]["gluon_skip_files"]:
        assert f"`{name}`" in summary
    assert "does not introduce these exclusions" in summary
    assert "not a new skip decision" in summary
