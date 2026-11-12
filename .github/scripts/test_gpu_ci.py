"""GPU CI orchestration tests; no GPU, AWS account, or compiler build needed."""
import io
import json
from pathlib import Path
import subprocess

from botocore.exceptions import ClientError
import pytest
import yaml

from gpu_test_selection import PATH_FILTERS, bump_ref, changed_paths, passed_push_run, select_platforms
from wheel_cache import BUCKET, LEASE_SECONDS, RETENTION_SECONDS, WheelCache, supplied_wheel

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


@pytest.mark.parametrize("state", ["missing", "building", "ready", "expired", "malformed", "denied"])
def test_fork_lookup_reuses_only_ready_wheels_and_never_writes(cache, monkeypatch, state):
    # Fork PRs have no credentials: lookup reuses a ready wheel, and anything else is a local build, not a lease.
    producer, s3, now = cache
    _, lease = producer.acquire("trusted")
    if state == "missing":
        s3.objects.clear()
    elif state in ("ready", "expired"):
        producer.publish(lease, upload(producer))
        if state == "expired":
            now[0] += RETENTION_SECONDS
    elif state == "malformed":
        s3.put_object(Bucket=BUCKET, Key=producer.key, Body=b'{"status": "ready"}')
    elif state == "denied":
        monkeypatch.setattr(s3, "get_object", lambda **kwargs: s3.fail("AccessDenied"))
    before = dict(s3.objects)
    monkeypatch.setattr(s3, "put_object", lambda **kwargs: pytest.fail("fork lookup wrote to the cache"))
    reused = WheelCache(s3, producer.identity, clock=producer.clock, sleep=producer.sleep).lookup()
    assert (reused is not None) == (state == "ready")
    assert s3.objects == before


def test_supplied_wheel_is_reused_only_for_the_exact_build():
    # A consumer skips its own build only for a wheel of exactly the identity it would build, with a usable source.
    identity = {"repository": "owner/repo", "sha": "a" * 40, "abi": "cp312", "platform": "Linux-X64", "recipe": "r"}
    artifact = {"identity": identity, "url": "", "filename": "fbtriton.whl", "sha256": "b" * 64, "artifact": "run_1"}
    s3_wheel = artifact | {"url": "https://example.org/fbtriton.whl", "artifact": ""}
    for wheel in (artifact, s3_wheel):
        assert supplied_wheel(json.dumps(wheel), identity) == wheel
    for field in identity:
        assert supplied_wheel(json.dumps(artifact), identity | {field: "other"}) is None
    for damage in ({"filename": ""}, {"sha256": "b" * 63}, {"artifact": ""}, {"url": "http://example.org/w.whl"}):
        assert supplied_wheel(json.dumps(artifact | damage), identity) is None
    assert supplied_wheel("", identity) is None


def test_codegen_sides_install_their_own_wheel_when_head_has_no_url():
    # Each side installs its own wheel: a head wheel without a URL (a fork PR's artifact) must not borrow the base
    # wheel's S3 URL, whether the head wheel was supplied by Fast correctness or built by the trace.
    import re
    trace = yaml.load((ROOT / ".github/workflows/tlx-codegen-trace.yml").read_text(), Loader=yaml.BaseLoader)
    action = yaml.load((ROOT / ".github/actions/install-prebuilt-triton/action.yml").read_text(),
                       Loader=yaml.BaseLoader)
    snapshot = trace["jobs"]["snapshot"]
    install = next(s for s in snapshot["steps"] if s.get("name", "").startswith("Install prebuilt Triton"))
    routes = {s["name"]: s["if"] for s in action["runs"]["steps"] if s["name"].startswith("Download")}

    def evaluate(template, context):
        # Just enough of the expression language for these values: names, ==, !=, &&, || and fromJSON(...).field.
        def expression(match):
            text = re.sub(r"fromJSON\(matrix\.wheel\)\.(\w+)",
                          lambda m: repr(json.loads(context["matrix.wheel"])[m[1]]), match[1])
            text = re.sub(r"\b(?:matrix|github|inputs|needs)\.[\w.-]+", lambda m: repr(context[m[0]]), text)
            return str(eval(text.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}}))

        return re.sub(r"\$\{\{(.*?)\}\}", expression, template if "${{" in template else "${{ " + template + " }}")

    fork_pr = {
        "github.run_id": "1", "github.event_name": "pull_request", "github.repository": "owner/triton",
        "github.event.pull_request.head.repo.full_name": "fork/triton"
    }
    base = {"url": "https://example.org/base.whl", "filename": "base.whl", "sha256": "b" * 64, "artifact": ""}
    for head_artifact in ("run_1", "run_1_head"):  # Fast correctness's head wheel, or the trace's own build
        head = {"url": "", "filename": "head.whl", "sha256": "a" * 64, "artifact": head_artifact}
        supplied = head_artifact == "run_1"
        context = fork_pr | {
            "needs.resolve.outputs.head_wheel": json.dumps(head) if supplied else "",
            "needs.resolve.outputs.base_wheel": ""
        }
        for side, wheel in (("head", {} if supplied else head), ("base", base)):
            context[f"needs.build-{side}.outputs.wheel"] = json.dumps(wheel) if wheel else ""
            for field in ("url", "filename", "sha256"):
                context[f"needs.build-{side}.outputs.wheel_{field}"] = wheel.get(field, "")
        entries = snapshot["strategy"]["matrix"]["include"]
        assert [entry["side"] for entry in entries] == ["head", "base"]
        for entry, expected in zip(entries, (head, base)):
            side = {"matrix.side": entry["side"], "matrix.wheel": evaluate(entry.get("wheel", "''"), context)}
            inputs = {key: evaluate(value, context | side) for key, value in install["with"].items()}
            assert (inputs["wheel-filename"], inputs["wheel-sha256"]) == (expected["filename"], expected["sha256"])
            action_context = context | {f"inputs.{key}": value for key, value in inputs.items()}
            taken = [name for name, condition in routes.items() if evaluate(condition, action_context) == "True"]
            if expected["url"]:
                assert taken == ["Download prebuilt Triton wheel from S3"] and inputs["wheel-url"] == expected["url"]
            else:
                assert taken == ["Download prebuilt Triton wheel from GitHub artifacts"]
                assert inputs["artifact-name"] == expected["artifact"]


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
    result = subprocess.run([sys.executable, "-c", script],
                            env=os.environ | {"NEEDS": json.dumps(needs), "EVENT_NAME": "push"}, capture_output=True)
    assert (result.returncode == 0) == success, result.stderr.decode()


def test_workflows_share_one_build_and_pin_all_checkouts():

    def workflow(name):
        return yaml.load((ROOT / ".github/workflows" / f"{name}.yml").read_text(), Loader=yaml.BaseLoader)

    parent = workflow("gpu-tests")
    assert sum(job.get("uses") == "./.github/workflows/build.yml" for job in parent["jobs"].values()) == 1
    for platform in FILTERS:
        caller = parent["jobs"][platform]
        assert caller["needs"] == ["select", "build"]
        assert caller["with"]["wheel_url"] == "${{ inputs.wheel_url || needs.build.outputs.wheel_url }}"
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
    suites = json.loads(SUITES.read_text())
    suites["reviewed_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = plan_tests(["lib/Analysis/Allocation.cpp"], ROOT, FILTERS, suites)
    assert plan["cpu"]["codegen"]["files"]
    assert plan["build"]


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
    (["lib/Analysis/Allocation.cpp"], True, "CPU", 6, "match the CPU audit"),
    (["python/triton/runtime/jit.py"], True, "GPU", 6, "current diff changes Python"),
    (["README.md"], True, "Not selected", 0, "No work for this suite"),
    (["lib/Analysis/Allocation.cpp"], False, "GPU", 6, "CPU offload is disabled"),
])
def test_run_summary_explains_actual_placement(paths, offload, placement, gpu_jobs, reason):
    from gpu_test_selection import SUITES, selection_summary, test_plan as plan_tests
    suites = json.loads(SUITES.read_text())
    suites["reviewed_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    suites["cpu"]["gluon"]["dependencies"].append("lib/")
    plan = plan_tests(paths, ROOT, FILTERS, suites, offload=offload)
    summary = selection_summary(plan, suites, "tested-sha", "Selection note")
    assert f"Planned GPU jobs: {gpu_jobs}." in summary
    assert f"| codegen | {placement} |" in summary
    assert reason in summary and "Selection note" in summary
    assert "Selected commit: `tested-sha`." in summary
    assert "Tested commit:" not in summary
    assert "not collected test cases" in summary
    for name in suites["cpu"]["gluon"]["gluon_skip_files"]:
        assert f"`{name}`" in summary
    assert "does not introduce these exclusions" in summary
    assert "not a new skip decision" in summary


def test_unreadable_kernel_test_retains_full_coverage(tmp_path):
    from gpu_test_selection import select_kernel_tests
    path = tmp_path / "python/test/unit/language/test_tlx_unreadable.py"
    path.parent.mkdir(parents=True)
    path.symlink_to("missing-source.py")
    plan = select_kernel_tests(["third_party/tlx/tutorials/changed.py"], tmp_path)
    assert plan["kernel_only"] is False
    assert plan["correctness_mode"] == "all"


@pytest.mark.parametrize("policy", [None, "def broken(", "_VERSION_SKEW_FILES = unknown"])
def test_unreadable_gluon_audit_only_blocks_gluon_cpu_placement(tmp_path, policy):
    from gpu_test_selection import SUITES, cpu_ineligible_reason
    suites = json.loads(SUITES.read_text())
    spec = suites["cpu"]["gluon"]
    directory = tmp_path / "python/test/gluon"
    directory.mkdir(parents=True)
    for path in spec["files"]:
        (tmp_path / path).write_text("")
    for name in spec["gluon_skip_files"]:
        (directory / name).write_text("")
    if policy is not None:
        (directory / "conftest.py").write_text(policy)
    assert cpu_ineligible_reason(["lib/Analysis/Allocation.cpp"], tmp_path, spec) is not None
    assert cpu_ineligible_reason(["lib/Analysis/Allocation.cpp"], tmp_path, suites["cpu"]["codegen"]) is None


def workflow_condition(expression, context):
    import re
    expression = expression.removeprefix('${{').removesuffix('}}').strip()
    has_status = 'cancelled()' in expression or 'always()' in expression
    if not has_status and any(value != 'success' for key, value in context.items() if key.endswith('.result')):
        return False
    expression = expression.replace('!cancelled()', str(not context.get('cancelled', False)))
    expression = expression.replace('always()', 'True').replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'(github\.event_name|(?:needs|steps)\.[\w-]+\.(?:result|outputs\.[\w]+))',
                        lambda match: repr(context[match.group()]), expression)
    return eval(expression, {'__builtins__': {}}, {})


@pytest.mark.parametrize('gate', ['success', 'failure', 'cancelled', 'skipped'])
def test_pr_full_allocation_requires_the_verified_fast_gate(gate):
    fast = yaml.load((ROOT / '.github/workflows/fast-gpu-tests.yml').read_text(), Loader=yaml.BaseLoader)
    full = yaml.load((ROOT / '.github/workflows/gpu-tests.yml').read_text(), Loader=yaml.BaseLoader)
    call = fast['jobs']['full']
    assert call['uses'] == './.github/workflows/gpu-tests.yml'
    assert fast['jobs']['result']['name'] == 'Fast correctness'
    context = {'github.event_name': 'pull_request', 'needs.result.result': gate, 'needs.build.result': 'skipped'}
    assert workflow_condition(call['if'], context) == (gate == 'success')
    assert not workflow_condition(call['if'], context | {'cancelled': True})
    assert not workflow_condition(call['if'], context | {'github.event_name': 'workflow_dispatch'})
    assert 'pull_request' not in full['on'] and {'push', 'schedule', 'workflow_dispatch'} <= set(full['on'])
    for key in ('commit', 'wheel_url', 'wheel_filename', 'wheel_sha256'):
        assert call['with'][key] == '${{ needs.build.outputs.' + key + ' }}'
    assert not workflow_condition(
        full['jobs']['build']['if'],
        {'github.event_name': 'pull_request', 'needs.select.outputs.build': 'true', 'needs.select.result': 'success'})
    for platform in ('h100', 'b200', 'mi350', 'cpu'):
        worker = full['jobs']['cpu-tests' if platform == 'cpu' else platform]
        context = {
            'github.event_name': 'pull_request', 'needs.select.result': 'success', 'needs.build.result': 'skipped',
            f'needs.select.outputs.{platform}': 'true'
        }
        assert workflow_condition(worker['if'], context)
        assert not workflow_condition(worker['if'], context | {'needs.select.result': 'failure'})
        assert not workflow_condition(worker['if'], context | {'cancelled': True})
        assert not workflow_condition(worker['if'], context | {f'needs.select.outputs.{platform}': 'false'})


@pytest.mark.parametrize('fork,url,valid', [(False, 'https://example.org/w.whl', True), (False, '', False),
                                            (True, '', True)])
def test_shared_full_wheel_validates_source_and_fork_artifact_identity(fork, url, valid):
    import os
    import sys
    full = yaml.load((ROOT / '.github/workflows/gpu-tests.yml').read_text(), Loader=yaml.BaseLoader)
    step = next(s for s in full['jobs']['select']['steps'] if s.get('name') == 'Validate shared wheel source')
    source = step['run'].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    env = os.environ | {
        'COMMIT': 'a' * 40, 'EXPECTED': 'a' * 40, 'WHEEL_URL': url, 'WHEEL_FILENAME': 'fbtriton.whl', 'WHEEL_SHA256':
        'b' * 64, 'IS_FORK_PR': str(fork).lower()
    }

    def run(extra):
        return subprocess.run([sys.executable, '-c', source], env=env | extra, capture_output=True).returncode

    assert (run({}) == 0) == valid
    assert run({'COMMIT': 'c' * 40}) != 0
    assert run({'WHEEL_SHA256': ''}) != 0
    assert run({'WHEEL_FILENAME': ''}) != 0
    assert not workflow_condition(step['if'],
                                  {'github.event_name': 'pull_request', 'steps.select.outputs.build': 'false'})


def test_fast_automatic_runs_are_pr_only():
    workflow = yaml.load((ROOT / ".github/workflows/fast-gpu-tests.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}


@pytest.mark.parametrize("paths,expected", [
    (["README.md"], set()),
    (["test/Conversion/reduce.mlir"], set()),
    (["third_party/tlx/ops/kernels/mm/sm100.py"], {"b200"}),
    (["third_party/amd/backend/compiler.py"], {"mi350"}),
    (["python/triton/runtime/jit.py"], {"h100", "b200", "mi350"}),
    ([".github/fast-test-suites.json"], {"h100", "b200", "mi350"}),
    ([".github/actions/setup-test-env/action.yml"], {"h100", "b200", "mi350"}),
    ([".github/actions/install-torchao/action.yml"], {"h100", "b200", "mi350"}),
    (None, {"h100", "b200", "mi350"}),
])
def test_fast_suites_preserve_affected_platform_routing(paths, expected):
    from fast_tests import select
    selected = select(paths, FILTERS)
    assert {key for key in FILTERS if selected[key]} == expected
    assert selected["cpu"] == selected["build"] == bool(expected)


@pytest.mark.parametrize("path,platform,targets", [
    ("python/test/unit/tlx_ops/test_flash_attn_gfx950.py", "mi350",
     ["python/test/unit/tlx_ops/test_flash_attn_gfx950.py::test_flash_attn_gfx950_forward_and_backward"]),
    ("third_party/tlx/tutorials/blackwell_gemm_clc.py", "b200",
     ["third_party/tlx/tutorials/testing/test_correctness.py::test_blackwell_gemm_clc"]),
    ("python/test/unit/tlx_ops/test_mm_sm100.py", "b200", [
        "python/test/unit/tlx_ops/test_mm_sm100.py::test_kernel_legacy_positional_launch_matches_reference",
        "python/test/unit/tlx_ops/test_mm_sm100.py::test_npot_block_n_matches_reference"
    ]),
])
def test_fast_pr_selects_corresponding_critical_tests(path, platform, targets):
    from fast_tests import fast_plan, SUITES
    suites = json.loads(SUITES.read_text())
    selected, plans = fast_plan([path], ROOT, FILTERS, suites, True)
    assert {key for key in FILTERS if selected[key]} == {platform}
    assert plans[platform]["baseline"] == suites[platform]
    assert any(group["file"] == targets[0].split("::", 1)[0] for group in plans[platform]["touched"])


@pytest.mark.parametrize("paths,narrow", [
    (["python/triton/runtime/jit.py"], True),
    (["python/test/unit/tlx_ops/conftest.py"], True),
    (["python/test/unit/tlx_ops/test_addmm_sm100.py"], True),
    ([
        "third_party/tlx/tutorials/testing/test_correctness_autows.py",
        "third_party/tlx/tutorials/blackwell_gemm_clc.py"
    ], True),
    (["python/test/unit/tlx_ops/test_addmm_sm100.py", "third_party/tlx/tutorials/blackwell_gemm_clc.py"], True),
    (["python/test/unit/tlx_ops/test_flash_attn_sm100.py"], False),
])
def test_fast_shared_unknown_and_manual_changes_retain_critical_coverage(paths, narrow):
    from fast_tests import fast_plan, SUITES
    suites = json.loads(SUITES.read_text())
    selected, plans = fast_plan(paths, ROOT, FILTERS, suites, narrow)
    assert selected["b200"] and plans["b200"]["baseline"] == suites["b200"]


@pytest.mark.parametrize("content", [
    '<testsuite/>',
    '<testsuite><testcase classname="tests.smoke" name="test_other"/></testsuite>',
    '<testsuite><testcase classname="tests.smoke" name="test_required"><skipped/></testcase></testsuite>',
    '<testsuite><testcase classname="tests.smoke" name="test_required"><failure/></testcase></testsuite>',
    '<testsuite><testcase classname="tests.smoke" name="test_required"><error/></testcase></testsuite>',
])
def test_fast_suite_requires_executed_passing_evidence(tmp_path, content):
    from fast_tests import verify_evidence
    path = tmp_path / "results.xml"
    path.write_text(content)
    with pytest.raises(ValueError):
        verify_evidence(["tests/smoke.py::test_required"], path)


def test_fast_suite_accepts_passing_parameterized_evidence(tmp_path):
    from fast_tests import verify_evidence
    path = tmp_path / "results.xml"
    path.write_text('<testsuites><testsuite><testcase classname="tests.smoke" name="test_required[fp16]"/> '
                    '<testcase classname="tests.smoke" name="test_required[bf16]"/></testsuite></testsuites>')
    assert verify_evidence(["tests/smoke.py::test_required"], path) == 2


@pytest.mark.parametrize("kernel,platform", [
    ("third_party/tlx/ops/kernels/flash_attn_mxfp8/gfx950_train.py", "mi350"),
    ("third_party/tlx/ops/kernels/flash_attn_mxfp8/sm100.py", "b200"),
])
def test_fast_baseline_unions_touched_tests_outside_the_baseline(kernel, platform):
    from fast_tests import fast_plan, SUITES
    suites = json.loads(SUITES.read_text())
    _, plans = fast_plan([kernel], ROOT, FILTERS, suites, True)
    plan = plans[platform]
    file = f"python/test/unit/tlx_ops/test_flash_attn_mxfp8_{'gfx950' if platform == 'mi350' else 'sm100'}.py"
    assert all(target.split("::", 1)[0] != file for target in suites[platform])
    assert plan["baseline"] == suites[platform]
    assert {group["file"] for group in plan["touched"]} == {file, "python/test/unit/tlx_ops/test_invariance_api.py"}
    _, mixed = fast_plan(["lib/Analysis/Allocation.cpp", kernel], ROOT, FILTERS, suites, True)
    assert mixed[platform]["touched"] == plan["touched"]
    assert all(mixed[key]["baseline"] == suites[key] for key in FILTERS)


def test_fast_shared_kernel_mapping_uses_its_supported_platform():
    from fast_tests import fast_plan, SUITES
    suites = json.loads(SUITES.read_text())
    _, plans = fast_plan(["third_party/tlx/ops/kernels/bmm/gfx950_torch.py"], ROOT, FILTERS, suites, True)
    assert all(not plans[platform]["unmapped"] for platform in FILTERS)
    assert any(group["file"].endswith("test_torchtlx_bmm_gfx950.py") for group in plans["mi350"]["touched"])
    assert not plans["h100"]["touched"] and not plans["b200"]["touched"]


def test_fast_shared_data_deleted_and_helper_inputs_keep_baseline(tmp_path):
    from fast_tests import fast_plan
    directory = tmp_path / "python/test/unit"
    directory.mkdir(parents=True)
    (directory / "test_new.py").write_text("def test_op():\n    assert True\n")
    (directory / "test_perf.py").write_text("import argparse\nparser = argparse.ArgumentParser()\n")
    suites = {"b200": ["test_base.py::test_baseline"], "cpu": []}
    paths = [
        "third_party/tlx/ops/_catalog.py", "third_party/tlx/ops/kernels/mm/_shapes.py", "python/test/unit/test_old.py",
        "python/test/unit/test_perf.py", "python/test/unit/test_new.py"
    ]
    _, plans = fast_plan(paths, tmp_path, {"b200": ["**"]}, suites, True)
    assert plans["b200"]["baseline"] == suites["b200"]
    assert plans["b200"]["touched"] == [{"file": "python/test/unit/test_new.py", "k": ""}]
    reasons = {item["path"]: item["reason"] for item in plans["b200"]["limited"]}
    assert reasons["python/test/unit/test_old.py"] == "deleted test input"
    assert reasons["python/test/unit/test_perf.py"] == "no pytest functions"
    assert not plans["b200"]["unmapped"]


def fast_fixture_run(tmp_path, source, environment=None, separate_baseline=False):
    import os
    import sys
    (tmp_path / "test_sample.py").write_text(source)
    plan = {
        "platform": "b200", "baseline": ["test_sample.py::test_baseline"], "touched":
        [{"file": "test_sample.py", "k": ""}], "unmapped": [], "limited": []
    }
    if separate_baseline:
        (tmp_path / "test_base.py").write_text("def test_baseline():\n    assert True\n")
        plan["baseline"] = ["test_base.py::test_baseline"]
    junit = tmp_path / "result.xml"
    result = subprocess.run(
        [
            sys.executable, "-c", "import json,sys; from pathlib import Path; from fast_tests import run_plan; "
            "run_plan(json.loads(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])",
            json.dumps(plan),
            str(tmp_path),
            str(junit)
        ], env=os.environ | {
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "PYTHONPATH": str(ROOT / ".github/scripts"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"
        } | (environment or {}), capture_output=True, text=True)
    manifest = json.loads(junit.with_suffix(".selection.json").read_text())
    return result, manifest


def test_fast_representatives_try_compatible_cases_without_removing_baseline(tmp_path):
    from fast_tests import verify_manifests
    result, manifest = fast_fixture_run(
        tmp_path, '''import pytest
@pytest.mark.parametrize("value", [0, 1])
def test_baseline(value):
    assert value in (0, 1)
@pytest.mark.parametrize("value", [0, 1, 2])
def test_touched(value):
    if value == 0:
        pytest.skip("incompatible parameter")
    assert value > 0
class TestNested:
    @pytest.mark.parametrize("value", [0, 1])
    def test_expected(self, value):
        if value == 0:
            pytest.xfail("unsupported parameter")
        assert value == 1
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest["baseline"] == ["test_sample.py::test_baseline[0]", "test_sample.py::test_baseline[1]"]
    assert manifest["selected"] == [
        *manifest["baseline"], "test_sample.py::test_touched[1]", "test_sample.py::TestNested::test_expected[1]"
    ]
    assert len(manifest["exclusions"]) == 2
    summary = (tmp_path / "summary.md").read_text()
    assert "test_sample.py::test_baseline[0] | both | pass |" in summary
    assert "test_sample.py::test_touched[0] | touched | skipped |" in summary
    assert "test_sample.py::test_touched[1] | touched | pass |" in summary
    assert "test_sample.py::test_touched[2] | touched | not executed |" in summary
    assert "JUnit seconds" in summary
    verify_manifests([manifest])


def test_fast_failures_and_claimed_pass_integrity_still_block_coverage_warnings(tmp_path):
    import copy
    from fast_tests import verify_manifests
    result, manifest = fast_fixture_run(
        tmp_path, '''def test_baseline():
    assert True
def test_touched():
    assert True
''')
    assert result.returncode == 0, result.stdout + result.stderr
    function = "test_sample.py::test_touched"
    passing_platform = manifest | {"platform": "h100"}
    for problem in ("failed", "wrong_node", "missing_selected"):
        broken = copy.deepcopy(manifest)
        broken["unmapped"].append("unmapped_kernel.py")
        if problem == "failed":
            broken["functions"][function]["status"] = "failed"
        elif problem == "wrong_node":
            broken["functions"][function]["passed"] = broken["baseline"][0]
        else:
            broken["selected"].remove(broken["functions"][function]["passed"])
        with pytest.raises(ValueError):
            verify_manifests([broken, passing_platform])


def test_fast_failure_cannot_be_replaced_by_a_passing_parameter(tmp_path):
    result, manifest = fast_fixture_run(
        tmp_path, '''import pytest
def test_baseline():
    assert True
@pytest.mark.parametrize("value", [0, 1])
def test_touched(value):
    assert value == 1
''')
    assert result.returncode != 0
    assert manifest["functions"]["test_sample.py::test_touched"]["status"] == "failed"
    assert manifest["functions"]["test_sample.py::test_touched"]["attempts"] == 1
    assert "test_sample.py::test_touched[1]" not in manifest["selected"]
    summary = (tmp_path / "summary.md").read_text()
    assert "Run failed:" in summary
    assert "test_sample.py::test_touched[0] | touched | fail |" in summary
    assert "test_sample.py::test_touched[1] | touched | not executed |" in summary
    assert "test_sample.py::test_touched[1] | touched | pass |" not in summary


@pytest.mark.parametrize("baseline_skip", [False, True])
def test_fast_skips_never_count_as_required_coverage(tmp_path, baseline_skip):
    from fast_tests import verify_manifests
    source = '''import pytest
def test_baseline():
    %s
@pytest.mark.skipif(True, reason="unsupported platform")
def test_touched():
    assert True
''' % ('pytest.skip("baseline cannot skip")' if baseline_skip else 'assert True')
    result, manifest = fast_fixture_run(tmp_path, source)
    if baseline_skip:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert manifest["functions"]["test_sample.py::test_touched"]["status"] == "excluded"
        assert manifest["functions"]["test_sample.py::test_touched"]["reason"] == "unsupported platform"
        verify_manifests([manifest])
    assert "test_sample.py::test_touched" not in manifest["selected"]


@pytest.mark.parametrize("cases", [1, 40])
def test_fast_runtime_unavailability_is_not_a_declared_exclusion(tmp_path, cases):
    from fast_tests import verify_manifests
    result, manifest = fast_fixture_run(
        tmp_path, '''import pytest
def test_baseline():
    assert True
@pytest.mark.parametrize("value", range(%d))
def test_touched(value):
    pytest.skip("required runtime dependency unavailable")
''' % cases)
    assert result.returncode == 0, result.stdout + result.stderr
    gaps = verify_manifests([manifest])
    assert [gap["target"] for gap in gaps] == ["test_sample.py::test_touched"]
    assert "test_sample.py::test_touched" not in manifest["selected"]
    summary = (tmp_path / "summary.md").read_text()
    assert "test_sample.py::test_touched[0] | touched | skipped |" in summary
    assert "required runtime dependency unavailable" in summary
    assert "runtime_unavailable" in summary if cases == 1 else "limit_reached" in summary
    import os
    import sys
    needs = {
        "select": {
            "outputs":
            {platform: "true" if platform == "b200" else "false"
             for platform in ("h100", "b200", "mi350", "cpu")}
        }
    }
    verified = subprocess.run(
        [sys.executable,
         str(ROOT / ".github/scripts/fast_tests.py"), "verify", "--evidence",
         str(tmp_path)],
        env=os.environ | {"NEEDS": json.dumps(needs), "GITHUB_STEP_SUMMARY": str(tmp_path / "aggregate.md")},
        capture_output=True, text=True)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "Fast correctness coverage warnings" in (tmp_path / "aggregate.md").read_text()
    assert "coverage_gaps" in verified.stdout and "::warning::" in verified.stdout
    assert "test_touched" in (tmp_path / "aggregate.md").read_text()


def test_fast_platform_coverage_gaps_are_advisory_and_clear_with_a_pass(tmp_path):
    from fast_tests import verify_manifests
    source = '''import os
import pytest
@pytest.mark.skipif(os.environ["SUPPORTED"] == "no", reason="requires supported hardware")
def test_touched():
    assert True
'''
    manifests = []
    for supported in ("no", "yes"):
        result, manifest = fast_fixture_run(tmp_path, source, {"SUPPORTED": supported}, separate_baseline=True)
        assert result.returncode == 0, result.stdout + result.stderr
        manifest["platform"] = "h100" if supported == "no" else "b200"
        manifests.append(manifest)
    gaps = verify_manifests(manifests[:1])
    assert gaps and gaps[0]["target"] == "test_sample.py"
    assert "requires supported hardware" in gaps[0]["details"][0]["reason"]
    assert not verify_manifests(manifests)


@pytest.mark.parametrize("test_body", [
    "def test_touched():\n    assert True\n",
    "import unittest\nclass DownloadTest(unittest.TestCase):\n    def test_touched(self):\n        assert True\n",
])
def test_fast_collection_cannot_silently_remove_a_touched_function(tmp_path, test_body):
    from fast_tests import verify_manifests
    (tmp_path / "conftest.py").write_text('''def pytest_collection_modifyitems(items):
    items[:] = [item for item in items if item.name != "test_touched"]
''')
    result, manifest = fast_fixture_run(tmp_path, "def test_baseline():\n    assert True\n" + test_body)
    assert result.returncode == 0, result.stdout + result.stderr
    gaps = verify_manifests([manifest])
    assert gaps and gaps[0]["details"][0]["status"] == "missing"
    assert "test_touched" in gaps[0]["target"]
