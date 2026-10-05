"""GPU CI orchestration tests; no GPU, AWS account, or compiler build needed."""
import io
import json
from pathlib import Path
import subprocess

from botocore.exceptions import ClientError
import pytest
import yaml

from gpu_test_selection import PATH_FILTERS, changed_paths, select_platforms
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


def test_cache_identity_uses_workflow_tooling_for_old_source_refs(tmp_path, monkeypatch):
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
    before = wheel_cache.build_identity()
    assert before["sha"] == "a" * 40
    recipe.write_text("changed build recipe")
    assert wheel_cache.build_identity()["recipe"] != before["recipe"]


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
        "build": str(selected).lower(), "h100": str(selected).lower(), "b200": "false", "mi350": "false"
    }
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
                if step.get("name") == "Download prebuilt Triton wheel from S3":
                    assert "sha256sum --check" in step["run"]
    assert LEASE_SECONDS > int(workflow("build")["jobs"]["build"]["timeout-minutes"]) * 60
