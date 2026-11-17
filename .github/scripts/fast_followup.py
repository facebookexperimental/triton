"""Hand a PR's GPU tests head wheel to its workflow_run follow-up (the TLX codegen trace).

GPU tests (pull_request) writes a versioned manifest naming its run, attempt, PR, PR head, test merge commit
and validated wheel. A follow-up runs on the default branch with its own token, so it trusts nothing in that
manifest it cannot check: the producer run comes from the workflow_run event, the PR and its current head from the
API, the merge commit's parents from git, and the wheel's identity is recomputed from the build recipe at that merge
commit. It reads all of this as data and never runs code from the PR.

GPU tests writes its manifest with the standard library only; wheel_cache (boto3) is imported only by the
follow-up commands that check wheels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

VERSION = 1
PRODUCER_WORKFLOW = ".github/workflows/gpu-tests.yml"
MANIFEST_FILE = "fast-followup.json"
SHA = re.compile(r"[0-9a-f]{40}")


class Rejected(Exception):
    """The triggering run or its manifest is not a source this follow-up may test."""


def manifest_name(attempt):
    return f"fast-followup-v{VERSION}-attempt-{attempt}"


def manifest(env):
    """The producer's manifest, written by GPU tests after its build."""
    wheel = json.loads(env["WHEEL"]) if env.get("WHEEL") else None
    return {
        "version": VERSION, "repository": env["GITHUB_REPOSITORY"], "run_id": int(env["GITHUB_RUN_ID"]), "run_attempt":
        int(env["GITHUB_RUN_ATTEMPT"]), "pr_number": int(env["PR_NUMBER"]), "pr_head_sha": env["PR_HEAD_SHA"],
        "test_merge_sha": env["GITHUB_SHA"], "wheel": wheel
    }


def find_artifact(api, repo, run_id, name):
    """The id of the unexpired artifact with exactly this name in exactly this run, or None."""
    listing = api(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&name={name}")
    found = [
        a for a in listing.get("artifacts", [])
        if a.get("name") == name and (a.get("workflow_run") or {}).get("id") == run_id and not a.get("expired")
    ]
    if len(found) > 1:
        raise Rejected(f"Run {run_id} has {len(found)} artifacts named {name}")
    return found[0]["id"] if found else None


def producer(event, repo):
    """The triggering GPU tests run, if it is a successful pull_request run of this repository's workflow."""
    run = event.get("workflow_run") or {}
    if (run.get("event") != "pull_request" or run.get("conclusion") != "success"
            or (run.get("repository") or {}).get("full_name") != repo
            or str(run.get("path", "")).split("@")[0] != PRODUCER_WORKFLOW):
        raise Rejected("The triggering run is not a successful pull_request run of GPU tests")
    if not SHA.fullmatch(str(run.get("head_sha"))):
        raise Rejected("The triggering run has no PR head SHA")
    return run


def producer_merge(run, repo, api, text):
    from ci_health_logs import checkout_sha

    jobs, page = [], 1
    while True:
        listing = api(f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=all&per_page=100&page={page}")
        batch = listing.get("jobs", [])
        jobs.extend(batch)
        if len(jobs) >= listing["total_count"]:
            break
        if not batch:
            raise Rejected("The producer job listing is incomplete")
        page += 1
    selected = [job for job in jobs if job.get("name") == "select" and job.get("run_attempt", 1) <= run["run_attempt"]]
    if not selected:
        raise Rejected("The GPU tests producer has no source selection job")
    job = max(selected, key=lambda item: (item.get("run_attempt", 1), item["id"]))
    if job.get("conclusion") != "success":
        raise Rejected("The GPU tests producer source selection did not pass")
    merge = checkout_sha(job, text(f"repos/{repo}/actions/jobs/{job['id']}/logs"))
    if not merge:
        raise Rejected("The GPU tests producer checkout log does not identify one tested commit")
    return merge


def verify(event, manifest, repo, api, git, recipe_identity, text=None):
    """Check the manifest against the triggering run, the API and git. Returns the verified source and wheel.

    `recipe_identity(sha)` is the identity a build of `sha` would have with the build recipe at `sha`.
    Status "stale" means the run was re-run or the PR head moved. A base-only update does not invalidate the source
    that this GPU tests attempt tested; when GitHub's current merge differs, authenticate that source from its checkout log.
    """
    run = producer(event, repo)
    if (manifest.get("version") != VERSION or manifest.get("repository") != repo or manifest.get("run_id") != run["id"]
            or manifest.get("run_attempt") != run.get("run_attempt") or manifest.get("pr_head_sha") != run["head_sha"]):
        raise Rejected("The manifest does not describe the triggering run and attempt")
    live = api(f"repos/{repo}/actions/runs/{run['id']}")
    if (live.get("id") != run["id"] or live.get("event") != "pull_request" or live.get("head_sha") != run["head_sha"]
            or (live.get("repository") or {}).get("full_name") != repo
            or str(live.get("path", "")).split("@")[0] != PRODUCER_WORKFLOW):
        raise Rejected("The API does not describe the triggering run as GPU tests for this PR head")
    live_attempt = live.get("run_attempt")
    head_repo = (run.get("head_repository") or {}).get("full_name")
    number = manifest.get("pr_number")
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise Rejected("The manifest has no PR number")
    # Open or closed: these checks may finish after the PR is closed for landing. The PR must still be the
    # triggering run's branch with the same head and test merge commit below.
    pr = api(f"repos/{repo}/pulls/{number}")
    if (pr.get("number") != number or pr["head"]["repo"]["full_name"] != head_repo
            or pr["head"]["ref"] != run.get("head_branch") or pr["base"]["repo"]["full_name"] != repo):
        raise Rejected(f"PR {number} is not the triggering run's {head_repo}:{run.get('head_branch')}")
    source = {
        "pr": pr["number"], "pr_head": run["head_sha"], "base_ref": pr["base"]["ref"], "head_ref": pr["head"]["ref"],
        "head_repo": head_repo, "fork": head_repo != repo, "producer_run_id": run["id"], "producer_attempt":
        run["run_attempt"]
    }
    if live_attempt != run["run_attempt"] or pr["head"]["sha"] != run["head_sha"]:
        return {"status": "stale", **source}
    if live.get("status") != "completed" or live.get("conclusion") != "success":
        raise Rejected("The API does not report the triggering attempt as a successful run")
    merge = manifest.get("test_merge_sha")
    if not SHA.fullmatch(str(merge)):
        raise Rejected("The manifest has no test merge SHA")
    if pr.get("merge_commit_sha") != merge:
        if text is None or producer_merge(run, repo, api, text) != merge:
            raise Rejected("The manifest merge does not match the GPU tests producer checkout")
    git("fetch", "--no-tags", "--quiet", "origin", merge,
        f"+refs/heads/{source['base_ref']}:refs/remotes/origin/{source['base_ref']}")
    parents = git("rev-list", "--parents", "-n", "1", merge).split()
    if len(parents) != 3 or parents[0] != merge or parents[2] != run["head_sha"]:
        raise Rejected("The test merge commit does not merge the PR head")
    try:
        git("merge-base", "--is-ancestor", parents[1], f"origin/{source['base_ref']}")
    except subprocess.CalledProcessError as exc:
        raise Rejected("The test merge commit's first parent is not on the PR's base branch") from exc
    wheel, reason = manifest.get("wheel"), ""
    if wheel:
        from wheel_cache import supplied_wheel
        wheel = supplied_wheel(json.dumps(wheel), recipe_identity(merge))
        reason = "" if wheel else "the producer's wheel does not match the merge commit's build identity"
    else:
        reason = "GPU tests built no head wheel"
    return {"status": "ok", "merge": merge, "base": parents[1], "wheel": wheel, "wheel_reason": reason, **source}


def gh_api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def gh_text(path):
    return subprocess.check_output(["gh", "api", path, "--allow-escape-sequences"], text=True)


def git(*args):
    return subprocess.check_output(["git", *args], text=True)


def recipe_identity(sha):
    """Identity of a build of `sha` with the recipe at `sha`, read from git as data."""
    from wheel_cache import build_identity
    with tempfile.TemporaryDirectory() as tooling:
        for path in ("workflows/build.yml", "scripts/wheel_cache.py"):
            target = Path(tooling, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{sha}:.github/{path}"]))
        return build_identity(sha, tooling=tooling)


def extract_member(archive, name, dest):
    """Write the one member `name` of the producer's artifact zip to `dest`.

    Producer artifacts may come from a fork, so the archive must hold exactly that one plain file name; anything else
    (extra entries, directories, paths) is rejected and nothing is written.
    """
    if not name or Path(name).name != name or name in (".", "..") or "\\" in name:
        raise Rejected(f"Unsafe artifact member name {name!r}")
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if [m.filename for m in members] != [name] or members[0].is_dir():
            raise Rejected(f"The producer artifact must hold only {name}, found {[m.filename for m in members]}")
        data = zf.read(members[0])
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def emit(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "locate", "extract", "verify", "relayed"))
    parser.add_argument("--out", type=Path, help="manifest, extract: file to write")
    parser.add_argument("--zip", type=Path, help="extract, relayed: the downloaded producer artifact zip")
    parser.add_argument("--manifest", type=Path, help="verify: the extracted manifest")
    parser.add_argument("--dir", type=Path, help="relayed: directory receiving the producer's wheel")
    args = parser.parse_args()
    env = os.environ
    repo = env["GITHUB_REPOSITORY"]
    if args.command == "manifest":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest(env)))
        emit(name=manifest_name(env["GITHUB_RUN_ATTEMPT"]))
        return
    if args.command == "extract":
        extract_member(args.zip, MANIFEST_FILE, args.out)
        return
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
    if args.command == "locate":
        run = producer(event, repo)
        artifact = find_artifact(gh_api, repo, run["id"], manifest_name(run["run_attempt"]))
        if artifact is None:
            # A run without a build publishes no manifest.
            print(f"GPU tests run {run['id']} attempt {run['run_attempt']} published no follow-up manifest v{VERSION}")
        emit(manifest_id=artifact or "")
        return
    if args.command == "verify":
        result = verify(event, json.loads(args.manifest.read_text()), repo, gh_api, git, recipe_identity, gh_text)
        wheel = result.pop("wheel", None)
        relay = ""
        if wheel and not wheel["url"]:
            # A fork's wheel is an artifact of the producer run; this run copies it after checking its checksum.
            relay = find_artifact(gh_api, repo, result["producer_run_id"], wheel["artifact"]) or ""
            if not relay:
                wheel, result["wheel_reason"] = None, "the producer's wheel artifact has expired or is missing"
        emit(**{k: str(v).lower() if isinstance(v, bool) else v for k, v in result.items()})
        emit(wheel=json.dumps(wheel) if wheel and not relay else "", relay_wheel=json.dumps(wheel) if relay else "",
             relay_id=relay)
        print(json.dumps({**result, "wheel": wheel}, indent=2))
        return
    # relayed: the producer's wheel artifact, holding only the verified wheel file, becomes this run's artifact.
    wheel = json.loads(env["RELAY_WHEEL"])
    path = extract_member(args.zip, wheel["filename"], args.dir / wheel["filename"])
    with open(path, "rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != wheel["sha256"]:
            sys.exit("The producer artifact's wheel does not match the manifest checksum")
    emit(wheel=json.dumps({**wheel, "artifact": env["RELAY_ARTIFACT"]}))


if __name__ == "__main__":
    try:
        main()
    except Rejected as exc:
        sys.exit(f"::error::{exc}")
