import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from meta_llvm_assets import (github_json, published_assets, read_json_asset, recover_starters, release_assets)

REPOSITORY = "facebookexperimental/triton"
LABEL = "llvm-bundle-ready"


class BundleNotReady(ValueError):
    pass


def ready_name(kind, sha, build):
    prefix = "llvm" if kind == "core" else "amd-llvm"
    return f"{prefix}-{sha[:8]}-{build}.ready.json"


def validate_sdk_records(source, sums, records):
    if set(records) != set(sums):
        raise ValueError("Ready record lacks SDK validation provenance")
    for platform, digest in sums.items():
        record = records[platform]
        if (record.get("schema_version") != 1 or record.get("source") != source or record.get("platform") != platform
                or record.get("sha256") != digest):
            raise ValueError("SDK validation provenance differs from the bundle")
        if not re.fullmatch(r"https://github\.com/facebookexperimental/triton/actions/runs/[0-9]+", record.get(
                "run", "")):
            raise ValueError("SDK validation has an invalid run")
        if type(record.get("attempt")) is not int or record['attempt'] < 1:
            raise ValueError("SDK validation has an invalid attempt")
        if not re.fullmatch(r"[0-9a-f]{40}", record.get("workflow_sha", "")):
            raise ValueError("SDK validation has an invalid workflow revision")


def ready_receipt(repo, kind, sha, tree, build, assets, platforms):
    asset = assets.get(ready_name(kind, sha, build))
    if asset is None:
        return None
    receipt = read_json_asset(repo, asset)
    expected = {
        "schema_version": 1, "kind": kind, "llvm_repository": "facebookexperimental/llvm-project", "llvm_sha": sha,
        "llvm_tree": tree, "build_number": build
    }
    prefix = "llvm" if kind == "core" else "amd-llvm"
    if receipt.get("schema_version") != 1 or receipt.get("source") != expected:
        raise ValueError("Ready record has mismatched source provenance")
    sums = {
        platform: assets[f"{prefix}-{sha[:8]}-{platform}-{build}.tar.gz"]["digest"].removeprefix("sha256:")
        for platform in platforms
    }
    if receipt.get("sha256sum") != sums:
        raise ValueError("Ready record has mismatched platform digests")
    if not re.fullmatch(r"https://github\.com/facebookexperimental/triton/actions/runs/[0-9]+",
                        receipt.get("build_run", "")):
        raise ValueError("Ready record has an invalid build run")
    if not re.fullmatch(r"[0-9a-f]{40}", receipt.get("workflow_sha", "")):
        raise ValueError("Ready record has an invalid workflow revision")
    if type(receipt.get("publication_attempt")) is not int or receipt["publication_attempt"] < 1:
        raise ValueError("Ready record has an invalid publication attempt")
    validate_sdk_records(expected, sums, receipt.get('sdk_validation', {}))
    return receipt


def publish_receipt(repo, source, platforms, validated_dir, run, attempt, workflow_sha):
    from select_meta_llvm_branches import bundle_complete

    _, raw = release_assets(repo, "meta-llvm")
    assets = published_assets(raw)
    kind, sha, build = source["kind"], source["llvm_sha"], source["build_number"]
    if not bundle_complete(kind, sha, source["llvm_tree"], assets, platforms, build):
        raise ValueError("Bundle is incomplete after publication")
    name = ready_name(kind, sha, build)
    if ready_receipt(repo, kind, sha, source["llvm_tree"], build, assets, platforms):
        return
    prefix = "llvm" if kind == "core" else "amd-llvm"
    sums, records = {}, {}
    for platform in platforms:
        archive = f"{prefix}-{sha[:8]}-{platform}-{build}.tar.gz"
        digest = assets[archive]["digest"].removeprefix("sha256:")
        content = (validated_dir / f"{archive}.sha256").read_text()
        if content != f"{digest}  {archive}\n":
            raise ValueError(f"{archive}: SDK validation did not use the published archive")
        sums[platform] = digest
        records[platform] = json.loads((validated_dir / f"{archive}.validation.json").read_text())
    validate_sdk_records(source, sums, records)
    receipt = {
        "schema_version": 1, "source": source, "sha256sum": sums, "build_run": run, "publication_attempt": attempt,
        "workflow_sha": workflow_sha, "sdk_validation": records
    }
    path = validated_dir / name
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    recover_starters(repo, raw, {name})
    subprocess.run(["gh", "release", "upload", "meta-llvm", str(path), "--repo", repo], check=True)


def issue_marker(kind, sha, build):
    return f"<!-- llvm-bundle-ready beta/{kind}/{sha}/{build} -->"


def existing_issue(repo, marker):
    pages = github_json(f"repos/{repo}/issues?state=all&per_page=100", paginate=True)
    for page in pages:
        for issue in page:
            if "pull_request" not in issue and marker in (issue.get("body") or ""):
                return issue["html_url"]
    return None


def claim_issue_creation(repo, kind, sha, build, marker):
    # INVARIANT: the caller holds the readiness-issue workflow lock.
    name = ready_name(kind, sha, build).replace('.ready.json', '.issue-request.json')
    intent = {'schema_version': 1, 'candidate': marker}
    _, raw = release_assets(repo, 'meta-llvm')
    assets = {asset['name']: asset for asset in raw}
    if name in assets:
        if read_json_asset(repo, assets[name]) != intent:
            raise ValueError('Issue submission intent has a different candidate')
        raise RuntimeError('Previous issue-create outcome is unresolved; verify it before retrying creation')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        path.write_text(json.dumps(intent, sort_keys=True) + '\n')
        result = subprocess.run(
            ['gh', 'release', 'upload', 'meta-llvm', str(path), '--repo', repo], capture_output=True, text=True)
        result.check_returncode()
    return name, intent


def clear_rejected_issue_intent(repo, name, intent):
    _, raw = release_assets(repo, 'meta-llvm')
    asset = published_assets(raw).get(name)
    if asset is not None and read_json_asset(repo, asset) == intent:
        github_json(f"repos/{repo}/releases/assets/{asset['id']}", method='DELETE')


def ensure_issue(repo, candidate, platforms):
    from select_meta_llvm_branches import bundle_complete, BRANCHES

    kind, sha, tree = candidate["kind"], candidate["llvm_sha"], candidate["llvm_tree"]
    build = int(candidate["build_number"])
    tip = github_json(f"repos/facebookexperimental/llvm-project/branches/{BRANCHES[kind]}")["commit"]
    if tip["sha"] != sha:
        return "Candidate superseded before readiness notification"
    if tip["commit"]["tree"]["sha"] != tree:
        raise ValueError("Candidate tree differs from the LLVM commit")
    _, raw = release_assets(repo, "meta-llvm")
    assets = published_assets(raw)
    if not bundle_complete(kind, sha, tree, assets, platforms, build):
        raise BundleNotReady("Cannot file an issue for an incomplete bundle")
    receipt = ready_receipt(repo, kind, sha, tree, build, assets, platforms)
    if receipt is None:
        raise BundleNotReady("Cannot file an issue without successful SDK validation")
    marker = issue_marker(kind, sha, build)
    existing = existing_issue(repo, marker)
    if existing:
        return existing
    label = github_json(f"repos/{repo}/labels/{LABEL}", missing_ok=True)
    if label is None:
        subprocess.run([
            "gh", "label", "create", LABEL, "--repo", repo, "--color", "5319e7", "--description",
            "Validated LLVM bundle ready for a manifest-update review diff"
        ], check=True)
    prefix = "llvm" if kind == "core" else "amd-llvm"
    source_asset = assets[f"{prefix}-{sha[:8]}-{build}.source.json"]
    record = {
        "schema_version": 1, "release_line": "beta", "kind": kind, "llvm_repository":
        "facebookexperimental/llvm-project", "llvm_branch": BRANCHES[kind], "llvm_sha": sha, "llvm_tree": tree,
        "build_number": build, "source_asset": source_asset["browser_download_url"], "ready_asset":
        assets[ready_name(kind, sha, build)]["browser_download_url"], "build_run": receipt["build_run"],
        "publication_attempt": receipt["publication_attempt"], "workflow_sha": receipt["workflow_sha"],
        "sdk_validation": receipt["sdk_validation"]
    }
    body = (f"{marker}\n\nThe beta {kind} LLVM bundle is published and its required SDK validations passed.\n\n"
            "Use this candidate to prepare a manifest-update review diff after verifying the current fbsource "
            "LLVM sources and upstream base. This issue does not authorize landing.\n\n"
            f"```json\n{json.dumps(record, indent=2)}\n```\n\n"
            "Keep this issue open until the manifest update lands and the exported consumer pin is verified.\n")
    title = f"[LLVM bundle ready] beta {kind} {sha} build {build}"
    intent_name, intent = claim_issue_creation(repo, kind, sha, build, marker)
    payload = {'title': title, 'body': body, 'labels': [LABEL]}
    result = subprocess.run(['gh', 'api', f'repos/{repo}/issues', '--method', 'POST', '--input', '-'],
                            input=json.dumps(payload), capture_output=True, text=True)
    if result.returncode:
        existing = existing_issue(repo, marker)
        if existing:
            return existing
        rejected = tuple(f'(HTTP {status})' for status in (401, 403, 404, 422))
        if result.stderr.rstrip().endswith(rejected):
            clear_rejected_issue_intent(repo, intent_name, intent)
        result.check_returncode()
    return json.loads(result.stdout)['html_url']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["publish", "issue"])
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    if repo != REPOSITORY:
        raise ValueError("Readiness publication is restricted to the Triton repository")
    import yaml
    with Path(".github/workflows/llvm-build.yml").open() as file:
        configs = yaml.safe_load(file)["jobs"]["build"]["strategy"]["matrix"]["config"]
    platforms = [f"{c['target-os']}-{c['arch']}" for c in configs if c["target-os"] in ("ubuntu", "almalinux")]
    if args.action == "publish":
        publish_receipt(repo, json.loads(os.environ["SOURCE_JSON"]), platforms, Path(os.environ["VALIDATED_DIR"]),
                        f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                        int(os.environ["GITHUB_RUN_ATTEMPT"]), os.environ["GITHUB_SHA"])
    else:
        try:
            print(ensure_issue(repo, json.loads(os.environ["CANDIDATE"]), platforms))
        except BundleNotReady as error:
            print(error)


if __name__ == "__main__":
    main()
