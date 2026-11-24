import hashlib
import json
import os
from pathlib import Path
import re

from meta_llvm_assets import github_json, published_assets, release_assets

BRANCHES = {
    "core": "triton_beta",
    "amd": "fbtriton-beta-amd-codegen",
}
FORK = "facebookexperimental/llvm-project"
BUILD_NUMBER = 1


def source_digest(kind, sha, tree):
    source = {
        "schema_version": 1,
        "kind": kind,
        "llvm_repository": FORK,
        "llvm_sha": sha,
        "llvm_tree": tree,
        "build_number": BUILD_NUMBER,
    }
    content = json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n"
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def bundle_complete(kind, sha, tree, assets, platforms):
    prefix = "llvm" if kind == "core" else "amd-llvm"
    stem = f"{prefix}-{sha[:8]}"
    source_name = f"{stem}-{BUILD_NUMBER}.source.json"
    source = assets.get(source_name)
    if source is None:
        return False
    if source.get("digest") != source_digest(kind, sha, tree):
        raise ValueError(f"{source_name}: published LLVM source provenance differs")
    for platform in platforms:
        name = f"{stem}-{platform}-{BUILD_NUMBER}.tar.gz"
        archive = assets.get(name)
        checksum = assets.get(f"{name}.sha256")
        if archive is None or checksum is None:
            return False
        digest = archive.get("digest") or ""
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name}: missing GitHub archive digest")
        content = f"{digest.removeprefix('sha256:')}  {name}\n"
        expected = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        if checksum.get("digest") != expected:
            raise ValueError(f"{name}: published checksum differs from archive digest")
    return True


def branch_requests(tips, assets, platforms):
    assets = published_assets(assets)
    requests = []
    for kind, tip in tips.items():
        sha, tree = tip["sha"], tip["commit"]["tree"]["sha"]
        if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ValueError(f"{BRANCHES[kind]}: expected a full immutable LLVM SHA")
        if not bundle_complete(kind, sha, tree, assets, platforms):
            requests.append({"kind": kind, "llvm_sha": sha, "build_number": str(BUILD_NUMBER)})
    return requests


def main():
    import yaml

    with Path(".github/workflows/llvm-build.yml").open() as source:
        configs = yaml.safe_load(source)["jobs"]["build"]["strategy"]["matrix"]["config"]
    platforms = [
        f"{config['target-os']}-{config['arch']}" for config in configs
        if config["target-os"] in ("ubuntu", "almalinux")
    ]
    kind = os.environ.get("KIND") or "all"
    kinds = BRANCHES if kind == "all" else {kind: BRANCHES[kind]}
    tips = {kind: github_json(f"repos/{FORK}/branches/{branch}")["commit"] for kind, branch in kinds.items()}
    repo = os.environ["GITHUB_REPOSITORY"]
    _, assets = release_assets(repo, "meta-llvm")
    requests = branch_requests(tips, assets, platforms)
    print(json.dumps(requests))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("requests=" + json.dumps(requests) + "\n")


if __name__ == "__main__":
    main()
