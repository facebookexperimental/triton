#!/usr/bin/env python3
"""Bump a branch's cmake/meta-llvm-info.json to the fork release built from its tracked fork branch.

Reads everything through the GitHub API, so --dry-run works from any machine.
Without --dry-run it must run in a checkout of the target branch: it commits
the new file to the bot branch, force-pushes it, and opens or updates one PR.

Emits status=no-meta|pin-moved|no-release|incomplete|current|bump to GITHUB_OUTPUT.
pin-moved exits 1: the fork has to be rebuilt on OAI's new pin by hand.
"""
import argparse
import base64
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meta_llvm_bump import (commit_message, decide, oai_pins, parse_sha256_file, release_for, render,  # noqa: E402
                            same_tree_chain, variants)

META_PATH = "cmake/meta-llvm-info.json"
DEFAULT_REQUIRED = "almalinux-arm64,almalinux-x64,ubuntu-arm64,ubuntu-x64"


def run(*args, capture=True):
    result = subprocess.run(args, capture_output=capture, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip() if capture else ''}")
    return result.stdout if capture else ""


def gh_json(path):
    return json.loads(run("gh", "api", path))


def gh_raw(url):
    return run("gh", "api", url, "-H", "Accept: application/octet-stream")


def read_json(repo, path, ref):
    try:
        data = gh_json(f"repos/{repo}/contents/{path}?ref={ref}")
    except RuntimeError:
        return None
    return json.loads(base64.b64decode(data["content"]))


def fork_chain(fork, branch):
    commits = gh_json(f"repos/{fork}/commits?sha={branch}&per_page=30")
    return same_tree_chain([(c["sha"], c["commit"]["tree"]["sha"]) for c in commits])


def sha256sums(release):
    sums = {}
    for suffix, (tarball, sha_asset) in variants(release).items():
        digest = parse_sha256_file(gh_raw(sha_asset["url"]), tarball["name"])
        published = tarball.get("digest")
        if published and published != f"sha256:{digest}":
            raise RuntimeError(f"{tarball['name']}: .sha256 says {digest}, GitHub says {published}")
        sums[suffix] = digest
    return sums


def descends_from(fork, base, sha):
    return gh_json(f"repos/{fork}/compare/{base}...{sha}")["status"] in ("ahead", "identical")


def fork_log(fork, old, new):
    try:
        commits = gh_json(f"repos/{fork}/compare/{old}...{new}")["commits"]
    except RuntimeError:
        return []
    # ShipIt's empty sync commits carry only an fbshipit-source-id line.
    subjects = [(c["sha"], c["commit"]["message"].splitlines()[0]) for c in commits]
    return [(sha, subject) for sha, subject in subjects if not subject.startswith("fbshipit-source-id:")]


def emit(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for key, value in kv.items():
                f.write(f"{key}={value}\n")


def open_pr(repo, target, title, message):
    # scratch/** is excluded from the "Meta CLA Enforcement" ruleset, whose required
    # "Meta CLA Check" cannot exist on a commit before its first push.
    bot_branch = "scratch/meta-llvm-bump/" + target.replace("/", "-")
    run("git", "checkout", "-B", bot_branch)
    run("git", "add", META_PATH)
    run("git", "commit", "-m", message)
    run("git", "push", "--force", "origin", f"HEAD:refs/heads/{bot_branch}", capture=False)
    body = message.split("\n", 2)[2]  # The PR title carries the first line.
    existing = gh_json(f"repos/{repo}/pulls?head={repo.split('/')[0]}:{bot_branch}&base={target}&state=open")
    if existing:
        number = existing[0]["number"]
        run("gh", "pr", "edit", str(number), "--repo", repo, "--title", title, "--body", body)
        return existing[0]["html_url"]
    return run("gh", "pr", "create", "--repo", repo, "--base", target, "--head", bot_branch, "--title", title, "--body",
               body).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton"))
    parser.add_argument("--target", required=True, help="Triton branch to bump, e.g. release/3.8.x")
    parser.add_argument("--fork", default="facebookexperimental/llvm-project")
    parser.add_argument("--fork-branch", required=True, help="Fork branch the target tracks, e.g. triton_3.8.x")
    parser.add_argument("--require", default=DEFAULT_REQUIRED, help="Platforms a release must publish")
    parser.add_argument("--dry-run", action="store_true", help="Print the new file and PR body; push nothing")
    args = parser.parse_args()

    meta = read_json(args.repo, META_PATH, args.target)
    if meta is None:
        print(f"{args.target} has no {META_PATH}; nothing to bump")
        emit(status="no-meta")
        return
    pins = oai_pins(read_json(args.repo, "cmake/llvm-info.json", args.target),
                    read_json(args.repo, "cmake/llvm-build-info.json", args.target))
    chain = fork_chain(args.fork, args.fork_branch)
    release = release_for(chain, gh_json(f"repos/{args.fork}/releases?per_page=100"))
    status, detail = decide(meta, pins, chain, release, [s for s in args.require.split(",") if s])
    print(f"{args.target} <- {args.fork_branch}: {status}: {detail}")
    emit(status=status)
    if status == "pin-moved":
        sys.exit(1)
    if status != "bump":
        return

    if not descends_from(args.fork, meta["based_on_oai_llvm_hash"], release["target_commitish"]):
        sys.exit(f"{release['tag_name']} does not descend from OAI pin {meta['based_on_oai_llvm_hash'][:12]}")
    new = render(meta, release, sha256sums(release), args.fork)
    title = f"[Meta LLVM] Bump {args.target} to {release['tag_name']}"
    log = fork_log(args.fork, meta["llvm_hash"], new["llvm_hash"])
    message = commit_message(title, args.repo, args.target, args.fork, args.fork_branch, meta, new, log)
    content = json.dumps(new, indent=2) + "\n"
    if args.dry_run:
        print(f"--- {META_PATH} ---\n{content}--- commit message and PR description ---\n{message}")
        return
    remote_tip = gh_json(f"repos/{args.repo}/branches/{args.target}")["commit"]["sha"]
    if run("git", "rev-parse", "HEAD").strip() != remote_tip:
        sys.exit(f"checkout is not at {args.target} tip {remote_tip[:12]}; the pin was read from there")
    with open(META_PATH, "w") as f:
        f.write(content)
    print(open_pr(args.repo, args.target, title, message))


if __name__ == "__main__":
    main()
