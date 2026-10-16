#!/usr/bin/env python3
"""Bump a branch's cmake/meta-llvm-info.json to the fork release built from its tracked fork branch.

Reads everything through the GitHub API, so --dry-run works from any machine.
Without --dry-run it must run in a checkout of the target branch: it commits
the new file to the bot branch and force-pushes it. If a person already opened
the branch's PR it updates that PR; otherwise it opens or updates a tracking
issue with the command to open it (the Meta CLA check rejects bot-authored PRs).

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
from meta_llvm_bump import (commit_message, decide, issue_body, issue_title, oai_pins, parse_sha256_file,  # noqa: E402
                            pick_tracking, pr_body, release_for, render, same_tree_chain, variants)

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


def bot_branch_for(target):
    # scratch/** is excluded from the "Meta CLA Enforcement" ruleset, whose required
    # "Meta CLA Check" cannot exist on a commit before its first push.
    return "scratch/meta-llvm-bump/" + target.replace("/", "-")


def tracking_issue(repo, target):
    """The open tracking issue for target, or None; closes older duplicates of it.

    Lists through GraphQL (gh issue list): with the app token, the REST issues list
    fills its page with pull requests and returns no issues.
    """
    issues = json.loads(
        run("gh", "issue", "list", "--repo", repo, "--state", "open", "--limit", "500", "--json", "number,body,url"))
    issue, duplicates = pick_tracking(issues, target)
    for dup in duplicates:
        run("gh", "issue", "close", str(dup["number"]), "--repo", repo, "--reason", "not planned", "--comment",
            f"Duplicate of #{issue['number']}.")
    return issue


def close_tracking_issue(repo, target, reason):
    issue = tracking_issue(repo, target)
    if issue:
        run("gh", "issue", "close", str(issue["number"]), "--repo", repo, "--comment", reason)


def publish(repo, target, title, message, body, release):
    """Push the bot branch, then update its PR if a person opened one, else the tracking issue."""
    bot_branch = bot_branch_for(target)
    run("git", "checkout", "-B", bot_branch)
    run("git", "add", META_PATH)
    run("git", "commit", "-m", message)
    try:
        remote_tree = gh_json(f"repos/{repo}/branches/{bot_branch}")["commit"]["commit"]["tree"]["sha"]
    except RuntimeError:
        remote_tree = None
    changed = remote_tree != run("git", "rev-parse", "HEAD^{tree}").strip()
    if changed:
        run("git", "push", "--force", "origin", f"HEAD:refs/heads/{bot_branch}", capture=False)
    # Same tree: same target tip and same pin, so the last build check still applies.
    emit(changed=str(changed).lower(), sha=run("git", "rev-parse", "HEAD").strip())
    prs = gh_json(f"repos/{repo}/pulls?head={repo.split('/')[0]}:{bot_branch}&base={target}&state=open")
    if prs:
        run("gh", "pr", "edit", str(prs[0]["number"]), "--repo", repo, "--title", title, "--body", body)
        return prs[0]["html_url"]
    issue_text = issue_body(repo, target, bot_branch, title, body, release)
    issue = tracking_issue(repo, target)
    if issue:
        moved = release["tag_name"] not in issue["body"]
        run("gh", "issue", "edit", str(issue["number"]), "--repo", repo, "--title", issue_title(target, release),
            "--body", issue_text)
        if moved:  # Edits notify no one; a comment tells subscribers the bump moved.
            run(
                "gh", "issue", "comment", str(issue["number"]), "--repo", repo, "--body",
                f"Moved `{bot_branch}` to `{release['tag_name']}` (fork commit "
                f"`{release['target_commitish'][:12]}`, published {release['published_at'][:10]}).")
        return issue["url"]
    return run("gh", "issue", "create", "--repo", repo, "--title", issue_title(target, release), "--body",
               issue_text).strip()


def report(repo, target, result, run_url):
    """Comment the build check's outcome on the bot branch's PR, else on the tracking issue."""
    bot_branch = bot_branch_for(target)
    sha = gh_json(f"repos/{repo}/branches/{bot_branch}")["commit"]["sha"]
    verdict = {"success": "passed", "failure": "failed", "cancelled": "was cancelled"}.get(result, f"ended: {result}")
    comment = (f"Build check for `{sha[:12]}` {verdict}: cp312 wheels (x86_64, aarch64) built by `wheels_fb.yml` "
               f"against the new pin. Run: {run_url}")
    prs = gh_json(f"repos/{repo}/pulls?head={repo.split('/')[0]}:{bot_branch}&base={target}&state=open")
    if prs:
        run("gh", "pr", "comment", str(prs[0]["number"]), "--repo", repo, "--body", comment)
        return
    issue = tracking_issue(repo, target)
    if issue:
        run("gh", "issue", "comment", str(issue["number"]), "--repo", repo, "--body", comment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton"))
    parser.add_argument("--target", required=True, help="Triton branch to bump, e.g. release/3.8.x")
    parser.add_argument("--fork", default="facebookexperimental/llvm-project")
    parser.add_argument("--fork-branch", help="Fork branch the target tracks, e.g. triton_3.8.x")
    parser.add_argument("--require", default=DEFAULT_REQUIRED, help="Platforms a release must publish")
    parser.add_argument("--dry-run", action="store_true", help="Print the new file and PR body; push nothing")
    parser.add_argument("--report", metavar="RESULT", help="Comment a build check result (success|failure|...)")
    parser.add_argument("--run-url", help="Build check run URL, with --report")
    args = parser.parse_args()
    if args.report:
        report(args.repo, args.target, args.report, args.run_url)
        return
    if not args.fork_branch:
        parser.error("--fork-branch is required unless --report is given")

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
    if status == "current" and not args.dry_run:
        close_tracking_issue(args.repo, args.target, f"{args.target} now pins this release ({detail}).")
    if status != "bump":
        return

    if not descends_from(args.fork, meta["based_on_oai_llvm_hash"], release["target_commitish"]):
        sys.exit(f"{release['tag_name']} does not descend from OAI pin {meta['based_on_oai_llvm_hash'][:12]}")
    new = render(meta, release, sha256sums(release), args.fork)
    title = f"[Meta LLVM] Bump {args.target} to {release['tag_name']}"
    log = fork_log(args.fork, meta["llvm_hash"], new["llvm_hash"])
    message = commit_message(title, args.repo, args.target, args.fork, args.fork_branch, meta, new, log)
    body = pr_body(args.repo, args.target, args.fork, args.fork_branch, meta, new, log)
    content = json.dumps(new, indent=2) + "\n"
    if args.dry_run:
        print(f"--- {META_PATH} ---\n{content}--- commit message ---\n{message}\n"
              f"--- tracking issue: {issue_title(args.target, release)} ---\n"
              f"{issue_body(args.repo, args.target, bot_branch_for(args.target), title, body, release)}")
        return
    remote_tip = gh_json(f"repos/{args.repo}/branches/{args.target}")["commit"]["sha"]
    if run("git", "rev-parse", "HEAD").strip() != remote_tip:
        sys.exit(f"checkout is not at {args.target} tip {remote_tip[:12]}; the pin was read from there")
    with open(META_PATH, "w") as f:
        f.write(content)
    print(publish(args.repo, args.target, title, message, body, release))


if __name__ == "__main__":
    main()
