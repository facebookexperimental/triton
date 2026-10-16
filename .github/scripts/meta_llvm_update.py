#!/usr/bin/env python3
"""Bump a branch's cmake/meta-llvm-info.json to the fork release built from its tracked fork branch.

Each run, for one target branch:
  * pushes scratch/meta-llvm-bump/<target> when the fork has a newer complete release, which starts
    the target's GPU workflows on that commit (push trigger);
  * once those runs end, opens one issue for the bump: "Bump ready" with the command a person runs to
    open the PR (the Meta CLA check rejects bot-authored PRs), or "Bump failed verify";
  * files a timeout issue if the runs are still going after --timeout-hours, and holds newer
    releases until the current commit's runs end;
  * leaves an open PR alone, closing the issue in its favour;
  * closes the issue once the target pins its release.
It runs hourly and when a bump branch's GPU run completes (meta-llvm-verify-done.yml), and is the only
writer of these issues.

Reads everything through the GitHub API, so --dry-run works from any machine and prints the writes it
would make. Without --dry-run it must run in a checkout of the target branch, to commit the bump.
pin-moved exits 1: the fork has to be rebuilt on OAI's new pin by hand.
"""
import argparse
import base64
import datetime
import json
import os
import subprocess
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meta_llvm_bump import (BAD_CONCLUSIONS, bot_branch_for, commit_message, decide, failed_body,  # noqa: E402
                            issue_body, issue_release, issue_result, issue_title, oai_pins, parse_sha256_file,
                            pick_tracking, pr_body, release_for, render, same_tree_chain, variants, verdict,
                            verify_state)

META_PATH = "cmake/meta-llvm-info.json"
DEFAULT_REQUIRED = "almalinux-arm64,almalinux-x64,ubuntu-arm64,ubuntu-x64"

DRY_RUN = False


def run(*args, capture=True):
    result = subprocess.run(args, capture_output=capture, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip() if capture else ''}")
    return result.stdout if capture else ""


def write(*args, capture=True):
    """run() for commands that change GitHub state; --dry-run prints them instead."""
    if DRY_RUN:
        print("would run:", " ".join(a if len(a) < 120 else a[:117] + "..." for a in args))
        return ""
    return run(*args, capture=capture)


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


def tracking_issue(repo, target):
    """The open tracking issue for target, or None; closes older duplicates of it.

    Lists through GraphQL (gh issue list): with the app token, the REST issues list
    fills its page with pull requests and returns no issues.
    """
    issues = json.loads(
        run("gh", "issue", "list", "--repo", repo, "--state", "open", "--limit", "500", "--json", "number,body,url"))
    issue, duplicates = pick_tracking(issues, target)
    for dup in duplicates:
        write("gh", "issue", "close", str(dup["number"]), "--repo", repo, "--reason", "not planned", "--comment",
              f"Duplicate of #{issue['number']}.")
    return issue


def tag_of(meta):
    return meta["release_base_url"].rsplit("/", 1)[-1]


def close_issue(repo, issue, reason):
    write("gh", "issue", "close", str(issue["number"]), "--repo", repo, "--comment", reason)


def open_pr(repo, target):
    prs = gh_json(f"repos/{repo}/pulls?head={repo.split('/')[0]}:{bot_branch_for(target)}&base={target}&state=open")
    return prs[0] if prs else None


def leave_pr_alone(repo, pr, issue, newer):
    """A person opened the bump PR: it is the tracker now, and the bot never rewrites it."""
    if issue:
        close_issue(repo, issue, f"Tracked in #{pr['number']}.")
    if newer and newer not in pr["title"]:
        marker = f"<!-- meta-llvm-bump-next {newer} -->"
        comments = gh_json(f"repos/{repo}/issues/{pr['number']}/comments?per_page=100")
        if not any(marker in (c.get("body") or "") for c in comments):
            write(
                "gh", "pr", "comment", str(pr["number"]), "--repo", repo, "--body",
                f"{marker}\nThe fork published `{newer}`. The bot leaves this PR as it is; `{newer}` "
                "follows after this lands.")


def bump_branch(repo, target):
    """{"sha", "age_hours", "meta"} for the bump branch's tip, or None if it does not exist."""
    branch = bot_branch_for(target)
    try:
        commit = gh_json(f"repos/{repo}/branches/{urllib.parse.quote(branch, safe='')}")["commit"]
    except RuntimeError:
        return None
    pushed = datetime.datetime.fromisoformat(commit["commit"]["committer"]["date"].replace("Z", "+00:00"))
    age = (datetime.datetime.now(datetime.timezone.utc) - pushed).total_seconds() / 3600
    return {"sha": commit["sha"], "age_hours": age, "meta": read_json(repo, META_PATH, commit["sha"])}


def failed_jobs(repo, latest):
    names = []
    for name, run_obj in sorted(latest.items()):
        if run_obj["conclusion"] in BAD_CONCLUSIONS:
            jobs = gh_json(f"repos/{repo}/actions/runs/{run_obj['id']}/jobs?per_page=100")["jobs"]
            names += [job["name"] for job in jobs if job["conclusion"] in BAD_CONCLUSIONS]
    return names


def pr_text(repo, target, fork, fork_branch, meta, new):
    """(title, body) of the PR that bumps target from meta to new."""
    title = f"[Meta LLVM] Bump {target} to {tag_of(new)}"
    return title, pr_body(repo, target, fork, fork_branch, meta, new, fork_log(fork, meta["llvm_hash"],
                                                                               new["llvm_hash"]))


def file_result(repo, issue, title, body):
    """Make the line's one open issue say title/body. Returns True if anything was written."""
    if issue and issue_result(issue["body"]) == issue_result(body):
        return False
    if issue and issue_result(issue["body"])[1] == issue_result(body)[1]:
        # Same commit, new outcome (e.g. a timed-out run finished): edit, then comment, since edits notify no one.
        write("gh", "issue", "edit", str(issue["number"]), "--repo", repo, "--title", title, "--body", body)
        write("gh", "issue", "comment", str(issue["number"]), "--repo", repo, "--body", f"Updated: {title}")
        return True
    if issue:
        close_issue(repo, issue, "Superseded by a newer bump commit.")
    print(write("gh", "issue", "create", "--repo", repo, "--title", title, "--body", body).strip())
    return True


def settle(args, meta, branch, issue):
    """Turn the bump branch tip's GPU runs into the line's issue.

    Returns "running" (runs still going, or timed out), "filed" (an issue was written this run) or "done".
    """
    sha, new = branch["sha"], branch["meta"]
    runs = gh_json(f"repos/{args.repo}/actions/runs?head_sha={sha}&event=push&per_page=100")["workflow_runs"]
    state, latest = verify_state(runs, args.verify_workflows, branch["age_hours"], args.timeout_hours)
    print(f"verify {sha[:12]} ({tag_of(new)}): {state}, {branch['age_hours']:.1f}h, " +
          (", ".join(f"{k}={v['status']}/{v['conclusion']}" for k, v in sorted(latest.items())) or "no runs"))
    if state == "running":
        return "running"
    release = gh_json(f"repos/{args.fork}/releases/tags/{tag_of(new)}")
    if state == "timeout":
        file_result(args.repo, issue, issue_title(args.target, release, "timeout"),
                    failed_body(args.repo, args.target, release, sha, latest, [], args.timeout_hours))
        return "running"
    if verdict(latest) == "ready":
        title, body = pr_text(args.repo, args.target, args.fork, args.fork_branch, meta, new)
        text = issue_body(args.repo, args.target, title, body, release, sha, latest)
        wrote = file_result(args.repo, issue, issue_title(args.target, release), text)
    else:
        text = failed_body(args.repo, args.target, release, sha, latest, failed_jobs(args.repo, latest))
        wrote = file_result(args.repo, issue, issue_title(args.target, release, "failed"), text)
    return "filed" if wrote else "done"


def push_bump(args, meta, release):
    """Commit the bump for release on the target tip and force-push the bump branch; opens no issue."""
    if not descends_from(args.fork, meta["based_on_oai_llvm_hash"], release["target_commitish"]):
        sys.exit(f"{release['tag_name']} does not descend from OAI pin {meta['based_on_oai_llvm_hash'][:12]}")
    new = render(meta, release, sha256sums(release), args.fork)
    title = f"[Meta LLVM] Bump {args.target} to {release['tag_name']}"
    log = fork_log(args.fork, meta["llvm_hash"], new["llvm_hash"])
    message = commit_message(title, args.repo, args.target, args.fork, args.fork_branch, meta, new, log)
    content = json.dumps(new, indent=2) + "\n"
    if DRY_RUN:
        print(f"would push {bot_branch_for(args.target)} with:\n--- {META_PATH} ---\n{content}"
              f"--- commit message ---\n{message}")
        return
    remote_tip = gh_json(f"repos/{args.repo}/branches/{urllib.parse.quote(args.target, safe='')}")["commit"]["sha"]
    if run("git", "rev-parse", "HEAD").strip() != remote_tip:
        sys.exit(f"checkout is not at {args.target} tip {remote_tip[:12]}; the pin was read from there")
    with open(META_PATH, "w") as f:
        f.write(content)
    run("git", "checkout", "-B", bot_branch_for(args.target))
    run("git", "add", META_PATH)
    run("git", "commit", "-m", message)
    run("git", "push", "--force", "origin", f"HEAD:refs/heads/{bot_branch_for(args.target)}", capture=False)
    print(f"pushed {bot_branch_for(args.target)} for {release['tag_name']}")


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton"))
    parser.add_argument("--target", required=True, help="Triton branch to bump, e.g. release/3.8.x")
    parser.add_argument("--fork", default="facebookexperimental/llvm-project")
    parser.add_argument("--fork-branch", required=True, help="Fork branch the target tracks, e.g. triton_3.8.x")
    parser.add_argument("--require", default=DEFAULT_REQUIRED, help="Platforms a release must publish")
    parser.add_argument(
        "--verify-workflows", default="", type=lambda v: [w for w in v.split(",") if w],
        help="Workflow files that must pass on the bump commit, e.g. gpu-tests.yml; "
        "none means the bump is ready as soon as it is pushed")
    parser.add_argument("--timeout-hours", type=float, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Print the writes instead of making them")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    meta = read_json(args.repo, META_PATH, args.target)
    if meta is None:
        print(f"{args.target} has no {META_PATH}; nothing to bump")
        return
    pins = oai_pins(read_json(args.repo, "cmake/llvm-info.json", args.target),
                    read_json(args.repo, "cmake/llvm-build-info.json", args.target))
    chain = fork_chain(args.fork, args.fork_branch)
    release = release_for(chain, gh_json(f"repos/{args.fork}/releases?per_page=100"))
    status, detail = decide(meta, pins, chain, release, [s for s in args.require.split(",") if s])
    print(f"{args.target} <- {args.fork_branch}: {status}: {detail}")
    if status == "pin-moved":
        sys.exit(1)
    pinned = tag_of(meta)
    newer = release["tag_name"] if status == "bump" else None
    issue = tracking_issue(args.repo, args.target)

    pr = open_pr(args.repo, args.target)
    if pr:
        leave_pr_alone(args.repo, pr, issue, newer)
        return
    if issue and (status == "current" or issue_release(issue["body"]) == pinned):
        close_issue(args.repo, issue, f"{args.target} now pins {pinned}.")
        issue = None

    branch = bump_branch(args.repo, args.target)
    # A bump the target has not landed yet: its pin moves past the target's.
    pending = (branch and branch["meta"] and branch["meta"]["llvm_hash"] != meta["llvm_hash"]
               and descends_from(args.fork, meta["llvm_hash"], branch["meta"]["llvm_hash"]))
    state = settle(args, meta, branch, issue) if pending else None
    if not newer or (pending and tag_of(branch["meta"]) == newer):
        return
    if state in ("running", "filed"):
        # Never cancel a bump: the current commit's runs end and get their issue before the next one goes out.
        print(f"holding {newer} until the current bump's issue is out")
        return
    issue = tracking_issue(args.repo, args.target)
    if issue:
        close_issue(args.repo, issue, f"Superseded by {newer}; the bot is verifying it.")
    push_bump(args, meta, release)


if __name__ == "__main__":
    main()
