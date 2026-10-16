"""Decide whether cmake/meta-llvm-info.json should move to a newer fork release (pure, unit-tested).

The Meta LLVM fork publishes immutable releases named llvm-<sha8>-<build> whose
assets are llvm-<sha8>-<suffix>-<build>.tar.gz plus a .sha256 file each. A
target branch tracks one fork branch; its pin moves to the release built from
that branch's tip once every required platform is published.
"""
from __future__ import annotations

import re
import textwrap

_TAG = re.compile(r"^llvm-([0-9a-f]{8})-([0-9]+)$")


def oai_pins(llvm_info, llvm_build_info):
    """OAI core revisions a fork build may sit on.

    Which manifest is authoritative differs per branch: on main llvm-build-info.json
    leads llvm-info.json until OAI publishes; on release/3.8.x llvm-build-info.json is
    a stale leftover and llvm-info.json is the pin. Each branch's build_helpers.py
    enforces its own rule, so accept either and flag only a pin matching neither.
    """
    return {info["llvm_hash"] for info in (llvm_info, llvm_build_info) if info}


def same_tree_chain(commits):
    """Leading commits that share the tip's tree, i.e. the tip plus any empty sync commits under it.

    commits: [(sha, tree), ...] newest first along first parents.
    """
    if not commits:
        return []
    tip_tree = commits[0][1]
    chain = []
    for sha, tree in commits:
        if tree != tip_tree:
            break
        chain.append(sha)
    return chain


def release_for(chain, releases):
    """Newest build of a release whose commit is in chain, or None.

    releases: GitHub release objects (tag_name, target_commitish, assets).
    """
    best = None
    for sha in chain:
        for release in releases:
            match = _TAG.match(release.get("tag_name", ""))
            if not match or release.get("target_commitish") != sha or not sha.startswith(match.group(1)):
                continue
            if best is None or int(match.group(2)) > best[1]:
                best = (release, int(match.group(2)))
        if best:
            return best[0]
    return None


def variants(release):
    """{suffix: (tarball_asset, sha256_asset)} for every platform with both files."""
    match = _TAG.match(release["tag_name"])
    short, build = match.group(1), match.group(2)
    names = {asset["name"]: asset for asset in release.get("assets", [])}
    found = {}
    pattern = re.compile(rf"^llvm-{short}-(.+)-{build}\.tar\.gz$")
    for name, asset in names.items():
        suffix = pattern.match(name)
        if suffix and f"{name}.sha256" in names:
            found[suffix.group(1)] = (asset, names[f"{name}.sha256"])
    return found


def parse_sha256_file(text, tarball_name):
    """Digest from a sha256sum-format line naming tarball_name."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == tarball_name and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            return parts[0]
    raise ValueError(f"no sha256 for {tarball_name}")


def render(meta, release, sha256sums, repository):
    """New meta-llvm-info.json content for release, keeping existing key order and unrelated keys."""
    match = _TAG.match(release["tag_name"])
    updated = dict(meta)
    updated["llvm_hash"] = release["target_commitish"]
    updated["build_number"] = int(match.group(2))
    updated["release_base_url"] = f"https://github.com/{repository}/releases/download/{release['tag_name']}"
    updated["sha256sum"] = dict(sorted(sha256sums.items()))
    return updated


def decide(meta, pins, chain, release, required):
    """One of ("pin-moved"|"no-release"|"incomplete"|"current"|"bump", detail)."""
    if meta["based_on_oai_llvm_hash"] not in pins:
        return "pin-moved", (f"meta-llvm-info.json is based on {meta['based_on_oai_llvm_hash'][:12]} but OAI pins "
                             f"{', '.join(sorted(p[:12] for p in pins))}")
    if release is None:
        return "no-release", f"no release yet for fork tip {chain[0][:12]}"
    missing = sorted(set(required) - set(variants(release)))
    if missing:
        return "incomplete", f"{release['tag_name']} lacks {', '.join(missing)}"
    if meta["llvm_hash"] == release["target_commitish"] and set(meta["sha256sum"]) >= set(variants(release)):
        return "current", f"already pins {release['tag_name']}"
    return "bump", release["tag_name"]


def _bump_text(repo, target, fork, fork_branch, meta, new):
    """(problem, solution, test plan, log header) prose shared by the commit message and the PR body."""
    old_tag = meta["release_base_url"].rsplit("/", 1)[-1]
    tip, base = new["llvm_hash"], new["based_on_oai_llvm_hash"]
    tag = new["release_base_url"].rsplit("/", 1)[-1]
    return (f"{repo}@{target} tracks the Meta LLVM fork branch {fork}@{fork_branch}. It pins {old_tag}, but the "
            "fork branch has a newer published release.",
            f"This points cmake/meta-llvm-info.json at {tag}, built from fork tip {tip[:12]} on the same OAI pin "
            f"{base[:12]}.", "The pinned digests match the release's .sha256 assets and GitHub's asset digests "
            f"({', '.join(sorted(new['sha256sum']))}).",
            f"Fork changes in this bump ({meta['llvm_hash'][:12]}...{tip[:12]}):")


def commit_message(title, repo, target, fork, fork_branch, meta, new, log, width=72):
    """Commit message: bare Summary:/Test Plan: labels, wrapped at width for git log."""
    problem, solution, test, log_header = _bump_text(repo, target, fork, fork_branch, meta, new)

    def para(text):
        return textwrap.fill(text, width)

    def entry(sha, subject):
        line = f"  {sha[:12]} {subject}"
        return line if len(line) <= width else line[:width - 3] + "..."

    lines = [
        title, "", "Summary:", "**Problem**", "",
        para(problem), "", "**Solution**", "",
        para(solution), "", "Test Plan:",
        para(test), "", log_header
    ]
    lines += [entry(sha, subject) for sha, subject in log] or ["  (none)"]
    return "\n".join(lines)


def pr_body(repo, target, fork, fork_branch, meta, new, log):
    """PR description: markdown headings, unwrapped paragraphs (GitHub turns each newline into a break), and the
    fork log in a fence with full subjects."""
    problem, solution, test, log_header = _bump_text(repo, target, fork, fork_branch, meta, new)
    entries = [f"{sha[:12]} {subject}" for sha, subject in log] or ["(none)"]
    return "\n".join([
        "# Summary:", "", "**Problem**", "", problem, "", "**Solution**", "", solution, "", "# Test Plan:", "", test,
        "", log_header, "", "```", *entries, "```"
    ])


def issue_marker(target):
    """Hidden tag that identifies a target's tracking issue."""
    return f"<!-- meta-llvm-bump {target} -->"


def pick_tracking(issues, target):
    """(newest open tracking issue for target or None, older duplicates newest first).

    issues: [{"number": int, "body": str, ...}, ...] open issues in any order.
    """
    matches = sorted((i for i in issues if issue_marker(target) in (i.get("body") or "")), key=lambda i: i["number"],
                     reverse=True)
    return (matches[0] if matches else None), matches[1:]


def issue_release(body):
    """Release tag a tracking issue proposes, read from its Release: line, or None."""
    match = re.search(r"^Release: `(llvm-[0-9a-f]{8}-[0-9]+)`", body or "", re.MULTILINE)
    return match.group(1) if match else None


BAD_CONCLUSIONS = ("failure", "timed_out", "cancelled", "startup_failure")


def bot_branch_for(target):
    """Branch the bot pushes a target's bump to. scratch/** is excluded from the "Meta CLA Enforcement"
    ruleset, whose required "Meta CLA Check" cannot exist on a commit before its first push."""
    return "scratch/meta-llvm-bump/" + target.replace("/", "-")


def result_marker(kind, sha):
    """Hidden tag recording which commit an issue reports on and how: ready, failed or timeout."""
    return f"<!-- meta-llvm-bump-result {kind} {sha} -->"


def issue_result(body):
    """(kind, sha) from an issue's result marker, or (None, None)."""
    match = re.search(r"<!-- meta-llvm-bump-result (ready|failed|timeout) ([0-9a-f]{40}) -->", body or "")
    return (match.group(1), match.group(2)) if match else (None, None)


def verify_state(runs, workflows, age_hours, timeout_hours):
    """("running"|"timeout"|"done", {workflow file: latest run}) for one commit's push-triggered runs.

    runs: GitHub workflow run objects for the commit; workflows: the files that must run on it. A file with
    no run yet counts as running, so a missing push trigger surfaces as a timeout instead of a silent pass.
    """
    latest = {}
    for run in runs:
        name = run["path"].split("@")[0].rsplit("/", 1)[-1]
        if name in workflows and (name not in latest or run["id"] > latest[name]["id"]):
            latest[name] = run
    if all(name in latest and latest[name]["status"] == "completed" for name in workflows):
        return "done", latest
    return ("timeout" if age_hours >= timeout_hours else "running"), latest


def verdict(latest):
    """"ready" when every required run concluded success, else "failed"."""
    return "ready" if all(run["conclusion"] == "success" for run in latest.values()) else "failed"


def _date(release):
    return release["published_at"][:10]


def _release_line(release):
    return (f"Release: `{release['tag_name']}`, built from fork commit `{release['target_commitish'][:12]}`, "
            f"published {release['published_at'].replace('T', ' ').rstrip('Z')} UTC.")


def _runs_line(latest):
    return ", ".join(f"[{name}]({run['html_url']})" for name, run in sorted(latest.items())) or "none started"


def issue_title(target, release, kind="ready"):
    """Tracking issue title; names the outcome, the release and the day it was published."""
    outcome = {"ready": "Bump ready", "failed": "Bump failed verify", "timeout": "Bump failed verify (timed out)"}
    return f"[Meta LLVM] {outcome[kind]}: {target} to {release['tag_name']} ({_date(release)})"


def issue_body(repo, target, title, body, release, sha, latest):
    """Ready issue: the GPU result and the command a person runs to open the PR.

    A person opens the PR because the Meta CLA check and the fbsource importer accept only
    PR authors with a CLA on file; the bot account has none.
    """
    bot_branch = bot_branch_for(target)
    gh_cmd = (f"gh pr create -R {repo} --base {target} --head {bot_branch} \\\n"
              f"  --title '{title}' --body-file - <<'PR_BODY'\n{body}\nPR_BODY")
    lines = [
        issue_marker(target),
        result_marker("ready", sha),
        f"A Meta LLVM bump for `{target}` passed GPU verification on `{bot_branch}` at `{sha[:12]}` "
        f"([view the change](https://github.com/{repo}/compare/{target}...{bot_branch})).",
        "",
        _release_line(release),
        "",
        f"GPU runs: {_runs_line(latest)}.",
    ]
    return "\n".join(lines + [
        "",
        "Open its PR (a person must open it, since the Meta CLA check reads the PR author), then import and "
        "land it as usual. The PR skips the GPU suites, since this commit already passed them:",
        "",
        "````",  # Four backticks: the PR body inside has its own fences.
        gh_cmd,
        "````",
        "",
        "The bot leaves the branch alone from here; a newer fork release replaces this bump only while no PR is "
        f"open. This issue closes when the PR opens or `{target}` pins the release.",
    ])


def failed_body(repo, target, release, sha, latest, failed, timeout_hours=None):
    """Failed issue: what failed or is stuck, and how to retry. No PR command."""
    bot_branch = bot_branch_for(target)
    kind = "timeout" if timeout_hours else "failed"
    lines = [issue_marker(target), result_marker(kind, sha)]
    if timeout_hours:
        lines.append(f"GPU verification of the `{target}` bump on `{bot_branch}` at `{sha[:12]}` is still queued "
                     f"or running after {timeout_hours}h.")
    else:
        lines.append(f"GPU verification of the `{target}` bump on `{bot_branch}` at `{sha[:12]}` failed.")
    lines += ["", _release_line(release), "", f"GPU runs: {_runs_line(latest)}."]
    if failed:
        lines.append(f"Failed jobs: {', '.join(f'`{job}`' for job in failed)}.")
    lines.append("")
    if timeout_hours:
        lines.append("The bot holds newer releases for this line until these runs end. Cancel or re-run them by "
                     "hand if they are stuck; if no run started, check the line's push trigger.")
    else:
        ids = " ".join(str(run["id"]) for run in latest.values() if run["conclusion"] in BAD_CONCLUSIONS)
        lines.append(f"Re-run the failed jobs with `gh run rerun --failed -R {repo} <id>` (runs: {ids or 'none'}). "
                     "If they pass, the bot replaces this issue with a ready one. A newer fork release also "
                     "replaces this bump.")
    return "\n".join(lines)
