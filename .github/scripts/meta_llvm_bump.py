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


def commit_message(title, repo, target, fork, fork_branch, meta, new, log, width=72):
    """Commit message and PR description: bare Summary:/Test Plan: labels, wrapped at width."""
    old_tag = meta["release_base_url"].rsplit("/", 1)[-1]
    tip, base = new["llvm_hash"], new["based_on_oai_llvm_hash"]
    tag = new["release_base_url"].rsplit("/", 1)[-1]

    def para(text):
        return textwrap.fill(text, width)

    def entry(sha, subject):
        line = f"  {sha[:12]} {subject}"
        return line if len(line) <= width else line[:width - 3] + "..."

    lines = [
        title,
        "",
        "Summary:",
        "**Problem**",
        "",
        para(f"{repo}@{target} tracks the Meta LLVM fork branch {fork}@{fork_branch}. It pins {old_tag}, but the "
             "fork branch has a newer published release."),
        "",
        "**Solution**",
        "",
        para(f"This points cmake/meta-llvm-info.json at {tag}, built from fork tip {tip[:12]} on the same OAI pin "
             f"{base[:12]}."),
        "",
        "Test Plan:",
        para("The pinned digests match the release's .sha256 assets and GitHub's asset digests "
             f"({', '.join(sorted(new['sha256sum']))})."),
        "",
        f"Fork changes in this bump ({meta['llvm_hash'][:12]}...{tip[:12]}):",
    ]
    lines += [entry(sha, subject) for sha, subject in log] or ["  (none)"]
    return "\n".join(lines)


def issue_marker(target):
    """Hidden tag that identifies a target's tracking issue."""
    return f"<!-- meta-llvm-bump {target} -->"


def issue_body(repo, target, bot_branch, title, message):
    """Tracking issue: where the bump is and the command a person runs to open its PR.

    A person opens the PR because the Meta CLA check and the fbsource importer accept only
    PR authors with a CLA on file; the bot account has none.
    """
    gh_cmd = (f'gh pr create -R {repo} --base {target} --head {bot_branch} \\\n'
              f'  --title "{title}" \\\n'
              f'  --body "$(gh api repos/{repo}/commits/{bot_branch} -q .commit.message | tail -n +3)"')
    return "\n".join([
        issue_marker(target),
        f"A Meta LLVM bump for `{target}` is ready on `{bot_branch}` "
        f"([view the change](https://github.com/{repo}/compare/{target}...{bot_branch})).",
        "",
        "Open its PR (a person must open it, since the Meta CLA check reads the PR author), then import and "
        "land it as usual:",
        "",
        "```",
        gh_cmd,
        "```",
        "",
        f"The bot keeps the branch, and the PR once it exists, up to date. This issue closes when `{target}` pins "
        "the release.",
        "",
        "Commit message and PR description:",
        "",
        "```",
        message,
        "```",
    ])
