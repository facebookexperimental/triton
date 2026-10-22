#!/usr/bin/env python3
"""Compute a ranked list of suspect PRs for a nightly regression.

Given the last-known-green nightly SHA and the failing SHA, this walks the commit
range ``good..bad``, extracts the squash-merge PR number from each commit subject
(``... (#1234)``), and ranks the PRs by how much their changed files overlap the
failing test's subsystem. The result is a Markdown block injected into the
auto-filed nightly issue as a cheap, immediate blame hint (the async git-bisect
job later confirms the single culprit).

The git-touching parts are isolated in thin wrappers so the ranking logic is unit
testable without a repo. ``main`` shells out to ``git``; the pure helpers accept
already-collected data.
"""

import argparse
import os
import re
import subprocess
import sys

# Squash-merge subjects end with the PR number, e.g. "Fix foo (#1972)".
_PR_RE = re.compile(r"\(#(\d+)\)\s*$")


def extract_pr(subject):
    """Return the trailing PR number in a commit subject, or None."""
    m = _PR_RE.search(subject or "")
    return int(m.group(1)) if m else None


def parse_commits(log_output):
    """Parse ``git log --format=%H%x09%s`` output into ordered commit dicts.

    Output order is preserved (newest first, as git emits it). Each dict has
    ``sha``, ``subject`` and ``pr`` (int or None).
    """
    commits = []
    for line in (log_output or "").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        sha, _, subject = line.partition("\t")
        commits.append({"sha": sha.strip(), "subject": subject.strip(), "pr": extract_pr(subject)})
    return commits


def test_subsystem_dir(normalized_failure_id):
    """Best-effort source directory for a failing test identity, or None.

    - pytest path id ``a/b/test_x.py::test_y``       -> ``a/b``
    - pytest dotted classname ``a.b.test_x::test_y`` -> ``a/b``  (JUnit classname form)
    - LIT id ``TRITON :: Conversion/foo.mlir``       -> ``test/Conversion``
    - job-level bucket (no path)                     -> None
    """
    nid = (normalized_failure_id or "").strip()
    if not nid or "::" not in nid:
        return None
    left, right = (s.strip() for s in nid.split("::", 1))
    # LIT: "TRITON :: <relpath-under-test/>"
    if left == "TRITON" or right.endswith((".mlir", ".ll", ".td")):
        rel_dir = os.path.dirname(right)
        return ("test/" + rel_dir).rstrip("/") if rel_dir else "test"
    # pytest, path form: "a/b/test_x.py"
    if left.endswith(".py"):
        return os.path.dirname(left) or None
    # pytest, dotted classname form: "a.b.test_x" -> "a/b/test_x.py" -> "a/b"
    if "." in left:
        return os.path.dirname(left.replace(".", "/") + ".py") or None
    return None


def test_location(normalized_failure_id):
    """``(path, function)`` of a failing test, or None for a job-level bucket.

    ``function`` is None for a LIT test, whose identity is the whole file.

    - pytest path id ``a/b/test_x.py::TestC::test_y``       -> ``("a/b/test_x.py", "test_y")``
    - pytest dotted classname ``a.b.test_x.TestC::test_y``  -> ``("a/b/test_x.py", "test_y")``
    - LIT id ``TRITON :: Conversion/foo.mlir``              -> ``("test/Conversion/foo.mlir", None)``
    """
    nid = (normalized_failure_id or "").strip()
    if "::" not in nid:
        return None
    left, right = (s.strip() for s in nid.split("::", 1))
    if left == "TRITON" or right.endswith((".mlir", ".ll", ".td")):
        return "test/" + right, None
    func = right.split("::")[-1].strip()
    if left.endswith(".py"):
        return left, func
    # The dotted classname may end in a test class; the module is the last test_* part.
    parts = left.split(".")
    modules = [i for i, part in enumerate(parts) if part.startswith("test_")]
    if not modules:
        return None
    return "/".join(parts[:modules[-1] + 1]) + ".py", func


def render_new_test(path, func, good_sha, commit):
    """Markdown note for a failing test that did not exist at the last green commit."""
    name = f"`{func}` in `{path}`" if func else f"`{path}`"
    if commit:
        added = f"it was added by `{commit['sha'][:9]}`" + (f" (#{commit['pr']})" if commit["pr"] else "")
    else:
        added = "it was added in this range"
    return f"**New test:** {name} did not exist at the last green commit `{good_sha[:9]}`; {added}."


def _overlap(path, test_dir):
    """Shared leading-directory-component count between a file and the test dir."""
    if not test_dir:
        return 0
    a = os.path.dirname(path).split("/")
    b = test_dir.split("/")
    n = 0
    for x, y in zip(a, b):
        if x != y or x == "":
            break
        n += 1
    return n


def rank_prs(commits, files_by_sha, test_dir):
    """Rank unique PRs by max file-path overlap with the failing test's dir.

    Ties (and the no-test-dir case) keep git order (most recent first). Returns a
    list of dicts: ``pr``, ``subject``, ``score``, ``top_path``.
    """
    best = {}
    order = []
    for c in commits:
        pr = c["pr"]
        if pr is None:
            continue
        files = files_by_sha.get(c["sha"], [])
        score, top = 0, ""
        for f in files:
            o = _overlap(f, test_dir)
            if o > score:
                score, top = o, f
        if pr not in best:
            best[pr] = {"pr": pr, "subject": c["subject"], "score": score, "top_path": top}
            order.append(pr)
        elif score > best[pr]["score"]:
            best[pr].update(score=score, top_path=top)
    ranked = [best[pr] for pr in order]
    # Stable sort by score desc; equal scores retain git (recency) order.
    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked


def render_markdown(ranked, good_sha, bad_sha, test_dir, limit=10):
    """Render the ranked suspect list as a Markdown table."""
    if not ranked:
        return ("_No merged PRs found in range "
                f"`{good_sha[:9]}..{bad_sha[:9]}` — suspect list unavailable._")
    lines = [
        f"**Suspected PRs** (range `{good_sha[:9]}..{bad_sha[:9]}`" +
        (f", ranked by overlap with `{test_dir}`" if test_dir else ", newest first") + "):",
        "",
        "| Rank | PR | Overlap | Title |",
        "| --- | --- | --- | --- |",
    ]
    for i, d in enumerate(ranked[:limit], 1):
        overlap = f"`{d['top_path']}`" if d["score"] > 0 else "—"
        subject = d["subject"].replace("|", "\\|")
        lines.append(f"| {i} | #{d['pr']} | {overlap} | {subject} |")
    if len(ranked) > limit:
        lines.append("")
        lines.append(f"_…and {len(ranked) - limit} more PR(s) in range._")
    return "\n".join(lines)


# --- git wrappers (not unit-tested; exercised in CI) ---
def _git(args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def collect_range(good_sha, bad_sha):
    commits = parse_commits(_git(["log", "--format=%H%x09%s", f"{good_sha}..{bad_sha}"]))
    files_by_sha = {}
    for c in commits:
        out = _git(["show", "--no-notes", "--pretty=format:", "--name-only", c["sha"]])
        files_by_sha[c["sha"]] = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return commits, files_by_sha


def test_exists(sha, path, func):
    """Whether the test file, and the test function if given, exist at ``sha``."""
    if subprocess.run(["git", "cat-file", "-e", f"{sha}:{path}"], capture_output=True, check=False).returncode:
        return False
    return not func or re.search(rf"^\s*(async\s+)?def {re.escape(func)}\(", _git(["show", f"{sha}:{path}"]),
                                 re.MULTILINE) is not None


def first_adding_commit(good_sha, bad_sha, path, func):
    """The oldest commit in ``good..bad`` that added the test, or None."""
    # git log -G takes a POSIX regex, where "\(" opens a group; "[(]" is a literal parenthesis.
    pick = ["-G", f"def {func}[(]"] if func else ["--diff-filter=A"]
    commits = parse_commits(
        _git(["log", "--reverse", "--format=%H%x09%s", *pick, f"{good_sha}..{bad_sha}", "--", f":(top){path}"]))
    return commits[0] if commits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--good-sha", required=True)
    ap.add_argument("--bad-sha", required=True)
    ap.add_argument("--normalized-id", default="")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    if not args.good_sha:
        print("_No prior green run found — suspect list unavailable._")
        return 0
    commits, files_by_sha = collect_range(args.good_sha, args.bad_sha)
    test_dir = test_subsystem_dir(args.normalized_id)
    ranked = rank_prs(commits, files_by_sha, test_dir)
    location = test_location(args.normalized_id)
    if location and test_exists(args.bad_sha, *location) and not test_exists(args.good_sha, *location):
        print(render_new_test(*location, args.good_sha, first_adding_commit(args.good_sha, args.bad_sha, *location)))
        print()
    print(render_markdown(ranked, args.good_sha, args.bad_sha, test_dir, args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
