#!/usr/bin/env python3
"""Write one platform's section into the PR's single perf smoke comment.

  tlx_ops_perf_smoke_comment.py --repo OWNER/REPO --pr N --platform P --head SHA \
      --section SECTION_MD [--note TEXT]

Each GPU's report job calls this when its own runner finishes, so a fast
platform's numbers show up without waiting for a slow or queued one. The
comment keeps one marked section per platform; this rewrites only PLATFORM's
and leaves the others, noting any that came from an older head.

Two report jobs can finish at the same moment, read the same body, and the
second write would drop the first one's section. So after writing, wait and
read again, and re-apply this platform's section if it is missing; each job
only ever has to restore its own.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

MARKER = "<!-- tlx-ops-perf-smoke -->"
BOT = "github-actions[bot]"
#: Section order in the comment; a platform not listed sorts after these.
ORDER = ("b200", "h100", "mi350")
SECTION = re.compile(
    r"<!-- tlx-ops-perf-smoke:(?P<platform>[a-z0-9]+) head=(?P<head>[0-9a-f]+)(?: pr_head=(?P<pr_head>[0-9a-f]+))? -->\n"
    r"(?P<body>.*?)\n<!-- /tlx-ops-perf-smoke:(?P=platform) -->", re.S)
#: A section whose title starts with this has no measurements (the bench produced nothing).
NO_RESULT = "### ⚪"
INTRO = ("Checks whether this PR makes tlx.ops kernels slower or breaks them: the first 8 shapes of matmul, "
         "attention and norm, forward only. It runs each time the PR's head passes CI. Each GPU times the kernels "
         "for the PR's base and for the PR, on the same machine, and fills in its own section below when it "
         "finishes. Advisory only: it never blocks the PR, and the full benchmark stays in the nightly run.")


def parse_sections(body):
    """{platform: (head, text)} from a comment body; empty for a legacy or foreign body.

    HEAD is "<benched commit>" or "<benched commit> <PR head>" when the two differ.
    """
    if not body or not body.startswith(MARKER):
        return {}
    return {
        m["platform"]: (f"{m['head']} {m['pr_head']}" if m["pr_head"] else m["head"], m["body"])
        for m in SECTION.finditer(body)
    }


def benched(head):
    """The benched commit of a section HEAD ("<commit>" or "<commit> <PR head>")."""
    return head.split()[0]


def measured_head(body):
    """The benched commit every measured section shares, or '' when they differ or none measured."""
    heads = {benched(head) for head, text in parse_sections(body).values() if not text.startswith(NO_RESULT)}
    return heads.pop() if len(heads) == 1 else ""


def marker(platform, head):
    commit, _, pr_head = head.partition(" ")
    return f"<!-- tlx-ops-perf-smoke:{platform} head={commit}{f' pr_head={pr_head}' if pr_head else ''} -->"


def order_key(platform):
    return (ORDER.index(platform) if platform in ORDER else len(ORDER), platform)


def compose(sections, head, note=""):
    lines = [MARKER, "## TLX.ops perf smoke (advisory)", "", INTRO]
    if note:
        lines += ["", note]
    stale = [p for p in sorted(sections, key=order_key) if benched(sections[p][0]) != benched(head)]
    if stale:
        lines += [
            "", "From an older head, not yet re-run for this one: " +
            ", ".join(f"{p} (`{benched(sections[p][0])[:12]}`)" for p in stale) + "."
        ]
    for platform in sorted(sections, key=order_key):
        section_head, text = sections[platform]
        lines += ["", marker(platform, section_head), text.rstrip("\n"), f"<!-- /tlx-ops-perf-smoke:{platform} -->"]
    return "\n".join(lines) + "\n"


def set_note(body, head, note):
    """The comment with only its note replaced, sections kept; for a head that is not re-run."""
    return compose(parse_sections(body), head, note)


def merge(body, platform, head, text, note=""):
    sections = parse_sections(body)
    sections[platform] = (head, text.rstrip("\n"))
    return compose(sections, head, note)


def has_section(body, platform, head, text):
    return parse_sections(body).get(platform) == (head, text.rstrip("\n"))


class GitHub:

    def __init__(self, repo, token, api=None):
        self.repo, self.token = repo, token
        self.api = (api or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")

    def call(self, method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f"{self.api}{path}", data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read() or b"null")

    def find_comment(self, pr):
        """The newest bot comment carrying the marker, or None."""
        found, page = None, 1
        while True:
            batch = self.call("GET", f"/repos/{self.repo}/issues/{pr}/comments?per_page=100&page={page}")
            for comment in batch:
                if (comment.get("user") or {}).get("login") == BOT and (comment.get("body") or "").startswith(MARKER):
                    found = comment
            if len(batch) < 100:
                return found
            page += 1

    def write(self, pr, comment, body):
        if comment:
            return self.call("PATCH", f"/repos/{self.repo}/issues/comments/{comment['id']}", {"body": body})
        return self.call("POST", f"/repos/{self.repo}/issues/{pr}/comments", {"body": body})


def publish(gh, pr, platform, head, text, note="", attempts=4, settle=10.0, sleep=time.sleep):
    """Write PLATFORM's section, then re-read and restore it until it survives a settle period."""
    for _ in range(attempts):
        comment = gh.find_comment(pr)
        body = (comment or {}).get("body") or ""
        if not has_section(body, platform, head, text):
            comment = gh.write(pr, comment, merge(body, platform, head, text, note))
        sleep(settle)
        current = gh.find_comment(pr)
        if current and has_section(current.get("body") or "", platform, head, text):
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--platform", help="the section to write (with --section)")
    parser.add_argument("--head", help="the benched commit; sections from other heads are marked")
    parser.add_argument("--pr-head", default="", help="the PR head, when the benched commit is its test merge")
    parser.add_argument("--section", help="markdown file with this platform's section")
    parser.add_argument("--note", default="", help="line under the intro, e.g. which platforms are paused")
    parser.add_argument("--measured-head", action="store_true",
                        help="print the benched commit all measured sections share (blank if none) and exit")
    parser.add_argument("--note-only", action="store_true", help="replace only the note, keeping every section")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN is not set", file=sys.stderr)
        return 1
    gh = GitHub(args.repo, token)
    try:
        if args.measured_head:
            print(measured_head((gh.find_comment(args.pr) or {}).get("body") or ""))
            return 0
        if args.note_only:
            comment = gh.find_comment(args.pr)
            if comment:
                body = comment.get("body") or ""
                gh.write(args.pr, comment, set_note(body, measured_head(body) or args.head or "", args.note))
            return 0
        if not (args.platform and args.head and args.section):
            parser.error("--platform, --head and --section are required to write a section")
        with open(args.section) as fh:
            text = fh.read()
        head = f"{args.head} {args.pr_head}" if args.pr_head and args.pr_head != args.head else args.head
        ok = publish(gh, args.pr, args.platform, head, text, args.note)
    except urllib.error.HTTPError as exc:
        print(f"GitHub API error: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    print(f"{args.platform} section {'published' if ok else 'could not be confirmed after retries'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
