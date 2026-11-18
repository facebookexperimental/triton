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
    r"<!-- tlx-ops-perf-smoke:(?P<platform>[a-z0-9]+) head=(?P<head>[0-9a-f]+) -->\n"
    r"(?P<body>.*?)\n<!-- /tlx-ops-perf-smoke:(?P=platform) -->", re.S)
INTRO = ("Same-runner A/B of a small tlx.ops subset (`bench_mm`, `bench_flash_attn*`, `bench_norm`; "
         "`--head 8 --fwd-only`): each GPU benches the PR's base and head back to back on one GPU and "
         "updates its own section when it finishes. Advisory only. Full numbers stay on the nightly suite.")


def parse_sections(body):
    """{platform: (head, text)} from a comment body; empty for a legacy or foreign body."""
    if not body or not body.startswith(MARKER):
        return {}
    return {m["platform"]: (m["head"], m["body"]) for m in SECTION.finditer(body)}


def order_key(platform):
    return (ORDER.index(platform) if platform in ORDER else len(ORDER), platform)


def compose(sections, head, note=""):
    lines = [MARKER, "## TLX.ops perf smoke (advisory)", "", INTRO]
    if note:
        lines += ["", note]
    stale = [p for p in sorted(sections, key=order_key) if sections[p][0] != head]
    if stale:
        lines += [
            "", "From an older head, not yet re-run for this one: " +
            ", ".join(f"{p} (`{sections[p][0][:12]}`)" for p in stale) + "."
        ]
    for platform in sorted(sections, key=order_key):
        section_head, text = sections[platform]
        lines += [
            "", f"<!-- tlx-ops-perf-smoke:{platform} head={section_head} -->",
            text.rstrip("\n"), f"<!-- /tlx-ops-perf-smoke:{platform} -->"
        ]
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--platform", required=True)
    parser.add_argument("--head", required=True, help="the benched commit; sections from other heads are marked")
    parser.add_argument("--section", required=True, help="markdown file with this platform's section")
    parser.add_argument("--note", default="", help="line under the intro, e.g. which platforms are paused")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN is not set", file=sys.stderr)
        return 1
    with open(args.section) as fh:
        text = fh.read()
    try:
        ok = publish(GitHub(args.repo, token), args.pr, args.platform, args.head, text, args.note)
    except urllib.error.HTTPError as exc:
        print(f"GitHub API error: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    print(f"{args.platform} section {'published' if ok else 'could not be confirmed after retries'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
