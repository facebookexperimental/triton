"""Unit tests for reopen_failure_issue (stdlib-only; run with pytest)."""
import datetime as dt
import re

import reopen_failure_issue as r

REPO = "o/r"
TITLE = "[nightly] W / j / t::test_a"
LABELS = ["ci_issues", "nightly"]
NOW = dt.datetime(2026, 10, 26, 12, tzinfo=dt.timezone.utc)
BODY = "> header\n<!-- STATUS -->\n### Summary\n```\nAssertionError: Tensor-likes are not close!\n```\n"


class FakeApi:
    """Answers the calls decide/reopen make from a list of issues; records every write."""

    def __init__(self, issues, reopened=0):
        self.issues = {i["number"]: i for i in issues}
        self.reopened = reopened
        self.writes = []

    def __call__(self, method, path, fields=None):
        if method != "GET":
            self.writes.append((method, path, fields))
            return {}
        if path.startswith("search/issues"):
            state = "open" if "is%3Aopen" in path else "closed"
            return {"items": [i for i in self.issues.values() if i["state"] == state]}
        if path.endswith("/events?per_page=100"):
            return [{"event": "reopened", "actor": {"login": r.BOT}}] * self.reopened
        return self.issues[int(re.search(r"issues/(\d+)$", path)[1])]


def closed(number=7, hours_ago=5, by=r.BOT, body=BODY, title=TITLE):
    at = (NOW - dt.timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")
    return {
        "number": number, "title": title, "state": "closed", "closed_at": at, "closed_by": {"login": by}, "body": body
    }


def decide(api, summary="AssertionError: Mismatched elements: 3 / 1024"):
    return r.decide(api, REPO, TITLE, LABELS, summary, NOW)


def test_reopens_a_recent_bot_closed_issue_with_the_same_error_type():
    issue, reopens, flaky = decide(FakeApi([closed()]))
    assert (issue["number"], reopens, flaky) == (7, 0, False)


def test_an_open_issue_is_left_to_the_report_step():
    api = FakeApi([closed(), dict(closed(number=8), state="open")])
    assert decide(api) is None


def test_a_person_closed_issue_is_not_reopened():
    assert decide(FakeApi([closed(by="wychi")])) is None


def test_only_the_latest_close_counts():
    api = FakeApi([closed(number=7, hours_ago=30), closed(number=8, hours_ago=2, by="wychi")])
    assert decide(api) is None


def test_an_issue_closed_over_a_week_ago_is_not_reopened():
    assert decide(FakeApi([closed(hours_ago=7 * 24 + 1)])) is None
    assert decide(FakeApi([closed(hours_ago=7 * 24 - 1)])) is not None


def test_a_different_error_type_files_a_new_issue():
    assert decide(FakeApi([closed()]), summary="RuntimeError: CUDA error") is None


def test_a_title_that_only_contains_the_search_text_does_not_match():
    assert decide(FakeApi([closed(title=TITLE + "[x]")])) is None


def test_the_third_recurrence_is_flaky():
    assert decide(FakeApi([closed()], reopened=1))[2] is False
    assert decide(FakeApi([closed()], reopened=2))[2] is True


def test_reopen_comments_and_labels_only_when_flaky():
    api = FakeApi([closed()])
    r.reopen(api, REPO, api.issues[7], 1, False, "RUN")
    assert [w[:2] for w in api.writes] == [("PATCH", "repos/o/r/issues/7"), ("POST", "repos/o/r/issues/7/comments")]
    assert "reopened (2 of 2)" in api.writes[1][2]["body"]

    api = FakeApi([closed()])
    r.reopen(api, REPO, api.issues[7], 2, True, "RUN")
    assert ("POST", "repos/o/r/issues/7/labels", {"labels": ["flaky"]}) in api.writes
    assert "close it by hand" in api.writes[-1][2]["body"]


def test_error_type_and_summary_of():
    assert r.error_type("AssertionError: Tensor-likes are not close!") == "AssertionError"
    assert r.error_type("") == ""
    assert r.summary_of(BODY) == "AssertionError: Tensor-likes are not close!"
    assert r.summary_of("no summary") == ""
