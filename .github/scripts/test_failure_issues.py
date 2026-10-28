"""Unit tests for failure_issues (stdlib-only; run with pytest)."""
import datetime as dt
import json
import re
import subprocess
from urllib.parse import parse_qs, urlsplit

import failure_issues as fi

REPO = "o/r"
LABELS = ["ci_issues", "nightly"]
PREFIX = "[nightly] W / j / "
TITLE = PREFIX + "t::test_a"
NOW = dt.datetime(2026, 10, 26, 12, tzinfo=dt.timezone.utc)
RUN = "https://github.com/o/r/actions/runs/"


def body(summary="AssertionError: assert 0.17 < 0.1", run_id=100):
    return (f"> header\n<!-- STATUS -->\n| SHA | `old` |\n| Run | {RUN}{run_id} |\n"
            f"### Summary\n```\n{summary}\n```\n")


def issue(number, state="closed", hours_ago=5, by=fi.BOT, title=TITLE, labels=LABELS, **kw):
    at = (NOW - dt.timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")
    return dict(number=number, title=title, state=state, closed_at=None if state == "open" else at,
                closed_by={"login": by}, labels=[{"name": l} for l in labels], body=body(), **kw)


class FakeApi:
    """GitHub as the scripts see it. Search filters by label and title substring, honors per_page, returns
    results unsorted unless sort=created is asked, and may report a stale state (search_state)."""

    def __init__(self, issues, reopened=0, search_state=None, fail=None):
        self.issues = {i["number"]: i for i in issues}
        self.reopened, self.search_state, self.fail = reopened, search_state or {}, fail
        self.writes = []

    def __call__(self, method, path, fields=None, paginate=False):
        if self.fail and self.fail(method, path):
            raise subprocess.CalledProcessError(1, "gh")
        if method != "GET":
            self.writes.append((method, path, fields))
            return {}
        url = urlsplit(path)
        args = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "search/issues":
            labels = re.findall(r'label:"([^"]+)"', args["q"])
            title = re.search(r'in:title "(.*)"$', args["q"])[1]
            hits = [
                dict(i, state=self.search_state.get(i["number"], i["state"]))
                for i in self.issues.values()
                if title.lower() in i["title"].lower() and set(labels) <= {l["name"]
                                                                           for l in i["labels"]}
            ]
            if args.get("sort") == "created":
                hits.sort(key=lambda i: -i["number"])
            return {"items": hits[:int(args.get("per_page", 30))]}
        if url.path.endswith("/events"):
            return [{"event": "reopened", "actor": {"login": fi.BOT}}] * self.reopened
        if url.path == f"repos/{REPO}/issues":
            want = set(args["labels"].split(","))
            return [
                i for i in self.issues.values() if i["state"] == "open" and want <= {l["name"]
                                                                                     for l in i["labels"]}
            ]
        return self.issues[int(url.path.rsplit("/", 1)[1])]


def decide(api, summary="AssertionError: assert 0.23 < 0.1", run_id=150):
    """The reopen decision: the issue number to reopen, or "skip" / "file"."""
    action, found = fi.decide(api, REPO, TITLE, LABELS, summary, NOW, run_id)
    return found[0]["number"] if action == "reopen" else action


def found(api):
    return fi.decide(api, REPO, TITLE, LABELS, "AssertionError: assert 0.23 < 0.1", NOW, 150)[1]


# --- reopen ---


def test_a_recurrence_with_different_numbers_reopens_the_issue():
    issue_, reopens, flaky = found(FakeApi([issue(7)]))
    assert (issue_["number"], reopens, flaky) == (7, 0, False)


def test_numbers_in_any_format_do_not_split_one_error():
    api = FakeApi([dict(issue(7), body=body("AssertionError: max diff: 9e-05"))])
    assert decide(api, "AssertionError: max diff: 0.00012") == 7


def test_a_different_error_files_a_new_issue_even_with_the_same_exception_type():
    api = FakeApi([dict(issue(7), body=body("RuntimeError: all CUDA-capable devices are busy"))])
    assert decide(api, "RuntimeError: Triton Error [CUDA]: an illegal memory access") == "file"
    api = FakeApi([dict(issue(7), body=body("[memcheck]\nInvalid __global__ read of size 4"))])
    assert decide(api, "[memcheck]\nInvalid __global__ write of size 4") == "file"
    assert decide(api, "[memcheck]\nInvalid __global__ read of size 8") == 7


def test_an_open_issue_is_left_to_the_report_step_unless_a_newer_run_reported_it():
    open_ = lambda run_id: dict(issue(8, state="open"), body=body(run_id=run_id))
    assert decide(FakeApi([issue(7), open_(149)]), run_id=150) == "file"
    assert decide(FakeApi([issue(7), open_(151)]), run_id=150) == "skip"


def test_an_issue_search_still_shows_open_is_reopened_once_confirmed_closed():
    assert decide(FakeApi([issue(7)], search_state={7: "open"})) == 7


def test_only_a_bot_close_within_a_week_reopens():
    assert decide(FakeApi([issue(7, by="wychi")])) == "file"
    assert decide(FakeApi([issue(7, hours_ago=7 * 24 + 1)])) == "file"
    assert decide(FakeApi([issue(7, hours_ago=7 * 24 - 1)])) == 7


def test_the_latest_close_is_chosen_by_close_time_not_result_order():
    assert decide(FakeApi([issue(7, hours_ago=2), issue(8, hours_ago=30)])) == 7


def test_only_the_latest_close_counts():
    assert decide(FakeApi([issue(7, hours_ago=30), issue(8, hours_ago=2, by="wychi")])) == "file"
    assert decide(FakeApi([issue(7, hours_ago=30, by="wychi"), issue(8, hours_ago=2)])) == 8


def test_the_latest_close_is_found_among_many_refiles():
    refiles = [issue(n, hours_ago=13 - n) for n in range(1, 13)]  # #12 closed most recently, an hour ago
    assert decide(FakeApi(refiles)) == 12


def test_issues_of_other_titles_and_labels_are_not_reopened():
    assert decide(FakeApi([issue(7, title=TITLE + "[x]")])) == "file"
    assert decide(FakeApi([issue(7, labels=["cutracer"])])) == "file"


def test_the_third_recurrence_is_flaky():
    assert found(FakeApi([issue(7)], reopened=1))[2] is False
    assert found(FakeApi([issue(7)], reopened=2))[2] is True


def test_reopen_records_the_run_and_labels_only_when_flaky():
    for reopens, flaky in ((1, False), (2, True)):
        api, events = FakeApi([issue(7)]), []
        fi.reopen(api, REPO, api.issues[7], reopens, flaky, "new", RUN + "200", lambda: events.append(len(api.writes)))
        method, path, fields = api.writes[0]
        assert (method, path, fields["state"]) == ("PATCH", f"repos/{REPO}/issues/7", "open")
        assert fi.run_id_of(fields["body"]) == 200 and "| SHA | `new` |" in fields["body"]
        assert events == [1]  # reported reopened right after the PATCH
        assert (("POST", f"repos/{REPO}/issues/7/labels", {"labels": ["flaky"]}) in api.writes) == flaky


def run_main(monkeypatch, tmp_path, api):
    out = tmp_path / "out"
    monkeypatch.setattr(fi, "gh", api)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setattr("sys.argv", [
        "x", "reopen", "--title", TITLE, "--labels", ",".join(LABELS), "--summary", "AssertionError: assert 1 < 0",
        "--sha", "s", "--run-url", RUN + "200"
    ])
    monkeypatch.setattr(fi.dt, "datetime", type("D", (dt.datetime, ), {"now": staticmethod(lambda tz=None: NOW)}))
    fi.main()
    return out.read_text()


def test_an_api_error_before_reopening_leaves_filing_to_the_report_step(monkeypatch, tmp_path):
    api = FakeApi([issue(7)], fail=lambda m, p: p.startswith("search/"))
    assert run_main(monkeypatch, tmp_path, api) == "action=file\n"


def test_a_stale_report_is_skipped(monkeypatch, tmp_path):
    newer = dict(issue(7, state="open"), body=body(run_id=201))
    assert run_main(monkeypatch, tmp_path, FakeApi([newer])) == "action=skip\n"


def test_an_api_error_after_reopening_does_not_file_a_duplicate(monkeypatch, tmp_path):
    api = FakeApi([issue(7)], fail=lambda m, p: p.endswith("/comments"))
    assert run_main(monkeypatch, tmp_path, api) == "action=reopened\n"


# --- reconcile ---


def reconcile(api, failures, run_id=150):
    return fi.reconcile(api, REPO, PREFIX, LABELS, json.dumps(failures), RUN + str(run_id), run_id)


def test_reconcile_closes_recovered_issues_of_this_job_only():
    api = FakeApi([
        issue(1, state="open", title=PREFIX + "t::gone"),
        issue(2, state="open", title=PREFIX + "t::still"),
        issue(3, state="open", title="[nightly] W / other / t::gone"),
    ])
    assert reconcile(api, [{"issue_title": PREFIX + "t::still"}]) == [1]
    assert ("PATCH", f"repos/{REPO}/issues/1", {"state": "closed", "state_reason": "completed"}) in api.writes


def test_reconcile_leaves_flaky_issues_open():
    assert reconcile(FakeApi([issue(1, state="open", labels=LABELS + ["flaky"])]), []) == []


def test_an_older_run_does_not_close_an_issue_a_newer_run_reported():
    newer = dict(issue(1, state="open"), body=body(run_id=151))
    older = dict(issue(2, state="open", title=PREFIX + "t::b"), body=body(run_id=149))
    assert reconcile(FakeApi([newer, older]), [], run_id=150) == [2]


def test_reconcile_closes_nothing_without_a_per_test_signal():
    api = FakeApi([issue(1, state="open")])
    assert reconcile(api, [{"issue_title": PREFIX + "bucket", "fallback": True}]) == []
    assert fi.reconcile(api, REPO, PREFIX, LABELS, "not json", RUN, 150) == []
    assert api.writes == []
