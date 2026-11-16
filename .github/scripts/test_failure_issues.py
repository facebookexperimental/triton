"""Tests for CI failure episodes and verified recovery."""
import json
import re
import subprocess
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

import failure_issues as fi
import parse_junit_failures as junit
import parse_lit_failures as lit
from test_results import encode_recovery

REPO = "o/r"
LABELS = ["ci_issues", "nightly"]
PREFIX = "[nightly] W / j / "
TITLE = PREFIX + "t::test_a"
RUN = "https://github.com/o/r/actions/runs/"


def body(ids=("t::test_a[one]", ), run_id=100, sha="a" * 40, norm="t::test_a"):
    return (f"<!-- bisect-meta\nbad_sha={sha}\nrepro=\n-->\n<!-- STATUS -->\n"
            f"| SHA | `{sha}` |\n| Run | {RUN}{run_id} |\n"
            f"| Normalized failing identity | `{norm}` |\n"
            "### Raw failing test IDs\n```\n" + "\n".join(ids) + "\n```\n"
            "### Summary\n```\nAssertionError: assert 1 == 0\n```\n")


def issue(number, state="open", title=TITLE, labels=LABELS, **kw):
    return dict(number=number, id=number * 10, title=title, state=state, closed_at="2026-11-12T16:40:00Z",
                closed_by={"login": "github-actions[bot]"}, labels=[{"name": l} for l in labels], body=body(), **kw)


class FakeApi:

    def __init__(self, issues):
        self.issues = {i["number"]: i for i in issues}
        self.writes = []

    def __call__(self, method, path, fields=None, paginate=False):
        url = urlsplit(path)
        if method == "GET" and url.path == f"repos/{REPO}/issues":
            args = {k: v[0] for k, v in parse_qs(url.query).items()}
            want = set(args["labels"].split(","))
            return [i for i in self.issues.values() if args["state"] in ("all", i["state"]) and want <= labels_of(i)]
        number = int(url.path.rsplit("/", 1)[-1]) if method == "GET" else None
        if method == "GET":
            return self.issues[number]
        self.writes.append((method, path, fields))
        if method == "PATCH":
            self.issues[int(url.path.rsplit("/", 1)[-1])].update(fields)
        return {}


def reconcile(api, failures=(), passed=(), run_id=150, complete=True, resolved=()):
    proof = encode_recovery(passed, resolved, complete)
    return fi.reconcile(api, REPO, PREFIX, LABELS, json.dumps(failures), RUN + str(run_id), run_id, proof)


def xml(tmp_path, cases, name="result.xml", seconds=1):
    suite = ET.Element("testsuite", tests=str(len(cases)), time=str(seconds))
    for rid, status in cases:
        module, _, name_ = rid.partition("::")
        tc = ET.SubElement(suite, "testcase", classname=module, name=name_)
        if status != "passed":
            ET.SubElement(tc, status, message="AssertionError: wrong result")
    path = tmp_path / name
    ET.ElementTree(suite).write(path)
    return str(path)


def parsed_recovery(paths, expected=None, budgets=(), failed=False):
    return junit.recovery_results(paths, paths if expected is None else expected, budgets, 0.8, failed, "job-failed")


def test_continuous_failures_keep_the_open_episode():
    api = FakeApi([issue(7)])
    old = dict(api.issues[7])
    assert fi.decide(api, REPO, TITLE, LABELS, 150) == "file"
    assert reconcile(api, [{"issue_title": TITLE}]) == []
    assert api.issues[7] == old and api.writes == []


def test_a_failure_after_verified_recovery_starts_a_new_episode():
    api = FakeApi([issue(7)])
    old_body = api.issues[7]["body"]
    assert reconcile(api, passed=["t::test_a[one]"]) == [7]
    assert api.issues[7]["state"] == "closed"
    assert fi.first_seen_sha(api.issues[7]["body"]) == "a" * 40
    assert old_body.rstrip() in api.issues[7]["body"]
    writes = len(api.writes)
    assert fi.decide(api, REPO, TITLE, LABELS, 151) == "file"
    assert len(api.writes) == writes and api.issues[7]["state"] == "closed"


def test_an_older_run_cannot_replace_a_newer_failure_or_recovery():
    for state in ("open", "closed"):
        newer = dict(issue(7, state=state), body=body(run_id=201))
        api = FakeApi([newer])
        assert fi.decide(api, REPO, TITLE, LABELS, 200) == "skip"
        assert reconcile(api, passed=["t::test_a[one]"], run_id=200) == []
        assert api.writes == []
    api = FakeApi([dict(issue(7, state="closed"), body=fi.record_resolution(body(), 201))])
    assert fi.decide(api, REPO, TITLE, LABELS, 200) == "skip"
    assert fi.decide(api, REPO, TITLE, LABELS, 202) == "file"


def test_a_preflight_read_error_does_not_drop_the_failure(monkeypatch, tmp_path):
    output = tmp_path / "output"

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gh")

    monkeypatch.setattr(fi, "gh", unavailable)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_RUN_ID", "150")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        "sys.argv",
        ["failure_issues.py", "check", "--title", TITLE, "--labels", ",".join(LABELS), "--run-url", RUN + "150"])
    fi.main()
    assert output.read_text() == "action=file\n"


def test_a_skipped_or_unselected_case_does_not_prove_recovery(tmp_path):
    for cases in ([('t::test_a[one]', 'skipped'), ('t::other', 'passed')], [('t::other', 'passed')]):
        path = xml(tmp_path, cases)
        api = FakeApi([issue(7)])
        assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, parsed_recovery([path])) == []
        assert api.writes == []


def test_every_previously_failing_parameter_must_pass(tmp_path):
    both = dict(issue(7), body=body(ids=("t::test_a[one]", "t::test_a[two]")))
    for second, expected in (("skipped", []), ("passed", [7])):
        path = xml(tmp_path, [("t::test_a[one]", "passed"), ("t::test_a[two]", second)])
        api = FakeApi([dict(both)])
        assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, parsed_recovery([path])) == expected


def test_missing_malformed_or_unfinished_results_do_not_close_issues(tmp_path):
    path = xml(tmp_path, [("t::test_a[one]", "passed")])
    missing = str(tmp_path / "missing.xml")
    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<testsuite>")
    for expected, completed in (([path, missing], [path]), ([path, str(malformed)], [path,
                                                                                     str(malformed)]), ([path], [])):
        api = FakeApi([issue(7)])
        assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, parsed_recovery(completed,
                                                                                               expected)) == []
        assert api.writes == []


def test_no_evidence_means_no_automatic_close():
    api = FakeApi([issue(7)])
    assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150) == []
    assert api.writes == []


def test_a_completed_failing_sibling_does_not_hide_a_real_recovery(tmp_path):
    path = xml(tmp_path, [("t::test_a[one]", "passed"), ("t::other", "failure")])
    api = FakeApi([issue(7)])
    failures = [{"issue_title": PREFIX + "t::other"}]
    assert fi.reconcile(api, REPO, PREFIX, LABELS, json.dumps(failures), RUN + "150", 150,
                        parsed_recovery([path], failed=True)) == [7]


def test_reconcile_leaves_flaky_and_other_jobs_issues_alone():
    api = FakeApi([issue(7, labels=LABELS + [fi.FLAKY]), issue(8, title="[nightly] W / other / t::test_a")])
    assert reconcile(api, passed=["t::test_a[one]"]) == []
    assert api.writes == []


def test_a_job_level_fallback_is_not_a_pass():
    api = FakeApi([issue(7)])
    assert reconcile(api, [{"issue_title": PREFIX + "bucket", "fallback": True}], passed=["t::test_a[one]"]) == []
    assert fi.reconcile(api, REPO, PREFIX, LABELS, "not json", RUN + "150", 150) == []
    assert api.writes == []


def test_slow_steps_need_a_completed_passing_suite_under_budget(tmp_path):
    old = dict(issue(7, title=PREFIX + "slow-step:result"), body=body(ids=(), norm="slow-step:result"))
    for status, seconds, expected in (("passed", 49, []), ("skipped", 1, []), ("passed", 1, [7])):
        path = xml(tmp_path, [("t::test_a[one]", status), ("t::unsupported_arch", "skipped")], seconds=seconds)
        api = FakeApi([dict(old)])
        proof = parsed_recovery([path], budgets=[(path, 1)])
        assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, proof) == expected


def test_a_complete_test_run_can_resolve_a_job_level_failure(tmp_path):
    path = xml(tmp_path, [("t::test_a[one]", "passed")])
    old = dict(issue(7, title=PREFIX + "pytest-junit-missing"), body=body(ids=(), norm="pytest-junit-missing"))
    api = FakeApi([old])
    assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, parsed_recovery([path])) == [7]


def test_lit_requires_complete_output_and_a_real_pass(tmp_path):
    path = tmp_path / "lit.log"
    old = dict(issue(7, title=PREFIX + "TRITON :: a.mlir"), body=body(ids=("TRITON :: a.mlir", ),
                                                                      norm="TRITON :: a.mlir"))
    for status, tail, expected in (("UNSUPPORTED", "Testing Time: 1.0s\n", []), ("PASS", "", []),
                                   ("PASS", "Testing Time: 1.0s\n", [7])):
        path.write_text(f"-- Testing: 1 tests, 1 workers --\n{status}: TRITON :: a.mlir (1 of 1)\n{tail}")
        api = FakeApi([dict(old)])
        proof = lit.recovery_results(str(path), True, False, "compiler-lit")
        assert fi.reconcile(api, REPO, PREFIX, LABELS, "[]", RUN + "150", 150, proof) == expected


def labels_of(issue_):
    return {l["name"] for l in issue_["labels"]}


class GroupApi:
    """Issues, parents and sub-issue links as the group step sees them; records writes."""

    def __init__(self, issues):
        self.issues = {i["number"]: i for i in issues}
        self.writes = []

    def __call__(self, method, path, fields=None, paginate=False):
        url = urlsplit(path)
        number = lambda: int(re.search(r"issues/(\d+)", url.path)[1])
        if method == "GET" and url.path == f"repos/{REPO}/issues":
            args = {k: v[0] for k, v in parse_qs(url.query).items()}
            want = set(args["labels"].split(","))
            return [i for i in self.issues.values() if args["state"] in ("all", i["state"]) and want <= labels_of(i)]
        if method == "GET" and url.path.endswith("/sub_issues"):
            return [i for i in self.issues.values() if i.get("parent_issue_url", "").endswith(f"/issues/{number()}")]
        if method == "GET" and "/commits/" in url.path:
            return {"author": None, "commit": {"message": "Change (#77)\n\nbody"}}
        if method == "GET":
            return self.issues.get(number(), {"user": {"login": "author"}})
        self.writes.append((method, path, fields))
        if method == "POST" and url.path == f"repos/{REPO}/issues":
            n = 900 + sum(fi.PARENT in labels_of(i) for i in self.issues.values())
            parent_ = dict(fields, number=n, id=n * 10, state="open", labels=[{"name": fi.PARENT}])
            self.issues[n] = parent_
            return parent_
        if method == "POST" and url.path.endswith("/sub_issues"):
            child = next(i for i in self.issues.values() if i["id"] == fields["sub_issue_id"])
            child["parent_issue_url"] = f"https://api.github.com/repos/{REPO}/issues/{number()}"
        if method == "PATCH":
            self.issues[number()].update(fields)
        return {}


def failing(number, first_seen, parent=None, labels=LABELS):
    i = dict(issue(number, title=f"{PREFIX}t::test_{number}", labels=labels), body=body(sha=first_seen * 40))
    return dict(i, parent_issue_url=f"https://api.github.com/repos/{REPO}/issues/{parent}") if parent else i


def parent(number, state="open"):
    return issue(number, state=state, title=f"CI failures first seen at {number}", labels=[fi.PARENT])


def test_a_failure_belongs_to_the_commit_where_its_episode_started():
    api = GroupApi([parent(800), failing(1, "a", parent=800), failing(4, "a"), failing(2, "b"), failing(3, "b")])
    fi.group(api, REPO, PREFIX, LABELS, "b" * 40, "main", RUN + "200")
    assert api.issues[1]["parent_issue_url"].endswith("/issues/800")
    assert "parent_issue_url" not in api.issues[4]
    new_parent = next(i for n, i in api.issues.items() if n != 800 and fi.PARENT in labels_of(i))
    assert new_parent["assignees"] == ["author"] and new_parent["title"] == "Test failures at bbbbbbb (#77): j"
    assert all(api.issues[n]["parent_issue_url"].endswith(f"/issues/{new_parent['number']}") for n in (2, 3))


def test_a_commit_has_one_parent_whichever_jobs_fail():
    lit_prefix = "[nightly] Compiler LIT Tests / lit-tests / "
    api = GroupApi([failing(1, "b"), dict(failing(2, "b"), title=lit_prefix + "a.mlir")])
    fi.group(api, REPO, PREFIX, LABELS, "b" * 40, "main", RUN + "200")
    fi.group(api, REPO, lit_prefix, LABELS, "b" * 40, "main", RUN + "201")
    parents = [i for i in api.issues.values() if fi.PARENT in labels_of(i)]
    assert len(parents) == 1 and parents[0]["title"] == "Test failures at bbbbbbb (#77): j, lit-tests"
    assert api.issues[2]["parent_issue_url"].endswith(f"/issues/{parents[0]['number']}")
    parents[0]["state"] = "closed"
    api.issues[3] = failing(3, "b")
    fi.group(api, REPO, PREFIX, LABELS, "b" * 40, "main", RUN + "202")
    assert [i for i in api.issues.values() if fi.PARENT in labels_of(i)] == parents
    assert parents[0]["state"] == "open"
    assert api.issues[3]["parent_issue_url"] == api.issues[2]["parent_issue_url"]


def test_a_recovered_episode_keeps_its_history_and_new_children_get_a_new_parent():
    old = dict(failing(1, "a", parent=800), state="closed")
    old_body = old["body"]
    api = GroupApi([parent(800, state="closed"), old, failing(2, "b")])
    fi.group(api, REPO, PREFIX, LABELS, "b" * 40, "main", RUN + "200")
    assert api.issues[1]["body"] == old_body and api.issues[1]["state"] == "closed"
    assert api.issues[1]["parent_issue_url"].endswith("/issues/800")
    assert not api.issues[2]["parent_issue_url"].endswith("/issues/800")
    assert api.issues[800]["state"] == "closed"


def test_a_parent_stays_open_exactly_while_a_non_flaky_sub_issue_is_open():
    flaky = failing(1, "a", parent=800, labels=LABELS + [fi.FLAKY])
    fixed = dict(failing(2, "a", parent=800), state="closed")
    api = GroupApi([parent(800), flaky, fixed])
    fi.group(api, REPO, PREFIX, LABELS, "c" * 40, "main", RUN + "200")
    assert api.issues[800]["state"] == "closed"
    api = GroupApi([parent(800, state="closed"), failing(2, "a", parent=800)])
    fi.group(api, REPO, PREFIX, LABELS, "c" * 40, "main", RUN + "200")
    assert api.issues[800]["state"] == "open"
