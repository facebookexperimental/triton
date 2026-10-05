from nightly_select import canonical_check_name, check_green, commit_green, select

CUT = "2026-07-17T04:00:00Z"


def r(t, c, e="schedule"):
    return {"completed_at": t, "conclusion": c, "event": e}


def test_latest_verdict_wins_over_earlier_success():
    runs = [r("2026-07-16T10:00:00Z", "success"), r("2026-07-16T14:00:00Z", "failure")]
    assert check_green(runs, CUT) is False


def test_cancelled_is_ignored_not_a_verdict():
    runs = [r("2026-07-16T00:59:00Z", "cancelled"), r("2026-07-17T03:01:00Z", "success")]
    assert check_green(runs, CUT) is True


def test_runs_after_cut_are_invisible():
    runs = [r("2026-07-17T18:46:00Z", "success")]  # after 04:00 cut
    assert check_green(runs, CUT) is False


def test_no_runs_is_not_green():
    assert check_green([], CUT) is False


def test_commit_green_requires_all_four():
    ok = {
        "LIT Tests": [r("2026-07-16T01:00:00Z", "success")], "h100-tlx-test": [r("2026-07-17T03:01:00Z", "success")],
        "mi350-tlx-test": [r("2026-07-16T01:00:00Z",
                             "success")], "b200-tlx-test": [r("2026-07-16T01:10:00Z", "success")]
    }
    req = ["LIT Tests", "h100-tlx-test", "mi350-tlx-test", "b200-tlx-test"]
    assert commit_green(ok, req, CUT) is True
    missing = dict(ok)
    missing["b200-tlx-test"] = []
    assert commit_green(missing, req, CUT) is False


def test_select_walks_to_first_green():
    data = {
        "newest": {"LIT Tests": []},  # not green
        "older": {"LIT Tests": [r("2026-07-16T01:00:00Z", "success")]}
    }
    req = ["LIT Tests"]
    assert select(["newest", "older"], lambda s: data[s], req, CUT) == "older"


def test_select_returns_none_when_cap_exhausted():
    req = ["LIT Tests"]
    assert select(["a", "b"], lambda s: {"LIT Tests": []}, req, CUT) is None


def test_reusable_workflow_check_names():
    for platform in ("h100", "b200", "mi350"):
        name = f"{platform}-tlx-test"
        assert canonical_check_name(name) == name
        assert canonical_check_name(f"{platform} / {name}") == name
    assert canonical_check_name("LIT Tests") == "LIT Tests"
    assert canonical_check_name("unrelated / b200-tlx-test") == "unrelated / b200-tlx-test"


def test_fetch_normalizes_names_and_paginates(monkeypatch):
    import nightly_run

    pages = []

    def gh_json(path):
        pages.append(path)
        if path.endswith("page=1"):
            return {
                "check_runs": [{"name": "other"}] * 99 +
                [{"name": "b200-tlx-test", "completed_at": "2026-07-16T01:00:00Z", "conclusion": "success"}]
            }
        return {
            "check_runs":
            [{"name": "b200 / b200-tlx-test", "completed_at": "2026-07-17T01:00:00Z", "conclusion": "failure"}]
        }

    monkeypatch.setattr(nightly_run, "gh_json", gh_json)
    checks = nightly_run.fetch("sha")
    assert len(pages) == 2
    assert len(checks["b200-tlx-test"]) == 2
    assert not check_green(checks["b200-tlx-test"], CUT)
