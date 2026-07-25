from nightly_select import check_green, commit_green, select

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
