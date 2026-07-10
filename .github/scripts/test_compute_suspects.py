#!/usr/bin/env python3
"""Unit tests for compute_suspects pure helpers (stdlib-only; run with pytest)."""

import compute_suspects as cs


def test_extract_pr():
    assert cs.extract_pr("Fix 2-CTA TMEM verify crash (#1905)") == 1905
    assert cs.extract_pr("Add CLC GEMM support (#1882) ") == 1882
    assert cs.extract_pr("no pr number here") is None
    assert cs.extract_pr("mentions (#123) mid-line but ends elsewhere") is None


def test_parse_commits_preserves_order_and_pr():
    log = "aaa\tFirst (#10)\nbbb\tSecond change (#11)\nccc\tNo pr\n"
    commits = cs.parse_commits(log)
    assert [c["sha"] for c in commits] == ["aaa", "bbb", "ccc"]
    assert [c["pr"] for c in commits] == [10, 11, None]


def test_test_subsystem_dir_pytest():
    nid = "third_party/tlx/tutorials/testing/test_correctness.py::test_blackwell_fa_ws"
    assert cs.test_subsystem_dir(nid) == "third_party/tlx/tutorials/testing"


def test_test_subsystem_dir_pytest_dotted_classname():
    # JUnit stores the classname dotted (pytest default), not as a path.
    nid = "third_party.tlx.tutorials.testing.test_correctness::test_blackwell_fa_ws"
    assert cs.test_subsystem_dir(nid) == "third_party/tlx/tutorials/testing"


def test_test_subsystem_dir_lit():
    assert cs.test_subsystem_dir("TRITON :: Conversion/foo.mlir") == "test/Conversion"


def test_test_subsystem_dir_bucket_is_none():
    assert cs.test_subsystem_dir("b200-tritonbench") is None
    assert cs.test_subsystem_dir("") is None


def test_rank_prs_orders_by_overlap():
    commits = [
        {"sha": "a", "subject": "unrelated (#1)", "pr": 1},
        {"sha": "b", "subject": "touches the test dir (#2)", "pr": 2},
        {"sha": "c", "subject": "no pr", "pr": None},
    ]
    files_by_sha = {
        "a": ["docs/readme.md"],
        "b": ["third_party/tlx/tutorials/testing/test_correctness.py"],
        "c": ["whatever.py"],
    }
    test_dir = "third_party/tlx/tutorials/testing"
    ranked = cs.rank_prs(commits, files_by_sha, test_dir)
    assert ranked[0]["pr"] == 2
    assert ranked[0]["score"] > ranked[1]["score"]
    assert {r["pr"] for r in ranked} == {1, 2}  # PR-less commit dropped


def test_rank_prs_dedupes_pr_keeping_best_score():
    commits = [
        {"sha": "a", "subject": "part 1 (#5)", "pr": 5},
        {"sha": "b", "subject": "part 2 (#5)", "pr": 5},
    ]
    files_by_sha = {"a": ["x/y.py"], "b": ["lib/Dialect/TritonGPU/foo.cpp"]}
    ranked = cs.rank_prs(commits, files_by_sha, "lib/Dialect/TritonGPU")
    assert len(ranked) == 1
    assert ranked[0]["pr"] == 5
    assert ranked[0]["score"] == 3  # from commit b: lib/Dialect/TritonGPU (3 components)


def test_rank_prs_no_test_dir_keeps_recency_order():
    commits = [
        {"sha": "a", "subject": "newest (#9)", "pr": 9},
        {"sha": "b", "subject": "older (#8)", "pr": 8},
    ]
    ranked = cs.rank_prs(commits, {"a": ["f.py"], "b": ["g.py"]}, None)
    assert [r["pr"] for r in ranked] == [9, 8]


def test_render_markdown_empty():
    md = cs.render_markdown([], "goodsha123", "badsha456", None)
    assert "suspect list unavailable" in md


def test_render_markdown_table():
    ranked = [{"pr": 2, "subject": "touches dir", "score": 3, "top_path": "a/b/c.py"}]
    md = cs.render_markdown(ranked, "goodsha123", "badsha456", "a/b")
    assert "#2" in md
    assert "`a/b/c.py`" in md
    assert "| Rank | PR | Overlap | Title |" in md
