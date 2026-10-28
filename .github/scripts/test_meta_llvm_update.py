"""Decision flow of meta_llvm_update.main() with every GitHub call faked."""
import sys

import pytest

import meta_llvm_update as m

PINNED = {"llvm_hash": "p" * 40, "release_base_url": "https://x/llvm-pppppppp-1", "based_on_oai_llvm_hash": "o" * 40}
X = {**PINNED, "llvm_hash": "x" * 40, "release_base_url": "https://x/llvm-xxxxxxxx-1"}
Y_RELEASE = {"tag_name": "llvm-yyyyyyyy-1"}
Y = {**PINNED, "llvm_hash": "y" * 40, "release_base_url": "https://x/llvm-yyyyyyyy-1"}


@pytest.fixture
def world(monkeypatch):
    """Fakes for main()'s reads; records its writes in calls."""
    state = {"status": "bump", "pr": None, "issue": None, "branch": None, "settle": "done", "calls": []}
    calls = state["calls"]
    monkeypatch.setattr(m, "read_json", lambda repo, path, ref: PINNED if ref == "main" else None)
    monkeypatch.setattr(m, "oai_pins", lambda *a: {PINNED["based_on_oai_llvm_hash"]})
    monkeypatch.setattr(m, "fork_chain", lambda *a: ["y" * 40])
    monkeypatch.setattr(m, "gh_json", lambda path: [])
    monkeypatch.setattr(m, "release_for", lambda *a: Y_RELEASE)
    monkeypatch.setattr(m, "decide", lambda *a: (state["status"], "detail"))
    monkeypatch.setattr(m, "tracking_issue", lambda *a: state["issue"])
    monkeypatch.setattr(m, "open_pr", lambda *a: state["pr"])
    monkeypatch.setattr(m, "bump_branch", lambda *a: state["branch"])
    monkeypatch.setattr(m, "descends_from", lambda fork, base, sha: sha != base)
    monkeypatch.setattr(m, "leave_pr_alone", lambda repo, pr, issue, newer: calls.append(("leave", newer)))
    monkeypatch.setattr(m, "close_issue", lambda repo, issue, reason: calls.append(("close", reason)))
    monkeypatch.setattr(m, "settle", lambda *a: calls.append(("settle", )) or state["settle"])
    monkeypatch.setattr(m, "push_bump", lambda args, meta, release: calls.append(("push", release["tag_name"])))
    monkeypatch.setattr(
        sys, "argv", ["x", "--target", "main", "--fork-branch", "triton_beta", "--verify-workflows", "gpu-tests.yml"])
    return state


def branch(meta):
    return {"sha": "b" * 40, "age_hours": 1, "meta": meta}


def test_open_pr_is_left_alone(world):
    world["pr"] = {"number": 9, "title": "[Meta LLVM] Bump main to llvm-xxxxxxxx-1"}
    world["branch"] = branch(X)
    m.main()
    assert world["calls"] == [("leave", "llvm-yyyyyyyy-1")]


def test_first_bump_pushes_without_an_issue(world):
    m.main()
    assert world["calls"] == [("push", "llvm-yyyyyyyy-1")]


@pytest.mark.parametrize("settled", ["running", "filed"])
def test_newer_release_waits_for_the_current_bump(world, settled):
    world["branch"], world["settle"] = branch(X), settled
    m.main()
    assert world["calls"] == [("settle", )]


def test_newer_release_replaces_a_settled_bump(world):
    world["branch"], world["issue"] = branch(X), {"number": 1, "body": "Release: `llvm-xxxxxxxx-1`"}
    m.main()
    assert world["calls"] == [("settle", ), ("close", "Superseded by llvm-yyyyyyyy-1; the bot is verifying it."),
                              ("push", "llvm-yyyyyyyy-1")]


def test_bump_already_on_newest_release_only_settles(world):
    world["branch"] = branch(Y)
    m.main()
    assert world["calls"] == [("settle", )]


def test_landed_release_closes_its_issue(world):
    world["status"], world["branch"] = "current", branch(PINNED)
    world["issue"] = {"number": 1, "body": "Release: `llvm-pppppppp-1`"}
    m.main()
    assert world["calls"] == [("close", "main now pins llvm-pppppppp-1.")]


def test_pending_bump_settles_while_the_fork_tip_has_no_release(world):
    world["status"], world["branch"] = "no-release", branch(X)
    m.main()
    assert world["calls"] == [("settle", )]
