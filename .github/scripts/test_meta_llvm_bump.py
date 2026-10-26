from meta_llvm_bump import (bot_branch_for, commit_message, decide, failed_body, issue_body, issue_marker,
                            issue_release, issue_result, issue_title, oai_pins, pick_tracking, pr_body,
                            parse_sha256_file, release_for, render, same_tree_chain, variants, verdict, verify_state)

OLD = "1889ed344a2ea019320c1a49fc5b7103d0d9468d"
NEW = "3dd0e43fed8836e77165a4f066b983efe1f13a82"
SYNC = "aaaaaaaa00000000000000000000000000000000"
BASE = "5f07f818b51b786b0a87b4e514882600ecba112f"
META = {
    "based_on_oai_llvm_hash": BASE,
    "llvm_hash": OLD,
    "llvm_repository": "https://github.com/facebookexperimental/llvm-project",
    "build_number": 1,
    "release_base_url": "https://github.com/facebookexperimental/llvm-project/releases/download/llvm-1889ed34-1",
    "sha256sum": {"almalinux-arm64": "a" * 64, "almalinux-x64": "b" * 64},
}


def _release(sha, build, suffixes):
    tag = f"llvm-{sha[:8]}-{build}"
    assets = []
    for suffix in suffixes:
        name = f"llvm-{sha[:8]}-{suffix}-{build}.tar.gz"
        assets += [{"name": name}, {"name": f"{name}.sha256"}]
    return {"tag_name": tag, "target_commitish": sha, "assets": assets}


def test_oai_pins_accepts_either_manifest():
    # main: llvm-build-info.json leads llvm-info.json; release/3.8.x: llvm-build-info.json is a stale leftover.
    assert oai_pins({"llvm_hash": "info"}, {"llvm_hash": "build"}) == {"info", "build"}
    assert oai_pins({"llvm_hash": "info"}, None) == {"info"}


def test_same_tree_chain_skips_empty_sync_commits():
    commits = [(SYNC, "t2"), ("bbbb", "t2"), (NEW, "t2"), (OLD, "t1")]
    assert same_tree_chain(commits) == [SYNC, "bbbb", NEW]


def test_release_for_finds_build_under_empty_sync_commit_and_newest_build():
    releases = [
        _release(OLD, 1, ["almalinux-x64"]),
        _release(NEW, 1, ["almalinux-x64"]),
        _release(NEW, 2, ["almalinux-x64"])
    ]
    assert release_for([SYNC, NEW], releases)["tag_name"] == "llvm-3dd0e43f-2"
    assert release_for([SYNC], releases) is None


def test_release_for_ignores_tag_commit_mismatch():
    bogus = _release(NEW, 1, [])
    bogus["target_commitish"] = OLD
    assert release_for([OLD], [bogus]) is None


def test_variants_requires_tarball_and_sha256():
    release = _release(NEW, 1, ["almalinux-x64"])
    release["assets"].append({"name": "llvm-3dd0e43f-ubuntu-x64-1.tar.gz"})
    assert set(variants(release)) == {"almalinux-x64"}


def test_parse_sha256_file():
    text = f"{'c' * 64}  llvm-3dd0e43f-almalinux-x64-1.tar.gz\n"
    assert parse_sha256_file(text, "llvm-3dd0e43f-almalinux-x64-1.tar.gz") == "c" * 64


def test_render_keeps_key_order_and_unrelated_keys():
    out = render(META, _release(NEW, 1, []), {"ubuntu-x64": "d" * 64, "almalinux-x64": "e" * 64},
                 "facebookexperimental/llvm-project")
    assert list(out) == list(META)
    assert out["llvm_hash"] == NEW
    assert out["based_on_oai_llvm_hash"] == BASE
    assert out["release_base_url"].endswith("/releases/download/llvm-3dd0e43f-1")
    assert list(out["sha256sum"]) == ["almalinux-x64", "ubuntu-x64"]


def test_decide():
    four = ["almalinux-arm64", "almalinux-x64", "ubuntu-arm64", "ubuntu-x64"]
    full = _release(NEW, 1, four)
    assert decide(META, {"f" * 40}, [NEW], full, four)[0] == "pin-moved"
    assert decide(META, {"f" * 40, BASE}, [NEW], full, four)[0] == "bump"
    assert decide(META, {BASE}, [NEW], None, four)[0] == "no-release"
    assert decide(META, {BASE}, [NEW], _release(NEW, 1, four[:2]),
                  four) == ("incomplete", "llvm-3dd0e43f-1 lacks ubuntu-arm64, ubuntu-x64")
    assert decide(META, {BASE}, [NEW], full, four) == ("bump", "llvm-3dd0e43f-1")
    current = {**META, "llvm_hash": NEW, "sha256sum": dict.fromkeys(four, "0" * 64)}
    assert decide(current, {BASE}, [NEW], full, four)[0] == "current"


def test_decide_bumps_when_release_adds_platforms():
    # Same commit, but the release gained ubuntu variants since the last bump.
    two = ["almalinux-arm64", "almalinux-x64"]
    pinned = {**META, "llvm_hash": NEW}
    assert decide(pinned, {BASE}, [NEW], _release(NEW, 1, two + ["ubuntu-x64"]), two)[0] == "bump"


def test_commit_message_plain_wrapped_bare_labels():
    release = _release(NEW, 1, ["almalinux-x64"])
    new = render(META, release, {"almalinux-x64": "e" * 64}, "facebookexperimental/llvm-project")
    long_subject = "[triton][3.8][LLVM] Cap SLSR candidate-basis SCEV differences per function (#13)"
    msg = commit_message("[Meta LLVM] Bump release/3.8.x to llvm-3dd0e43f-1", "facebookexperimental/triton",
                         "release/3.8.x", "facebookexperimental/llvm-project", "triton_3.8.x", META, new,
                         [(NEW, long_subject)])
    lines = msg.splitlines()
    assert lines[0] == "[Meta LLVM] Bump release/3.8.x to llvm-3dd0e43f-1" and lines[1] == ""
    # Bare labels: git's default cleanup deletes lines starting with '#'.
    assert "Summary:" in lines and "Test Plan:" in lines
    assert not any(line.startswith("#") for line in lines)
    assert all(len(line) <= 72 for line in lines)
    assert "](" not in msg and "https://" not in msg
    body = " ".join(lines)
    assert "facebookexperimental/triton@release/3.8.x tracks the Meta LLVM fork" in body
    assert "facebookexperimental/llvm-project@triton_3.8.x" in body
    assert "Fork changes in this bump (1889ed344a2e...3dd0e43fed88):" in lines
    assert "  3dd0e43fed88 [triton][3.8][LLVM] Cap SLSR candidate-basis SCEV diff..." in lines


RELEASE = {
    "tag_name": "llvm-3dd0e43f-1", "target_commitish": "3dd0e43fed88" + "0" * 28, "published_at": "2026-10-15T07:12:30Z"
}


def test_pr_body_markdown_unwrapped_linked_fork_changes():
    release = _release(NEW, 1, ["almalinux-x64"])
    new = render(META, release, {"almalinux-x64": "e" * 64}, "facebookexperimental/llvm-project")
    long_subject = "[triton][3.8][LLVM] Cap SLSR candidate-basis SCEV differences per function"
    body = pr_body("facebookexperimental/triton", "release/3.8.x", "facebookexperimental/llvm-project", "triton_3.8.x",
                   META, new, [(NEW, f"{long_subject} (#13)"), (BASE, "Direct push")], RELEASE)
    lines = body.splitlines()
    assert lines[:4] == ["# Summary:", "", "**Problem**", ""]
    # One line per paragraph: GitHub renders every newline as a hard break.
    assert any(
        line.startswith("facebookexperimental/triton@release/3.8.x tracks") and line.endswith("release.")
        for line in lines)
    solution = lines.index("**Solution**")
    assert "at llvm-3dd0e43f-1 (published 2026-10-15), built from" in lines[solution + 2]
    assert lines[solution + 2].endswith(". It adds 2 fork changes:")
    # Squash-merged fork commits link their fork PR; others link the commit.
    assert lines[solution + 4:solution + 6] == [
        f"- [#13](https://github.com/facebookexperimental/llvm-project/pull/13) {long_subject}",
        f"- [{BASE[:12]}](https://github.com/facebookexperimental/llvm-project/commit/{BASE}) Direct push",
    ]
    assert lines[solution + 6:solution + 8] == ["", "# Test Plan:"]
    assert "```" not in body


def test_pr_body_without_fork_changes():
    release = _release(NEW, 1, ["almalinux-x64"])
    new = render(META, release, {"almalinux-x64": "e" * 64}, "facebookexperimental/llvm-project")
    body = pr_body("facebookexperimental/triton", "release/3.8.x", "facebookexperimental/llvm-project", "triton_3.8.x",
                   META, new, [], RELEASE)
    assert ". It adds 0 fork changes.\n\n# Test Plan:" in body


SHA = "a" * 40
RUN = {
    "id": 7, "path": ".github/workflows/gpu-tests.yml", "status": "completed", "conclusion": "success", "html_url":
    "https://github.com/r/actions/runs/7"
}


def test_issue_body_has_markers_result_and_self_contained_command():
    body = issue_body("facebookexperimental/triton", "release/3.8.x",
                      "[Meta LLVM] Bump release/3.8.x to llvm-3dd0e43f-1", "# Summary:\n\n```\nlog\n```", RELEASE, SHA,
                      {"gpu-tests.yml": RUN})
    assert body.startswith(issue_marker("release/3.8.x"))
    assert issue_result(body) == ("ready", SHA)
    assert ("gh pr create -R facebookexperimental/triton --base release/3.8.x "
            "--head scratch/meta-llvm-bump/release-3.8.x") in body
    # The PR body rides in a quoted heredoc, inside a fence longer than the body's own fences.
    assert "--body-file - <<'PR_BODY'\n# Summary:\n\n```\nlog\n```\nPR_BODY\n````" in body
    assert "GPU runs: [gpu-tests.yml](https://github.com/r/actions/runs/7)." in body
    assert issue_marker("main") != issue_marker("release/3.8.x")
    assert "Release: `llvm-3dd0e43f-1`, built from fork commit `3dd0e43fed88`, published 2026-10-15 07:12:30 UTC." in body


def test_failed_and_timeout_bodies_have_no_pr_command():
    failed = {"gpu-tests.yml": {**RUN, "conclusion": "failure"}}
    body = failed_body("r", "main", RELEASE, SHA, failed, ["b200 / b200-tlx-test"])
    assert issue_result(body) == ("failed", SHA) and issue_release(body) == "llvm-3dd0e43f-1"
    assert "Failed jobs: `b200 / b200-tlx-test`." in body and "gh run rerun --failed -R r <id>` (runs: 7)" in body
    assert "gh pr create" not in body
    stuck = failed_body("r", "main", RELEASE, SHA, {}, [], 8)
    assert issue_result(stuck) == ("timeout", SHA)
    assert "still queued or running after 8h" in stuck and "GPU runs: none started." in stuck


def test_verify_state_waits_for_every_workflow_then_times_out():
    h100 = {**RUN, "id": 1, "path": ".github/workflows/h100.yml"}
    b200 = {**RUN, "id": 2, "path": ".github/workflows/b200.yml", "status": "in_progress", "conclusion": None}
    flows = ["h100.yml", "b200.yml"]
    assert verify_state([h100, b200], flows, 1, 8)[0] == "running"
    assert verify_state([h100], flows, 1, 8)[0] == "running"  # b200 never started
    assert verify_state([h100], flows, 8, 8)[0] == "timeout"
    rerun = {**b200, "id": 3, "status": "completed", "conclusion": "success"}
    state, latest = verify_state([h100, b200, rerun], flows, 9, 8)
    assert state == "done" and latest["b200.yml"]["id"] == 3
    # Runs of other workflows on the same commit do not count.
    assert verify_state([{**RUN, "path": ".github/workflows/pre-commit.yml"}], ["gpu-tests.yml"], 1, 8)[0] == "running"


def test_verdict_needs_every_run_green():
    assert verdict({"gpu-tests.yml": RUN}) == "ready"
    assert verdict({"gpu-tests.yml": RUN, "b200.yml": {**RUN, "conclusion": "cancelled"}}) == "failed"


def test_bot_branch_for():
    assert bot_branch_for("release/3.8.x") == "scratch/meta-llvm-bump/release-3.8.x"
    assert bot_branch_for("main") == "scratch/meta-llvm-bump/main"


def test_issue_release_reads_release_line_not_old_pin():
    body = issue_body("facebookexperimental/triton", "main", "T",
                      "It pins llvm-1889ed34-1, but the fork branch has a newer published release.", RELEASE, SHA, {})
    assert issue_release(body) == "llvm-3dd0e43f-1"
    assert issue_release("<!-- meta-llvm-bump main -->\nold format, no release line") is None
    assert issue_release(None) is None


def test_issue_title_dates_the_release():
    assert issue_title("main", RELEASE) == "[Meta LLVM] Bump ready: main to llvm-3dd0e43f-1 (2026-10-15)"
    assert issue_title(
        "main", RELEASE,
        "timeout") == ("[Meta LLVM] Bump failed verify (timed out): main to llvm-3dd0e43f-1 (2026-10-15)")


def test_pick_tracking_keeps_newest_and_returns_duplicates():
    issues = [{"number": n, "body": issue_marker(t) + "\nbody"}
              for n, t in [(4133, "main"), (4134, "release/3.8.x"), (4139, "main"), (4136, "main")]]
    issues.append({"number": 4126, "body": None})
    keep, dups = pick_tracking(issues, "main")
    assert keep["number"] == 4139
    assert [d["number"] for d in dups] == [4136, 4133]
    assert pick_tracking(issues, "release/3.8.x")[0]["number"] == 4134
    assert pick_tracking(issues[-1:], "main") == (None, [])
