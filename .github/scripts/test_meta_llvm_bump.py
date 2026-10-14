from meta_llvm_bump import (commit_message, decide, issue_body, issue_marker, oai_pins, parse_sha256_file, release_for,
                            render, same_tree_chain, variants)

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


def test_issue_body_has_marker_command_and_message():
    body = issue_body("facebookexperimental/triton", "release/3.8.x", "scratch/meta-llvm-bump/release-3.8.x",
                      "[Meta LLVM] Bump release/3.8.x to llvm-3dd0e43f-1", "TITLE\n\nSummary:\nbody text")
    assert body.startswith(issue_marker("release/3.8.x"))
    assert ("gh pr create -R facebookexperimental/triton --base release/3.8.x "
            "--head scratch/meta-llvm-bump/release-3.8.x") in body
    # The PR body comes from the branch's own commit message, minus the title line.
    assert ("gh api repos/facebookexperimental/triton/commits/scratch/meta-llvm-bump/release-3.8.x "
            "-q .commit.message | tail -n +3") in body
    assert "compare/release/3.8.x...scratch/meta-llvm-bump/release-3.8.x" in body
    assert "Summary:\nbody text" in body
    assert issue_marker("main") != issue_marker("release/3.8.x")
