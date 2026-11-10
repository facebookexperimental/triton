import json
import math
import os
import subprocess

import pytest

import tlx_ops_bench_bisect_round as rounds

ENV = {
    **os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"
}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=ENV).stdout.strip()


def commit(repo, message):
    git(repo, "commit", "--allow-empty", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def new_repo(path, count):
    """A linear main of `count` commits, published as origin/main."""
    path.mkdir()
    git(path, "init", "-q")
    git(path, "symbolic-ref", "HEAD", "refs/heads/main")
    shas = [commit(path, f"c{i}") for i in range(count)]
    git(path, "update-ref", "refs/remotes/origin/main", shas[-1])
    return shas


def spec(good, bad, key="case", **extra):
    return {
        "issue_number": 7, "op": "op", "direction": "fwd", "key": key, "platform": "b200", "good_sha": good, "bad_sha":
        bad, "required_ratio": 0.9, "space": None, **extra
    }


def ok(tflops, **extra):
    return {"status": "ok", "values": [tflops, tflops * 1.01], "tflops": tflops, "gpu_uuid": "GPU-1", **extra}


def drive(repo, specs, regressed, *, unbuildable=(), noisy=(), max_rounds=16):
    """Run the whole prepare/advance loop with stub measurements. `regressed(key, sha)` says if a commit is slow."""
    state = {"specs": specs, "max_rounds": max_rounds}
    measured = []
    for _ in range(100):
        prepared = rounds.prepare(state, repo)
        if prepared["done"]:
            return prepared, measured
        measurements = {}
        for name, sha in zip(("good", "bad", "candidate"), prepared["commits"]):
            if sha in unbuildable:
                measurements[name] = {"status": "build_failed", "reason": "no wheel"}
            elif sha in noisy:
                measurements[name] = {"status": "noisy", "runs": []}
            else:
                measurements[name] = ok(50 if regressed(prepared["spec"]["key"], sha) else 100, gpu_uuid="GPU-1")
        if prepared["commits"]:
            measured.append(prepared["commits"])
        state = rounds.advance(prepared, measurements)
    raise AssertionError("the loop did not finish")


def culprit_after(repo, culprit):
    return lambda key, sha: subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", culprit, sha], env=ENV).returncode == 0


def test_every_round_measures_one_new_commit_and_isolates_the_first_bad_commit(tmp_path):
    shas = new_repo(tmp_path / "r", 20)
    prepared, measured = drive(tmp_path / "r", [spec(shas[0], shas[19])], culprit_after(tmp_path / "r", shas[13]))
    [final] = prepared["results"]
    assert final["status"] == "conclusive" and final["first_bad_sha"] == shas[13]
    assert final["first_bad_parents"] == [shas[12]] and final["verified"] is True
    assert len(measured) <= math.ceil(math.log2(19)) + 2
    assert all(len(set(commits)) == len(commits) <= 3 for commits in measured)
    assert all(commits[:2] == [shas[0], shas[19]] for commits in measured)
    assert measured[-1] == [shas[0], shas[19]]  # the last round re-verifies the endpoints without a candidate


def test_the_graph_is_followed_through_a_merged_side_branch(tmp_path):
    repo = tmp_path / "r"
    shas = new_repo(repo, 5)
    git(repo, "checkout", "-q", "-b", "side", shas[2])
    side = [commit(repo, f"s{i}") for i in range(3)]
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "--no-ff", "-q", "side", "-m", "merge")
    tail = [commit(repo, f"t{i}") for i in range(2)]
    git(repo, "update-ref", "refs/remotes/origin/main", tail[-1])
    prepared, _ = drive(repo, [spec(shas[0], tail[-1])], culprit_after(repo, side[1]))
    assert prepared["results"][0]["first_bad_sha"] == side[1]


def test_adjacent_endpoints_conclude_without_a_candidate_once_they_reproduce(tmp_path):
    shas = new_repo(tmp_path / "r", 6)
    prepared = rounds.prepare({"specs": [spec(shas[4], shas[5])]}, tmp_path / "r")
    assert prepared["candidate_sha"] is None and prepared["first_bad_sha"] == shas[5]
    assert prepared["commits"] == [shas[4], shas[5]] and prepared["first_bad_parents"] == [shas[4]]
    final = rounds.advance(prepared, {"good": ok(100), "bad": ok(50)})
    assert final["results"][0]["status"] == "conclusive" and final["results"][0]["first_bad_sha"] == shas[5]
    refuted = rounds.advance(prepared, {"good": ok(100), "bad": ok(99)})
    assert refuted["results"][0]["status"] == "inconclusive"
    assert "first_bad_sha" not in refuted["results"][0]


def test_endpoints_that_stop_reproducing_end_the_spec_and_the_next_one_still_runs(tmp_path):
    shas = new_repo(tmp_path / "r", 10)
    repo = tmp_path / "r"
    specs = [spec(shas[0], shas[9], key="steady"), spec(shas[0], shas[9], key="slow")]
    prepared, _ = drive(repo, specs, lambda key, sha: key == "slow" and sha in shas[5:])
    first, second = prepared["results"]
    assert first["status"] == "inconclusive" and "not reproduced" in first["reason"]
    assert second["status"] == "conclusive" and second["first_bad_sha"] == shas[5]
    assert second["observations"] and not first["observations"]


@pytest.mark.parametrize("failing,reason", [("good", "good endpoint"), ("bad", "bad endpoint")])
def test_unbuildable_or_noisy_endpoints_are_inconclusive_never_a_finding(tmp_path, failing, reason):
    shas = new_repo(tmp_path / "r", 8)
    broken = {"good": shas[0], "bad": shas[7]}[failing]
    for kind in ({"unbuildable": [broken]}, {"noisy": [broken]}):
        prepared, _ = drive(tmp_path / "r", [spec(shas[0], shas[7])], culprit_after(tmp_path / "r", shas[4]), **kind)
        [final] = prepared["results"]
        assert final["status"] == "inconclusive" and "first_bad_sha" not in final
        assert reason in final["reason"]


def test_unmeasurable_commits_are_skipped_and_never_counted_as_bad(tmp_path):
    shas = new_repo(tmp_path / "r", 12)
    repo = tmp_path / "r"
    # Git's first two candidates in this range are c5 and then c6 once c5 is skipped.
    prepared, measured = drive(repo, [spec(shas[0], shas[11])], culprit_after(repo, shas[9]), unbuildable={shas[5]},
                               noisy={shas[6]})
    [final] = prepared["results"]
    assert final["status"] == "conclusive" and final["first_bad_sha"] == shas[9]
    assert final["observations"][shas[5]] == rounds.SKIP and final["observations"][shas[6]] == rounds.SKIP
    assert all(code in (0, 1, 125) for code in final["observations"].values())


def test_when_git_can_only_say_the_culprit_is_among_skipped_commits_nothing_is_claimed(tmp_path):
    shas = new_repo(tmp_path / "r", 12)
    repo = tmp_path / "r"
    prepared, _ = drive(repo, [spec(shas[0], shas[11])], culprit_after(repo, shas[6]), unbuildable={shas[5], shas[6]})
    [final] = prepared["results"]
    assert final["status"] == "inconclusive" and "first_bad_sha" not in final
    assert "Only skipped commits remain" in final["reason"]
    assert shas[6] in final["candidates"] and shas[5] in final["candidates"]


def test_the_round_limit_ends_a_spec_inconclusively(tmp_path):
    shas = new_repo(tmp_path / "r", 30)
    prepared, measured = drive(tmp_path / "r", [spec(shas[0], shas[29])], culprit_after(tmp_path / "r", shas[17]),
                               max_rounds=2)
    [final] = prepared["results"]
    assert final["status"] == "inconclusive" and "max_rounds" in final["reason"] and len(measured) == 2


def test_a_candidate_round_records_only_what_the_candidate_measured(tmp_path):
    shas = new_repo(tmp_path / "r", 10)
    prepared = rounds.prepare({"specs": [spec(shas[0], shas[9])]}, tmp_path / "r")
    candidate = prepared["candidate_sha"]
    cases = [({"status": "noisy", "runs": []}, 125), ({"status": "error", "runs": []}, 125),
             ({"status": "build_failed"}, 125), (None, 125), (ok(100), 0), (ok(50), 1),
             ({**ok(100), "values": [100, 50]}, 125)]
    for measurement, code in cases:
        base = {"good": ok(100), "bad": ok(50)}
        if measurement is not None:
            base["candidate"] = measurement
        nxt = rounds.advance(prepared, base)
        assert nxt["observations"] == {candidate: code} and nxt["round"] == 1 and nxt["verified"] is True
        assert nxt["index"] == 0 and nxt["log"][0]["candidate"] == candidate


def test_contradictory_measurements_are_inconclusive(tmp_path):
    shas = new_repo(tmp_path / "r", 12)
    state = {"specs": [spec(shas[0], shas[11])], "observations": {shas[5]: 1, shas[8]: 0}}
    prepared = rounds.prepare(state, tmp_path / "r")
    assert prepared["finish"]["status"] == "inconclusive" and "monotone" in prepared["finish"]["reason"]
    assert prepared["commits"] == [] and prepared["needs_measurement"] is False
    assert rounds.advance(prepared, {})["results"][0]["status"] == "inconclusive"


def test_specs_that_are_not_on_main_history_never_reach_git_bisect(tmp_path):
    repo = tmp_path / "r"
    shas = new_repo(repo, 8)
    git(repo, "checkout", "-q", "-b", "off", shas[3])
    off = commit(repo, "off main")
    git(repo, "checkout", "-q", "main")
    bad_specs = [
        spec(shas[0], off),
        spec(shas[6], shas[2]),
        spec("abc", shas[5]),
        spec(shas[0], shas[0]),
        spec(shas[0], shas[5] + "\n"),
        spec(shas[0], "f" * 40),
        spec(shas[0], shas[5], required_ratio=1.5), {k: v
                                                     for k, v in spec(shas[0], shas[5]).items()
                                                     if k != "key"}
    ]
    for bad in bad_specs:
        prepared = rounds.prepare({"specs": [bad]}, repo)
        assert prepared["finish"]["status"] == "inconclusive" and prepared["commits"] == [], bad
        assert rounds.advance(prepared, {})["index"] == 1


def test_endpoints_and_candidate_must_come_from_one_gpu(tmp_path):
    shas = new_repo(tmp_path / "r", 10)
    prepared = rounds.prepare({"specs": [spec(shas[0], shas[9])]}, tmp_path / "r")
    mixed = {"good": ok(100, gpu_uuid="A"), "bad": ok(50, gpu_uuid="A"), "candidate": ok(50, gpu_uuid="B")}
    final = rounds.advance(prepared, mixed)["results"][0]
    assert final["status"] == "inconclusive" and "same GPU" in final["reason"]


def test_measurements_without_gpu_identity_cannot_attribute_a_commit(tmp_path):
    shas = new_repo(tmp_path / "r", 2)
    prepared = rounds.prepare({"specs": [spec(shas[0], shas[1])]}, tmp_path / "r")
    missing = ok(100)
    del missing["gpu_uuid"]
    final = rounds.advance(prepared, {"good": missing, "bad": ok(50)})["results"][0]
    assert final["status"] == "inconclusive" and "first_bad_sha" not in final


def test_prepare_is_deterministic_and_leaves_the_checkout_untouched(tmp_path):
    repo = tmp_path / "r"
    shas = new_repo(repo, 15)
    before = (git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain"), git(repo, "for-each-ref"))
    state = {"specs": [spec(shas[0], shas[14])], "observations": {shas[7]: 1, shas[3]: 0, shas[5]: 125}}
    first, second = rounds.prepare(state, repo), rounds.prepare(json.loads(json.dumps(state)), repo)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert (git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain"), git(repo, "for-each-ref")) == before
    assert not [p for p in (repo / ".git").iterdir() if "BISECT" in p.name]
    assert first["candidate_sha"] not in state["observations"]


def test_cli_round_trip_reaches_the_same_conclusion(tmp_path):
    repo = tmp_path / "r"
    shas = new_repo(repo, 9)
    slow = culprit_after(repo, shas[4])
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"specs": [spec(shas[0], shas[8])]}))
    for _ in range(20):
        prepared_file, measured_file = tmp_path / "prepared.json", tmp_path / "measured.json"
        assert rounds.main(["prepare", "--state",
                            str(state_file), "--source",
                            str(repo), "--out",
                            str(prepared_file)]) == 0
        prepared = json.loads(prepared_file.read_text())
        if prepared["done"]:
            break
        measured_file.write_text(
            json.dumps({
                name: ok(50 if slow("k", sha) else 100)
                for name, sha in zip(("good", "bad", "candidate"), prepared["commits"])
            }))
        assert rounds.main(
            ["advance", "--state",
             str(prepared_file), "--measurements",
             str(measured_file), "--out",
             str(state_file)]) == 0
    [final] = prepared["results"]
    assert final["status"] == "conclusive" and final["first_bad_sha"] == shas[4]
    state_file.write_text(json.dumps({"specs": [spec(shas[0], shas[8])]}))
    rounds.main(["prepare", "--state", str(state_file), "--source", str(repo), "--out", str(prepared_file)])
    with pytest.raises(SystemExit):  # a round that measures commits cannot advance without measurements
        rounds.main(["advance", "--state", str(prepared_file), "--out", str(state_file)])
