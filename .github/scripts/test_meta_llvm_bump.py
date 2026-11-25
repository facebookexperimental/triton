import hashlib
import json

import pytest

from meta_llvm_bump import (bot_branch_for, commit_message, failed_body, issue_body, issue_marker, issue_release,
                            issue_result, issue_title, pick_tracking, pr_body, parse_sha256_file, release_for, render,
                            same_tree_chain, variants, verdict, verify_state)

OLD = "1889ed344a2ea019320c1a49fc5b7103d0d9468d"
NEW = "3dd0e43fed8836e77165a4f066b983efe1f13a82"
SYNC = "aaaaaaaa00000000000000000000000000000000"
BASE = "5f07f818b51b786b0a87b4e514882600ecba112f"
META = {
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
    assert out["release_base_url"].endswith("/releases/download/llvm-3dd0e43f-1")
    assert list(out["sha256sum"]) == ["almalinux-x64", "ubuntu-x64"]


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

SHA = "a" * 40
RUN = {
    "id": 7, "path": ".github/workflows/gpu-tests.yml", "status": "completed", "conclusion": "success", "html_url":
    "https://github.com/r/actions/runs/7"
}


def test_pr_body_markdown_unwrapped_linked_fork_changes():
    release = _release(NEW, 1, ["almalinux-x64"])
    new = render(META, release, {"almalinux-x64": "e" * 64}, "facebookexperimental/llvm-project")
    long_subject = "[triton][3.8][LLVM] Cap SLSR candidate-basis SCEV differences per function"
    body = pr_body("facebookexperimental/triton", "release/3.8.x", "facebookexperimental/llvm-project", "triton_3.8.x",
                   META, new, [(NEW, f"{long_subject} (#13)"), (BASE, "Direct push")], RELEASE, SHA,
                   {"gpu-tests.yml": RUN}, [("build / build", "success"), ("b200 / b200-tlx-test", "success")])
    lines = body.splitlines()
    assert lines[:4] == ["# Summary:", "", "**Problem**", ""]
    # One line per paragraph: GitHub renders every newline as a hard break.
    assert any(
        line.startswith("facebookexperimental/triton@release/3.8.x tracks") and line.endswith("release.")
        for line in lines)
    solution = lines.index("**Solution**")
    assert "at llvm-3dd0e43f-1 (published 2026-10-15), built from" in lines[solution + 2]
    compare = f"https://github.com/facebookexperimental/llvm-project/compare/{META['llvm_hash']}...{NEW}"
    assert lines[solution + 2].endswith(f". It adds [2 fork changes]({compare}):")
    fork = "https://github.com/facebookexperimental/llvm-project"
    # Every fork commit links itself; squash-merged ones also link their fork PR.
    assert lines[solution + 4:solution + 6] == [
        f"- [`{NEW[:12]}`]({fork}/commit/{NEW}) [#13]({fork}/pull/13) {long_subject}",
        f"- [`{BASE[:12]}`]({fork}/commit/{BASE}) Direct push",
    ]
    plan = lines.index("# Test Plan:")
    assert lines[plan + 2] == ("GPU tests passed on the bump commit `aaaaaaaaaaaa`: "
                               "[gpu-tests.yml](https://github.com/r/actions/runs/7).")
    assert lines[plan + 4:plan + 8] == ["```", "build / build         success", "b200 / b200-tlx-test  success", "```"]
    assert lines[-1].startswith("The pinned digests match")


def test_pr_body_without_fork_changes():
    release = _release(NEW, 1, ["almalinux-x64"])
    new = render(META, release, {"almalinux-x64": "e" * 64}, "facebookexperimental/llvm-project")
    body = pr_body("facebookexperimental/triton", "release/3.8.x", "facebookexperimental/llvm-project", "triton_3.8.x",
                   META, new, [], RELEASE, SHA, {"gpu-tests.yml": RUN}, [])
    assert "fork changes](" in body and ").\n\n# Test Plan:" in body


def test_issue_body_has_markers_result_and_self_contained_command():
    body = issue_body("facebookexperimental/triton", "release/3.8.x",
                      "[Meta LLVM] Bump release/3.8.x to llvm-3dd0e43f-1", "# Summary:\n\n```\nlog\n```", RELEASE, SHA,
                      {"gpu-tests.yml": RUN}, "facebookexperimental/llvm-project", [(NEW, "Fix (#13)")], "https://c",
                      [("b200 / b200-tlx-test", "success")])
    assert body.startswith(issue_marker("release/3.8.x"))
    assert issue_result(body) == ("ready", SHA)
    assert ("gh pr create -R facebookexperimental/triton --base release/3.8.x "
            "--head scratch/meta-llvm-bump/release-3.8.x") in body
    # The PR body rides in a quoted heredoc, inside a fence longer than the body's own fences.
    assert "--body-file - <<'PR_BODY'\n# Summary:\n\n```\nlog\n```\nPR_BODY\n````" in body
    # Fork changes and jobs render on the issue page, outside the command's fence.
    rendered = body.split("````")[0]
    fork = "https://github.com/facebookexperimental/llvm-project"
    assert (f"It adds [1 fork change](https://c):\n\n"
            f"- [`{NEW[:12]}`]({fork}/commit/{NEW}) [#13]({fork}/pull/13) Fix\n") in rendered
    assert ("GPU runs: [gpu-tests.yml](https://github.com/r/actions/runs/7).\n\n```\nb200 / b200-tlx-test  success\n```"
            in rendered)
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
                      "It pins llvm-1889ed34-1, but the fork branch has a newer published release.", RELEASE, SHA, {},
                      "facebookexperimental/llvm-project", [], "https://c", [])
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


BRANCH_PLATFORMS = ["almalinux-arm64", "almalinux-x64", "ubuntu-arm64", "ubuntu-x64"]
BRANCH_TREE = "b" * 40


def _branch_tip(sha=NEW):
    return {"sha": sha, "commit": {"tree": {"sha": BRANCH_TREE}}}


def _branch_bundle(kind, sha=NEW):
    prefix = "llvm" if kind == "core" else "amd-llvm"
    stem = f"{prefix}-{sha[:8]}"
    record = {
        "schema_version": 1,
        "kind": kind,
        "llvm_repository": "facebookexperimental/llvm-project",
        "llvm_sha": sha,
        "llvm_tree": BRANCH_TREE,
        "build_number": 1,
    }
    content = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    assets = [{"name": f"{stem}-1.source.json", "digest": "sha256:" + hashlib.sha256(content.encode()).hexdigest()}]
    for platform in BRANCH_PLATFORMS:
        name = f"{stem}-{platform}-1.tar.gz"
        digest = hashlib.sha256(name.encode()).hexdigest()
        checksum = f"{digest}  {name}\n"
        assets += [{"name": name, "digest": f"sha256:{digest}"},
                   {"name": f"{name}.sha256", "digest": "sha256:" + hashlib.sha256(checksum.encode()).hexdigest()}]
    for asset_id, asset in enumerate(assets, 1):
        asset.update(id=asset_id, state="uploaded", size=1)
    return assets


@pytest.mark.parametrize("status", [404, 502])
def test_release_lookup_only_not_found_means_absent(monkeypatch, status):
    import subprocess
    import meta_llvm_assets

    result = subprocess.CompletedProcess([], 1, "", f"gh: request failed (HTTP {status})")
    monkeypatch.setattr(meta_llvm_assets.subprocess, "run", lambda *args, **kwargs: result)
    if status == 404:
        assert meta_llvm_assets.release_assets("facebookexperimental/triton", "meta-llvm") == (None, [])
    else:
        with pytest.raises(subprocess.CalledProcessError):
            meta_llvm_assets.release_assets("facebookexperimental/triton", "meta-llvm")


@pytest.mark.parametrize("current_state", ["starter", "uploaded"])
def test_recovery_never_deletes_uploaded_or_unrelated_assets(monkeypatch, current_state):
    import meta_llvm_assets

    published = dict(_branch_bundle("core")[0], id=101)
    pending = dict(_branch_bundle("core")[2], id=102, state="starter", size=0, digest=None)
    unrelated = dict(_branch_bundle("amd")[0], id=103, state="starter", size=0, digest=None)
    current = dict(
        _branch_bundle("core")[2], id=102, state=current_state, size=0 if current_state == "starter" else 1,
        digest=None if current_state == "starter" else _branch_bundle("core")[2]["digest"])
    calls = []

    def github(path, **kwargs):
        calls.append((path, kwargs))
        if kwargs.get("method") == "DELETE":
            return None
        return {101: published, 102: current, 103: unrelated}[int(path.rsplit("/", 1)[1])]

    monkeypatch.setattr(meta_llvm_assets, "github_json", github)
    meta_llvm_assets.recover_starters("facebookexperimental/triton", [published, pending, unrelated],
                                      {published["name"], pending["name"]})
    path = "repos/facebookexperimental/triton/releases/assets/102"
    expected = [(path, {})]
    if current_state == "starter":
        expected.append((path, {"method": "DELETE"}))
    assert calls == expected


def _ready_fixture(kind):
    assets = _branch_bundle(kind)
    prefix = 'llvm' if kind == 'core' else 'amd-llvm'
    receipt = {
        'schema_version': 1,
        'source': {
            'schema_version': 1, 'kind': kind, 'llvm_repository': 'facebookexperimental/llvm-project', 'llvm_sha': NEW,
            'llvm_tree': BRANCH_TREE, 'build_number': 1
        },
        'sha256sum': {
            platform:
            next(a
                 for a in assets
                 if a['name'] == f'{prefix}-{NEW[:8]}-{platform}-1.tar.gz')['digest'].removeprefix('sha256:')
            for platform in BRANCH_PLATFORMS
        },
        'build_run': 'https://github.com/facebookexperimental/triton/actions/runs/123',
        'publication_attempt': 1,
        'workflow_sha': 'c' * 40,
    }
    receipt['sdk_validation'] = {
        platform: {
            'schema_version': 1, 'source': receipt['source'], 'platform': platform, 'sha256': digest, 'run':
            receipt['build_run'], 'attempt': 1, 'workflow_sha': 'c' * 40
        }
        for platform, digest in receipt['sha256sum'].items()
    }
    assets.append({'name': f'{prefix}-{NEW[:8]}-1.ready.json', 'id': 99, 'state': 'uploaded', 'size': 1})
    for asset in assets:
        asset[
            'browser_download_url'] = f"https://github.com/facebookexperimental/triton/releases/download/meta-llvm/{asset['name']}"
    candidate = {'kind': kind, 'llvm_sha': NEW, 'llvm_tree': BRANCH_TREE, 'build_number': '1'}
    return assets, receipt, candidate


@pytest.mark.parametrize('kind', ['core', 'amd'])
def test_ready_issue_requires_complete_validated_publication(monkeypatch, kind):
    import meta_llvm_ready as ready
    from select_meta_llvm_builds import pending_builds

    assets, receipt, candidate = _ready_fixture(kind)
    writes = []
    marker = ready.issue_marker(kind, NEW, 1)
    existing_url = 'https://github.com/facebookexperimental/triton/issues/7'

    def api(path, **kwargs):
        if '/branches/' in path:
            return {'commit': _branch_tip()}
        return [[{'number': 7, 'body': marker, 'html_url': existing_url}]]

    monkeypatch.setattr(ready, 'github_json', api)
    monkeypatch.setattr(ready, 'read_json_asset', lambda *args: receipt)
    monkeypatch.setattr(ready.subprocess, 'run', lambda *args, **kwargs: writes.append(args))
    monkeypatch.setattr(ready, 'release_assets', lambda *args: ({'id': 1}, assets))
    assert pending_builds(ready.REPOSITORY, [candidate], assets, BRANCH_PLATFORMS) == []
    assert ready.ensure_issue(ready.REPOSITORY, candidate, BRANCH_PLATFORMS, {'url': None}) == existing_url

    for missing in assets:
        partial = [a for a in assets if a is not missing]
        variants = [partial]
        for digest in (None, 'sha256:' + hashlib.sha256(b'').hexdigest()):
            variants.append(partial + [dict(missing, state='starter', size=0, digest=digest)])
        for incomplete in variants:
            expected = [{'kind': kind, 'llvm_sha': NEW, 'build_number': '1'}]
            assert pending_builds(ready.REPOSITORY, [candidate], incomplete, BRANCH_PLATFORMS) == expected
            monkeypatch.setattr(ready, 'release_assets', lambda *args: ({'id': 1}, incomplete))
            with pytest.raises(ready.BundleNotReady):
                ready.ensure_issue(ready.REPOSITORY, candidate, BRANCH_PLATFORMS, {'url': None})
    assert writes == []


@pytest.mark.parametrize('closed', [False, True])
def test_readiness_issue_reuses_candidate_after_timeout_or_close(monkeypatch, closed):
    import subprocess
    import meta_llvm_ready as ready
    assets, receipt, candidate = _ready_fixture('amd')
    marker = ready.issue_marker('amd', NEW, 1)
    saved = {
        'number': 7, 'body': marker, 'state': 'closed' if closed else 'open', 'html_url':
        'https://github.com/facebookexperimental/triton/issues/7'
    }
    pages = [[], []]
    calls = []

    def api(path, **kwargs):
        if '/branches/' in path: return {'commit': _branch_tip()}
        if '/labels/' in path: return {'name': ready.LABEL}
        if kwargs.get('method') == 'POST':
            result = create(['gh', 'api', path], input=json.dumps(kwargs['payload']))
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        return pages

    def create(command, **kwargs):
        calls.append(command)
        pages[1].append(saved)
        assert marker in kwargs['input'] and NEW in kwargs['input']
        return subprocess.CompletedProcess(command, 1, '', 'timeout after creation')

    monkeypatch.setattr(ready, 'github_json', api)
    monkeypatch.setattr(ready, 'read_json_asset', lambda *args: receipt)
    monkeypatch.setattr(ready, 'release_assets', lambda *args: ({'id': 1}, assets))
    monkeypatch.setattr(ready.subprocess, 'run', create)
    monkeypatch.setattr(ready, 'claim_issue_creation', lambda *args: ('request.json', {'candidate': marker}))
    assert ready.ensure_issue(ready.REPOSITORY, candidate, BRANCH_PLATFORMS, {'url': None}) == saved['html_url']
    assert ready.ensure_issue(ready.REPOSITORY, candidate, BRANCH_PLATFORMS, {'url': None}) == saved['html_url']
    assert len(calls) == 1


def test_unresolved_issue_creation_does_not_submit_again(monkeypatch):
    import meta_llvm_ready as ready
    assets, receipt, candidate = _ready_fixture('amd')
    marker = ready.issue_marker('amd', NEW, 1)
    intent = {'schema_version': 1, 'candidate': marker}
    name = ready.ready_name('amd', NEW, 1).replace('.ready.json', '.issue-request.json')
    assets.append({'name': name, 'id': 100, 'state': 'uploaded', 'size': 1})

    def api(path, **kwargs):
        if '/branches/' in path: return {'commit': _branch_tip()}
        if '/labels/' in path: return {'name': ready.LABEL}
        return [[]]

    monkeypatch.setattr(ready, 'github_json', api)
    monkeypatch.setattr(ready, 'read_json_asset', lambda repo, asset: intent if asset['name'] == name else receipt)
    monkeypatch.setattr(ready, 'release_assets', lambda *args: ({'id': 1}, assets))
    writes = []
    monkeypatch.setattr(ready.subprocess, 'run', lambda *args, **kwargs: writes.append(args))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='unresolved'):
            ready.ensure_issue(ready.REPOSITORY, candidate, BRANCH_PLATFORMS, {'url': None})
    assert writes == []


def test_validation_attempts_survive_publication_retry(monkeypatch, tmp_path):
    import subprocess
    import meta_llvm_ready as ready
    assets, receipt, candidate = _ready_fixture('amd')
    assets = [asset for asset in assets if not asset['name'].endswith('.ready.json')]
    for index, platform in enumerate(BRANCH_PLATFORMS):
        archive = f'amd-llvm-{NEW[:8]}-{platform}-1.tar.gz'
        digest = receipt['sha256sum'][platform]
        (tmp_path / (archive + '.sha256')).write_text(f'{digest}  {archive}\n')
        validation = dict(receipt['sdk_validation'][platform], attempt=1 + index % 2)
        (tmp_path / (archive + '.validation.json')).write_text(json.dumps(validation))
    monkeypatch.setattr(ready, 'release_assets', lambda *args: ({'id': 1}, assets))
    monkeypatch.setattr(ready, 'recover_starters', lambda *args: assets)
    monkeypatch.setattr(ready.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    ready.publish_receipt(ready.REPOSITORY, receipt['source'], BRANCH_PLATFORMS, tmp_path, receipt['build_run'], 3,
                          'c' * 40)
    saved = json.loads((tmp_path / ready.ready_name('amd', NEW, 1)).read_text())
    assert saved['publication_attempt'] == 3
    assert [saved['sdk_validation'][p]['attempt'] for p in BRANCH_PLATFORMS] == [1, 2, 1, 2]


def test_failed_intent_upload_cannot_adopt_an_earlier_claim(monkeypatch):
    import subprocess
    import meta_llvm_ready as ready
    marker = ready.issue_marker('amd', NEW, 1)
    intent = {'schema_version': 1, 'candidate': marker}
    name = ready.ready_name('amd', NEW, 1).replace('.ready.json', '.issue-request.json')
    earlier = {'name': name, 'id': 100, 'state': 'uploaded', 'size': 1}
    snapshots = iter([({'id': 1}, []), ({'id': 1}, [earlier])])
    monkeypatch.setattr(ready, 'release_assets', lambda *args: next(snapshots))
    monkeypatch.setattr(ready, 'read_json_asset', lambda *args: intent)
    calls = []

    def upload(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '', 'gh: duplicate asset (HTTP 422)')

    monkeypatch.setattr(ready.subprocess, 'run', upload)
    with pytest.raises(subprocess.CalledProcessError):
        ready.claim_issue_creation(ready.REPOSITORY, 'amd', NEW, 1, marker)
    assert len(calls) == 1 and calls[0][:3] == ['gh', 'release', 'upload']


def test_explicit_validation_is_not_skipped_for_a_complete_bundle(monkeypatch):
    import meta_llvm_ready as ready
    from select_meta_llvm_builds import pending_builds
    assets, receipt, candidate = _ready_fixture('amd')
    monkeypatch.setattr(ready, 'read_json_asset', lambda *args: receipt)
    assert pending_builds(ready.REPOSITORY, [candidate], assets, BRANCH_PLATFORMS) == []
    assert pending_builds(ready.REPOSITORY, [candidate], assets, BRANCH_PLATFORMS,
                          validate_existing=True) == [{'kind': 'amd', 'llvm_sha': NEW, 'build_number': '1'}]


def test_closed_candidate_is_not_recreated(monkeypatch):
    import meta_llvm_ready as ready
    assets, receipt, candidate = _ready_fixture('amd')
    monkeypatch.setattr(ready, 'validated_candidate', lambda *args: ({a['name']: a for a in assets}, receipt))
    monkeypatch.setattr(ready, 'read_manifest', lambda *args: dict(META))
    calls = []

    def api(path, **kwargs):
        calls.append(kwargs)
        if path.endswith('/commits/main'):
            return {'sha': 'b' * 40}
        if 'state=open' in path:
            return [[]]
        assert 'state=closed' in path
        return [[{'number': 7, 'state': 'closed'}]]

    monkeypatch.setattr(ready, 'github_json', api)
    assert ready.ensure_candidate_pr(ready.REPOSITORY, candidate, BRANCH_PLATFORMS) is None
    assert all('method' not in call for call in calls)


def test_newer_candidate_closes_only_older_drafts(monkeypatch):
    import meta_llvm_ready as ready
    old, kept = '1' * 40, '2' * 40
    pulls = [{
        'number': n, 'draft': draft, 'head': {'ref': f'scratch/llvm-canary/amd-{sha[:8]}-1'}, 'body':
        f'<!-- llvm-candidate beta/amd/{sha}/1/{"a" * 40} -->'
    } for n, sha, draft in ((1, old, True), (2, kept, False))]
    issues = [{'number': 10 + n, 'body': ready.issue_marker('amd', sha, 1)} for n, sha in ((1, old), (2, kept))]
    closed = []

    def api(path, method=None, payload=None, **kwargs):
        if method == 'PATCH':
            closed.append(int(path.rsplit('/', 1)[1]))
        if method:
            return {}
        return [issues] if 'labels=' in path else [pulls]

    monkeypatch.setattr(ready, 'github_json', api)
    ready.close_older_drafts(ready.REPOSITORY, 'amd', f'scratch/llvm-canary/amd-{NEW[:8]}-1', 9)
    assert closed == [1, 11]
