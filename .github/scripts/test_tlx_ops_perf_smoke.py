import argparse
import base64
import hashlib
import json
import zipfile

import pytest

import tlx_ops_perf_smoke_ab as ab
import tlx_ops_perf_smoke_comment as comment

KEY = "mm/sm90/float16/1024x1024x1024"


def case(key=KEY, tflops=100.0, status="ok", flop_count=2e9, notes=()):
    return {"key": key, "status": status, "tflops": tflops, "flop_count": flop_count, "notes": list(notes)}


def results(base_rounds, head_rounds, env=None):
    """results.json with one run per (side, round); each round is a list of cases (None = the run failed)."""
    runs = []
    for side, rounds in (("base", base_rounds), ("head", head_rounds)):
        for i, cases in enumerate(rounds, 1):
            runs.append({"side": side, "round": i, "cases": cases or [], "error": None if cases else "boom"})
    return {"runs": runs, "env": env or {}}


def rows_by_key(res, threshold=0.05):
    return {r["key"]: r for r in ab.compare(res, threshold)}


def test_consistent_slowdown_is_flagged():
    res = results([[case(tflops=100)]] * 3, [[case(tflops=93)], [case(tflops=94)], [case(tflops=92)]])
    row = rows_by_key(res)[KEY]
    assert row["state"] == "slower"
    assert row["ratio"] == pytest.approx(0.93)
    assert (row["ratio_min"], row["ratio_max"]) == (pytest.approx(0.92), pytest.approx(0.94))
    assert ab.verdict_for(list(rows_by_key(res).values()), True) == "INVESTIGATE"


def test_slowdown_that_one_round_contradicts_is_noise():
    # Median 0.93 but one round came out faster: the rounds disagree, so nothing is claimed.
    res = results([[case(tflops=100)]] * 3, [[case(tflops=93)], [case(tflops=101)], [case(tflops=90)]])
    row = rows_by_key(res)[KEY]
    assert row["state"] == "same"
    assert ab.verdict_for([row], True) == "PASS"


def test_small_consistent_change_is_below_threshold():
    res = results([[case(tflops=100)]] * 3, [[case(tflops=97)]] * 3)
    assert rows_by_key(res)[KEY]["state"] == "same"


def test_consistent_speedup_is_reported_but_passes():
    res = results([[case(tflops=100)]] * 3, [[case(tflops=110)], [case(tflops=108)], [case(tflops=112)]])
    row = rows_by_key(res)[KEY]
    assert row["state"] == "faster"
    assert ab.verdict_for([row], True) == "PASS"


def test_ratio_pairs_rounds_not_pooled_medians():
    # Round 2 ran on a hot GPU for both sides; pairing by round keeps it from looking like a change.
    res = results([[case(tflops=100)], [case(tflops=80)], [case(tflops=100)]],
                  [[case(tflops=100)], [case(tflops=80)], [case(tflops=100)]])
    row = rows_by_key(res)[KEY]
    assert row["ratio"] == pytest.approx(1.0) and row["state"] == "same"


def test_latency_comes_from_flop_count():
    res = results([[case(tflops=100, flop_count=2e9)]] * 2, [[case(tflops=50, flop_count=2e9)]] * 2)
    row = rows_by_key(res)[KEY]
    assert row["base_us"] == pytest.approx(20.0)
    assert row["head_us"] == pytest.approx(40.0)


def test_head_only_error_is_a_problem():
    res = results([[case()]] * 2, [[case(status="error", tflops=None, notes=["wrong answer"])]] * 2)
    row = rows_by_key(res)[KEY]
    assert row["state"] == "head-error" and row["notes"] == ["wrong answer"]
    assert ab.verdict_for([row], True) == "INVESTIGATE"


def test_failures_collapse_per_op_and_message():
    err = "RuntimeError: PassManager::run failed: descriptor block and tensor element types must match"
    keys = [f"flash_attn_mxfp8/sm100/bfloat16/case{i}" for i in range(3)]
    res = results([[case(key=k) for k in keys + [KEY]]] * 2,
                  [[case(key=k, status="error", tflops=None, notes=[err])
                    for k in keys] + [case(status="error", tflops=None, notes=["wrong answer"])]] * 2)
    rows = ab.compare(res, 0.05)
    assert ab.findings(rows, 0.05) == [
        f"- **flash_attn_mxfp8**: 3 of 3 kernels fail on this PR but run on base: `{err}`",
        "- **mm**: 1 of 1 kernel fail on this PR but run on base: `wrong answer`",
    ]
    assert ab.headline(rows, ab.verdict_for(rows, True), True) == ("❌", "4 kernels fail on this PR")
    groups = {g["group"]: g for g in ab.summarize_groups(rows)}
    assert groups["flash_attn_mxfp8/sm100"]["problems"] == 3 and groups["flash_attn_mxfp8/sm100"]["cases"] == 0
    assert len(ab.shorten("x" * 500)) == 300


def test_changes_read_as_words():
    assert [ab.change_text(r) for r in (0.81, 1.06, 0.997, None)] == ["19% slower", "6% faster", "no change", "-"]
    # No "-0%": a spread that rounds to zero reads 0%, and an all-equal spread is one value.
    assert ab.range_text(0.997, 1.004) == "0%"
    assert ab.range_text(0.92, 1.08) == "−8% to +8%"
    assert ab.range_words(0.92, 1.08) == "from 8% slower to 8% faster"
    assert ab.within([0.99, 1.004]) == "within ±1%" and ab.within([0.97, 1.02]) == "within ±3%"


def test_noisy_move_is_called_noise_not_a_change():
    other = "flash_attn/sm90/bfloat16/4x48x1024x128xFalse"
    res = results([[case(), case(key=other)]] * 3,
                  [[case(tflops=92), case(key=other)], [case(tflops=108), case(key=other)],
                   [case(tflops=94), case(key=other)]])
    rows = ab.compare(res, 0.05)
    assert ab.headline(rows, ab.verdict_for(rows, True), True) == ("✅", "no slowdowns found")
    assert ab.findings(rows, 0.05) == [
        f"All 2 kernels are within ±1% of base, except `{KEY}` (6% slower, but its runs ranged from 8% slower to "
        "8% faster, so that is noise)."
    ]


def test_faster_only_still_passes():
    res = results([[case()]] * 3, [[case(tflops=110)], [case(tflops=109)], [case(tflops=111)]])
    rows = ab.compare(res, 0.05)
    assert ab.headline(rows, ab.verdict_for(rows, True), True) == ("✅", "no slowdowns, 1 kernel faster")
    assert ab.group_result(rows) == "✅ 1 faster, by 10%"


def test_nightly_age():
    assert ab.nightly_age("20261117", "20261112") == "; it is 5 days older than the base commit"
    assert ab.nightly_age("20261117", "20261116") == ""
    assert ab.nightly_age("20261117", "20261118") == "; it is 1 day newer than the base commit"
    assert ab.nightly_age("", "20261118") == ""


def test_base_only_error_is_not_compared_and_passes():
    res = results([[case(status="error", tflops=None)]] * 2, [[case()]] * 2)
    row = rows_by_key(res)[KEY]
    assert row["state"] == "base-error"
    assert ab.verdict_for([row, {**row, "state": "same", "ratio": 1.0}], True) == "PASS"


def test_cases_on_one_side_only():
    other = "norm_fwd/sm90/bfloat16/4096x8192"
    res = results([[case(), case(key=other)]] * 2, [[case(), case(key=other + "x2")]] * 2)
    rows = rows_by_key(res)
    assert rows[other]["state"] == "only-base"
    assert rows[other + "x2"]["state"] == "only-head"


def test_failed_round_drops_out_of_the_pairs():
    res = results([[case(tflops=100)], None, [case(tflops=100)]], [[case(tflops=90)]] * 3)
    row = rows_by_key(res)[KEY]
    assert row["pairs"] == 2 and row["state"] == "slower"


def test_one_pair_is_not_enough():
    res = results([[case()], None], [[case(tflops=50)]] * 2)
    assert rows_by_key(res)[KEY]["state"] == "unmeasured"


def test_groups_summarize_per_op_and_arch():
    norm = "norm_fwd/sm90/bfloat16/4096x8192"
    res = results([[case(), case(key=norm)]] * 3, [[case(tflops=90), case(key=norm, tflops=100)]] * 3)
    groups = {g["group"]: g for g in ab.summarize_groups(ab.compare(res, 0.05))}
    assert groups["mm/sm90"]["slower"] == 1 and groups["mm/sm90"]["geomean"] == pytest.approx(0.9)
    assert groups["norm_fwd/sm90"]["slower"] == 0 and groups["norm_fwd/sm90"]["cases"] == 1


def test_pick_baseline(tmp_path):
    for date in ("20261110", "20261115", "20261117"):
        (tmp_path / f"{date}.h100.json").write_text("{}")
    (tmp_path / "20261116.b200.json").write_text("{}")
    assert ab.pick_baseline(tmp_path, "20261116", "h100")[0] == "20261115"
    assert ab.pick_baseline(tmp_path, "20261101", "h100")[0] == "20261110"
    assert ab.pick_baseline(tmp_path, "20261116", "mi350") == (None, None)


def test_head_only_falls_back_to_the_nightly_thresholds():
    rows = [{"state": "only-head", "ratio": None, "nightly_ratio": r} for r in (0.95, 0.75, 0.70)]
    assert ab.verdict_for(rows, False) == "INVESTIGATE"
    rows = [{"state": "only-head", "ratio": None, "nightly_ratio": r} for r in (0.95, 0.75)]
    assert ab.verdict_for(rows, False) == "PASS"
    assert ab.verdict_for([], False) == "NO SIGNAL"


def report_args(tmp_path, **overrides):
    args = dict(results=str(tmp_path / "results.json"), platform="h100", history=str(tmp_path / "history"),
                base_date="20261117", base="a" * 40, head="b" * 40, pr_head="c" * 40, run_url="https://run",
                bench_status="success", threshold=0.05, out_md=str(tmp_path / "section.md"),
                out_json=str(tmp_path / "verdict.json"))
    args.update(overrides)
    return argparse.Namespace(**args)


def write_nightly(tmp_path, date, platform, tflops):
    history = tmp_path / "history"
    history.mkdir(exist_ok=True)
    summary = {"ops": {"mm": {"cases": [{"key": KEY, "status": "ok", "tlx_tflops": tflops}]}}}
    (history / f"{date}.{platform}.json").write_text(json.dumps(summary))


def test_report_end_to_end(tmp_path):
    env = {
        "gpu": "NVIDIA H100 80GB HBM3", "driver": "580.82.07", "cuda": "13.0", "torch": "2.14.0+cu130", "governed":
        ["persistence mode", "power cap 700 W", "clock lock 1980 MHz"]
    }
    res = results([[case(tflops=100)]] * 3, [[case(tflops=90)], [case(tflops=91)], [case(tflops=89)]], env)
    (tmp_path / "results.json").write_text(json.dumps(res))
    write_nightly(tmp_path, "20261116", "h100", 100.0)
    ab.report(report_args(tmp_path))
    md = (tmp_path / "section.md").read_text()
    # The verdict and what moved come first, in words; numbers and setup follow, collapsed.
    assert md.startswith("### ⚠️ H100: 1 kernel slower\n\n"
                         f"- **mm**: 1 of 1 kernel slower, by 10% (`{KEY}`). Every run agreed.\n")
    assert "| `mm` | 1 | ⚠️ 1 slower, by 10% |" in md
    assert f"| `{KEY}` | 20.0 µs | 22.2 µs | ⚠️ 10% slower | −11% to −9% |" in md
    assert md.index("<details><summary>All 1 kernels") < md.index("<details><summary>How this was measured")
    assert ("- **GPU:** NVIDIA H100 80GB HBM3, driver 580.82.07, CUDA 13.0, torch 2.14.0+cu130, clock locked at "
            "1980 MHz.") in md
    assert ("- **Compared:** base `aaaaaaaaaaaa` and this PR `bbbbbbbbbbbb` (GitHub's merge of PR head "
            "`cccccccccccc` into base).") in md
    assert "each kernel ran 3 times on each side" in md and "changed by 5% or more" in md
    assert f"typically differ by 2.2%; the most was 2% (`{KEY}`)." in md
    # Base is what the nightly measured, so it carries the runner context; head moved by the PR would not.
    assert "this machine measured base within 1% of the nightly `20261116.h100.json` (median over kernels)." in md
    assert "- [Workflow run](https://run)" in md
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "INVESTIGATE" and verdict["same_runner"] is True


def test_flagged_cases_collapse_per_op():
    keys = [f"flash_attn_varlen/sm90/bfloat16/case{i}" for i in range(3)]
    res = results([[case(key=k) for k in keys + [KEY]]] * 3,
                  [[case(key=k, tflops=75 + i) for i, k in enumerate(keys)] + [case(tflops=80)]] * 3)
    assert ab.findings(ab.compare(res, 0.05), 0.05) == [
        "- **flash_attn_varlen**: 3 of 3 kernels slower, by 23–25%. Every run agreed.",
        f"- **mm**: 1 of 1 kernel slower, by 20% (`{KEY}`). Every run agreed.",
    ]


def test_report_without_results_says_no_signal(tmp_path):
    ab.report(report_args(tmp_path, bench_status="failure"))
    md = (tmp_path / "section.md").read_text()
    assert md.startswith("### ⚪ H100: no result\n")
    assert "The benchmark produced no results (bench job: failure), so there is nothing to compare." in md


def test_report_head_only(tmp_path):
    res = results([], [[case(tflops=70)], [case(tflops=72)]])
    (tmp_path / "results.json").write_text(json.dumps(res))
    write_nightly(tmp_path, "20261116", "h100", 100.0)
    ab.report(report_args(tmp_path))
    md = (tmp_path / "section.md").read_text()
    assert md.startswith("### ✅ H100: no large slowdowns vs the nightly (no base to compare)\n")
    assert "1 kernel below 80% of the nightly. It takes two such kernels" in md
    assert "| Kernel | This PR | vs nightly |" in md and "29% slower" in md
    assert "because the base build did not run" in md
    assert "this machine measured this PR 29% slower than the nightly" in md
    # A manual run benches the PR head itself, so the section must not call it a merge commit.
    ab.report(report_args(tmp_path, pr_head="b" * 40))
    md = (tmp_path / "section.md").read_text()
    assert "this PR `bbbbbbbbbbbb` with the nightly" in md and "merge" not in md


def test_sanitize_drops_host_identity(tmp_path):
    artifact = {
        "env": {
            "host": "runner-17", "torch": "2.14.0+cu130", "triton": "3.9.0", "cuda": "13.0", "gpu":
            {"name": "NVIDIA B200", "driver": "580", "uuid": "GPU-123", "index":
             3}, "foreign_processes": [{"pid": 1}], "governed": {"applied": ["locked clocks"]}
        }, "results": [{
            "case": {"key": KEY, "op": "mm", "arch": "sm90", "direction": "fwd"}, "status": "ok", "tlx":
            {"mean": 100.0, "cv": 0.01}, "flop_count": 2e9, "notes": []
        }]
    }
    (tmp_path / "mm.sm90.json").write_text(json.dumps(artifact))
    cases, env, clocks = ab.sanitize_results(tmp_path)
    assert env == {
        "gpu": "NVIDIA B200", "driver": "580", "cuda": "13.0", "torch": "2.14.0+cu130", "triton": "3.9.0", "governed":
        ["locked clocks"]
    }
    assert clocks == {}
    assert cases[0]["tflops"] == 100.0 and cases[0]["flop_count"] == 2e9
    assert "runner-17" not in json.dumps([cases, env]) and "GPU-123" not in json.dumps([cases, env])


def test_env_line_reports_rocm_and_unlocked_clocks():
    env = {"gpu": "AMD Instinct MI350X", "torch": "2.14.0+rocm7.2", "cuda": None, "governed": ["NUMA node 1 (64 CPUs)"]}
    assert ab.env_line(env) == "AMD Instinct MI350X, ROCm 7.2, torch 2.14.0+rocm7.2, clock not locked (noisier)"
    assert ab.env_line({**env, "governed": ["perf determinism 1900 MHz"]}).endswith("clock locked at 1900 MHz")
    assert "clock" not in ab.env_line({**env, "governed": None})


def test_env_line_takes_the_amd_gpu_from_the_device():
    # The harness records no GPU name or driver on AMD; the bench's own device lookup fills them in.
    env = {"gpu": None, "driver": None, "torch": "2.14.0+rocm7.2", "cuda": None, "governed": ["NUMA node 0 (128 CPUs)"]}
    device = {"index": 4, "name": "MI350X", "vendor": "amd", "arch": "gfx950", "driver": "6.14.14"}
    assert ab.env_line(env, device) == ("AMD MI350X (gfx950), driver 6.14.14, ROCm 7.2, torch 2.14.0+rocm7.2, "
                                        "clock not locked (noisier)")
    nvidia = {"gpu": "NVIDIA H100 80GB HBM3", "driver": "580.82.07", "cuda": "13.0", "torch": "2.14.0+cu130"}
    assert ab.env_line(nvidia, {"name": "NVIDIA H100", "vendor": "nvidia"}).startswith("NVIDIA H100 80GB HBM3,")


def test_describe_gpu_parses_the_harness_pick(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["code"] = cmd[-1]
        return argparse.Namespace(
            stdout='log line\n{"index": 4, "name": "MI350X", "vendor": "amd", "arch": "gfx950"}\n')

    monkeypatch.setattr(ab.subprocess, "run", fake_run)
    monkeypatch.setattr(ab, "amd_driver_version", lambda: "6.14.14")
    device = ab.describe_gpu("python", ".", "4")
    assert device == {"index": 4, "name": "MI350X", "vendor": "amd", "arch": "gfx950", "driver": "6.14.14"}
    assert "select_device('4')" in seen["code"]

    def failing_run(cmd, **kwargs):
        raise ab.subprocess.CalledProcessError(1, cmd, stderr="RuntimeError: No benchmark GPU discovered")

    monkeypatch.setattr(ab.subprocess, "run", failing_run)
    assert ab.describe_gpu("python", ".", "auto") is None


def test_wheel_digest_matches_installed_file(tmp_path):
    payload = b"\x7fELF fake libtriton"
    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    wheel = tmp_path / "triton-3.9.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(ab.LIBTRITON, payload)
        zf.writestr("triton-3.9.0.dist-info/RECORD", f"{ab.LIBTRITON},{digest},{len(payload)}\n"
                    "triton-3.9.0.dist-info/RECORD,,\n")
    installed = tmp_path / "libtriton.so"
    installed.write_bytes(payload)
    assert ab.wheel_lib_digest(wheel) == digest == ab.file_digest(installed)
    installed.write_bytes(payload + b"!")
    assert ab.file_digest(installed) != digest


def test_bench_alternates_sides_on_one_gpu(tmp_path, monkeypatch):
    calls, installs = [], []

    def fake_run(args, side, rnd, wheel, gpu, out, installed=False):
        calls.append((rnd, side, gpu, installed))
        return {"side": side, "round": rnd, "cases": [case()], "error": None, "seconds": 1.0}, {"gpu": "H100"}

    monkeypatch.setattr(ab, "run_side", fake_run)
    monkeypatch.setattr(ab, "install", lambda python, wheel: installs.append(str(wheel)))
    picked = {"index": 5, "name": "NVIDIA H100", "vendor": "nvidia", "arch": None}
    monkeypatch.setattr(ab, "describe_gpu", lambda python, repo, spec: picked if spec == "auto" else None)
    args = argparse.Namespace(out=str(tmp_path), base_wheel="base.whl", head_wheel="head.whl", gpu="auto", repo=".",
                              platform="h100", rounds=3, select=ab.SMOKE_SELECT, head_cases=8)
    ab.bench(args)
    # The GPU is picked once, after the first wheel is installed (the picker imports triton), then kept.
    assert installs == ["base.whl"]
    assert calls == [(1, "base", "5", True), (1, "head", "5", False), (2, "head", "5", False), (2, "base", "5", False),
                     (3, "base", "5", False), (3, "head", "5", False)]
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["gpu_index"] == "5" and len(saved["runs"]) == 6 and saved["env"] == {"gpu": "H100"}
    assert saved["device"] == picked


def test_bench_without_base_wheel_runs_head_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ab, "run_side", lambda args, side, rnd, wheel, gpu, out, installed=False: (calls.append(
            (side, gpu)) or ({"side": side, "round": rnd, "cases": []}, {})))
    monkeypatch.setattr(ab, "install", lambda python, wheel: None)
    # An unidentified GPU keeps the requested index.
    monkeypatch.setattr(ab, "describe_gpu", lambda python, repo, spec: None)
    args = argparse.Namespace(out=str(tmp_path), base_wheel="", head_wheel="head.whl", gpu="2", repo=".",
                              platform="h100", rounds=2, select=ab.SMOKE_SELECT, head_cases=8)
    ab.bench(args)
    assert calls == [("head", "2"), ("head", "2")]


# --------------------------------------------------------------------------------------------------------------------
# comment


def test_compose_orders_sections_and_marks_old_heads():
    body = comment.merge("", "h100", "1" * 40, "### h100: PASS\n", note="b200 paused.")
    body = comment.merge(body, "b200", "2" * 40, "### b200: PASS")
    assert body.startswith(comment.MARKER + "\n## TLX.ops perf smoke (advisory)")
    assert body.index("### b200") < body.index("### h100")
    assert "From an older head, not yet re-run for this one: h100 (`111111111111`)." in body
    assert comment.parse_sections(body) == {"b200": ("2" * 40, "### b200: PASS"), "h100": ("1" * 40, "### h100: PASS")}


def test_sections_keep_the_pr_head_of_a_merge_commit():
    merge, pr_head = "1" * 40, "a" * 40
    body = comment.merge("", "h100", f"{merge} {pr_head}", "### ✅ H100: no slowdowns found")
    assert f"<!-- tlx-ops-perf-smoke:h100 head={merge} pr_head={pr_head} -->" in body
    assert comment.parse_sections(body) == {"h100": (f"{merge} {pr_head}", "### ✅ H100: no slowdowns found")}
    # A section from an older head is marked by its benched commit alone.
    body = comment.merge(body, "mi350", "2" * 40, "### ✅ MI350: no slowdowns found")
    assert f"From an older head, not yet re-run for this one: h100 (`{merge[:12]}`)." in body


def test_measured_head_is_the_head_every_measured_section_shares():
    merge, older = "1" * 40, "2" * 40
    body = comment.merge("", "h100", f"{merge} {'a' * 40}", "### ✅ H100: no slowdowns found")
    body = comment.merge(body, "mi350", merge, "### ⚠️ MI350: 1 kernel slower")
    assert comment.measured_head(body) == merge
    # A section without results does not count; sections that disagree give no head.
    assert comment.measured_head(comment.merge(body, "b200", older, "### ⚪ B200: no result")) == merge
    assert comment.measured_head(comment.merge(body, "b200", older, "### ✅ B200: no slowdowns found")) == ""
    assert comment.measured_head("") == ""


def test_note_only_keeps_every_section():
    body = comment.merge("", "h100", "1" * 40, "### ✅ H100: no slowdowns found", note="b200 paused.")
    noted = comment.set_note(body, "1" * 40, "Not re-run for PR head `bbbbbbbbbbbb`: only tests changed.")
    assert comment.parse_sections(noted) == comment.parse_sections(body)
    assert "Not re-run for PR head `bbbbbbbbbbbb`" in noted and "b200 paused." not in noted
    assert "From an older head" not in noted


def test_relevant_ignores_tests_ci_and_docs(tmp_path, capsys):
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.com")
    git("config", "user.name", "ci")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.cpp").write_text("int a;\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "python" / "test").mkdir(parents=True)
    (tmp_path / "python" / "test" / "test_a.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("hi\n")
    git("add", "-A")
    git("commit", "-qm", "tests only")
    tests_only = git("rev-parse", "HEAD")
    (tmp_path / "lib" / "a.cpp").write_text("int b;\n")
    git("commit", "-qam", "source")
    source = git("rev-parse", "HEAD")
    for head, expected in ((tests_only, "false"), (source, "true")):
        ab.main(["relevant", "--repo", str(tmp_path), "--base", base, "--head", head])
        assert capsys.readouterr().out.strip() == expected


def test_merge_replaces_only_its_own_section():
    body = comment.merge("", "b200", "1" * 40, "### b200: old")
    body = comment.merge(body, "h100", "1" * 40, "### h100: PASS")
    body = comment.merge(body, "b200", "1" * 40, "### b200: new")
    assert comment.parse_sections(body) == {"b200": ("1" * 40, "### b200: new"), "h100": ("1" * 40, "### h100: PASS")}
    assert "older head" not in body


def test_legacy_comment_is_replaced():
    legacy = comment.MARKER + "\n## TLX.ops perf smoke (advisory)\n\n### h100\n\n**Verdict: PASS**\n"
    body = comment.merge(legacy, "h100", "1" * 40, "### h100: PASS")
    assert "**Verdict: PASS**" not in body
    assert comment.parse_sections(body) == {"h100": ("1" * 40, "### h100: PASS")}


class FakeGitHub:
    """One comment; `clobber` replays a concurrent writer that read the body before this job's write."""

    def __init__(self, body=None, clobber=None):
        self.comment = {"id": 7, "body": body} if body is not None else None
        self.clobber = clobber
        self.writes = 0

    def find_comment(self, pr):
        return dict(self.comment) if self.comment else None

    def write(self, pr, existing, body):
        self.writes += 1
        stale = self.comment["body"] if self.comment else ""
        self.comment = {"id": 7, "body": body}
        if self.clobber:
            self.comment["body"], self.clobber = self.clobber(stale), None
        return self.comment


def test_publish_creates_the_comment():
    gh = FakeGitHub()
    assert comment.publish(gh, 1, "h100", "1" * 40, "### h100: PASS", sleep=lambda s: None)
    assert gh.writes == 1 and comment.has_section(gh.comment["body"], "h100", "1" * 40, "### h100: PASS")


def test_publish_restores_a_section_lost_to_a_concurrent_write():
    start = comment.merge("", "b200", "1" * 40, "### b200: running")
    # The other job read `start` before our write landed and wrote its own section on top of it.
    gh = FakeGitHub(start, clobber=lambda stale: comment.merge(stale, "b200", "1" * 40, "### b200: PASS"))
    assert comment.publish(gh, 1, "h100", "1" * 40, "### h100: PASS", sleep=lambda s: None)
    sections = comment.parse_sections(gh.comment["body"])
    assert sections == {"b200": ("1" * 40, "### b200: PASS"), "h100": ("1" * 40, "### h100: PASS")}
    assert gh.writes == 2


def test_publish_skips_the_write_when_already_current():
    gh = FakeGitHub(comment.merge("", "h100", "1" * 40, "### h100: PASS"))
    assert comment.publish(gh, 1, "h100", "1" * 40, "### h100: PASS", sleep=lambda s: None)
    assert gh.writes == 0
