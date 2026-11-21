import json
import math
import os
import re
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import tlx_ops_bench_dashboard as dash


def timed(key, tflops, M=1024, status="ok"):
    return {"key": f"mm/sm100/{key}", "status": status, "tlx_tflops": tflops, "params": {"M": M, "N": 1024, "K": 1024}}


def bench_night(date, sha, *cases, **env):
    return {
        "platform": "b200", "date": date, "sha": sha, "governed": env.pop("governed", []), "ops":
        {"mm": {"cases": list(cases)}}, **env
    }


def test_a_perf_event_needs_the_next_night_to_hold_it(monkeypatch):
    monkeypatch.setattr(dash, "commits", lambda a, b: [])
    nights = [
        bench_night("20260901", "a" * 40, timed("x", 100), timed("y", 100)),
        bench_night("20260902", "b" * 40, timed("x", 80), timed("y", 80)),
        bench_night("20260903", "c" * 40, timed("x", 80), timed("y", 100))
    ]
    ev = dash.events(nights)
    perf = ev[1]["perf"]["mm"]
    assert [r["key"] for r in perf["slower"]] == ["x"] and perf["n_faster"] == 0 and not perf["unconfirmed"]
    assert ev[2]["perf"]["mm"]["unconfirmed"] and [r["key"] for r in ev[2]["perf"]["mm"]["faster"]] == ["y"]


def test_suite_and_environment_changes_are_events(monkeypatch):
    monkeypatch.setattr(dash, "commits", lambda a, b: None)
    nights = [
        bench_night("20260901", "a" * 40, timed("x", 100), timed("y", 100), driver="570", governed=["NUMA 0"]),
        bench_night("20260902", "b" * 40, timed("x", 100), timed("y", 0, status="error"), timed("z", 100), driver="575",
                    governed=["NUMA 1"])
    ]
    ev = dash.events(nights)[1]
    suite = ev["suite"]["mm"]
    assert (suite["added"], suite["removed"], suite["broke"], suite["fixed"]) == (1, 0, 1, 0)
    assert {c["field"] for c in ev["env"]} == {"driver", "NUMA"}
    assert ev["commits"] is None and ev["perf"] == {}


def test_the_same_microseconds_lost_at_every_size_is_a_host_signature():
    host = [{"dus": 5.0, "dt": dt} for dt in (-0.5, -0.2, -0.1)]
    kernel = [{"dus": us, "dt": -0.2} for us in (1.0, 5.0, 20.0)]
    assert dash.signature(host)[0] == "host" and dash.signature(kernel)[0] == "kernel"


def test_file_rank_drops_other_ops_arches_and_backends():
    assert dash.file_rank("third_party/tlx/ops/kernels/mm/gemm_sm100.py", "mm", "b200") == 1
    assert dash.file_rank("third_party/tlx/ops/kernels/mm/gemm_gfx950.py", "mm", "b200") is None
    assert dash.file_rank("third_party/tlx/ops/kernels/flash_attn/fwd.py", "mm", "b200") is None
    assert dash.file_rank("third_party/amd/lib/foo.cpp", "mm", "b200") is None
    assert dash.file_rank("third_party/nvidia/lib/foo.cpp", "mm", "b200") == 5
    assert dash.file_rank("lib/Analysis/Alias.cpp", "mm", "b200") == 6


def test_file_rank_covers_the_shared_kernel_files():
    assert dash.file_rank("third_party/tlx/ops/kernels/_shape_suites.py", "mm", "mi350") == 2
    assert dash.file_rank("third_party/tlx/ops/kernels/__init__.py", "mm", "b200") == 3


def test_suspects_rank_by_area_and_a_host_signature_promotes_dispatch():
    found = [{"sha": "c1", "title": "compiler", "pr": 1, "files": ["lib/Analysis/Alias.cpp"]},
             {"sha": "c2", "title": "dispatch", "pr": 2, "files": ["third_party/tlx/ops/__init__.py"]},
             {"sha": "c3", "title": "kernel", "pr": 3, "files": ["third_party/tlx/ops/kernels/mm/gemm_sm100.py"]}, {
                 "sha": "c4", "title": "other op", "pr": 4, "files":
                 ["third_party/tlx/ops/kernels/flash_attn/fwd.py", "third_party/tlx/ops/_catalog.py"]
             }, {"sha": "c5", "title": "docs", "pr": 5, "files": ["README.md"]}]
    assert [c["sha"] for c in dash.suspects(found, "mm", "b200", "kernel")] == ["c3", "c2", "c1"]
    assert [c["sha"] for c in dash.suspects(found, "mm", "b200", "host")] == ["c2", "c3", "c1"]


def test_a_change_the_next_night_cannot_check_stays_unconfirmed(monkeypatch):
    monkeypatch.setattr(dash, "commits", lambda a, b: [])
    nights = [
        bench_night("20260901", "a" * 40, timed("x", 100)),
        bench_night("20260902", "b" * 40, timed("x", 80)), {**bench_night("20260903", "c" * 40), "ops": {}}
    ]
    perf = dash.events(nights)[1]["perf"]["mm"]
    assert perf["unconfirmed"] and perf["n_slower"] == 1 and perf["slower"][0]["next"] is None


def test_skipped_operator_remains_visible_alongside_measurements(tmp_path):
    raw = {
        "env": {"ref": "torch.matmul", "space": "heuristic"},
        "results": [{
            "case":
            {"op": "mm", "key": "mm/gfx950/float16/256", "input": "((), {})", "direction": "fwd", "dtype": "float16"},
            "status": "ok",
            "speedup": 1.0,
            "tlx": {"mean": 10.0},
            "ref": {"mean": 10.0},
        }],
    }
    (tmp_path / "mm.json").write_text(json.dumps(raw))
    suite = ET.Element("testsuite")
    test = ET.SubElement(suite, "testcase", name="test_op_perf[bench_addmm]")
    ET.SubElement(test, "skipped", message="no implementation")
    ET.ElementTree(suite).write(tmp_path / "bench.xml")
    out = tmp_path / "summary.json"
    dash.summarize(
        SimpleNamespace(dir=str(tmp_path), platform="mi350", version="3.9.0.dev20261029", sha="a" * 40, out=str(out),
                        ci="", ci_run=""))
    summary = json.loads(out.read_text())
    ops = summary["ops"]
    assert ops["addmm"]["reason"] == "no implementation"
    assert ops["addmm"]["cases"] == []
    assert len(ops["mm"]["cases"]) == 1
    assert ops["mm"]["cases"][0]["tlx_tflops"] == 10.0
    page = dash.page_data({"mi350": [summary]})["platforms"]["mi350"]
    assert page["ops"]["addmm"]["reason"] == "no implementation"
    assert page["ops"]["addmm"]["cases"] == []
    assert page["nights"][0]["ops"]["addmm"] == []


def test_summary_preserves_autotune_choices_and_accepts_older_artifacts(tmp_path):
    decisions = [{
        "kernel": "kernels.attention",
        "phase": "warmup",
        "key": ["64", "64", "480", "1024", "4"],
        "config": {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2, "num_ctas": 1},
        "selection": "cached",
        "selected_by_case": "serving/Z37",
        "candidates": [],
    }]
    old = {
        "case": {
            "op": "hstu_cross_attn", "key": "serving/Z37", "input": "((), {'Z': 37})", "direction": "fwd", "dtype":
            "bfloat16"
        },
        "status": "ok",
        "tlx": {"mean": 226.0, "p50": 227.0},
    }
    new = {
        **old, "case": {**old["case"], "key": "serving/Z40", "input": "((), {'Z': 40})"}, "best_config":
        "BM=128 BN=64 w4 s2", "autotune": decisions
    }
    (tmp_path / "raw.json").write_text(json.dumps({"env": {}, "results": [old, new]}))
    out = tmp_path / "summary.json"
    dash.summarize(
        SimpleNamespace(dir=str(tmp_path), platform="mi350", version="3.9.0.dev20261118", sha="a" * 40, out=str(out),
                        ci="", ci_run=""))
    cases = json.loads(out.read_text())["ops"]["hstu_cross_attn"]["cases"]
    assert cases[0]["best_config"] is None and cases[0]["autotune"] == []
    assert cases[1]["best_config"] == new["best_config"] and cases[1]["autotune"] == decisions
    assert cases[1]["tlx_tflops"] == 226.0 and cases[1]["tlx_p50"] == 227.0


def test_overview_uses_valid_comparisons_and_separates_directions():

    def case(key, ratio, direction="fwd", status="ok"):
        return {**timed(key, 100, status=status), "speedup": ratio, "ref_tflops": 100 / ratio, "direction": direction}

    night = bench_night("20261109", "a" * 40, case("a", 4), case("b", 1, status="noisy"),
                        case("error", 100, status="error"), case("bwd", 3, direction="bwd"))
    night["ops"]["mm"]["ref"] = "torch.matmul"
    night["ops"]["no_reference"] = {"cases": [case("no-ref", 9)], "ref": ""}
    overview = dash.overview_data({"b200": [night]})[0]
    rows = {(r["op"], r["direction"]): r for r in overview["rows"]}
    fwd = rows["mm", "fwd"]
    assert math.isclose(fwd["speedup"], 2)
    assert (fwd["compared"], fwd["cases"], fwd["errors"], fwd["noisy"]) == (2, 3, 1, 1)
    assert math.isclose(rows["mm", "bwd"]["speedup"], 3)
    assert rows["no_reference", "fwd"]["speedup"] is None
    assert rows["no_reference", "fwd"]["compared"] == 0
    assert overview["source"].endswith("20261109.b200.json")
    svg = ET.fromstring(dash.overview_svg([overview]))
    assert "2.00×" in list(svg.itertext())


def test_overview_does_not_fill_a_skipped_latest_night_from_history():
    older = bench_night("20261108", "a" * 40, {**timed("old", 100), "direction": "fwd", "speedup": 2, "ref_tflops": 50})
    older["ops"]["mm"]["ref"] = "torch.matmul"
    latest = bench_night("20261109", "b" * 40)
    latest["ops"]["mm"].update(ref=None, reason="no implementation")
    overview = dash.overview_data({"mi350": [latest, older]})[0]
    assert overview["date"] == "20261109"
    row = overview["rows"][0]
    assert row["speedup"] is None and row["compared"] == row["cases"] == 0
    assert row["reason"] == "no implementation"


def test_render_publishes_the_measured_overview_diagram(tmp_path):
    source = tmp_path / "summaries"
    source.mkdir()
    night = bench_night("20261109", "a" * 40)
    night["schema"] = 2
    (source / "20261109.b200.json").write_text(json.dumps(night))
    output = tmp_path / "site"
    dash.render(SimpleNamespace(source=str(source), outdir=str(output)))
    assert 'src="tlx_ops_bench_overview.svg"' in (output / "index.html").read_text()
    svg = ET.parse(output / "tlx_ops_bench_overview.svg")
    assert "B200 · 2026-11-09" in list(svg.getroot().itertext())


@pytest.mark.parametrize("input_value,group,has_reference", [
    ("mxfp8:layer:4096x256:default:q1:j0", "none", False),
    ("mxfp8:layer:4096x256:default:q1:j0", "dir", False),
    ("plain:layer:128x384:default:q1:j0", "none", True),
    ("", "none", True),
])
def test_chart_units_follow_the_selected_measurements(tmp_path, input_value, group, has_reference):
    chrome = os.environ.get("CHROME_BIN") or shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chromium is needed to check rendered chart units")
    cases = [{
        "key": f"norm_fwd/sm100/{family}",
        "params": {"input": value},
        "direction": "fwd",
        "dtype": "bf16",
        "status": "ok",
        "speedup": ratio,
        "tlx_tflops": throughput,
        "ref_tflops": reference,
        "notes": [],
    } for family, value, ratio, throughput, reference in [
        ("mxfp8", "mxfp8:layer:4096x256:default:q1:j0", None, 0.8196, None),
        ("plain", "plain:layer:128x384:default:q1:j0", 1.5, 3.0, 2.0),
    ]]
    night = bench_night("20261111", "a" * 40)
    night.update(schema=2, ops={"norm_fwd": {"ref": "torch (plain/softmax); CUDA graphs", "cases": cases}})
    source = tmp_path / "summaries"
    source.mkdir()
    (source / "20261111.b200.json").write_text(json.dumps(night))
    site = tmp_path / "site"
    dash.render(SimpleNamespace(source=str(source), outdir=str(site)))
    url = (site / "index.html").as_uri() + f"#tab=trending&platform=b200&op=norm_fwd&dir=fwd&group={group}"
    if input_value:
        url += f"&in.input={input_value}"
    result = subprocess.run([
        chrome,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-background-networking",
        f"--user-data-dir={tmp_path / 'chrome'}",
        "--virtual-time-budget=1500",
        "--dump-dom",
        url,
    ], capture_output=True, text=True, check=True, timeout=30)
    trend = "Geomean speedup per night" if has_reference else "norm_fwd: TLX TFLOP/s per night"
    if group == "dir":
        trend += ", by direction"
    chart = re.search(r'<svg[^>]*aria-label="' + re.escape(trend) + r'"[^>]*>(.*?)</svg>', result.stdout, re.S)
    assert chart is not None
    assert re.search(r'<circle[^>]*cx="[\d.]+"[^>]*cy="[\d.]+"', chart.group(1))
    if not has_reference:
        assert "0.820" in chart.group(1)
    per_case = "TLX vs reference" if has_reference else "TLX"
    assert f'aria-label="norm_fwd: {per_case} TFLOP/s per case, latest night"' in result.stdout
    assert not re.search(r'(?:cx|cy|points|x|y)="[^"]*(?:NaN|Infinity)', result.stdout)


@pytest.mark.parametrize("values,kind,start,recovered", [
    ([100, 100, 120, 120, 121], "lift", 2, None),
    ([100, 100, 80, 80, 88, 92], "regression", 2, None),
    ([100, 100, 80, 80, 92, 96, 97], "regression", 2, 5),
])
def test_episodes_compare_with_the_pre_change_level(values, kind, start, recovered):
    series = [timed("x", v) for v in values]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert len(rows) == 1
    row = rows[0]
    assert (row["kind"], row["start"], row["recovered"]) == (kind, start, recovered)
    assert row["baseline"] == 100 and row["confirmed"] == 3
    assert row["last"] == len(values) - 1


def test_recovery_is_not_a_lift_unless_it_exceeds_the_healthy_level():
    series = [timed("x", v) for v in [100, 80, 80, 100, 100]]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert len(rows) == 1 and rows[0]["recovered"] == 3
    series[-2:] = [timed("x", 120), timed("x", 120)]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert [r["kind"] for r in rows] == ["regression", "lift"]
    assert rows[1]["baseline"] == 100 and rows[1]["start"] == 3


@pytest.mark.parametrize("bad", [None, timed("x", 100, status="noisy"), timed("x", 100, status="error")])
def test_missing_or_untrusted_measurements_cannot_confirm_recovery(bad):
    series = [timed("x", 100), timed("x", 80), timed("x", 80), timed("x", 100), bad, timed("x", 100)]
    row = dash.case_episodes(series, ["same"] * len(series))[0]
    assert row["recovered"] is None and row["baseline"] == 100
    series.append(timed("x", 100))
    assert dash.case_episodes(series, ["same"] * len(series))[0]["recovered"] == 5


@pytest.mark.parametrize("values,kind", [
    ([100, 100, 100, 96, 92, 88, 84], "regression"),
    ([100, 100, 100, 104, 108, 112, 116], "lift"),
])
def test_gradual_movements_do_not_redefine_the_normal_baseline(values, kind):
    series = [timed("x", v) for v in values]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert len(rows) == 1 and rows[0]["kind"] == kind
    assert rows[0]["baseline"] == 100 and rows[0]["start"] == 5 and rows[0]["confirmed"] == 6


@pytest.mark.parametrize("values,lift_start", [
    ([100, 80, 80, 100, 120, 120], 4),
    ([100, 80, 80, 100, 100, 120, 120], 5),
])
def test_a_lift_on_the_recovery_boundary_keeps_the_healthy_baseline(values, lift_start):
    series = [timed("x", v) for v in values]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert [r["kind"] for r in rows] == ["regression", "lift"]
    assert rows[0]["recovered"] == 3
    assert rows[1]["baseline"] == 100 and rows[1]["start"] == lift_start
    assert rows[1]["confirmed"] == lift_start + 1


def test_reference_only_changes_are_not_tlx_movements():
    series = [{**timed("x", 100), "speedup": ratio, "ref_tflops": 100 / ratio} for ratio in [1, 1, 2, 2]]
    assert dash.case_episodes(series, ["same"] * len(series)) == []


def test_a_later_lift_does_not_rewrite_the_earlier_lift():
    series = [timed("x", v) for v in [100, 120, 120, 120, 150, 150]]
    rows = dash.case_episodes(series, ["same"] * len(series))
    assert len(rows) == 2
    assert rows[0]["end"] == 3 and rows[0]["peak"] == 1.2
    assert rows[1]["start"] == 4 and rows[1]["baseline"] == 120


def test_a_transient_drop_does_not_become_a_completed_episode():
    series = [timed("x", v) for v in [100, 80, 100, 100]]
    assert dash.case_episodes(series, ["same"] * len(series)) == []
    series = [timed("x", 100), timed("x", 80)]
    row = dash.case_episodes(series, ["same"] * len(series))[0]
    assert row["confirmed"] is None and row["recovered"] is None


def test_an_environment_change_does_not_recover_an_old_regression():
    series = [timed("x", v) for v in [100, 80, 80, 100, 100]]
    rows = dash.case_episodes(series, ["old", "old", "old", "new", "new"])
    assert len(rows) == 1 and rows[0]["recovered"] is None
    assert rows[0]["context_changed"] and rows[0]["last"] == 2


def test_episode_membership_is_not_truncated_or_hidden_by_partial_recovery():
    nights = []
    for i in range(5):
        cases = [timed(str(k), 100 if i == 0 or i >= 3 and k < 74 else 80) for k in range(75)]
        nights.append(bench_night(f"2026110{i + 1}", "a" * 40, *cases))
    catalog = {"mm": {c["key"]: i for i, c in enumerate(nights[0]["ops"]["mm"]["cases"])}}
    groups = dash.episodes(nights, catalog)
    assert len(groups) == 1
    rows = groups[0]["rows"]
    assert len(rows) == 75 and len({r["case"] for r in rows}) == 75
    assert sum(r["recovered"] is not None for r in rows) == 74
    assert rows[-1]["recovered"] is None and rows[-1]["baseline"] == 100


def test_episode_plot_does_not_use_measurements_from_a_later_context(tmp_path, monkeypatch):
    chrome = os.environ.get("CHROME_BIN") or shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chromium is needed to check episode plots")
    monkeypatch.setattr(dash, "commits", lambda a, b: [])
    source = tmp_path / "summaries"
    source.mkdir()
    for i, value in enumerate([100, 80, 80, 100, 100]):
        case = {**timed("x", value), "direction": "fwd", "dtype": "bf16", "speedup": 1, "ref_tflops": 100, "notes": []}
        night = bench_night(f"2026110{i + 1}", "a" * 40, case, driver="570" if i < 3 else "575")
        night["schema"] = 2
        (source / f"{night['date']}.b200.json").write_text(json.dumps(night))
    site = tmp_path / "site"
    dash.render(SimpleNamespace(source=str(source), outdir=str(site)))
    result = subprocess.run([
        chrome,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-background-networking",
        f"--user-data-dir={tmp_path / 'chrome'}",
        "--virtual-time-budget=1500",
        "--dump-dom",
        (site / "index.html").as_uri() + "#tab=episodes",
    ], capture_output=True, text=True, check=True, timeout=30)
    assert "last joint point 2026-11-03: −20.0%" in result.stdout
    assert re.search(r'<span class="ep-b ep-b-warn">[^<]*unresolved[^<]*</span>', result.stdout)
    assert not re.search(r'(?:cx|cy|points)="[^"]*(?:NaN|Infinity)', result.stdout)


@pytest.mark.parametrize("values,expected_text,lift_count", [
    ([100, 120, 120, 80], "latest downturn · awaiting confirmation", 1),
    ([100, 120, 120, 150, 150], "1 input · 1 op(s) · 1 GPU(s)", 2),
])
def test_lift_summaries_distinguish_current_measurements_and_unique_inputs(tmp_path, monkeypatch, values, expected_text,
                                                                           lift_count):
    chrome = os.environ.get("CHROME_BIN") or shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chromium is needed to check episode summaries")
    monkeypatch.setattr(dash, "commits", lambda a, b: [])
    source = tmp_path / "summaries"
    source.mkdir()
    for i, value in enumerate(values):
        case = {**timed("x", value), "direction": "fwd", "dtype": "bf16", "speedup": 1, "ref_tflops": 100, "notes": []}
        night = bench_night(f"2026110{i + 1}", "a" * 40, case)
        night["schema"] = 2
        (source / f"{night['date']}.b200.json").write_text(json.dumps(night))
    site = tmp_path / "site"
    dash.render(SimpleNamespace(source=str(source), outdir=str(site)))
    result = subprocess.run([
        chrome,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-background-networking",
        f"--user-data-dir={tmp_path / 'chrome'}",
        "--virtual-time-budget=1500",
        "--dump-dom",
        (site / "index.html").as_uri() + "#tab=episodes",
    ], capture_output=True, text=True, check=True, timeout=30)
    section = re.search(
        r'<div class="tiles ep-tiles" id="ep-tiles">(.*?)<div id="ep-sections">(.*?)</section>\s*</div>', result.stdout,
        re.S)
    assert section is not None and expected_text in section.group(0)
    assert f'<div class="k">Lifts</div><div class="v">{lift_count}</div>' in section.group(1)


def test_historical_episode_inputs_remain_addressable_after_an_op_disappears(monkeypatch):
    monkeypatch.setattr(dash, "commits", lambda a, b: [])
    case = {**timed("x", 100), "direction": "fwd", "dtype": "bf16", "speedup": 1, "ref_tflops": 100, "notes": []}
    nights = [bench_night(f"2026110{i + 1}", "a" * 40, {**case, "tlx_tflops": v}) for i, v in enumerate([100, 80, 80])]
    nights.append({**bench_night("20261104", "a" * 40), "ops": {}})
    p = dash.page_data({"b200": nights})["platforms"]["b200"]
    assert p["ops"] == {}
    row = p["episodes"][0]["rows"][0]
    assert p["episode_cases"]["mm"][row["case"]][3] == case["key"]
    assert row["last"] == 2 and row["recovered"] is None
