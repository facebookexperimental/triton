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
