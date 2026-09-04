import json
import os

import pytest
import triton

from _harness import (COLD_COMPILE_CAP_S, DEFAULT_REPLICATES, DEFAULT_WARMUP_ITERS, MAX_CLOCK_IDR, MAX_CV,
                      MAX_REPLICATE_DEVIATION, MIN_TOTAL_SAMPLES, MIN_SPEEDUP, SCHEMA_VERSION, Case, ClockTrace,
                      CompileStat, GpuState, Result, Status, artifact, decode_event_reasons, fresh_triton_cache,
                      parse_cpulist, reject_outliers_iqr, relative_interdecile_range, resolve_warmup_and_rep, summarize,
                      to_tflops)

# --------------------------------------------------------------------------
# measure: sizing the window
# --------------------------------------------------------------------------


def test_resolve_warmup_and_rep_scales_with_kernel_cost():
    # Sub-millisecond and few-millisecond kernels share the short window.
    assert resolve_warmup_and_rep(None, None, 0.05) == (25, 100)
    assert resolve_warmup_and_rep(None, None, 8.0) == (25, 100)
    # A slow kernel needs a much longer window or the sample count collapses.
    assert resolve_warmup_and_rep(None, None, 50.0) == (3000, 3000)


def test_resolve_warmup_and_rep_explicit_wins():
    assert resolve_warmup_and_rep(7, 11, 0.05) == (7, 11)
    assert resolve_warmup_and_rep(7, None, 50.0) == (7, 3000)


def test_warmup_is_an_iteration_count_not_a_duration():
    assert DEFAULT_WARMUP_ITERS == 100
    # The estimate table is still reachable, but only via auto_window=True.
    assert resolve_warmup_and_rep(None, None, 1.0) == (25, 100)


def test_one_replicate_by_default_and_the_quota_binds_directly():
    from _harness import window_for

    assert DEFAULT_REPLICATES == 1
    assert MIN_TOTAL_SAMPLES == 500
    assert window_for(1.0, DEFAULT_REPLICATES) == 500.0  # 500 samples of a 1ms kernel
    assert window_for(0.01, DEFAULT_REPLICATES) == 200.0  # floor still guards fast kernels


def test_p99_needs_the_quota_to_be_worth_printing():
    from _harness import percentiles

    at_100 = percentiles(list(range(100)), (50, 95, 99))
    assert at_100 == (49, 94, 98)  # p99 is the second-largest of 100: one observation
    at_500 = percentiles(list(range(500)), (50, 95, 99))
    assert at_500 == (249, 474, 494)  # p99 is now five samples deep


def test_single_replicate_reports_deviation_as_unmeasured():
    stat = summarize([[1.0, 2.0, 3.0] * 40], remove_outliers=False)
    assert stat.replicates == 1
    assert stat.rel_max_deviation is None
    assert stat.cv > 0


# --------------------------------------------------------------------------
# measure: outlier rejection and dispersion
# --------------------------------------------------------------------------


def test_reject_outliers_drops_the_spike_and_keeps_order():
    data = [10.0, 10.1, 9.9, 10.2, 400.0, 10.0, 9.8]
    kept = reject_outliers_iqr(data)
    assert 400.0 not in kept
    assert kept == [10.0, 10.1, 9.9, 10.2, 10.0, 9.8]


def test_reject_outliers_leaves_tiny_samples_alone():
    # Quartiles of three points are meaningless; rejecting there would throw
    # away real data rather than noise.
    assert reject_outliers_iqr([1.0, 50.0, 2.0]) == [1.0, 50.0, 2.0]


def test_summarize_rejects_outliers_before_summarizing():
    stat = summarize([10.0, 10.0, 10.0, 10.0, 10.5, 9.5, 400.0])
    assert stat.n_raw == 7
    assert stat.n_kept == 6
    assert stat.p50 == 10.0
    assert stat.max == 10.5


def test_dispersion_survives_outlier_rejection():
    stat = summarize([10.0, 12.0, 8.0, 11.0, 9.0, 10.5, 9.5, 11.5, 8.5])
    assert stat.n_kept == stat.n_raw
    assert stat.cv > 0.10


def test_between_run_deviation_is_not_within_run_dispersion():
    wide_but_reproducible = [[9.0, 10.0, 11.0] * 40, [9.0, 10.0, 11.0] * 40, [9.0, 10.0, 11.0] * 40]
    stat = summarize(wide_but_reproducible, remove_outliers=False)
    assert stat.replicates == 3
    assert stat.cv > 0.05  # each run is wide
    assert stat.rel_max_deviation == pytest.approx(0.0)  # but they agree exactly


def test_between_run_deviation_catches_a_drifting_machine():
    drifting = [[10.0] * 120, [10.6] * 120, [11.2] * 120]
    stat = summarize(drifting, remove_outliers=False)
    assert stat.rel_idr > 0.05  # pooled, so the drift shows here too
    assert stat.rel_max_deviation == pytest.approx(0.06, abs=0.005)


def test_rel_idr_is_a_decile_width_not_an_extreme():
    times = [10.0] * 200 + [30.0]
    stat = summarize(times, remove_outliers=False)
    assert (stat.max - stat.p50) / stat.p50 == pytest.approx(2.0)
    assert stat.rel_idr == pytest.approx(0.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


# --------------------------------------------------------------------------
# measure: percentiles and the tail
# --------------------------------------------------------------------------


def test_percentiles_are_observed_samples_not_interpolations():
    from _harness import percentiles

    values = [1.0] * 98 + [5.0, 9.0]
    p50, p90, p99 = percentiles(values)
    assert (p50, p90, p99) == (1.0, 1.0, 5.0)
    assert all(v in values for v in (p50, p90, p99))


def test_summarize_reports_the_tail():
    stat = summarize([[1.0] * 98 + [5.0, 9.0]] * 3, remove_outliers=False)
    assert stat.p50 == 1.0
    assert stat.p99 == 5.0
    assert stat.p99 / stat.p50 == 5.0


# --------------------------------------------------------------------------
# measure: TFLOP/s is the measured unit, not a rendering of one
# --------------------------------------------------------------------------


def test_to_tflops_inverts_per_sample():
    assert to_tflops([1.0, 2.0, 4.0], flop_count=1e12) == [1000.0, 500.0, 250.0]


def test_to_tflops_drops_nonpositive_samples():
    assert to_tflops([1.0, 0.0, -1.0, 2.0], flop_count=1e12) == [1000.0, 500.0]


def test_conversion_is_per_sample_so_the_tail_is_not_flattened():
    latencies = [1.0] * 98 + [5.0, 9.0]
    stat = summarize([to_tflops(latencies, flop_count=1e12)], remove_outliers=False)
    assert stat.min == pytest.approx(1000 / 9)  # the 9 ms iteration
    assert stat.p50 == pytest.approx(1000.0)


def test_throughput_percentiles_ascend_so_p99_is_the_best_case():
    latencies = [10.0] * 90 + [9.0] * 10  # ten fast iterations
    stat = summarize([to_tflops(latencies, flop_count=1e12)], remove_outliers=False)
    assert stat.p50 < stat.p95 <= stat.p99
    assert stat.p99 == pytest.approx(1000 / 9)  # the fast ones
    assert stat.min == pytest.approx(100.0)  # the slow ones


def test_summarize_records_its_unit():
    assert summarize([1.0, 2.0]).unit == "tflops"
    assert summarize([1.0, 2.0], unit="ms").unit == "ms"


# --------------------------------------------------------------------------
# contract: cases, statuses, the artifact
# --------------------------------------------------------------------------


def test_case_key_is_stable_and_readable():
    case = Case(op="mm", arch="sm100", dtype="float16", shape=(1024, 2048, 512, True, False))
    assert case.key == "mm/sm100/float16/1024x2048x512xTruexFalse"


def test_status_vocabulary_is_exactly_the_four_documented():
    assert {s.value for s in Status} == {"ok", "pip", "noisy", "error"}


def test_noisy_is_not_a_failure_but_pip_and_error_are():
    from _harness.report import FAILING

    assert Status.NOISY not in FAILING
    assert set(FAILING) == {Status.PIP, Status.ERROR}


def test_artifact_is_json_serializable_and_versioned():
    case = Case(op="mm", arch="sm100", dtype="float16", shape=(256, 256, 256, True, True))
    result = Result(case=case, status=Status.OK, tlx=summarize([1.0, 1.1, 0.9]), speedup=1.2)
    doc = json.loads(json.dumps(artifact([result], env={"gpu": "B200"})))
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["results"][0]["case"]["key"] == case.key
    assert doc["results"][0]["status"] == "ok"
    assert doc["results"][0]["ref"] is None
    # Schema 2: the throughput lives in the Stat and only there.
    assert doc["results"][0]["tlx"]["unit"] == "tflops"
    assert "tlx_tflops" not in doc["results"][0]


# --------------------------------------------------------------------------
# verdict: the four statuses
# --------------------------------------------------------------------------


def _stat(cv=0.0, mean=1.0):
    from _harness.contract import Stat

    return Stat(mean=mean, cv=cv, p50=mean, p95=mean, p99=mean, min=mean, max=mean, rel_max_deviation=0.0, rel_idr=0.0,
                replicates=5, n_kept=1000, n_raw=1000)


def _case():
    return Case(op="mm", arch="gfx942", dtype="float16", shape=(1024, 1024, 1024, True, True))


def test_gate_thresholds_are_the_documented_ones():
    assert MAX_CV == 0.03
    assert MIN_SPEEDUP == 0.9
    assert COLD_COMPILE_CAP_S == 120.0
    # rel_max_deviation is still computed and still in the artifact, but it is a
    # diagnostic now -- the README makes CV the gate.
    assert MAX_REPLICATE_DEVIATION == 0.02


def test_wrong_answer_outranks_every_perf_claim():
    from _harness import verdict

    # Fast enough to pass on speed, and perfectly steady -- but wrong.
    r = verdict.judge(_case(), _stat(mean=2.0), _stat(mean=1.0), correct=False, accuracy_note="49.9% of elements wrong")
    assert r.status is Status.ERROR
    assert "49.9%" in r.notes[0]


def test_pip_outranks_noisy_so_a_slow_shape_is_never_hidden_by_jitter():
    from _harness import verdict

    r = verdict.judge(_case(), _stat(cv=0.10, mean=1.0), _stat(mean=10.0), correct=True)
    assert r.status is Status.PIP
    assert r.speedup < MIN_SPEEDUP
    # The softness of the number is still reported, just not as the verdict.
    assert any("CV" in n for n in r.notes)


def test_noisy_decides_only_among_cases_that_are_otherwise_fine():
    from _harness import verdict

    r = verdict.judge(_case(), _stat(cv=0.10, mean=1.0), _stat(mean=1.0), correct=True)
    assert r.status is Status.NOISY


def test_compile_cap_outranks_noisy_because_it_is_a_separate_pass():
    from _harness import verdict
    from _harness.compile import CompileStat

    over = CompileStat(t_cold_s=300.0, cap_s=COLD_COMPILE_CAP_S)
    r = verdict.judge(_case(), _stat(cv=0.10, mean=1.0), _stat(mean=1.0), compile_stat=over, correct=True)
    assert r.status is Status.PIP
    assert "300s" in r.notes[0]


def test_speedup_is_a_throughput_ratio_so_above_one_is_still_faster():
    from _harness import verdict

    r = verdict.judge(_case(), _stat(mean=1200.0), _stat(mean=1000.0), correct=True)
    assert r.speedup == pytest.approx(1.2)


def test_pip_fires_on_the_absolute_speedup_floor():
    from _harness import verdict

    slow = verdict.judge(_case(), _stat(mean=0.5), _stat(mean=1.0), correct=True)  # 0.5x
    assert slow.status is Status.PIP
    fast = verdict.judge(_case(), _stat(mean=1.0), _stat(mean=1.0), correct=True)  # 1.0x
    assert fast.status is Status.OK


def test_pip_fires_on_the_compile_cap_even_when_the_kernel_is_fast():
    from _harness import verdict

    over = CompileStat(t_cold_s=300.0, cap_s=COLD_COMPILE_CAP_S)
    r = verdict.judge(_case(), _stat(mean=2.0), _stat(mean=1.0), compile_stat=over, correct=True)
    assert r.status is Status.PIP
    assert "300s" in r.notes[0]


def test_judging_reads_nothing_from_disk():
    import ast
    import inspect

    from _harness import verdict

    # Strip docstrings first: the module explains at length why there is no
    # baseline, and matching that prose would be matching the opposite of what
    # this test is for.
    tree = ast.parse(inspect.getsource(verdict))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    for stateful in ("open(", "pathlib", "json", "baseline", "Path"):
        assert stateful not in code, f"verdict.py must stay stateless; found {stateful!r}"


# --------------------------------------------------------------------------
# compile: the cold-compile guard
# --------------------------------------------------------------------------


def test_over_cap():
    assert CompileStat(t_cold_s=119.0).over_cap is False
    assert CompileStat(t_cold_s=121.0).over_cap is True
    # Measured: tlx.ops.mm at space="full", 1024^3, on B200.
    assert CompileStat(t_cold_s=284.6, n_compiles=350, n_configs=348).over_cap is True


def test_to_dict_carries_the_diagnostic_counts():
    d = CompileStat(t_cold_s=284.6, n_compiles=350, n_configs=348).to_dict()
    assert d == {"t_cold_s": 284.6, "n_compiles": 350, "n_configs": 348, "cap_s": 120.0, "over_cap": True}


def test_fresh_triton_cache_restores_both_knob_and_env():
    before_knob = triton.knobs.cache.dir
    before_env = os.environ.get("TRITON_CACHE_DIR")
    with fresh_triton_cache() as tmp:
        assert triton.knobs.cache.dir == tmp
        assert os.environ["TRITON_CACHE_DIR"] == tmp
        assert os.path.isdir(tmp)
    assert triton.knobs.cache.dir == before_knob
    assert os.environ.get("TRITON_CACHE_DIR") == before_env
    assert not os.path.exists(tmp)


def test_fresh_triton_cache_restores_after_an_exception():
    before = triton.knobs.cache.dir
    try:
        with fresh_triton_cache():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert triton.knobs.cache.dir == before


# --------------------------------------------------------------------------
# denoise: environment verification
#
# The GPU readings below were all taken from a B200 on devgpu006 -- unmanaged,
# under ``third_party/tlx/denoise.sh``, and under sustained load -- so these
# tests pin the decision logic against hardware that was observed rather than
# against invented numbers.
# --------------------------------------------------------------------------


def test_decode_event_reasons():
    assert decode_event_reasons(0x1) == ["gpu_idle"]
    assert decode_event_reasons(0x4) == ["sw_power_cap"]
    assert decode_event_reasons(0x44) == ["hw_thermal_slowdown", "sw_power_cap"]
    assert decode_event_reasons(0x0) == []


def test_sw_power_cap_is_not_degrading():
    from _harness.denoise import DEGRADING_REASONS

    assert not DEGRADING_REASONS & 0x4
    assert not DEGRADING_REASONS & 0x1  # gpu_idle says nothing about the window
    assert DEGRADING_REASONS & 0x40  # hw_thermal_slowdown does count
    assert DEGRADING_REASONS & 0x8  # hw_slowdown too


def test_no_clock_lock_check_exists():
    import _harness

    assert not hasattr(_harness, "clocks_locked")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0-3", {0, 1, 2, 3}),
        ("0,2,4", {0, 2, 4}),
        ("0-2,8", {0, 1, 2, 8}),
        # The real node0 cpulist from devgpu006.
        ("0-95,192-287", set(range(0, 96)) | set(range(192, 288))),
        ("", set()),
    ],
)
def test_parse_cpulist(text, expected):
    assert parse_cpulist(text) == expected


def test_gpu_state_to_dict_expands_reason_names():
    state = GpuState(available=True, sm_clock_mhz=1845, max_sm_clock_mhz=1965, event_reasons=0x4)
    d = state.to_dict()
    assert d["event_reason_names"] == ["sw_power_cap"]
    assert d["sm_clock_mhz"] == 1845


def test_gpu_state_unknown_without_nvidia_smi():
    assert GpuState(available=False).event_reason_names == []


def test_relative_interdecile_range_ignores_the_idle_ramp():
    steady = [830] * 110
    ramp = [990, 985, 970, 950, 900]
    values = ramp + steady
    assert (max(values) - min(values)) / 830 > 0.15  # what min/max would have said
    assert relative_interdecile_range(values) < MAX_CLOCK_IDR


def test_relative_interdecile_range_still_sees_real_movement():
    drifting = list(range(700, 1000, 3))  # a card sliding across the window
    assert relative_interdecile_range(drifting) > MAX_CLOCK_IDR


def test_relative_interdecile_range_degrades_to_min_max_on_tiny_samples():
    assert relative_interdecile_range([100, 110], median=100) == pytest.approx(0.1)
    assert relative_interdecile_range([]) is None


def test_clock_trace_stability():
    ramping = ClockTrace(samples=21, min_mhz=802, median_mhz=832, max_mhz=990, rel_idr=0.23)
    assert ramping.stable is False

    steady = ClockTrace(samples=120, min_mhz=828, median_mhz=832, max_mhz=990, rel_idr=0.014)
    assert steady.rel_idr < MAX_CLOCK_IDR
    assert steady.stable is True


def test_clock_trace_degradation_overrides_a_tight_spread():
    trace = ClockTrace(samples=100, min_mhz=828, median_mhz=832, max_mhz=840, rel_idr=0.01,
                       reasons=("hw_thermal_slowdown", ), degrading=("hw_thermal_slowdown", ))
    assert trace.stable is False


def test_clock_trace_unknown_without_samples():
    assert ClockTrace(samples=0).stable is None
    assert ClockTrace(samples=0).to_dict()["stable"] is None


def test_power_target_matches_the_part():
    from _harness.denoise import AMD, NVIDIA, Device

    assert Device(NVIDIA, 0, "NVIDIA B200").power_target_w == 750
    assert Device(NVIDIA, 0, "NVIDIA H100 80GB HBM3").power_target_w == 700
    assert Device(NVIDIA, 0, "NVIDIA GB200").power_target_w == 1200
    assert Device(AMD, 0, "MI350X").power_target_w == 1000
    assert Device(AMD, 0, "MI355X").power_target_w == 1400
    # Unknown parts get no target rather than a guessed one; the governor then
    # leaves the card's own limit alone.
    assert Device(NVIDIA, 0, "NVIDIA L4").power_target_w is None


def test_visibility_variable_follows_the_vendor():
    from _harness.denoise import AMD, NVIDIA, Device

    assert Device(NVIDIA, 2, "NVIDIA B200").visibility_env == "CUDA_VISIBLE_DEVICES"
    assert Device(AMD, 2, "MI350X").visibility_env == "HIP_VISIBLE_DEVICES"


def test_sampler_join_does_not_shadow_thread_internals():
    from _harness.denoise import _Sampler

    sampler = _Sampler(uuid=None)
    sampler.start()
    trace = sampler.finish()  # raised TypeError before the rename
    assert trace.samples == 0  # no NVML handle, so nothing was collected
    assert not sampler.is_alive()


def test_arch_matches_the_part():
    from _harness.denoise import AMD, NVIDIA, Device

    assert Device(NVIDIA, 0, "NVIDIA B200").arch == "sm100"
    assert Device(AMD, 0, "MI300X").arch == "gfx942"
    # "GB200" contains "B200"; both map to sm100, but the longer key must win so
    # the table stays correct if they ever diverge.
    assert Device(NVIDIA, 0, "NVIDIA GB200").arch == "sm100"
    # No entry yet -- None, not a guess. bench_mm turns this into a skip.
    assert Device(AMD, 0, "MI350X").arch is None


def test_amd_numa_node_resolves_through_pci_not_the_drm_index(tmp_path, monkeypatch):
    from _harness import denoise

    pci = tmp_path / "0000:c8:00.0"
    pci.mkdir()
    (pci / "numa_node").write_text("1\n")
    monkeypatch.setattr(denoise, "_PCI_DEVICES", str(tmp_path))
    # rocm-smi reports the domain in uppercase; sysfs spells it lowercase.
    monkeypatch.setattr(denoise, "_rocm", lambda args: "device,PCI Bus\ncard6,0000:C8:00.0")

    assert denoise._amd_numa_node(6) == 1
    # A device whose bus id is not in the listing gets None, not another's node.
    assert denoise._amd_numa_node(7) is None


def test_auto_selection_picks_the_least_used_gpu(monkeypatch):
    import _harness.denoise as denoise_mod
    from _harness.denoise import NVIDIA, Device

    fleet = [
        Device(NVIDIA, 0, "NVIDIA B200", memory_used_mib=14178),
        Device(NVIDIA, 1, "NVIDIA B200", memory_used_mib=4),
        Device(NVIDIA, 2, "NVIDIA B200", memory_used_mib=1254),
    ]
    monkeypatch.setattr(denoise_mod, "list_devices", lambda: fleet)
    assert denoise_mod.select_device("auto").index == 1
    assert denoise_mod.select_device("2").index == 2
    with pytest.raises(ValueError):
        denoise_mod.select_device("9")


def test_governor_is_inert_without_a_device():
    from _harness.denoise import Governor

    with Governor(None) as g:
        pass
    assert g.applied == []
    assert g.skipped and "no GPU" in g.skipped[0]


def test_governor_can_be_disabled():
    from _harness.denoise import Governor, NVIDIA, Device

    with Governor(Device(NVIDIA, 0, "NVIDIA B200"), enable=False) as g:
        pass
    assert g.applied == []


@pytest.mark.parametrize("message", [
    "CUDA error: unspecified launch failure",
    "CUDA error: an illegal memory access was encountered",
    "device-side assert triggered",
])
def test_mm_benchmark_stops_after_fatal_cuda_error(message):
    import bench_mm

    seen = []
    case_list = [Case(op="mm", arch="sm100", dtype="bfloat16", shape=(i, i, i)) for i in (1, 2, 3)]

    def run_one(case):
        seen.append(case)
        if len(seen) == 2:
            raise RuntimeError(message)
        return Result(case=case)

    results = bench_mm._run_cases(case_list, run_one)
    assert len(seen) == 2
    assert len(results) == 2
    assert results[-1].status is Status.ERROR
    assert "stopping:" in results[-1].notes[-1]


def test_mm_benchmark_continues_after_recoverable_error():
    import bench_mm

    case_list = [Case(op="mm", arch="sm100", dtype="bfloat16", shape=(i, i, i)) for i in (1, 2, 3)]

    def run_one(case):
        if case.shape[0] == 2:
            raise ValueError("unsupported layout")
        return Result(case=case)

    results = bench_mm._run_cases(case_list, run_one)
    assert len(results) == 3
    assert [result.status for result in results] == [Status.OK, Status.ERROR, Status.OK]
