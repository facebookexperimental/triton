import inspect
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


def test_report_groups_notes_after_the_table(tmp_path):
    from _harness import report

    pip = Result(case=_case(), status=Status.PIP, notes=["speedup 0.746x is under the 0.9x floor"])
    noisy_case = Case(op="mm", arch="gfx942", dtype="float16", shape=(2048, 1024, 512, True, True))
    noisy = Result(case=noisy_case, status=Status.NOISY, notes=["CV 4.2% over the 3% limit"])

    rendered = report.render([pip, noisy], {}, tmp_path / "mm.json")
    table_text, footer = rendered.split("\n\n", 1)

    assert "speedup 0.746x is under" not in table_text
    assert "CV 4.2% over" not in table_text
    artifact_at = footer.index("artifact:")
    summary_at = footer.index("1 PIP, 1 noisy")
    noisy_at = footer.index("Noisy data:")
    issues_at = footer.index("Issues:")
    assert artifact_at < summary_at < noisy_at < issues_at
    assert "CV 4.2% over the 3% limit" in footer[noisy_at:issues_at]
    assert "speedup 0.746x is under the 0.9x floor" in footer[issues_at:]


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
    assert Device(AMD, 0, "MI350X").arch == "gfx950"
    assert Device(AMD, 0, "MI355X").arch == "gfx950"


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


# --------------------------------------------------------------------------
# direction: the one op-specific axis that is a Case field
# --------------------------------------------------------------------------


def test_forward_keys_are_unchanged_by_the_direction_field():
    # Schema 3 added `direction`. Every key an op with no backward has ever
    # written has to survive it, or old artifacts stop being diffable.
    case = Case(op="mm", arch="sm100", dtype="float16", shape=(1024, 2048, 512, True, False))
    assert case.direction == "fwd"
    assert case.key == "mm/sm100/float16/1024x2048x512xTruexFalse"


def test_forward_and_backward_do_not_collide_in_the_artifact():
    shape = (4, 32, 4096, 128, False)
    fwd = Case(op="flash_attn", arch="sm100", dtype="bfloat16", shape=shape)
    bwd = Case(op="flash_attn", arch="sm100", dtype="bfloat16", shape=shape, direction="bwd")
    assert fwd.key != bwd.key
    assert bwd.key.endswith("/bwd")
    assert json.loads(json.dumps(fwd.to_dict()))["direction"] == "fwd"


# --------------------------------------------------------------------------
# extra: op-specific derived metrics
# --------------------------------------------------------------------------


def test_extra_defaults_to_empty_and_round_trips():
    result = Result(case=_case(), status=Status.OK)
    assert result.extra == {}
    doc = json.loads(json.dumps(artifact([result], env={})))
    assert doc["results"][0]["extra"] == {}


def test_declared_extra_columns_are_rendered_and_undeclared_ones_are_not():
    from _harness import report

    result = Result(case=_case(), status=Status.OK, tlx=summarize([1.0]),
                    extra={"ref_backend": "cudnn", "tokens": 32768})
    without = report.table([result])
    assert "cudnn" not in without and "32768" not in without

    with_col = report.table([result], (("ref bknd", "ref_backend"), ))
    assert "ref bknd" in with_col and "cudnn" in with_col
    # Only what was declared: `tokens` stays in the artifact.
    assert "32768" not in with_col


def test_a_missing_extra_key_renders_as_absent_not_as_a_crash():
    from _harness import report

    result = Result(case=_case(), status=Status.OK, tlx=summarize([1.0]), extra={})
    # The data row, not the dashed separator: the cell itself must be a dash.
    row = report.table([result], (("Mtok/s", "mtokens_per_s"), )).splitlines()[2]
    # ... <extra cell>  <status>  <best config>
    assert row.split()[-3:] == ["-", "ok", "-"]


# --------------------------------------------------------------------------
# verdict: the absolute floor, for an op with no reference
# --------------------------------------------------------------------------


def test_floor_gates_an_op_that_has_no_reference():
    from _harness import verdict

    slow = verdict.judge(_case(), _stat(mean=40.0), None, floor_tflops=100.0)
    assert slow.status is Status.PIP
    assert "under the 100 TFLOP/s floor" in "; ".join(slow.notes)

    fast = verdict.judge(_case(), _stat(mean=140.0), None, floor_tflops=100.0)
    assert fast.status is Status.OK


def test_no_floor_means_a_reference_less_op_reports_rather_than_gates():
    from _harness import verdict

    assert verdict.judge(_case(), _stat(mean=1.0), None).status is Status.OK
    assert verdict.judge(_case(), _stat(mean=1.0), None, floor_tflops=None).status is Status.OK


def test_the_speedup_floor_wins_when_a_reference_exists():
    from _harness import verdict

    # A floor is the substitute for a ratio, not an addition to one: with a
    # reference present the ratio decides, so the two can never disagree.
    result = verdict.judge(_case(), _stat(mean=100.0), _stat(mean=50.0), floor_tflops=1e6)
    assert result.status is Status.OK


# --------------------------------------------------------------------------
# driver: the adapter contract
# --------------------------------------------------------------------------

BENCH_MODULES = ("bench_mm", "bench_flash_attn", "bench_hstu_attn", "bench_kda")


@pytest.mark.parametrize("module_name", BENCH_MODULES)
def test_every_bench_module_satisfies_the_adapter_contract(module_name):
    import importlib

    bench = importlib.import_module(module_name)
    for name in ("OP", "REF_NAME", "EXTRA_COLUMNS", "cases", "prepare", "supported", "default_json", "run", "main"):
        assert hasattr(bench, name), f"{module_name} is missing {name}"
    assert all(len(col) == 2 for col in bench.EXTRA_COLUMNS)
    # An op with no reference must say so rather than leave it implicit.
    assert isinstance(bench.REF_NAME, str)


@pytest.mark.parametrize("module_name", BENCH_MODULES)
def test_synthetic_cases_are_well_formed_without_a_gpu(module_name):
    import importlib

    bench = importlib.import_module(module_name)
    cases = bench.cases(synthetic=True)
    assert cases, f"{module_name} has no synthetic shapes"
    for case in cases:
        assert case.op == bench.OP
        assert case.direction in ("fwd", "bwd")
        assert case.label, "an op must render its own shape tuple"
        assert case.key.count("/") >= 3


def test_space_resolves_to_each_ops_own_default():
    import importlib

    from _harness import driver

    # mm has a heuristic_config; the attention ops do not, and their kernels
    # accept only "full"/"smoke" -- a shared "heuristic" default is a KeyError.
    assert driver.resolve_space(importlib.import_module("bench_mm"), None) == "heuristic"
    assert driver.resolve_space(importlib.import_module("bench_flash_attn"), None) == "full"
    # An explicit choice still wins.
    assert driver.resolve_space(importlib.import_module("bench_mm"), "full") == "full"


def test_bind_returns_the_four_entry_points_bound_to_the_module():
    import importlib

    from _harness import driver

    bench = importlib.import_module("bench_mm")
    supported, default_json, run, main = driver.bind(bench)
    assert callable(supported) and callable(run) and callable(main)
    assert default_json() == bench.default_json()


# --------------------------------------------------------------------------
# driver: the best-config column
# --------------------------------------------------------------------------


def test_a_config_is_abbreviated_whichever_tile_spelling_a_kernel_uses():
    from _harness import driver

    config = triton.Config({"BLOCK_M": 128, "BLOCK_SIZE_N": 256, "SPLIT_K": 2}, num_warps=8, num_stages=3)
    assert driver._fmt_config(config) == "BM=128 BN=256 SPLIT_K=2 w8 s3"


def test_every_autotuned_kernel_a_case_launches_is_recorded(monkeypatch):
    """The column is the harness's, not an op's: whatever ran is what is named."""
    import types

    from triton.runtime.autotuner import Autotuner

    from _harness import driver

    stub = lambda self, *a, **k: None  # noqa: E731
    monkeypatch.setattr(Autotuner, "run", stub)

    def attn_fwd():
        pass

    def attn_bwd():
        pass

    captured: dict = {}
    with driver._record_configs(captured):
        for fn, warps in ((attn_fwd, 4), (attn_bwd, 8)):
            Autotuner.run(
                types.SimpleNamespace(base_fn=fn, best_config=triton.Config({"BLOCK_M": 64}, num_warps=warps,
                                                                            num_stages=2)))
    assert Autotuner.run is stub, "the patch must not outlive the launch"
    assert driver._render_configs(captured) == "attn_fwd: BM=64 w4 s2 | attn_bwd: BM=64 w8 s2"


def test_an_op_whose_kernels_are_not_autotuned_reports_no_config():
    from _harness import driver, report

    assert driver._render_configs({}) is None
    result = Result(case=_case(), status=Status.OK, tlx=summarize([1.0]))
    assert report.table([result]).splitlines()[2].endswith("-")


# --------------------------------------------------------------------------
# driver: how often the cold-compile pass is paid
# --------------------------------------------------------------------------


class _FakeBench:
    """A bench module stand-in whose `prepare` records what it was asked for."""

    OP = "fake"
    REF_NAME = ""
    EXTRA_COLUMNS = ()

    def __init__(self, cold_compile=None, directions=("fwd", "bwd"), n=3):
        if cold_compile is not None:
            self.COLD_COMPILE = cold_compile
        self._directions = directions
        self._n = n

    def cases(self, synthetic=False):
        return [
            Case(op=self.OP, arch="sm100", dtype="bfloat16", shape=(i, ), direction=d)
            for i in range(self._n)
            for d in self._directions
        ]


def _cold_flags(bench, monkeypatch, mode=None):
    """Which cases the driver would pay a cold pass for."""
    from _harness import driver

    seen = []

    def fake_run_case(_bench, case, *, space, cold=True, latency_mode="wallclock"):
        seen.append((case.direction, cold))
        return Result(case=case, status=Status.OK)

    monkeypatch.setattr(driver, "run_case", fake_run_case)
    monkeypatch.setattr(driver, "capture_env", lambda *a, **k: {})
    monkeypatch.setattr(driver, "stable", __import__("contextlib").contextmanager(lambda *a, **k: iter([{}])))
    driver.run(bench, space="full", cold_compile_mode=mode)
    return seen


def test_first_mode_samples_one_cold_pass_per_direction(monkeypatch):
    # The whole point: hstu_attn compiles 48 forward configs per cold pass, and
    # paying that once per case is most of the suite's runtime for a question
    # that is not per-shape.
    seen = _cold_flags(_FakeBench(cold_compile="first"), monkeypatch)
    assert [c for _, c in seen].count(True) == 2  # one fwd, one bwd
    assert {d for d, c in seen if c} == {"fwd", "bwd"}
    # And it is the FIRST of each, so a later shape never pays it.
    assert seen[0] == ("fwd", True) and seen[1] == ("bwd", True)
    assert all(not c for _, c in seen[2:])


def test_all_is_the_default_so_an_op_that_says_nothing_is_unchanged(monkeypatch):
    seen = _cold_flags(_FakeBench(), monkeypatch)
    assert all(cold for _, cold in seen)


def test_none_skips_every_cold_pass(monkeypatch):
    seen = _cold_flags(_FakeBench(cold_compile="none"), monkeypatch)
    assert not any(cold for _, cold in seen)


def test_an_explicit_mode_overrides_the_ops_own(monkeypatch):
    seen = _cold_flags(_FakeBench(cold_compile="first"), monkeypatch, mode="all")
    assert all(cold for _, cold in seen)


def test_an_unknown_cold_compile_mode_is_rejected():
    from _harness import driver

    with pytest.raises(ValueError, match="cold_compile"):
        driver.resolve_cold_compile(_FakeBench(), "sometimes")


def test_mm_still_times_every_case_and_the_full_space_ops_do_not():
    import importlib

    from _harness import driver

    # mm's cold pass is under a second at heuristic space, so sampling it would
    # drop information for no saving.
    assert driver.resolve_cold_compile(importlib.import_module("bench_mm"), None) == "all"
    for name in ("bench_flash_attn", "bench_hstu_attn", "bench_kda"):
        assert driver.resolve_cold_compile(importlib.import_module(name), None) == "first"


def test_a_case_with_no_cold_pass_reports_no_compile_time_and_is_not_gated():
    from _harness import report, verdict

    result = verdict.judge(_case(), _stat(mean=100.0), _stat(mean=100.0), compile_stat=None)
    assert result.t_cold_s is None
    assert result.status is Status.OK
    # input, ref, tlx, speedup, compile, ...
    assert report.table([result]).splitlines()[2].split()[4] == "-"


# --------------------------------------------------------------------------
# report: one table per direction
# --------------------------------------------------------------------------


def _result(direction="fwd", shape=(4, 32, 4096, 128, False)):
    case = Case(op="flash_attn", arch="sm100", dtype="bfloat16", shape=shape, direction=direction)
    return Result(case=case, status=Status.OK, tlx=summarize([1.0]), ref=summarize([1.0]), speedup=1.0)


def test_an_op_with_one_direction_gets_one_unlabelled_table():
    from _harness import report

    rendered = report.tables([_result(), _result(shape=(2, 32, 8192, 128, True))])
    assert "[fwd]" not in rendered
    assert len(rendered.splitlines()) == 4  # header, rule, two rows


def test_forward_and_backward_are_reported_as_two_tables():
    from _harness import report

    # Interleaved on the way in, as `cases()` produces them.
    rendered = report.tables([
        _result("fwd"),
        _result("bwd"),
        _result("fwd", (2, 32, 8192, 128, True)),
        _result("bwd", (2, 32, 8192, 128, True))
    ])
    assert "[fwd]" in rendered and "[bwd]" in rendered
    assert rendered.index("[fwd]") < rendered.index("[bwd]")
    # Each table has its own header, and each case landed under its own heading.
    fwd_block, bwd_block = rendered.split("[bwd]")
    assert fwd_block.count("ref TF/s") == 1 and bwd_block.count("ref TF/s") == 1
    assert fwd_block.count("4096") == 1 and bwd_block.count("4096") == 1


def test_the_summary_and_the_artifact_stay_whole_across_the_split(tmp_path):
    from _harness import report

    rendered = report.render([_result("fwd"), _result("bwd")], {}, tmp_path / "fa.json")
    assert rendered.count("2 ok") == 1  # one summary over both tables
    doc = json.loads((tmp_path / "fa.json").read_text())
    assert {r["case"]["direction"] for r in doc["results"]} == {"fwd", "bwd"}


def _picked(monkeypatch, bench, **kwargs):
    """The cases the driver would actually run."""
    from _harness import driver

    picked = []

    def fake_run_case(_bench, case, *, space, cold=True, latency_mode="wallclock"):
        picked.append(case)
        return Result(case=case, status=Status.OK)

    monkeypatch.setattr(driver, "run_case", fake_run_case)
    monkeypatch.setattr(driver, "capture_env", lambda *a, **k: {})
    monkeypatch.setattr(driver, "stable", __import__("contextlib").contextmanager(lambda *a, **k: iter([{}])))
    _, env = driver.run(bench, space="full", **kwargs)
    return picked, env


def test_head_counts_per_direction_not_overall(monkeypatch):
    # A flat slice of the interleaved fwd/bwd list would give `head/2` shapes of
    # each, so `--head 2` would quietly measure one shape.
    picked, _ = _picked(monkeypatch, _FakeBench(cold_compile="none", n=5), head=2)
    assert [(c.shape[0], c.direction) for c in picked] == [(0, "fwd"), (0, "bwd"), (1, "fwd"), (1, "bwd")]


def test_head_is_unchanged_for_an_op_with_one_direction(monkeypatch):
    picked, _ = _picked(monkeypatch, _FakeBench(cold_compile="none", directions=("fwd", ), n=5), head=2)
    assert [c.shape[0] for c in picked] == [0, 1]


def test_direction_filter_runs_before_head(monkeypatch):
    picked, env = _picked(monkeypatch, _FakeBench(cold_compile="none"), head=2, directions=("bwd", ))
    assert [c.direction for c in picked] == ["bwd", "bwd"]
    assert env["directions"] == ["bwd"]


# --------------------------------------------------------------------------
# driver: the selected device is the single source of truth
# --------------------------------------------------------------------------


@pytest.fixture
def unpinned_driver():
    """Restore the module-level device pin, which is process-wide."""
    from _harness import driver

    saved = list(driver._SELECTED)
    driver._SELECTED.clear()
    yield driver
    driver._SELECTED[:] = saved


def test_arch_follows_the_selected_device_not_physical_gpu_zero(unpinned_driver, monkeypatch):
    from _harness.denoise import NVIDIA, Device

    # A heterogeneous host: `--device 1` must not report GPU 0's arch, and
    # nvidia-smi enumerates physical devices whatever the visibility variable
    # says, so list_devices()[0] is simply the wrong answer here.
    monkeypatch.setattr(
        unpinned_driver, "list_devices",
        lambda: [Device(NVIDIA, 0, "NVIDIA H100"), Device(NVIDIA, 1, "NVIDIA B200")])
    # Unpinned falls back to the first device, i.e. what torch calls cuda:0.
    # H100 is not in ARCH_BY_PART, so that answer is None -- and None is exactly
    # what the old code would have reported for a run pinned to the B200.
    assert unpinned_driver.arch() is None

    unpinned_driver.select(Device(NVIDIA, 1, "NVIDIA B200"))
    assert unpinned_driver.arch() == "sm100"
    assert unpinned_driver.device_index() == 1


def test_every_arch_consumer_reads_the_same_pin(unpinned_driver, monkeypatch):
    import importlib

    from _harness.denoise import NVIDIA, Device

    monkeypatch.setattr(unpinned_driver, "list_devices", lambda: [Device(NVIDIA, 0, "NVIDIA H100")])
    unpinned_driver.select(Device(NVIDIA, 3, "NVIDIA B200"))
    bench = importlib.import_module("bench_mm")
    # The artifact name, the support check and the cases all agree.
    assert unpinned_driver.default_json(bench).endswith("mm.sm100.json")
    assert unpinned_driver.supported(bench)
    assert {c.arch for c in bench.cases(synthetic=True)} == {"sm100"}


def test_no_gpu_stays_distinguishable_from_an_unset_pin(unpinned_driver, monkeypatch):
    monkeypatch.setattr(unpinned_driver, "list_devices", lambda: [])
    assert unpinned_driver.arch() is None
    assert unpinned_driver.device_index() == 0  # the denoise helpers still need an int


# --------------------------------------------------------------------------
# measure: the two latency modes
# --------------------------------------------------------------------------


def test_wallclock_is_the_default_and_the_documented_vocabulary():
    from _harness import LATENCY_MODES
    from _harness.measure import measure

    assert LATENCY_MODES == ("wallclock", "gpu_events")
    assert "wallclock" == inspect.signature(measure).parameters["mode"].default


def test_an_unknown_latency_mode_is_rejected():
    from _harness.measure import measure

    with pytest.raises(ValueError, match="mode must be one of"):
        measure(lambda: None, mode="profiler")
