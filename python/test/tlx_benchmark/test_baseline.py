"""Unit tests for the verdict. No GPU required.

The numbers are the real ones from a denoised B200 at mm 8192^3 fp16, so the
tolerances are exercised at the magnitudes they will actually see.
"""

import json

import pytest

from _harness import Case, Status, summarize
from _harness import report as report_mod
from _harness.baseline import SPEEDUP_TOLERANCE, judge, load, save
from _harness.compile import CompileStat

BIG = Case(op="mm", arch="sm100", dtype="float16", shape=(8192, 8192, 8192, True, True))
SMALL = Case(op="mm", arch="sm100", dtype="float16", shape=(2048, 2048, 2048, False, False))

TLX = summarize([[1.080] * 120, [1.081] * 120, [1.079] * 120])  # ~1.08 ms
REF = summarize([[1.146] * 120, [1.145] * 120, [1.147] * 120])  # ~1.146 ms
HOST_US = 43.0


def test_speedup_is_the_ratio_of_reference_to_candidate():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US)
    assert r.speedup == pytest.approx(1.146 / 1.080, rel=1e-3)


def test_pass_within_tolerance():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, baseline={"speedup": 1.06})
    assert r.status is Status.PASS


def test_regression_beyond_tolerance():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, baseline={"speedup": 1.18})
    assert r.status is Status.REGRESSED
    assert "-10.1%" in " ".join(r.notes)


def test_just_inside_tolerance_passes():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, baseline={"speedup": 1.105})
    assert r.status is Status.PASS
    assert SPEEDUP_TOLERANCE == 0.05


def test_compile_cap_is_absolute_and_independent_of_the_baseline():
    """Measured: tlx.ops.mm at space="full", 1024^3, took 284.6s for 348
    configs. A relative gate would have called that fine against a slow
    baseline."""
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, compile_stat=CompileStat(t_cold_s=284.6, n_configs=348),
              baseline={"speedup": 1.06})
    assert r.status is Status.SLOW_COMPILE
    assert "348 configs" in " ".join(r.notes)


def test_noisy_suppresses_the_perf_verdict():
    """An untrustworthy number must not be reported as a regression. Claiming
    a regression from noise is worse than claiming nothing, because it is
    actionable and wrong."""
    shaky = summarize([[1.0] * 120, [1.10] * 120, [1.20] * 120])
    r = judge(BIG, shaky, REF, tlx_host_us=HOST_US, baseline={"speedup": 1.18})
    assert r.status is Status.NOISY


def test_host_bound_takes_priority_over_everything():
    tiny = summarize([[0.052] * 120, [0.0521] * 120, [0.0519] * 120])
    r = judge(SMALL, tiny, REF, tlx_host_us=62.0, compile_stat=CompileStat(t_cold_s=999.0), baseline={"speedup": 5.0})
    assert r.status is Status.HOST_BOUND


def test_missing_baseline_records_without_gating():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, baseline=None)
    assert r.status is Status.PASS
    assert "no baseline entry" in " ".join(r.notes)


def test_failures_selects_only_actionable_statuses():
    """NOISY and HOST_BOUND are not claims that the code got worse. Failing on
    them would teach people to ignore the suite."""
    results = [
        judge(BIG, TLX, REF, tlx_host_us=HOST_US, baseline={"speedup": 1.18}),  # regressed
        judge(BIG, summarize([[1.0] * 120, [1.2] * 120, [1.4] * 120]), REF, tlx_host_us=HOST_US),  # noisy
        judge(SMALL, summarize([[0.052] * 120] * 3), REF, tlx_host_us=62.0),  # host-bound
    ]
    assert [r.status for r in report_mod.failures(results)] == [Status.REGRESSED]


def test_save_refuses_to_baseline_untrustworthy_cases(tmp_path, monkeypatch):
    """Baselining a noisy run bakes the noise in as the thing to beat, and
    every later comparison inherits it."""
    import _harness.baseline as baseline_mod

    monkeypatch.setattr(baseline_mod, "BASELINE_DIR", tmp_path)
    good = judge(BIG, TLX, REF, tlx_host_us=HOST_US)
    noisy = judge(BIG, summarize([[1.0] * 120, [1.2] * 120, [1.4] * 120]), REF, tlx_host_us=HOST_US)
    noisy.case = Case(op="mm", arch="sm100", dtype="bfloat16", shape=BIG.shape)

    path = save("mm", "sm100", [good, noisy], env={"host": "test"})
    document = json.loads(path.read_text())
    assert list(document["cases"]) == [good.case.key]
    assert document["not_baselined"] == [noisy.case.key]
    assert load("mm", "sm100")[good.case.key]["speedup"] == pytest.approx(good.speedup)


def test_load_missing_baseline_is_not_an_error(tmp_path, monkeypatch):
    import _harness.baseline as baseline_mod

    monkeypatch.setattr(baseline_mod, "BASELINE_DIR", tmp_path)
    assert load("nosuchop", "sm100") == {}


def test_table_reports_compile_time_replicates_and_throughput_error():
    """The headline columns a reviewer actually reads.

    Throughput stays a single number and its uncertainty gets its own column,
    derived from the same replicate-to-replicate spread the gate reads -- one
    definition of "uncertainty" across the table and the verdict beside it.
    """
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US, compile_stat=CompileStat(t_cold_s=0.69, n_configs=None))
    r.tlx_tflops = 1004.0
    rendered = report_mod.table([r])

    header, _, row = rendered.splitlines()[:3]
    for column in ("input", "ref us", "tlx us", "speedup", "compile", "reps", "CV%", "p50 us", "p90 us", "p99 us"):
        assert column in header
    assert "shape" not in header and "TF/s" not in header

    assert "0.69s" in row  # sub-10s compile keeps two decimals
    assert "3x120" in row  # 3 replicates of 120 timed iterations each


def test_table_marks_compile_as_absent_when_not_measured():
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US)
    r.tlx_tflops = 1004.0
    assert r.t_cold_s is None
    assert report_mod.table([r]).splitlines()[2].rstrip().endswith("ok")


def test_input_column_is_op_supplied_and_defined_in_the_legend():
    """``(8192, 8192, 8192, True, False)`` says nothing on its own; only the op
    module knows the tuple means a product shape plus operand layouts."""
    labelled = Case(op="mm", arch="sm100", dtype="float16", shape=(8192, 8192, 8192, True, False),
                    label="8192x8192x8192 A:row B:col")
    assert labelled.input == "8192x8192x8192 A:row B:col"
    assert labelled.to_dict()["input"] == labelled.input
    # An op that supplies no label still renders.
    assert BIG.input == "8192x8192x8192xTruexTrue"

    r = judge(labelled, TLX, REF, tlx_host_us=HOST_US)
    rendered = report_mod.render([r], env={"input_spec": "(M x K) @ (K x N), with operand layouts"})
    assert "A:row B:col" in rendered
    assert "input  = (M x K) @ (K x N), with operand layouts" in rendered


def test_one_table_carries_the_tail_too():
    """A single row, not a summary table plus a percentile table."""
    r = judge(BIG, TLX, REF, tlx_host_us=HOST_US)
    rendered = report_mod.render([r], env={})
    assert "Percentiles" not in rendered
    assert len([ln for ln in rendered.splitlines() if ln.startswith("-" * 20)]) == 1


def test_speedup_direction_is_stated_and_correct():
    """`ref us / tlx us`, so >1 means TLX is faster.

    The direction is only checkable if the reader knows these are latencies
    rather than throughput -- under the throughput reading the column looks
    inverted -- so the units are in the headers and the definition is in the
    legend.
    """
    faster = judge(BIG, summarize([[1.0] * 120] * 3), summarize([[2.0] * 120] * 3), tlx_host_us=HOST_US)
    slower = judge(BIG, summarize([[2.0] * 120] * 3), summarize([[1.0] * 120] * 3), tlx_host_us=HOST_US)
    assert faster.speedup == pytest.approx(2.0)
    assert slower.speedup == pytest.approx(0.5)

    rendered = report_mod.render([faster], env={})
    assert "ref us" in rendered and "tlx us" in rendered
    assert "speedup = ref us / tlx us" in rendered
    assert "MICROSECONDS" in rendered
