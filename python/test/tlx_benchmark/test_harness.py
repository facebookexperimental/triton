"""Unit tests for the harness statistics and contract. No GPU required.

The measurement policy is the part of this suite most likely to be wrong in a
way that silently produces plausible numbers, so it gets tested directly rather
than only through a benchmark run.
"""

import json

import pytest

from _harness import (DEFAULT_REP_MS, DEFAULT_WARMUP_MS, HOST_BOUND_RATIO, NOISE_FLOOR, SCHEMA_VERSION, Case, Result,
                      Status, artifact, reject_outliers_iqr, resolve_warmup_and_rep, summarize)


def test_resolve_warmup_and_rep_scales_with_kernel_cost():
    # Sub-millisecond and few-millisecond kernels share the short window.
    assert resolve_warmup_and_rep(None, None, 0.05) == (25, 100)
    assert resolve_warmup_and_rep(None, None, 8.0) == (25, 100)
    # A slow kernel needs a much longer window or the sample count collapses.
    assert resolve_warmup_and_rep(None, None, 50.0) == (3000, 3000)


def test_resolve_warmup_and_rep_explicit_wins():
    assert resolve_warmup_and_rep(7, 11, 0.05) == (7, 11)
    assert resolve_warmup_and_rep(7, None, 50.0) == (7, 3000)


def test_reject_outliers_drops_the_spike_and_keeps_order():
    data = [10.0, 10.1, 9.9, 10.2, 400.0, 10.0, 9.8]
    kept = reject_outliers_iqr(data)
    assert 400.0 not in kept
    assert kept == [10.0, 10.1, 9.9, 10.2, 10.0, 9.8]


def test_reject_outliers_leaves_tiny_samples_alone():
    # Quartiles of three points are meaningless; rejecting there would throw
    # away real data rather than noise.
    assert reject_outliers_iqr([1.0, 50.0, 2.0]) == [1.0, 50.0, 2.0]


def test_summarize_reports_spread_after_rejection():
    stat = summarize([10.0, 10.0, 10.0, 10.0, 10.5, 9.5, 400.0])
    assert stat.n_raw == 7
    assert stat.n_kept == 6
    assert stat.p50 == 10.0
    assert stat.max == 10.5


def test_summarize_spread_survives_rejection():
    """A wide-but-not-outlying distribution must stay wide.

    This is the property the noise gate depends on: IQR rejection must not be
    able to launder an unstable machine into a tight-looking result.
    """
    stat = summarize([10.0, 12.0, 8.0, 11.0, 9.0, 10.5, 9.5, 11.5, 8.5])
    assert stat.n_kept == stat.n_raw
    assert stat.spread > 0.15


def test_spread_measures_reproducibility_not_distribution_width():
    """The distinction the gate rests on.

    Measured on B200, mm 8192^3 has a ~6% decile width -- the power-governed
    clock wanders -- while its p50 reproduces to 1.7%. Gating on width rejected
    a case whose reported number was solid, so `spread` is the replicate-to-
    replicate figure and the width is kept separately as a diagnostic.
    """
    wide_but_reproducible = [[9.0, 10.0, 11.0] * 40, [9.0, 10.0, 11.0] * 40, [9.0, 10.0, 11.0] * 40]
    stat = summarize(wide_but_reproducible, remove_outliers=False)
    assert stat.replicates == 3
    assert stat.within_spread > 0.15
    assert stat.spread == pytest.approx(0.0)


def test_spread_catches_a_drifting_machine():
    """Tight within each replicate, but the level moves between them."""
    drifting = [[10.0] * 120, [10.6] * 120, [11.2] * 120]
    stat = summarize(drifting, remove_outliers=False)
    assert stat.within_spread > 0.05  # pooled, so the drift shows here too
    assert stat.spread == pytest.approx(0.06, abs=0.005)


def test_single_replicate_falls_back_to_the_wider_number():
    """With one replicate reproducibility is unmeasurable, so report the
    distribution width -- the conservative direction."""
    stat = summarize([[9.0, 10.0, 11.0] * 40], remove_outliers=False)
    assert stat.replicates == 1
    assert stat.spread == stat.within_spread > 0.15


def test_spread_is_a_decile_width_not_an_extreme():
    """A single hiccup in a long run must not read as instability.

    (max - p50) does not shrink as samples accumulate, so a threshold against
    it measures sample count rather than stability. One 3x sample in 200 tight
    ones is what a descheduled iteration looks like.
    """
    times = [10.0] * 200 + [30.0]
    stat = summarize(times, remove_outliers=False)
    assert (stat.max - stat.p50) / stat.p50 == pytest.approx(2.0)
    assert stat.within_spread == pytest.approx(0.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_case_key_is_stable_and_readable():
    case = Case(op="mm", arch="sm100", dtype="float16", shape=(1024, 2048, 512, True, False))
    assert case.key == "mm/sm100/float16/1024x2048x512xTruexFalse"


def test_default_window_is_not_the_estimate_table():
    """The fixed 3s/3s window is load-bearing, not incidental.

    Sizing the window from the estimate table hands a ~1ms kernel a 25ms
    warmup, which measured 13.9% across-run spread on a clock-locked B200
    against 1.7% at 3s/3s. If someone reverts the default to the table, this
    test should be the thing that argues back.
    """
    assert (DEFAULT_WARMUP_MS, DEFAULT_REP_MS) == (3000, 3000)
    assert resolve_warmup_and_rep(None, None, 1.0) == (25, 100)


def test_gate_thresholds_are_the_measured_ones():
    assert NOISE_FLOOR == 0.02
    # Just above 1: the host issues iteration N+1 while the GPU runs N, so host
    # cost only starves the GPU once it exceeds kernel time. At 5.0 this flagged
    # mm 8192x8192x1024 (138us measured, 42us host) as unmeasurable.
    assert HOST_BOUND_RATIO == 1.5


def test_host_bound_is_its_own_status():
    # NOISY and HOST_BOUND must stay distinct: one is the machine, the other is
    # the op's launch path against a too-small shape, and they are read and
    # fixed differently.
    assert Status.HOST_BOUND.value == "host_bound"
    assert Status.HOST_BOUND is not Status.NOISY


def test_artifact_is_json_serializable_and_versioned():
    case = Case(op="mm", arch="sm100", dtype="float16", shape=(256, 256, 256, True, True))
    result = Result(case=case, status=Status.PASS, tlx=summarize([1.0, 1.1, 0.9]), speedup=1.2)
    doc = json.loads(json.dumps(artifact([result], env={"gpu": "B200"})))
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["results"][0]["case"]["key"] == case.key
    assert doc["results"][0]["status"] == "pass"
    assert doc["results"][0]["ref"] is None
