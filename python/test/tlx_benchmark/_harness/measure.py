"""Steady-state throughput measurement.

The reported unit is TFLOP/s, and it is the unit the samples are *in* -- each
timed iteration is converted the moment ``do_bench`` hands it back, before
outlier rejection, before the percentiles, before the CV. Converting at render
time instead (``flops / mean_latency``) makes every dispersion figure a latency
statistic wearing a throughput label, and inverts the tail: the p99 of a
latency distribution is the p1 of the throughput one. Everything downstream of
``summarize`` therefore describes the number the report prints.

``summarize`` itself stays unit-agnostic -- ``denoise`` summarizes clock MHz
through the same helpers -- so the conversion lives in ``measure`` alone. Pass
no ``flop_count`` and the samples stay in milliseconds.

Sampling is **blocked**, deliberately: a provider's whole warmup+measure window
runs to completion before the next provider starts. That is what tritonbench
does -- ``utils/triton_op.py`` reduces over providers per input, each fully
measured -- and matching it is what keeps our absolute numbers comparable to a
tritonbench run of the same shape.

The measurement loop itself is ``triton.testing.do_bench``, which already
flushes L2 before every rep and records per-rep events. What this module adds
around it is tritonbench's default policy, ported rather than imported:

* size warmup/rep from a cheap runtime estimate
  (``components/do_bench/utils.py``, ``utils/constants.py``)
* reject outliers by IQR and summarize on the median
  (``Latency`` in ``components/do_bench/run.py``)

Ported and not imported because this suite must not depend on the tritonbench
wheel. The behaviour is intended to match; the debt is that it can drift, so
each port names its source.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, Iterable, Optional

import triton

from .contract import Stat

#: Estimated per-iteration ms -> (warmup ms, rep ms).
#: tritonbench ``utils/constants.py::DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS``.
_WARMUP_REP_BY_ESTIMATE = ((1.0, (25, 100)), (10.0, (25, 100)), (float("inf"), (3000, 3000)))

#: Warmup before the measured window, in ITERATIONS rather than milliseconds.
#:
#: A duration was the wrong unit. Expressed as 3000ms it handed a 0.1ms kernel
#: 30,000 warmup iterations and a 30ms kernel 100 -- the fast shapes, which are
#: most of a sweep, paid the most for the least reason. Across 300 shapes that
#: was 97% of all launches, discarded.
#:
#: What a case actually needs here is *its own* steady state: cold L2 for these
#: operands, and the one-time allocator and autotuner work. Thermal steady
#: state, which the old 3000ms was sized for (13.9% -> 1.7% between-run spread
#: on a single shape measured in isolation), is a property of the sweep -- the
#: GPU never cools between 300 consecutive cases -- not of each case.
DEFAULT_WARMUP_ITERS = 100

#: Total timed iterations, floor.
#:
#: The measurement window is derived from this rather than fixed, because what
#: matters is how many samples stand behind the number, not how many
#: milliseconds were spent collecting them -- a fixed window silently gives a
#: fast kernel 50x the samples of a slow one.
#:
#: 500 rather than 100 because of the tail columns. Percentiles are nearest-rank
#: so every one is an observed sample: at n=100, p99 is the 99th smallest, i.e.
#: a single observation, and p95 is the sixth-largest. 500 puts p99 five samples
#: deep, which is the least that makes the column worth printing.
MIN_TOTAL_SAMPLES = 500

#: Floor on the measurement window itself, in ms.
#:
#: Without it a fast kernel meets the sample quota in ~10 ms, which is a short
#: enough slice of time to catch a single scheduling artefact and call it the
#: answer. Costs nothing: warmup dominates a replicate regardless.
MIN_REP_MS = 200

#: Number of independent measurement replicates per case.
#:
#: One. Replicates existed only to measure ``rel_max_deviation`` -- drift
#: BETWEEN runs, which a single contiguous window is blind to because its
#: samples share one thermal state, one clock trajectory and one allocation.
#: That figure was the gate; the README made ``cv`` the gate instead, and cv is
#: a within-run quantity that needs no second run.
#:
#: So replicates now cost 5x the warmup for a statistic nothing reads. In a
#: sweep the sweep itself is the drift detector: a machine that moves shows up
#: as later shapes reading differently. Raise this if drift per case ever needs
#: to be a per-case verdict again.
DEFAULT_REPLICATES = 1

#: A case whose reported p50 does not reproduce this closely across replicates
#: gets no perf verdict.
#:
#: Compared against ``Stat.rel_max_deviation`` -- BETWEEN-run reproducibility of
#: the reported mean, NOT the within-run ``cv``. Measured on a denoised B200,
#: compute-bound shapes reproduce to 0.4-1.7%, so 2% is the achievable floor
#: plus a little margin. An earlier version compared this number against a
#: within-run figure (5-7% for the same shapes, because the power-governed
#: clock wanders) and rejected healthy cases.
MAX_REPLICATE_DEVIATION = 0.02

#: Host-side per-call cost is still measured by ``host_overhead_us`` and still
#: recorded in the artifact, but it no longer produces a status of its own --
#: the README's four statuses have no ``host_bound``. It stays as a diagnostic
#: because it is what distinguishes "the kernel got slower" from "the launch
#: path got slower", and the two have nothing to do with each other.
#:
#: Measured on B200: ``tlx.ops.mm`` costs 43-63us of host time per call -- 4
#: TensorDescriptor constructions, the autotuner key lookup, and, when
#: SPLIT_K > 1, a fresh ``torch.empty((SPLIT_K*M, N))`` in the config pre_hook
#: -- against ~9us for ``torch.matmul``. At mm 2048x2048x2048 the whole measured
#: latency was 54us while host cost alone was 63us: the GPU idles waiting for
#: the launch and the number describes Python. Such a case now reports its
#: numbers like any other; read ``tlx_host_us`` before trusting a small shape.


def resolve_warmup_and_rep(warmup: Optional[int], rep: Optional[int], estimate_ms: float) -> tuple[int, int]:
    """Pick warmup/rep windows (in ms) for a kernel of this cost.

    Long kernels get much longer windows, since a 100 ms rep window around a
    50 ms kernel is two samples.
    """
    for upper, (default_warmup, default_rep) in _WARMUP_REP_BY_ESTIMATE:
        if estimate_ms <= upper:
            break
    return (default_warmup if warmup is None else warmup, default_rep if rep is None else rep)


def estimate_runtime_ms(fn: Callable, iters: int = 5, grad_to_none: Optional[Iterable] = None) -> float:
    """Rough per-iteration cost, used only to size the real measurement.

    Deliberately cheap and deliberately not trusted as a result: the GPU is not
    at steady state here.
    """
    di = triton.runtime.driver.active.get_device_interface()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    def run_once():
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        triton.runtime.driver.active.clear_cache(cache)
        fn()

    run_once()
    di.synchronize()

    start, end = di.Event(enable_timing=True), di.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    di.synchronize()
    return start.elapsed_time(end) / iters


def reject_outliers_iqr(data: list[float]) -> list[float]:
    """Drop points outside 1.5 IQR of the quartiles, preserving order.

    Ported from tritonbench ``Latency._remove_outliers_iqr``. Its job is to
    remove the occasional descheduled sample, not to make a noisy run look
    clean -- dispersion is computed *after* rejection and still gates the
    verdict, so a genuinely unstable machine cannot be filtered into a PASS.
    """
    if len(data) <= 3:
        return list(data)
    quantiles = statistics.quantiles(sorted(data), n=100)
    q1, q3 = quantiles[25], quantiles[75]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [x for x in data if lo <= x <= hi]


def relative_interdecile_range(values, median: Optional[float] = None) -> Optional[float]:
    """``(p90 - p10) / median``: the width of a distribution, robustly.

    Not ``(max - min)`` and not ``(max - p50)``. Those are extreme-value
    statistics: with thousands of samples they measure the worst single
    scheduling hiccup in the run and do not shrink as evidence accumulates, so
    a threshold against them says more about sample count than about stability.
    Deciles converge.
    """
    if not values:
        return None
    med = statistics.median(values) if median is None else median
    if not med:
        return None
    if len(values) < 10:  # too few points for deciles to mean anything
        return (max(values) - min(values)) / med
    deciles = statistics.quantiles(sorted(values), n=10)
    return (deciles[8] - deciles[0]) / med


def window_for(estimate_ms: float, replicates: int) -> float:
    """Measurement window, in ms, that meets the total-sample quota.

    ``do_bench`` derives its iteration count as ``rep_ms / estimate_ms``, so a
    sample count has to be expressed to it as a duration. Deriving the window
    instead of fixing it is what keeps the evidence behind a number from
    depending on how fast the kernel happens to be.
    """
    if not estimate_ms:
        return MIN_REP_MS
    per_replicate = math.ceil(MIN_TOTAL_SAMPLES / max(1, replicates))
    return max(MIN_REP_MS, per_replicate * estimate_ms)


def to_tflops(samples: Iterable[float], flop_count: float) -> list[float]:
    """Per-iteration latencies in ms -> per-iteration throughputs in TFLOP/s.

    Applied per sample rather than to the summary, which is the whole point:
    ``flop_count / mean(latency)`` is a harmonic mean of throughput and has no
    dispersion attached to it at all, so a CV or a percentile derived from it
    describes latency. A zero or negative sample would be a clock artefact
    rather than an infinitely fast kernel, so it is dropped.
    """
    scale = flop_count / 1e12 * 1e3  # FLOP -> TFLOP, and ms -> s
    return [scale / ms for ms in samples if ms > 0]


def percentiles(values, wanted=(50, 90, 99)) -> tuple:
    """Nearest-rank percentiles, so every returned value is an observed sample.

    Interpolating would invent a throughput the kernel never reached, which is
    exactly the wrong thing for a tail figure.

    Percentiles are literal: over TFLOP/s samples, p99 is the *fast* tail --
    only 1% of iterations beat it -- and the pessimistic end of the
    distribution is ``Stat.min``. Reading the columns as if they descended,
    which they do for latency, gets the sign of every tail wrong.
    """
    ordered = sorted(values)
    n = len(ordered)
    return tuple(ordered[min(n - 1, max(0, math.ceil(q / 100 * n) - 1))] for q in wanted)


def summarize(replicates: list[list[float]], remove_outliers: bool = True, unit: str = "tflops") -> Stat:
    """Collapse independent replicates into the reported statistic.

    Accepts a list of per-replicate sample lists. A single flat list is also
    accepted and treated as one replicate, in which case ``rel_max_deviation``
    is None -- it is not measurable from one run.

    Unit-agnostic -- it is handed TFLOP/s by ``measure`` and MHz by ``denoise``
    -- so ``unit`` is carried rather than assumed, and lands in the artifact so
    a consumer never has to infer it from the magnitudes.
    """
    if replicates and not isinstance(replicates[0], list):
        replicates = [replicates]  # a bare sample list
    replicates = [r for r in replicates if r]
    if not replicates:
        raise ValueError("no samples to summarize")

    kept_per_replicate = [reject_outliers_iqr(r) if remove_outliers else list(r) for r in replicates]
    kept_per_replicate = [k or r for k, r in zip(kept_per_replicate, replicates)]
    means = [statistics.fmean(k) for k in kept_per_replicate]
    pooled = [x for k in kept_per_replicate for x in k]

    mean = statistics.median_low(means)
    p50, p95, p99 = percentiles(pooled, (50, 95, 99))
    cv = (statistics.stdev(pooled) / mean) if len(pooled) > 1 and mean else 0.0
    rel_idr = relative_interdecile_range(pooled, p50) if p50 else float("inf")
    if len(means) > 1 and mean:
        rel_max_deviation = max(max(means) - mean, mean - min(means)) / mean
    else:
        # One replicate has nothing to deviate from. None, not 0.0 and not cv:
        # the first would claim perfect reproducibility and the second would
        # quietly relabel a within-run figure as a between-run one, which is
        # the exact conflation this field exists to avoid.
        rel_max_deviation = None

    return Stat(
        mean=mean,
        cv=cv,
        p50=p50,
        p95=p95,
        p99=p99,
        min=min(pooled),
        max=max(pooled),
        rel_max_deviation=rel_max_deviation,
        rel_idr=rel_idr,
        replicates=len(means),
        n_kept=len(pooled),
        n_raw=sum(len(r) for r in replicates),
        unit=unit,
    )


def host_overhead_us(fn: Callable, iters: int = 300) -> float:
    """Median wall-clock cost of *issuing* ``fn``, in microseconds.

    Times the call without synchronizing, so it captures the host-side work --
    argument marshalling, descriptor construction, allocation, launch -- and
    not the kernel. Stays in microseconds rather than being folded into the
    reported TFLOP/s: it is not device work, so expressing it as throughput
    would assert that the GPU did those FLOPs during it, which is the opposite
    of what a host-bound case means. Against ``flop_count / stat.mean``, this
    is what says whether a number describes the kernel or describes Python.
    """
    import time

    di = triton.runtime.driver.active.get_device_interface()
    fn()
    di.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    di.synchronize()
    return statistics.median(samples)


def measure(fn: Callable, *, flop_count: Optional[float] = None, warmup: Optional[int] = None,
            rep: Optional[int] = None, auto_window: bool = False, replicates: int = DEFAULT_REPLICATES,
            grad_to_none: Optional[Iterable] = None, remove_outliers: bool = True) -> Stat:
    """Measure ``fn`` to steady state and return its throughput.

    ``fn`` must be callable with no arguments and should launch exactly the
    work under test -- input construction belongs outside, or it is measured
    too.

    ``flop_count`` is the useful FLOPs of one call, and supplying it is what
    makes the returned ``Stat`` a throughput: every timed iteration is
    converted to TFLOP/s here, so the mean, the CV and the percentiles all
    describe the reported quantity. Omit it and the ``Stat`` is in
    milliseconds, which is what the timing internals below need anyway.

    ``replicates`` defaults to 1. More than one re-warms each time and measures
    drift BETWEEN runs (``rel_max_deviation``), which a single contiguous window
    cannot see -- but that figure is no longer the gate, so paying its warmup
    per case is not worth it in a sweep. The window is sized so the replicates
    together time at least ``MIN_TOTAL_SAMPLES`` iterations.

    The warmup defaults to ``DEFAULT_WARMUP_ITERS`` iterations and the window is
    derived from ``MIN_TOTAL_SAMPLES`` (see ``window_for``). Pass
    ``auto_window=True`` to size it from a runtime estimate instead, which is
    tritonbench's default policy but under-warms sub-10ms kernels badly enough
    to dominate the result -- see ``DEFAULT_WARMUP_ITERS``.
    """
    # Window sizing is inherently temporal -- `do_bench` derives its iteration
    # count from a duration -- so this half of the function stays in ms no
    # matter what the caller wants reported.
    estimate_ms = estimate_runtime_ms(fn, grad_to_none=grad_to_none)
    replicates = max(1, replicates)
    if auto_window:
        warmup_ms, rep_ms = resolve_warmup_and_rep(warmup, rep, estimate_ms)
    else:
        # do_bench takes a duration, so the iteration count is converted here
        # against the same estimate that sizes the window.
        warmup_ms = (DEFAULT_WARMUP_ITERS * estimate_ms) if warmup is None else warmup
        rep_ms = rep if rep is not None else window_for(estimate_ms, replicates)
    runs = [
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, grad_to_none=grad_to_none, return_mode="all")
        for _ in range(max(1, replicates))
    ]
    if flop_count:
        return summarize([to_tflops(r, flop_count) for r in runs], remove_outliers=remove_outliers, unit="tflops")
    return summarize(runs, remove_outliers=remove_outliers, unit="ms")
