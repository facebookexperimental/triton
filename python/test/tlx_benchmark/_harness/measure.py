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
    for upper, (default_warmup, default_rep) in _WARMUP_REP_BY_ESTIMATE:
        if estimate_ms <= upper:
            break
    return (default_warmup if warmup is None else warmup, default_rep if rep is None else rep)


def estimate_runtime_ms(fn: Callable, iters: int = 5, grad_to_none: Optional[Iterable] = None) -> float:
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
    if len(data) <= 3:
        return list(data)
    quantiles = statistics.quantiles(sorted(data), n=100)
    q1, q3 = quantiles[25], quantiles[75]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [x for x in data if lo <= x <= hi]


def relative_interdecile_range(values, median: Optional[float] = None) -> Optional[float]:
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
    if not estimate_ms:
        return MIN_REP_MS
    per_replicate = math.ceil(MIN_TOTAL_SAMPLES / max(1, replicates))
    return max(MIN_REP_MS, per_replicate * estimate_ms)


def to_tflops(samples: Iterable[float], flop_count: float) -> list[float]:
    scale = flop_count / 1e12 * 1e3  # FLOP -> TFLOP, and ms -> s
    return [scale / ms for ms in samples if ms > 0]


def percentiles(values, wanted=(50, 90, 99)) -> tuple:
    ordered = sorted(values)
    n = len(ordered)
    return tuple(ordered[min(n - 1, max(0, math.ceil(q / 100 * n) - 1))] for q in wanted)


def summarize(replicates: list[list[float]], remove_outliers: bool = True, unit: str = "tflops") -> Stat:
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
    # replicates>1 re-warms each time and measures drift BETWEEN runs
    # (rel_max_deviation); that is no longer the gate, hence the default of 1.
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
