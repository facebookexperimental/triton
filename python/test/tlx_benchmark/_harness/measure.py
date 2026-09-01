"""Steady-state latency measurement.

Sampling is **blocked**, deliberately: a provider's whole warmup+measure window
runs to completion before the next provider starts. That is what tritonbench
does -- ``utils/triton_op.py`` reduces over providers per input, each fully
measured -- and matching it is what keeps our absolute milliseconds comparable
to a tritonbench run of the same shape.

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

import statistics
from typing import Callable, Iterable, Optional

import triton

from .contract import Stat

#: Estimated per-iteration ms -> (warmup ms, rep ms).
#: tritonbench ``utils/constants.py::DEFAULT_WARMUP_REP_BY_ESTIMATED_KERNEL_MS``.
_WARMUP_REP_BY_ESTIMATE = ((1.0, (25, 100)), (10.0, (25, 100)), (float("inf"), (3000, 3000)))

#: Default measurement window, in ms. NOT the estimate-based table above, which
#: is opt-in via ``auto_window=True``.
#:
#: The table gives a ~1 ms kernel a 25 ms warmup -- about 24 iterations -- which
#: is nowhere near thermal steady state. Measured on a clock-locked B200,
#: mm 8192x8192x8192 fp16 moved from 13.9% to 1.7% across-run p50 spread purely
#: by going from the table's window to 3s/3s. 3000/3000 is also what the team
#: already passes to tritonbench for TLX perf runs, so this matches practice.
DEFAULT_WARMUP_MS = 3000
DEFAULT_REP_MS = 3000

#: Number of independent measurement replicates per case.
#:
#: Three is the smallest number from which a spread can be read at all, and the
#: cost is linear: at the 3s/3s window each replicate is ~6s per provider.
DEFAULT_REPLICATES = 3

#: A case whose reported p50 does not reproduce this closely across replicates
#: gets no perf verdict.
#:
#: Compared against ``Stat.spread``, which is replicate-to-replicate
#: reproducibility -- NOT the width of one replicate's distribution. Measured on
#: a denoised B200, compute-bound shapes reproduce to 0.4-1.7%, so 2% is the
#: achievable floor plus a little margin. An earlier version compared this
#: number against the distribution width (5-7% for the same shapes, because the
#: power-governed clock wanders) and rejected healthy cases.
NOISE_FLOOR = 0.02

#: A case is host-bound when its latency is below this multiple of the host
#: cost of issuing one call.
#:
#: The host issues iteration N+1 while the GPU runs iteration N, so host
#: overhead does not add to latency -- it only starves the GPU once it exceeds
#: kernel time. The threshold is therefore just above 1, not far above it. An
#: earlier value of 5.0 flagged mm 8192x8192x1024 (138us measured against 42us
#: host, and torch needs 128us for the same work, so the kernel plainly
#: dominates) as unmeasurable.
#:
#: Measured on B200: ``tlx.ops.mm`` costs 43-63us of host time per call -- 4
#: TensorDescriptor constructions, the autotuner key lookup, and, when
#: SPLIT_K > 1, a fresh ``torch.empty((SPLIT_K*M, N))`` in the config pre_hook
#: -- against ~9us for ``torch.matmul``. At mm 2048x2048x2048 the whole measured
#: latency was 54us while host cost alone was 63us: the GPU idles waiting for
#: the launch and the number describes Python.
HOST_BOUND_RATIO = 1.5


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
    clean -- the spread is computed *after* rejection and still gates the
    verdict, so a genuinely unstable machine cannot be filtered into a PASS.
    """
    if len(data) <= 3:
        return list(data)
    quantiles = statistics.quantiles(sorted(data), n=100)
    q1, q3 = quantiles[25], quantiles[75]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [x for x in data if lo <= x <= hi]


def quantile_spread(values, median: Optional[float] = None) -> Optional[float]:
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


def summarize(replicates: list[list[float]], remove_outliers: bool = True) -> Stat:
    """Collapse independent replicates into the reported statistic.

    Accepts a list of per-replicate sample lists. A single flat list is also
    accepted and treated as one replicate, in which case ``spread`` cannot be
    measured and is reported as the within-replicate width instead -- the
    conservative direction, since that width is the larger number.
    """
    if replicates and not isinstance(replicates[0], list):
        replicates = [replicates]  # a bare sample list
    replicates = [r for r in replicates if r]
    if not replicates:
        raise ValueError("no samples to summarize")

    kept_per_replicate = [reject_outliers_iqr(r) if remove_outliers else list(r) for r in replicates]
    kept_per_replicate = [k or r for k, r in zip(kept_per_replicate, replicates)]
    p50s = [statistics.median_low(k) for k in kept_per_replicate]
    pooled = [x for k in kept_per_replicate for x in k]

    p50 = statistics.median_low(p50s)
    within = quantile_spread(pooled, p50) if p50 else float("inf")
    if len(p50s) > 1 and p50:
        spread = max(max(p50s) - p50, p50 - min(p50s)) / p50
    else:
        spread = within

    return Stat(
        p50=p50,
        min=min(pooled),
        max=max(pooled),
        mean=statistics.fmean(pooled),
        spread=spread,
        within_spread=within,
        replicates=len(p50s),
        n_kept=len(pooled),
        n_raw=sum(len(r) for r in replicates),
    )


def host_overhead_us(fn: Callable, iters: int = 300) -> float:
    """Median wall-clock cost of *issuing* ``fn``, in microseconds.

    Times the call without synchronizing, so it captures the host-side work --
    argument marshalling, descriptor construction, allocation, launch -- and
    not the kernel. Compared against the measured latency, this is what tells
    us whether a case is host-bound and therefore ungateable.
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


def measure(fn: Callable, *, warmup: Optional[int] = None, rep: Optional[int] = None, auto_window: bool = False,
            replicates: int = DEFAULT_REPLICATES, grad_to_none: Optional[Iterable] = None,
            remove_outliers: bool = True) -> Stat:
    """Measure ``fn`` to steady state and return its latency.

    ``fn`` must be callable with no arguments and should launch exactly the
    work under test -- input construction belongs outside, or it is measured
    too.

    The measurement is repeated ``replicates`` times because that, and only
    that, measures the quantity the guard depends on: how reproducible the
    reported p50 is. Each replicate re-warms, so between-replicate variation
    picks up the slow drift a single long window cannot see.

    The window defaults to ``DEFAULT_WARMUP_MS`` / ``DEFAULT_REP_MS``. Pass
    ``auto_window=True`` to size it from a runtime estimate instead, which is
    tritonbench's default policy but under-warms sub-10ms kernels badly enough
    to dominate the result -- see ``DEFAULT_WARMUP_MS``.
    """
    if auto_window:
        estimate_ms = estimate_runtime_ms(fn, grad_to_none=grad_to_none)
        warmup_ms, rep_ms = resolve_warmup_and_rep(warmup, rep, estimate_ms)
    else:
        warmup_ms = DEFAULT_WARMUP_MS if warmup is None else warmup
        rep_ms = DEFAULT_REP_MS if rep is None else rep
    runs = [
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, grad_to_none=grad_to_none, return_mode="all")
        for _ in range(max(1, replicates))
    ]
    return summarize(runs, remove_outliers=remove_outliers)
