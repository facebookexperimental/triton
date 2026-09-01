"""Baselines and the verdict.

The enforcing metric is **speedup against a reference measured in the same
process and the same run** -- for mm, ``torch.matmul``. A ratio taken that way
is immune to clock state, thermal history, driver version and machine identity,
all of which move absolute milliseconds by more than the regression we are
trying to detect. Phase 1 measured that directly: the same kernel spanned 13.9%
across-run spread unlocked and 1.7% denoised, while its ratio to torch barely
moved.

Absolute milliseconds are still recorded, and still compared, but only as an
informational signal. They drift; the ratio does not.

Compile time is the exception and is judged against an absolute cap rather than
a baseline at all -- see ``compile.py``.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Optional, Sequence

from .compile import CompileStat
from .contract import Case, Result, Stat, Status
from .measure import HOST_BOUND_RATIO, MAX_REPLICATE_DEVIATION

#: A case may be this much slower than its baseline speedup before it fails.
#: Set against a measured 0.4-1.7% between-run deviation on denoised
#: compute-bound shapes, so it is roughly three times that -- tight enough to catch
#: a real regression, loose enough not to fire on one.
SPEEDUP_TOLERANCE = 0.05

#: Absolute latency is reported against baseline but never fails a run, because
#: it legitimately moves between machines and driver versions. This only
#: controls when the report calls the drift out.
LATENCY_DRIFT_NOTE = 0.10

BASELINE_DIR = pathlib.Path(__file__).resolve().parent.parent / "baselines"


def baseline_path(op: str, arch: str) -> pathlib.Path:
    return BASELINE_DIR / f"{op}.{arch}.json"


def load(op: str, arch: str, space: Optional[str] = None) -> dict:
    """Baseline entries keyed by ``Case.key``. Missing file is not an error --
    a new op has no baseline until someone records one.

    A ``space`` mismatch returns nothing rather than comparing: measured on
    B200, mm 1024^3 is 40.6us at ``space="heuristic"`` and 10.4us at
    ``space="full"``. Comparing across the two would report a 4x "regression"
    that is only a different search space.
    """
    path = baseline_path(op, arch)
    if not path.exists():
        return {}
    with path.open() as fh:
        document = json.load(fh)
    recorded = document.get("env", {}).get("space")
    if space is not None and recorded is not None and recorded != space:
        warnings.warn(f"baseline for {op}/{arch} was recorded at space={recorded!r} but this run uses "
                      f"space={space!r}; not comparing")
        return {}
    return document.get("cases", {})


def save(op: str, arch: str, results: Sequence[Result], env: dict) -> pathlib.Path:
    """Record the passing cases as the new baseline.

    Refuses anything that did not produce a trustworthy number. Baselining a
    noisy or host-bound run would bake the noise in as the thing to beat, and
    every later comparison would inherit it.
    """
    trustworthy = [r for r in results if r.status is Status.PASS and r.speedup is not None]
    rejected = [r.case.key for r in results if r not in trustworthy]
    path = baseline_path(op, arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "op": op,
        "arch": arch,
        "env": env,
        "not_baselined": rejected,
        "cases": {
            r.case.key: {
                "speedup": r.speedup,
                "tlx_p50_ms": r.tlx.p50 if r.tlx else None,
                "ref_p50_ms": r.ref.p50 if r.ref else None,
                "tlx_tflops": r.tlx_tflops,
                "t_cold_s": r.t_cold_s,
            }
            for r in trustworthy
        },
    }
    with path.open("w") as fh:
        json.dump(document, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def judge(
    case: Case,
    tlx: Stat,
    ref: Optional[Stat],
    *,
    tlx_host_us: Optional[float] = None,
    compile_stat: Optional[CompileStat] = None,
    baseline: Optional[dict] = None,
) -> Result:
    """Turn one case's measurements into a verdict.

    Order matters, and it runs cheapest-to-disqualify first: a host-bound or
    noisy case has no perf claim to make, so it must not be compared against a
    baseline at all. Reporting "regressed" for a number we already know is
    untrustworthy would be worse than reporting nothing.
    """
    result = Result(case=case, tlx=tlx, ref=ref, tlx_host_us=tlx_host_us, ref_host_us=None)
    if ref is not None and ref.p50 and tlx.p50:
        result.speedup = ref.p50 / tlx.p50

    if compile_stat is not None:
        result.t_cold_s = compile_stat.t_cold_s
        result.n_configs = compile_stat.n_configs

    if tlx_host_us is not None and tlx.p50 * 1e3 < HOST_BOUND_RATIO * tlx_host_us:
        result.status = Status.HOST_BOUND
        result.notes.append(f"latency {tlx.p50 * 1e3:.0f}us is under {HOST_BOUND_RATIO:g}x the "
                            f"{tlx_host_us:.0f}us host cost of issuing the call; this measures the "
                            f"launch path, not the kernel")
        return result

    if tlx.rel_max_deviation > MAX_REPLICATE_DEVIATION:
        result.status = Status.NOISY
        result.notes.append(f"p50 varied {tlx.rel_max_deviation * 100:.1f}% across {tlx.replicates} replicates, over "
                            f"the {MAX_REPLICATE_DEVIATION * 100:.0f}% limit; no perf verdict claimed")
        return result

    # Compile time is absolute and independent of the perf comparison, so it is
    # checked even when there is no baseline to compare latency against.
    if compile_stat is not None and compile_stat.over_cap:
        result.status = Status.SLOW_COMPILE
        result.notes.append(f"first call took {compile_stat.t_cold_s:.0f}s against a {compile_stat.cap_s:.0f}s cap" +
                            (f" ({compile_stat.n_configs} configs benchmarked)" if compile_stat.n_configs else ""))
        return result

    if not baseline:
        result.notes.append("no baseline entry; recorded but not gated")
        return result

    was = baseline.get("speedup")
    if was and result.speedup is not None:
        change = result.speedup / was - 1.0
        if change < -SPEEDUP_TOLERANCE:
            result.status = Status.REGRESSED
            result.notes.append(f"speedup {result.speedup:.3f}x against baseline {was:.3f}x "
                                f"({change * 100:+.1f}%, tolerance {-SPEEDUP_TOLERANCE * 100:.0f}%)")
        else:
            result.notes.append(f"speedup {result.speedup:.3f}x vs baseline {was:.3f}x ({change * 100:+.1f}%)")

    was_ms = baseline.get("tlx_p50_ms")
    if was_ms and abs(tlx.p50 / was_ms - 1.0) > LATENCY_DRIFT_NOTE:
        result.notes.append(f"absolute latency drifted {((tlx.p50 / was_ms) - 1) * 100:+.0f}% from baseline "
                            f"({was_ms * 1e3:.0f}us); informational, machines differ")
    return result
