"""One case's measurements, turned into one of the four statuses.

Every threshold here is **absolute**. There is no recorded baseline: a case is
judged against fixed numbers and against a reference measured in the same
process and the same run, so a verdict depends only on the run that produced
it. That is what lets the suite be a single stateless command with nothing to
record, promote, or keep in sync across machines.

The reference matters as much as the absoluteness. ``speedup`` is TLX against
``torch.matmul`` timed back to back on one GPU, so clock state, thermal
history, driver version and machine identity all cancel -- they move absolute
throughput by more than the regression being looked for, and they move both
providers together.
"""

from __future__ import annotations

from typing import Optional

from .compile import CompileStat
from .contract import Case, Result, Stat, Status

#: Below this multiple of the reference, a case is ``pip``. Absolute, not
#: relative to a previous run: the question is "is TLX competitive with the
#: vendor library today", which does not need history to answer.
MIN_SPEEDUP = 0.9

#: Coefficient of variation above which no perf verdict is claimed. CV is a
#: WITHIN-run figure -- the width of one run's distribution -- so this asks
#: whether the machine held still while measuring, not whether the number
#: reproduces.
#:
#: Read off the TFLOP/s samples, which is the same question as reading it off
#: the latencies only because the threshold is small: throughput is 1/latency,
#: so the two CVs agree to first order and diverge as the distribution widens.
#: At 3% that is a fraction of a tenth of a percent.
MAX_CV = 0.03


def judge(
    case: Case,
    tlx: Stat,
    ref: Optional[Stat],
    *,
    tlx_host_us: Optional[float] = None,
    compile_stat: Optional[CompileStat] = None,
    correct: Optional[bool] = None,
    accuracy_note: str = "",
) -> Result:
    """Turn one case's measurements into a verdict.

    Order is cheapest-to-disqualify first, and it is load-bearing. A wrong
    answer outranks every perf claim -- benchmarking a kernel that computes the
    wrong thing produces a number that looks like signal and is not. A noisy
    case has no perf claim to make either, so it must not be called ``pip``:
    reporting "too slow" from a measurement we already know is untrustworthy is
    worse than reporting nothing.
    """
    result = Result(case=case, tlx=tlx, ref=ref, tlx_host_us=tlx_host_us, ref_host_us=None)
    # Throughput, so TLX is the numerator; >1 still means TLX is faster.
    if ref is not None and ref.mean and tlx.mean:
        result.speedup = tlx.mean / ref.mean

    if compile_stat is not None:
        result.t_cold_s = compile_stat.t_cold_s
        result.n_configs = compile_stat.n_configs

    if correct is False:
        result.status = Status.ERROR
        result.notes.append(accuracy_note or "output does not match the reference")
        return result

    if tlx.cv > MAX_CV:
        result.status = Status.NOISY
        result.notes.append(f"CV {tlx.cv * 100:.1f}% over the {MAX_CV * 100:.0f}% limit "
                            f"across {tlx.n_kept} samples; no perf verdict claimed")
        return result

    # Compile time is its own axis: a kernel can be fast and still take too long
    # to get there, and the fix is different, so it is named separately in the
    # note even though it shares the `pip` status.
    if compile_stat is not None and compile_stat.over_cap:
        result.status = Status.PIP
        result.notes.append(f"first call took {compile_stat.t_cold_s:.0f}s against a {compile_stat.cap_s:.0f}s cap" +
                            (f" ({compile_stat.n_configs} configs benchmarked)" if compile_stat.n_configs else ""))
        return result

    if result.speedup is not None and result.speedup < MIN_SPEEDUP:
        result.status = Status.PIP
        result.notes.append(f"speedup {result.speedup:.3f}x is under the {MIN_SPEEDUP:g}x floor")
        return result

    if result.speedup is not None:
        result.notes.append(f"speedup {result.speedup:.3f}x")
    return result
