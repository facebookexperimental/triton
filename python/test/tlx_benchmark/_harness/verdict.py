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
    # Precedence is the README's order: error, pip, noisy, ok. pip ahead of noisy is
    # deliberate -- a shape that is slow AND jittery is the one most worth seeing, and
    # noisy does not fail the run. MI300X 512x4096x1024 was 0.512x at CV 4.7%.
    # Compile is checked before latency: it is a separate cold pass, so the window's
    # CV says nothing about it.
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

    # Its own axis: a kernel can be fast and still take too long to get there,
    # and the fix is different, so it is named separately in the note even
    # though it shares the `pip` status.
    if compile_stat is not None and compile_stat.over_cap:
        result.status = Status.PIP
        result.notes.append(f"first call took {compile_stat.t_cold_s:.0f}s against a {compile_stat.cap_s:.0f}s cap" +
                            (f" ({compile_stat.n_configs} configs benchmarked)" if compile_stat.n_configs else ""))
        return result

    if result.speedup is not None and result.speedup < MIN_SPEEDUP:
        result.status = Status.PIP
        result.notes.append(f"speedup {result.speedup:.3f}x is under the {MIN_SPEEDUP:g}x floor")
        if tlx.cv > MAX_CV:  # still worth saying the number is soft
            result.notes.append(f"CV {tlx.cv * 100:.1f}% is also over the {MAX_CV * 100:.0f}% limit")
        return result

    if tlx.cv > MAX_CV:
        result.status = Status.NOISY
        result.notes.append(f"CV {tlx.cv * 100:.1f}% over the {MAX_CV * 100:.0f}% limit "
                            f"across {tlx.n_kept} samples; no perf verdict claimed")
        return result

    if result.speedup is not None:
        result.notes.append(f"speedup {result.speedup:.3f}x")
    return result
