"""Rendering: a table for a person, a JSON artifact for a machine.

The JSON is the interface a review agent consumes -- it must never have to
parse the table. The table exists so that a person reading CI output can see
what happened without downloading anything.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional, Sequence

from .contract import Result, Status, artifact

#: Marks that survive a terminal with no colour and a diff with no rendering.
_MARK = {
    Status.PASS: "ok",
    Status.REGRESSED: "REGRESSED",
    Status.SLOW_COMPILE: "SLOW-COMPILE",
    Status.NOISY: "noisy",
    Status.HOST_BOUND: "host-bound",
    Status.ERROR: "ERROR",
}

#: Statuses that should fail an enforcing run. ``NOISY`` and ``HOST_BOUND`` are
#: deliberately absent: neither is a claim that the code got worse, and failing
#: on them would train people to ignore the suite.
FAILING = (Status.REGRESSED, Status.SLOW_COMPILE, Status.ERROR)


def _us(value) -> str:
    return f"{value * 1e3:9.1f}"


def _fmt_us(stat) -> str:
    return _us(stat.mean) if stat else "        -"


def _fmt_tflops_err(result) -> str:
    """Absolute uncertainty on throughput, in TFLOP/s.

    Derived from the same replicate-to-replicate spread the gate reads, rather
    than from a separately computed standard deviation. One definition of
    "uncertainty" across the table and the pass/fail decision is worth more
    than a more conventional statistic that could disagree with the verdict
    printed beside it. Throughput is inversely proportional to latency, so it
    inherits the latency's relative uncertainty directly.
    """
    if not result.tlx_tflops or not result.tlx:
        return "-"
    return f"{result.tlx_tflops * result.tlx.rel_max_deviation:.1f}"


def _fmt_cv(result) -> str:
    """Coefficient of variation of the within-run samples, as a percentage."""
    return f"{result.tlx.cv * 100:.1f}" if result.tlx else "-"


def _fmt_compile(result) -> str:
    if result.t_cold_s is None:
        return "-"  # --measure latency; no cold pass was run
    return f"{result.t_cold_s:.2f}s" if result.t_cold_s < 10 else f"{result.t_cold_s:.0f}s"


def table(results: Sequence[Result]) -> str:
    lines = [
        f"{'input':<34} {'dtype':<8} {'tlx mean':>9} {'ref mean':>9} {'speedup':>8} "
        f"{'TFLOP/s':>8} {'+-TF/s':>7} {'CV%':>6} {'reps':>4} {'compile':>8}  status",
        "-" * 134,
    ]
    for r in results:
        lines.append(f"{r.case.input:<34} {r.case.dtype:<8} {_fmt_us(r.tlx)} {_fmt_us(r.ref)} "
                     f"{(f'{r.speedup:.3f}x' if r.speedup else '-'):>8} "
                     f"{(f'{r.tlx_tflops:.0f}' if r.tlx_tflops else '-'):>8} "
                     f"{_fmt_tflops_err(r):>7} "
                     f"{_fmt_cv(r):>6} "
                     f"{(str(r.tlx.replicates) if r.tlx else '-'):>4} "
                     f"{_fmt_compile(r):>8}  {_MARK[r.status]}")
        for note in r.notes:
            lines.append(f"{'':<34} {'':<8} -> {note}")
    return "\n".join(lines)


def percentile_table(results: Sequence[Result]) -> str:
    """The tail, which a mean and a CV together still cannot show.

    Two kernels can share a mean and a CV and differ entirely at p99, and for a
    kernel inside a larger pipeline the tail is often what is actually felt.
    """
    lines = [
        f"{'input':<34} {'dtype':<8} {'tlx p50':>9} {'tlx p95':>9} {'tlx p99':>9} "
        f"{'p99/p50':>8} {'ref p50':>9}",
        "-" * 92,
    ]
    for r in results:
        if not r.tlx:
            continue
        ratio = r.tlx.p99 / r.tlx.p50 if r.tlx.p50 else float("nan")
        lines.append(f"{r.case.input:<34} {r.case.dtype:<8} {_us(r.tlx.p50)} {_us(r.tlx.p95)} "
                     f"{_us(r.tlx.p99)} {ratio:>7.2f}x {(_us(r.ref.p50) if r.ref else '        -')}")
    return "\n".join(lines)


def summary(results: Sequence[Result]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[_MARK[r.status]] = counts.get(_MARK[r.status], 0) + 1
    return ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))


def failures(results: Sequence[Result]) -> list[Result]:
    return [r for r in results if r.status in FAILING]


def write_json(results: Sequence[Result], env: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(artifact(results, env), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


#: Explains the columns whose meaning is not obvious from the header.
LEGEND = ("mean   = headline latency, the median of the per-replicate means.\n"
          "CV%    = coefficient of variation of the within-run samples, sd/mean, after IQR rejection.\n"
          "+-TF/s = between-run uncertainty: max deviation of the replicate means from their median,\n"
          "         in absolute throughput. This -- not CV -- is what the gate reads, because it is\n"
          "         the uncertainty on the headline number rather than the width of one run.")


def render(results: Sequence[Result], env: dict, json_path: Optional[str] = None) -> str:
    legend = LEGEND
    if env.get("input_spec"):
        legend = f"input  = {env['input_spec']}\n{legend}"
    out = [
        table(results), "", legend, "", "Percentiles (microseconds):",
        percentile_table(results), "",
        summary(results)
    ]
    if json_path:
        out.append(f"artifact: {write_json(results, env, json_path)}")
    bad = failures(results)
    if bad:
        out.append("")
        out.append("FAILING:")
        out.extend(f"  {r.case.key}: {'; '.join(r.notes) or _MARK[r.status]}" for r in bad)
    return "\n".join(out)
