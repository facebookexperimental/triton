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
    Status.OK: "ok",
    Status.PIP: "PIP",
    Status.NOISY: "noisy",
    Status.ERROR: "ERROR",
}

#: Statuses that fail the run. ``NOISY`` is deliberately absent: it is not a
#: claim that the code is slow, only that the machine would not hold still, and
#: failing on it would train people to ignore the suite.
FAILING = (Status.PIP, Status.ERROR)


def _tf(value) -> str:
    """A TFLOP/s figure, column-width. No arithmetic: the harness measures in
    this unit, so the report's job here is formatting and nothing else."""
    if not value:
        return "        -"
    return f"{value:9.0f}"


def _stat(stat, field: str):
    return getattr(stat, field) if stat else None


def _fmt_samples(result) -> str:
    """Total timed kernel invocations behind the row.

    One number, not a replicates-by-iterations pair: the split matters to the
    harness -- replicates make the result reproducible, iterations make each
    run precise -- but a reader of the table only needs to know how much
    evidence is behind it. The breakdown is in the artifact.
    """
    return str(result.tlx.n_kept) if result.tlx else "-"


def _fmt_cv(result) -> str:
    """Coefficient of variation of the within-run TFLOP/s samples, as a
    percentage."""
    return f"{result.tlx.cv * 100:.1f}" if result.tlx else "-"


def _fmt_compile(result) -> str:
    if result.t_cold_s is None:
        return "-"  # no cold pass was run for this case
    return f"{result.t_cold_s:.2f}s" if result.t_cold_s < 10 else f"{result.t_cold_s:.0f}s"


def table(results: Sequence[Result]) -> str:
    """One row per case.

    The input column is sized to its contents rather than fixed. An op names its
    own inputs and mm's rendering is ~100 characters, so a fixed width silently
    pushed every later column out of alignment. There is no separate dtype
    column: mm carries dtype inside the input string, and a second copy in its
    own column is noise.
    """
    width = max([len(r.case.input) for r in results] + [len("input")])
    lines = [
        f"{'input':<{width}} {'ref TF/s':>9} {'tlx TF/s':>9} {'speedup':>8} {'compile':>8} "
        f"{'samples':>8} {'CV%':>6} {'p50 TF/s':>9} {'p95 TF/s':>9} {'p99 TF/s':>9}  status",
        "-" * (width + 84),
    ]
    for r in results:
        lines.append(f"{r.case.input:<{width}} "
                     f"{_tf(_stat(r.ref, 'mean'))} {_tf(_stat(r.tlx, 'mean'))} "
                     f"{(f'{r.speedup:.3f}x' if r.speedup else '-'):>8} "
                     f"{_fmt_compile(r):>8} "
                     f"{_fmt_samples(r):>8} "
                     f"{_fmt_cv(r):>6} "
                     f"{_tf(_stat(r.tlx, 'p50'))} "
                     f"{_tf(_stat(r.tlx, 'p95'))} "
                     f"{_tf(_stat(r.tlx, 'p99'))}  {_MARK[r.status]}")
        for note in r.notes:
            lines.append(f"{'':<{width}} -> {note}")
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
LEGEND = ("Every number here is TFLOP/s, HIGHER is better -- each timed iteration is converted to\n"
          "        TFLOP/s before any statistic is taken, so CV and the percentiles describe the\n"
          "        throughput distribution, not a latency one. `ref`/`tlx` are the mean (median of the\n"
          "        per-replicate means).\n"
          "speedup = tlx TF/s / ref TF/s, so >1 means TLX is faster.\n"
          "pNN TF/s = literal percentile of the TFLOP/s samples, so the columns ASCEND: p99 is the\n"
          "        BEST case, beaten by 1% of iterations. The worst case is `min` in the JSON\n"
          "        artifact. Nearest-rank over the pooled samples, so each is an observed iteration.\n"
          "samples = total timed kernel invocations behind the row, over all replicates.\n"
          "CV%  = coefficient of variation of the TFLOP/s samples within a run, sd/mean, after IQR\n"
          "        rejection. This is the noise gate: over 3% and the row is `noisy` with no perf\n"
          "        verdict claimed.\n"
          "status: ok | PIP (speedup < 0.9x, or cold compile over 2 min) | noisy | ERROR (raised, or\n"
          "        the output did not match the reference). Every threshold is absolute -- there is no\n"
          "        recorded baseline, so a verdict depends only on the run that produced it.")


def render(results: Sequence[Result], env: dict, json_path: Optional[str] = None) -> str:
    legend = LEGEND
    if env.get("input_spec"):
        legend = f"input  = {env['input_spec']}\n{legend}"
    out = [table(results), "", legend, "", summary(results)]
    if json_path:
        out.append(f"artifact: {write_json(results, env, json_path)}")
    bad = failures(results)
    if bad:
        out.append("")
        out.append("FAILING:")
        out.extend(f"  {r.case.key}: {'; '.join(r.notes) or _MARK[r.status]}" for r in bad)
    return "\n".join(out)
