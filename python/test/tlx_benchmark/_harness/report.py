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


def _tf(result, latency_ms) -> str:
    """Throughput at a given latency, in TFLOP/s."""
    if not result.flop_count or not latency_ms:
        return "        -"
    return f"{result.flop_count / (latency_ms * 1e-3) / 1e12:9.0f}"


def _fmt_samples(result) -> str:
    """Total timed kernel invocations behind the row.

    One number, not a replicates-by-iterations pair: the split matters to the
    harness -- replicates make the result reproducible, iterations make each
    run precise -- but a reader of the table only needs to know how much
    evidence is behind it. The breakdown is in the artifact.
    """
    return str(result.tlx.n_kept) if result.tlx else "-"


def _fmt_cv(result) -> str:
    """Coefficient of variation of the within-run samples, as a percentage."""
    return f"{result.tlx.cv * 100:.1f}" if result.tlx else "-"


def _fmt_compile(result) -> str:
    if result.t_cold_s is None:
        return "-"  # --measure latency; no cold pass was run
    return f"{result.t_cold_s:.2f}s" if result.t_cold_s < 10 else f"{result.t_cold_s:.0f}s"


def table(results: Sequence[Result]) -> str:
    lines = [
        f"{'input':<34} {'dtype':<8} {'ref TF/s':>9} {'tlx TF/s':>9} {'speedup':>8} {'compile':>8} "
        f"{'samples':>8} {'CV%':>6} {'p50 TF/s':>9} {'p90 TF/s':>9} {'p99 TF/s':>9}  status",
        "-" * 150,
    ]
    for r in results:
        lines.append(f"{r.case.input:<34} {r.case.dtype:<8} "
                     f"{_tf(r, r.ref.mean if r.ref else None)} {_tf(r, r.tlx.mean if r.tlx else None)} "
                     f"{(f'{r.speedup:.3f}x' if r.speedup else '-'):>8} "
                     f"{_fmt_compile(r):>8} "
                     f"{_fmt_samples(r):>8} "
                     f"{_fmt_cv(r):>6} "
                     f"{_tf(r, r.tlx.p50 if r.tlx else None)} "
                     f"{_tf(r, r.tlx.p90 if r.tlx else None)} "
                     f"{_tf(r, r.tlx.p99 if r.tlx else None)}  {_MARK[r.status]}")
        for note in r.notes:
            lines.append(f"{'':<34} {'':<8} -> {note}")
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
LEGEND = ("Every throughput column is TFLOP/s, HIGHER is better. `ref`/`tlx` are at the mean latency\n"
          "        (median of the per-replicate means). Latencies are in the JSON artifact.\n"
          "speedup = tlx TF/s / ref TF/s, so >1 means TLX is faster.\n"
          "pNN TF/s = throughput at the pNN *latency*, so the columns descend: p99 is the worst-case\n"
          "        throughput, not the best. Percentiles are nearest-rank over the pooled samples.\n"
          "samples = total timed kernel invocations behind the row, over all replicates.\n"
          "CV%  = coefficient of variation of the latency samples within a run, sd/mean, after IQR\n"
          "        rejection.\n"
          "The gate reads NEITHER CV nor the percentiles: it reads the between-run deviation of the\n"
          "replicate means, which is the uncertainty on the headline rather than the width of one run.\n"
          "That figure is in the JSON artifact, and in the note printed when it trips.")


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
