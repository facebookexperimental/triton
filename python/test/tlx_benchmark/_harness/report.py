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


def _fmt_us(stat) -> str:
    return f"{stat.p50 * 1e3:9.1f}" if stat else "        -"


def table(results: Sequence[Result]) -> str:
    lines = [
        f"{'shape':<28} {'dtype':<8} {'tlx us':>9} {'ref us':>9} {'speedup':>8} "
        f"{'TFLOP/s':>8} {'repro':>7} {'width':>7} {'t_cold':>8}  status",
        "-" * 118,
    ]
    for r in results:
        shape = "x".join(str(s) for s in r.case.shape)
        lines.append(f"{shape:<28} {r.case.dtype:<8} {_fmt_us(r.tlx)} {_fmt_us(r.ref)} "
                     f"{(f'{r.speedup:.3f}x' if r.speedup else '-'):>8} "
                     f"{(f'{r.tlx_tflops:.0f}' if r.tlx_tflops else '-'):>8} "
                     f"{(f'{r.tlx.spread * 100:.1f}%' if r.tlx else '-'):>7} "
                     f"{(f'{r.tlx.within_spread * 100:.1f}%' if r.tlx else '-'):>7} "
                     f"{(f'{r.t_cold_s:.0f}s' if r.t_cold_s else '-'):>8}  {_MARK[r.status]}")
        for note in r.notes:
            lines.append(f"{'':<28} {'':<8} -> {note}")
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


def render(results: Sequence[Result], env: dict, json_path: Optional[str] = None) -> str:
    out = [table(results), "", summary(results)]
    if json_path:
        out.append(f"artifact: {write_json(results, env, json_path)}")
    bad = failures(results)
    if bad:
        out.append("")
        out.append("FAILING:")
        out.extend(f"  {r.case.key}: {'; '.join(r.notes) or _MARK[r.status]}" for r in bad)
    return "\n".join(out)
