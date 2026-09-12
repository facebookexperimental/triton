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
    if not value:
        return "        -"
    return f"{value:9.0f}"


def _stat(stat, field: str):
    return getattr(stat, field) if stat else None


def _fmt_samples(result) -> str:
    return str(result.tlx.n_kept) if result.tlx else "-"


def _fmt_cv(result) -> str:
    return f"{result.tlx.cv * 100:.1f}" if result.tlx else "-"


def _fmt_compile(result) -> str:
    if result.t_cold_s is None:
        return "-"  # no cold pass was run for this case
    return f"{result.t_cold_s:.2f}s" if result.t_cold_s < 10 else f"{result.t_cold_s:.0f}s"


def _fmt_extra(result, key: str) -> str:
    value = result.extra.get(key)
    if value is None:
        return "-"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _fmt_config(result) -> str:
    return result.best_config or "-"


def table(results: Sequence[Result], extra_columns: Sequence[tuple] = ()) -> str:
    """The common columns, then whichever op-specific ones the op asked for.

    ``extra_columns`` is a sequence of ``(header, key)`` where ``key`` indexes
    ``Result.extra``. Nothing here knows what any of them mean -- an op that
    wants one of its derived metrics in the table declares it and the width is
    taken from the data.
    """
    width = max([len(r.case.input) for r in results] + [len("input")])
    extra_widths = [max([len(head)] + [len(_fmt_extra(r, key)) for r in results]) for head, key in extra_columns]
    heads = "".join(f" {head:>{w}}" for (head, _), w in zip(extra_columns, extra_widths))
    lines = [
        f"{'input':<{width}} {'ref TF/s':>9} {'tlx TF/s':>9} {'speedup':>8} {'compile':>8} "
        f"{'samples':>8} {'CV%':>6} {'p50 TF/s':>9} {'p95 TF/s':>9} {'p99 TF/s':>9}{heads}  status  best config",
        "-" * (width + 97 + sum(w + 1 for w in extra_widths)),
    ]
    for r in results:
        cells = "".join(f" {_fmt_extra(r, key):>{w}}" for (_, key), w in zip(extra_columns, extra_widths))
        lines.append(f"{r.case.input:<{width}} "
                     f"{_tf(_stat(r.ref, 'mean'))} {_tf(_stat(r.tlx, 'mean'))} "
                     f"{(f'{r.speedup:.3f}x' if r.speedup else '-'):>8} "
                     f"{_fmt_compile(r):>8} "
                     f"{_fmt_samples(r):>8} "
                     f"{_fmt_cv(r):>6} "
                     f"{_tf(_stat(r.tlx, 'p50'))} "
                     f"{_tf(_stat(r.tlx, 'p95'))} "
                     f"{_tf(_stat(r.tlx, 'p99'))}{cells}  {_MARK[r.status]:<6}  {_fmt_config(r)}")
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


def _details(results: Sequence[Result], statuses: tuple[Status, ...]) -> list[str]:
    return [f"  {r.case.key}: {'; '.join(r.notes) or _MARK[r.status]}" for r in results if r.status in statuses]


def by_direction(results: Sequence[Result]) -> dict:
    """Results grouped by `Case.direction`, in the order the directions appear."""
    groups: dict = {}
    for r in results:
        groups.setdefault(r.case.direction, []).append(r)
    return groups


def tables(results: Sequence[Result], extra_columns: Sequence[tuple] = ()) -> str:
    """One table per direction, or just the one when the op has a single one.

    Split because forward and backward are different kernels doing different
    amounts of work: interleaving them puts two unrelated TFLOP/s scales in one
    column, and the eye reads down a column.
    """
    groups = by_direction(results)
    if len(groups) <= 1:
        return table(results, extra_columns)
    return "\n\n".join(f"[{direction}]\n{table(group, extra_columns)}" for direction, group in groups.items())


def render(results: Sequence[Result], env: dict, json_path: Optional[str] = None,
           extra_columns: Sequence[tuple] = ()) -> str:
    out = [tables(results, extra_columns), ""]
    if json_path:
        out.append(f"artifact: {write_json(results, env, json_path)}")
    out.append(summary(results))
    noisy = _details(results, (Status.NOISY, ))
    if noisy:
        out.extend(("", "Noisy data:", *noisy))
    bad = failures(results)
    if bad:
        out.extend(("", "Issues:", *_details(bad, FAILING)))
    return "\n".join(out)
