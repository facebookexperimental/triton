"""Pure-python (no GPU) gate: run the sched2tlx emitter on a produced graph and
fail if it emitted any placeholder marker, meaning the schedule is outside the
emitter's supported family. Runs before any GPU time.
"""

from __future__ import annotations

from pathlib import Path

# Placeholder / bail-out markers the emitter can produce (from emitter.py).
_MARKERS = (
    "<unhandled",
    "<unsupported",
    "_unsupported>",
    "<missing:",
    "<inner_iter_arg",
    "falling back to single MMA",
)


def emit_source(graph_path: str | Path) -> str:
    """Emit TLX source for a schedule graph without importing torch/triton."""
    from sched2tlx import emitter, schedule_graph  # local, pure-python
    g = schedule_graph.load_graph(str(graph_path))
    return emitter.emit(g)


def dry_run(graph_path: str | Path) -> tuple[bool, list[str]]:
    """Return (ok, markers_found). ok is False if any placeholder was emitted."""
    try:
        src = emit_source(graph_path)
    except Exception as e:  # noqa: BLE001 - surface any emitter failure as a gate fail
        return False, [f"emitter raised: {type(e).__name__}: {e}"]
    found = [m for m in _MARKERS if m in src]
    return (len(found) == 0), found
