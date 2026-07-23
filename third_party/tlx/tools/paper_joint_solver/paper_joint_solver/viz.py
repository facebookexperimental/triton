"""Figure-9-style rendering of the loop DDG (paper's G) as Graphviz DOT.

The paper's Figure 9 draws the FMHA forward dependence graph with every node
filled by its assigned warp group: the variable-latency (TMA) group, the
Tensor-Core GEMM group, two softmax groups and the accumulator-rescale group.
This module reproduces that view from a loaded Problem plus a JointSolution's
warp map (and optionally its modulo-schedule cycles, shown as a second label
line ``t=<cycle>``).

Zero-cost scalar helpers (latency == occupancy == 0: address arithmetic,
expand_dims/broadcast, view-only allocs) carry no schedule information; they
are contracted out of the picture, with their edges re-routed so dependence
chains between the surviving tile ops stay visible.  Loop-carried edges
(distance >= 1) are dashed, matching the paper's back-edge convention.

Colors: a fixed 8-slot qualitative pastel palette, indexed by warp id (never
re-ranked), CVD-checked for adjacent-slot separation; the legend and per-node
labels carry warp identity in text so color is never the only channel.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .ddg import Problem, load_problem

# One fill per warp-group slot (machine.MAX_WARPGROUPS == 8), assigned by
# warp id.  Pastel so black labels stay readable; adjacent slots alternate
# hue family and lightness (validated: adjacent-pair CVD deltaE >= 8,
# normal-vision >= 18).  Slot 0 (the paper's W_vl variable-latency warp)
# is the blue.
PALETTE = ("#9ecae1", "#fdae6b", "#6fc7ae", "#f5e17d",
           "#b5a6d4", "#a1d99b", "#e57368", "#f7c5da")
UNASSIGNED_FILL = "#e8e8e4"

# Short display names for the ops the paper's figure labels specially; other
# kinds fall back to the op name with its dialect prefix stripped.
_SHORT_NAMES = {
    "tc_gen5_mma": "TC GEMM",
    "descriptor_load": "TMA load",
    "exp2": "exp",
    "reduce": "row_red",
}


def _short_label(op_kind: str) -> str:
    name = op_kind.rsplit(".", 1)[-1]
    return _SHORT_NAMES.get(name, name)


def _contract(prob: Problem) -> tuple[set[int], set[tuple[int, int, int]]]:
    """Drop zero-latency/zero-occupancy helper nodes, re-routing edges so
    every dependence between two kept nodes survives (distances add up along
    the contracted path).  Returns (kept ids, {(src, dst, distance)})."""
    kept = {v.id for v in prob.nodes.values()
            if v.latency > 0 or v.occupancy > 0}
    succs: dict[int, list[tuple[int, int]]] = {}
    for e in prob.edges:
        succs.setdefault(e.src, []).append((e.dst, e.distance))

    def kept_targets(v: int, dist: int,
                     seen: frozenset[int]) -> list[tuple[int, int]]:
        out = []
        for dst, d in succs.get(v, ()):
            if dst in kept:
                out.append((dst, dist + d))
            elif dst not in seen:
                out.extend(kept_targets(dst, dist + d, seen | {dst}))
        return out

    edges = set()
    for e in prob.edges:
        if e.src not in kept:
            continue
        if e.dst in kept:
            edges.add((e.src, e.dst, e.distance))
        else:
            for dst, dist in kept_targets(e.dst, e.distance,
                                          frozenset((e.dst,))):
                edges.add((e.src, dst, dist))
    return kept, edges


def to_dot(prob: Problem, warp: dict[int, int] | None,
           cycles: dict[int, int] | None = None) -> str:
    """Emit the warp-colored dependence graph (paper Fig 9) as DOT."""
    kept, edges = _contract(prob)
    lines = [
        "digraph ddg {",
        "  rankdir=TB;",
        '  node [shape=box, style="filled,rounded", fontname="Helvetica",'
        ' fontsize=11, color="#3a3a37"];',
        '  edge [color="#6b6b66", arrowsize=0.7];',
    ]
    for vid in sorted(kept):
        v = prob.nodes[vid]
        label = _short_label(v.op_kind)
        if cycles is not None and vid in cycles:
            label += f"\\nt={cycles[vid]}"
        w = warp.get(vid) if warp is not None else None
        fill = PALETTE[w % len(PALETTE)] if w is not None else UNASSIGNED_FILL
        tip = f"node {vid}: {v.op_kind} [{v.pipeline}]"
        if w is not None:
            tip += f", warp {w}"
        lines.append(f'  n{vid} [label="{label}", fillcolor="{fill}",'
                     f' tooltip="{tip}"];')
    for src, dst, distance in sorted(edges):
        attrs = f' [style=dashed, label="+{distance}"]' if distance else ""
        lines.append(f"  n{src} -> n{dst}{attrs};")
    if warp is not None:
        used = sorted({warp[vid] for vid in kept if vid in warp})
        if used:
            lines.append("  subgraph cluster_legend {")
            lines.append('    label="warp groups"; fontname="Helvetica";'
                         ' fontsize=10; style=rounded; color="#9a9a94";')
            for w in used:
                fill = PALETTE[w % len(PALETTE)]
                lines.append(f'    legend_w{w} [label="warp {w}",'
                             f' fillcolor="{fill}", fontsize=10];')
            for a, b in zip(used, used[1:]):
                lines.append(f"    legend_w{a} -> legend_w{b} [style=invis];")
            lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(prob: Problem, warp: dict[int, int] | None, out_path: str | Path,
           cycles: dict[int, int] | None = None) -> Path:
    """Write ``out_path`` (.dot) and, when the ``dot`` binary is on PATH,
    render the .svg next to it (skipped silently otherwise)."""
    out_path = Path(out_path)
    out_path.write_text(to_dot(prob, warp, cycles))
    dot_bin = shutil.which("dot")
    if dot_bin:
        try:
            subprocess.run(
                [dot_bin, "-Tsvg", str(out_path), "-o",
                 str(out_path.with_suffix(".svg"))],
                check=False, capture_output=True)
        except OSError:
            pass
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Render a ddg.json + solution.json as the paper's "
        "Fig-9-style warp-colored dependence graph.")
    ap.add_argument("ddg", help="ddg.json dump (schema ddg-0.1)")
    ap.add_argument("solution", nargs="?", default=None,
                    help='JSON with {"warp": {id: w}, "cycles": {id: t}};'
                    " omit for an uncolored graph")
    ap.add_argument("-o", "--out", default="ddg_warp.dot",
                    help="output .dot path (default: %(default)s)")
    ap.add_argument("--loop-index", type=int, default=0)
    args = ap.parse_args(argv)

    prob = load_problem(args.ddg, loop_index=args.loop_index)
    warp = cycles = None
    if args.solution:
        sol = json.loads(Path(args.solution).read_text())
        warp = {int(k): int(v) for k, v in sol.get("warp", {}).items()}
        if sol.get("cycles"):
            cycles = {int(k): int(v) for k, v in sol["cycles"].items()}
    render(prob, warp, args.out, cycles=cycles)


if __name__ == "__main__":
    main()
