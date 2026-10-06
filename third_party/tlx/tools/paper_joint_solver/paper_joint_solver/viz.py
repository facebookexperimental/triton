"""Figure-9-style rendering of the loop DDG (paper's G) as Graphviz DOT.

The paper's Figure 9 draws the FMHA forward dependence graph with every node
filled by its assigned warp group: the variable-latency (TMA) group, the
Tensor-Core GEMM group, two softmax groups and the accumulator-rescale group.
This module reproduces that view from a loaded Problem plus a JointSolution's
warp-set map (and optionally its modulo-schedule cycles, shown as a second
label line ``t=<cycle>``).  The paper assigns each operation a set of physical
warps, so the colored entity is the warp set, not a group label.

Zero-cost scalar helpers (latency == occupancy == 0: address arithmetic,
expand_dims/broadcast, view-only allocs) carry no schedule information; they
are contracted out of the picture, with their edges re-routed so dependence
chains between the surviving tile ops stay visible.  Loop-carried edges
(distance >= 1) are dashed, matching the paper's back-edge convention.

Colors: a fixed 8-slot qualitative pastel palette, assigned to the sorted set
of distinct warp sets.  Sparse or non-canonical ids therefore remain distinct;
graphs with more than eight distinct warp sets are rejected rather than
silently reusing a color.  The legend and per-node labels carry warp identity
in text so color is never the only channel.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .ddg import Problem, load_problem
from .schedule_plan import PAPER_SOLUTION_SCHEMA, load_schedule_context

# One fill per used warp group, up to the paper figure's eight-color limit.
# Pastel so black labels stay readable; adjacent slots alternate
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


def _warp_label(warps: tuple[int, ...]) -> str:
    return "warp " + ",".join(str(warp) for warp in warps)


def _warp_colors(
    warp_sets: dict[int, tuple[int, ...]] | None,
) -> dict[tuple[int, ...], str]:
    if warp_sets is None:
        return {}
    used = sorted({tuple(warps) for warps in warp_sets.values()})
    if len(used) > len(PALETTE):
        raise ValueError(
            f"cannot render {len(used)} distinct warp sets with the "
            f"{len(PALETTE)}-color palette: {used}"
        )
    return {warps: PALETTE[index] for index, warps in enumerate(used)}


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


def to_dot(prob: Problem, warp_sets: dict[int, tuple[int, ...]] | None,
           cycles: dict[int, int] | None = None) -> str:
    """Emit the warp-colored dependence graph (paper Fig 9) as DOT."""
    kept, edges = _contract(prob)
    warp_colors = _warp_colors(warp_sets)
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
        w = tuple(warp_sets[vid]) if warp_sets and vid in warp_sets else None
        fill = warp_colors[w] if w is not None else UNASSIGNED_FILL
        tip = f"node {vid}: {v.op_kind} [{v.pipeline}]"
        if w is not None:
            tip += f", {_warp_label(w)}"
        lines.append(f'  n{vid} [label="{label}", fillcolor="{fill}",'
                     f' tooltip="{tip}"];')
    for src, dst, distance in sorted(edges):
        attrs = f' [style=dashed, label="+{distance}"]' if distance else ""
        lines.append(f"  n{src} -> n{dst}{attrs};")
    if warp_sets is not None:
        used = sorted(warp_colors)
        if used:
            lines.append("  subgraph cluster_legend {")
            lines.append('    label="warp sets"; fontname="Helvetica";'
                         ' fontsize=10; style=rounded; color="#9a9a94";')
            for index, w in enumerate(used):
                fill = warp_colors[w]
                lines.append(f'    legend_w{index} [label="{_warp_label(w)}",'
                             f' fillcolor="{fill}", fontsize=10];')
            for index in range(len(used) - 1):
                lines.append(f"    legend_w{index} -> legend_w{index + 1}"
                             " [style=invis];")
            lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(prob: Problem, warp_sets: dict[int, tuple[int, ...]] | None,
           out_path: str | Path,
           cycles: dict[int, int] | None = None) -> Path:
    """Write ``out_path`` (.dot) and, when the ``dot`` binary is on PATH,
    render the .svg next to it (skipped silently otherwise)."""
    out_path = Path(out_path)
    out_path.write_text(to_dot(prob, warp_sets, cycles))
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
    ap.add_argument("ddg", help="curated-ddg-0.2 artifact (see curate_ddg)")
    ap.add_argument("solution", nargs="?", default=None,
                    help='JSON with {"warp_sets": {id: [w]}, "cycles":'
                    " {id: t}}; omit for an uncolored graph")
    ap.add_argument("-o", "--out", default="ddg_warp.dot",
                    help="output .dot path (default: %(default)s)")
    ap.add_argument("--loop-index", type=int, default=0)
    args = ap.parse_args(argv)

    prob = None
    warp_sets = cycles = None
    if args.solution:
        sol = json.loads(Path(args.solution).read_text())
        if sol.get("schema_version") == PAPER_SOLUTION_SCHEMA:
            if args.loop_index != 0:
                ap.error("versioned solution rendering only supports --loop-index 0")
            prob, plan = load_schedule_context(args.solution, args.ddg)
            warp_sets = plan.warp_sets
            cycles = plan.cycles
        else:
            warp_sets = {
                int(k): tuple(sorted(int(w) for w in v))
                for k, v in sol.get("warp_sets", {}).items()
            }
            if sol.get("cycles"):
                cycles = {int(k): int(v) for k, v in sol["cycles"].items()}
    if prob is None:
        prob = load_problem(args.ddg, loop_index=args.loop_index)
    render(prob, warp_sets, args.out, cycles=cycles)


if __name__ == "__main__":
    main()
