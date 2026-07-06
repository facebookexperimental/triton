"""CLI: python -m twill_solver <schedule_graph.json> -o <optimized_graph.json>

Reads a modulo schedule graph, runs the Twill-inspired solver, and writes an
optimized schedule graph that ``python -m sched2tlx`` lowers to TLX.
"""

from __future__ import annotations

import argparse
import sys

from . import cpsat_model, ddg_model, emit_check, graph_writer
from .baseline import extract as extract_baseline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", help="input schedule_graph.json (modulo dump)")
    p.add_argument("-o", "--output", help="optimized schedule_graph.json path")
    p.add_argument("--ddg", help="companion ddg.json (default: sibling ddg.json)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--reproduce", action="store_const", dest="mode",
                      const="reproduce", help="emit the baseline schedule verbatim")
    mode.add_argument("--improve", action="store_const", dest="mode",
                      const="buffering",
                      help="propose Twill optimizations (eager MMA operand release "
                      "when double-buffered; latency-hiding buffer depth). This is a "
                      "PROPOSAL — run twill_solver.testing.ab --ship to A/B it on GPU; "
                      "that gate is what guarantees no regression.")
    mode.add_argument("--joint", action="store_const", dest="mode",
                      const="joint", help="[M2] full joint schedule + WS")
    p.add_argument("--time-limit", type=float, default=10.0,
                   help="CP-SAT wall-clock budget per solve (s)")
    p.add_argument("--emit-check", action="store_true",
                   help="run the sched2tlx emitter dry-run gate before writing")
    p.set_defaults(mode="reproduce")
    args = p.parse_args(argv)

    model = ddg_model.load(args.graph, args.ddg)
    base = extract_baseline(model)

    opts = cpsat_model.SolveOptions(mode=args.mode, time_limit_s=args.time_limit)
    sol = cpsat_model.solve(model, opts)

    out = args.output or (str(args.graph).rsplit(".json", 1)[0] + ".optimized.json")
    graph_writer.write(model, sol, out)

    # Report what changed vs baseline.
    _report(base, sol)

    if args.emit_check:
        ok, markers = emit_check.dry_run(out)
        if not ok:
            print(f"[twill] EMIT-CHECK FAILED: {markers}", file=sys.stderr)
            return 2
        print("[twill] emit-check: OK (no placeholders)", file=sys.stderr)

    print(f"[twill] wrote {out} (mode={args.mode})", file=sys.stderr)
    return 0


def _report(base, sol) -> None:
    for lid in sorted(sol.loops):
        b, s = base.loops[lid], sol.loops[lid]
        deltas = []
        for bid in sorted(s.buffer_count):
            if s.buffer_count[bid] != b.buffer_count.get(bid):
                deltas.append(f"buf{bid}:{b.buffer_count.get(bid)}->{s.buffer_count[bid]}")
        wgs = f"{len(b.warp_groups)}->{len(s.warp_groups)} WGs" if len(b.warp_groups) != len(s.warp_groups) else f"{len(s.warp_groups)} WGs"
        tag = ", ".join(deltas) if deltas else "no buffer change"
        print(f"[twill] loop {lid}: II {b.II}->{s.II}, {wgs}, {tag}",
              file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
