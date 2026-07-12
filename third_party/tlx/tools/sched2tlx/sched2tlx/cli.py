# pyre-strict
"""CLI: python -m sched2tlx <schedule_graph.json> -o <out.py>"""

from __future__ import annotations

import argparse
import sys

from . import emitter, schedule_graph


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("graph", help="schedule_graph.json (modulo dump)")
    p.add_argument("-o", "--output", help="output .py path (default: stdout)")
    p.add_argument(
        "--phases",
        action="store_true",
        help="multi-phase template lowering for two sibling GEMM nests "
        "(pools SMEM rings via storage_alias_spec when the dump carries "
        "smem_phase_group)",
    )
    args = p.parse_args(argv)

    g = schedule_graph.load_graph(args.graph)
    if args.phases:
        from . import phases

        src = phases.emit(g)
    else:
        src = emitter.emit(g)
    if args.output:
        with open(args.output, "w") as f:
            f.write(src)
        print(f"wrote {args.output} ({len(src)} bytes)", file=sys.stderr)
    else:
        sys.stdout.write(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
