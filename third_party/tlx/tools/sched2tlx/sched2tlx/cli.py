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
        "--pingpong",
        action="store_true",
        help="paired-tile ping-pong lowering (FA fwd pattern only; two M-tiles "
        "per CTA, replicated softmax WGs; see sched2tlx/pingpong.py)",
    )
    args = p.parse_args(argv)

    g = schedule_graph.load_graph(args.graph)
    if args.pingpong:
        from . import pingpong

        src = pingpong.emit_pingpong(g)
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
