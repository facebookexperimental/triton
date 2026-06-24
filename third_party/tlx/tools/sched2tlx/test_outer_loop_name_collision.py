#!/usr/bin/env python3
"""Reproducer + regression test for the outer-persistent-loop variable-name
collision in the sched2tlx emitter.

Root cause
----------
The emitter auto-names ops (`div_2`, `mul_7`, ...) with a counter that USED TO
restart at 0 in every scope. The function-scope preamble and each per-warp-group
outer-loop body therefore minted overlapping names. An outer-loop body op then
gets a name already used by the preamble — and because a name reassigned inside
a loop is loop-carried, the body's `mul_7 = ...` (or `div_4 = tile_id // div_4`)
CLOBBERS the preamble's `mul_7 = cdiv(M,128)*cdiv(N,128)` (the loop bound) /
`div_4 = cdiv(N,128)`. The address arithmetic is then corrupt on every iteration
after the first. It only manifests once a program executes >1 outer tile (a
persistent kernel at scale), so small-shape / few-tile runs pass and the bug
surfaces as a CUDA "illegal instruction" / wrong result only at scale.

Fixture
-------
`test_outer_loop_name_collision.schedule_graph.json` is a hand-written minimal
persistent kernel (4 ops, 2 loops): the preamble computes the loop bound as an
`arith.muli` (auto-named `mul_0`), and the persistent outer-loop body has an
`arith.muli` tile offset that the old per-scope counter ALSO auto-names `mul_0`.
The emitted kernel is then `for tile_id in range(0, mul_0, 1): ...; mul_0 =
(tile_id * 128)` — the body reassigns the loop bound `mul_0`, the exact clobber
this bug causes. With the old per-scope counter `mul_0 = ...` appears TWICE; with
the global counter (`RenderCtx.fresh_idx()`) every auto-named variable is unique.

Run: `python3 test_outer_loop_name_collision.py` (stdlib only, no GPU).
Exit 0 / "PASS" after the fix; raises AssertionError before it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE = SCRIPT_DIR / "test_outer_loop_name_collision.schedule_graph.json"

# Op-name prefixes minted by the emitter's `_auto_name` (see _OP_KIND_NAME_PREFIX)
# plus the `trunc_` epilogue names — these are the collision-prone auto-names.
_AUTO_NAME_RE = re.compile(
    r"^(mul|div|rem|add|sub|ext|trunc|exp2|log2|maxf|minf|sqrt|rsqrt|"
    r"addf|subf|mulf|divf|red|expand|bcast|splat|reshape|trans|join|"
    r"addptr|cvt|mdt|tmload)_\d+$"
)


def _load_emitter():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from sched2tlx import emitter, schedule_graph  # noqa: E402

    return emitter, schedule_graph


def _duplicate_auto_names(src: str) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for tgt in re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s", src, re.M):
        if not _AUTO_NAME_RE.match(tgt):
            continue
        if tgt in seen:
            dups.add(tgt)
        seen.add(tgt)
    return sorted(dups)


def main() -> int:
    emitter, schedule_graph = _load_emitter()
    if not FIXTURE.exists():
        raise SystemExit(f"fixture not found: {FIXTURE}")

    src = emitter.emit(schedule_graph.load_graph(str(FIXTURE)))

    dups = _duplicate_auto_names(src)
    assert not dups, (
        "outer-loop-body auto-names collide with the preamble: "
        f"{dups} each assigned more than once. A loop-carried reassignment of a "
        "preamble variable (e.g. `mul_7 = ...` / `div_4 = tile_id // div_4`) "
        "corrupts the tile address arithmetic after the first iteration. Fix: "
        "give every auto-named variable a globally-unique index via "
        "RenderCtx.fresh_idx()."
    )
    print("PASS: all auto-named variables are unique across scopes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
