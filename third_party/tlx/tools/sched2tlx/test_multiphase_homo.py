#!/usr/bin/env python3
"""Multi-phase emitter regression test: HOMOGENEOUS sibling phases.

Fixture: `test_multiphase_homo.schedule_graph.json` — the pool-mode schedule
dump of the retired case8_dual_gemm example (two identical fp16 128x128x128
GEMM K-loops, smem_phase_group 0/1). GPU-free: asserts on the emitted source.

Asserts the general multi-phase driver's contract:
  * emission succeeds (no NotImplementedError) and auto-dispatches;
  * exactly ONE SMEM storage_alias_spec, with ONE set_buffer_overlap tree of
    shape shared(distinct(phase0 rings), distinct(phase1 rings));
  * exactly ONE inter-phase join barrier (N=2 -> 1 boundary), every task
    arrives (arrive_count == task count) and waits;
  * same-signature epilogues SHARE one staging alloc (single c_smem) —
    per-(shape,dtype) sharing, not per-phase duplication;
  * the legacy canonical `acc_tmem` name is never minted in multi-phase
    output (every phase accumulator is per-var, e.g. acc_tmem_<id>).

Run: `python3 test_multiphase_homo.py` (stdlib only, no GPU).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE = SCRIPT_DIR / "test_multiphase_homo.schedule_graph.json"

sys.path.insert(0, str(SCRIPT_DIR))
from sched2tlx import emitter, schedule_graph  # noqa: E402


def main() -> int:
    g = schedule_graph.load_graph(str(FIXTURE))
    assert [loop.smem_phase_group for loop in g.loops] == [0, 1], (
        "fixture must carry pool-mode phase groups 0/1"
    )
    src = emitter.emit(g)

    # One pooled spec, one overlap tree: shared(distinct(...), distinct(...)).
    assert src.count("tlx.storage_alias_spec(storage=tlx.storage_kind.smem)") == 1
    overlaps = [l for l in src.splitlines() if "set_buffer_overlap" in l]
    assert len(overlaps) == 1, overlaps
    tree = overlaps[0]
    assert tree.count("reuse_group_type.distinct") == 2, tree
    assert tree.count("reuse_group_type.shared") == 1, tree
    # Both phases' rings are members (L0_* and L1_* referenced in the tree).
    assert "L0_smem_0" in tree and "L1_smem_0" in tree, tree

    # Every phase ring alloc opts into the pool.
    ring_allocs = [
        l for l in src.splitlines()
        if re.search(r"L[01]_smem_\d+ = tlx\.local_alloc", l)
    ]
    assert ring_allocs and all("reuse=smem_pool" in l for l in ring_allocs), (
        ring_allocs
    )

    # Exactly one join barrier (N=2), full task-set rendezvous.
    join_allocs = re.findall(r"mp_join_0 = tlx\.alloc_barriers\((.*)\)", src)
    assert len(join_allocs) == 1, join_allocs
    assert re.search(r"mp_join_1\b", src) is None, "N=2 must have 1 boundary"
    m = re.search(r"arrive_count=(\d+)", join_allocs[0])
    # async_task( / async_task("default") only — NOT the async_tasks() region.
    n_tasks = len(re.findall(r"with tlx\.async_task\(", src))
    assert m and int(m.group(1)) == n_tasks, (join_allocs[0], n_tasks)

    # Same-signature epilogues share ONE staging alloc.
    staging = [
        l.split(" = ")[0].strip()
        for l in src.splitlines()
        if "= tlx.local_alloc" in l and "c_smem" in l
    ]
    assert staging == ["c_smem"], staging

    # Canonical acc_tmem is never minted in multi-phase output.
    assert re.search(r"\bacc_tmem = ", src) is None
    assert re.search(r"\bacc_tmem\[", src) is None

    print("PASS test_multiphase_homo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
