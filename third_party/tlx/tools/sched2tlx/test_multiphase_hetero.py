#!/usr/bin/env python3
"""Multi-phase emitter regression test: HETEROGENEOUS sibling phases.

Fixture: `test_multiphase_hetero.schedule_graph.json` — the pool-mode schedule
dump of the retired case9_hetero_dual_gemm example (phase 0 fp16 (128,128)x2
rings II 512; phase 1 bf16 (128,64)/(64,128)x3 rings II 256). GPU-free.

Asserts what the homogeneous case cannot:
  * HETEROGENEOUS pool membership — different shapes, dtypes, AND ring
    depths share the one storage_alias_spec;
  * TWO distinct epilogue staging signatures (f16 c_smem + bf16 c_smem_1),
    i.e. staging is shared per-(shape,dtype), duplicated across dtypes;
  * per-phase ring depths preserved from the schedule (2 vs 3).

Run: `python3 test_multiphase_hetero.py` (stdlib only, no GPU).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE = SCRIPT_DIR / "test_multiphase_hetero.schedule_graph.json"

sys.path.insert(0, str(SCRIPT_DIR))
from sched2tlx import emitter, schedule_graph  # noqa: E402


def main() -> int:
    g = schedule_graph.load_graph(str(FIXTURE))
    assert [loop.smem_phase_group for loop in g.loops] == [0, 1]
    src = emitter.emit(g)

    # One pooled spec; members span both dtypes, shapes, and depths.
    assert src.count("tlx.storage_alias_spec(storage=tlx.storage_kind.smem)") == 1
    ring_allocs = {
        m.group(1): m.group(0)
        for l in src.splitlines()
        if (m := re.search(r"(L[01]_smem_\d+) = tlx\.local_alloc\(.*", l))
    }
    assert all("reuse=smem_pool" in a for a in ring_allocs.values()), ring_allocs
    phase0 = [a for v, a in ring_allocs.items() if v.startswith("L0_")]
    phase1 = [a for v, a in ring_allocs.items() if v.startswith("L1_")]
    assert phase0 and all("tl.float16, 2" in a for a in phase0), phase0
    assert phase1 and all("tl.bfloat16, 3" in a for a in phase1), phase1
    assert any("(128, 64)" in a for a in phase1) and any(
        "(64, 128)" in a for a in phase1
    ), phase1

    # Overlap tree covers both phases.
    overlaps = [l for l in src.splitlines() if "set_buffer_overlap" in l]
    assert len(overlaps) == 1 and "L0_smem_0" in overlaps[0] and "L1_smem_0" in overlaps[0]

    # Two staging signatures: f16 c_smem and bf16 c_smem_1.
    staging = {
        l.split(" = ")[0].strip(): l
        for l in src.splitlines()
        if "= tlx.local_alloc" in l and "c_smem" in l
    }
    assert set(staging) == {"c_smem", "c_smem_1"}, staging
    dtypes = {re.search(r"tl\.(float16|bfloat16)", l).group(1) for l in staging.values()}
    assert dtypes == {"float16", "bfloat16"}, staging

    print("PASS test_multiphase_hetero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
