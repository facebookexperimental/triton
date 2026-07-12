#!/usr/bin/env python3
"""Multi-phase emitter regression test: REJECTION paths.

The multi-phase driver's safety contract: any sibling-loop schedule graph it
does not support must raise (NotImplementedError), never silently emit a
kernel with a phase dropped or mis-sequenced (the pre-driver emitter silently
dropped every unclaimed sibling loop — the failure mode these tests pin).

Both fixtures are derived in-memory from `test_multiphase_homo.schedule_graph
.json`:
  * persistent/nested phase among siblings (loop marked is_outer);
  * cross-phase SSA dependency (a pre-phase-0 function-scope op consuming a
    phase's scf.for result — violates segment/E validation).

Run: `python3 test_multiphase_reject.py` (stdlib only, no GPU).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE = SCRIPT_DIR / "test_multiphase_homo.schedule_graph.json"

sys.path.insert(0, str(SCRIPT_DIR))
from sched2tlx import emitter, schedule_graph  # noqa: E402


def _emit_variant(d: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        path = f.name
    emitter.emit(schedule_graph.load_graph(path))


def main() -> int:
    base = json.loads(FIXTURE.read_text())

    # 1. Persistent/nested phase among siblings must be rejected.
    d = copy.deepcopy(base)
    d["loops"][1]["is_outer"] = True
    try:
        _emit_variant(d)
        raise AssertionError("outer-among-siblings emitted silently")
    except NotImplementedError as e:
        assert "multi-phase" in str(e), e

    # 2. Cross-phase SSA dependency must be rejected: make an early
    # function-scope op consume a phase's scf.for result.
    d = copy.deepcopy(base)
    loop_op_id = next(
        oid for oid, op in d["ops"].items()
        if op["kind"] == "scf.for" and op.get("scope") == "function"
    )
    victim = next(
        op for oid, op in d["ops"].items()
        if op.get("scope") == "function"
        and op["kind"] == "tt.make_tensor_descriptor"
    )
    victim["operands"].append({"op": loop_op_id, "result": 0})
    try:
        _emit_variant(d)
        raise AssertionError("cross-phase SSA dep emitted silently")
    except NotImplementedError as e:
        assert "multi-phase" in str(e), e

    print("PASS test_multiphase_reject")
    return 0


if __name__ == "__main__":
    sys.exit(main())
