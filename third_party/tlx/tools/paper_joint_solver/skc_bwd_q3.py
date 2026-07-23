"""SKC instance — generated, do not edit.

Solution : None
DDG      : None
Skeleton : skc.skeleton_bwd (verified handwritten-kernel protocol)

Binding audit:
  {
    "mode": "default-params (M1 baseline; no solution bound)",
    "manual_overrides": [
      "NUM_BUFFERS_Q=3"
    ],
    "register_budget_check": "17920 <= 65536 OK"
  }
"""

import sys
from pathlib import Path

# instances live in the paper_joint_solver dir, next to the skc package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skc.skeleton_bwd import bwd_attention as _skeleton_entry  # noqa: E402

PARAMS = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "NUM_BUFFERS_Q": 3,
    "REGS_MMA": 88,
    "REGS_REDUCTION": 88,
    "REGS_LOAD": 24
}

AUDIT = {
  "mode": "default-params (M1 baseline; no solution bound)",
  "manual_overrides": [
    "NUM_BUFFERS_Q=3"
  ],
  "register_budget_check": "17920 <= 65536 OK"
}


def bwd_attention(*args, **kwargs):
    return _skeleton_entry(*args, params=PARAMS, **kwargs)
