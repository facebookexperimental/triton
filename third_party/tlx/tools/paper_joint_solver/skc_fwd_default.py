"""SKC instance — generated, do not edit.

Solution : None
DDG      : None
Skeleton : skc.skeleton_fwd (verified handwritten-kernel protocol)

Binding audit:
  {
    "mode": "default-params (M1 baseline; no solution bound)",
    "register_budget_check": "43520 <= 65536 OK"
  }
"""

import sys
from pathlib import Path

# instances live in the paper_joint_solver dir, next to the skc package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skc.skeleton_fwd import attention as _skeleton_entry  # noqa: E402

PARAMS = {
    "BLOCK_M": 256,
    "BLOCK_N": 128,
    "NUM_BUFFERS_KV": 3,
    "NUM_BUFFERS_QK": 1,
    "NUM_MMA_GROUPS": 2,
    "MMA_PV_SKEW": 0,
    "REGS_SOFTMAX": 152,
    "REGS_MMA": 24,
    "REGS_LOAD": 24
}

AUDIT = {
  "mode": "default-params (M1 baseline; no solution bound)",
  "register_budget_check": "43520 <= 65536 OK"
}


def attention(*args, **kwargs):
    return _skeleton_entry(*args, params=PARAMS, **kwargs)
