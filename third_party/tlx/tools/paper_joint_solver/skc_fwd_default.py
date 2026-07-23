"""SKC instance — generated, do not edit.

Solution : None
DDG      : None
Skeleton : skc.skeleton_fwd (verified blackwell_fa_ws protocol)

Binding audit:
  {
    "mode": "default-params (M1 baseline; no solution bound)",
    "register_budget_check": "43520 <= 65536 OK"
  }
"""

import sys
from pathlib import Path

sys.path.insert(0, '/projects/kzhou6/hwu27/triton-beta-3-paper-solver-wt/third_party/tlx/tools/paper_joint_solver')

from skc.skeleton_fwd import attention as _skeleton_attention  # noqa: E402

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


def attention(q, k, v, sm_scale):
    return _skeleton_attention(q, k, v, sm_scale, PARAMS)
