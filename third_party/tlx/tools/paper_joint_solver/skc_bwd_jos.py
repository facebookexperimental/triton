"""SKC instance — generated, do not edit.

Solution : bwd_skc_solution.json
DDG      : ../sched2tlx/examples/case4_FA_bwd/ddg_hd128.json
Skeleton : skc.skeleton_bwd (verified handwritten-kernel protocol)

Binding audit:
  {
    "mode": "solution-bound",
    "roles": {
      "load": 0,
      "mma": 2,
      "compute": 3,
      "reduction": 1,
      "fingerprints": {
        "0": {
          "kinds": [
            "tt.descriptor_load",
            "tt.load"
          ],
          "pipes": [
            "TMA"
          ]
        },
        "2": {
          "kinds": [
            "arith.truncf",
            "tt.broadcast",
            "tt.expand_dims",
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "ttng.tc_gen5_mma",
            "ttng.tmem_load"
          ],
          "pipes": [
            "CUDA",
            "NONE",
            "TC",
            "TMEM"
          ]
        },
        "1": {
          "kinds": [
            "tt.expand_dims",
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "ttng.tmem_load"
          ],
          "pipes": [
            "NONE",
            "TMEM"
          ]
        },
        "3": {
          "kinds": [
            "arith.addi",
            "arith.mulf",
            "arith.subf",
            "arith.truncf",
            "math.exp2",
            "tt.addptr",
            "tt.broadcast",
            "tt.descriptor_reduce",
            "tt.splat",
            "ttg.convert_layout",
            "ttng.tmem_alloc",
            "ttng.tmem_load"
          ],
          "pipes": [
            "CUDA",
            "NONE",
            "SFU",
            "TMEM"
          ]
        }
      }
    },
    "ii": 95,
    "length": 273,
    "copies": 3,
    "protocol_overrides": [
      {
        "node": 25,
        "op_kind": "ttng.tmem_load",
        "solver_warp": 2,
        "rule": "R1: mma issuer only issues tcgen05+barriers"
      },
      {
        "node": 7,
        "op_kind": "tt.load",
        "solver_warp": 0,
        "rule": "skeleton loads M/D via tl.load in the compute task (model keeps them on the VL warp)"
      },
      {
        "node": 9,
        "op_kind": "tt.load",
        "solver_warp": 0,
        "rule": "skeleton loads M/D via tl.load in the compute task (model keeps them on the VL warp)"
      }
    ],
    "depth_detail": {
      "load:0": {
        "birth": 0,
        "death": 197,
        "span": 197,
        "consumers": [
          [
            36,
            0,
            197
          ],
          [
            11,
            0,
            69
          ]
        ],
        "copies": 3,
        "mma_consumers": 2
      },
      "load:2": {
        "birth": 37,
        "death": 254,
        "span": 217,
        "consumers": [
          [
            22,
            0,
            254
          ],
          [
            21,
            0,
            107
          ]
        ],
        "copies": 3,
        "mma_consumers": 2
      }
    },
    "geometry": {
      "block_m": 64,
      "block_n": 128,
      "source": "Q descriptor rows / qkT read rows",
      "realized_block_m": 128
    },
    "dropped": [
      "solver M block 64 < 128-row TMEM tile minimum of this build; realized at 128 (schedule-preserving scale-up)",
      "liveness Q copies=3 exceeds the skeleton's SMEM accounting (max ring 2 after barrier/padding overhead the model does not price) \u2014 clamped"
    ],
    "mma_frame_lags": {
      "11": 0,
      "21": 0,
      "22": 2,
      "31": 1,
      "36": 1
    },
    "register_budget_check": "17920 <= 65536 OK"
  }
"""

import sys
from pathlib import Path

sys.path.insert(0, '/projects/kzhou6/hwu27/triton-beta-3-paper-solver-wt/third_party/tlx/tools/paper_joint_solver')

from skc.skeleton_bwd import bwd_attention as _skeleton_entry  # noqa: E402

PARAMS = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "NUM_BUFFERS_Q": 2,
    "REGS_MMA": 88,
    "REGS_REDUCTION": 88,
    "REGS_LOAD": 24
}

AUDIT = {
  "mode": "solution-bound",
  "roles": {
    "load": 0,
    "mma": 2,
    "compute": 3,
    "reduction": 1,
    "fingerprints": {
      "0": {
        "kinds": [
          "tt.descriptor_load",
          "tt.load"
        ],
        "pipes": [
          "TMA"
        ]
      },
      "2": {
        "kinds": [
          "arith.truncf",
          "tt.broadcast",
          "tt.expand_dims",
          "ttg.local_alloc",
          "ttg.memdesc_trans",
          "ttng.tc_gen5_mma",
          "ttng.tmem_load"
        ],
        "pipes": [
          "CUDA",
          "NONE",
          "TC",
          "TMEM"
        ]
      },
      "1": {
        "kinds": [
          "tt.expand_dims",
          "ttg.local_alloc",
          "ttg.memdesc_trans",
          "ttng.tmem_load"
        ],
        "pipes": [
          "NONE",
          "TMEM"
        ]
      },
      "3": {
        "kinds": [
          "arith.addi",
          "arith.mulf",
          "arith.subf",
          "arith.truncf",
          "math.exp2",
          "tt.addptr",
          "tt.broadcast",
          "tt.descriptor_reduce",
          "tt.splat",
          "ttg.convert_layout",
          "ttng.tmem_alloc",
          "ttng.tmem_load"
        ],
        "pipes": [
          "CUDA",
          "NONE",
          "SFU",
          "TMEM"
        ]
      }
    }
  },
  "ii": 95,
  "length": 273,
  "copies": 3,
  "protocol_overrides": [
    {
      "node": 25,
      "op_kind": "ttng.tmem_load",
      "solver_warp": 2,
      "rule": "R1: mma issuer only issues tcgen05+barriers"
    },
    {
      "node": 7,
      "op_kind": "tt.load",
      "solver_warp": 0,
      "rule": "skeleton loads M/D via tl.load in the compute task (model keeps them on the VL warp)"
    },
    {
      "node": 9,
      "op_kind": "tt.load",
      "solver_warp": 0,
      "rule": "skeleton loads M/D via tl.load in the compute task (model keeps them on the VL warp)"
    }
  ],
  "depth_detail": {
    "load:0": {
      "birth": 0,
      "death": 197,
      "span": 197,
      "consumers": [
        [
          36,
          0,
          197
        ],
        [
          11,
          0,
          69
        ]
      ],
      "copies": 3,
      "mma_consumers": 2
    },
    "load:2": {
      "birth": 37,
      "death": 254,
      "span": 217,
      "consumers": [
        [
          22,
          0,
          254
        ],
        [
          21,
          0,
          107
        ]
      ],
      "copies": 3,
      "mma_consumers": 2
    }
  },
  "geometry": {
    "block_m": 64,
    "block_n": 128,
    "source": "Q descriptor rows / qkT read rows",
    "realized_block_m": 128
  },
  "dropped": [
    "solver M block 64 < 128-row TMEM tile minimum of this build; realized at 128 (schedule-preserving scale-up)",
    "liveness Q copies=3 exceeds the skeleton's SMEM accounting (max ring 2 after barrier/padding overhead the model does not price) \u2014 clamped"
  ],
  "mma_frame_lags": {
    "11": 0,
    "21": 0,
    "22": 2,
    "31": 1,
    "36": 1
  },
  "register_budget_check": "17920 <= 65536 OK"
}


def bwd_attention(*args, **kwargs):
    return _skeleton_entry(*args, params=PARAMS, **kwargs)
