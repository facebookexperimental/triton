"""SKC instance — generated, do not edit.

Solution : subtiled_fa4exact_solution.json
DDG      : ../sched2tlx/examples/case3_FA_fp16_subtiled/ddg.json
Skeleton : skc.skeleton_fwd (verified blackwell_fa_ws protocol)

Binding audit:
  {
    "mode": "solution-bound",
    "roles": {
      "load": 0,
      "mma": 3,
      "softmax": [
        1,
        2
      ],
      "correction": 4,
      "fingerprints": {
        "4": {
          "kinds": [
            "arith.addi",
            "arith.mulf",
            "arith.muli",
            "ttg.local_alloc",
            "ttng.tmem_load"
          ],
          "pipes": [
            "CUDA",
            "NONE",
            "TMEM"
          ]
        },
        "0": {
          "kinds": [
            "tt.descriptor_load"
          ],
          "pipes": [
            "TMA"
          ]
        },
        "3": {
          "kinds": [
            "ttg.local_alloc",
            "ttg.memdesc_trans",
            "ttng.tc_gen5_mma",
            "ttng.tmem_load",
            "ttng.tmem_store"
          ],
          "pipes": [
            "NONE",
            "TC",
            "TMEM"
          ]
        },
        "1": {
          "kinds": [
            "arith.addf",
            "arith.maxnumf",
            "arith.mulf",
            "arith.subf",
            "arith.truncf",
            "math.exp2",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reduce",
            "ttg.convert_layout",
            "ttng.tmem_alloc"
          ],
          "pipes": [
            "CUDA",
            "NONE",
            "SFU"
          ]
        },
        "2": {
          "kinds": [
            "arith.addf",
            "arith.mulf",
            "arith.subf",
            "arith.truncf",
            "math.exp2",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reduce",
            "ttg.convert_layout",
            "ttng.tmem_alloc"
          ],
          "pipes": [
            "CUDA",
            "NONE",
            "SFU"
          ]
        }
      }
    },
    "ii": 66,
    "length": 148,
    "copies": 3,
    "qk_mmas": [
      7,
      31
    ],
    "pv_mmas": [
      28,
      52
    ],
    "protocol_overrides": [
      {
        "node": 8,
        "op_kind": "ttng.tmem_load",
        "solver_warp": 3,
        "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
      },
      {
        "node": 27,
        "op_kind": "ttng.tmem_store",
        "solver_warp": 3,
        "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
      },
      {
        "node": 32,
        "op_kind": "ttng.tmem_load",
        "solver_warp": 3,
        "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
      },
      {
        "node": 49,
        "op_kind": "ttng.tmem_load",
        "solver_warp": 3,
        "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
      },
      {
        "node": 51,
        "op_kind": "ttng.tmem_store",
        "solver_warp": 3,
        "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
      }
    ],
    "chain_order": [
      1,
      2
    ],
    "pingpong_offset_norm": 9,
    "mma_issue_order": {
      "frame": [
        [
          7,
          5
        ],
        [
          31,
          14
        ],
        [
          28,
          104
        ],
        [
          52,
          113
        ]
      ],
      "steady_state_mod_ii": [
        [
          7,
          5
        ],
        [
          31,
          14
        ],
        [
          28,
          38
        ],
        [
          52,
          47
        ]
      ],
      "pv_skew_stages": 1
    },
    "depth_detail": {
      "K:2": {
        "birth": 2,
        "death": 70,
        "span": 68,
        "consumers": [
          [
            7,
            0,
            61
          ],
          [
            31,
            0,
            70
          ]
        ],
        "copies": 2
      },
      "V:3": {
        "birth": 98,
        "death": 148,
        "span": 50,
        "consumers": [
          [
            28,
            0,
            139
          ],
          [
            52,
            0,
            148
          ]
        ],
        "copies": 1
      },
      "QK:7": {
        "birth": 5,
        "death": 69,
        "span": 64,
        "consumers": [
          [
            8,
            0,
            69
          ]
        ],
        "copies": 1
      },
      "QK:31": {
        "birth": 14,
        "death": 80,
        "span": 66,
        "consumers": [
          [
            32,
            0,
            80
          ]
        ],
        "copies": 2
      }
    },
    "geometry": {
      "sub_m": 64,
      "block_n": 64,
      "source": "qk tmem_load result type",
      "realized_split": 128
    },
    "dropped": [
      "solver row block SUB_M=64 < 128-row TMEM tile minimum of this build; row block realized at 128 per chain (schedule-preserving scale-up)",
      "solver cycle values are kept only as issue order / chain order / liveness depths; exact normalized offsets are not representable in a static-program skeleton"
    ],
    "manual_overrides": [
      "MMA_PV_SKEW=0"
    ],
    "register_budget_check": "43520 <= 65536 OK"
  }
"""

import sys
from pathlib import Path

sys.path.insert(0, '/projects/kzhou6/hwu27/triton-beta-3-paper-solver-wt/third_party/tlx/tools/paper_joint_solver')

from skc.skeleton_fwd import attention as _skeleton_attention  # noqa: E402

PARAMS = {
    "BLOCK_M": 256,
    "BLOCK_N": 64,
    "NUM_BUFFERS_KV": 3,
    "NUM_BUFFERS_QK": 2,
    "NUM_MMA_GROUPS": 2,
    "MMA_PV_SKEW": 0,
    "REGS_SOFTMAX": 152,
    "REGS_MMA": 24,
    "REGS_LOAD": 24
}

AUDIT = {
  "mode": "solution-bound",
  "roles": {
    "load": 0,
    "mma": 3,
    "softmax": [
      1,
      2
    ],
    "correction": 4,
    "fingerprints": {
      "4": {
        "kinds": [
          "arith.addi",
          "arith.mulf",
          "arith.muli",
          "ttg.local_alloc",
          "ttng.tmem_load"
        ],
        "pipes": [
          "CUDA",
          "NONE",
          "TMEM"
        ]
      },
      "0": {
        "kinds": [
          "tt.descriptor_load"
        ],
        "pipes": [
          "TMA"
        ]
      },
      "3": {
        "kinds": [
          "ttg.local_alloc",
          "ttg.memdesc_trans",
          "ttng.tc_gen5_mma",
          "ttng.tmem_load",
          "ttng.tmem_store"
        ],
        "pipes": [
          "NONE",
          "TC",
          "TMEM"
        ]
      },
      "1": {
        "kinds": [
          "arith.addf",
          "arith.maxnumf",
          "arith.mulf",
          "arith.subf",
          "arith.truncf",
          "math.exp2",
          "tt.broadcast",
          "tt.expand_dims",
          "tt.reduce",
          "ttg.convert_layout",
          "ttng.tmem_alloc"
        ],
        "pipes": [
          "CUDA",
          "NONE",
          "SFU"
        ]
      },
      "2": {
        "kinds": [
          "arith.addf",
          "arith.mulf",
          "arith.subf",
          "arith.truncf",
          "math.exp2",
          "tt.broadcast",
          "tt.expand_dims",
          "tt.reduce",
          "ttg.convert_layout",
          "ttng.tmem_alloc"
        ],
        "pipes": [
          "CUDA",
          "NONE",
          "SFU"
        ]
      }
    }
  },
  "ii": 66,
  "length": 148,
  "copies": 3,
  "qk_mmas": [
    7,
    31
  ],
  "pv_mmas": [
    28,
    52
  ],
  "protocol_overrides": [
    {
      "node": 8,
      "op_kind": "ttng.tmem_load",
      "solver_warp": 3,
      "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
    },
    {
      "node": 27,
      "op_kind": "ttng.tmem_store",
      "solver_warp": 3,
      "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
    },
    {
      "node": 32,
      "op_kind": "ttng.tmem_load",
      "solver_warp": 3,
      "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
    },
    {
      "node": 49,
      "op_kind": "ttng.tmem_load",
      "solver_warp": 3,
      "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
    },
    {
      "node": 51,
      "op_kind": "ttng.tmem_store",
      "solver_warp": 3,
      "rule": "R1: mma issuer only issues tcgen05+barriers; TMEM traffic follows the skeleton protocol (softmax reads qk / stores p, correction rescales acc)"
    }
  ],
  "chain_order": [
    1,
    2
  ],
  "pingpong_offset_norm": 9,
  "mma_issue_order": {
    "frame": [
      [
        7,
        5
      ],
      [
        31,
        14
      ],
      [
        28,
        104
      ],
      [
        52,
        113
      ]
    ],
    "steady_state_mod_ii": [
      [
        7,
        5
      ],
      [
        31,
        14
      ],
      [
        28,
        38
      ],
      [
        52,
        47
      ]
    ],
    "pv_skew_stages": 1
  },
  "depth_detail": {
    "K:2": {
      "birth": 2,
      "death": 70,
      "span": 68,
      "consumers": [
        [
          7,
          0,
          61
        ],
        [
          31,
          0,
          70
        ]
      ],
      "copies": 2
    },
    "V:3": {
      "birth": 98,
      "death": 148,
      "span": 50,
      "consumers": [
        [
          28,
          0,
          139
        ],
        [
          52,
          0,
          148
        ]
      ],
      "copies": 1
    },
    "QK:7": {
      "birth": 5,
      "death": 69,
      "span": 64,
      "consumers": [
        [
          8,
          0,
          69
        ]
      ],
      "copies": 1
    },
    "QK:31": {
      "birth": 14,
      "death": 80,
      "span": 66,
      "consumers": [
        [
          32,
          0,
          80
        ]
      ],
      "copies": 2
    }
  },
  "geometry": {
    "sub_m": 64,
    "block_n": 64,
    "source": "qk tmem_load result type",
    "realized_split": 128
  },
  "dropped": [
    "solver row block SUB_M=64 < 128-row TMEM tile minimum of this build; row block realized at 128 per chain (schedule-preserving scale-up)",
    "solver cycle values are kept only as issue order / chain order / liveness depths; exact normalized offsets are not representable in a static-program skeleton"
  ],
  "manual_overrides": [
    "MMA_PV_SKEW=0"
  ],
  "register_budget_check": "43520 <= 65536 OK"
}


def attention(q, k, v, sm_scale):
    return _skeleton_attention(q, k, v, sm_scale, PARAMS)
