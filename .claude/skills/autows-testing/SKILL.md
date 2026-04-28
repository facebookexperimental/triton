---
name: autows-testing
description: >
  Run autoWS (automatic warp specialization) correctness tests. Use when
  working on autoWS compiler code — files under WarpSpecialization/, partition
  scheduling, warp_specialize ops, WSCodePartition, WSDataPartition,
  WSTaskPartition, WSMemoryPlanner, or related passes. Do NOT use TLX
  correctness tests (third_party/tlx/tutorials/testing/test_correctness.py)
  for autoWS work — those test manual warp specialization via TLX, not the
  automatic compiler pipeline.
---

# AutoWS Correctness Testing

**Do NOT run `third_party/tlx/tutorials/testing/test_correctness.py` for autoWS.**
Those tests cover manual warp specialization via TLX, which is a separate system.

The canonical test list lives in `third_party/nvidia/hopper/run_all.sh` — check
that file if the list below seems out of date.

## Python tests

```bash
# GEMM autoWS Python test
pytest python/test/unit/language/test_tutorial09_warp_specialization.py

# Addmm autoWS Python test
pytest python/test/unit/language/test_autows_addmm.py

# FA autoWS tutorial kernels
TRITON_ALWAYS_COMPILE=1 pytest python/tutorials/fused-attention-ws-device-tma.py
TRITON_ALWAYS_COMPILE=1 python python/tutorials/test_tlx_bwd_from_fused_attention.py

# FA autoWS Hopper tutorial kernel
TRITON_USE_META_PARTITION=1 TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 pytest python/tutorials/fused-attention-ws-device-tma-hopper.py
```

## LIT tests

Run all WarpSpecialization LIT tests:

```bash
lit test/Hopper/WarpSpecialization/
```

## If tests hang

Run `third_party/tlx/killgpu.sh` to kill GPU processes that have been running too long.
