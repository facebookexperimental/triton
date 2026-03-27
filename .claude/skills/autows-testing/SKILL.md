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

## Python tests

```bash
# GEMM autoWS Python test
pytest python/test/unit/language/test_tutorial09_warp_specialization.py

# Addmm autoWS Python test
pytest python/test/unit/language/test_autows_addmm.py

# Flash Attention autoWS Python test
pytest python/test/unit/language/test_autows_flash_attention.py

# Additional autoWS tests
pytest python/test/unit/language/test_warp_specialization.py
```

## LIT tests

Run individual LIT tests with `triton-opt` (after rebuilding if C++ changed):

```bash
# Run all WarpSpecialization LIT tests
python bin/run-lit.py test/Hopper/WarpSpecialization/

# Or run individual tests:
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_task_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_task_id_propagation.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_data_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_code_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_code_partition_merged_barrier.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_memory_planner.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_memory_planner_bwd.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_memory_planner_merged_barrier.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/ws_memory_planner_split_copy.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/fa_code_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/blackwell_fa_code_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/blackwell_ws_data_partition.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/blackwell_ws_matmul_tma.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/swap_transposed_local_alloc.mlir
python bin/run-lit.py test/Hopper/WarpSpecialization/1D_tmem.mlir
```

### Additional upstream LIT tests

```bash
python bin/run-lit.py test/Conversion/warp_specialize_to_llvm.mlir
python bin/run-lit.py test/TritonGPU/partition-scheduling.mlir
python bin/run-lit.py test/TritonGPU/partition-loops.mlir
```

## If tests hang

Run `third_party/tlx/killgpu.sh` to kill GPU processes that have been running too long.
