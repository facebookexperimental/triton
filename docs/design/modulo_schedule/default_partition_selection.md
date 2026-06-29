# Modulo Native WS — Default-Partition Selection & Warp/Register Allocation

When the modulo scheduler lowers a loop to warp-specialized partitions, one
partition must be the **default** (index 0): it runs the non-specialized /
leftover code (including the function-scope epilogue) and, critically, it is the
**register sink**. This doc defines the resource-grounded rule for (a) which
partition is the default and (b) per-partition warp counts, and records why the
epilogue placement lives in modulo (not the backend).

## Hardware mechanism: the default partition is the register sink
PTXAS distributes the 65,536 registers/SM evenly across all allocated warps
(fewer warps ⇒ more registers/thread). Warp specialization then redistributes
with `setmaxnreg`:

- Async producer warp groups (TMA, MMA — 1 warp each) `setmaxnreg.dec` to
  ~24 registers/thread, releasing their share.
- The released surplus flows to the **default** partition, which `setmaxnreg.inc`
  (capped at 256).

Backend references (this tree):
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`
  (`nTotalRegs = 1 << 16`).
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`
  (workers → `lowRegs = 24`; default → surplus).
- `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` (leftover registers go
  to the front/default partition).

Consequence: the default should be the **highest register-footprint** partition,
so the producers' released registers are actually used. A light epilogue as the
default wastes the surplus.

## Existing algorithm (industry + literature)
- **CudaDMA** (Bauer et al., NVIDIA Research, SC'11): the foundational warp
  specialization — split warps into DMA producers + compute consumers. *Singe*
  generalized it into a compiler.
- **CUTLASS 3.x Hopper/Blackwell warp-specialized persistent GEMM; FlashAttention-3:**
  specialize the lightweight async **producer** (TMA) into its own small warp
  group that `setmaxnreg.dec`s; keep the heavy **consumer** (MMA + softmax +
  epilogue) as the main/**default** warp group that `setmaxnreg.inc`s. The
  consumer/default is the register sink.
- **Performance model:** "A Performance Model for Warp Specialization Kernels"
  (arXiv 2506.11209).
- **Backend heuristic already present:** `PartitionSchedulingMeta.cpp`
  `needs4Warps` promotion swaps a 4-warp (TMEM/wgmma) partition to index 0;
  `getDefaultPartition()` fallback = correction → reduction → epilogue →
  last-computation.

Canonical pattern: **specialize producers OUT; leave the heavy consumer as the
default** (the inverse of reserving an empty epilogue default).

## Per-op minimum warps
`LatencyModel::getMinWarps`: async (TMA, MMA, barriers) = 1; tile-parallel
(CUDA arith, SFU/exp2, reduce, TMEM, SMEM/cvt on tensors) = 4 (inferred from a
`RankedTensorType` result). A partition's `num_warps` = `snapWarps(max op
min_warps)` ∈ {1, 2, 4, 8}.

## The general rule
Per schedule partition `p`:
1. `num_warps[p] = snapWarps(max over ops of min_warps)` ∈ {1, 2, 4, 8}.
2. `reg_footprint[p]` ≈ Σ(register-resident tensor result elements × dtype bytes)
   (skip `memdesc` results, which live in SMEM/TMEM, not registers).
3. **default = argmax `reg_footprint[p]` among partitions with `num_warps >= 4`.**
   Tie-break: most CUDA/SFU tile-parallel work.
4. Fold leftover/unassigned ops (epilogue, scalar setup) into the default.
5. Producers (TMA/MMA, 1 warp, ~24 regs) are never the default — they are the
   register donors.
6. Budget: `num_warps[default] + Σ workers <= 16` (`kMaxWarps`); kernel num_warps
   `% 4 == 0`; per-partition power-of-2. If over budget, merge the lightest
   workers or drop WS.

## What modulo emits (`emitScheduleFromGraph`)
- Per-op `ttg.partition` (warp group), with index 0 reserved as the default.
- The function-scope **epilogue** is stamped `ttg.partition = 0` via a forward
  dataflow walk from `loop.getResults()`, so the backend anchors it in the
  default warp group (it would otherwise hoist above the loop and read the
  accumulator before the MMA writes it). This keeps `PartitionSchedulingMeta`
  pristine — it just honors `ttg.partition`.
- Per-loop `ttg.partition_num_warps` and `ttg.partition.stages`.

## Register-footprint default selection (gated)
`TRITON_MODULO_REG_DEFAULT` (default off): instead of reserving a separate empty
default warp group, pick the highest register-footprint 4-warp consumer as the
default (index 0) and fold the epilogue into it. This implements the register-sink
rule above. With the flag off, behavior is the legacy reserved-empty default.

Example (FA forward): the flag drops the kernel from 16 → 12 warps by eliminating
the redundant empty default warp group (one fewer partition), giving the consumer
more registers/thread from the 65,536 pool.
