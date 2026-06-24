# Subtiled Region Epilogue Buffer Fusion Investigation

## Status — RESOLVED

Root cause confirmed exactly as hypothesized below: in `doCodePartitionPost`,
the reuse-group **consumer-merge** fires for the subtiled-region path (both
staging channels feed the same `ttng.subtiled_region`, so `getDstOp()` matches),
which removes the non-representative channel from `orderedChannels`.
`replaceBufferReuse` iterated `orderedChannels` and only inspected
`it->second.front()`, so the merged-out channel was never visited and its
duplicate physical `local_alloc` was never collapsed onto the representative —
leaving two double-buffered `2x256x64xf16` staging buffers (+64 KiB → OOM).

**Fix (implemented):** `replaceBufferReuse` now iterates `config->groups`
directly and folds every non-representative channel into `channels[0]`,
independent of merge/order bookkeeping. The generic SMEM `replaceUsesOfWith`
already rewrites the `ttng.subtiled_region` operands. The consumer-merge is
unchanged (it is still required for shared barrier/comm generation). See
`WSCodePartition.cpp::replaceBufferReuse` and the `ReuseGroups.md`
"Buffer Replacement" section.

**Regression test:**
`test/Hopper/WarpSpecialization/ws_subtiled_region_buffer_reuse_collapse.mlir`
(asserts a single `allocation.shareGroup` epilogue staging alloc survives).
E2E: `test_tutorial09_matmul_tma_persistent_warp_specialize[...generate_subtiled_region=True...EPILOGUE_SUBTILE=2...BLOCK_SIZE_M=256...]`
now passes (was the OOM).

The original investigation notes are retained below for context.

## Repro

Failing test:

```bash
LD_LIBRARY_PATH=/home/njriasan/miniconda3/envs/triton_oss/lib:$LD_LIBRARY_PATH \
TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=1 \
/home/njriasan/miniconda3/envs/triton_oss/bin/python -m pytest -q -s \
'python/test/unit/language/test_tutorial09_warp_specialization.py::test_tutorial09_matmul_tma_persistent_warp_specialize[True-True-1-True-False-False-2-False-4-3-64-128-256-8192-8192-1024]'
```

This is the `generate_subtiled_region=True` variant. The matching passing case is the same node id with `True-False-1-...`, i.e. `generate_subtiled_region=False`.

Observed failure:

```text
OutOfResources: shared memory, Required: 279044, Hardware limit: 232448
```

Useful dump files from the investigation:

- Full failing MLIR dump: `/tmp/triton_mlir_subtile_true_full.log`
- Final IR dumps for false: `/tmp/triton_ir_subtile_false/5FMRTWZYPI3EW4HHOKGZS7L2ZFOEZDSI6ME7R56HVLASKK4YMSHQ/`
- Final IR dumps for true: `/tmp/triton_ir_subtile_true/5FMRTWZYPI3EW4HHOKGZS7L2ZFOEZDSI6ME7R56HVLASKK4YMSHQ/`

## Key Observation

The TTIR is identical between `generate_subtiled_region=False` and `True`.

Before `doGenerateSubtiledRegion`, the memory planner already has two logical epilogue staging allocs, one per static epilogue slice:

```mlir
%_0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32, buffer.tmaStaging = 1 : i32}
  : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
%_1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 2 : i32, buffer.tmaStaging = 1 : i32}
  : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
```

In the `generate_subtiled_region=False` path, `doCodePartitionPost` fuses these into one physical double-buffered allocation:

```mlir
%_0_36 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 2 : i32, buffer.id = 2 : i32}
  : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
```

In the `generate_subtiled_region=True` path, final TTGIR keeps two physical double-buffered allocations:

```mlir
%_0_28 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 2 : i32, buffer.id = 2 : i32}
  : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
%_1 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 2 : i32, buffer.id = 2 : i32}
  : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
```

Each allocation is `2 * 256 * 64 * sizeof(f16) = 65536` bytes, so the second physical allocation explains the extra 64 KiB shared-memory usage and the OOM.

## Where The Divergence Appears

`MLIR_ENABLE_DUMP=1` shows the second physical `2x256x64xf16` allocation first appears after:

```text
// -----// WarpSpec internal IR Dump After: doCodePartition
```

It does not first appear in `doMemoryPlanner`; memory planner only has two scalar logical staging allocs.

With `generate_subtiled_region=True`, `GenerateSubtiledRegion.cpp` wraps the epilogue chain and then creates a second `ttng.subtiled_region` for the TMA store users. Relevant area:

```text
lib/Dialect/TritonNvidiaGPU/Transforms/GenerateSubtiledRegion.cpp
  "Build a second SubtiledRegionOp for TMA store ops..."
```

The generated IR has:

```mlir
ttng.subtiled_region ... %_1, %_0 ... {
  ttg.local_store ... %acc_slices_19
}

ttng.subtiled_region ... %_1, %_0 ... {
  ttng.async_tma_copy_local_to_global ... %acc_slices_19
}
```

So the two logical SMEM buffers become per-tile operands to subtiled regions.

## Suspected Root Cause

The issue appears to be in buffer-fusion/reuse replacement after channel merging, not in the initial creation of two logical subtile buffers.

In `doCodePartitionPost`:

```text
third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp
```

The code builds reuse groups from channels sharing the same `buffer.id`, then merges channels with the same consumer:

```cpp
channelsGroupedByConsumers[rep].push_back(ch);
channelsGroupedByConsumers.erase(ch);
mergedChannels.insert(ch);
...
orderedChannels.erase(... mergedChannels ...);
```

In the subtiled-region path, both channels' consumer can look like the same `ttng.subtiled_region`, so this merge fires.

Later, `replaceBufferReuse` only iterates `orderedChannels`:

```cpp
for (auto *key : orderedChannels) {
  ...
}
```

If a channel was removed from `orderedChannels` by the consumer merge, it may never have its allocation users rewritten to the representative allocation. That leaves both physical SMEM allocations alive, even though both carry the same `allocation.shareGroup`.

## Fix Direction

Investigate making buffer-reuse replacement cover all channels in each reuse group, including channels merged out of `orderedChannels`.

Likely fix shape:

- Keep consumer-channel merging for barrier/comm generation.
- Change the SMEM buffer replacement step so it iterates the reuse groups themselves, or otherwise includes merged channels.
- For SMEM reuse entries with identical memdesc types, replace all uses of the non-representative alloc result with the representative alloc result.
- This replacement must include `ttng.subtiled_region` operands.

Expected fixed final TTGIR for `generate_subtiled_region=True`:

- Only one epilogue staging allocation of type `!ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>`.
- Both subtile paths use views/indices of that one allocation.

## Files To Inspect

- `lib/Dialect/TritonNvidiaGPU/Transforms/GenerateSubtiledRegion.cpp`
  - Around the second TMA-store subtiled-region construction.
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`
  - Channel merge in `doCodePartitionPost`.
  - `createBufferPost`.
  - `replaceBufferReuse`.
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.cpp`
  - `createChannelPost`.
  - `ChannelPost::getSrcOp`.
  - `ChannelPost::getDstOp`.
