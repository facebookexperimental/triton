# Cross-Partition Run-Once Atomic Support (Dynamic Persistent)

**Files**: `WSAtomicBroadcast.cpp` (the pass), `WarpSpecialization.cpp`
(pipeline wiring), with supporting fixes in `WSCodePartition.cpp` and
`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`.

> The original design exploration lives in
> [`CrossPartitionAtomicSupport.proposal.md`](CrossPartitionAtomicSupport.proposal.md).
> This document describes what was actually implemented.

## Problem

A dynamic (work-stealing) persistent kernel claims its next tile from a global
counter inside a persistent `scf.while`:

```python
tile_id = tl.program_id(0)
while tile_id < num_tiles:
    ... compute tile ...
    tile_id = tl.atomic_add(tile_counter, 1)   # claim next tile
```

AutoWS clones the persistent `scf.while` once per partition. That is correct for
the *pure* static update (`tile_id += NUM_SMS`) but **wrong** for the
side-effecting `tl.atomic_add`: task-id propagation assigns the atomic (whose
result is the loop-carried value used by every partition) to **all** partitions,
so each warp group bumps the counter independently, diverges onto a different
tile, and the producer/consumer barriers never match → **runtime deadlock**.

## Pass: `doAtomicBroadcast`

Runs in `WarpSpecialization.cpp::runOnFuncOp` immediately **after**
`doTaskIdPropagate` (so partitions are materialized as `async_task_id`) and
before `doBufferAllocation`. For each `tt.atomic_rmw` it classifies into one of
three cases:

| Case | Condition | Action |
|------|-----------|--------|
| **1. Pass-through** | mapped to a single partition | leave unchanged |
| **2. Transform** | scalar, mapped to *every* partition, and its result is the loop-carried value of an enclosing persistent `scf.while` | run once + broadcast (below) |
| **3. Reject** | any other shape (non-scalar/scatter, strict-subset replication, or not loop-carried) | bail out of WS gracefully |

### Case 2 — transform

`transformAtomic` installs only the **data path** and lets the existing AutoWS
channel machinery provide synchronization:

1. Retag the atomic to the **owner** partition alone (the TMA-load/producer
   partition, located via the loop's `tt.descriptor_load` /
   `async_tma_copy_global_to_local`).
2. In the owner: `tt.splat` the scalar result into a single-element SMEM slot
   (`ttg.local_alloc` at function scope) via `ttg.local_store`.
3. In every partition: `ttg.local_load` the slot and `tt.unsplat` back to a
   scalar.
4. Rewire the `scf.while`'s loop-carried tile id to the broadcast (unsplat)
   value, so the atomic now feeds only the slot store and is cloned into the
   owner partition alone.

The producer→consumer synchronization is **not** hand-synthesized. Because the
`local_store` (owner) and `local_load` (all partitions) form an ordinary
cross-partition SMEM dependency, `doCodePartitionPost` turns it into (N) SMEM
channels with the correct full/empty mbarriers and phase — i.e. the broadcast is
expressed in terms of an existing, tested AutoWS channel rather than a bespoke
handshake.

### Case 3 — graceful reject

`doAtomicBroadcast` returns `failure()`. The caller then calls the canonical
`removeWarpSpecializeAttr(funcOp)`, which strips **both** the partition ids
(`ttg.partition`) and the task ids (`async_task_id`) from every op, plus the WS
loop attributes (`tt.warp_specialize`, `ttg.partition.stages`,
`ttg.partition.types`, `ttg.warp_specialize.tag`) from `scf.for`/`scf.while`
loops. The kernel is left unspecialized-but-compilable (never a crash/assert).
This is the same teardown used by the other AutoWS bail-outs (`numWarps < 4`,
`scf.if` else-block); `removeWarpSpecializeAttr` clears `async_task_id` too
because this reject runs *after* propagation, unlike the earlier bail-outs.

## Two supporting fixes this feature required

1. **`scf.while` phase for a direct channel producer** (`WSCodePartition.cpp`).
   The broadcast channel's producer (`local_store`) lives directly in the
   persistent `scf.while` after-region. The buffer-index/phase computation only
   handled a producer inside an `scf.for`; a producer directly in a `scf.while`
   fell back to a **constant phase**, so the channel's mbarrier parity did not
   toggle across persistent iterations → deadlock. The fix routes a
   `scf::WhileOp`-parented producer through `getBufferIdxAndPhase` (which, via
   `getAccumCount`, resolves the counter from the while's after-region args),
   exactly as for `scf.for`.

2. **Single-element tensors excluded from register-class estimation**
   (`OptimizePartitionWarps.cpp`). The scalar round-trips through SMEM as a
   single-element `tensor<1xi32>`. Counting it would flip an otherwise
   non-tensor partition into the "tensor" register class (request `-1` instead
   of the fixed minimum), which shifts warp-group register budgeting. The fix
   ignores single-element tensors for the register estimate while still
   relayouting them when a partition's warp count changes.

## Tunable: broadcast depth

The broadcast-channel depth is a compile-time knob, not an env var:

- Python: `knobs.nvidia.ws_tile_prefetch_depth` (default 1).
- Threaded to the `nvgpu-warp-specialization` pass as the `tile-prefetch-depth`
  option (`Passes.td` → `triton_nvidia.cc` → `compiler.py`).

Depth 1 is the single-stage broadcast. (Multi-stage run-ahead, `depth > 1`, is
future work: it additionally requires drain-on-exit because the `atomic_add` is
a destructive claim.)

## Tests

- E2E: `test_tutorial09_matmul_tma_dynamic_persistent_while_loop_warp_specialize`
  (un-xfailed; runs on Hopper or Blackwell) asserts correctness for
  `EPILOGUE_SUBTILE ∈ {1,2,4}`.
- LIT: `test/Hopper/WarpSpecialization/ws_atomic_broadcast_transform.mlir`
  (case 2 — exactly one atomic + broadcast) and `ws_atomic_broadcast_reject.mlir`
  (case 3 — no WS / no partition ids / no task ids left, atomic preserved).
