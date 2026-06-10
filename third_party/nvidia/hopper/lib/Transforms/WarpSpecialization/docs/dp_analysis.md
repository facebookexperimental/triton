# FA BWD Data Partition Analysis

## Overview

This document analyzes Triton's `NVGPUWSDataPartition` pass, why it fails on the FA backward kernel, and how CUTLASS FA3 handles the equivalent data partitioning.

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp`

---

## 1. Triton DP Pass Architecture

### Entry Flow

```
NVGPUWSDataPartitionPass::runOnOperation()
  └─ runOnFuncOp()
     └─ doDataPartition(funcOp, dataPartitionFactor)
        ├─ computePartitionScheme()          # Decide WHAT to partition
        │    └─ getSliceToPartition()        # Closure: backward + forward slice
        ├─ rewriteRematerializedOps()        # Create MemDescSubsliceOp views
        ├─ for each partition i:
        │    sliceOp() on all ops            # Clone + retype + adjust
        │    post-slice cleanup (to_be_removed)
        ├─ doDeepCleanup()                   # Remove dead original ops
        │    ├─ Delete unused ops in partitionScheme.ops
        │    └─ ForOpDeadArgElimination      # Remove dead for-op iter args
        ├─ fixTaskId()                       # Propagate async task IDs
        ├─ Handle unpartitioned stores/loads
        ├─ Update FuncOp arg types           # Slice TMA descriptor types
        └─ reorderLoadsToFirstUse()
```

### DataPartitionScheme

The central bookkeeping structure for the pass:

| Field | Purpose |
|-------|---------|
| `numPartitions` | How many slices (typically 2, one per consumer warp group) |
| `ops` (SetVector) | The set of ops that will be partitioned (order-preserving) |
| `opPartitionDims` | For each op, which tensor dimension to slice along (0=M, 1=N) |
| `dotPartitionOperand` | For dot ops, which operand (0=A, 1=B) is sliced |
| `rematerializedOps` | Ops appearing in both partition dimensions — need cloning. Maps op → set of dims |
| `opsToSkip` | Ops explicitly excluded from partitioning (via `undoPartition`) |
| `funcArgPartitionDims` | Function arguments (TMA descriptors) whose block type must be halved. Maps arg index → dim |

**Sentinel:** `noOpPartitionDim = ~0U - 2` — means the op is duplicated (not resized). Used for K-slice dots whose outputs go to atomic stores.

**How fields get updated during closure computation:**

- **`ops`**: `getBackwardSliceToPartition` and `getForwardSliceToPartition` call
  `partitionScheme.ops.insert(op)` for every op they visit. The SetVector preserves
  insertion order. If `insert` returns false (already in set), the dim conflict check
  runs instead.

- **`opPartitionDims`**: Set to `currentDim` when an op is first added to `ops`
  (line ~302: `partitionScheme.opPartitionDims[op] = currentDim`). NOT updated later —
  the first dim assignment wins. If the same op is reached along a different dim, it
  triggers rematerialization rather than overwriting.

- **`dotPartitionOperand`**: Set when a dot is encountered in backward or forward slice.
  For backward: `dotPartitionOperand[dot] = currentDim == 0 ? 0 : 1` (A for dim=0,
  B for dim=1). For forward (K-slice with atomic): also set before switching to
  `noOpPartitionDim`. For forward (normal): same formula.

- **`rematerializedOps`**: Populated when an op is found already in `ops` with a
  DIFFERENT dim. The `rematerializeOp` helper adds the op to `rematerializedOps` with
  both the existing dim and the new dim. The op stays in `ops` with its original dim.

- **`funcArgPartitionDims`**: Set when `getBackwardSliceToPartition` reaches a
  `DescriptorLoadOp` whose descriptor is a function argument (`BlockArgument` of
  `FuncOp`). Maps `argIndex → dim` so the func arg's TMA descriptor type can be
  halved later.

- **`opsToSkip`**: Populated by `undoPartition(op)`, which removes the op from `ops`
  and `opPartitionDims` and adds it to `opsToSkip`. Used to mark ops as explicitly
  excluded. `isPartitioned(op)` returns false for skipped ops.

### Algorithm Detail

#### Phase 1: Closure Computation (`getSliceToPartition`)

Three-pass closure from the dot accumulator:

1. **Backward slice** (`getBackwardSliceToPartition`): Walk producers from the accumulator. For dim=0, traces operand A; for dim=1, traces operand B. At each step:
   - Transpose ops flip the dimension via the inverse permutation
   - ExpandDimsOp adjusts the dimension index
   - Records `dotPartitionOperand` for each dot encountered
   - **Dim conflicts → rematerialization**: When an op is already in `ops` with a different dim, it gets added to `rematerializedOps` with both dims. This happens when the same op is reachable from different dots along different dimensions.

2. **Forward slice** (`getForwardSliceToPartition`): Walk consumers from the accumulator. At each user:
   - **K-slice detection**: When a partitioned value feeds the contraction dimension (K) of a dot, check if the dot's output goes exclusively to atomic stores (`onlyUsedByAtomicStore`). If yes → assign `noOpPartitionDim` (duplicate without resize, each partition atomically reduces). If no → reject (can't partition along K without atomics).
   - Normal dots record their `dotPartitionOperand`
   - **Dim conflicts** here also trigger rematerialization (same as backward)

3. **Second backward slice**: For forward-discovered ops (stores, dots), backward-slice their operands to catch address-computation chains not reachable from the original root.

**How `rematerializedOps` gets populated**: During the backward or forward slice, when
`partitionScheme.ops.insert(op)` finds the op already exists with a DIFFERENT dim in
`opPartitionDims`, it calls `rematerializeOp(op, existingDim, newDim)`. This records
the op in `rematerializedOps` with both dims. The op itself stays in `ops` with the
FIRST dim it was assigned.

**FA BWD example**: The zero constant `%cst_0 = arith.constant dense<0.0> : tensor<128x128xf32>`
is the accumulator for multiple dots:
- qkT, dpT (dim=0) — backward-sliced from the root
- dQ (noOpPartitionDim = 4294967293) — reached via forward slice with K-slice detection

The constant appears along dim=0 AND noOpPartitionDim → rematerialized. The original
serves dim-0 dots (qkT, dpT, dV, dK). A clone serves dQ (noOpPartitionDim — not resized,
stays 128×128). No LocalAllocOps are rematerialized in FA BWD.

#### Phase 2: Threshold & Dimension Selection (`computePartitionScheme`)

For each dot op:
- Compute `sliceSizeM = shapePerCTA[0] / numPartitions`, `sliceSizeN = shapePerCTA[1] / numPartitions`
- **M threshold**: sliceSizeM ≥ 64 → dim 0 is a candidate
- **N threshold**: sliceSizeN ≥ 128 → dim 1 is a candidate
- Try dim 0 first, then dim 1. Use `getSliceToPartition` as a trial — if it succeeds, adopt that scheme

#### Phase 3: Rematerialization (`rewriteRematerializedOps`)

For ops in `rematerializedOps`: skip the first dim (the original op serves it), then
for each additional dim, create a clone and redirect matching users:

- **LocalAllocOp**: Create a `MemDescSubsliceOp` view (NOT added to partition scheme).
  The view provides a sliced window into the original full-size alloc for users
  partitioned along a different dimension.
- **ConstantOp**: Simply clone the constant.
- **User replacement via `dimMatches`**: For each user of the original op, check if the
  user's effective partition dim matches the rematerialized dim (accounting for dim flips
  through TransOp/MemDescTransOp and cross-dim dot operands). If yes, redirect to the clone.

#### Phase 4: Slicing (`sliceOp` + `cloneAndSetResultType`)

For each partition i (0..numPartitions-1), with fresh IRMapping:

1. **Elementwise / alloc / store ops**: Recursively slice operands, then clone and resize via `cloneAndSetResultType`:
   - **Dot guard**: Don't resize if cross-dim partitioned (contraction dim split)
   - Resize `MemDescType`, `RankedTensorType`, or `TensorDescType` along dim

2. **Dot handler**: Slice the partition operand, accumulator, and optionally the other operand.

3. **ForOp handler**: Create a new ForOp with additional init args for sliced values. Transfer the body via `takeBody` (moves, not clones). Map new region args and results. Mark old ForOp for removal.

4. **YieldOp handler**: Append sliced yield operands (only if the parent ForOp actually added corresponding init args).

5. **Post-slice cleanup**: Erase all ops marked `to_be_removed` (old ForOp shells).

#### Phase 5: Deep Cleanup (`doDeepCleanup`)

Iterative loop until fixpoint:
1. Walk ops in `partitionScheme.ops`. Delete ops with no result users (except ForOp/IfOp — let canonicalization handle those). Side-effectful ops go into `opsCanBeTriviallyDead`.
2. Run `ForOpDeadArgElimination` (treats `opsCanBeTriviallyDead` ops as non-existent for liveness analysis) + standard ForOp/IfOp canonicalization via `applyPatternsGreedily`.
3. Repeat.

#### Phase 6: Finalization

1. **fixTaskId**: Propagate `async_task_id` through def-use chains.
2. **Func arg type update**: For each func arg in `funcArgPartitionDims`, halve the TMA descriptor block type.
3. **Reorder loads**: Stable-sort loads by first-use position.

### Key Functions

| Function | Description |
|----------|-------------|
| `computePartitionScheme` | Drives partition: iterates dots, tries dim 0 then dim 1, calls closure walk |
| `getBackwardSliceToPartition` | Walks producers from dot accumulator, adds to ops |
| `getForwardSliceToPartition` | Walks consumers, handles K-slice detection with atomic output |
| `getSliceToPartition` | Combines backward + forward + second backward pass |
| `rewriteRematerializedOps` | Creates MemDescSubsliceOp views for multi-dim allocs |
| `sliceOp` | Core rewriting: clones and resizes ops for a partition offset |
| `cloneAndSetResultType` | Lambda inside sliceOp: clones op, resizes result types |
| `doDeepCleanup` | Iterative deletion of dead original ops + dead arg elimination |
| `fixTaskId` | Propagates async_task_id through def-use chains |

---

## FA BWD: Phase-by-Phase Trace

### Phase 1: Closure Computation

The 5 dots in program order: qkT, dpT, dV, dQ, dK. `computePartitionScheme` iterates
them, but the **first dot's closure walk discovers all 5**, so subsequent dots are
skipped (`isPartitioned() == true`).

**Starting from qkT (dim=0):**

**Backward slice:**
- Accumulator = zero constant → added to ops
- Partition operand (dim=0 → A): K's `local_alloc` → backward-sliced into ops
  (K's alloc, descriptor_load all added to ops with dim=0)
- `dotPartitionOperand[qkT] = 0`

**Forward slice from qkT output:**
- qkT → subf → exp2 (pT) → truncf (ppT) → reaches **dV** dot as A (operand 0)
  - dim=0, feeding A → normal (not K-slice). dV added with dim=0, operand=0
- pT → truncf → mulf → truncf (dsT) → reaches **dQ** dot as A (via tt.trans → cvt_layout)
  - dim=0, feeding A → normal. dQ added with dim=0, operand=0
  - dQ output → reshape/split/descriptor_reduce (atomic) → chain added
- dsT → cvt_layout → reaches **dK** dot as A (operand 0)
  - dim=0, feeding A → normal. dK added with dim=0, operand=0
  - dK output → yield (loop-carried)
- Forward from pT also reaches **dpT** dot indirectly? No — dpT's A is V (outer-loop
  alloc), dpT's B is dO^T. dpT is reached via the second backward pass.

**Second backward pass:**
- From dV: backward-slice its partition operand (ppT, A). ppT → truncf → exp2 →
  already in ops ✓
- From dpT (discovered how?): Actually, dpT is discovered because its output chain
  (dsT computation) is reached by the forward slice. dpT's output feeds into the
  dsT → dQ/dK chain. Let me re-check...

Actually, the exact discovery path for dpT: the forward slice from pT (softmax output)
reaches dpT through the dsT computation chain:
- pT_87 (exp2 output) → `arith.mulf %pT_87, %dsT_95` (line 148) — but dsT_95 uses
  dpT_89 (dpT output). So the backward connection is: dsT_95 ← arith.subf ← dpT_89.
  The second backward pass from dsT operations discovers dpT.

**Result after closure (confirmed by debug trace):**

Discovery order and assignments:

| # | Dot | How discovered | dim | K-slice? | noOpPartitionDim? |
|---|-----|----------------|-----|----------|-------------------|
| 1 | **qkT** | backward from root | 0 | no | no |
| 2 | **dV** | forward (ppT→dV.A) | 0 | no | no |
| 3 | **dQ** | forward (dsT→trans→dQ.A) | 1 (flipped) | **yes** | **yes** (atomic output) |
| 4 | **dK** | forward (dsT→cvt→dK.A) | 0 | no | no |
| 5 | **dpT** | second backward pass | 0 | no | no |

**Key insight on transposes**: dsT feeds BOTH dQ and dK, but dQ's path goes through
`tt.trans {order=[1,0]}` which flips dim 0→1. This means:
- dK: dsT (dim=0) directly → A (dim=0) → **normal partition** (M-dim sliced)
- dQ: dsT (dim=0) → transpose → A (dim=1) → **K-slice detected** (dim=1 feeding A
  means the partitioned value enters the contraction dimension). Since dQ output goes
  exclusively to `descriptor_reduce` (atomic add), `noOpPartitionDim` is assigned.
  Each partition independently computes full dQ and atomically reduces.

K and V: IN ops (backward slice walks through them). They get sliced to 64×128.
Each partition gets its own sliced K/V alloc. dQ uses sliced K as B operand with
noOpPartitionDim — output duplicated at full size, atomically reduced.

### Phase 2: Threshold Selection

- All dots are 128×128, numPartitions=2
- sliceSizeM = 64 ≥ 64 → dim 0 candidate ✓
- sliceSizeN = 64 < 128 → dim 1 NOT candidate (standard threshold)
- Tries dim 0 → succeeds → uses dim 0

### Phase 3–8: Slicing and Cleanup

K/V allocs ARE in ops (backward slice walks through them). They get sliced to 64×128.
Each partition gets its own sliced K/V alloc. dQ uses sliced K (64×128) as B operand
with noOpPartitionDim — the output is duplicated at full size and atomically reduced
via `descriptor_reduce`. No `memdesc_subslice` views needed. Cleanup removes all
original ops normally.

---

## Solution

### Root Cause

The parent code's `onlyUsedByAtomicStore` only found ONE atomic store (broke after
first `DescriptorReduceOp`). FA BWD's dQ has FOUR `descriptor_reduce` ops. The
connectivity check from just one didn't cover all ops in the forward slice, so it
returned false → K-slice rejected → dim 0 failed → pass failed.

### Fix: Multi-Atomic Store Detection

**One change** to `onlyUsedByAtomicStore` in `getForwardSliceToPartition`:

```cpp
// Before (parent): find first atomic store only
Operation *atomicStore;
for (auto op : forwardSlice) {
  if (isa<AtomicRMWOp, DescriptorReduceOp>(op)) {
    atomicStore = op;
    break;  // BUG: only finds first
  }
}

// After (fix): collect ALL atomic stores
SmallVector<Operation *> atomicStores;
for (auto op : forwardSlice) {
  if (isa<AtomicRMWOp, DescriptorReduceOp>(op)) {
    atomicStores.push_back(op);
  }
}
// Walk backward from ALL atomic stores simultaneously
SmallVector<Operation *> queue(atomicStores);
for (auto op : atomicStores)
  forwardSlice.remove(op);
```

### Test Results

All 56 WS lit tests pass (55 passed + 1 expected failure):

| Test | Status |
|------|--------|
| ws_data_partition.mlir (all 5 sub-tests) | PASS |
| ws_data_partition_fa_bwd.mlir | PASS |
| blackwell_ws_data_partition.mlir | PASS |
| ws_data_partition_epilogue_subtile.mlir | PASS |
| ws_data_partition_host_tma_store.mlir | PASS |
| All other WS tests (51 tests) | PASS |

### Why This Works

The parent code's existing infrastructure handles FA BWD correctly once
`onlyUsedByAtomicStore` works:

1. K/V allocs stay IN ops (backward slice walks through them normally)
2. K/V get sliced to 64×128 — each partition gets its own copy
3. dQ's K-slice is detected (dsT→transpose→dQ.A, dim flipped 0→1)
4. `onlyUsedByAtomicStore(dQ.D)` returns true (all 4 descriptor_reduce found)
5. dQ gets `noOpPartitionDim` — duplicated at full size, atomically reduced
6. No `isOuterAlloc`, `memdesc_subslice`, `feedsOuterAlloc`, or func arg type
   propagation changes needed — the parent's slicing, cleanup, and func arg
   update all work correctly as-is
