# Memory Lowering

Memory lowering creates the actual async copy operations that transfer data
between partitions. While code partitioning (`WSCodePartition.cpp`) identifies
cross-partition data dependencies and creates abstract channels, memory
lowering materializes the copies — inserting producer-side store/copy
operations and consumer-side load operations through shared memory or tensor
memory.

## Files

| File | Scope |
|------|-------|
| `WSLowerMem.cpp` | Core memory lowering: async copies, TMA fusion |
| `WSTMAStoreLowering.cpp` | Pre-pass: TMA store lowering for WS visibility |
| `TMEMAlloc1D.cpp` | Special case: 1D tensor communication via TMEM |

## Entry Point: `insertAsyncCopy`

**File**: `WSLowerMem.cpp`

`insertAsyncCopy` is the main dispatcher, called from `doCodePartitionPost`
in `WSCodePartition.cpp`. It groups channels by producer operation and
calls the appropriate copy creation function based on the channel type.

## Copy Types

### 1. `createAsyncCopy` — Global-to-Local TMA Copy

For `tt::LoadOp` producers (global memory loads not using TMA descriptors):

**Producer side**:
- Allocates an SMEM buffer (`LocalAllocOp`)
- Creates `AsyncCopyGlobalToLocalOp` to copy from global to shared memory
- The copy is asynchronous — the producer continues after initiating it

**Consumer side**:
- `LocalLoadOp` reads from the SMEM buffer
- A barrier wait ensures the copy has completed before reading

### 2. `createLocalCopy` — Register-to-SMEM Copy

For channels where the source value is in registers:

**Producer side**:
- `LocalStoreOp` writes the register value into an SMEM buffer

**Consumer side**:
- `LocalLoadOp` reads from the SMEM buffer

This is used for non-TMA data that needs to cross partition boundaries
(e.g., intermediate computation results).

### 3. `createSMEMCopy` — SMEM Buffer Replacement

For channels where the source is already a `LocalAllocOp` in shared memory:

Instead of creating a new allocation, the existing alloc is replaced with a
store into the multi-buffered allocation managed by the memory planner. The
consumer reads from the same multi-buffered buffer at the appropriate slot.

### 4. `createTMEMCopy` — Tensor Memory Copy

For TMEM channels (Blackwell only):

**Producer side**:
- `TMEMStoreOp` writes the value into the TMEM allocation

**Consumer side**:
- References to the old `TMEMAllocOp` are replaced with a buffer subview
  (`MemDescIndexOp`) into the multi-buffered TMEM allocation

### 5. `createBufferView` — Multi-Buffer Indexing

A shared helper that creates `MemDescIndexOp` subviews into multi-buffered
allocations. Given an accumulation counter (`accumCnt`), it computes:

```
bufferIdx = accumCnt % numBuffers
```

and returns a view of the corresponding buffer slot.

## TMA Barrier Fusion (`optimizeTMALoads`)

**File**: `WSLowerMem.cpp`

When multiple TMA descriptor loads feed the same consumer (e.g., two operand
loads for the same MMA), they are fused onto a single barrier:

1. **Group by consumer**: Channels sharing the same dominant consumer are
   grouped together.
2. **Shared barrier**: A single pair of barriers (ready + empty) is allocated
   for the group.
3. **Combined expect**: One `BarrierExpectOp` is emitted with the total byte
   count across all loads.
4. **Multiple copies, one wait**: Each `AsyncTMACopyGlobalToLocalOp` references
   the shared barrier. The consumer issues a single `WaitBarrierOp`.

See [Barrier Fusion](BarrierFusion.md) for more details.

## TMA Store Lowering

**File**: `WSTMAStoreLowering.cpp`

TMA store lowering is a **pre-pass** that runs before the main WS pipeline
(`doTMAStoreLowering`). It converts `tt::DescriptorStoreOp` (register-to-global
via TMA) into a three-step sequence visible to the WS pipeline:

1. **`LocalAllocOp`**: Allocate SMEM and store the register data.
2. **`AsyncTMACopyLocalToGlobalOp`**: Async TMA copy from SMEM to global
   memory, producing a token.
3. **`TMAStoreTokenWaitOp`**: Wait for the TMA store to finish reading from
   SMEM before the buffer can be reused.

### Why This Pre-Pass Is Needed

Without this lowering, the WS pipeline would see only the high-level
`DescriptorStoreOp` and would not know about the intermediate SMEM buffer.
By lowering early, the SMEM buffer becomes visible to the memory planner
for allocation and the barrier becomes visible for synchronization.

### `TMAStoreTokenWaitLowering` Pass

A separate pass (`NVGPUTMAStoreTokenWaitLoweringPass`) lowers the abstract
`TMAStoreTokenWaitOp` into concrete operations:
- `TMAStoreWaitOp`: waits for the async TMA store to complete
- `ArriveBarrierOp`: signals the associated barrier that the SMEM buffer
  is now free

## 1D TMEM Allocation

**File**: `TMEMAlloc1D.cpp`

The `TMEM1DAllocator` handles the special case of 1D tensor values that need
to be communicated between partitions via TMEM. TMEM is inherently 2D (M × N
matrix), so 1D values require expansion.

### Algorithm

1. **Expand shape**: The 1D input `[K]` is expanded to 2D `[M, N]` where
   `M × N ≥ K`, choosing dimensions compatible with TMEM layout constraints.

2. **Allocate**: A 2D `TMEMAllocOp` is created with the expanded shape.

3. **Producer side** (`TMEMStore1D`):
   - `ExpandDimsOp`: reshape 1D → 2D
   - Optional `ConvertLayoutOp` for TMEM-compatible layout
   - `TMEMStoreOp`: write to TMEM

4. **Consumer side** (`TMEMLoad1D`):
   - `TMEMLoadOp`: read from TMEM
   - `ReshapeOp`: 2D → 1D
   - `ConvertLayoutOp`: convert to target encoding

### Entry Point

`generate1DAllocations()` walks the function for ops with `tmem.start`
attributes and creates the 1D TMEM channel infrastructure.

### TMEM Subslicing Utilities

`TMEMUtils.h` also provides utilities for carving sub-regions from TMEM
allocations:

- **`sliceAndReinterpretMDTMEM`**: Creates `TMEMSubSliceOp` +
  `MemDescReinterpretOp` to extract a sub-region with a different N dimension
  or element type.
- **`createTMEMDesc`**: Creates a `MemDescType` with
  `TensorMemoryEncodingAttr` for given M/N dimensions.
