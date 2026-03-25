# Token & Barrier Lowering

Token lowering is the step that converts abstract synchronization primitives
(NVWS dialect tokens) into concrete hardware mbarrier operations. Tokens are
created during code partitioning to represent producer-consumer synchronization
points. This pass materializes them as SMEM-allocated mbarrier arrays.

**File**: `WSLowerToken.cpp`
**Function**: `doTokenLowering(funcOp, numConsumerGroups)`

## Pipeline Context

```
doCodePartitionPost     ← creates CreateTokenOp, ProducerAcquireOp, etc.
  → specializeRegion    ← clones ops into WarpSpecializeOp regions
  → doPingPongSync      ← inserts named barrier ops
  → doTokenLowering     ← THIS STEP: tokens become hardware barriers
```

Token lowering runs **after** code specialization, operating on the ops
inside `WarpSpecializeOp` regions.

## Abstract Token Operations

The NVWS dialect defines these abstract synchronization ops:

| Op | Purpose |
|----|---------|
| `CreateTokenOp` | Allocates a synchronization token with `numBuffers` slots |
| `ProducerAcquireOp` | Producer waits for a buffer slot to be free |
| `ProducerCommitOp` | Producer signals that data is ready |
| `ConsumerWaitOp` | Consumer waits for data to be available |
| `ConsumerReleaseOp` | Consumer signals that it has finished reading |
| `TMAStoreTokenWaitOp` | Special wait for TMA store completion |

## Lowering Algorithm

### Step 1: Allocate Barrier Arrays

For each `CreateTokenOp`, allocate two mbarrier arrays in SMEM:

- **`bufferFull`** (ready barriers): `numBuffers` entries. Signals data
  availability from producer to consumer.
- **`bufferEmpty`** (empty barriers): `numBuffers` entries. Signals buffer
  slot availability from consumer to producer.

Each barrier is initialized with `InitBarrierOp` with the appropriate arrive
count (typically 1 for single-producer or single-consumer).

### Step 2: Lower Token Operations

Each abstract token op is converted to the corresponding hardware barrier
operation:

| Abstract Op | Lowered To | Barrier Array | Description |
|-------------|-----------|---------------|-------------|
| `ProducerAcquireOp` | `WaitBarrierOp` | `bufferEmpty[i]` | Wait for consumer to release buffer slot |
| `ProducerCommitOp` | `ArriveBarrierOp` | `bufferFull[i]` | Signal data is ready for consumer |
| `ConsumerWaitOp` | `WaitBarrierOp` | `bufferFull[i]` | Wait for producer to fill buffer slot |
| `ConsumerReleaseOp` | `ArriveBarrierOp` | `bufferEmpty[i]` | Signal buffer slot is free for producer |

The barrier index `i` is derived from the buffer index (which buffer slot
in the multi-buffered pipeline).

### Step 3: Phase Computation

Each barrier wait requires a **phase bit** that alternates across uses:

- **Ready barriers** (`bufferFull`): Phase is computed directly from
  `accumCnt / numBuffers`.
- **Empty barriers** (`bufferEmpty`): Phase is XORed with 1 relative to the
  ready barrier phase, ensuring proper initial synchronization (the producer
  must be able to acquire the first slot without waiting).

The phase computation via `getMBarrierPhaseBit()`:
```
phase = (accumCnt / numBuffers) & 1
emptyPhase = phase ^ 1  // inverted for empty barriers
```

### Step 4: Update Captures

Token values that cross the `WarpSpecializeOp` boundary are replaced with
their materialized barrier array values in the capture list. Each token
capture becomes two captures (the ready and empty barrier arrays).

### Step 5: Handle TMA Store Tokens

`TMAStoreTokenWaitOp` is handled specially — it is lowered by adding real
barriers for the TMA store's SMEM buffer. This ensures the SMEM buffer is
not reused before the TMA store finishes reading from it.

## Relationship to Barrier Fusion

Token lowering happens **after** barrier fusion. By the time tokens are
lowered, channels that share barriers (from TMA fusion or channel grouping
in `doCodePartitionPost`) already share the same `CreateTokenOp`. This means
the lowering naturally produces shared mbarrier allocations for fused
channels.

See [Barrier Fusion](BarrierFusion.md) for details on how barriers are
shared before lowering.
