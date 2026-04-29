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

## Why Tokens Exist

Tokens are an IR-level abstraction that separates **where and what to
synchronize** from **how to synchronize on hardware**. Every cross-partition
data dependency (TMA-backed, software async copy, local store, TMEM) uses
tokens for its producer-consumer protocol — they are not specific to any
single channel type.

The compiler could in principle emit raw `LocalAllocOp` (for mbarrier SMEM),
`InitBarrierOp`, `WaitBarrierOp`, and `ArriveBarrierOp` directly during code
partitioning. Tokens exist because that would tangle synchronization
placement logic with hardware-specific barrier management in a pass that is
already ~950 lines (`insertAsyncComm`). The concrete advantages:

### Separation of concerns across pipeline stages

Code partitioning (`WSCodePartition.cpp`) focuses on **what** needs to be
synchronized — which data flows cross partition boundaries, which channels
can share barriers, and where acquire/commit/wait/release should be placed.
It does not need to know:

- How many threads are in a warp group (needed for arrive counts)
- Whether the barrier should use TMA hardware auto-arrive (arrive count 1)
  vs. software arrive (arrive count = `THREADS_PER_WARP * numWarps`)
- How to compute the phase bit and its XOR inversion for empty barriers
- How to thread mbarrier memdescs through `WarpSpecializePartitionsOp`
  capture lists

All of that is deferred to `WSLowerToken.cpp`.

### Clean survival across code specialization

Code specialization (`specializeRegion`) clones the IR into per-partition
regions inside `WarpSpecializeOp`. Token SSA values cross the region
boundary via the op's capture list and become block arguments — trivial
because a token is a single opaque `!nvws.token` value.

If raw mbarrier memdescs were used instead, specialization would need to
capture **two** barrier arrays per channel (full + empty), correctly map
indices, and handle the fact that different regions use them for different
purposes (producer vs. consumer). Token lowering handles this cleanly
afterward — it replaces each token capture with the two materialized barrier
array captures.

### Same-partition elision

Token lowering detects when a `ProducerCommitOp` and `ConsumerWaitOp` share
the same `async_task_id` — meaning the producer and consumer are in the same
warp group partition. In this case, the synchronization is redundant (program
order within a partition already guarantees correctness), so both ops are
erased. This happens for OperandD channels where the MMA accumulator is both
produced and consumed by the same partition. At the abstract token level this
is a straightforward task-ID check; at the raw mbarrier level it would
require pattern-matching wait/arrive pairs in the same region.

### Barrier sharing composes naturally

Before tokens are lowered, channels grouped by their dominant consumer share
a single `CreateTokenOp`. When lowered, they naturally share the same
mbarrier pair with no extra deduplication. Without the token layer, barrier
fusion would need to run as a post-pass that merges already-allocated
mbarrier arrays — requiring SMEM deallocation, use-chain rewriting, and
careful phase synchronization.

### Centralized phase management

The phase bit logic is subtle: ready barriers (`bufferFull`) use the
computed phase directly, while empty barriers (`bufferEmpty`) XOR the phase
with 1 so that the producer can acquire the first slot without waiting. This
inversion is implemented once in `getMBarrierPhaseBit` during token lowering,
rather than being sprinkled across every site that inserts synchronization.

### Producer-type-aware arrive counts

Each `CreateTokenOp` carries a `TokenLoadType` enum (`TMALoadOp`,
`AsyncLoadOp`, `LocalStoreOp`, `TmemLoadOp`, `None`). During lowering, TMA
loads get an arrive count of 1 (hardware auto-arrive), while non-TMA loads
get `THREADS_PER_WARP * numWarps` (software arrive from every thread). This
decision is made once in `WSLowerToken.cpp` rather than at every barrier
insertion site.

## Abstract Token Operations

The NVWS dialect defines these abstract synchronization ops:

| Op | Purpose |
|----|---------|
| `CreateTokenOp` | Allocates a synchronization token with `numBuffers` slots and a `TokenLoadType` |
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

Each barrier is initialized with `InitBarrierOp` with arrive count 1. The
arrive count depends on the `TokenLoadType`:

- **TMA loads**: `bufferFullCount = 1` (hardware auto-arrives)
- **Non-TMA loads**: `bufferFullCount = THREADS_PER_WARP * producerWarps`
  (software arrives from every thread)
- **Empty barriers**: `bufferEmptyCount = THREADS_PER_WARP * consumerWarps`
  (always software arrive)

### Step 2: Elide Same-Partition Synchronization

Before lowering individual ops, the pass detects `ProducerCommitOp` /
`ConsumerWaitOp` pairs that share the same `async_task_id`. These are in the
same warp-specialize partition where program order already guarantees
correctness, so they are erased. This typically occurs for OperandD channels.

### Step 3: Lower Token Operations

Each remaining abstract token op is converted to the corresponding hardware
barrier operation:

| Abstract Op | Lowered To | Barrier Array | Description |
|-------------|-----------|---------------|-------------|
| `ProducerAcquireOp` | `WaitBarrierOp` | `bufferEmpty[i]` | Wait for consumer to release buffer slot |
| `ProducerCommitOp` | `ArriveBarrierOp` | `bufferFull[i]` | Signal data is ready for consumer |
| `ConsumerWaitOp` | `WaitBarrierOp` | `bufferFull[i]` | Wait for producer to fill buffer slot |
| `ConsumerReleaseOp` | `ArriveBarrierOp` | `bufferEmpty[i]` | Signal buffer slot is free for producer |

The barrier index `i` is derived from the buffer index (which buffer slot
in the multi-buffered pipeline).

### Step 4: Phase Computation

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

### Step 5: Update Captures

Token values that cross the `WarpSpecializeOp` boundary are replaced with
their materialized barrier array values in the capture list. Each token
capture becomes two captures (the ready and empty barrier arrays).

### Step 6: Handle TMA Store Tokens

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
