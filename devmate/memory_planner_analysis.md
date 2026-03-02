# MemoryPlanner Analysis

**File**: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`

## Overview

The MemoryPlanner is responsible for assigning buffer IDs, copy counts, and memory offsets to shared memory (SMEM) and tensor memory (TMEM) allocations in warp-specialized kernels. It runs as part of the `doMemoryPlanner` top-level function, which orchestrates two sub-planners: `MemoryPlanner` (SMEM) and `MemoryPlannerTmem` (TMEM).

---

## Architecture

```
MemoryPlannerBase (abstract)
├── MemoryPlanner       — SMEM allocations (buffer.id, buffer.copy)
└── MemoryPlannerTmem   — TMEM allocations (buffer.id, buffer.copy, buffer.offset)
```

Both inherit common infrastructure for:
- **Operation ID mapping** (`buildOperationIdMap`): post-order walk assigning monotonic IDs
- **Liveness interval computation** (`computeIntervalFromOps`): min/max operation ID across live ops
- **Channel lookup** (`findChannelForOp`/`findChannelForAlloc`): mapping allocations to their producer/consumer channels

---

## Entry Point: `doMemoryPlanner`

```
doMemoryPlanner(funcOp, numBuffers, readDecisionFile, writeDecisionFile)
```

### Step 0 — Reorder Ops by SWP Schedule (line ~1765)
```cpp
reorderOpsBySchedule(funcOp);
```
For each ForOp with `loop.stage`/`loop.cluster` attributes, reorders ops within each stage so that smaller cluster IDs appear before larger cluster IDs. This ensures the post-order operation ID assignment reflects the pipelined execution order.

### Step 1 — Collect Channels (line ~1768)
```cpp
collectPostChannels(channelsOrigin, funcOp);
```
Gathers all `SMEMPost` and `TMEMPost` channels representing producer→consumer communication pairs. Each channel has:
- An allocation op (the shared buffer)
- A source op (producer)
- Destination ops (consumers)

### Step 1.5 — Identify Cross-Stage Channels (line ~1791)
Iterates all channels and identifies those where the source and destination are in different pipeline stages (`loop.stage` attribute). Logs debug information for SWP-aware planning.

### Step 2 — Deserialize Decisions (optional, line ~1836)
If `TRITON_WS_DECISION_FILE` env var or `readDecisionFile` is set, skip the planner and apply pre-computed buffer decisions from a JSON file. Falls back to running the planner if deserialization fails.

### Step 3 — Run SMEM Planner (line ~1867)
```cpp
triton::MemoryPlanner planner(funcOp, &allocation, &channels);
planner.run(numBuffers);
unsigned bufferId = planner.getLastBufferId();
```

### Step 4 — Dump Visualization (line ~1880)
Dumps combined channel + key-ops graph for debugging (to stderr and/or file).

### Step 5 — Run TMEM Planner (line ~1900)
```cpp
triton::MemoryPlannerTmem planner(funcOp, &allocation, &channels);
planner.run(bufferId);  // continues buffer ID numbering from SMEM planner
```

### Step 6 — Serialize Decisions (optional, line ~1910)
If `writeDecisionFile` is set, serialize all `buffer.id`/`buffer.copy`/`buffer.offset` attributes to JSON.

---

## SMEM Planner (`MemoryPlanner::run`)

### Step 1 — Get Values and Sizes (`getValuesAndSizes`, line 491)
Walks all ops, finds `LocalAllocOp` with shared memory, computes byte sizes (handling padded encodings), and registers them as explicit buffers in the `Allocation`.

### Step 2 — Resolve Liveness (`resolveLiveness`, line 492)
1. **Build operation ID map** — post-order walk assigns IDs.
2. **For each buffer value**, compute liveness via `livenessForSmemChannel`:
   - Find the channel for this allocation
   - Collect all actual users (source + actual consumers of destinations)
   - Call `updateLiveOpsAcrossScopes` to normalize users to the same scope level and collect all ops between first and last user
3. **Map each buffer to its liveness interval** (min/max operation IDs).

### Step 3 — Assign Buffer IDs and Copy Counts (line 524)
Iterates over buffers in liveness order:

| Condition | `buffer.id` | `buffer.copy` |
|-----------|-------------|---------------|
| Users in innermost loop AND shape is 2D+ | **Shared** `bufferIdInnermost` (same ID for same element type) | `numBuffers` (multi-buffered) |
| Otherwise | **Unique** per buffer | `1` (single-buffered) |

Multi-buffered allocations with the same element type share a `buffer.id`, enabling the downstream buffer materialization to allocate a single multi-buffered region.

---

## TMEM Planner (`MemoryPlannerTmem::run`) — Detailed

### Step 1 — Collect TMEM Allocs and Compute Liveness (line ~848)

1. **Build operation ID map** (post-order via `buildOperationIdMap()`).
2. Walk all ops to find `TMEMAllocOp` operations.
3. For each alloc, compute:
   - **Liveness interval** via `getLiveIntervals()`:
     - Starts with the alloc's direct users
     - Follows `MemDescIndexOp` / `MemDescReinterpretOp` chains recursively
     - Computes min/max operation IDs across all live ops found by `livenessForTmemChannel()`
   - **Size** (`numRows × numCols`) via `getTmemAllocSizes()`
   - **Associated channel** via `findChannelForAlloc()` — casts to `TmemDataChannelPost *` if `TMEMPost`
4. Results stored in `allocToIntervals`, `allocToSize`, `allocToChannel` maps.

### Step 2 — Sort Allocs (line ~884)
Priority ordering (stable sort):
1. **`isOperandD`** (accumulator) first — these have read-modify-write semantics
2. **Larger size** first: rows then cols, tiebreak by total area
3. **Earlier liveness start** first (for same-size allocs)
4. Null-channel allocs go last

### Step 3 — Create Buffer Objects (line ~940)
For each sorted alloc, creates a `BufferT` entry:
```cpp
tBuf->rowSize = allocSize.numRows;
tBuf->colSize = allocSize.numCols;
tBuf->rowOffset = MAX;       // unallocated
tBuf->colOffset = MAX;       // unallocated
tBuf->isOwnerOfSpace = false;
tBuf->reuseOwner = nullptr;
```

### Step 4 — Per-Loop Allocation (line ~996)
Iterates innermost loops in program order. For each loop:
1. **Collect candidates**: allocs whose liveness intersects the loop interval (not yet handled)
2. **Call `allocateTMemAllocs()`** — the core heuristic

---

### TMEM Allocation Heuristic (`allocateTMemAllocs`) — Detailed

**Location**: Lines 1048–1511

This is a closure-heavy function with several key lambdas:

#### Key Lambdas

| Lambda | Purpose |
|--------|---------|
| `isDataDependent(src, dst)` | Forward SSA slice check: does `dst` transitively use `src`? Also follows memory deps (local_store, tmem_store). |
| `hasTransitiveDependency(consumer, producer, ...)` | Cross-partition dependency: finds a TMEM channel where chSrc is data-dependent on consumer and producer is data-dependent on chDst. |
| `alongDependencyChain(src, dst, cond)` | Combines: same partition? OR direct data dep? OR transitive dep via channel? |
| `sameLoop(alloc)` | Does alloc's liveness intersect the current loop's interval? |
| `samePartition(alloc, cand, cond)` | Partition matching with configurable strictness (0=any, 1=src/dst match, 2=combined tasks match). |
| `findReuseSpace(cand, owner, depChain)` | Find column offset within `owner`'s row region. Checks existing reuses for overlap. |
| `checkOtherReuses(cand, owner, colOffset)` | Verify no other buffer reusing `owner` has both liveness AND column overlap with `cand`. |
| `findReuseChannel(cand, partCond, depCond)` | Main reuse search: iterates all space-owning buffers, checking reuse feasibility. |
| `allInterfere(cand)` | Does `cand` overlap in liveness with ALL currently allocated buffers? |
| `allocateNewSpace(cand, allocate)` | Assign a new row region; checks 512-row hardware limit. |

#### Main Allocation Loop (line ~1456)

```
for each alloc (in priority order):
  candBuf = getBuffer(alloc)
  
  if allInterfere(candBuf):
    // No existing buffer is non-overlapping → must allocate new space
    allocateNewSpace(candBuf, allocate=true)
    if out of space: ERROR
    
  else:
    // Try reuse with strict partition matching first
    reuseBuf = findReuseChannel(cand, partitionCondition=2, depChainCondition=1)
    if !reuseBuf:
      // Relax to less strict matching
      reuseBuf = findReuseChannel(cand, partitionCondition=1, depChainCondition=1)
    
    if reuseBuf:
      set buffer.id = reuseBuf's buffer.id
      set buffer.offset = colOffset
    else:
      allocateNewSpace(candBuf)  // fallback
  
  set buffer.copy = 1  // always single-copy for TMEM
```

#### Reuse Conditions Detail (`findReuseChannel`)

A candidate `cand` can reuse space-owner `alloc` when **ALL** of:
1. `alloc.isOwnerOfSpace == true`
2. **No liveness overlap**: `!bufferRange[alloc].intersects(bufferRange[cand])`
3. **Column size fits**: `alloc.colSize >= cand.colSize`
4. **Either**:
   - **Same loop + dependency chain**: `sameLoop(alloc) && alongDependencyChain(alloc, cand)` — respecting temporal order (earlier-starting buffer is "src")
   - **Different loop + same partitions**: `!sameLoop(alloc) && samePartition(alloc, cand, partitionCondition)`
5. **Column offset available**: `findReuseSpace()` finds a valid column offset
6. **No reuse conflicts**: `checkOtherReuses()` confirms no column/liveness overlap with existing reuses

---

## Buffer Decision Serialization/Deserialization

### Data Structures (line ~1519)

```cpp
struct BufferDecision {
  unsigned channelId;     // channel's uniqID
  unsigned bufferId;      // buffer.id attribute
  unsigned bufferCopy;    // buffer.copy attribute
  unsigned bufferOffset;  // buffer.offset attribute
};

struct BufferDecisionList {
  SmallVector<BufferDecision> decisions;
};
```

### Serialization Flow

```
serializeBufferDecisions(channels)
  → sort channels by program order (alloc position)
  → for each channel: extractBufferDecision() reads buffer.id/copy/offset attrs
  → return BufferDecisionList

serializeBufferDecisionsToString(list)
  → JSON: { "version": 1, "decisions": [...] }

writeDecisionsToFile(channels, filePath)
  → serialize → write JSON to file
```

### Deserialization Flow

```
readDecisionsFromFile(channels, filePath)
  → read JSON → deserializeBufferDecisionsFromString()
  → deserializeBufferDecisions(channels, decisions)
    → sort channels by program order
    → check: channel count matches?
    → check: each channel's uniqID matches?
    → applyBufferDecision(): set buffer.id/copy/offset attrs
```

### Current Validation in Deserialization

The current `deserializeBufferDecisions()` only checks:
1. **Channel count match**: `sortedChannels.size() == decisions.decisions.size()`
2. **Channel ID match**: `ch->uniqID == decision.channelId` (per-index)

It does **NOT** verify:
- Whether the assigned `buffer.id` groupings are internally consistent (e.g., buffers sharing an ID don't overlap in liveness)
- Whether `buffer.offset` values cause column conflicts within shared row regions
- Whether the 512-row hardware limit is respected
- Whether reuse relationships are valid (dependency chain, partition matching)

---

## Bug: Deserialization Fails for Zero-Valued Fields

**Location**: `deserializeBufferDecisionsFromString()`, line ~1699

```cpp
auto channelId = obj->getInteger("channelId");
auto bufferId = obj->getInteger("bufferId");
auto bufferCopy = obj->getInteger("bufferCopy");
auto bufferOffset = obj->getInteger("bufferOffset");

if (!channelId || !bufferId || !bufferCopy || !bufferOffset) {
  LDBG("Missing required field in decision");
  return std::nullopt;
}
```

`getInteger()` returns `std::optional<int64_t>`. When the value is `0`, the optional *does* contain a value (`std::optional<int64_t>(0)`), but `!bufferId` tests `!*bufferId` (the contained `int64_t`), which is `!0 == true`. This means **any decision with `bufferId=0`, `channelId=0`, or `bufferOffset=0` fails parsing**.

In the example `bwd_decisions_v2.json`, channelId 0 has `bufferId: 0` and every entry has `bufferOffset: 0`. The current code would fail to parse this file.

**Fix**: Use `!channelId.has_value()` instead of `!channelId`, or check each separately:

```cpp
if (!channelId.has_value() || !bufferId.has_value() ||
    !bufferCopy.has_value() || !bufferOffset.has_value()) {
```

---

## Proposal: Verifying Deserialized Decisions

### Refactoring Opportunity

The core logic in `allocateTMemAllocs` can be decomposed into two separable concerns:

1. **Decision-making**: Which buffers share space? What offsets to use? (the current heuristic)
2. **Feasibility verification**: Given a set of decisions, are they valid?

Currently these are intertwined. The feasibility checks (`checkOtherReuses`, `findReuseSpace`, `allInterfere`, liveness overlap) are embedded in the allocation loop. They could be extracted into a standalone **verification pass** that validates any `BufferDecisionList` — whether computed by the heuristic or loaded from a file.

### Proposed Verification Checks

A `verifyBufferDecisions()` function would check:

| # | Check | Description | Existing Code to Reuse |
|---|-------|-------------|----------------------|
| 1 | **Row capacity** | Total row usage across all distinct `buffer.id` groups must not exceed 512 | `allocateNewSpace()` check at line 1436 |
| 2 | **Row size consistency** | Buffers sharing a `buffer.id` must have the **same** `rowSize` (they share the same row region) | Implicit in the owner-reuse model: `reuseOwner->rowSize` is the shared region |
| 3 | **Column capacity** | For any reusing buffer, `buffer.offset + colSize <= ownerColSize` where `ownerColSize` is the maximum colSize in the group | `findReuseSpace()` check at line 1291 |
| 4 | **Liveness × column conflict** | Within the same `buffer.id`, two buffers whose liveness intervals overlap must have non-overlapping column ranges `[offset, offset+colSize)` | `checkOtherReuses()` logic at line 1253 |
| 5 | **buffer.copy consistency** | TMEM buffers should always have `buffer.copy = 1` (multi-buffering is SMEM-only) | Hardcoded at line 1508 |

### What We Do NOT Verify (intentionally)

The heuristic also checks dependency chains (`alongDependencyChain`) and partition matching (`samePartition`) during reuse search. These are **heuristic quality signals**, not hard correctness constraints. A valid TMEM allocation only needs:
- Non-overlapping liveness/columns (no two live buffers occupy the same TMEM cells)
- Total rows ≤ 512

The dependency/partition checks guide the heuristic toward safe reuse patterns, but a manually crafted decision that violates them is still *technically feasible* if the liveness/column constraints are satisfied.

### Implementation Plan

```
static LogicalResult verifyBufferDecisions(
    SmallVector<Channel *> &channels,
    SmallVector<triton::nvidia_gpu::TMEMAllocOp> &allocs,
    DenseMap<Operation *, Interval<size_t>> &allocToIntervals,
    DenseMap<Operation *, ttng::TMemAllocation> &allocToSize) {

  // Step 1: Build channel→alloc mapping (channelId → allocOp)
  //         Read buffer.id, buffer.offset, buffer.copy from alloc attrs

  // Step 2: Group allocs by buffer.id
  //   For each group:
  //     a. Find the "owner" = the alloc with offset 0 (or largest colSize)
  //     b. Verify all allocs in the group have the same rowSize
  //     c. For each alloc: verify offset + colSize <= owner.colSize
  //     d. For each pair in the group with overlapping liveness:
  //        verify column ranges [offset, offset+colSize) do not overlap

  // Step 3: Across all groups:
  //   Compute total row usage = sum of each group's rowSize
  //   Verify total <= 512

  // Step 4: Verify all TMEM buffer.copy == 1

  return success(); // or failure() with diagnostic
}
```

### Integration Points

The verification function should be called at three sites:

1. **After deserialization** (in `doMemoryPlanner`, line ~1855): validate loaded decisions before returning `success()`. If verification fails, fall back to the heuristic planner.

   ```cpp
   if (succeeded(readDecisionsFromFile(channels, effectiveReadFile))) {
     if (succeeded(verifyBufferDecisions(channels, allocs, ...))) {
       return success();
     }
     LDBG("Decisions failed verification, falling back to planner");
   }
   ```

2. **After the heuristic** (after `MemoryPlannerTmem::run()`, line ~1905): as a self-check / assertion to catch bugs in the heuristic itself.

3. **In tests** — to verify hand-crafted decision files produce valid allocations.

### What Needs Refactoring

To make this work, we need to slightly restructure `doMemoryPlanner`:

- **Collect TMEM allocs and compute liveness** *before* the decision deserialization branch, so that the verification function has access to `allocToIntervals` and `allocToSize` regardless of whether we use the heuristic or file-based decisions.
- Currently, liveness computation happens *inside* `MemoryPlannerTmem::run()`. We'd need to either:
  - (a) Extract liveness computation into a standalone helper callable from `doMemoryPlanner`, or
  - (b) Move deserialization *inside* `MemoryPlannerTmem::run()` so it has access to the liveness data, or
  - (c) Re-compute liveness in the verification function (simplest but redundant).

Option (a) is the cleanest: extract `collectTMemAllocsAndLiveness()` from `MemoryPlannerTmem::run()` into a public method, then call it from `doMemoryPlanner` before the deserialization branch.

---

## Key Data Structures

| Structure | Purpose |
|-----------|---------|
| `operationId: DenseMap<Operation*, size_t>` | Post-order IDs for ordering |
| `bufferRange: MapVector<BufferT*, Interval<size_t>>` | Liveness interval per buffer |
| `Channel` / `ChannelPost` / `TmemDataChannelPost` | Producer→consumer communication info |
| `BufferT` | Buffer metadata: size, offset, reuse owner, `isOwnerOfSpace` |
| `BufferDecision` / `BufferDecisionList` | Serializable buffer assignments |

## Output Attributes (set on allocation ops)

| Attribute | Meaning |
|-----------|---------|
| `buffer.id` | Logical buffer group ID (shared by reusable buffers) |
| `buffer.copy` | Number of copies for multi-buffering (SMEM) or `1` (TMEM) |
| `buffer.offset` | Column offset within reused buffer space (TMEM only) |

---

## Liveness Computation — Current Approach

The current liveness model is based on **operation IDs** assigned via a post-order walk of the entire IR tree. This has implications for SWP (Software Pipelining) awareness:

1. **Operation IDs are static** — they reflect the IR structure at the time the memory planner runs, *before* software pipelining transforms the loop. There is no notion of pipeline stages or clusters.

2. **Liveness is channel-based** — for each allocation, liveness spans from the first to the last user (producer/consumer), normalized to the same scope level. This is correct for non-pipelined code but may be overly conservative or incorrect for pipelined loops where a buffer's producer and consumer are in different pipeline stages.

3. **Multi-buffering is heuristic** — SMEM buffers in innermost loops with 2D+ shapes get `buffer.copy = numBuffers`. This is a blanket policy not informed by the actual pipeline schedule (number of stages, which stage produces/consumes each buffer).

4. **TMEM reuse uses data dependency** — `alongDependencyChain` and `hasTransitiveDependency` check SSA reachability and cross-partition channels. These checks are pipeline-unaware; they don't consider whether two allocs are live at the same pipeline stage.

### Opportunities for SWP Awareness

- **Stage-aware liveness**: If the SWP schedule is available, liveness intervals could be computed per-stage, allowing more precise overlap detection and more aggressive reuse.
- **Stage-aware multi-buffering**: The number of copies (`buffer.copy`) could be derived from the actual number of pipeline stages between producer and consumer, rather than a fixed `numBuffers` parameter.
- **Cluster-aware ordering**: Operations within the same stage but different clusters have a defined order; this could refine the liveness intervals within a pipelined loop body.
