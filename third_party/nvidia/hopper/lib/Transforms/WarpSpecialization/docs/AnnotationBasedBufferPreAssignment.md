# Annotation-Based Buffer Pre-Assignment in WSMemoryPlanner

## Overview

Users can annotate `tl.dot` operations with per-operand channel specifications via the `attrs` dict. These annotations flow through the compiler as a `tt.autows` JSON string attribute on `ttng.tc_gen5_mma` ops and can be consumed by WSMemoryPlanner to **pre-assign** `buffer.copy`, `buffer.id`, and `buffer.offset` — bypassing heuristic allocation for annotated buffers while leaving un-annotated buffers unchanged.

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| SMEM algo 1 (WSBuffer-based) | ✅ **Pre-assignment** | Annotated buffers pinned in Phase 1; skip Phases 2–4 |
| SMEM algo 0 (original MemoryPlanner) | ❌ **Not implemented** | No annotation support |
| TMEM algo 1 (greedy) | ⚠️ **Post-processing override** | Heuristic runs first; buffer.id/buffer.copy overridden after |
| TMEM algo 2 (backtracking) | ⚠️ **Post-processing override** | Same as TMEM algo 1 |

### Known Gaps

1. **SMEM algo 0**: The original `MemoryPlanner` class doesn't receive annotations at all.
   All annotated kernels should use `tt.smem_alloc_algo = 1`.

2. **TMEM post-processing vs. pre-assignment**: The current TMEM implementation is a
   post-processing override — the heuristic allocator runs first (computing its own
   reuse groups and offsets), then we override `buffer.id`/`buffer.copy` for annotated
   allocs. This can lead to inconsistencies:
   - `buffer.offset` was computed for the heuristic's reuse groups, not the annotated ones
   - If two annotated allocs share a `bufferId` (reuse group), the heuristic doesn't
     know about this and may compute conflicting offsets
   - The proper fix is to **pre-assign** annotated TMEM allocs before the heuristic runs,
     partitioning them out of `allocsForThisLoop` (see Step 4 design below)

3. **Alloc-to-annotation mapping**: `buildAllocToAnnotationMap` traces from channels to
   MMA consumers to look up annotations. This tracing is incomplete for some cases
   (e.g., SMEM allocs feeding MMA through multiple levels of indirection). The
   `getSmemOperandIndex` helper only finds MMA operand A vs B when there's a direct or
   single-hop path.

### User-Facing API

```python
tl.dot(k, qT, attrs={
    "stage": "0", "cluster": "0",
    "channels": ["opndA,smem,2,0", "opndB,smem,2,1", "opndD,tmem,1,2"]
})
```

### Channel Format

Each channel string: `"operand,memoryType,numCopies,bufferId"`

| Field | Values | Description |
|-------|--------|-------------|
| `operand` | `opndA`, `opndB`, `opndD` | Which MMA operand this channel feeds |
| `memoryType` | `smem`, `tmem` | Memory backing for the channel |
| `numCopies` | integer | Multi-buffering depth |
| `bufferId` | integer | Buffer identity; shared IDs form reuse groups |

### MLIR Representation

```mlir
%qkT = ttng.tc_gen5_mma %k, %qT, %acc ...
  {tt.autows = "{\"stage\": \"0\", \"cluster\": \"0\",
                 \"channels\": [\"opndA,smem,2,0\", \"opndB,smem,2,1\", \"opndD,tmem,1,2\"]}"}
```

The `tt.autows` attribute survives through `AccelerateMatmul` (which propagates discardable attrs from `tt.dot` to `ttng.tc_gen5_mma`) and persists when WSMemoryPlanner runs.

---

## Current Memory Planner Architecture

### SMEM Allocation (`allocateSmemBuffers()`)

5-phase algorithm:

| Phase | Action | Annotated Buffer Behavior |
|-------|--------|---------------------------|
| 1. Initialize | Create `WSBuffer` per `local_alloc`, `bufferId = nextId++`, `numCopies = 1` | **Override**: set `bufferId` and `numCopies` from annotation, mark `isPinned = true` |
| 2. Cross-stage minimum | `numCopies = 2` for cross-stage buffers | **Skip** pinned buffers |
| 3. Classify priorities | P0 (TMA+innermost), P1, P2 | **Skip** pinned buffers |
| 4. Iterative copy increase | Increment copies within SMEM budget; optional circular reuse pairing | **Exclude** pinned buffers from candidates |
| 5. Emit attributes | Write `buffer.id`, `buffer.copy` on each `local_alloc` | No change — emits from WSBuffer fields |

### TMEM Allocation (`MemoryPlannerTmem::run()`)

- Collects TMEM allocs, builds `allocToChannel` map
- Sorts: operand D first, larger first, earlier liveness first
- Two algorithms (`tt.tmem_alloc_algo`): greedy (1) or backtracking (2)
- Outputs: `buffer.id`, `buffer.copy` (always 1), `buffer.offset` (column offset for reuse)

### Channel → MMA Operand Mapping

| Operand | Channel Type | Key Field | Memory |
|---------|-------------|-----------|--------|
| A | `ChannelPost` (SMEM) or `TmemDataChannelPost` (TMEM) | `operandIdx` / trace through users | smem or tmem |
| B | `ChannelPost` (SMEM) or `TmemDataChannelPost` (TMEM) | `operandIdx` / trace through users | smem or tmem |
| D | `TmemDataChannelPost` | `isOperandD = true` | tmem (always) |

---

## Implementation Steps

### Step 1: Channel Annotation Parsing Utility

**File**: `WSMemoryPlanner.cpp` — add near line 630 (after `WSBuffer` struct)

Add a `ChannelAnnotation` struct and parser function:

```cpp
struct ChannelAnnotation {
  std::string operand;   // "opndA", "opndB", "opndD"
  std::string memType;   // "smem", "tmem"
  unsigned numCopies;
  unsigned bufferId;
};

/// Parse tt.autows channels from all MMA ops.
/// Returns a map keyed by (mmaOp, operandName) → ChannelAnnotation.
static DenseMap<std::pair<Operation*, StringRef>, ChannelAnnotation>
parseChannelAnnotations(triton::FuncOp funcOp) {
  DenseMap<std::pair<Operation*, StringRef>, ChannelAnnotation> result;

  funcOp->walk([&](Operation *op) {
    if (!isa<ttng::MMAv5OpInterface>(op))
      return;
    auto attr = op->getAttrOfType<StringAttr>("tt.autows");
    if (!attr)
      return;
    auto parsed = llvm::json::parse(attr.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return;
    }
    auto *obj = parsed->getAsObject();
    if (!obj)
      return;
    auto *channelsArr = obj->getArray("channels");
    if (!channelsArr)
      return;
    for (auto &elem : *channelsArr) {
      auto str = elem.getAsString();
      if (!str) continue;
      // Parse "opndA,smem,2,0"
      SmallVector<StringRef, 4> parts;
      StringRef(*str).split(parts, ',');
      if (parts.size() != 4) continue;
      ChannelAnnotation ann;
      ann.operand = parts[0].str();
      ann.memType = parts[1].str();
      ann.numCopies = std::stoi(parts[2].str());
      ann.bufferId = std::stoi(parts[3].str());
      result[{op, StringRef(ann.operand)}] = ann;
    }
  });
  return result;
}
```

### Step 2: Build Alloc-to-Annotation Mapping

**File**: `WSMemoryPlanner.cpp` — add helper function

For each channel in the collected channels list, trace from `allocOp` → consumer MMA → look up annotation:

```cpp
/// Map each alloc op → its ChannelAnnotation (if the consumer MMA has one).
static DenseMap<Operation*, ChannelAnnotation>
buildAllocToAnnotationMap(
    SmallVector<Channel*> &channels,
    const DenseMap<std::pair<Operation*, StringRef>, ChannelAnnotation> &annotations) {
  DenseMap<Operation*, ChannelAnnotation> result;

  for (auto *ch : channels) {
    Operation *allocOp = ch->getAllocOp();
    if (!allocOp) continue;

    Operation *mmaOp = ch->getDstOp();
    if (!mmaOp || !isa<ttng::MMAv5OpInterface>(mmaOp))
      continue;

    StringRef operandName;
    if (ch->channelKind == DataChannelKind::TMEMPost) {
      auto *tmemCh = static_cast<ttng::TmemDataChannelPost*>(ch);
      operandName = tmemCh->isOperandD ? "opndD" : "opndA"; // TODO: distinguish A vs B
    } else if (ch->channelKind == DataChannelKind::SMEMPost) {
      operandName = "opndA"; // TODO: distinguish A vs B by tracing operand index
    } else {
      continue;
    }

    auto it = annotations.find({mmaOp, operandName});
    if (it != annotations.end())
      result[allocOp] = it->second;
  }
  return result;
}
```

**Note**: Distinguishing `opndA` vs `opndB` requires tracing from the `allocOp` through its users to determine which MMA input it feeds. For SMEM, follow `local_alloc` → `memdesc_trans` → MMA operand index. For TMEM non-D, check the channel's operand index.

### Step 3: SMEM Pre-Assignment in `allocateSmemBuffers()`

**File**: `WSMemoryPlanner.cpp` — modify lines 788–1022

#### 3a. Add `isPinned` field to `WSBuffer`

```cpp
struct WSBuffer {
    Operation *allocOp;
    unsigned sizeBytes;
    Interval<size_t> liveness;
    bool isInnermost, isTMA, isCrossStage;
    unsigned bufferId;
    unsigned numCopies;
    WSBufferPriority priority;
    bool isPinned = false;  // NEW: set by annotation, skips heuristic phases
};
```

#### 3b. Phase 1: Apply annotations

After creating each `WSBuffer`, check `allocToAnnotation`:

```cpp
// In Phase 1, after populating WSBuffer fields:
if (auto it = allocToAnnotation.find(alloc.getOperation());
    it != allocToAnnotation.end() && it->second.memType == "smem") {
  buf.bufferId = it->second.bufferId;
  buf.numCopies = it->second.numCopies;
  buf.isPinned = true;
  LDBG("Phase 1: WSBuffer pinned by annotation: bufferId="
       << buf.bufferId << " numCopies=" << buf.numCopies);
}
```

#### 3c. Adjust `nextBufferId`

After Phase 1, ensure heuristic IDs don't collide:

```cpp
unsigned maxAnnotatedId = 0;
for (auto &buf : wsBuffers)
  if (buf.isPinned)
    maxAnnotatedId = std::max(maxAnnotatedId, buf.bufferId + 1);
nextBufferId = std::max(nextBufferId, maxAnnotatedId);
```

#### 3d. Phases 2–4: Skip pinned buffers

```cpp
// Phase 2 (cross-stage enforcement):
for (auto &buf : wsBuffers) {
  if (buf.isPinned) continue;  // NEW
  if (buf.isCrossStage && numBuffers >= 2) { ... }
}

// Phase 3 (priority classification):
for (auto &buf : wsBuffers) {
  if (buf.isPinned) continue;  // NEW
  // ... classify priority ...
}

// Phase 4 (iterative copy increase):
// When building candidateIndices:
for (unsigned i = 0; i < wsBuffers.size(); ++i) {
  if (wsBuffers[i].isPinned) continue;  // NEW: exclude pinned
  if (wsBuffers[i].priority == currentPriority)
    candidateIndices.push_back(i);
}
```

### Step 4: TMEM Pre-Assignment

**File**: `WSMemoryPlanner.cpp` — modify `MemoryPlannerTmem::run()`

Add a pre-assignment step before the heuristic allocation loop:

#### 4a. Partition annotated vs. un-annotated allocs

```cpp
// After building allocToChannel, get annotations:
auto annotations = parseChannelAnnotations(funcOp);
auto allocToAnnotation = buildAllocToAnnotationMap(*channels, annotations);

// Separate annotated and un-annotated allocs
SmallVector<ttng::TMEMAllocOp> annotatedAllocs, heuristicAllocs;
for (auto alloc : allocsForThisLoop) {
  if (allocToAnnotation.count(alloc.getOperation()))
    annotatedAllocs.push_back(alloc);
  else
    heuristicAllocs.push_back(alloc);
}
```

#### 4b. Group annotated allocs by `bufferId`

```cpp
// Group by bufferId: first alloc per ID is owner, rest are reusers
DenseMap<unsigned, SmallVector<ttng::TMEMAllocOp>> annotatedGroups;
for (auto alloc : annotatedAllocs) {
  auto &ann = allocToAnnotation[alloc.getOperation()];
  annotatedGroups[ann.bufferId].push_back(alloc);
}
```

#### 4c. Validate reuse and assign attributes

For each group:

```cpp
for (auto &[bid, group] : annotatedGroups) {
  // First alloc is owner
  auto ownerAlloc = group[0];
  ownerAlloc->setAttr("buffer.id", IntegerAttr::get(i32, bid));
  ownerAlloc->setAttr("buffer.copy", IntegerAttr::get(i32, 1));

  // Subsequent allocs are reusers
  size_t colOffset = 0;
  for (size_t i = 1; i < group.size(); ++i) {
    auto reuserAlloc = group[i];

    // Validate liveness non-overlap
    auto &ownerInterval = allocToIntervals[ownerAlloc.getOperation()];
    auto &reuserInterval = allocToIntervals[reuserAlloc.getOperation()];
    if (ownerInterval.intersects(reuserInterval)) {
      LDBG("WARNING: annotated reuse group bufferId=" << bid
           << " has overlapping liveness — falling back to heuristic");
      heuristicAllocs.push_back(reuserAlloc);
      continue;
    }

    // Validate size compatibility
    auto ownerSize = allocToSize[ownerAlloc.getOperation()];
    auto reuserSize = allocToSize[reuserAlloc.getOperation()];
    if (reuserSize.numCols > ownerSize.numCols) {
      LDBG("WARNING: reuser columns exceed owner — falling back to heuristic");
      heuristicAllocs.push_back(reuserAlloc);
      continue;
    }

    // Assign attributes
    reuserAlloc->setAttr("buffer.id", IntegerAttr::get(i32, bid));
    reuserAlloc->setAttr("buffer.copy", IntegerAttr::get(i32, 1));
    reuserAlloc->setAttr("buffer.offset", IntegerAttr::get(i32, colOffset));

    colOffset += reuserSize.numCols;
  }
}
```

#### 4d. Coordinate bufferId for heuristic allocation

```cpp
unsigned maxAnnotatedBid = 0;
for (auto &[bid, _] : annotatedGroups)
  maxAnnotatedBid = std::max(maxAnnotatedBid, bid + 1);
bufferId = std::max(bufferId, maxAnnotatedBid);

// Run heuristic on remaining un-annotated allocs only
if (!heuristicAllocs.empty()) {
  result = allocateTMemAllocs2(heuristicAllocs, buffers, allocToChannel,
                               operationId, ctrlOp, bufferId);
}
```

### Step 5: Validation and Diagnostics

Add throughout the implementation:

- **memType mismatch**: Warn if SMEM channel annotated with `"tmem"` or vice versa
- **Cross-stage numCopies**: Warn if annotated SMEM `numCopies == 1` for a cross-stage buffer
- **TMEM reuse validity**: Warn on liveness overlap or size incompatibility
- **LDBG logging** for all annotation decisions, matching existing style

---

## Attribute Flow Summary

```
Python: tl.dot(..., attrs={"channels": ["opndA,smem,2,0", ...]})
  ↓
core.py: _unwrap_if_constexpr(attrs), pass to _semantic.dot()
  ↓
semantic.py: json.dumps(attrs) → set_attr("tt.autows", json_string) on tt.dot
  ↓
AccelerateMatmul: propagate discardable attrs from tt.dot → ttng.tc_gen5_mma
  ↓
WSMemoryPlanner: parse tt.autows → ChannelAnnotation → allocToAnnotation map
  ↓
SMEM: WSBuffer.isPinned → skip phases 2-4 → emit buffer.id/buffer.copy
TMEM: pre-assign buffer.id/buffer.copy/buffer.offset → validate reuse → exclude from heuristic
```

## Key Attributes

| Attribute | Set By | Read By | Pre-assigned? |
|-----------|--------|---------|---------------|
| `buffer.id` | WSMemoryPlanner (SMEM Phase 5 / TMEM alloc) | `doCodePartitionPost` (reuse group formation) | ✅ From annotation |
| `buffer.copy` | WSMemoryPlanner (SMEM Phase 5 / TMEM alloc) | Buffer allocation, `needAccumCntForReuse` | ✅ From annotation |
| `buffer.offset` | WSMemoryPlanner (TMEM only) | `replaceBufferReuse` (TMEM column slice) | ✅ Computed from reuse group |

## Files Modified

| File | Changes |
|------|---------|
| `WSMemoryPlanner.cpp` | `ChannelAnnotation` struct, `parseChannelAnnotations()`, `buildAllocToAnnotationMap()`, WSBuffer `isPinned` field, SMEM phases 1–4 pinning, TMEM pre-assignment with reuse validation |

## Testing

1. **Regression**: Run existing WS memory planner lit tests to verify no change for un-annotated kernels
2. **New lit test**: MLIR test with `tt.autows` channel annotations on `tc_gen5_mma` ops, verify `buffer.id`/`buffer.copy`/`buffer.offset` match annotations
3. **Integration**: Run bwd attention tutorial with channel annotations, dump MLIR, verify buffer attributes
4. **Edge cases**: Partially annotated kernels, invalid reuse annotations (overlapping liveness), memType mismatches

---

## Future Work: Proper TMEM Pre-Assignment

The current TMEM implementation (Step 4) is a **post-processing override** — the heuristic
runs first, then we override `buffer.id`/`buffer.copy`. This has limitations:

1. `buffer.offset` remains from the heuristic's reuse computation, which may be wrong
   for the annotated reuse groups
2. Annotated reuse groups (same `bufferId`) are not validated for liveness non-overlap
3. The heuristic may allocate new row space for allocs that should reuse annotated space

### Proper Pre-Assignment Design (from Step 4 in this document)

The correct approach is to:

1. **Partition** `allocsForThisLoop` into annotated and un-annotated before calling
   `allocateTMemAllocs`/`allocateTMemAllocs2`
2. **Pre-assign** annotated allocs: group by `bufferId`, validate reuse (liveness
   non-overlap, column size compatibility), compute `buffer.offset`
3. **Update `BufferT` fields** for pre-assigned allocs (`rowOffset`, `colOffset`,
   `isOwnerOfSpace`, `reuseOwner`) so the heuristic sees them as already placed
4. **Exclude** annotated allocs from `allocsForThisLoop` passed to the heuristic
5. **Coordinate** `bufferId` so heuristic IDs don't collide with annotated IDs

### Also Needed: SMEM Algo 0 Support

The original `MemoryPlanner` class (algo 0) does not receive annotations. To support it:
- Pass `allocToAnnotation` map to `MemoryPlanner::run()`
- In the original allocation loop, check for annotations and pin buffer.id/buffer.copy
- Or: require `tt.smem_alloc_algo = 1` for annotated kernels (simpler)
