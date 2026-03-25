# Reuse Groups

Reuse groups are the autoWS memory planner's mechanism for letting multiple
channels with non-overlapping lifetimes share a single physical buffer
allocation. When two channels never hold live data at the same time, the planner
assigns them the same `buffer.id` so that downstream code partitioning replaces
all but one allocation with views into a single representative buffer. This
reduces SMEM and TMEM pressure without changing program semantics.

## Requirements for Reuse

Two channels can share a buffer when:

1. They have the **same `buffer.id`** assigned by the memory planner.
2. They reference **different `allocOp`s**. If all channels with the same
   `buffer.id` point to the same `allocOp`, they are lifecycle phases of one
   buffer (e.g., multi-buffered pipeline stages), not reuse candidates.

Beyond these common requirements, SMEM and TMEM have additional constraints:

### SMEM Circular Reuse

Handled in `WSMemoryPlanner.cpp` Phase 4 (`allocateSmemBuffers`). Requires:

- Exactly **2 innermost-loop candidates** in the same priority group
- **Compatible element types** (both allocs must have the same `elemType`)
- Multi-dimensional allocs (`numD >= 2`) whose users live in the innermost loop

When these conditions hold, buffer B is given buffer A's `bufferId` and both
receive the same `numCopies`. The number of copies is then maximized by the
SMEM memory planner's incremental allocation algorithm described in
[SMEM Allocation Design](SmemAllocationDesign.md).

### TMEM Packing

Handled in `WSMemoryPlanner.cpp` (`applyAllocationState`). Requires:

- **Non-overlapping liveness intervals** in the column dimension, checked by
  `hasPotentialReuse` during allocation planning
- A valid column offset found by the backtracking allocator `tryAllocate`

Owner buffers get a fresh `buffer.id`; non-owner (reusing) buffers receive the
same `buffer.id` as their owner plus a `buffer.offset` encoding the column
offset within the owner's TMEM row.

## Data Structures

Defined in `CodePartitionUtility.h`:

```cpp
struct ReuseGroup {
  std::vector<unsigned> channelIDs;
  std::vector<Channel *> channels;
};

struct ReuseConfig {
  std::vector<ReuseGroup> groups;
  unsigned getGroupSize() { return groups.size(); }
  ReuseGroup *getGroup(unsigned idx);
};
```

`ReuseGroup` holds a set of channels that all share the same physical buffer.
The first channel (`channels[0]`) is always the **representative** — the owner
of the physical memory. `ReuseConfig` is the collection of all reuse groups for
a given kernel.

## Formation Algorithm

Reuse groups are formed in `doCodePartitionPost` (`WSCodePartition.cpp`):

1. **Group by `buffer.id`**: Iterate over all ordered channels. For each
   channel, look up the `buffer.id` attribute on its `allocOp` and insert the
   channel into a `bufferIdToChannels` map.

2. **Filter same-allocOp sets**: For each `buffer.id` with more than one
   channel, check whether all channels reference the same `allocOp`. If so,
   they are lifecycle phases of one buffer — skip them.

3. **Order channels**: Stable-partition the channels so that the one
   **without** a `buffer.offset` attribute comes first. This channel becomes
   the representative (`channels[0]`), the owner of the physical allocation.

4. **Create `ReuseGroup`**: Push the ordered channel list into a new
   `ReuseGroup` and append it to `config.groups`.

## What Reuse Groups Affect

### 1. Accumulation Counters

When channels in a reuse group share a multi-buffered circular buffer, a shared
**accumulation counter** (`accumCnt`) tracks which buffer slot to use. The
counter is carried as a loop argument and incremented as channels are consumed.

Key functions:
- `needAccumCntForReuse` — returns true when a loop/if region contains at
  least one src or dst op of the reuse group and the group is multi-buffered
- `getAccumForReuseGroup` — computes the `accumCnt` SSA value at a given
  operation by walking back through the channel list to find the nearest
  preceding region op, then arithmetically adding the remaining offset
- `getBufferIdxAndPhase` — for the first channel in the ordered list, uses
  `accumCnt` directly; each subsequent channel at position N adds N to stagger
  its slot within the shared circular buffer
- `getReuseAccumArgIdx` — returns the position of a group's `accumCnt`
  argument within the region's full argument list

### 2. Token/Barrier Sharing

In `createTokenPost`, the representative channel (first in the group) creates
barriers; non-representative channels reuse them. `channelInReuseGroup` looks
up which group a channel belongs to (returning -1 if none). The `reuseBarrier`
flag skips groups whose representative has `numBuffers <= 1` (single-buffered
channels share no circular barrier).

### 3. Buffer Replacement

`replaceBufferReuse` rewrites all IR uses of non-representative alloc ops to
point at the representative's alloc:

- **SMEM channels**: When the alloc types match, uses direct
  `replaceUsesOfWith` to swap the alloc result, then erases the old alloc.
  Type mismatches are skipped (SMEM cannot be reinterpreted like TMEM).

- **TMEM channels**: Inserts a `sliceAndReinterpretMDTMEM` op at the
  `buffer.offset` column within the representative's TMEM allocation. If the
  primary representative's type cannot accommodate the slice, other group
  representatives are tried before emitting an error.

### 4. `allocation.shareGroup` Attribute

Buffers in a reuse group are tagged with an `allocation.shareGroup` attribute
for consumption by downstream passes.

## Key Attributes

| Attribute | Description | Set by | Read by |
|-----------|-------------|--------|---------|
| `buffer.id` | Groups channels that share physical memory | `WSMemoryPlanner` (SMEM + TMEM) | `doCodePartitionPost` (group formation) |
| `buffer.copy` | Number of pipeline copies (multi-buffering depth) | `WSMemoryPlanner` | Buffer allocation, `needAccumCntForReuse` |
| `buffer.offset` | Column offset within the owner's TMEM allocation | `WSMemoryPlanner` (`applyAllocationState`) | `replaceBufferReuse` (TMEM slice offset) |
| `allocation.shareGroup` | Tags buffers for downstream passes | `doCodePartitionPost` | Downstream passes |

## Key Functions Reference

| Function | File | Purpose |
|----------|------|---------|
| `ReuseGroup`, `ReuseConfig` | `CodePartitionUtility.h` | Data structures |
| `channelInReuseGroup` | `CodePartitionUtility.cpp` | Look up reuse group index for a channel |
| `needAccumCntForReuse` | `CodePartitionUtility.cpp` | Check if a region needs an `accumCnt` argument |
| `getReuseChannels` | `CodePartitionUtility.cpp` | Build ordered list of dst ops in a region |
| `getReuseAccumArgIdx` | `CodePartitionUtility.cpp` | Position of group's `accumCnt` in argument list |
| `getBufferIdxAndPhase` | `CodePartitionUtility.cpp` | Compute buffer index with per-channel stagger |
| `getAccumForReuseGroup` | `WSBuffer.cpp` | Compute `accumCnt` SSA value at a given op |
| `replaceBufferReuse` | `WSCodePartition.cpp` | Rewrite alloc uses to point at representative |
| Reuse group formation | `WSCodePartition.cpp` (`doCodePartitionPost`) | Group channels by `buffer.id`, form `ReuseConfig` |
| SMEM `buffer.id` assignment | `WSMemoryPlanner.cpp` | Assign `buffer.id` to SMEM allocs |
| SMEM circular reuse (Phase 4) | `WSMemoryPlanner.cpp` | Form SMEM reuse pairs, maximize copies |
| TMEM `applyAllocationState` | `WSMemoryPlanner.cpp` | Assign `buffer.id` + `buffer.offset` to TMEM allocs |
