# Barrier Constraints Design

## Overview

Barrier and token ops (`wait_barrier`, `arrive_barrier`, `producer_acquire`,
`producer_commit`, `consumer_wait`, `consumer_release`) accept an optional
`constraints` argument of type `DictionaryAttr`. This provides a generic,
extensible mechanism for passes to attach context-dependent metadata to
barrier operations without modifying the op definitions.

## Motivation

Different compilation stages need to annotate barrier ops with different
metadata:

- **Subtile lowering** needs to know which tiles should emit a barrier and
  how many buffers to use for phase computation.
- **Pipeline scheduling** needs to track pipeline stages and clusters.
- **Barrier fusion** needs to know which barriers can be merged.

Rather than adding a new attribute to the op definition for each use case
(which couples the op to specific passes), the `constraints` dict provides
a single extensible slot. Each consuming pass defines its own key namespace
and ignores keys it doesn't recognize.

## Design Principles

1. **Optional**: The attribute is `OptionalAttr<DictionaryAttr>`. When absent
   (the default), the barrier behaves exactly as before. All existing code
   is unchanged.

2. **Dict-based**: A `DictionaryAttr` rather than a structured attribute.
   This avoids defining a new TableGen attribute for every combination of
   constraints. Passes validate the keys they care about at use time.

3. **Namespace by convention**: Each pass owns a set of keys. Keys are
   plain strings. No formal namespace enforcement — collisions are avoided
   by using descriptive names.

4. **Argument, not discardable attr**: The `constraints` is declared in
   the op's `arguments` list, not as a discardable attribute. This means:
   - It participates in the op's builder signatures.
   - It's part of the op's identity for comparison/hashing.
   - It won't be silently stripped by passes that drop unknown attrs.
   - It appears in `attr-dict` in the assembly format.

5. **Forward-compatible**: A pass that doesn't understand a key simply
   ignores it. Adding new constraint keys doesn't require changing any
   existing pass.

## Constraint Keys

### Subtile Lowering (`LowerSubtiledRegionPass`)

| Key | Type | Description |
|-----|------|-------------|
| `loweringMask` | `DenseI32ArrayAttr` | Per-tile mask: emit barrier only for tiles where mask[i] != 0. Length must equal number of tiles. Absent = all tiles. |
| `numBuffers` | `I32Attr` | Number of buffer slots for phase computation: `phase = (accumCnt + tileIdx) / numBuffers & 1`. Default 1. |

Example:
```mlir
// Wait only on tile 0, use 2-buffer phase rotation
ttng.wait_barrier %bar, %phase {
  constraints = {loweringMask = array<i32: 1, 0>, numBuffers = 2 : i32}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// Arrive only on tile 1
ttng.arrive_barrier %bar, 1 {
  constraints = {loweringMask = array<i32: 0, 1>}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

### WS Barrier Analysis (`WSBarrierAnalysis.h`)

These keys annotate barriers with the channel-graph metadata needed for
barrier reordering analysis (e.g., pushing a `tmem_load` arrive past
intervening waits).

| Key | Type | Description |
|-----|------|-------------|
| `dstTask` | `I32Attr` | Destination task ID — the foreign partition this barrier communicates with. The source task is the partition where the barrier lives (available via `async_task_id`). |
| `channelGraph` | `DenseI32ArrayAttr` | Set of task IDs reachable from the destination through the channel adjacency graph (excluding the source). Used by `canAdvanceWSBarrier` to check if two barriers can be safely reordered. |
| `parentId` | `I32Attr` | Local ID for the nearest ordered parent scope (`scf.for`, `scf.while`, or containing `tt.func`). Used only inside the current function for V2 ordered-region checks. |
| `minRegionId` | `I32Attr` | Earliest ordered region reached by this channel summary. |
| `maxRegionId` | `I32Attr` | Latest ordered region reached by this channel summary. |

**Lifecycle:**
1. `dstTask` is set when token ops are created in `insertAsyncComm`
   (before code partitioning).
2. `channelGraph`, `parentId`, `minRegionId`, and `maxRegionId` are injected
   after code partitioning via `buildChannelGraph()`,
   `buildWSBarrierOrderedRegionRanges()`, and `injectChannelGraph()`.
3. These fields propagate through `doTokenLowering` to the resulting barrier
   ops.

**Reordering rule:** Same-direction WS barriers can be swapped directly. An
arrive can be delayed past a wait if the graphs are disjoint, or if both
barriers have valid ordered-region metadata in the same parent and the wait's
reached regions are strictly earlier than the arrive's reached regions. This is
checked by `canAdvanceWSBarrier()` and
`canAdvanceWSBarrierArrivePastWait()` (see
[Barrier Reordering](#barrier-reordering) below).

Example:
```mlir
// Producer commit to consumer task 2
nvws.producer_commit %tok, %idx {
  constraints = {WSBarrier = {dstTask = 2 : i32}}
} : tensor<1x!nvws.token>, i32

// After channelGraph and ordered-region injection
ttng.arrive_barrier %bar, 1 {
  constraints = {WSBarrier = {
    dstTask = 2 : i32,
    channelGraph = array<i32: 1, 2>,
    parentId = 0 : i32,
    minRegionId = 3 : i32,
    maxRegionId = 3 : i32
  }}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

See [WS Barrier Ordered Region Tracking](WSBarrierOrderedRegionTracking.md)
for the V2 construction details.

### Pipeline Scheduling (future)

| Key | Type | Description |
|-----|------|-------------|
| `pipelineStage` | `I32Attr` | Which pipeline stage this barrier belongs to. |
| `cluster` | `I32Attr` | Loop cluster for scheduling. |

### Token Ops

The same `constraints` dict is available on the NVWS token ops.
`doTokenLowering` propagates constraints from token ops to the resulting
barrier ops, so any key set on a token op will appear on the lowered
`wait_barrier` / `arrive_barrier`.

```mlir
// dstTask is set during insertAsyncComm
nvws.producer_acquire %tok, %idx, %phase {
  constraints = {WSBarrier = {dstTask = 2 : i32}}
} : tensor<1x!nvws.token>, i32, i1

nvws.consumer_wait %tok, %idx, %phase {
  constraints = {WSBarrier = {dstTask = 0 : i32}}
} : tensor<1x!nvws.token>, i32, i1
```

Token-specific constraint keys can signal to `doTokenLowering` how to
convert the token op — e.g., `subtileChannel = true` could indicate that
the resulting barrier should use per-subtile phase tracking.

## Assembly Format

The constraints appear in the `attr-dict` portion of the assembly:

```mlir
// Without constraints (default)
ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// With constraints
ttng.wait_barrier %bar, %phase {constraints = {numBuffers = 2 : i32}}
    : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// Multiple constraint keys
ttng.arrive_barrier %bar, 1 {
  constraints = {loweringMask = array<i32: 0, 1>, pipelineStage = 0 : i32}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

## Builder API

Custom builders default `constraints` to null so existing callers are
unchanged:

```cpp
// Existing call — still works
WaitBarrierOp::create(builder, loc, barrier, phase);

// With constraints
auto constraints = DictionaryAttr::get(ctx, {
  NamedAttribute(StringAttr::get(ctx, "loweringMask"),
                 DenseI32ArrayAttr::get(ctx, {1, 0})),
  NamedAttribute(StringAttr::get(ctx, "numBuffers"),
                 builder.getI32IntegerAttr(2)),
});
WaitBarrierOp::create(builder, loc, barrier, phase,
                       /*pred=*/Value(), /*deps=*/{}, constraints);
```

## Accessing Constraints

```cpp
if (auto constraints = waitOp.getConstraints()) {
  if (auto mask = constraints.getAs<DenseI32ArrayAttr>("loweringMask")) {
    // Use mask for selective tile emission
  }
  if (auto numBuf = constraints.getAs<IntegerAttr>("numBuffers")) {
    unsigned n = numBuf.getInt();
    // Use n for phase computation
  }
}
```

## Interaction with SubtiledRegionOp

The WSBarrier marker ops (`ws_wait_barrier`, `ws_arrive_barrier`) defined
inside SubtiledRegionOp tile bodies serve a different purpose: they use
attribute-based barrier references (`barrierIdx`) to avoid SSA captures
across `IsolatedFromAbove` boundaries. The `constraints` dict on real
barrier ops is complementary — it annotates the actual `wait_barrier` /
`arrive_barrier` ops that exist outside or after lowering.

The migration path:
1. `doCodePartitionPost` creates token annotations on SubtiledRegionOps
2. `doTokenLowering` converts tokens to real barrier ops with `constraints`
   encoding the subtile context (loweringMask, numBuffers)
3. `LowerSubtiledRegionPass` reads constraints when expanding tiles

Alternatively, WSBarrier marker ops can carry their own `loweringMask`
attribute directly (as currently defined). The two approaches can coexist:
- WSBarrier ops for barriers inside the tile body (attribute-based refs)
- `constraints` dict for barriers outside the SubtiledRegionOp or after
  lowering

## Barrier Reordering

**Files:**
- `nvidia/hopper/include/Transforms/WSBarrierReorder.h` — `canAdvanceWSBarrier`, `canAdvanceWSBarrierArrivePastWait`, `sinkWSArrives`, `raiseWSWaits`, `buildBarrierToMemoryOpMap`, `optimizeWSBarrierLocations`
- `lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp` — consumer of the above

### Motivation

After token lowering, the epilogue region contains interleaved barrier
ops from multiple channels. For example, a `tmem_load` channel's arrive
barrier may sit between a store channel's wait/arrive barriers, preventing
the `tmem_load` from sinking closer to its use. The barrier reordering
step separates barriers from independent channels, unblocking tmem_load
sinking and reducing register pressure.

### Algorithm

The reordering runs as part of the `triton-nvidia-interleave-tmem` pass,
before the existing tmem_load sinking. Four steps:

1. **`buildBarrierToMemoryOpMap`** — For each WS-annotated barrier, record
   its nearest associated memory op (scan backward for arrives, forward for
   waits). This map is used in step 4 to restore barriers near their ops.

2. **`sinkWSArrives` / `raiseWSWaits`** — Push arrive barriers down and
   pull wait barriers up within each basic block. An arrive can move past
   any non-barrier op (delaying the signal is always safe) and past another
   WS arrive. It can move past a wait only if
   `canAdvanceWSBarrierArrivePastWait` accepts either the V1 disjoint-graph
   rule or the V2 ordered-region rule. Waits follow the mirror rule, with an
   additional check to not move past definitions of their operands.

3. **tmem_load sinking (WSBarrier-aware)** — Each `tmem_load` inherits the
   full `WSBarrier` dictionary from its associated arrive barrier, including
   `channelGraph` and ordered-region metadata. When the sinking loop
   encounters a barrier, it calls `canAdvanceWSBarrier` with the `tmem_load`'s
   constraints to decide whether to pass it. All tmem_loads in the same channel
   region (between the arrive and the preceding same-channel barrier) get the
   same constraints, so split tmem_loads are treated uniformly.

4. **`optimizeWSBarrierLocations`** — After sinking, relocate each barrier
   back to an optimal position right next to its associated memory op
   (arrives after, waits before), respecting SSA dominance.

### `canAdvanceWSBarrier`

```cpp
bool canAdvanceWSBarrier(optional<DictionaryAttr> constraintsA,
                         optional<DictionaryAttr> constraintsB);
```

Returns true when both barriers have `WSBarrier.channelGraph` and the two sets
are disjoint (no shared task ID). Returns false conservatively if either
barrier lacks `WSBarrier` or `channelGraph`.

```cpp
bool canAdvanceWSBarrierArrivePastWait(optional<DictionaryAttr> arrive,
                                       optional<DictionaryAttr> wait);
```

This is the stricter opposite-direction check. It first accepts the same V1
disjoint-graph case. If the graphs overlap, it accepts only when both barriers
have valid `parentId`, `minRegionId`, and `maxRegionId`, both `parentId`
values match, and:

```text
wait.maxRegionId < arrive.minRegionId
```

Invalid ordered-region metadata, including `parentId = -1` or region IDs of
`-1`, preserves the conservative V1 behavior. Channels nested under `scf.if`
currently receive invalid ordered-region metadata for this reason.

### Barrier Movement Rules

| Pair | Safety |
|------|--------|
| Arrive, Arrive | Always safe |
| Wait, Wait | Always safe |
| Arrive, Wait | Safe only if `canAdvanceWSBarrierArrivePastWait` returns true |
| Wait, Arrive | Same check (mirror direction) |

### IR Example

Before (barriers block tmem_load sinking):
```mlir
ttng.wait_barrier %bar0, %phase : ...                           // tmem_load wait
ttng.tmem_load %s0 → %v0                                        // stuck here
ttng.tmem_load %s1 → %v1
ttng.arrive_barrier %bar0, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 1, 3>}}} : ...
ttng.wait_barrier %bar1, %phase {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttg.local_store %v0, %smem
ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttg.local_store %v1, %smem
ttng.arrive_barrier %bar2, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
```

After (tmem_loads interleaved with store pipeline):
```mlir
ttng.wait_barrier %bar0, %phase : ...                           // tmem_load wait
ttng.wait_barrier %bar1, %phase {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttng.tmem_load %s0 → %v0                                        // sunk past store wait
ttg.local_store %v0, %smem
ttng.arrive_barrier %bar1, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttng.wait_barrier %bar2, %phase {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
ttng.tmem_load %s1 → %v1                                        // sunk past store wait
ttg.local_store %v1, %smem
ttng.arrive_barrier %bar0, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 1, 3>}}} : ...
ttng.arrive_barrier %bar2, 1 {constraints = {WSBarrier = {
  channelGraph = array<i32: 2>}}} : ...
```
