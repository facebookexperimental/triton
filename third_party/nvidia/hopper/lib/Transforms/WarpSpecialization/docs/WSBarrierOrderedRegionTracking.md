# WS Barrier Ordered Region Tracking

This document describes the V2 ordered-region metadata used by WSBarrier
reordering. It complements [Barrier Constraints](BarrierConstraints.md), which
documents the `constraints.WSBarrier` attribute contract and the lowering-time
consumers.

The design is based on the region-based WSBarrier design doc and the
implementation in:

- `WSBarrierAnalysis.h`
- `WSCodePartition.cpp::injectChannelGraphOnTokenOps`
- `nvidia/hopper/include/Transforms/WSBarrierReorder.h`
- `lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp`

## Motivation

V1 WSBarrier analysis represents each channel by a `channelGraph`: the set of
foreign task IDs reachable from the destination partition through the channel
graph. This proves simple cases by allowing an arrive to move past a wait only
when their reachable task sets are disjoint.

That is enough for isolated GEMM epilogues, but too conservative for addmm with
TMA bias loads. In addmm, the epilogue partition needs to delay the TMEM load
arrive past waits for later logical work, while the TMA load partition contains
both loop-body A/B loads and loop-epilogue bias loads. V1 sees overlapping
partitions and rejects the move, even when the relevant wait is in an earlier
same-iteration region than the arrive being delayed.

V2 keeps the V1 fallback and adds ordered parent/region metadata. The extra
metadata allows some overlapping `channelGraph` cases when both barriers belong
to the same ordered parent and the wait is known to be earlier than the arrive.

## Metadata

The nested `constraints.WSBarrier` dictionary can contain:

| Field | Meaning |
|-------|---------|
| `dstTask` | Destination partition for the channel. The source partition is the op's `async_task_id`. |
| `channelGraph` | V1 reachable foreign task set, excluding the source task. |
| `parentId` | Function-local ID for the nearest ordered parent scope. |
| `minRegionId` | Earliest ordered region reached by the channel summary. |
| `maxRegionId` | Latest ordered region reached by the channel summary. |

`parentId` is local compiler metadata, not an ABI. It only needs to be unique
within the current function. The implementation assigns IDs deterministically as
ordered parents are first encountered during an IR walk, which keeps lit checks
readable.

Invalid ordered metadata uses `-1`. Any barrier with missing metadata or a
negative `parentId`, `minRegionId`, or `maxRegionId` falls back to the V1
disjoint-graph rule.

## Ordered Parents

`getNearestWSBarrierParent()` walks from a token op to its enclosing ordered
parent. The valid parents are:

- nearest `scf.for`
- nearest `scf.while`
- containing `tt.func`

If the walk crosses `scf.if`, the token receives invalid ordered metadata.
Conditional channels therefore keep the V1 behavior for now.

Barrier movement also stops at region-bearing operations. `canAdvanceWSBarrier`
returns false for any operation with regions, so WSBarrier sinking and raising
stay within one basic-block region and do not move barriers into or out of
control flow.

## Region Numbering

Each ordered parent is split into ordered regions by its direct child
region-bearing operations. For each block owned by the parent, the first
segment starts as one region, and every direct child op with regions starts the
next segment.

For a parent with `N` ordered regions:

- Forward tokens use region IDs `1..N`.
- Backward tokens use region IDs `N+1..2N`.

Forward tokens are:

- `nvws.producer_commit`
- `nvws.consumer_wait`

Backward resource-reuse tokens are:

- `nvws.producer_acquire`
- `nvws.consumer_release`

Splitting forward and backward ranges prevents a data-availability edge from
being confused with a resource-reuse edge in the same program region.

## Channel Range Construction

`buildWSBarrierOrderedRegionRanges()` runs after `insertAsyncComm()` has
created token ops with `WSBarrier.dstTask`. It starts from the V1
`buildChannelGraph()` result and computes a region summary for every WS token.

For each token node, the initial range is its own ordered region:

```text
minRegionId = regionId
maxRegionId = regionId
```

The range is then extended in two conservative same-iteration cases:

1. Same-region peer channels are grouped when they share the same parent,
   direction, and ordered region, and are visible through the starting token's
   V1 graph. This makes the same relationship visible from both participating
   partitions.
2. For wait tokens, later regions in the same source partition, parent, and
   direction are unioned into the range. This evaluates an arrive-past-wait
   move from the delayed arrive's perspective.

The range does not try to prove moves across multiple loop iterations. It only
summarizes same-iteration ordered regions.

## Injection

`WSCodePartition.cpp::injectChannelGraphOnTokenOps()` writes the V1 and V2
metadata back to token constraints:

1. Build the V1 partition reachability graph with `buildChannelGraph()`.
2. Build ordered-region ranges with `buildWSBarrierOrderedRegionRanges()`.
3. For each WS token with one `async_task_id` and a valid `dstTask`, inject:
   - `channelGraph`
   - `parentId`
   - `minRegionId`
   - `maxRegionId`

`doTokenLowering()` then propagates the token constraints to the lowered
`ttng.wait_barrier` and `ttng.arrive_barrier` ops.

## Reordering Rule

For an arrive delayed past a wait, the implementation checks:

```text
if channelGraph sets are disjoint:
  allow
else if parentId matches and wait.maxRegionId < arrive.minRegionId:
  allow
else:
  reject
```

The first branch is V1. The second branch is the V2 same-parent,
same-iteration ordered-region relaxation. Missing or invalid region metadata
rejects the second branch.

Same-direction moves are simpler:

- Arrive past arrive is safe when both are WSBarrier arrives.
- Wait before wait is safe when both are WSBarrier waits.

Opposite-direction movement always goes through
`canAdvanceWSBarrierArrivePastWait()`, including the mirror case where a wait
is raised before an arrive.

## TMEM Load Sinking

`InterleaveTMem` uses the same WSBarrier constraints for `tmem_load` sinking.
For each WS arrive in a block with TMEM loads, it walks backward through the
same channel region and assigns the arrive's `WSBarrier` dictionary to every
`tmem_load` it finds. The sinking pass then calls `canAdvanceWSBarrier()` when
the load encounters another barrier.

This is necessary because the `tmem_load` represents work protected by the same
wait/arrive pair as the channel. Split TMEM loads must all carry the same
constraints, otherwise one split could move past a barrier that another split
cannot.

## Conservative Cases

The implementation intentionally keeps these cases conservative:

- Channels nested under `scf.if` get invalid ordered metadata and use V1 only.
- Barriers are not moved across region-bearing operations.
- V2 proves only same-iteration ordering.
- Resource-aware reasoning, such as proving additional safety from buffer
  independence or aliasing information, is not modeled yet.
- If either compared barrier lacks `WSBarrier` or `channelGraph`, reordering is
  rejected.

## Debugging

Useful checks:

- Before token lowering, inspect `nvws.*` token ops for
  `constraints = {WSBarrier = {...}}`.
- After token lowering, inspect `ttng.wait_barrier` and
  `ttng.arrive_barrier` for the same nested dictionary.
- For V1 behavior, check `channelGraph`.
- For V2 behavior, check that both barriers have the same `parentId` and that
  the wait's `maxRegionId` is less than the arrive's `minRegionId`.

Lit coverage should prefer checking the nested `WSBarrier` dictionary rather
than relying on comments around a specific barrier placement.
