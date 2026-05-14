# TMEM Interleave

`triton-nvidia-interleave-tmem` tries to reduce overlapping liveness between
`ttng.tmem_load` values. It does this by moving TMEM loads and their pure
single-use chains closer to their consumers, while also moving warp-specialized
barriers when those barriers are known to protect independent channels.

The pass is scheduled as a module pass, but its algorithm is block-local:

1. Build a worklist of blocks with at least two direct `ttng.tmem_load` ops.
2. For each block, collect the TMEM loads and TMEM allocation ops that may move.
3. Reorder WS barriers within that block to unblock legal movement.
4. Sink eligible TMEM loads and allocation ops within that block.
5. Restore WS barriers near the memory operations they guard.
6. Run rollback analysis for that block before processing the next block.

Blocks with fewer than two direct `ttng.tmem_load` ops are skipped. They cannot
benefit from the pass objective because there is no in-block TMEM load overlap
to isolate.

## Barrier Movement

Barrier movement uses the WS barrier constraints already attached to
`ttng.wait_barrier` and `ttng.arrive_barrier` ops. Barriers from disjoint
`channelGraph`s can move past one another, while barriers with overlapping or
unknown constraints stay ordered.

The pass first sinks WS arrives and raises WS waits. This can expose legal
positions for TMEM loads that were previously blocked by unrelated barrier
traffic. After load sinking, the pass restores barriers to better positions near
the memory operations they protect.

## Load Sinking

For each candidate load, the pass forms a movable chain starting at the
`ttng.tmem_load` and continuing through adjacent pure single-use users. The
chain can sink as a unit as long as the move remains legal for the underlying
TMEM buffer and for the channel constraints associated with the load.

Split TMEM loads can inherit the `channelGraph` constraints from the guarding
arrive barrier. This lets loads from the same TMEM allocation, but different
subtiles, sink independently around store-channel waits when the channels are
disjoint.

## Rollback

The pass keeps a block transformation only when finalized lowering improves the
overlapping liveness of comparable TMEM loads. The decision is made after final
barrier restoration, because the final barrier positions can affect load
liveness.

Rollback uses overlapping liveness occupancy, not total live-range length:

```cpp
struct OverlapLiveness {
  // One entry per contiguous block-order span where at least one candidate
  // tmem_load value is live.
  SmallVector<unsigned> numLiveTMEMLoads;
  // Entries from numLiveTMEMLoads greater than 1, sorted descending. This is
  // the comparison key.
  SmallVector<unsigned> overlapProfile;
};
```

The implementation may compute temporary start/end positions for each load, but
those ranges are not the acceptance metric. `numLiveTMEMLoads` records only
spans where at least one candidate load is live. If two TMEM loads are back to
back and both values are live after the second load, the important value is `2`;
the brief prefix where only one load is live is ignored for the pass objective.

`overlapProfile` is built by dropping all `1` entries from
`numLiveTMEMLoads` and sorting the remaining counts descending. A group is
improved when the final profile is lexicographically smaller than the original
profile.

Examples:

- `[4, 2] -> [4, 1, 1]` succeeds because the profiles are `[4, 2] -> [4]`.
- `[4, 2] -> [3, 3]` succeeds because the maximum overlap decreases.
- `[4, 2] -> [4, 3]` fails.
- `[2, 2, 2, 2] -> [4]` fails because the maximum overlap got worse.

Do not treat a shorter non-overlapping tail, earlier last use, or smaller total
live range as sufficient on its own. The pass goal is specifically to isolate
TMEM load values that were live at the same time.

### Candidate Groups

Rollback compares loads in block-local candidate groups. The current grouping
uses:

1. The derived `memOpConstraints` / `channelGraph` dictionary for the load.
2. The root TMEM allocation returned by `findBufferAccess(load.getSrc())`.

Groups with fewer than two loads are ignored. Groups whose original
`overlapProfile` is empty are also ignored because there was no overlap to
improve.

### Restore Behavior

Before mutating a block, the pass records the original order of non-terminator
ops. If no candidate group had initial overlap, or if any initially-overlapping
group fails to improve, the pass restores that block by moving the same
operations back into the recorded order.

Rollback is intentionally block-level. The pass moves both memory ops and
barriers inside a block, and a load's final liveness often depends on their
coupled placement. Per-load rollback is possible as a future refinement, but it
must preserve shared barrier placement and avoid making another load group's
overlap profile worse.

## Testing

Coverage lives in `test/TritonNvidiaGPU/interleave_tmem.mlir` and should
include:

- single-load blocks staying unchanged
- split-load cases where overlap improves and the transformation is kept
- rollback cases where the final overlap profile does not improve

After changing the C++ implementation, rebuild before testing:

```bash
pip install -e . --no-build-isolation
```

Then run the focused test:

```bash
triton-opt test/TritonNvidiaGPU/interleave_tmem.mlir \
  --triton-nvidia-interleave-tmem \
  --allow-unregistered-dialect | \
  FileCheck test/TritonNvidiaGPU/interleave_tmem.mlir
```
