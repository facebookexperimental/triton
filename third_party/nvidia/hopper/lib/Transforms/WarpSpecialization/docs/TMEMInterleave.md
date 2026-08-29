# TMEM Interleave

`triton-nvidia-interleave-tmem` moves `ttng.tmem_load` values and TMEM
allocation ops later in their block, and moves warp-specialized barriers when
those barriers are known to protect independent channels.

The pass has one objective: **distance a load from its producer.**
`ttng.tc_gen5_mma` is asynchronous, so the consuming `ttng.tmem_load` cannot
retire until the MMA completes. The gap between them is exactly the amount of
independent work available to cover that latency. When producer and consumer
sit in the same partition there is no other warpgroup to hide behind, so that
gap has to come from instruction scheduling. This applies to every load,
whether or not the accumulator is split.

Sinking also shortens the load's own register live range, and when an
accumulator is read as several subtiles it tends to keep the subtiles from
being live at once. That is a useful side effect, not a guarantee, and it is
not what the pass optimizes for. A chain grows through adjacent pure users, so
a sinking load absorbs its consumer and carries it along; that isolates the
subtile value but drags the consumer's *other* operands into a longer live
range. Peak register pressure can come out unchanged even when every subtile
value is separated. Treat subtile liveness as a property to observe, not one
the pass establishes — the measurements that justify this pass are latency
measurements.

The pass is scheduled as a module pass, but its algorithm is block-local:

1. Build a worklist of blocks holding at least one direct `ttng.tmem_load` or
   `ttng.tmem_alloc` op.
2. For each block, collect the TMEM loads and TMEM allocation ops that may move.
3. Reorder WS barriers within that block to unblock legal movement.
4. Sink eligible TMEM loads and allocation ops within that block.
5. Restore WS barriers near the memory operations they guard.

A block holding neither op is skipped. The skip is not just an optimization:
steps 3 and 5 walk the whole block rather than the collected worklist, so
without it a block with no TMEM at all would still have its WS barriers and its
plain TMA store token waits repositioned for no benefit.

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

Every load sinks greedily, as far as legality allows. Split loads are processed
in program order, so each one can move next to its own consumer without making
later subtile values live early. The first load is not kept as an anchor: it is
subject to the same legality checks and movement as every later load.

Split TMEM loads can inherit the `channelGraph` constraints from the guarding
arrive barrier. This lets loads from the same TMEM allocation, but different
subtiles, sink independently around store-channel waits when the channels are
disjoint.

A chain cannot sink past the arrive that releases the buffer it is reading,
because that would let the release fire before the read. Instead the chain
absorbs that arrive and the two move together, which delays the release rather
than reordering it against the load. Only an arrive that already follows the
load is a candidate: the chain walk starts after the chain, so a preceding
arrive is never visited. An arrive whose constraints differ belongs to another
channel and stays subject to the ordinary legality checks. Once a chain has
picked up its own arrive it may also pass a second constrained arrive, since
two arrives only delay signals.

Plain `ttng.async_tma_store_token_wait` ops do not block TMEM load sinking by
themselves. They wait for a TMA store to finish reading SMEM, but do not carry
WS barrier semantics unless they include attached barrier operands. Barrier-
bearing token waits still block movement like other arrive-like operations.

## No Rollback

The pass applies its transformation unconditionally. It does not snapshot the
block, score the result, and restore the original order on a bad score.

A previous version did, keyed on an overlapping-liveness profile. That metric
only ever measured how many candidate load values were simultaneously live, so
it was structurally unable to see producer-to-consumer distance — the thing the
pass exists to widen. A rewrite that opened that gap and shortened a live range
scored as "unchanged" and was discarded. The metric also could not fire at all
on a block with no multi-load group, and the rollback treated that absence of
evidence as grounds to reject, undoing the alloc sinking that had already
happened.

If a scoring heuristic is reintroduced, it has to see distance and not just
overlap. A peak simultaneous count captures multiplicative register pressure
and is blind to the durational pressure and latency exposure this pass targets;
an integral over the liveness curve would subsume the old metric rather than
trade against it.

## Out of Scope: Reshaping Barriers for Codegen

The barrier movement described above is in scope. It exists to unblock legal
TMEM movement, and every barrier still ends up near the memory operation it
guards.

What is out of scope is reshaping barrier placement to unlock a downstream
codegen optimization. The motivating example is hoisting several
`ttng.wait_barrier` ops to a common program point so that a value broadcast to
their consumers becomes optimizable in PTX. That is driven by codegen quality
rather than by TMEM liveness or MMA latency, and on a given block the two
objectives can pull in opposite directions: the placement that best exposes a
broadcast is not generally the placement that best distances a load from its
producer. Handle it in future work on barrier placement rather than adding
cases here.

## Testing

Coverage lives in `test/TritonNvidiaGPU/interleave_tmem.mlir` and should
include:

- single-load blocks, where the load sinks away from its producing MMA
- split-load cases where each load reaches its own consumer independently
- loads whose sinking is blocked by aliasing or barrier constraints, which must
  stay put

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
