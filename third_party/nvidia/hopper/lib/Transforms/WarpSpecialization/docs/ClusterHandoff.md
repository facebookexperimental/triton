# Cluster Handoff Utilities

`ClusterHandoff.cpp` contains the small set of TTGIR builders shared by the
atomic tile scheduler and CLC lowering when a value must be handed off across
physical CTAs. It does not define a new synchronization protocol. Each caller
still owns its leader election, phase management, wait placement, data
movement, and completion rules.

The public interface is declared in
`include/triton/Dialect/TritonNvidiaGPU/Transforms/ClusterHandoff.h`:

```cpp
Value createPersistentMBarrierAlloc(ImplicitLocOpBuilder &builder,
                                    int arriveCount);

Value captureInWarpPartition(Value value, Operation *user);

ArriveBarrierOp createRemoteMBarrierArrive(OpBuilder &builder, Location loc,
                                           Value barrier, Value rank,
                                           Value pred = {});
```

## Why This Is Shared

Both current users materialize cluster communication after optional AutoWS:

* `AtomicTileScheduler.cpp` publishes a base tile ID from physical cluster rank
  zero and waits for every CTA to consume it before reusing the slot.
* `CLCLowering.cpp` protects reuse of the CLC response buffer when a physical
  cluster consumes a valid response.

At that point AutoWS may already have moved the relevant operation into an
isolated `ttg.warp_specialize` partition. Both lowerings need the same mechanics
for making a function-lifetime allocation visible in that partition and for
arriving at an mbarrier owned by another CTA. Sharing those mechanics avoids
giving either lowering a private barrier operation, memory-semantics encoding,
or cluster-acquire abstraction.

## `createPersistentMBarrierAlloc`

This helper constructs a single-slot mbarrier allocation at the builder's
current insertion point:

1. Allocate one `i64` scalar with `createScalarAlloc`.
2. Create the slot-zero view with `createSingleBufferView`.
3. Initialize that view with `ttng.init_barrier` and the caller-provided arrival
   count.
4. Return the allocation, not the view.

Returning the allocation lets the caller create or capture the appropriate
view at the point where it assembles the complete handoff. The helper does not
choose the arrival count:

* The atomic scheduler's full barrier expects one arrival from rank zero.
* Its empty barrier expects one arrival from each of the `K` CTAs.
* CLC's cluster reuse barrier also expects the physical cluster size.

The helper is named "persistent" because both callers position the builder at
function entry and reuse the allocation from a persistent loop. Phase is
carried and toggled by the caller.

## `captureInWarpPartition`

Function-lifetime allocations are initially created in the function body. If
the user operation remains outside an isolated warp partition, the SSA value is
already visible and this helper returns it unchanged. The same is true for an
operation in the `ttg.warp_specialize` default region.

For a user inside a partition region, the helper extends the
`ttg.warp_specialize` partition interface:

1. Append the value to the partition operation's operands.
2. Add a matching block argument to every partition region.
3. Return the block argument belonging to the region that contains the user.

Adding an argument to every partition region keeps the operation's operand and
region signatures aligned. The helper asserts if the user was reported inside
a warp-specialize operation but cannot be found in one of its partition
regions.

This helper does not assign an operation to a partition or move it between
partitions. AutoWS has already chosen and materialized the owner; the helper
only makes an existing outer value available there.

## `createRemoteMBarrierArrive`

An mbarrier is allocated in each CTA's local shared memory. To arrive at the
instance owned by another CTA, this helper:

1. Copies the barrier memdesc type while replacing its memory space with
   `ttng.shared_cluster`.
2. Creates `ttng.map_to_remote_buffer` for the requested physical cluster
   rank.
3. Creates `ttng.arrive_barrier` with count one on the remote memdesc.
4. Applies the optional predicate to the arrival when one is supplied.

The result is the same remote-arrive/local-wait pattern used by the TLX cluster
paths. It uses the existing TTGIR DSM and mbarrier operations directly. There
is no cluster-wide barrier and no extra semaphore or memory-scope parameter:
the mapped remote memdesc identifies the target CTA, while the mbarrier carries
the synchronization state.

The optional predicate is used by CLC so an invalid response does not count as
consumed. The atomic scheduler uses unconditional arrivals because every CTA
consumes every published base tile ID.

## Ownership Boundary

`ClusterHandoff` deliberately does not provide a one-call "broadcast" helper.
The two protocols have different correctness conditions:

| Concern | Atomic tile scheduler | CLC lowering |
|---------|-----------------------|--------------|
| Published data | Base tile ID in a shared slot | Hardware CLC response |
| Producer | Physical cluster rank zero | CLC hardware completion |
| Local ready wait | One full barrier per CTA | CLC completion barrier |
| Reuse arrival | Every CTA, unconditionally | Valid consumers only |
| Reuse wait | Rank zero, before the next atomic claim | Rank zero, before the next CLC issue |

Keeping policy in the callers makes the synchronization counts and predicates
visible next to the data operation they protect. The shared file stays limited
to IR construction that is identical in both paths.

## Related Documentation

* [Atomic Tile Scheduler](AtomicTileScheduler.md)
* [Cross-Partition Atomic Support](CrossPartitionAtomicSupport.md)
* [Overview](Overview.md)
