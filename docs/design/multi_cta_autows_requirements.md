# AutoWS requirements for multi-CTA operations

## Summary

This document extends the
[multi-CTA control-flow contract](multi_cta_control_flow_contract.md) to
compiler-driven warp specialization.

AutoWS is a uniform transformation of the original CTA program. Until
multi-CTA lowering runs during code partitioning, every CTA has the same
partially transformed program, including the same task assignments, data
partitions, and channels. Physical warp-specialize regions are created later.
CTAs may evaluate the common program differently when the original program
contains CTA-dependent values, but AutoWS does not choose a different program
for each CTA rank.

Multi-CTA lowering is the phase that intentionally introduces asymmetric CTA
roles, such as a TMA multicast leader and its recipients. The design therefore
needs to answer four questions:

1. Which multi-CTA facts must be established before AutoWS, and which lowering
   preconditions can be checked during AutoWS code partitioning?
2. How does an AutoWS channel's ready/done synchronization protocol become
   multi-CTA barrier handling?
3. Under what conditions does a value transported through an AutoWS channel
   retain the equality facts required by the control-flow contract?
4. How must cluster communication account for independently executing
   warp-group partitions within each CTA?

The running example is a descriptor load selected for TMA multicast across
ranks 0 and 1 of a `(2, 1, 1)` cluster.

## Proof placement around AutoWS

The compilation model is:

```text
one original CTA program
  -> Multi-CTA Validation
  -> uniform AutoWS task and data partitioning
  -> channel discovery and buffer planning
  -> multi-CTA lowering during code partitioning
  -> physical warp specialization and token/barrier lowering
```

These phases do not create two versions of the kernel. Validation establishes
the semantic facts. AutoWS lowering selects a supported strategy using the
actual partition, channel, and buffer graph, then adds CTA roles and cross-CTA
synchronization as part of code partitioning.

### What code partitioning can use directly

Lowering is most direct when a candidate operation has no incoming
cross-partition data dependencies. The operation, its enclosing control flow,
and the values covered by its multi-CTA proof must all be available in the
owning partition. Code partitioning can then inspect:

- the partition that owns each candidate operation;
- its enclosing `scf.if`, `scf.for`, and `scf.while` operations and their task
  assignments;
- its direct SSA dependencies; and
- pure expressions whose operands have established CTA-group equality.

This commonly applies to producer operations. For example, a descriptor load in
a producer partition can be analyzed directly when its enclosing conditions and
loop bounds are captured from kernel arguments or rematerialized from
group-uniform operands. The uniform transformation also makes its owning
partition a static property of the program: the operation has the same owner in
every CTA.

This proof establishes the producer operation's control-flow trace. It is not
sufficient to determine which cluster communication operations are required.
That decision also depends on the consumer partitions, buffer lifetime, ready
and done paths, and the hardware participation scope. If any controlling value
reaches the producer through a channel, the stronger channel-provenance rules
below apply instead.

### Data dependencies through intermediate buffers

When a value crosses warp-group partitions, AutoWS may replace the original SSA
dependency with a synchronized intermediate SMEM or TMEM buffer. This applies
to ordinary operation operands and to values that control whether or when an
operation executes, including branch predicates and loop continuation or
termination conditions.

For example, a scheduler partition may compute whether a persistent loop has
more work, store that condition in an intermediate buffer, and broadcast it to
the other partitions. The channel proves that those partitions within one CTA
observe the scheduler partition's value. It does not prove that scheduler
partitions in different CTAs produced the same condition or exit on the same
iteration.

After this rewrite, a `local_load` exposes the immediate buffer dependency but
not whether the original value came from a kernel argument, a CTA coordinate, a
scheduler result, or mutable memory. Code partitioning may use that channel in a
multi-CTA lowering only when it can associate the payload with an established
equality fact and dynamic iteration. The local wait and load alone are
insufficient.

### Two-phase protocol

The design has two stages:

1. **Multi-CTA Validation.** This stage must run before any AutoWS pass. It
   proves CTA-group equality, whole-path reachability, and loop exits while the
   original SSA provenance is visible. It records which operations satisfy the
   multi-CTA control-flow contract but does not introduce CTA roles or
   synchronization.
2. **AutoWS lowering.** This stage runs during code partitioning, after the
   owning partitions and producer-consumer graph are known. It integrates the
   multi-CTA operation with AutoWS token and barrier handling and is responsible
   for producing correct cluster communication semantics. If the final
   partition or channel shape has no supported lowering, this stage aborts the
   optional multi-CTA transformation and retains the original per-CTA
   operation.

The later multi-CTA barrier-handling section describes the full/empty-path and
backend synchronization requirements of AutoWS lowering in detail.

If validation runs before data partitioning, AutoWS must retain a mapping from
each validated source operation to any resulting clone or slice, including its
execution count, tile, and pipeline epoch. If that mapping is unavailable, or
if no lowering pattern understands the result, AutoWS lowering aborts the
multi-CTA transformation.

## Multi-CTA barrier handling

The high-level challenge is that a multi-CTA operation cannot be lowered as an
isolated instruction. Every dynamic multi-CTA operation conceptually
participates in at least two coordinated events for its logical buffer or
resource:

- **Ready:** production is complete and every participating CTA may consume its
  destination.
- **Done:** every participating CTA has completed all uses of its destination,
  so the producer may reuse the logical buffer slot.

Both events are group-level properties. It is insufficient to prove only that
the leader can issue an operation or that one recipient has completed. The
lowering must connect the operation that produces ready state with the
operation or operations that establish done state.

Barrier handling includes both the abstract synchronization constructed during
AutoWS code partitioning and the concrete barriers emitted by the backend. Code
partitioning may call these states full and empty and represent them with
tokens or logical semaphores. A later backend stage may use mbarriers or another
hardware protocol. These are different implementations of the same ready/done
protocol.

### TMA multicast example

For the `(2, 1, 1)` TMA multicast example, multicast completion is the ready
event for the destination buffers in ranks 0 and 1. The corresponding done
event comes from completion of the consumers of those buffers, often an MMA.
The relevant lowering unit is therefore the complete ready/done barrier
protocol, not the TMA instruction in isolation.

### AutoWS lowering requirements

AutoWS splits the original CTA program into regions that may start and make
progress independently. Multi-CTA barrier lowering must preserve three
properties across those regions:

1. **Expected arrivals.** For each ready or done barrier, lowering must identify
   the specialized regions and hardware threads expected to arrive. The
   initialized arrival count and the emitted arrivals must still match after
   code partitioning. If several regions participate in one logical CTA
   arrival, lowering must coordinate them rather than accidentally counting the
   CTA more than once or omitting it. This will actually happen in the same location
   other barriers are updated but must be tracked, with the correct masks.
2. **Barrier lifetime.** Barrier storage and state must remain live for every
   specialized region that can use it, for the full duration of the kernel or
   enclosing persistent loop. Correctness cannot depend on producer and
   consumer regions launching or progressing concurrently.
3. **Iteration-count agreement.** Every participating region and CTA must
   execute the same ordered number of uses of the corresponding barrier. Code
   partitioning must not make that count data-dependent in a way that can vary
   between tiles, partitions, or CTA participants. Multi-CTA Validation covers
   source-level exits; lowering must preserve that result when control flow is
   distributed across specialized regions.

This document assumes the existing AutoWS indexing and phase construction are
valid. The requirement is agreement on the dynamic barrier-use count, not a new
analysis of arbitrary indices or phase values. If lowering cannot establish the
expected arrivals, lifetime, or iteration-count agreement, it must abort the
optional multi-CTA transformation.

Current AutoWS 2-CTA MMA is a narrow precedent. It inserts a pair-scoped remote
mbarrier protocol after warp specialization, when the owning MMA partition is
known. Both CTAs arrive on the even CTA's barrier, and only the even CTA waits
and issues the cooperative MMA. This avoids placing a full-cluster barrier
inside one consumer warp group, but still relies on the paired CTAs executing
the same dynamic MMA and barrier instances.

## Future work

### Dead-tile participation

Multi-CTA operations have a fixed participation width. The scheduled tile count
along the grouping dimension must therefore be divisible by that width. For
example, every dynamic 2-CTA GEMM requires two CTAs to participate, even when
the logical matrix has work for only one CTA in the final pair. The schedule
must add a **dead tile** to complete that pair.

The question is not whether the CTA assigned the dead tile may opt out. It is
how that CTA can participate in the required multi-CTA operation without
producing an incorrect result. The current implementation keeps it on the normal
cooperative execution path and requires its input and output data harmless. This
includes masking on any store or any intermediate value.

This differs from ordinary edge masking. A partially valid tile still produces
some logical output, but there are known issues with trying to write fully
past the bounds with TMA stores (although this may just be a bug),

The current 2-CTA matmul test demonstrates this for `M = 128` and
`BLOCK_M = 128`. Its launch wrapper starts two CTAs even though there is only
one logical M tile. Both CTAs execute the same K loop, descriptor loads,
cross-CTA synchronization, and 2-CTA MMA. The second CTA's A tile is entirely
out of bounds, so the descriptor load supplies padding values. The non-AutoWS
kernel suppresses its output with a masked store; the AutoWS kernel relies on
descriptor-store bounds behavior. The dead CTA therefore performs dummy work
rather than skipping the collective operation.

This is not yet in scope, but this will be an important followup for solidifying all
stable multi-cta handling. For AutoWS this may be relevant if need to perf the multi-CTA
op (e.g. load for TMA-multicast), but may not need to perform the rest of the kernel.
