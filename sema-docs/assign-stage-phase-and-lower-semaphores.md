# AssignStagePhase and LowerSemaphore

[← Back to README](README.md)

## Contents

- [Division of responsibility](#division-of-responsibility)
- [LowerSemaphore order](#lowersemaphore-order)
- [The state carried by AssignStagePhase](#the-state-carried-by-assignstagephase)
- [Updating `state.stage`](#updating-statestage)
  - [Schedule-local copies of `state.stage`](#schedule-local-copies-of-statestage)
- [Updating `state.phases`](#updating-statephases)
  - [Why clone a phase value for each `loop.stage`?](#why-clone-a-phase-value-for-each-loopstage)
  - [Proving that pipeline stages use disjoint buffer stages](#proving-that-pipeline-stages-use-disjoint-buffer-stages)
  - [Proving that circular acquires use partition-disjoint buffer stages](#proving-that-circular-acquires-use-partition-disjoint-buffer-stages)
  - [Cloning a split phase value into multiple partitions](#cloning-a-split-phase-value-into-multiple-partitions)
- [Structured control flow metadata](#structured-control-flow-metadata)
- [Additional lowering invariants](#additional-lowering-invariants)
- [Code map](#code-map)

## Division of responsibility

`InsertSemas` assigns ownership, `loop.stage`/`loop.cluster`, arrive counts,
optional stage offsets, and `pending_count` on every original
`semaphore.create`. LowerSemaphore uses that count unchanged for every
semaphore it keeps. When it combines several SMEM semaphore pairs, it creates
a replacement empty/full pair, derives `pending_count` from the replacement
acquire/release operations, and records it on the two new `semaphore.create`
operations. Hardware lowering consumes the recorded counts verbatim.
`AssignStagePhase` computes the final buffer stage for each acquire, buffer,
and release, plus the wait phase for each acquire. LowerSemaphore consumes
those values to emit hardware barrier operations.

```text
current buffer stage (advanced on a fresh write) + stage offset
  -> AssignStagePhase: buffer stage, wait phase
  -> LowerSemaphore: mbarrier view, wait, arrive/commit
```

A **semaphore group** is keyed by the first buffer operand of each
`nvws.semaphore.create`. A semaphore can contain several buffers. Using only
the first buffer as the key is safe because every used semaphore sharing any
backing buffer must have the exact same ordered buffer operand list.

`InsertSemas` creates related semaphores from one ordered buffer list.
`combineSemaphores` creates one replacement empty/full pair whose two
semaphores use the same concatenated ordered buffer list. `SemaphoreCreateOp`
verification rejects shared-buffer semaphores whose ordered lists differ. For
example:

```text
%empty = nvws.semaphore.create %a, %b released = -1
%full  = nvws.semaphore.create %a, %b
```

The optional `released` value is an i32 stage mask. Bit `i` set to one means
physical stage `i` begins released; a zero bit means it begins blocked. An
omitted mask is zero. `released = -1` is the legacy all-stages-released state.

Both semaphores belong to the group keyed by `%a`. The shared-buffer invariant
additionally guarantees that both contain the complete ordered list
`[%a, %b]`. The group shares one buffer depth and one canonical `state.stage`.
Its semaphores still have separate `state.phases` entries, further separated
by partition. Grouping therefore means “share the buffer cursor,” not “share
one phase.”

[↑ Back to contents](#contents)

## LowerSemaphore order

The `NVWSLowerSemaphore` pass runs these steps in order:

1. `combineSemaphores`
   - For each eligible set of two or more SMEM empty/full pairs, create exactly
     one replacement empty/full pair.
   - Concatenate their ordered buffer lists, build one acquire/buffer/release
     chain on each side, and derive `pending_count` for both replacement
     semaphores.

2. `multiBufferSemaphore`
   - For a TMA-load semaphore group without authored `buffer.copy`, widen its
     buffers and semaphores to the effective loop depth.
   - Repeat the old released-stage pattern across the resized ring. In
     particular, a released depth-one stage initializes every new stage as
     released.

3. `NVWSAssignStagePhase`
   - Assign the buffer stage to every acquire, buffer, and release.
   - Assign the wait phase to every acquire.

4. `LowerSemaphoreCreate`
   - Allocate and initialize the mbarriers.
   - Lower acquire, release, and buffer operations to waits, arrivals/commits,
     and buffer-stage-indexed views.
   - Remove the semaphore operations.

5. Cleanup
   - Hoist poison tokens with `hoistPoisonOps`.
   - Coalesce compatible TMEM allocations with
     `coalesceDuplicateTmemAllocsByBufferId`.
   - Remove TMEM memory-plan attributes with `stripTmemBufferAttrs`.

Rewrites derived from an annotated semaphore operation copy its partition,
optional WS tag, `loop.stage`, and `loop.cluster` (`assignStageCluster`).
Barrier allocation/initialization and invalidation/deallocation receive no
metadata. Poison tokens are placeholder values substituted for erased
semaphore tokens that WS loops still yield. A replacement poison copies a
partition only if its `semaphore.create` has one; creates are normally
unpartitioned, so these poisons are normally unannotated. After semaphore
lowering, all poison operations are hoisted to function entry so downstream
partitioning does not encounter an unannotated poison inside WS code.

[↑ Back to contents](#contents)

## The state carried by AssignStagePhase

`AssignStagePhase` walks each semaphore group in program order. Two pieces of
state determine the operands written on an acquire:

```text
state.stage       one canonical cursor for the group's circular buffer
state.phases[key] phase history for one partition and one semaphore
```

The normal phase key is `(partition, semaphore)`. Under the conditions proved
later, the pass may keep one copy per pipeline stage, keyed by
`(partition, semaphore, loop.stage)`. `loop.cluster` is never part of a phase
key.

Each phase value uses the multiphase representation: an `i32` bitmask with one
phase bit per physical buffer stage. Its exact update is shown below.

For a group of depth `D`, initialization is:

```text
state.stage = D - 1

state.phases[key] = ~released_mask
```

An omitted released mask is zero, so every buffer stage starts with phase 1.
`released = -1` produces phase zero for every stage. Other masks initialize
each stage independently; for example, `released = 5` produces phase mask
`-6`, with physical stages 0 and 2 released and stage 1 blocked.
The `i32` representation supports at most 32 physical buffer stages;
`AssignStagePhase` rejects a deeper semaphore group before rewriting it.

[↑ Back to contents](#contents)

## Updating `state.stage`

An acquire advances the canonical cursor only when its first reachable buffer
access is a fresh write. Reads and non-fresh writes leave it unchanged. The
stage offset selects the buffer stage for the current operation; it does not
change the canonical cursor:

```text
if acquire starts a fresh write:
    state.stage = (state.stage + 1) mod D

baseStage    = state.stage
acquireStage = positive_mod(baseStage + stageOffset, D)
```

The acquire receives `acquireStage`. The unshifted `baseStage` follows its token
through loops and `if` results. When `InsertSemas` supplies stage offsets, each
buffer and release applies its own offset to that base; without offsets, they
use the base directly.

For example, with `D = 3`:

```text
initial state.stage = 2

access                    state.stage after access    acquireStage
K fresh, offset  0                   0                     0
V fresh, offset  0                   1                     1
K read,  offset -1                   1                     0
```

The read addresses buffer stage 0, but `state.stage` remains 1. That distinction
is why the offset must never be written back into `state.stage`.

[↑ Back to contents](#contents)

### Schedule-local copies of `state.stage`

There is one logical cursor, the canonical `state.stage`. It is the only cursor
carried into the loop, yielded from the loop, and used after the loop.

When one SSA chain cannot obey the loop schedule, the pass also creates one or
more schedule-local copies of that cursor calculation:

```text
state.stage                                  canonical loop-carried cursor
state.localStages[(loop.stage,loop.cluster)] schedule-local SSA copy
```

These are SSA values, not additional hardware cursors. A copy is created only
for a `(loop.stage, loop.cluster)` that needs the same cursor value without a
backward SSA dependency.

The examples below write `schedule=(a,b)` as shorthand for
`{loop.stage=a, loop.cluster=b}`. `loop.stage` places an operation in a
software-pipeline stage. Within one `loop.stage`, `loop.cluster` orders the
operations: a value at `(0,1)` may feed an operation at `(0,4)` in the same
iteration, but a value at `(0,4)` may not feed an operation at `(0,1)` in that
iteration.

Here is a depth-2 shape before assignment. Types, phases, and releases are
omitted because this example shows only the stage dataflow. The first two
acquires lead to fresh writes, so each advances the buffer cursor. The last two
lead to reads, so neither advances it:

```text
scf.for ... {
  %t1 = acquire %empty[offset=0]  schedule=(0,1)
  W A through %t1                                    // fresh: advance

  %t2 = acquire %empty[offset=0]  schedule=(0,4)
  W B through %t2                                    // fresh: advance

  %t3 = acquire %full[offset=0]   schedule=(0,4)
  R B through %t3                                    // non-fresh: no advance

  %t4 = acquire %full[offset=0]   schedule=(0,1)
  R B through %t4                                    // non-fresh: no advance
}
```

With only the canonical chain, the fourth acquire would consume the cursor
produced by the second fresh write:

```text
first advance (0,1) -> second advance (0,4) -> fourth acquire (0,1)
                                                   ^ backward cluster edge
```

An SSA value has one defining operation, and that operation has one
`(loop.stage, loop.cluster)`. Here the logical cursor after the second advance
is required at two different schedule points:

```text
(0,4)  the second and third acquires need it
(0,1)  the fourth acquire needs it
```

Using the `(0,4)` definition at `(0,1)` creates a backward SSA dependency.
Using the cursor before the second advance selects the wrong buffer stage.

The cursor advance is pure integer bookkeeping: it does not execute either
memory access. The pass can therefore calculate the same logical cursor twice,
once at each required schedule point. `AssignStagePhase` emits identical
canonical and local first advances at `(0,1)`; the following CSE merges them.
The resulting stage dataflow is:

```text
scf.for ... iter_args(%stage = %stageIn, ...) {
  %s1_c1 = (%stage + 1) mod D       schedule=(0,1)
  acquire bufferStage=%s1_c1        schedule=(0,1)

  %s2_c4 = (%s1_c1 + 1) mod D       schedule=(0,4)  // canonical
  %s2_c1 = (%s1_c1 + 1) mod D       schedule=(0,1)  // local copy
  acquire bufferStage=%s2_c4        schedule=(0,4)

  acquire bufferStage=%s2_c4        schedule=(0,4)  // no advance
  acquire bufferStage=%s2_c1        schedule=(0,1)  // no advance

  scf.yield %s2_c4, ...
}
```

Immediately after the second advance, the walk's state is:

```text
state.stage              = %s2_c4  // canonical value at (0,4)
state.localStages[(0,1)] = %s2_c1  // copy requested at (0,1)
```

`%s2_c4` and `%s2_c1` contain the same cursor integer. The third acquire uses
the canonical value at `(0,4)`. The fourth uses the local value at `(0,1)`, so
there is no same-iteration `(0,4) -> (0,1)` dependency. Only `%s2_c4` is
loop-carried; `%s2_c1` is discarded when the walk leaves the loop.

The analysis selects at most the first qualifying loop. In that loop, copies
handle direct, same-iteration non-fresh acquires and direct authored-offset
buffer/release consumers only when the scalar loop backedge feeding each copy
is schedule-legal. Other loops and fresh-write, recurrence, or nested shapes
retain only the canonical scalar chain.

[↑ Back to contents](#contents)

## Updating `state.phases`

After `acquireStage` is known, the acquire updates one `state.phases[key]`
value for each applicable group-partition lane and receives the resulting
`waitPhase`. A partitioned acquire selects its listed partitions. An
unpartitioned acquire selects all partitions represented in the semaphore
group, so it can update multiple phase-state entries.

For each selected phase key, an acquire of buffer stage `s` changes only bit
`s`:

```text
word = state.phases[key] xor (1 << acquireStage)
state.phases[key] = word
waitPhase = (word >> acquireStage) & 1
```

The per-stage bitmask is distinct from phase cloning. The bitmask records the
history of physical buffer stages inside one `state.phases[key]` value. Phase
cloning, described next, controls how many such values exist for one
`(partition, semaphore)` pair.

[↑ Back to contents](#contents)

### Why clone a phase value for each `loop.stage`?

Suppose acquires X and Y use the same partition `P` and semaphore `S`, but X is
in `loop.stage 0` and Y is in `loop.stage 1`. One shared value creates this SSA
chain:

```text
state.phases[(P,S)] -> X(i) -> Y(i) -> X(i+1)
```

The software pipeline executes stage 0 of a newer iteration before stage 1 of
an older iteration. The shared chain can therefore make `X(i+1)` depend on
`Y(i)`, even though the schedule places X first. To remove that scheduling
conflict, the pass would like to clone the value:

```text
before:  state.phases[(P,S)]

after:   state.phases[(P,S,loop.stage=0)]   used and updated by X
         state.phases[(P,S,loop.stage=1)]   used and updated by Y
```

This transformation is called phase splitting in the code. It is legal only
under the buffer-stage disjointness condition below.

[↑ Back to contents](#contents)

### Proving that pipeline stages use disjoint buffer stages

For one `(partition, semaphore)`, phase splitting is allowed only when
different `loop.stage` values access disjoint sets of buffer stages across all
loop iterations:

```text
allowed                         forbidden

loop.stage 0 -> {0,2}           loop.stage 0 -> {0,1}
loop.stage 1 -> {1,3}           loop.stage 1 -> {1,2}
no shared buffer stage          both use buffer stage 1
```

In the allowed case, each phase value owns the complete phase history of its
buffer stages. In the forbidden case, two independent phase values would
update buffer stage 1 and split its phase history in two. Any overlap is
forbidden.

The pass must prove this without enumerating every loop iteration. The access
pattern repeats according to `D`, the circular-buffer depth, and `A`, the
number of fresh-write cursor advances per iteration. Their greatest common
divisor, `G = gcd(D,A)`, divides the accesses into repeating buffer-stage
classes. Splitting is permitted only when `G > 1` and no class is used by more
than one `loop.stage`; `G > 1` alone is not sufficient. The formulas below
compute those classes.

Let:

```text
D = circular-buffer depth
A = number of fresh-write cursor advances in one loop iteration
p = an acquire's position in that advance sequence
o = its stage offset
```

`p` starts at zero and increments before each fresh-write acquire is recorded;
a non-fresh acquire uses the number of advances already seen. The cursor starts
at `D - 1`. In iteration `i`, that acquire addresses:

```text
bufferStage(i) = positive_mod(D - 1 + p + o + i*A, D)
```

Each iteration therefore moves the acquire's buffer stage by `A` around a ring
of size `D`. The pass assigns the acquire this class:

```text
class = positive_mod(p + o, G)
```

Acquires with the same class eventually address the same buffer stages.
Acquires with different classes never do. Phase values may be cloned per
`loop.stage` only when every class belongs to exactly one `loop.stage`.

Safe example — `%S` is initially released and guards a four-copy buffer, so
the physical barrier array is `%S[0..3]`. Both acquires use partition `P` and
semaphore `%S`; their next buffer uses are fresh writes:

```text
// Input pseudo-IR. Both stage operands are offset 0.
scf.for ... {
  %tx = nvws.semaphore.acquire %S[offset=0]
          {partition=P, loop.stage=0, loop.cluster=1}
  W X through %tx

  %ty = nvws.semaphore.acquire %S[offset=0]
          {partition=P, loop.stage=1, loop.cluster=1}
  W Y through %ty
}
```

There are two fresh-write advances per iteration, so `D = 4`, `A = 2`, and
`G = 2`:

```text
acquire   loop.stage   p   o   class       buffer stages over iterations
X             0        1   0     1                      {0, 2}
Y             1        2   0     0                      {1, 3}
```

The buffer-stage sets are disjoint, so the output carries two phase words. The
suffix `LS0` or `LS1` names the owning `loop.stage`; it does not name a buffer
stage:

```text
// Output pseudo-IR. Every fresh write advances by exactly one.
scf.for ... iter_args(%stage = 3,
                      %phaseLS0 = 0,
                      %phaseLS1 = 0) {
  %xStage = (%stage + 1) mod 4                       loop.stage=0
  %xWord  = %phaseLS0 xor (1 << %xStage)             loop.stage=0
  %xWait  = (%xWord >> %xStage) & 1                  loop.stage=0
  %tx = nvws.semaphore.acquire %S[%xStage, %xWait]   loop.stage=0

  %yStage = (%xStage + 1) mod 4                      loop.stage=1
  %yWord  = %phaseLS1 xor (1 << %yStage)             loop.stage=1
  %yWait  = (%yWord >> %yStage) & 1                  loop.stage=1
  %ty = nvws.semaphore.acquire %S[%yStage, %yWait]   loop.stage=1

  scf.yield %yStage, %xWord, %yWord
}
```

The three loop-carried values are:

```text
state.stage              = %yStage    canonical buffer cursor
state.phases[(P,S,0)]    = %xWord     phase bits for X's buffer stages {0,2}
state.phases[(P,S,1)]    = %yWord     phase bits for Y's buffer stages {1,3}
```

Their first three iterations are:

```text
iteration   X buffer stage   phaseLS0 -> xWait   Y buffer stage   phaseLS1 -> yWait
    0              0              0 -> 1 / 1            1              0 ->  2 / 1
    1              2              1 -> 5 / 1            3              2 -> 10 / 1
    2              0              5 -> 4 / 0            1             10 ->  8 / 0
```

For example, decimal word 5 is binary `0101`: only the bits for buffer stages 0
and 2 have been changed by X. Decimal word 10 is binary `1010`: only the bits
for buffer stages 1 and 3 have been changed by Y. The pass cloned compiler SSA
phase values; it did not clone `%S[0..3]` or the backing buffers.

In map notation, the same updates are:

```text
state.phases[(P,S,0)] updates only bits for buffer stages {0,2}
state.phases[(P,S,1)] updates only bits for buffer stages {1,3}
```

Unsafe example — `D = 2`, `A = 2`:

```text
acquire   loop.stage   p    o   class       buffer stages over iterations
X             0        1   -1     0                       {1}
Y             1        2    0     0                       {1}
```

Both acquires use buffer stage 1, whose correct phase history is:

```text
X(i) -> Y(i) -> X(i+1)
```

But the expanded pipeline places `X(i+1)` before `Y(i)`. Preserving that buffer
stage's history makes X wait for Y and can deadlock before the same partition
reaches Y. Removing that dependency lets X accept a completed generation from
an earlier access and creates a race. There is no correct independent phase
value for each `loop.stage`, so the pass rejects the clone.

This restriction does not apply to `state.stage` replay. Cursor replay merely
recomputes the same buffer stage; it does not divide one buffer stage's phase
history into independent values.

Before applying this disjointness proof, the pass also requires a single direct
loop body, static `loop.stage` values, constant stage offsets, and at least one
cursor advance. These conditions make the acquire sequence and its classes
provable.

[↑ Back to contents](#contents)

### Proving that circular acquires use partition-disjoint buffer stages

Every physical barrier `%S[k]` has one ownership history, regardless of which
partition acquires it. The phase arithmetic can correctly assign alternating
phases to that history, but it does not order independently executing
partitions. If consecutive owners of `%S[k]` are distributed across partitions,
one partition can lap the other and issue its next same-phase wait before the
intervening opposite-phase acquire has completed. Because an mbarrier wait
observes rather than consumes a completed generation, the later wait can
accept the earlier generation. This is an acquire race and can give both
partitions ownership of the same buffer stage.

The pass reuses the slot-class proof above, but uses the partition ID as the
owner. For all statically scheduled, single-partition acquire operations of the
same circular semaphore, regardless of `loop.stage`, every class

```text
class = positive_mod(p + o, gcd(D,A))
```

must belong to exactly one partition. Acquires in different classes never
reach the same physical buffer stage, so every physical slot's acquire history
remains in one partition. A class owned by multiple partitions is rejected.
Different `loop.cluster` or `loop.stage` values do not make a cross-partition
overlap safe: after loop partitioning, neither attribute orders partition `P`
against partition `Q`.

Grouping this proof by the whole semaphore, rather than separately by
`loop.stage`, is necessary even when phase splitting itself is legal. Consider
`D = 4`, `A = 2`, and `G = 2`:

```text
acquire   partition   loop.stage   class
X             P           0          1
Y             Q           0          0
Z             P           1          0
```

Within `P`, X and Z use disjoint classes, so the phase-split proof succeeds.
Within `loop.stage 0`, X and Y also use disjoint classes. Nevertheless, Y and Z
both acquire class 0 from different partitions. Checking each split phase lane
in isolation would miss that diagonal overlap. The semaphore-wide partition
proof rejects it.

The failing depth-three case has two fresh writes in each loop iteration. X
and Y acquire the same initially released semaphore `%S` in the same
`loop.stage`, but execute in different partitions:

```text
// Input pseudo-IR. Both stage operands are offset 0.
scf.for ... {
  %tx = nvws.semaphore.acquire %S[offset=0]
          {partition=P, loop.stage=0, loop.cluster=0}
  fresh write X through %tx

  %ty = nvws.semaphore.acquire %S[offset=0]
          {partition=Q, loop.stage=0, loop.cluster=1}
  fresh write Y through %ty
}
```

Here `D = 3`, `A = 2`, and `G = gcd(D,A) = 1`:

```text
acquire   partition   p   o   class   buffer stages by iteration
X             P       1   0     0             0, 2, 1, 0, ...
Y             Q       2   0     0             1, 0, 2, 1, ...
```

Both acquires eventually use every physical buffer stage. The logically correct
ownership and phase sequence for `%S[0]` is:

```text
owner              partition   wait phase
X(iteration 0)          P           1
Y(iteration 1)          Q           0
X(iteration 3)          P           1
```

Those phases are correct. The race is that the partitions need not execute in
that order. Partition `P` can run ahead as follows:

```text
partition P: acquire %S[0], phase 1   // X(iteration 0)
partition P: ...
partition P: acquire %S[0], phase 1   // X(iteration 3), may pass old phase 1
partition Q: acquire %S[0], phase 0   // Y(iteration 1), not completed in between
```

The phase-0 acquire exists and its phase is computed correctly, but it has not
completed between the two phase-1 acquires. Both X waits can therefore accept
the same earlier phase-1 completion. Cloning or replaying the phase arithmetic
in both partitions does not add the missing runtime ordering. Repair would
require a new runtime handoff; `AssignStagePhase` only proves legality and
rejects the overlap.

For comparison, with `D = 4` and the same two advances, `G = 2`: X owns
buffer stages `{0,2}` and Y owns `{1,3}`. Their classes are disjoint, so the
two partition-local acquire histories are legal.

This check applies to both unsplit and split circular phase state. For split
state, it complements the phase-split proof: the partition proof assigns each
physical slot class to one partition globally, while the phase-split proof
assigns the classes used inside that partition to disjoint `loop.stage` lanes.
Only after both proofs succeed may a shared split phase computation be cloned
as described below.

[↑ Back to contents](#contents)

### Cloning a split phase value into multiple partitions

Phase state is normally separated by partition, and phase splitting further
separates it by `loop.stage`. Cross-partition cloning is therefore performed
separately for each `loop.stage`. For circular semaphores, the
stage-disjointness and semaphore-wide partition-ownership proofs above first
establish that every physical slot has one partition-local phase history. When
acquires in partitions `P` and `Q` use the same semaphore and the same
already-split `loop.stage`,
`AssignStagePhase` shares that split phase value across `P` and `Q`. Both
partitions must compute the same phase-update history. Each acquire remains in
exactly one partition; only the split phase computation is cloned into both.

For example, suppose X and Y use the same split phase value while Z uses the
other split phase value:

```text
acquire X    partition=P    loop.stage=0
acquire Y    partition=Q    loop.stage=0
acquire Z    partition=P    loop.stage=1
```

After assigning phase updates but before partitioning the loop, the relevant
shape is:

```text
%xWord = update %phaseLS0       {partition=[P,Q], loop.stage=0}
%tx = acquire X[%xWord]         {partition=P,     loop.stage=0}
%yWord = update %xWord          {partition=[P,Q], loop.stage=0}
%ty = acquire Y[%yWord]         {partition=Q,     loop.stage=0}

%zWord = update %phaseLS1       {partition=P,     loop.stage=1}
%tz = acquire Z[%zWord]         {partition=P,     loop.stage=1}
```

`AssignStagePhase` puts `[P,Q]` on every update and loop-carried value in the
`loop.stage 0` phase chain. `PartitionLoops` then clones that complete chain
into both partitions:

```text
partition P:  %phaseLS0.P -> X update -> Y update
partition Q:  %phaseLS0.Q -> X update -> Y update
```

X consumes its local value in `P`; Y consumes its local value in `Q`. The
other updates are replayed so that both local copies reach the same final phase
state for the next loop iteration. The `loop.stage 1` chain is needed only by
P, so it remains annotated with `P` and is not cloned into Q.

This does not make X or Y a multi-partition acquire, and it does not clone the
semaphore or its backing buffers. It clones only the compiler SSA computation
for the already-split phase value.

[↑ Back to contents](#contents)

## Structured control flow metadata

Buffer-stage and phase state is threaded through only the `for` and `if`
regions that use it. A loop carries one canonical `state.stage` plus every
`state.phases[key]` used in its body as iter-args and results; schedule-local
stage replays are not carried. A buffer-stage result is stamped with all of the
semaphore group's partitions, and that partition set is extended when another
partition consumes its block argument; phase result partitions are inferred
from their final SSA values. After assignment, a WS-loop iter-arg is removed
only when it is trivially forwarded unchanged (`scf.yield` supplies that same
region iter-arg). The pass replaces that slot with its init value and rebuilds
the loop without the slot; it does not prove or remove other loop invariants.

[↑ Back to contents](#contents)

## Additional lowering invariants

- Every `semaphore.create` that reaches physical lowering must carry a
  `pending_count`. InsertSemas records it on the original semaphores. When
  `LowerSemaphore` combines semaphore pairs, the combiner derives and records
  it on the two replacement semaphores.
- `arrive_count` is mandatory on every `semaphore.release`. Values above one
  are lowerable only when every release kind is `none` or `wgmma`; TC5MMA,
  TMEM-copy, and TMA-load completion paths reject them.
- Eligible SMEM semaphore pairs are combined before multibuffering and
  `AssignStagePhase`; pairs that do not meet the combiner's structural,
  ownership, and dominance requirements remain separate.
- Eligible TMEM aliases are coalesced only after dominance and
  compatible-view checks.

[↑ Back to contents](#contents)

## Code map

- Buffer-stage and phase analysis:
  [`AssignStagePhase.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp),
  `AssignStagePhase::run`, `analyzeStageCopies`, `assignStateInBlock`, and
  `propagateStage`.
- Multi-`loop.stage` phase proof: the same file,
  `computeMultiStagePhaseLanes` and `proveStageDisjointSlotOwnership`.
- Cross-partition circular-slot proof: the same file,
  `validateCircularPartitionSlotOwnership` and
  `provePartitionDisjointSlotOwnership`.
- Barrier lowering:
  [`LowerSempahores.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSempahores.cpp),
  `NVWSLowerSemaphore::runOnOperation`, `combineSemaphores`,
  `createCombinedSemaphoreOps`, `rewriteAcquire`, `rewriteRelease`, and
  `rewriteBuffer`.

[↑ Back to contents](#contents)
