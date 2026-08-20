# Multi-CTA control-flow contract

## Summary

A multi-CTA operation is correct only when all CTAs in its participation group
execute the same dynamic operation instance.

The compiler must prove that participating CTAs:

- reach the operation the same number of times and in the same order;
- take the same enclosing control-flow decisions;
- agree on loop trip counts and the current dynamic iteration; and
- agree on whether the operation is enabled by a predicate.

The controlling SSA values do not need to be the same SSA definition. They must
be shared values or be provably equal for every CTA in the participation group.
If the compiler cannot prove this, it must not create the multi-CTA operation.

This is a control-flow contract, not an operand-equality contract. Some operands
are intentionally CTA-specific. For example, two CTAs in a 2-CTA MMA load
different halves of B. What must agree is which cooperative MMA instance those
halves belong to.

## Motivation: TMA multicast

TMA multicast replaces several per-CTA loads with one transaction that delivers
the same source tile to multiple CTAs' shared memory. One CTA issues the TMA
instruction, while all recipient CTAs participate in its completion and buffer
lifetime protocol.

This example starts after multicast opportunity analysis. Assume that:

- the compiler has already identified one descriptor load as a multicast
  candidate;
- descriptor and coordinate analysis has proved that the CTAs request the same
  source tile;
- the physical cluster shape is exactly `ctas_per_cga=(2, 1, 1)`;
- cluster ranks 0 and 1 are both recipients of the multicast; and
- rank 0 is selected as the leader that issues the TMA instruction.

Finding the shared load, proving tile identity, choosing the cluster shape, and
deciding whether multicast is profitable are outside this document's scope. The
question here is narrower: given that plan, can the compiler prove that ranks 0
and 1 execute a compatible control-flow sequence so the multicast is safe?

TMA multicast operates across CTAs within one hardware cluster, not across
distinct hardware clusters. In this example, "both CTAs" always means ranks 0
and 1 in the same `(2, 1, 1)` cluster.

Consider two CTAs that should share one multicast load:

```text
CTA 0                          CTA 1
-----                          -----
prepare local destination      prepare local destination
leader issues multicast        do not issue
wait for local completion      wait for local completion
consume local copy             consume local copy
```

If CTA 1 skips the load, executes it in a different loop iteration, or reaches a
different multicast load first, the transaction no longer has a well-defined
set of participants. Depending on the lowering, the result is stale data,
incorrect barrier phase accounting, or deadlock.

It is therefore not enough to prove that the descriptor and coordinates select
the same source tile. The compiler must separately prove that the recipient CTAs
execute the same dynamic load instance.

## Tile padding and conditional execution

Edge tiles do not necessarily require divergent control flow. A kernel can pad
the scheduled tile space so every CTA in a participation group remains active,
execute the same multi-CTA operations for a partially valid or dummy tile, and
discard invalid output elements. This preserves the control-flow contract at
the cost of some extra work.

Two kinds of padding are relevant:

- **Data padding** defines the values read or written outside the logical tensor
  bounds. TMA tensor descriptors currently support out-of-bounds fill, with
  zero as the default padding option. Kernels also use masked stores or
  descriptor-store bounds behavior to avoid writing invalid output elements.
- **Schedule padding** adds logical tile slots so the launch grid or persistent
  scheduler remains aligned to the multi-CTA participation group. A CTA assigned
  a padded slot still follows the same collective operation trace; it must not
  return early merely because its output tile is invalid.

### Current ownership

Schedule padding is currently the kernel author's responsibility, including the
host-side launch-grid calculation and any in-kernel tile scheduler. The launcher
passes the requested grid and `ctas_per_cga` shape to CUDA; it does not generally
round the grid or synthesize dummy tile work. The compiler also does not turn a
CTA-dependent early exit into a padded collective operation.

A follow-up design will explore how to make this contract explicit for the
grid-based, non-persistent approach, for example through launch metadata or an
IR-level guarantee that the grid and tile mapping are participation-group
aligned. Until then, this design assumes that the launch grid is sufficiently
padded before compilation or launch.

For a statically persistent schedule, the compiler should be able to infer the
required participation from the control-flow contract rather than relying on a
separate grid-padding assertion. Proving equal loop entry, dynamic iteration
sequence, and exit across the participation group establishes that all members
continue to reach the same multi-CTA operations. The compiler must still prove
that the statically defined tile mapping assigns compatible work to those
members.

The implementation does provide mechanisms that the kernel author can use for
data padding: tensor descriptors define the logical shape and out-of-bounds fill
behavior, and stores can suppress invalid output elements. The author still
chooses the padded grid or schedule, descriptor shape, and output guards.

Tensor-descriptor padding does not provide a whole-tile validity mask or decide
whether a tile-level operation should execute. Even if every element requested
by a TMA load is out of bounds and the descriptor returns padding values, the
load still represents an executed tile operation. When an entire logical tile
may be empty, the kernel must explicitly choose between executing padded dummy
work for the whole participation group and using a group-shared condition to
skip the operation. A descriptor cannot make it safe for only one CTA to skip
the collective operation.

### Non-persistent kernels

A non-persistent kernel normally maps each launched CTA to one logical tile. The
host-side grid calculation must round the tile grid to the physical cluster
shape. For `ctas_per_cga=(2, 1, 1)`, the X grid must contain complete CTA pairs,
so an odd logical X-tile count is rounded up to an even launch count.

The CTA assigned the padded tile still executes the same multicast or 2-CTA MMA
sequence as its partner. Its descriptor loads use out-of-bounds padding where
needed, and invalid output writes are suppressed. It must not return early just
because its logical output tile is outside the problem shape.

Current 2-CTA test kernels follow this model by launching at least one complete
pair, using descriptor padding for edge inputs, and guarding output writes. The
general rounding of arbitrary grid shapes remains the kernel or launch wrapper's
job.

### Persistent kernels

A persistent kernel launches a fixed set of resident CTAs and processes multiple
logical tiles in a loop. Padding only the launch grid is insufficient because
the in-kernel scheduler determines which CTAs execute each dynamic tile
instance.

The scheduler must assign and retire work at participation-group granularity.
For a 2-CTA group, both CTAs advance to compatible logical tiles together and
make the same continue or exit decision. If the final work assignment does not
fill the group, the scheduler must either issue padded work that keeps both CTAs
on the collective trace or make the whole group exit before the next multi-CTA
operation.

Static 2-CTA persistent matmul schedules rely on this kernel-level invariant:
paired CTAs stay on the same N tile and cover adjacent M tiles, and an odd M-tile
count is padded so the final pair does not cross into an incompatible N tile.
The compiler does not prove arbitrary static persistent tile schedulers are
pair-aligned.

For the canonical atomic dynamic scheduler, the NVIDIA backend now recognizes
the ordinary frontend loop before warp specialization. It proves a uniform
counter, canonical direct loop carry, and a cluster-uniform tile bound divisible
by physical cluster size `K`. It then linearizes the seed in cluster-major
X-fastest order. AutoWS assigns the run-once owner and clones the physical
partitions without special cluster-scheduler handling. A late materialization
pass then rewrites the marked owner claim so rank zero atomically reserves `K`
consecutive PIDs and distributes the base to the cluster; rank `r` executes PID
`base + r`.
Ready and reuse mbarriers make the full physical cluster the scheduler
protocol's participation group. Failure of the proof is a compile error before
this collective protocol is introduced.

This proof establishes scheduler alignment, not problem mapping. The kernel
author remains responsible for mapping each linear PID to problem coordinates,
padding the scheduled tile space, and initializing the counter to the aligned
number of launched physical CTAs.

Cluster-level schedulers such as multi-CTA CLC can satisfy the contract by
distributing one shared work assignment to the cluster and deriving each CTA's
tile from that assignment. Independently acquired per-CTA work does not provide
the required alignment. Likewise, a persistent loop over jagged data requires
the kernel to bucket or pad work so every CTA in the participation group observes
the same continuation sequence, or multicast must remain disabled for that
load.

Padding data alone is insufficient if the kernel still branches around the
entire operation. For example, zero-filled descriptor loads do not make this
safe when only one member executes the load:

```python
if tile_id < num_tiles:
    tile = desc.load(offsets)
```

For a multi-CTA operation, the guard must be shared by the participation group,
or the kernel must hoist the collective operation out of the guard and use
padding to make the extra work safe.

## Participation groups

The contract is scoped to the CTAs that cooperate on one operation, called its
**participation group**.

- A TMA multicast group is the set of CTA ranks in one recipient mask.
- A 2-CTA MMA group is the even/odd CTA pair that executes one
  `tcgen05.mma.cta_group::2` operation.
- An operation using a full-cluster barrier has the entire physical cluster as
  its participation group, even if its data recipients are a smaller subset.

The synchronization scope is a lowering decision, not an intrinsic requirement
that every multi-CTA operation use a full-cluster barrier. That decision has an
explicit control-flow implication: if lowering chooses a primitive whose
participants are wider than the data-sharing group, the compiler must prove
agreement across that wider group. A full-cluster rendezvous requires
full-cluster agreement, while a pair-scoped remote mbarrier requires agreement
only within the pair. For the `(2, 1, 1)` multicast example, both scopes contain
the same two CTAs.

## The contract

For a multi-CTA operation `O`, participation group `G`, and dynamic instance
number `i`, every CTA in `G` must agree on:

```text
(operation identity, instance i, enabled predicate, synchronization phase)
```

The following rules define that agreement.

### Whole-path reachability

The proof covers the complete control-flow path from kernel entry to the
multi-CTA operation, not only its immediately enclosing condition or loop. Every
decision that can change whether, when, or how often the operation is reached
must agree within the participation group.

For example, a CTA-dependent early return or program exit near the start of the
kernel is unsafe if another member of the group continues to a later multicast.
The later load may have no divergent local condition, but one required recipient
will never reach it. Guards around calls, loop exits, returns, and other earlier
branches must therefore be included in the reachability proof.

Participating CTAs do not necessarily need to follow the same textual path if
the compiler can prove that the paths reconverge without changing the ordered
trace of multi-CTA operations. The required property is equal dynamic
reachability, not syntactically identical control flow.

### Conditional control flow

For an operation nested under `scf.if`, the condition must have the same value
for every CTA in the participation group.

The condition does not need to have the same value across the entire physical
cluster. It must have the same value for every CTA within each participation
group; different groups may make different decisions. For example, a multicast
group that spans cluster Y at a fixed X coordinate may use an X-dependent
condition, so long as X is shared by the group. If the condition depends on Y,
which varies among the group's members, the compiler must still prove that the
evaluated condition is identical for every member. Otherwise the multicast is
not safe.

Syntactic dependence is a conservative approximation, not the semantic rule.
For a 2-CTA pair, `program_id(0)` differs, while `program_id(0) // 2` can be the
same for both CTAs. A sufficiently strong analysis may prove the latter equal.

### Loops

For an operation nested in a loop, participating CTAs must agree on:

- whether the loop is entered;
- the number of iterations;
- which logical iteration is executing; and
- any early-exit or continuation condition.

For `scf.for`, this normally requires equal lower bounds, upper bounds, and
steps, or a direct proof that they produce the same iteration sequence. For
`scf.while`, every CTA in the participation group must make the same continue or
exit decision on each logical iteration. In other words, they must exit on the
same iteration and execute the loop body the same number of times. Agreement on
the initial condition is insufficient if loop-carried state could make one CTA
exit earlier than another.

In practice, the expected `scf.while` use cases fall into three main categories:

1. **Scheduling loops, such as CLC.** The scheduler's continue or exit decision
   must be derived from one source shared by the participation group. For
   example, CTAs in a cluster may consume one cluster-level work assignment and
   derive the same "more work" condition from it. If each CTA independently
   acquires work or observes unrelated scheduler state, the compiler cannot
   assume that their loop iterations remain aligned.
2. **Fixed-iteration loops.** These are semantically equivalent to `scf.for`
   loops even if represented as `scf.while`. The compiler proves a shared
   initial iteration, bound, step, and resulting trip count.
3. **Loops over jagged data.** The data usage that controls continuation must be
   identical within the participation group. For example, CTAs sharing a
   multicast load must observe the same relevant sequence length or termination
   marker and therefore exit on the same iteration. Sharing the address of the
   control data is insufficient if the observed values can differ; the compiler
   must prove equality of the resulting continue or exit decisions.

These categories are not an exhaustive restriction on supported `scf.while`
loops. Other patterns are eligible when the compiler can prove the same dynamic
iteration sequence. However, until a pattern is modeled by the equality and
control-flow analyses, it may conservatively fail the proof and prevent TMA
multicast.

Loop-carried descriptor or index values are checked separately for tile
identity. Loop-carried values that affect the condition are checked for
control-flow agreement.

### Predication

A predicate attached to a multi-CTA operation is part of its control flow. All
participants must agree whether the cooperative operation occurs.

Leader predication introduced by lowering is different: the compiler may
select one CTA to issue the hardware instruction after it has proved that the
logical operation is enabled for the whole group.

### Calls

Kernel entry arguments are shared launch inputs. Arguments to a non-inlined
function are not necessarily shared: a caller can pass a CTA-dependent value.

For example:

```python
@triton.jit(noinline=True)
def helper(desc, offset):
    return desc.load([offset, 0])

@triton.jit
def kernel(desc, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK
    tile = helper(desc, offset)
```

`desc` is a shared kernel argument, but `offset` is computed independently by
each CTA. Inside the non-inlined helper, both values appear as function block
arguments. Treating every function block argument as shared would therefore
lose the `program_id(0)` dependency and could incorrectly form a multicast.

The IR identifies the kernel entry function by symbol visibility. Triton code
generation emits the launched kernel as `tt.func public` and callable helpers as
private functions. Compiler passes should use `tt::isKernel(FunctionOpInterface)`,
which implements this public-visibility check, rather than assuming that every
`tt.func` is a kernel.

A load or MMA in a callable helper therefore needs an interprocedural equality
proof. Until that exists, the compiler should require complete inlining or
reject multi-CTA formation inside non-inlined helpers.

### Unknown control flow

Control-flow analysis fails closed. If an operation is nested under a region or
branch construct whose execution semantics are not modeled, the compiler does
not assume convergence.

This includes unmodeled `scf` operations, CFG branches, custom region
operations, and control flow introduced by later compiler passes.

## What is allowed to differ

Control-flow agreement does not require lockstep equality of every value. The
following may differ when the operation's semantics permit it:

- CTA rank and local cluster coordinates;
- rank-derived leader predicates;
- destination shared-memory addresses local to each CTA;
- operands intentionally partitioned across CTAs; and
- ordinary computation between multi-CTA operations.

Divergent computation may reconverge before the next multi-CTA operation. The
required property is that every participation group observes the same ordered
trace of dynamic multi-CTA operations.

## Comparison with 2-CTA MMA today

The current 2-CTA MMA implementation already relies on this contract, although
it does not yet verify it explicitly. See
[Support 2-CTA in Triton with ctas_per_cga](2cta-autoWS-sync.md).

### Cooperative execution

Each even/odd CTA pair contributes to one `tcgen05.mma.cta_group::2` operation.
The CTAs load different B halves. Before the MMA, both CTAs must have reached the
same dynamic MMA instance.

In non-warp-specialized lowering, the backend uses cluster synchronization
around the 2-CTA operation. This assumes the required CTAs and threads reach the
same synchronization point.

In Meta AutoWS lowering, a full cluster barrier inside the consumer warp group
would deadlock because the other warps do not execute that region. The
`Insert2CTASync` pass instead inserts a pair-scoped "arrive remote, wait local"
protocol:

1. both CTAs arrive on the even CTA's shared-memory barrier;
2. the even CTA waits for both arrivals; and
3. the leader issues the `cta_group::2` MMA after both CTAs are ready.

The pass assigns separate barrier slots to separate MMA sites and derives the
barrier phase from enclosing `scf.for` induction variables.

`CheckMatmulTwoCTAs` also enforces the PTX requirement that all `tcgen05` MMA
operations in a kernel use one consistent `cta_group` mode. That kernel-wide
mode check is orthogonal to this contract: it does not prove that paired CTAs
execute those MMA operations in the same dynamic order.

### Current implicit assumption

`Insert2CTASync` inserts synchronization at every 2-CTA MMA, but it does not
prove that both CTAs:

- take the same branches around the MMA;
- execute equal loop trip counts;
- reach multiple MMA sites in the same order; or
- agree on the original MMA predicate.

Correct kernels satisfy these conditions by construction. If one CTA skips an
MMA or advances to a different phase, the inserted barrier protocol can wait
forever or pair arrivals from different dynamic instances.

The proposed contract makes this existing requirement explicit and gives the
compiler a common validation rule for 2-CTA MMA, TMA multicast, and future
multi-CTA operations.

## Compiler model

Each multi-CTA operation declares:

- its participation group;
- its logical enable predicate;
- the values that identify its dynamic instance or synchronization phase; and
- whether it is optional or required by program semantics.

A multi-CTA control-flow analysis walks outward through enclosing conditions,
loops, and calls. It proves that each controlling value is equal within the
participation group.

The initial analysis may be deliberately conservative:

- shared constants and kernel arguments are equal;
- a value independent of every axis that varies within the group is equal;
- modeled pure operations preserve the union of their dependencies;
- supported SCF bounds and conditions are checked explicitly; and
- unknown values or control-flow constructs fail the proof.

Later analyses may prove additional equalities such as pair IDs, modulo classes,
or equivalent expressions.

## Failure behavior

Failure to prove the contract must happen before collective synchronization is
introduced.

- Optional optimizations such as TMA multicast fall back to ordinary per-CTA
  operations.
- Required multi-CTA operations such as an explicitly requested 2-CTA MMA must
  either use a semantics-preserving 1-CTA fallback or report a compile-time
  diagnostic.

Once barriers or leader-only issue have been introduced, dropping only the
multi-CTA instruction is not a valid fallback. The entire collective protocol
must be removed together.

## Warp specialization

All control-flow rules in this document apply unchanged to warp-specialized
kernels. Participating CTAs must still agree on whole-path reachability,
conditions, loop iterations, exit decisions, operation predicates, and the
ordered dynamic trace of multi-CTA operations. Moving an operation into a
specialized warp group does not relax the CTA-level contract.

Warp specialization does, however, add value-flow and execution structures that
the equality analysis must understand. In particular, it must resolve values
captured by or passed into warp-specialized regions, values forwarded between
partitions, and arguments to specialized warp-group functions. It must also
relate conditions evaluated inside those regions or functions back to the CTAs
in the multi-CTA participation group.

This document does not define that warp-specialized value-flow analysis. The IR
contract for passed values, partition boundaries, and specialized warp-group
functions will be specified in a follow-up design. Until that work is complete,
a multi-CTA operation whose control-flow proof crosses an unsupported
warp-specialization boundary must conservatively fail the proof.

The lowering must use synchronization whose participants match the executing
warps. The 2-CTA MMA implementation demonstrates this distinction: a full
cluster barrier is usable on the non-WS path, while the WS path uses a remote
mbarrier because only the consumer partition reaches the MMA.

After warp-specialization, pipelining, or code-partition transforms, the compiler
must revalidate or preserve the proof that each CTA group still observes the
same multi-CTA operation trace.
