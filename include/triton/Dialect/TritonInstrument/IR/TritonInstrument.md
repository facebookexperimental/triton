# Triton Instrument Dialect and Concurrency Sanitizer (ConSan)

## Overview

ConSan instruments Triton IR to detect illegal concurrent accesses to shared
and Tensor Core memory under warp specialization. It tracks per-buffer
visibility across threads and CTAs, models barrier-based synchronization, and
models commit-count synchronization such as cp.async and wgmma.

ConSan currently supports one public entry point in a module. BufferRegion
analysis collects shared-memory buffers, tensor-memory buffers, and barrier
allocations. Auxiliary state is then created on demand for each
warp-specialization partition using distributed tensors and shared-cluster
global scratch memory.

## Thread Model

- Base threads: 16 warp-specialization threads, allowing up to 16 partitions.
- Peer classes: 16 TMA, 16 Tensor Core, and 16 CLC threads model operations
  that are not ordered with base threads.
- Total logical threads: 64.

Logical thread IDs are in `[0, 64)`. State dimensions are power-of-two padded
where required by the distributed layout.

## Auxiliary State

Shape notation:

- `C`: CTAs in the cluster.
- `Cbar`, `Cbuf`, `Cthr`, `Cmask`: CTA dimensions qualifying barriers,
  buffers, threads, and visibility masks. Each has extent `C`.
- `B`: tracked buffers for one memory type.
- `K`: tracked mbarriers.
- `T`: logical ConSan thread slots.
- `P`: base-thread commit columns.

`tensor` denotes a distributed tensor value. `scratch` denotes a pointer to
shared-cluster global scratch memory. Logical state shapes are:

- `buffers` (tensor, `<B x i64>`): packed buffer descriptors.
- `barriers` (tensor, `<K x i64>`): packed mbarrier descriptors.
- `barrierStates` (scratch, `<Cbar x K x i64>`): packed lifecycle state.
- `writeVisibility` (scratch, `<Cbuf x B x Cmask x i64>`): latest-write
  visibility masks.
- `writeTracking` (scratch, `<Cbuf x B x Cbar x K x i8>`): barriers
  tracking writes.
- `readVisibility` (scratch, `<Cbuf x B x Cthr x T x Cmask x i64>`):
  per-thread read frontiers.
- `readTracking` (scratch, `<Cbuf x B x Cbar x K x Cmask x i64>`):
  barriers tracking read visibility.
- `outstandingCommits` (scratch, `<C x B x P x i8>`): staged and
  outstanding cp.async or wgmma accesses.
- `aliasMatrices` (tensor, `<B x B x i1>`): optional intra-CTA buffer
  alias information.
- `waiting` (scratch, `<Cbar x K x Cthr x i32>`): waiting flags and phases.
- `activeMasks` (scratch, `<C x i32>`): active base-thread masks.
- `lock` (scratch pointer, `i32`): serializes instrumentation updates.

Buffer and barrier descriptors are CTA-agnostic. CTA-qualified scratch axes
identify the buffer row, barrier row, logical thread, or visibility mask to
which a fact belongs.

Scratch state is initialized once before the instrumented body. A CTA or
cluster barrier follows initialization before any instrumented operation uses
the state.

At a `ttg.warp_specialize`, the pass copies the default thread's read and write
visibility to the destination partition peer masks so partition-local execution
starts with the visibility frontier that existed before specialization.
The generic-to-async proxy frontier is copied separately to the destination
base-thread rows; it does not use the TMA, Tensor Core, or CLC peer slots.

## CTA Model

The single-CTA case is the degenerate multi-CTA model: every CTA-qualified axis
has one row and the ordinary visibility rules apply unchanged.

For multiple CTAs, each CTA has its own logical threads. A multicast-layout
barrier has one live barrier row per multicast group, owned by the group's lead
CTA. Every CTA in the group may arrive or expect on that row, but only the lead
CTA initializes, waits on, and invalidates it. Non-leader barrier rows are not
live.

An operation has three distinct CTA roles:

- The issuer predicate selects which CTA executes the operation.
- The memory-effect CTA bitset selects the buffer rows read or written.
- The barrier-recipient CTA bitset selects the live barrier rows updated by
  arrivals, expectations, and completion signals.

A multicast TMA load is issued by the group leader, writes every result CTA
row, and signals the leader barrier row. A two-CTA Tensor Core operation is
issued by the even CTA but affects both CTA rows. CLC try-cancel is issued once
for the cluster and affects all CTA rows.

## Visibility and Legality

A read is legal when the reading thread sees the latest write to the buffer. A
write is legal when the writing thread sees all prior writes and completed
reads. ConSan emits:

- `experimental_verify_write_visibility`: verifies that no unseen write is
  in flight.
- `experimental_verify_read_visibility`: verifies that the writer's
  read-visibility lane covers all prior reads.

Checks account for aliases and CTA recipients. A selected buffer row is
expanded through alias metadata when BufferRegion analysis finds overlapping
regions. Aliasing is intra-CTA: descriptors may alias within one CTA row, but
not across different CTA rows.

## Barrier Synchronization

ConSan separates tracking from visibility transfer. Memory operations update
read and write frontiers. An arrive or commit snapshots visible reads and
writes into a barrier's tracking state. A later wait transfers that state to
the waiting thread and its peer classes.

Ordinary mbarrier operations follow the live-row rule: participating CTAs
address the lead barrier row and only the lead CTA waits. TMA-style and CLC
cross-CTA writes become visible in the CTA rows reached by the memory effect;
read transfers update the current CTA row.

A non-relaxed cluster barrier directly publishes synchronous base-thread work
to all CTA rows for the generic proxy. Barrier waits use a locked pre-wait
section and a locked post-wait section so instrumentation state remains
consistent while the hardware wait executes outside the ConSan lock.

## Generic-to-Async Proxy Ordering

On NVIDIA targets, some instructions access shared memory through the generic
proxy and others through an async proxy. ConSan requires every generic-proxy
access that precedes an async-proxy access to cross
`ttng.fence_async_shared` before the async access is issued. This rule applies
to both reads and writes. In particular, ConSan intentionally requires a fence
for a generic read followed by an async read even though that pair alone is not
a data hazard. This is the conservative rule exposed at the Gluon level.

The proxy state is maintained per buffer, CTA, and base thread. Each frontier
records which source base threads have made generic accesses visible and which
of those source accesses have been covered by a proxy fence. A new generic
access marks its source as seen and invalidates older fence coverage for that
source and buffer. `ttng.fence_async_shared` covers the generic accesses
currently visible to the issuing base thread; it does not fence another logical
thread. A CTA-scoped fence covers current-CTA buffer rows, while a
cluster-scoped fence covers buffer rows across the cluster.

Synchronization transports the packed access-and-fence frontier in the same
places that ordinary read visibility is transported:

- A frontier-tracked mbarrier arrive copies the issuing base thread's current
  proxy frontier into the barrier tracking row. The local copy remains live, so
  an arrive does not make a later async access by the arriving thread legal.
- A successful mbarrier wait merges the selected barrier row into the waiting
  base thread's row. A fence before the wait cannot cover accesses learned only
  by that wait; a fence after the wait can.
- Barrier invalidation clears the barrier's proxy tracking row.
- A non-relaxed publishing cluster barrier publishes the proxy frontier across
  CTA and base-thread rows.
- Warp specialization copies the parent's proxy frontier into the new
  partition base-thread rows.

Before an async-proxy shared-memory effect, ConSan checks only the issuing base
thread's row, restricted to the effect-recipient CTA rows and expanded through
the shared-memory alias matrix. The access is legal when every visible generic
source bit has corresponding fence coverage. This current-thread check is what
allows a producer to fence before publishing through an mbarrier, or a consumer
to wait and then fence, without treating a fence in an unrelated thread as
sufficient.

When ConSan initializes an otherwise uninitialized shared allocation with a
poison pattern, it emits a CTA-scoped async-shared fence after the poison store
and its CTA barrier. The poison operations are added after ordinary ConSan
instrumentation, so this fence prevents the sanitizer's own generic stores from
introducing an unmodeled proxy hazard.

ConSan does not add a symmetric proxy check for async-to-generic accesses.
PTX completion mechanisms for the modeled async operations provide the
async-to-generic proxy ordering, and ConSan already requires the corresponding
explicit completion wait before a later conflicting generic access is legal.
For example, a TMA-load mbarrier wait or an async commit-count wait transfers
ordinary visibility to the waiting thread. By contrast, `cp.async` completion
followed by a TCGen or WGMMA shared-memory access still needs
`fence_async_shared`: `cp.async` is recorded as a generic-side access for this
check, and its wait does not clear the generic-to-async proxy frontier.

## Barrier Lifecycle and Deadlock Detection

`barrierStates` packs the phase in bit 0, the initial arrival count in bits
`[1..20]`, the current arrival count in bits `[21..40]`, and a signed
transaction count in bits `[41..61]`.

- `experimental_init_barrier_state` initializes phase and counts.
- `experimental_verify_barrier_arrive` checks arrival-count underflow and
  transaction-count range.
- `experimental_update_barrier_state` applies count deltas, flips the phase
  when both counts reach zero, and reloads the initial arrival count.

Deadlock detection records a waiting flag and phase for each base thread.
`experimental_check_all_active_waiting` filters waits by the current barrier
phase and reports a deadlock when every active thread is waiting.

## Commit-Count Synchronization

Some hardware operations synchronize through outstanding commit counts instead
of mbarriers:

- Stage marks the current thread's buffer entry as uncommitted.
- Commit turns staged entries into committed accesses and advances existing
  outstanding-group distances.
- cp.async wait clears completed writes and publishes write visibility.
- wgmma wait clears completed reads and publishes read visibility.

`experimental_check_outstanding_commits` verifies that a selected buffer and
its aliases have no incompatible pending accesses. Commit counters apply to
base-thread columns; TMA, Tensor Core, and CLC peers are represented through
the visibility model.

## Modeled Operations

The common hook implementation covers these TritonGPU operations:

- `ttg.async_copy_global_to_local`: shared-memory write tracked with
  `AsyncCp` commit counts and recorded as a generic-side proxy access.
- `ttg.async_commit_group`: commits staged `AsyncCp` accesses.
- `ttg.async_wait`: clears `AsyncCp` entries beyond the pending-count threshold
  and transfers write visibility.
- `ttg.local_store`: barrier-tracked shared-memory write.
- `ttg.local_alloc` with a source: barrier-tracked shared-memory write.

All three ordinary shared-memory effects above are generic-proxy accesses for
the proxy-ordering model.

NVIDIA hooks additionally cover:

- `ttng.init_barrier`, `ttng.wait_barrier`, and `ttng.inval_barrier` lifecycle
  arrive path for multicast barriers.
- `ttng.arrive_barrier`.
- TMA loads as barrier-tracked writes with tx-count decrement and precise
  effect-write tracking. Their shared-memory destinations are async-proxy
  effects.
- TMA stores as `TmaStore` commit-count reads, with `ttng.tma_store_wait`
  transferring read visibility. Their shared-memory sources are async-proxy
  effects.
- TMEM load, store, alloc-with-source, and copy operations.
- TCGen5 MMA, scaled MMA, commit, and TMEM copy operations as Tensor Core peer
  thread effects. Shared A/B/scale operands and the shared source of TMEM copy
  are async-proxy effects.
- CLC try-cancel as a CLC peer-thread write with EffectWrites barrier tracking,
  and CLC load-result as a barrier-tracked read. The try-cancel result write is
  an async-proxy effect.
- WGMMA operands in shared memory as async-proxy effects. Async WGMMA also uses
  `Wgmma` commit-count reads, with `ttng.warp_group_dot_wait` transferring read
  visibility.
- `ttng.async_shared_store` as a generic-side shared-memory access.
