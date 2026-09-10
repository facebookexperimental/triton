# NVIDIA Warp Barrier Efficiency

Apply this guidance to NVIDIA kernels that use TLX mbarriers. Treat barrier changes as synchronization protocol changes, not mechanical API substitutions. Preserve correctness first and promote only with stable end-to-end benchmark evidence.

## Build A Barrier Ledger

Before proposing a barrier optimization, inventory every relevant barrier and record:

- its role: full/data-ready, empty/reuse, completion, scheduling, or handoff;
- its signal source: explicit software arrival, TMA transaction, MMA completion, named barrier, remote arrival, or multicast;
- the producer, consumer, and owner of the protected storage;
- every task replica, warp, lane, predicate, arrive site, and wait site;
- the expected arrival count and phase progression;
- the final access that must complete before the protected storage is reused.

Do not change a barrier whose role, participants, ownership, or phase cannot be proven from the complete kernel. Include prologue, steady-state, tail, boundary, and persistent outer-loop paths in the ledger.

## Distinguish Arrival Protocols

`tlx.alloc_barriers` uses ordinary leader-based software arrival lowering and may synchronize participating threads before the selected leader arrives. `tlx.alloc_warp_barrier` makes every participating thread arrive independently. Its initialized arrival count is:

```text
num_warps * 32 * num_arrivals
```

`num_warps` is the number of warps in one arriving task replica. `num_arrivals` is the number of all-thread arrival events targeting the same barrier in one phase, including replicated consumer tasks when each replica arrives once. Derive both values from the ledger; do not copy an ordinary barrier's logical arrival count into a warp-barrier allocation.

Consult `.claude/skills/tlx-api-reference/SKILL.md` and nearby same-architecture kernels before using `tlx.alloc_warp_barrier`. Reuse a demonstrated topology rather than inventing an arrival protocol.

## Select Only Safe Candidates

Local barriers signaled by explicit software `tlx.barrier_arrive` calls are the primary candidates. Empty/reuse notifications are especially useful to evaluate when all consumer lanes release a shared-memory slot after their final read.

Do not automatically convert:

- TMA full/data-ready or transaction-completion barriers;
- MMA or tensor-core completion barriers;
- named scheduling barriers;
- remote, multicast, or cross-CTA barriers;
- predicated or divergent arrivals where any counted lane can skip an arrival.

A warp barrier is safe only when every counted lane performs exactly the declared arrivals in every phase. For persistent kernels, prove the same topology across initial phases, ring wraparound, logical-task transitions, and the final tail.

## Isolate And Measure The Change

For the first A/B candidate, change only the allocator. Preserve the barrier's wait and arrive sites, ownership, protected buffer, buffer indices, and phase calculations. Keep paired TMA full barriers ordinary when evaluating warp barriers for software empty/reuse notifications.

Warp barriers may remove leader-path synchronization, but they issue more per-thread mbarrier arrivals. They are not universally faster. Verify all protected correctness cases, then use stable end-to-end benchmark and profile evidence to decide whether the reduced synchronization cost outweighs the additional arrival instructions. Revert the candidate when the topology proof fails or the measured result is noisy, neutral, or slower.
