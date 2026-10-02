# Synchronization and Phase Audit

Create `sync_manifest.json` before implementing barriers. The unit of review is
a dependency or resource-reuse obligation, not a convenient barrier copied
from another kernel.

## Required manifest structure

```json
{
  "schema_version": "paper-joint-manual-cuda-sync-v2",
  "authoring_mode": "agent-assisted-manual-cuda",
  "target": {"gpu": "B200", "cuda_arch": "sm_100a"},
  "pipelined_ir": {
    "schema_version": "paper-joint-pipelined-ir-v3",
    "sha256": "<sha256>"
  },
  "status": "manual_completion_required",
  "channels": [],
  "dependencies": [],
  "additional_obligations": [],
  "ordering_edges": [],
  "phase_traces": [],
  "coverage": {
    "ir_edges": 0,
    "mapped_ir_edges": 0,
    "cross_warp_edges": 0,
    "mapped_cross_warp_edges": 0,
    "ir_instance_dependencies": 0,
    "mapped_instance_dependencies": 0,
    "async_completion_obligations": 0,
    "mapped_async_completion_obligations": 0,
    "reuse_obligations": 0,
    "mapped_reuse_obligations": 0
  },
  "review": {}
}
```

The dependency list covers every IR edge, using either reviewed same-context
program order or a named channel. `additional_obligations` covers each TMA,
including TMA stores, TCGEN05, and TMEM-publication completion derived from the
IR. An instruction wait must name a real IR consumer and dependency distance;
a terminal TMA store instead names an explicit kernel-exit completion wait.
`ordering_edges` records release-to-next-producer edges, and the final bundle
derives their required coverage from multi-slot memory allocations. The final audit
requires full channel records, two consecutive slot wraps per phase trace,
all pipeline regions, complete coverage counters, and an `approved` review.

## Per-channel contract

For each channel, record:

- channel and barrier IDs, allocation ID, address, scope, and slot count;
- source/destination IR nodes and dynamic iteration relation;
- the edge's `producer_warps` and `consumer_warps`, and their CTA ranks;
- payload allocation and slot expression;
- initialization actor and expected software-arrival count;
- TMA transaction bytes, if any, and which actor calls `expect_tx`;
- producer completion witness: software arrival, TMA completion, TCGEN05
  completion, fence plus arrival, or another documented mechanism;
- consumer wait primitive, expected phase/parity expression, and predicate;
- release witness protecting producer reuse;
- initial state, prologue transitions, steady-state recurrence, epilogue drain,
  and reinitialization policy;
- behavior for empty, partial, and predicated iterations;
- specification/wrapper revision and reviewer.

One hardware barrier may discharge several obligations only when every edge,
participant, predicate, payload lifetime, and phase transition is listed and
proved compatible.

## Coverage obligations

Map all of the following:

1. Every `cross_warp_dependencies[]` edge.
2. Edges whose `producer_warps` and `consumer_warps` are equal but whose
   producer is asynchronous, such as TMA or TCGEN05.
3. Buffer-release edges before SMEM/TMEM/ring-slot reuse.
4. Descriptor or async-proxy ordering fences.
5. Epilogue store completion before source reuse or kernel exit.
6. CTA/cluster rendezvous and remote-memory visibility.

Program order is sufficient only for synchronous operations in one execution
context with the required memory ordering. State that proof explicitly.

## Phase model

Audit mbarriers by phase, not by comparing raw arrive and wait counts.

For each barrier slot, write a recurrence table that includes:

- logical iteration and pipeline region;
- producer slot and expected current phase;
- each software arrival and expected transaction-byte contribution;
- the asynchronous event that completes the phase;
- consumer wait parity and the state after the wait;
- release arrival/completion and the phase expected on next reuse.

Expand enough iterations to cover at least two complete slot wraps, plus the
full prologue and epilogue. Check the zero-trip and last-partial-tile paths.
The initial phase must be derived, not selected to make the first test pass.

An mbarrier phase completes only after its expected arrivals and tracked
transactions complete. `expect_tx` changes transaction accounting; it does not
replace required participant arrivals. A wait returning proves completion for
the targeted phase, not correctness of the payload layout.

## Completion witnesses

Use the witness matching the producer:

- **TMA load:** transaction completion on the associated mbarrier publishes
  the destination; issuing the TMA instruction is not completion.
- **TCGEN05 MMA:** a documented TCGEN05 completion/commit mechanism may publish
  MMA completion or release operands as specified. Record exactly which event
  it witnesses.
- **Synchronous register/SMEM/TMEM store:** use required memory/proxy fences and
  a software arrival. Do not use a TCGEN05 commit without prior asynchronous
  TCGEN05 work as a generic signal.
- **TMA store:** use the documented store-wait/completion sequence before
  reusing its source or exiting.

If a selected primitive cannot witness the required event, change the sync
design; do not weaken the obligation.

## Participant and predicate audit

For every channel:

- enumerate all arriving and waiting warps/CTAs;
- derive counts from that enumeration;
- verify each conditional producer has a conditionally compatible consumer;
- verify early exits cannot strand a waiter or skip a required release;
- verify named-barrier thread counts match exact participants;
- verify cluster-scoped operations have cluster-scoped initialization and
  visibility;
- verify only the intended actor performs TMA byte expectation and issue.

A CTA-wide barrier may change overlap and does not automatically witness async
completion. Record it as a schedule deviation if it is not represented by the
reviewed mapping.

## Deadlock and reuse audit

Perform a manual phase-state audit before running:

1. Build a wait-for graph for each expanded iteration and slot.
2. Confirm every wait has a reachable completion event on the same phase.
3. Confirm no completion depends transitively on the waiter proceeding.
4. Confirm release happens after the last asynchronous read of a slot.
5. Confirm prologue does not wait for a producer iteration that does not exist.
6. Confirm epilogue drains outstanding MMAs, loads, stores, and releases.
7. Confirm every barrier is quiescent before reinitialization or kernel exit.

The structural auditor expands enough dynamic iterations for two full channel
wraps plus the maximum dependency/reuse distance. It includes reviewed IR
dependency waits, async-consumer waits, per-warp issue order, and
allocation-specific release-to-next-producer edges. This check supports, but
does not replace, the manual phase-state review and runtime synchronization
tests.

Then run scale tests and synchronization tooling described in
`correctness-performance.md`. A small launch passing does not validate phase
reuse; exercise multiple wraps and enough CTAs to expose latent deadlocks.

## Review result

The sync reviewer signs the exact manifest hash and records one of:

- `approved`: all obligations are covered without a schedule deviation;
- `approved-experimental`: safe but intentionally non-faithful, with deviations
  linked from the mapping manifest;
- `rejected`: missing witness, phase proof, predicate match, or drain.

Only `approved` supports a faithful manual-lowering claim.
