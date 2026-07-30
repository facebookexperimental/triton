# B200 Primitive Selection

Read this after the instruction mapping exists and before writing inline PTX,
CUDA intrinsics, or wrapper calls. Public CUDA/PTX specifications and reviewed
local leaf wrappers are the source of truth for exact operand constraints.
Never infer a primitive protocol from FA4/FA3.

## Primitive selection record

For each lowered IR node, record in `mapping_manifest.json`:

- exact PTX/SASS family and shape, data types, transpose/layout modes, CTA
  group, scope, and required architecture;
- source anchor or leaf-wrapper symbol;
- operand storage and layout IDs from `memory_plan.json`;
- barriers or fences from `sync_manifest.json`;
- predication, dynamic bounds, and which warp/lane issues the instruction;
- the specification or reviewed-wrapper revision used to justify it.

Do not let a wrapper choose a warp role, schedule position, stage index, ring
slot, phase, or barrier ID. Those are visible manual-lowering decisions.

## TMA

For each TMA operation:

1. Record descriptor rank, element type, global shapes/strides, box shape,
   swizzle/interleave, address-space scope, and alignment.
2. Map the destination SMEM allocation and slot explicitly.
3. Set the transaction byte expectation before issuing the transfer and prove
   that the expected bytes match the actual transaction for every predicate.
4. Treat TMA completion as asynchronous. The completion mechanism, not the
   issuing warp's program counter, publishes the destination.
5. Do not issue a tensor tile whose starting coordinate is wholly outside the
   descriptor shape. TMA does not make an entirely out-of-range tile safe by
   silently masking it; fix tile mapping or launcher bounds.
6. Record any descriptor or generic-to-async proxy fence required before use.
7. For multicast or cluster operations, record CTA ranks, recipient mask,
   barrier scope, and which CTA performs transaction-byte accounting.

TMA store completion and TMA load completion are distinct obligations. Record
the wait required before source reuse or kernel completion for each store.

## TCGEN05 and TMEM

For each TCGEN05 MMA:

1. Record the exact MMA kind, M/N/K shape, operand types, accumulator type,
   CTA-group mode, descriptor/layout interpretation, and accumulate/overwrite
   mode.
2. Map each SMEM/TMEM operand to a reviewed memory-plan entry. Prove alignment,
   tile shape, and TMEM column allocation constraints.
3. Preserve the Twill per-group issue order. Do not reorder MMAs to imitate a
   baseline or to improve local aesthetics.
4. Treat MMA completion and operand-release completion as explicit asynchronous
   events. Identify which documented mechanism witnesses each event.
5. Use a TCGEN05 commit/completion operation only when it witnesses prior
   asynchronous TCGEN05 work. It is not a generic software signal.
6. A plain synchronous TMEM store followed by publication requires the
   documented software-arrival/fence sequence, not a TCGEN05 commit with no
   prior MMA to complete.
7. Keep MMA completion barriers distinct from barriers consumed by an MMA when
   the primitive contract requires distinct objects.

Disassembly must show the intended TCGEN05 family and shapes. A successful
compile that substituted a different operation is a mapping failure until
reviewed and reflected in the manifests.

## mbarrier

Model an mbarrier as a reusable phase state machine, not a raw counter:

- Initialization defines the expected software arrivals for the first phase.
- `expect_tx` adds asynchronous transaction bytes to the current phase; it is
  not a substitute for participant accounting.
- A phase completes only when required arrivals and tracked asynchronous work
  complete.
- A parity wait targets the phase expected by the consumer. Record the initial
  parity and the recurrence for every ring slot.
- Barrier reuse is valid only after every consumer has completed the previous
  wait and every producer has stopped adding work to that phase.

For every use, record address, scope, initialization count, transaction bytes,
arrivers, waiters, predicates, phase expression, and reuse owner. Never choose
counts by copying another kernel.

## Fences, scopes, and named barriers

- Distinguish ordinary memory ordering from async-proxy ordering. Record the
  producer write, the proxy that observes it, the fence, and its scope.
- A CTA fence cannot satisfy a cluster-scoped dependency. Remote CTA access
  requires an explicitly reviewed cluster contract.
- Named barriers synchronize a fixed participant set but do not by themselves
  witness TMA or TCGEN05 asynchronous completion. Record the exact participating
  warps/threads and the memory/completion event they guard.
- Do not use a CTA-wide synchronization to paper over an unproved cross-warp
  dependency; it may alter the schedule and still fail to witness async work.

## Leaf-wrapper exception

A reusable leaf wrapper is permitted only when all of these are true:

- it wraps one documented TMA, TCGEN05, fence, or mbarrier operation;
- callers provide operands, scope, predicate, barrier/phase, and schedule
  position;
- it has no loop, ring index, warp-role dispatch, phase recurrence, or hidden
  companion barrier;
- its generated instruction and clobber/order contract have been reviewed on
  B200;
- its provenance is listed in `manual_cuda_authoring.json`.

Composite FA helpers and wrappers that encode producer/consumer protocols are
outside this exception, even if they are convenient or already tuned.

## Primitive audit checklist

- Every scheduled node has exactly one semantic lowering or a reviewed
  expansion whose sub-operations are all listed.
- All asynchronous operations have completion witnesses in the sync manifest.
- All operands refer to memory-plan IDs; no implicit buffer or layout remains.
- All scopes and participant sets match the physical group mapping.
- Predicated-away producers cannot leave unconditional waiters.
- Prologue, steady state, and epilogue use the same documented phase model.
- Source and disassembly anchors are linked back to the mapping manifest.
