# PlanBundle schema 0.5

PlanBundle is a canonical JSON sidecar for final AMD Triton/TLX TTGIR. All
objects are serialized with sorted keys, and every independently comparable
layer has a SHA-256 hash.

## Top-level contract

| Field | Meaning |
|---|---|
| `schema_version` | Exact reader/writer contract, currently `0.5`; readers upgrade `0.1` through `0.4`. |
| `kernel` | TTGIR function symbol. |
| `case` | Shape, dtype, causal, and MHA/GQA contract from the baseline manifest. |
| `provenance` | Source/compiler revisions, schedule configuration, and captured artifact references. |
| `operations` | All parsed TTGIR operations in final order. |
| `dot_fragments` | Ordinary dots and statically materialized scheduled-MFMA sites. |
| `storage` | LDS allocation/deallocation sites and their memdesc types. |
| `synchronization` | Async commit/wait, barriers, and MFMA commit/wait sites. |
| `schedule` | Stable operation IDs in final textual order. |
| `layouts` | Expanded TTGIR layout aliases, including AMD MFMA layouts. |
| `normalized_ir_hash` | Hash after removing debug locations and alpha-renaming SSA values. |
| `values` | Native final-structured-TTGIR values, types, origins, uses, and identity quality. |
| `lineage_edges` | Derived-value and structured-loop edges, including dynamic iteration distance. |
| `blocks` | Structured blocks and their direct operation order; positions are local to each block. |
| `live_segments` | Half-open static TTGIR program-order intervals with live-in/live-out and loop-distance metadata. |
| `lds_aliases` | Logical LDS roots/views, static offsets/order, and normalized slot paths. |
| `memory_accesses` | LDS read/write/allocate/free effects, including whether an operation starts pending async work. |
| `lds_allocations` | Per-root logical size, aliases, and the block-local union of their static intervals. |
| `async_transactions` | Async LDS reads/writes with commit, completion, visibility, consumption, release, and overwrite frontiers. |
| `async_groups` | Static commit groups and their transactions. |
| `async_waits` | Partial-wait retained count, completed groups with iteration distance, and possibly outstanding groups. |
| `lds_reuse_hazards` | Missing commit, wait, visibility, or consumer-release relationships that prevent proving safe reuse. |
| `dependency_edges` | SSA, loop-carried, RAW/WAR/WAW, async, barrier, consumer-release, and slot-reuse dependencies with iteration distance and precision. |
| `peak_live_sets` | Per-block maximum static overlap of logical tensor values and logical LDS roots. |
| `resource_summary` | Logical overlap, slot depth, async depth, and TTGIR operation-class counts; physical resource fields remain null. |
| `unresolved_facts` | Typed open or accepted facts with importance and stable operation/value references. |
| `value_graph_fingerprint` | Semantic fingerprint of the native operation/value graph. |
| `layer_hashes` | Independent hashes for operation, dot, storage, sync, schedule, layout, value, lineage, liveness, and LDS layers. |
| `diagnostics` | Unresolved semantic information; never silently discarded. |

## Identities

Without a native sidecar, operation IDs hash the operation kind, structured
scope, and location-free operation text after replacing SSA names, plus a
deterministic occurrence number. With `--value-graph`, PlanBundle adopts the
native compiler IDs. Those IDs refine operation/result/operand/use signatures
to a fixed point, exclude debug locations and placement-only attributes, and
record `fallback_ordinal` when two sites remain structurally symmetric.

Dot-fragment IDs additionally include:

- semantic output role (`qk`, `dp`, `dq`, `dk`, or `dv`);
- static role ordinal;
- operand/result tensor shapes and accumulator slice offsets;
- expanded MMA result layout;
- scheduled-MFMA resident operand, accumulator lifetime, register class, and
  initialization contract.

This gives decomposed dots a first-class identity. A future plan may move the
static fragment without confusing it with a different output tile.

## Replay levels

`exact` means normalized TTGIR hashes match. `semantic_match` means all
PlanBundle layer hashes match. A report can therefore distinguish harmless
text/debug differences from a dot decomposition, layout, staging,
synchronization, or final-order change.

Version 0.4 adds M1.4c asynchronous LDS lifetime modeling for AMD commit-count
operations and CTA barriers. A transaction may have multiple completion
frontiers: for example, a steady-state loop wait and a conservative loop-exit
wait. `iteration_distance` is a structured-loop distance, not latency.

The five phases are pending write, completed-but-not-visible, readable,
awaiting consumer release, and reusable/overwritten. An async wait completes a
producer-to-consumer RAW edge; it does not release a consumer-to-next-producer
WAR edge. Reuse therefore requires a release barrier after consumption.

TDM/mbarrier epochs, physical LDS placement, physical VGPR allocation, backend
instruction order, and cycle timing remain outside this schema and are never
inferred from TTGIR.

Version 0.5 adds M1.4d. Dependency distance is a structured-loop iteration
distance, not latency. Peak tensor bytes are whole distributed logical tensor
bytes, not per-wave VGPR bytes. Peak LDS bytes use logical allocation roots and
do not imply assigned offsets or allocator overlap. `tlx_plan audit` rejects
error diagnostics, open important facts, and LDS reuse hazards while allowing
explicitly classified deterministic identity fallbacks.

## M1.5a schedule delta

`plan-schedule-delta/0.1` is a separate mutation request rather than a
PlanBundle layer. It names the kernel, input value-graph fingerprint, final
structured-TTGIR pass position, and one or more changed blocks. Each block
contains its complete baseline operation order and complete desired
permutation. The compiler rejects kernel, fingerprint, block, baseline-order,
or permutation mismatches before moving an operation.

M1.5a only changes order within an existing block and dynamic iteration. It
does not change storage, synchronization, loop placement, staging depth, or
dot decomposition. Positive dependency distance is structured-loop metadata,
not permission to move an operation into another iteration.

## M1.5b pipeline delta

`plan-pipeline-delta/0.1` is a separate intent contract for changes that cross
dynamic loop iterations or change iteration-scoped storage. It is pinned to a
kernel, input value-graph fingerprint, and the compiler position immediately
before async wait-count adjustment. An empty `loops` list is the identity
delta.

Each non-empty loop entry names a stable `scf.for` or `scf.while` operation and
one or both of:

- a complete async group with `set_prefetch_distance`, positive iteration
  distance, and buffer depth at least that distance;
- a logical tensor value with `global_to_lds` or `register_to_lds`, explicit
  in-loop consumers, non-negative iteration distance, positive buffer depth,
  and power-of-two alignment. Register staging is currently distance zero and
  single-slot. Global staging is either distance-zero/single-slot or buffered
  with positive distance and buffer depth strictly greater than distance.

Dry-run validation rejects stale fingerprints, unknown or non-loop targets,
groups committed or produced outside the selected loop, incomplete groups,
unresolved LDS slot paths, non-tensor staging values, and consumers that do not
actually use the staged value. The validation report distinguishes iteration
placement, storage, synchronization, and dot-decomposition effects. Dot
decomposition is always frozen in M1.5b.

This schema does not encode resolved cycles, physical LDS offsets, inserted
waits/barriers, a modulo schedule, or prologue/steady-state/epilogue TTGIR.
Those are outputs of M1.5b.2--b.5, not claims made by an M1.5b.1 request.

### M1.5b.2 native subset

The first native materializer consumes the same `plan-pipeline-delta/0.1`
schema immediately before async wait-count adjustment, but accepts only
existing-LDS transaction entries. It requires:

- the exact input value-graph fingerprint and pass position;
- complete positive-distance wait families;
- consumer-frontier distance equal to the requested distance;
- every resolved modulo slot depth equal to the requested buffer depth; and
- no `staging` entries.

It projects the verified Plan dependency graph into the selected structured
loop, preserves baseline ordering for operations outside the selected producer
slices, adds loop-carried release-to-overwrite constraints, and invokes Meta's
shared modulo scheduler. A projected distance-zero dependency that contradicts
the already-valid baseline order is not imported and is counted in
`skipped_inconsistent_dependencies`; this handles conservative inner-region
frontiers collapsing onto an outer-loop operation.

`plan-pipeline-apply-report/0.1` records acceptance, resolved groups, modulo II,
selected/moved operation counts, imported/skipped dependencies, and stable
pre/post fingerprints. All `changes_*` fields remain false. Materializing new
LDS allocations, waits/barriers, prefetch distances, or buffer depths is not
part of M1.5b.2.

### M1.5b.3 existing-ring materialization

The native materializer may change `distance` and `buffer_depth` for an
existing canonical LDS ring. All in-loop readers and writers of the allocation
must belong to the selected complete transaction/wait family. The materializer
resizes the allocation's leading ring dimension, rewrites producer and consumer
`memdesc_index` expressions to the requested modulo depth and distance, derives
retained counts while preserving unrelated partial-wait groups, and inserts
missing local visibility and consumer-release barriers.

Before mutation it rejects unknown allocation sizes, inconsistent requests for
one root, unsupported indirect/nested views, and target LDS-capacity overflow.
After mutation it rebuilds Plan IR and the physical DDG. Acceptance requires
the requested slot depth and consumer distance, completion/visibility/release
and overwrite frontiers, no open important fact, no LDS reuse hazard, a legal
second modulo schedule, and an unchanged dot/scheduled-MFMA contract. New
staging remains M1.5b.4.

### M1.5b.4a single-slot register staging

The original 4a subset accepts a staging-only loop with `register_to_lds`,
`buffer_depth: 1`, a power-of-two alignment, and one or more stable consumer
operation IDs. The staged value must be a produced ranked tensor with known
logical bytes. Its producer, every direct use, and every named consumer must be
direct children of the selected `scf.for`; all direct uses must be selected.

The materializer allocates one mutable shared-memory object outside the loop,
stores the original register value, inserts a local visibility barrier, reloads
the value in its exact original register layout, rewrites the named consumer
operands, inserts a consumer-release barrier, and deallocates after the loop.
It rejects target-capacity overflow, loop-carried values, and multi-slot
register staging. Derived uses are added by 4b below; nested uses remain
unsupported. M1.5b.4e permits this request alongside an independent complete
existing-ring transaction family.

After mutation it requires strict Plan IR verification, no open important fact
or LDS reuse hazard, exact requested alignment and logical bytes, a legal
rebuilt distance-zero DDG order, and an unchanged dot/scheduled-MFMA signature.

### M1.5b.4b derived register staging

Named consumer operands may be reached from the staged source through a DAG of
pure direct-loop operations: `amdg.extract_slice`, `tt.reshape`, `tt.trans`,
`ttg.convert_layout`, `amdg.in_thread_transpose`, `tlx.require_layout`, and
`tlx.release_layout`. Unsupported, nested, side-effecting, or ambiguous
derived paths are rejected.

The materializer inserts the same single-slot synchronous LDS path as 4a,
reloads the exact source register type, clones the union of selected derived
paths in original program order, maps shared prefixes once, rewrites only the
named consumer operands, and erases original pure derived operations that
become dead. Unselected branches keep their original SSA operands. The apply
report records cloned/pruned operation counts, selected operands, preserved
unselected consumers, and source live-range endpoints. Acceptance requires the
post-rewrite Plan IR interval length to be strictly smaller than the baseline;
otherwise it rejects with `staging_does_not_shorten_lifetime`.

### M1.5b.4c same-iteration global staging

The native reader also accepts `global_to_lds` with `buffer_depth: 1` when the
staged tensor is produced directly by `tt.load` or `amdg.buffer_load` in the
selected loop. Every use reachable through the supported 4b derived DAG must
terminate at a named consumer; partial-use plans are rejected so the rewrite
cannot duplicate global memory traffic. Volatile and unsupported load forms
are rejected.

The materializer replaces the register-producing load with
`ttg.async_copy_global_to_local` or `amdg.buffer_load_to_local`, preserving
pointer/offset, mask, `other`, stride, cache, eviction, and contiguity semantics
represented by the source and destination ops. It inserts one commit, wait,
visibility barrier, exact-layout local load, and consumer-release barrier,
then removes the original load and dead derived register path.

Post-rewrite acceptance requires one new LDS allocation, async transaction,
group, and wait; a proven completion/visibility/consumer/release chain; no old
source operation/value identity; unchanged dot/scheduled-MFMA contracts; and a
legal rebuilt distance-zero DDG. M1.5b.4e composes this with independent
existing-ring intents.

### M1.5b.4d buffered cross-iteration global staging

`global_to_lds` may additionally request positive `distance` and
`buffer_depth > distance`. The same complete-use, non-volatile
`tt.load`/`amdg.buffer_load`, access-semantics, derived-path, alignment, and
capacity requirements from 4c still apply. `register_to_lds` remains
distance-zero and single-slot.

The materializer adds a loop-carried i32 ring counter initialized to `-1`,
increments it modulo the requested depth, allocates a leading-dimension LDS
ring, and indexes the direct-to-LDS copy and local load through the resulting
single-buffer view. It serializes a `CoarseSchedule` with the copy/commit
backward slice in stage zero, the wait first in the requested consumer stage,
and the local load, derived path, consumers, and release barrier in that
consumer stage. The shared AMD pipeline expander then emits the prologue,
steady-state loop, and peeled epilogue. Distance one/depth two and distance
two/depth three are covered; depth must be greater than distance so a producer
cannot overwrite the slot still being consumed.

Acceptance is two-phase. Before expansion, strict Plan IR proves the new
allocation, modulo slot depth, direct-to-LDS transaction, access semantics,
completion/visibility/consumer/release chain, source-load elimination, LDS
capacity, and unchanged dot contract. After expansion, schedule markers must
be gone, MLIR and strict Plan IR must verify, no important fact or LDS hazard
may remain, overwrite distance must equal the requested depth, and the
producer-view to consumer-view SSA path must cross exactly the requested
number of structured-loop backedges. The apply report records `distance`,
`buffer_depth`, and `pipeline_expanded`.

### M1.5b.4e mixed-plan composition

One loop entry may contain both `transactions` and `staging`. Each family is
resolved against the same baseline graph and retains its earlier complete-use,
complete-wait-family, direct-view, alignment, and depth/distance requirements.
The implementation requires unique producer/derived paths for each new staging
family; a consumer may be shared only when every family assigns it the same
distance. It rejects any operation shared by an existing-ring and staging
family and any direct SSA dependency in either direction between those
families. This conservative contract prevents the unified scheduler from
silently changing an existing ring's producer/consumer relationship.

Capacity is computed once from baseline logical LDS bytes, every resized
existing-ring delta, and every new staging allocation multiplied by its buffer
depth. Synchronous mixed plans materialize existing-ring changes followed by
new staging and rebuild one DDG; acceptance verifies all emitted distance-zero
edges plus the existing Plan IR ring and staging contracts.

If any new staging entry is buffered, the pass builds one schedule with
`max(staging distance) + 1` stages. For each buffered family, the direct-to-LDS
copy/commit backward slice is assigned stage zero and its wait, visibility
barrier, local load, cloned derived path, named consumers, and release barrier
are assigned the requested consumer stage. Existing-ring operations and all
unassigned operations remain in the last logical stage. Dependencies are
completed once and the shared AMD expander is invoked once, so staging entries
with different distances share one prologue, steady state, and epilogue.

After materialization the existing structure-count, global-access, staging,
ring-mutation, DDG, and dot-contract audits still apply. After expansion, the
pass removes schedule markers, coalesces only adjacent barriers with identical
address-space semantics, and rebuilds strict Plan IR. Final acceptance requires
no LDS reuse hazard or open important fact and exact depth, structured SSA
consumer distance, visibility, release, and overwrite distance for every
expanded existing-ring and new-staging allocation. Async waits are deliberately
left separate so unrelated retained groups are preserved.

Focused lit coverage includes existing ring plus register staging, existing
ring plus same-iteration global staging, existing ring plus buffered global
staging, two staging distances in one expansion, combined-capacity overflow,
and overlapping/cross-dependent family rejection.
