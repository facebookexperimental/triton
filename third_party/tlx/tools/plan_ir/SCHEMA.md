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
  in-loop consumers, positive buffer depth, and power-of-two alignment.

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
`global_to_lds` or `register_to_lds` staging remains M1.5b.4.
