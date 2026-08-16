# PlanBundle schema 0.3

PlanBundle is a canonical JSON sidecar for final AMD Triton/TLX TTGIR. All
objects are serialized with sorted keys, and every independently comparable
layer has a SHA-256 hash.

## Top-level contract

| Field | Meaning |
|---|---|
| `schema_version` | Exact reader/writer contract, currently `0.3`; readers upgrade `0.1` and `0.2`. |
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

Version 0.3 adds M1.4b static liveness and logical LDS modeling. Intervals are
block-local TTGIR program order, not cycles or physical register allocation.
LDS sizes and offsets are logical, not the allocator's physical placement.
`pending_async` identifies operations that initiate asynchronous work, but this
version deliberately does not extend lifetimes through commit/wait/barrier.
That dynamic async model and mutation lowering belong to M1.4c-d and later
plan-application milestones.
