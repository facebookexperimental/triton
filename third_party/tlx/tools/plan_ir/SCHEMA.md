# PlanBundle schema 0.2

PlanBundle is a canonical JSON sidecar for final AMD Triton/TLX TTGIR. All
objects are serialized with sorted keys, and every independently comparable
layer has a SHA-256 hash.

## Top-level contract

| Field | Meaning |
|---|---|
| `schema_version` | Exact reader/writer contract, currently `0.2`; readers upgrade `0.1`. |
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
| `value_graph_fingerprint` | Semantic fingerprint of the native operation/value graph. |
| `layer_hashes` | Independent hashes for operation, dot, storage, sync, schedule, layout, value, and lineage layers. |
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

`exact` means normalized TTGIR hashes match. `semantic_match` means all eight
PlanBundle layer hashes match. A report can therefore distinguish harmless
text/debug differences from a dot decomposition, layout, staging,
synchronization, or final-order change.

Version 0.2 is M1.4a. It encodes stable native identities, loop init/backedge/
exit edges, branch yields, and lineage through AMD extract-slice, reshape,
transpose, in-thread-transpose, and layout conversion. It does not yet encode
complete liveness intervals, alias sets, asynchronous LDS lifetime, or mutation
lowering. Those belong to M1.4b-d and plan-application milestones.
