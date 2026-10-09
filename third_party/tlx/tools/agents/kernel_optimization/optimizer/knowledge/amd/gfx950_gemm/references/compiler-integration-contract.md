# gfx950 Compiler Integration Contract

A generated kernel is identified by source, workload, resolved compiler facts,
and framework path. If a framework drops an option or omits it from the
compilation cache key, an autotuning sweep can report different configurations
while executing one cached binary.

## Preserve Every Lowering-Affecting Fact

Examples include:

- matrix instruction and K width, wave policy, `waves_per_eu`, and packing;
- tensor, MFMA, shared-memory, and offset layouts;
- alignment, contiguity, pointer-range, and unaligned-load facts;
- direct-to-LDS eligibility and destination-view structure;
- intra-wave scheduling regions and post-RA scheduling policy;
- accumulator/register-class form and hazard guarantees; and
- stage count, persistence, and other options that change generated code.

An environment variable may provide an experimental default, but an explicit
per-configuration value should win. Store the resolved value in the backend
options object and include it in the cache identity.

## JIT And Framework Templates Are Different Paths

A direct JIT wrapper may infer specialization facts that a framework-generated
AST does not preserve. Do not use a fast JIT run as proof that the production
provider compiled the same kernel.

Compare across the boundary:

```text
resolved backend options and cache key
TTGIR operations, layouts, views, and encodings
MFMA opcode and accumulator form
global vector width and direct-to-LDS realization
intra-wave scheduling regions and realized machine groups
VGPR, AGPR, LDS, scratch, and occupancy
public dispatch, output/workspace ownership, and runtime result
```

When a physical backing carries authored offsets, use a semantic layout
requirement to preserve that exact layout through transformations; a weaker
hint that merely discourages rewriting may not prove the lowering contract.
Views such as static index/subslice/reinterpret operations are safe for
direct-to-LDS traversal only when they do not reorder the destination and the
lowering validates the chain back to the allocation.

## Fail Closed At Provider Boundaries

Offer a specialized provider only when dtype, rank, shape, strides, transpose,
address range, alignment, output, and compiler-option contracts are statically
proven. Unsupported workloads must remain on a valid general path.

For source consolidation, API migration, or rebase, compare dispatch manifests
and final objects before and after. Matching source helpers are insufficient if
a fallback rule, default option, or cache identity changed.
