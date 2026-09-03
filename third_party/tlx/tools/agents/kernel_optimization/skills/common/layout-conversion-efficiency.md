# TLX Layout Conversion Efficiency

Apply this guidance to Triton and TLX kernels on every target. Layout changes can be free metadata rewrites, register permutations, lane shuffles, or shared-memory round trips. Optimize only conversions whose material cost is demonstrated in the compiled pipeline.

## Inspect The Final Layout Flow

Do not infer cost from source-level `require_layout`, `release_layout`, transpose, reshape, or conversion counts. Compare early TTGIR with TTGIR after layout propagation and the final remove-layout-conversions pass. A conversion that disappears before lowering is not a runtime cost.

For each conversion that survives final TTGIR, record its shape, source and destination encodings, execution frequency, and producer-consumer path. Classify the expected lowering:

- equivalent layouts or metadata-only views: no physical movement;
- within-thread register reorder: usually low cost;
- cross-lane permutation or shuffle: potentially material;
- cross-warp, cross-block, or shared-memory redistribution: high priority because it may require stores, synchronization, and reloads.

Confirm the classification in lowered IR or generated code when possible. Do not estimate shared-memory capacity or prune configurations for scratch that the final compiler pipeline does not allocate.

## Trace Layout Anchors

Find why both sides require different layouts before changing either side. Common anchors include:

- global or descriptor loads and stores with coalescing or TMA geometry requirements;
- `local_load`, `local_store`, and explicitly laid-out local allocations;
- dot operands, accumulators, reductions, and epilogues;
- TMEM loads, stores, and subslices;
- warp-specialization captures and region-carried values;
- user-pinned `require_layout` boundaries.

Trace through elementwise operations, casts, broadcast, expand-dims, reshape, transpose, join, split, and loop-carried values. Prefer changing a flexible chain over forcing a fixed hardware-facing anchor into an incompatible layout.

## Choose Layouts At The Data Boundary

When a consumer needs a transposed or specialized layout, prefer expressing it at the memory-descriptor or load boundary rather than materializing data and converting it later. Consider:

- a metadata-only local transpose or reshape before `local_load`;
- a consumer-compatible register layout on `local_load` when the API and target support it;
- a shared-memory layout compatible with both the producer access and the dominant consumer;
- preserving a native dot-accumulator layout until narrowing or epilogue storage;
- relocating casts or elementwise work so they do not block packed conversion or layout propagation.

Keep layout selection aligned with vectorization, bank-conflict avoidance, descriptor legality, and tensor-core operand requirements. A conversion-free path that introduces uncoalesced traffic, bank conflicts, extra registers, or spills is not an improvement.

## Use Layout Constraints Deliberately

Treat layout constraints according to their semantics:

- use an unpinned requirement only when the layout is a preference the compiler may propagate or reconcile;
- use `release_layout` when downstream consumers should regain layout freedom;
- use a pinned requirement only for a proven correctness or performance boundary;
- never remove or bypass a pinned layout merely to reduce the visible conversion count.

Over-pinning can prevent propagation and create conversions on both sides of a boundary. Under-constraining can let an epilogue or store layout propagate backward into a hot accumulator or producer. Compare both directions and place the boundary where it minimizes total material movement.

## Preserve Memory And Task Contracts

Layout changes can alter thread ownership, vector width, local-memory addressing, and the point where data becomes visible. Before changing a layout across asynchronous or warp-specialized code, prove that:

- every producer and consumer agrees on tile shape and element ownership;
- descriptor and TMA block geometry remains legal;
- shared-memory and TMEM aliases retain their intended physical footprint;
- barriers, fences, arrival counts, and phase reuse still protect the same bytes;
- region-carried and cross-task values retain compatible layouts;
- boundary tiles and fallback paths remain valid.

Do not replace a physical transpose with a metadata-only view unless the resulting memory encoding and consumer interpretation are equivalent.

## Validate Material Savings

For every proposed layout change, retain evidence from before and after:

- final surviving conversion count and lowering category;
- allocated shared-memory or LDS scratch;
- register count, spills, occupancy, and code size;
- shared-memory or LDS transactions, bank conflicts, barriers, and dependency stalls;
- generated shuffle, load/store, and conversion instructions;
- correctness across shapes, dtypes, boundary cases, topology variants, and repeated launches;
- stable end-to-end benchmark latency.

Use conversion removal as a hypothesis, not the promotion criterion. Accept the change only when the intended physical movement or resource cost decreases and protected end-to-end performance improves without weakening numerical or synchronization contracts.
