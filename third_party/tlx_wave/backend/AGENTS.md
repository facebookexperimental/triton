# TLX Wave Backend Instructions

These instructions apply to `third_party/tlx_wave/backend/` and all
descendants. They are the working contract for the TLX Wave backend and its
from-scratch converter.

## Converter Architecture

- The production backend uses the structural converter for lowering.
- Unsupported semantics must fail with a structured diagnostic. Silent fallback
  or best-effort lowering is a bug.
- Lower supported TTGIR/ROCDL/AMDG ops to Wave/WaveAMD through structural Python
  bindings only. Do not handwrite Wave IR text, parse textual Wave snippets, or
  use text assembly as an internal lowering mechanism.
- Keep conversion decisions in analysis/rewriter stages. Emission is mechanical:
  it creates types, constants, Wave/WaveAMD ops, regions, and SSA bindings from
  already verified target IR.

## Hard Rules

- Rewriters are stateless. They receive the current source op, converted
  operands/results, facts, layout/dependency handles, and a target builder. They
  must not walk MLIR def-use chains, inspect users, or rediscover producers.
- Type conversion owns representation choice. Terminal lowerers must not infer
  representation by recursively inspecting operand producers.
- WaveAMD fragment types are an MMA-lowering detail, not a bridge value
  representation. They may exist only as immediate MMA operands/results inside
  MMA emission. Pack ordinary typed SIMD payloads immediately before MMA and
  unpack MMA results immediately afterward; fragment types must never cross a
  target-op, structured-control-flow, memory, or layout-conversion boundary.
- `ttg.convert_layout` is an explicit operation. It must produce an alias,
  structural `wave.redistribute`, or a diagnostic. Do not forward it and
  reinterpret it later in stores, DMA, masks, dot, or local-memory lowering.
- The only acceptable layout conversion boundary is an explicit
  `ttg.convert_layout` op in the source TTGIR. If a conversion is needed
  anywhere else, the layout model is wrong and the code must not be merged.
- Facts need provenance. Range, divisibility, pow2, no-overflow, uniformity, and
  contiguity facts must identify the source guarantee and fixed-width semantics.
  Do not invent facts from unsafe identities such as "nonnegative *
  nonnegative is nonnegative" unless no-overflow is proven.
- Target IR is schema-closed. Target ops contain target value IDs, primitive
  attrs, enum-like strings, fact IDs, layout IDs, and region IDs. They must not
  store source ops/values, converted-value objects, callables, lazy resolvers,
  emitter objects, or layout-analysis objects except as non-semantic debug
  provenance.
- Shared helpers must belong to one domain: source import, types, layouts,
  facts, tokens, target IR, diagnostics, verification, or emission. Do not add a
  helper that owns source program state, converted values, dependency graph,
  target program, and emitter state together.

## Memory And Tokens

- Memory dependency tokens encode required happens-before edges only. They are
  not a generic "this op has memory effects" chain or a scheduler preference.
- Plain loads use normal SSA value dependencies for data readiness. Retain or
  thread a load/store token only when a later operation requires memory
  ordering, such as a may-alias store, wait, barrier, volatile/atomic/fence, or
  source async protocol.
- Kernel pointer arguments may alias by default. Alignment, contiguity, and
  `pointer_range` attrs are not no-alias proofs.
- Global/buffer memory effects require conservative source-order edges for
  may-alias RAW, WAR, and WAW pairs unless a no-alias, disjoint-byte,
  inactive-mask, or equivalent proof exists. RAR pairs do not need ordering.
- Unknown address spaces are may-alias with mapped spaces unless proven
  otherwise.
- Async-copy packets and scalar/vector fallback chunks are group members, not a
  serial chain. Do not thread `after=previous_packet` or `after=previous_chunk`
  unless source semantics require packet order.
- Async DMA completion is synchronized exclusively by an explicit
  `wait_group`. Never consume async-DMA tokens through implicit memory
  dependencies, inferred local-memory hazards, destination accesses, ordinary
  CTA barriers, or unrelated waits. If correctness requires DMA completion,
  the source async protocol must contain the explicit wait; the backend must not
  synthesize one from dependency analysis.
- Direct-to-LDS DMA issue must not consult LDS alias classes, pending
  reads/writes, allocation history, or destination-access state. In particular,
  never infer a read-before-overwrite dependency or insert/reuse a CTA barrier
  from a DMA destination alias. A DMA `after` edge may come only from the
  explicit source async protocol. If that protocol does not order safe slot
  reuse, reject or fix the source program; do not make alias analysis more
  precise to manufacture the missing edge.
- An explicit full-memory source/compiler barrier may contribute one distinct,
  completion-free barrier-issue dependency to following memory issuers,
  including direct-to-LDS DMA. Build it structurally as
  `issue_token(memory completions) -> barrier -> issue_token(barrier result)`.
  The raw barrier result and all DMA/LDS completion events remain forbidden as
  DMA `after` operands. Pure operations and layout conversions must not consume
  this ordering token.
- A local load/store must never consume an in-flight async-DMA token through an
  inferred destination dependency. Data readiness comes only from the explicit
  wait associated with the source access (for example an explicit wait token or
  `syncedViaAsyncWait`). Missing readiness metadata is a source/protocol gap,
  not permission for the bridge to infer a DMA wait.
- The bridge must represent that readiness structurally: thread the explicit
  wait result token to every dominated DS consumer. Do not rely on textual
  ordering or change Wave scheduling to compensate for a missing token edge.
- `ttg.async_wait num = K` waits only groups that must complete while leaving
  the newest `K` committed groups live. A hot-loop wait for an older group must
  not become a full `vmcnt(0)`/`lgkmcnt(0)` drain while newer groups remain
  live.

## Layouts And Coordinates

- Layout maps are structural. Use TTGIR attr APIs and local structural helpers;
  do not parse layout strings when structural attributes are available.
- Distributed tensor layouts describe one Wave wave, not the whole CTA.
  `warpsPerCTA` contributes to the `warp` coordinate and must not be multiplied
  into per-wave component count.
- Coordinate-producing ops must go through the result layout map. The flat rule
  `start + component * lane_width + lane` is legal only when the layout map
  proves that relation.
- `ttg.convert_layout` lowers by composing source/result layout maps into either
  a physical identity alias or one structural `wave.redistribute` relation.
  Movement classification and scratch-exchange lowering belong to Wave.
- Facts do not survive layout conversion, casts, extension/truncation, or
  control-flow joins unless explicitly remapped, re-proven, or joined.
- Shared-memory physical offsets come from memdesc layout maps. Padded and
  swizzled layouts must use the same logical-to-physical remap as Triton/TLX.
- DMA legality is byte-for-byte equivalence. Prove source bytes, destination
  bytes, alignment, mask behavior, packet width, and padding/swizzle boundaries
  before selecting DMA.

## Operation Contracts

- Integer lowering preserves source-width fixed-width semantics. Widened
  arithmetic is legal only with facts proving equivalence, such as appropriate
  `nuw`/`nsw` or range facts.
- Div/rem lowering must prove nonzero divisors; signed div/rem must also prove
  the `INT_MIN / -1` overflow case impossible when relevant.
- Power-of-two assumptions must preserve source width. A fixed-width bit test is
  not the same as an unbounded symbolic predicate.
- FP ops and comparisons must preserve source flags and NaN/denorm/rounding
  semantics. Reject if target semantics cannot be proven equivalent.
- Masks are first-class values. Multi-component masks, `arith.andi`, and
  `ttg.convert_layout` on masks must lower explicitly or reject.
- Buffer promotion requires a uniform base pointer, bounded i32 active-lane byte
  offsets, legal mask behavior, and a carried or rejected cache modifier.
  Inactive masked lanes must not receive unconditional offset-range assumes.
- `other` values for masked loads/copies use the copied element type, not the
  pointer type.
- MFMA lowering is layout-driven. Fragment packs, unpacks, conversions, stores,
  and dot operands must use the MFMA layout bijection, not register-count or
  tuple-length guesses. gfx942/gfx950 differences belong in metadata.
- `scf.if` and `scf.for` lower as structured regions. Facts and required memory
  dependencies must have dominance/region scope and be yielded or joined when
  they cross structured boundaries.

## Module Boundaries

- `source_import.py`: import MLIR into source records.
- `types.py`: source/converted type records and type conversion.
- `layouts.py`: structural TTGIR layout maps and layout conversion.
- `facts.py`: range/divisibility/pow2/no-overflow/uniformity facts.
- `tokens.py`: async and memory-dependency graph.
- `target_ir.py`: target-program schema.
- `verifier.py`: target-program verification.
- `emission.py`: structural Wave Python binding emission only.
- `diagnostics.py`: consistent failure formatting.

Allowed dependency direction:

- Schema/data modules stay separate from stage implementations.
- Pre-emission stages must not import Wave bindings unless a file already has an
  explicit, narrow reason.
- Rewriters/op conversion may use source/type/layout/fact/token/target IR
  records and diagnostics, but must not import emission.
- Emission may use target IR, diagnostics, and Wave binding helpers. It must not
  import source analysis modules or make lowering-family decisions.
## Testing And Validation

- Keep tests layered: import, type/layout, facts, tokens, op conversion,
  verifier, emission, and end-to-end. Unit tests should exercise stages without
  requiring full GEMM compilation when possible.
- Negative tests should assert diagnostic code and key fields, not just that an
  exception was raised.
- Static tests should reject analysis-to-emitter cycles, text-fed MLIR/Wave
  parsing, source objects/callables in target IR attrs, recursive
  materialization, and hidden late decisions.
- Converter tests should include positive minimum-support cases so a stage
  cannot pass by rejecting everything.
- For GEMM/end-to-end checks, assert structural emitter path, instruction
  families, DMA/buffer/MFMA use, absence of known bloat patterns, and relevant
  metadata. Avoid exact tutorial-output checks unless the exact text is the
  behavior under test.
- When a feature has natural shape/layout degrees of freedom, add at least two
  shape/layout variants unless the task explicitly justifies narrower coverage.
- Runtime tests must be gated on HIP/torch availability and supported physical
  hardware. Compile-only tests may target gfx942/gfx950 without matching local
  hardware.

## Review Rejection Checklist

Reject or revise a patch if it:

- adds a source-user map outside `tokens.py`;
- gives a rewriter a source-program, owner-op, user-map, or producer lookup API;
- recursively materializes operand trees in emission;
- forwards `ttg.convert_layout` without an explicit converted result;
- formats or parses Wave IR text as lowering;
- adds silent fallback or a fallback that is not an explicit target IR op with a
  test-assertable reason;
- serializes RAR pairs, proven-disjoint effects, inactive masked effects, or
  independent packets only because they are memory effects;
- threads mutable "last token" state through packet, chunk, component, or store
  loops without a required happens-before edge;
- synchronizes an async DMA through an implicit dependency, inferred
  local-memory hazard, destination access, ordinary barrier, or unrelated wait
  instead of exclusively through an explicit `wait_group`;
- feeds a raw barrier, LDS-completion, or DMA-completion token to direct DMA,
  or derives a barrier-issue token from anything other than an explicit
  full-memory source/compiler barrier;
- queries local-memory alias/access state while emitting a direct-to-LDS DMA,
  or adds a DMA/read/store edge merely because their LDS destinations may
  overlap;
- lowers a steady-state async wait with newer live groups to full
  `vmcnt(0)`/`lgkmcnt(0)`;
- proves multiplication range without no-overflow;
- adds unmasked assumptions for masked lanes;
- treats pointer range as an element-index bound rather than an active-lane byte
  interval;
- uses facts after layout conversion/cast/control-flow without remap or join;
- emits DMA without byte-for-byte source/destination/alignment/mask proof;
- maps MFMA fragments by register count instead of layout bijection;
- adds kernel-shape-specific lowering without a general layout reason;
- centralizes multiple domains in a shared helper or service object.
