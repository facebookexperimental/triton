# Minimal TLX Wave Layout Model

## Contract

Triton layouts describe how logical tensor elements are distributed across the
compute hierarchy. Wave already has the value model.

The bridge needs only:

- existing Wave value types;
- symbolic expressions for coordinates and offsets;
- layout queries over Triton's coordinate algebra;
- target diagnostics when a layout relation cannot be proven or emitted.

Do not introduce a second register-layout hierarchy. Register layout is carried
by Wave value types:

- uniform scalar or pointer: ordinary scalar MLIR value;
- lane-varying scalar: `!wave.simd<T, W>`;
- lane-varying predicate: `!wave.mask<W>`;
- per-lane vector/register packet: `!wave.simd<vector<NxT>, W>`.

WaveAMD fragment values do not participate in layout handling. MMA lowering may
construct `!waveamd.fragment<...>` values on demand from SIMD/vector operands,
but those fragments are local emission artifacts, not bridge layout
representations.

Bridge type metadata may keep component counts and target representation names
for packing and verification. Component count is not layout semantics and must
not select a movement class.

## Layout Algebra

The primitive layout object is a map from named hardware coordinates to logical
tensor coordinates. The input names and sizes are part of the layout; the bridge
must not hard-code only one coordinate tuple.

Common distributed tensor layouts use:

```text
(register, lane, warp, block) -> (dim0, dim1, ...)
```

Common shared-memory layouts use:

```text
(offset, block) -> (dim0, dim1, ...)
```

Other Triton-generated layouts can introduce dimensions such as `iteration`,
`partition`, `message`, or target-specific scale dimensions. They still fit this
model when imported as named `LinearLayout` input dimensions plus explicit target
metadata.

This direction matters. A logical element can have zero, one, or many hardware
locations. Any operation that needs `logical -> hardware` must invert or solve
the map under explicit constraints.

The canonical representation is Triton's `LinearLayout` wherever the encoding
is linear:

- `blocked`: construct the distributed `LinearLayout` from `sizePerThread`,
  `threadsPerWarp`, `warpsPerCTA`, `order`, and `CGALayout`;
- `linear`: import the exact bases, output dimensions, and the fact that the
  encoding is constrained/bijective after broadcast removal;
- `generic_linear`: import the exact bases, output dimensions, and the fact
  that the encoding is only required to be surjective;
- `slice`: project the parent map;
- `dot_operand`: use the parent layout plus `opIdx` and `kWidth`;
- `amd_mfma`: use the AMD MFMA distributed map and MMA access metadata;
- `amd_wmma`: use the AMD WMMA distributed map and MMA access metadata if the
  target supports it, otherwise reject before emission;
- `shared_linear`: import the full `(offset, block) -> dim*` map;
- `swizzled_shared`: use Triton's swizzled shared `LinearLayout` for
  `(offset, block) -> dim*`;
- `partitioned_shared`: unwrap or compose the partition layout and preserve the
  physical partition/allocation metadata needed for addressing;
- `amd_rotating_shared`: import the linear relation and rotation semantics, or
  reject before emission;
- `padded_shared`: use the full linear component plus explicit interval padding.

`padded_shared` is the one required non-linear extension. Its logical map is
the linear component, but its physical address adds padding:

```text
raw = solve(linearComponent, logical).offset
padded = raw + sum(floor(raw / interval_i) * padding_i)
```

`toLinearLayout` must not be used as the full physical representation of a
padded layout. Order plus shape is sufficient only for identity-shorthand
padded layouts. Non-identity padded layouts require the full `linearComponent`
or must be rejected.

Target-specific encodings outside the AMD Wave target, such as NVIDIA MMA or
unsupported WMMA variants, may still be describable by Triton layout algebra.
They are not WaveAMD-lowerable unless an MMA lowering contract exists, so the
bridge must reject them with a layout/target diagnostic.

Triton also creates helper layouts that are not direct source attributes, such
as shared scratch conversion layouts, scale layouts, and descriptor-message
layouts. These enter the same query model as named-dimension `LinearLayout`
values with side metadata. If a helper layout cannot be imported structurally,
the bridge must reject the operation that requires it.

## Symbolic Queries

The bridge mechanically serializes Triton's layout maps. It may apply a layout
in its forward direction when importing coordinates that are already explicit
in the source contract, but it does not invert, compose, enumerate, classify, or
lower distributed layout movement. Wave/iXSimpl owns those operations over its
frontend-neutral packet-layout representation.

### `coords(layout, hw)`

Return logical coordinate expressions for a hardware point.

```text
coords(blocked_layout, (reg, lane, warp, block)) -> (m, n)
coords(shared_layout, (offset, block)) -> (m, n)
```

This is the direct Triton layout direction and should be implemented by applying
the imported `LinearLayout` or the MMA layout metadata.

### `solve(layout, logical, constraints)`

Return hardware coordinates that produce a logical coordinate under constraints.

```text
solve(register_layout, (m, n), lane = active_lane)
solve(shared_layout, (m, n), block = active_block)
```

The Wave-side result is one of:

- no solution: masked-out element or unsupported relation;
- one solution: direct movement or address;
- multiple solutions: choose a specified representative or reject.

When matching Triton's `invertAndCompose` semantics for non-injective maps, the
Wave model chooses a deterministic physical representative while preserving
matching physical dimensions when possible. The bridge never computes or
serializes that representative.

Wave's packet-layout solver uses the imported named bases plus structural
operation semantics. It derives symbolic `source_block`, `source_item`, and
`source_slot` expressions for `wave.redistribute`; Wave lowering then decides
whether those expressions are aliases, register extracts, shuffles, or require
another generic redistribution mechanism.

### `physical_offset(shared_layout, logical, unit)`

Return a shared-memory physical offset record.

The record must state:

- element offset expression;
- byte offset expression when needed;
- dword offset expression when needed;
- element byte width;
- layout kind and order/provenance;
- bindings and assumptions used by the expression;
- proof status or explicit fallback/reject reason.

All local-memory consumers use this query: scalarized paths, vector packets,
MMA operand loads/stores, and DMA fallbacks. The query owns padded and swizzled
physical addressing, so consumers do not rederive those formulas.

For swizzled shared layouts, use the swizzled shared `LinearLayout` for the
logical relation and prove that the selected packet does not cross an illegal
swizzle boundary.

For partitioned or rotating shared layouts, `physical_offset` must either
compose the physical partition/rotation mapping into the returned offset record
or reject the consumer before emission. A plain `(offset, block)` answer is not
enough when Triton's layout has additional physical dimensions.

### `mma_access(layout, payload)`

Return logical matrix coordinates for a SIMD/vector payload element used by MMA.

The payload remains ordinary Wave data, usually `!wave.simd<T, W>` or
`!wave.simd<vector<NxT>, W>`. If the physical value is a vector packet, evaluate
the underlying scalar register coordinates with:

```text
register = component * vector_length + vector_index
```

MMA lowering may pack those SIMD/vector values into WaveAMD fragments
immediately before emitting the WaveAMD MMA op. Layout analysis never treats the
fragment type as a layout-bearing value.

AMD MMA access helpers must mirror Triton's layout helpers. For MFMA this means
`AMDMfmaEncodingAttr::toLinearLayout` and `mfmaDotToLinearLayout`, including:

- instruction shape and version;
- `tilesPerWarp`;
- `warpsPerCTA` and `CGALayout`;
- `isTransposed`;
- element bit width;
- `opIdx` and `kWidth`;
- operand role and vector payload width;
- Triton's MFMA tile order for N-contiguous operands.

For AMD WMMA or future AMD MMA families, `mma_access` must import the equivalent
Triton linear-layout helper and target metadata, or reject the layout family.

## Operation Use

### Coordinate Producers

`tt.make_range` and similar coordinate-producing ops call
`coords(result_layout, hw)` and emit the resulting expressions through
`wave.index_expr`.

The flat expression:

```text
start + component * wave_size + lane
```

is legal only when it is the expression produced by the result layout map.

### Local Memory

Local load/store lowering composes:

```text
value logical coords
  -> memdesc view transform
  -> physical_offset(memdesc layout, logical coords)
  -> Wave pointer/index expression
```

The physical offset record decides the units used by the consumer. A Wave load
or store sees a pointer and a value type, not the original Triton layout.

### DMA

DMA selection is a byte-for-byte proof over source and destination expressions.

The proof must cover:

- source byte interval;
- destination byte interval from `physical_offset`;
- packet width;
- alignment;
- active/inactive mask behavior;
- zero-fill behavior;
- padding and swizzle boundaries;
- address-field and M0 planning obligations for WaveAMDMachine.

`waveamd.dma_load_lds` has stricter target constraints than generic Wave memory
ops. A legal lowering must select a supported 4-byte or 16-byte DMA mode, use a
SIMD global/buffer source pointer, produce a uniform shared destination pointer,
and satisfy the machine address-field proofs. If any proof fails, lower an
explicit scalar/vector fallback or reject.

### `ttg.convert_layout`

`ttg.convert_layout` crosses the bridge as one structural operation with source
and result layout IDs. The operation carries no movement mode, physical source
map, scratch plan, register permutation, or component-source table. Reshape and
transpose additionally carry only their literal structural transform, while
broadcast, expand, join, and split retain their source operation semantics.

Emission mechanically converts each serialized `TargetLinearLayout` into a
frontend-neutral Wave `PacketLayout`, packs the already selected Wave value
representation, and asks Wave to build a symbolic `wave.redistribute`. It does
not inspect producers or users and does not select a lowering family.

For distributed tensor layouts, define:

```text
S = toLinearLayout(source_type)
D = toLinearLayout(result_type)
C = D.invertAndCompose(S)
```

`C` maps each result hardware coordinate to the source hardware coordinate that
contains the same logical tensor element:

```text
D(result_hw) == S(C(result_hw))
```

Wave derives `C` symbolically from the two packet layouts and the structural
transform. Matching physical dimensions remain direct when that is a valid
solution; otherwise the generic solver constructs the required relation. The
`wave-lower-redistribute` pass alone classifies and lowers the result:

- an identity relation folds to an alias;
- register-only relations become ordinary pack/extract operations;
- lane relations become generic Wave shuffles;
- warp or block relations use Wave's generic cross-wave redistribution path;
- unsupported or non-total relations are diagnosed by Wave.

The same solver handles blocked, linear, generic-linear, slice, dot-operand,
and MMA-derived packet layouts. Component count affects only the mechanical
Wave packet type. It never chooses a source component or a movement class.

### Dot / MMA

Dot lowering consumes SIMD/vector operands whose payload order is described by
layout algebra. It may construct WaveAMD fragments on demand inside MMA
emission, then discard that representation immediately after the target MMA
operation.

Checks:

- operand role and element type match the selected MMA;
- wave size, shape, and vector payload width match the selected MMA contract;
- `kWidth`, transpose, and tiling metadata match SIMD/vector payload
  coordinates;
- shared MMA operand loads and register vector packs use the same `mma_access`
  query;
- gfx-specific differences live in metadata, not tuple-length guesses.

## Import Requirements

The importer must structurally preserve enough data to build the queries:

- blocked: `sizePerThread`, `threadsPerWarp`, `warpsPerCTA`, `order`,
  `CGALayout`;
- linear/generic-linear: attr kind, input bases for register/lane/warp/block,
  output dimension names, output dimension sizes, and `CGALayout`;
- slice: dimension and parent layout;
- dot operand: `opIdx`, `kWidth`, parent layout, parent kind, and parent
  MMA access metadata when applicable;
- AMD MFMA: version, instruction shape, transpose flag, `warpsPerCTA`,
  `tilesPerWarp`, element bit width, `CGALayout`, and rank/order metadata;
- AMD WMMA or other AMD MMA families: target family, instruction shape,
  `ctaLayout`, swizzled warp layout data, element bit width, and rank/order
  metadata, or an explicit unsupported-target diagnostic;
- shared linear: full `LinearLayout` for `(offset, block) -> dim*`;
- swizzled shared: full swizzled `LinearLayout`, `vec`, `perPhase`,
  `maxPhase`, `order`, and `CGALayout`;
- padded shared: intervals, paddings, and full `linearComponent`;
- partitioned or nested shared wrappers: enough data to unwrap or compose them
  before querying;
- AMD rotating shared: full linear relation and rotation metadata;
- helper-generated layouts: full named-dimension `LinearLayout` plus any side
  metadata needed by the consumer.

If any required field is unavailable, the bridge must reject the layout before
emission and name the missing field.

## Target IR

Target IR is self-contained before emission. `TargetContract` versions the
schema, states that address arithmetic does not overflow, and carries the
compiler's FP-fusion permission. `TargetLayout`
records mechanically copy layout identity, kind, shape, element type,
component count, lane width, encoding parameters, and serialized linear-layout
dimensions and bases. `TargetAssumption` records bind proven range or
divisibility predicates directly to target value IDs.

Target operations carry only schema data needed for emission:

- chosen Wave representation kind from type conversion;
- expression records: expression payload, binding names, binding target IDs,
  binding types, and assumptions;
- structural transform kind and literal axis/order where the source operation
  has one;
- source and result IDs that mechanically reference serialized symbolic layouts;
- physical-offset records with units and proof status;
- MMA access metadata: operand role, instruction shape, element type, wave size,
  vector payload width, target family, and metadata ID;
- DMA proof data: packet width, alignment, contiguity, mask behavior, boundary
  proof, and machine address obligations.

Target IR must not carry source MLIR objects, Python layout objects, fact or
layout analysis records, lazy resolvers, callables, emitter state, or
unverified analysis objects. Verification and emission consume the target
program without a side-channel fact or layout program.

Emission may switch on verified structural operation kinds. It must not inspect
source layouts, derive a physical map, or choose a lowering family.

## Verifier Requirements

Verifier checks required for this model:

- schema version and no-overflow address contract are supported;
- every target layout and assumption ID is local, dense, and well formed;
- every value and operation layout reference names a target layout;
- every assumption use names one of its target subjects;
- every layout-sensitive op has expression records, physical-offset records, or
  source/result serialized symbolic layouts;
- every expression binding is present and has a compatible Wave type;
- coordinate expression rank matches tensor rank;
- shared offsets declare units and element byte width;
- layout-changing operations carry only their closed structural attr set;
- distributed source/result layouts have well-formed named dimensions and bases;
- DMA attrs include alignment, contiguity, mask, boundary, packet-width, and
  machine-address proof data;
- MMA attrs match the selected SIMD/vector operand contract and the on-demand
  WaveAMD fragment construction requirements;
- named layout dimensions in target attrs match the imported layout metadata;
- target IR contains no raw layout objects or source objects.

## Coverage

The model covers Triton's layout algebra when:

- all linear layout data is imported as `LinearLayout` data;
- padded shared layouts are represented as linear component plus padding
  intervals;
- MMA operand access is represented by target metadata that matches Triton's
  AMD MMA and dot-operand layout helpers;
- named input dimensions are preserved for helper-generated layouts and shared
  wrappers rather than collapsed to a fixed tuple;
- each layout-sensitive operation asks for the query it actually needs.

The model does not promise that every algebraic relation is target-lowerable.
Unsupported movement, unsupported MMA families, missing metadata,
non-unique representatives, and unprovable machine constraints are valid
diagnostics.

## Acceptance Criteria

- No bridge-owned replacement for Wave uniform/SIMD/vector types.
- WaveAMD fragments are constructed only on demand inside MMA emission.
- Triton distributed layouts are mechanically serialized into Wave's symbolic
  packet-layout representation.
- Shared memory, DMA, scalar fallback, and MMA operand local-load paths share the
  same physical-offset query.
- Wave/iXSimpl derives layout movement from source/result symbolic layouts plus
  structural operation semantics, never from bridge component tables.
- Dot operands, MMA operand packs, and accumulator repacks use AMD MMA layout
  metadata over SIMD/vector payloads, not tuple-length guesses.
- Unsupported layouts fail with a diagnostic naming the missing symbolic
  layout, metadata field, or proof.
