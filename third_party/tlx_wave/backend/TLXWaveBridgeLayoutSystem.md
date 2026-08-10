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
tensor coordinates. The input names and sizes are part of the layout. Each
consumer declares the physical domain it can bind; a layout must be composed
into that domain before the query runs.

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
model, but a packet operation cannot materialize those coordinates directly.
The producing transform must compose them into `(register, lane, warp, block)`;
otherwise the packet query rejects the unbound input before emission.

This direction matters. A logical element can have zero, one, or many hardware
locations. Any operation that needs `logical -> hardware` must use a proved
inverse composition under explicit constraints.

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
linearComponent(raw, active_block) = logical
padded = raw + sum(floor(raw / interval_i) * padding_i)
```

Here `raw` is the proved physical representative for `active_block`.

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
layouts. Their producer must compose the helper coordinates into the consuming
packet or address domain. If that composition or required side metadata is not
available, the bridge rejects the operation before emission.

## Symbolic Queries

The bridge derives every packet relation through the same logical-coordinate
path. It evaluates the result `LinearLayout` at the destination packet,
applies the operation's exact logical transform to those expressions, and
evaluates one canonical inverse of the source layout:

```text
source_inverse = logical_identity.invert_and_compose(source)
result_logical = evaluate(result, destination_physical)
source_logical = logical_transform(result_logical, structural_inputs)
source_physical = evaluate(source_inverse, source_logical)
source_physical[replicated_bit] = destination_physical[replicated_bit]
```

Identity, transpose, broadcast, expand, join, split, reduction, and reshape all
use this derivation. A reshape is the literal mixed-radix delinearization of
`result_logical`; it does not select a second layout model. Split and reduction
add their one literal structural coordinate to the same transform. The inverse
selects the canonical representative, except that each literal-zero source
physical basis retains the corresponding bounded destination-physical bit. Such
a basis is exact value replication, so this changes ownership locality without
changing the logical value. Coordinates introduced by a structural operation,
such as a reduction coordinate or a fixed split selector, remain explicit
relation inputs.

The bridge translates each `LinearLayout` through the Wave DSL symbolic facade
into its shared symbolic context. The bridge never imports the underlying
solver directly.
Each logical coordinate is the canonical combination of every basis
contribution. For a power-of-two coordinate that combination is Triton's GF(2)
XOR; for a non-power-of-two coordinate it is addition modulo the coordinate
extent. There is no separate origin decomposition or additive approximation of
the layout:

```text
logical[dim] = combine(layout_basis[input_bit][dim] * bit(input))
```

The proof applies the logical transform independently to the result expressions.
It maps that point through Triton's canonical pseudoinverse and checks logical
preservation through the pseudoinverse/source `LinearLayout` composition.  The
source is required to be surjective, so this composition is its exact logical
right-inverse contract; the transform itself remains outside the composition.
The mapped physical expressions are independently canonicalized and proved
in-bounds before serialization.  Replicated physical bits are retained only
after the logical check because their source bases are zero and cannot change a
logical coordinate.  No family-specific coordinate formula validates itself.

One fact set bounds the destination packet coordinates and any structural input.
The symbolic query proves the composed inverse maps to the required logical
coordinates, every selected source coordinate bound, the packed
source-coordinate range, and the exact pack/unpack round trip. The separate
coordinate bounds prevent a carry in one coordinate from cancelling an
out-of-range value in another. A false or unknown result rejects the relation.
The only relation payload serialized into target IR is the proved packed
expression:

```text
R = source_slot + source_slots * (source_item + items * source_block)
```

Emission structurally deserializes `R` once, projects the three existing
`wave.redistribute` expressions with floor and modulo, and normalizes those
projections under the same exact destination packet-coordinate facts (including
the reduction extent when present). This keeps the producer and consumer on one
canonical symbolic carrier. No layout bases, inverse relation state, or
unproved analysis object crosses this boundary.

### `coords(layout, hw)`

Return logical coordinate expressions for a hardware point.

```text
coords(blocked_layout, (reg, lane, warp, block)) -> (m, n)
coords(shared_layout, (offset, block)) -> (m, n)
```

This is the direct Triton layout direction and should be implemented by applying
the imported `LinearLayout` or the MMA layout metadata.

### `local_memory_offsets(distributed, shared, view)`

Compose the distributed logical-coordinate expression, the memdesc view origin,
and the shared physical layout into one bit-offset expression over the full
physical slot domain:

```text
bit_offset[item, full_slot]
  = element_bits * shared_offset(
      distributed(item, full_slot) + view_origin)
```

The query proves every logical and physical coordinate bound, every swizzle
minor bound, and the final allocation bound through the same Wave DSL symbolic
context. A false or unknown proof rejects the operation. Only the serialized
bit-offset expression crosses into target IR; coordinate bases, memdesc shapes,
swizzle parameters, padding plans, and proof helper state remain private to the
query.

All local-memory consumers use this composition: scalarized paths, vector
packets, MMA operand loads/stores, and DMA copies. The query owns dense,
shared-linear, padded, and swizzled physical addressing, so consumers do not
rederive those formulas.

Dense shared storage always uses the same ordered mixed-radix row-major formula,
regardless of whether any shape extent is a power of two. A `LinearLayout`
inverse is reserved for shared layouts that encode a nontrivial physical map.

For swizzled shared layouts, use the swizzled shared `LinearLayout` for the
logical relation and prove that the selected packet does not cross an illegal
swizzle boundary.

For partitioned or rotating shared layouts, the query must either compose the
physical partition/rotation mapping into the returned expression
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

The imported `LinearLayout` remains the only layout object. For payload width
`P`, the bridge evaluates that one map at

```text
register = component * P + intra
```

and derives fragment and tile coordinates with `Mod` and `floor`. Component
and warp/workgroup quotient bits may be interleaved; compact/deposit
expressions compose those subspaces directly. One `ixs_check(exprs, facts)`
batch proves component order, fragment independence, tile composition, bounds,
and quotient/remainder reconstruction. The bridge never samples basis points
or constructs fragment, component, or partition `LinearLayout` substitutes.

For AMD WMMA or future AMD MMA families, `mma_access` must import the equivalent
Triton linear-layout helper and target metadata, or reject the layout family.

## Operation Use

### Coordinate Producers

`tt.make_range` and similar coordinate-producing ops call
`coords(result_layout, hw)` and emit the resulting expressions through
`wave.index_expr`.

`tt.make_range` emits one scalar relation over `(block, item, slot)`, so its
component count must equal the distributed layout's register-domain size.
Emission specializes only `slot` when materializing physical components and
records the unspecialized relation as the value's producer-carried SSA relation.

The flat expression:

```text
start + component * wave_size + lane
```

is legal only when it is the expression produced by the result layout map.

### Local Memory

Local load/store lowering composes:

```text
distributed(item, full_slot)
  -> memdesc view transform
  -> shared physical bit offset
```

The root converter serializes that one proved expression. Emission deserializes
it once and mechanically substitutes
`full_slot = packet * packet_width + slot`; it does not rebuild coordinates or
inspect the original Triton layout. A Wave load or store sees a base pointer,
the specialized bit-offset expression, and a value packet.

Wave keeps logical packet slots distinct from physical addresses. Its complete
semantics are one proved point transaction per logical slot; therefore duplicate
gather addresses still produce every requested output, permuted gather
addresses are repacked in logical slot order, and scatter values remain tied to
their original logical slots. A wider transaction replaces those points only
when the index-map proof establishes the address displacement and activity for
each `within` position. Coalescing must not silently deduplicate or reorder a
packet.

Replicated store layouts are restricted by the RREF kernel of the imported
`LinearLayout`. If `J` includes the basic register-column subspace with free
columns fixed to zero, the packet layout is the ordinary composition `L o J`.
The bridge selects payload components by that same `J` and evaluates the index
family through the restricted bases; it does not build a component address
table or a Piecewise selector. Lane and warp ownership remain one direct-parent
`wave.where` activity condition because those are execution coordinates rather
than packet coordinates. Source ownership masks are accepted only when they
name columns in the proved kernel.

`ttg.memdesc_index` uses the same shared address map. For child logical element
count `C`, the view base is the physical image of the parent logical element
`slot * C`; mixed-radix delinearization, swizzle, and padding all stay in that
one expression. The query also proves over every child element `e` that

```text
parent_address(slot * C + e)
  = parent_address(slot * C) + child_address(e)
```

so advancing the base pointer preserves the child view's layout. False or
unknown translation proofs reject the view. The target carrier contains only
the serialized element-offset expression and its slot-domain extent. Emission
binds `slot` directly to the index SSA through `wave.index_expr`; constants and
dynamic SSA chains take the identical path. No slot table, sampled stride,
static-offset channel, or child-size fallback exists. Shared allocation size is
the exclusive output extent of the same physical map; padding after the final
logical element is not allocated.

### DMA

DMA selection is a byte-for-byte proof over source and destination expressions.
The gather/scatter pair is a closed copy domain, so the physical issue domain
need not use the packet's original `(item, slot)` parameterization. Instead,
Wave constructs one transaction domain and pulls both address maps back through
it. It proves source address, destination address, activity, and bounds for
every point before selecting a DMA transaction. This is an algebraic
factorization, not candidate search and not a special case for one blocked
layout.

For example, a slot-major eight-element packet over one 64-lane wave has the
original linear position `item + 64 * slot`. A legal eight-element-per-lane DMA
domain uses:

```text
p = 8 * lane + within
original_item = p mod 64
original_slot = floor(p / 64)
```

Pulling both maps back through this relation preserves every source/destination
pair while exposing the contiguous physical transaction. Lane-major packets
use the same proof and reduce to the identity parameterization.

The proof must cover:

- source byte interval;
- destination byte interval from the composed bit-offset relation;
- packet width;
- alignment;
- active/inactive mask behavior;
- zero-fill behavior;
- padding and swizzle boundaries;
- WaveAMDMachine address-field and M0 constraints.

`waveamd.dma_load_lds` has stricter target constraints than generic Wave memory
ops. A legal lowering must select a supported 4-byte or 16-byte DMA mode, use a
SIMD global/buffer source pointer, produce a uniform shared destination pointer,
and satisfy the machine address-field proofs. If the DMA proof fails, lower an
independently proved scalar/vector transaction or reject.

### `ttg.convert_layout`

`ttg.convert_layout` crosses the bridge as one structural operation whose normal
operand and result metadata identifies the layouts and whose attributes contain
the private physical-relation carrier. The carrier contains no movement mode,
register permutation, or component-source table.
Reshape, transpose, broadcast, expand, join, split, and reduction all derive the
same carrier from their literal logical transform and Triton's layout algebra.

Emission packs the already selected Wave value representation, deserializes the
proved packed relation, and emits the existing `wave.redistribute` relation. It
does not inspect producers or users and does not select a lowering family.

For distributed tensor layouts, define:

```text
S = toLinearLayout(source_type)
D = toLinearLayout(result_type)
P = logical_identity.invertAndCompose(S)
C(h) = P(T(D(h)))
```

Here `T` is the operation's exact logical transform. `C` maps each result
hardware coordinate to a source hardware coordinate containing the same
logical tensor element:

```text
T(D(result_hw)) == S(C(result_hw))
```

After evaluating `C`, the bridge retains `h` for every individual physical bit
whose source basis is the zero vector. This is the locality-preserving section
of an explicitly replicated value; every nonzero source basis remains under
the canonical inverse. The `wave-lower-redistribute` pass only classifies and
lowers the resulting expressions:

- an identity relation folds to an alias;
- register-only relations become ordinary pack/extract operations;
- lane relations become generic Wave shuffles;
- warp or block relations use Wave's generic cross-wave redistribution path;
- unsupported or non-total relations are diagnosed by Wave.

The same Triton composition handles blocked, linear, generic-linear, slice,
dot-operand, and MMA-derived layouts. Component count affects only the
mechanical Wave packet type. It never chooses a source component or a movement
class.

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
- serialized Wave DSL symbolic relations for redistribution and local-memory
  bit offsets;
- structural transform kind and literal axis/order where the source operation
  has one;
- scalar packet facts such as component count, packet count, packet width,
  element width, and lane width;
- MMA access metadata: operand role, instruction shape, element type, wave size,
  vector payload width, target family, and metadata ID;
- ordinary memory metadata such as cache, contiguity, mask behavior, and pointer
  range.

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
- every layout-sensitive op has the required serialized relation payload;
- every expression binding is present and has a compatible Wave type;
- coordinate expression rank matches tensor rank;
- local-memory relations have the declared packet count and element width;
- layout-changing operations carry only their closed structural attr set;
- distributed source/result layouts have well-formed named dimensions and bases;
- DMA copies have one destination bit-offset relation and a matching packet
  domain;
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
- Triton distributed layouts are consumed by direct composition in the bridge;
  only the proved packed relation is serialized for redistribution.
- Shared memory, DMA, exact scalar transactions, and MMA operand local-load paths share the
  same composed bit-offset query.
- The bridge derives and proves layout movement through the Wave DSL symbolic
  facade; Wave consumes the packed relation without rederiving it.
- Dot operands, MMA operand packs, and accumulator repacks use AMD MMA layout
  metadata over SIMD/vector payloads, not tuple-length guesses.
- Unsupported layouts fail with a diagnostic naming the missing symbolic
  layout, metadata field, or proof.
