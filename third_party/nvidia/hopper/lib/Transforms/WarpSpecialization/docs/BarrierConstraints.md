# Barrier Constraints Design

## Overview

Barrier and token ops (`wait_barrier`, `arrive_barrier`, `producer_acquire`,
`producer_commit`, `consumer_wait`, `consumer_release`) accept an optional
`constraints` argument of type `DictionaryAttr`. This provides a generic,
extensible mechanism for passes to attach context-dependent metadata to
barrier operations without modifying the op definitions.

## Motivation

Different compilation stages need to annotate barrier ops with different
metadata:

- **Subtile lowering** needs to know which tiles should emit a barrier and
  how many buffers to use for phase computation.
- **Pipeline scheduling** needs to track pipeline stages and clusters.
- **Barrier fusion** needs to know which barriers can be merged.

Rather than adding a new attribute to the op definition for each use case
(which couples the op to specific passes), the `constraints` dict provides
a single extensible slot. Each consuming pass defines its own key namespace
and ignores keys it doesn't recognize.

## Design Principles

1. **Optional**: The attribute is `OptionalAttr<DictionaryAttr>`. When absent
   (the default), the barrier behaves exactly as before. All existing code
   is unchanged.

2. **Dict-based**: A `DictionaryAttr` rather than a structured attribute.
   This avoids defining a new TableGen attribute for every combination of
   constraints. Passes validate the keys they care about at use time.

3. **Namespace by convention**: Each pass owns a set of keys. Keys are
   plain strings. No formal namespace enforcement — collisions are avoided
   by using descriptive names.

4. **Argument, not discardable attr**: The `constraints` is declared in
   the op's `arguments` list, not as a discardable attribute. This means:
   - It participates in the op's builder signatures.
   - It's part of the op's identity for comparison/hashing.
   - It won't be silently stripped by passes that drop unknown attrs.
   - It appears in `attr-dict` in the assembly format.

5. **Forward-compatible**: A pass that doesn't understand a key simply
   ignores it. Adding new constraint keys doesn't require changing any
   existing pass.

## Constraint Keys

### Subtile Lowering (`LowerSubtiledRegionPass`)

| Key | Type | Description |
|-----|------|-------------|
| `loweringMask` | `DenseI32ArrayAttr` | Per-tile mask: emit barrier only for tiles where mask[i] != 0. Length must equal number of tiles. Absent = all tiles. |
| `numBuffers` | `I32Attr` | Number of buffer slots for phase computation: `phase = (accumCnt + tileIdx) / numBuffers & 1`. Default 1. |

Example:
```mlir
// Wait only on tile 0, use 2-buffer phase rotation
ttng.wait_barrier %bar, %phase {
  constraints = {loweringMask = array<i32: 1, 0>, numBuffers = 2 : i32}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// Arrive only on tile 1
ttng.arrive_barrier %bar, 1 {
  constraints = {loweringMask = array<i32: 0, 1>}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

### Pipeline Scheduling (future)

| Key | Type | Description |
|-----|------|-------------|
| `pipelineStage` | `I32Attr` | Which pipeline stage this barrier belongs to. |
| `cluster` | `I32Attr` | Loop cluster for scheduling. |

### Token Ops

The same `constraints` dict is available on the NVWS token ops:

```mlir
nvws.producer_acquire %tok, %idx, %phase {
  constraints = {subtileChannel = true}
} : tensor<1x!nvws.token>, i32, i1

nvws.consumer_wait %tok, %idx, %phase {
  constraints = {subtileChannel = true}
} : tensor<1x!nvws.token>, i32, i1
```

Token-specific constraint keys can signal to `doTokenLowering` how to
convert the token op — e.g., `subtileChannel = true` could indicate that
the resulting barrier should use per-subtile phase tracking.

## Assembly Format

The constraints appear in the `attr-dict` portion of the assembly:

```mlir
// Without constraints (default)
ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// With constraints
ttng.wait_barrier %bar, %phase {constraints = {numBuffers = 2 : i32}}
    : !ttg.memdesc<1xi64, #shared, #smem, mutable>

// Multiple constraint keys
ttng.arrive_barrier %bar, 1 {
  constraints = {loweringMask = array<i32: 0, 1>, pipelineStage = 0 : i32}
} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

## Builder API

Custom builders default `constraints` to null so existing callers are
unchanged:

```cpp
// Existing call — still works
WaitBarrierOp::create(builder, loc, barrier, phase);

// With constraints
auto constraints = DictionaryAttr::get(ctx, {
  NamedAttribute(StringAttr::get(ctx, "loweringMask"),
                 DenseI32ArrayAttr::get(ctx, {1, 0})),
  NamedAttribute(StringAttr::get(ctx, "numBuffers"),
                 builder.getI32IntegerAttr(2)),
});
WaitBarrierOp::create(builder, loc, barrier, phase,
                       /*pred=*/Value(), /*deps=*/{}, constraints);
```

## Accessing Constraints

```cpp
if (auto constraints = waitOp.getConstraints()) {
  if (auto mask = constraints.getAs<DenseI32ArrayAttr>("loweringMask")) {
    // Use mask for selective tile emission
  }
  if (auto numBuf = constraints.getAs<IntegerAttr>("numBuffers")) {
    unsigned n = numBuf.getInt();
    // Use n for phase computation
  }
}
```

## Interaction with SubtiledRegionOp

The WSBarrier marker ops (`ws_wait_barrier`, `ws_arrive_barrier`) defined
inside SubtiledRegionOp tile bodies serve a different purpose: they use
attribute-based barrier references (`barrierIdx`) to avoid SSA captures
across `IsolatedFromAbove` boundaries. The `constraints` dict on real
barrier ops is complementary — it annotates the actual `wait_barrier` /
`arrive_barrier` ops that exist outside or after lowering.

The migration path:
1. `doCodePartitionPost` creates token annotations on SubtiledRegionOps
2. `doTokenLowering` converts tokens to real barrier ops with `constraints`
   encoding the subtile context (loweringMask, numBuffers)
3. `LowerSubtiledRegionPass` reads constraints when expanding tiles

Alternatively, WSBarrier marker ops can carry their own `loweringMask`
attribute directly (as currently defined). The two approaches can coexist:
- WSBarrier ops for barriers inside the tile body (attribute-based refs)
- `constraints` dict for barriers outside the SubtiledRegionOp or after
  lowering
