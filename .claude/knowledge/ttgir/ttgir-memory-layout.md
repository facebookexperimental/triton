# TTGIR Memory Layout Ops

Ops for creating views, transforming descriptors, and converting layouts.
These ops do not move data — they reinterpret how existing memory is
addressed.

## Memory Descriptor Views

All view ops are `Pure` (no side effects) and carry the `MemDescViewTrait`.
They return a new `MemDescType` pointing to the same underlying memory.

### `ttg.memdesc_index`
Index into dimension 0 of a memdesc, reducing rank by 1. Commonly used to
select a pipeline stage from a multi-buffered allocation.

```mlir
// Select stage `i` from a 3-stage buffer:
// 3x128x64xf16 → 128x64xf16
%stage = ttg.memdesc_index %buf[%i]
    : !ttg.memdesc<3x128x64xf16, #shared, #smem>
    -> !ttg.memdesc<128x64xf16, #shared, #smem>
```

### `ttg.memdesc_subslice`
Take a subview with static offsets. Each offset must be zero or a multiple
of the tile size for that dimension. The output shape is the result type's
shape.

```mlir
// Take a 8x16 subview starting at [2, 1]:
%sub = ttg.memdesc_subslice %buf[2, 1]
    : !ttg.memdesc<32x16xf16, #shared, #smem>
    -> !ttg.memdesc<8x16xf16, #shared, #smem>
```

### `ttg.memdesc_trans`
Transpose view of a memdesc. Permutes dimensions according to the `order`
attribute. The underlying memory layout must support the transposition.

```mlir
%transposed = ttg.memdesc_trans %buf {order = array<i32: 1, 0>}
    : !ttg.memdesc<128x64xf16, #shared, #smem>
    -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
```

### `ttg.memdesc_reshape`
Reshape a memdesc to a different shape. Only valid if the underlying memory
is contiguous. No data movement.

```mlir
%reshaped = ttg.memdesc_reshape %buf
    : !ttg.memdesc<1x32x2x64xf32, #shared, #smem>
    -> !ttg.memdesc<64x64xf32, #shared, #smem>
```

### `ttg.memdesc_reinterpret`
Reinterpret a memdesc with a different shape and element type. Only valid
for contiguous descriptors. Useful for bitcasting.

```mlir
%reinterp = ttg.memdesc_reinterpret %buf
    : !ttg.memdesc<128x64xf16, #shared, #smem>
    -> !ttg.memdesc<128x128xi8, #shared, #smem>
```

### `ttng.tmem_subslice` (Blackwell only)
Subslice of a TMEM allocation along the inner (column) dimension only.
TMEM layout restricts slicing to the inner dimension.

```mlir
%sub = ttng.tmem_subslice %tmem {N = 64}
    : !ttg.memdesc<128x128xf32, #tmem, #tensor_memory>
    -> !ttg.memdesc<128x64xf32, #tmem, #tensor_memory>
```

## Cluster Buffer Mapping

### `ttng.map_to_remote_buffer`
Given a local SMEM memdesc, returns a new memdesc referring to the
corresponding buffer in another CTA within the cluster. Pure, no data
movement. Used for distributed algorithms across CTAs.

```mlir
%remote = ttng.map_to_remote_buffer %local_buf, %cta_rank
    : !ttg.memdesc<128x64xf16, #shared, #smem>
    -> !ttg.memdesc<128x64xf16, #shared, #smem>
```

## TMA Descriptor Ops

### `ttng.reinterpret_tensor_descriptor`
Reinterprets a raw pointer as a typed TMA tensor descriptor. Transitional
op for converting between untyped and typed TMA objects. Pure.

```mlir
%desc = ttng.reinterpret_tensor_descriptor %raw_ptr
    : !tt.ptr<i8> to !tt.tensordesc<tensor<128x64xf16>>
```

### `ttng.tensormap_create`
Creates a TMA descriptor on device. Takes global address, box dimensions,
global dimensions, strides, element type, swizzle mode, etc. Has
global memory read/write effects (writes the descriptor to global memory).

```mlir
ttng.tensormap_create %desc_ptr, %global_addr,
    [%box_d0, %box_d1], [%glob_d0, %glob_d1],
    [%stride0], [%elem_stride0]
    {elem_type = 4, interleave_layout = 0, swizzle_mode = 3, fill_mode = 0}
    : ...
```

## Register Layout Conversion

### `ttg.convert_layout`
Converts a distributed tensor between register layouts (e.g., blocked ↔
MMA ↔ dot_operand). Pure operation at the TTGIR level, but may lower to
SMEM-mediated shuffles. Same shape and element type, different encoding.

```mlir
%converted = ttg.convert_layout %tensor
    : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot_op>
```
