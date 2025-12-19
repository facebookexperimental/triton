# Placeholder Layouts in TLX

## Motivating Problem

In Triton, layout encodings (such as `BlockedEncodingAttr`, `NvidiaMmaEncodingAttr`, `DotOperandEncodingAttr`, etc.) determine how tensor data is distributed across threads, warps, and CTAs. Many of these layouts depend on the **number of warps** (`num_warps`) to compute the correct distribution.

A critical issue arises when TLX functions are defined separately from their call sites:

1. **Separate function definition**: When a TLX kernel helper is written as a separate function, any layout computation during lowering sees the **global module's `num_warps`**.

2. **Inlined context**: After function inlining, the same code may execute in a different context (e.g., inside a `tlx.async_task` region) where the **effective `num_warps` is different** from the global value.

This mismatch causes incorrect or inconsistent layouts. For example:
- A function lowered with `num_warps=4` at the global level
- Gets inlined into an `async_task` that executes with `num_warps=2`
- The pre-computed layout is now wrong for the actual execution context

**Solution**: We use **placeholder (dummy) layouts** during initial lowering that defer the actual layout computation until after function inlining. A dedicated pass (`TLXResolvePlaceholderLayouts`) then resolves these placeholders to concrete layouts when the correct `num_warps` and other context information is available.

---

## Overview

The placeholder layout system consists of three components:

1. **Placeholder Layout Attributes**: MLIR attributes that carry shape and type information but defer concrete layout decisions
2. **Python Encoding Classes**: Frontend classes that generate placeholder layout attributes during lowering
3. **Resolution Pass**: A C++ pass that replaces placeholder layouts with concrete layouts after inlining

---

## Placeholder Layout Types

We define five placeholder layout types, organized by memory space and use case:

| Placeholder Type | Memory Space | Resolves To |
|------------------|--------------|-------------|
| `DummyTMemLayoutAttr` | Tensor Memory (TMEM) | `TensorMemoryEncodingAttr` |
| `DummySMemLayoutAttr` | Shared Memory (SMEM) | `SwizzledSharedEncodingAttr` or `NVMMASharedEncodingAttr` |
| `DummyRegisterLayoutAttr` | Registers | `BlockedEncodingAttr` |
| `DummyMMALayoutAttr` | Registers (MMA) | `NvidiaMmaEncodingAttr` |
| `DummyDotOperandLayoutAttr` | Registers (Dot Operand) | `DotOperandEncodingAttr` |

> **Note**: `DummyMMALayoutAttr` and `DummyDotOperandLayoutAttr` are only used for Hopper (version 3) MMA operations. We may look to remove these in the future if we can derive them from the IR.

### IR Examples

**Before resolution:**
```mlir
// SMEM with placeholder layout
%0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #tlx.dummy_smem_layout<[128, 64], f16>, #smem>

// TMEM with placeholder layout
%1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #tlx.dummy_tmem_layout<[128, 64], f32, unpacked=true>, #tmem>

// Register tensor with placeholder layout
%2 = tlx.require_layout %arg : tensor<128x64xf16, #tlx.dummy_register_layout<[128, 64], f16>>
```

**After resolution:**
```mlir
// SMEM resolved to NVMMAShared
%0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #ttg.nvmma_shared<...>, #smem>

// TMEM resolved to TensorMemory encoding
%1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #ttng.tensor_memory<blockM=128, blockN=64, unpacked=true>, #tmem>

// Register resolved to Blocked encoding
%2 = tlx.require_layout %arg : tensor<128x64xf16, #ttg.blocked<...>>
```

---

## Python Frontend Classes

The following Python classes generate placeholder layouts during lowering:

### DummyTMemLayoutEncoding
```python
class DummyTMemLayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype, unpacked: bool = True):
        self.shape = shape
        self.element_type = element_type
        self.unpacked = unpacked
```

### DummySMemLayoutEncoding
```python
class DummySMemLayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype):
        self.shape = shape
        self.element_type = element_type
```

### DummyRegisterLayoutEncoding
```python
class DummyRegisterLayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype):
        self.shape = shape
        self.element_type = element_type
```

### DummyMMALayoutEncoding
```python
class DummyMMALayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype, operand_a_element_type: tl.dtype):
        self.shape = shape
        self.element_type = element_type
        self.operand_a_element_type = operand_a_element_type
```

### DummyDotOperandLayoutEncoding
```python
class DummyDotOperandLayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype, op_idx: int):
        self.shape = shape
        self.element_type = element_type
        self.op_idx = op_idx  # 0 for operand A, 1 for operand B
```

---

## Resolution Pass

The `TLXResolvePlaceholderLayouts` pass runs after function inlining and resolves all placeholder layouts to concrete layouts.

### Pipeline Location

```python
# In nvidia/backend/compiler.py
passes.common.add_inliner(pm)
tlx.tlx_passes.add_tlx_resolve_placeholder_layouts(pm)  # <-- Runs here
passes.ttir.add_rewrite_tensor_pointer(pm)
```

### Resolution Logic

Each placeholder type has a dedicated resolution function:

| Placeholder | Resolution Function | Key Parameters Used |
|-------------|---------------------|---------------------|
| `DummyTMemLayoutAttr` | `resolveTMemLayout()` | shape, unpacked |
| `DummySMemLayoutAttr` | `resolveSMemLayout()` | shape, elementType, rank |
| `DummyRegisterLayoutAttr` | `resolveRegisterLayout()` | shape, numWarps, threadsPerWarp, numCTAs |
| `DummyMMALayoutAttr` | `resolveMMALayout()` | shape, operandAElementType, numWarps |
| `DummyDotOperandLayoutAttr` | `resolveDotOperandLayout()` | shape, elementType, opIdx, numWarps |

The resolution functions use `ttg::lookupNumWarps()` and similar utilities to obtain the correct context-dependent values after inlining.

---

## TableGen Definitions

The placeholder layout attributes are defined in `TLXAttrDefs.td`:

```tablegen
def TLX_DummyTMemLayoutAttr : TLX_Attr<"DummyTMemLayout", []> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "bool":$unpacked
  );
}

def TLX_DummySMemLayoutAttr : TLX_Attr<"DummySMemLayout", []> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
}

def TLX_DummyRegisterLayoutAttr : TLX_Attr<"DummyRegisterLayout", []> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
}

def TLX_DummyMMALayoutAttr : TLX_Attr<"DummyMMALayout", []> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Type":$operandAElementType
  );
}

def TLX_DummyDotOperandLayoutAttr : TLX_Attr<"DummyDotOperandLayout", []> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "unsigned":$opIdx
  );
}
```

---

## Usage Examples

### Allocating SMEM with Placeholder Layout

```python
@tl.builtin
def local_alloc(shape, dtype, num, storage=tlx.storage_kind.smem, ...):
    if layout is None:
        if storage == tlx.storage_kind.smem:
            layout = tlx.DummySMemLayoutEncoding(unwrapped_shape, dtype)
        else:  # TMEM
            layout = tlx.DummyTMemLayoutEncoding(unwrapped_shape, dtype)
    ...
```

### MMA Operations with Placeholder Layouts

```python
# Hopper path in async_dot
mma_layout = tlx.DummyMMALayoutEncoding(
    list(acc.shape),
    acc.dtype,
    A.dtype,  # operand A element type
)
acc = builder.create_require_layout(acc_handle, mma_layout.to_ir(builder))

if isinstance(A, tl.tensor):
    dot_op_layout = tlx.DummyDotOperandLayoutEncoding(
        list(A.shape),
        A.dtype,
        op_idx=0,
    )
    A_handle = builder.create_require_layout(A.handle, dot_op_layout.to_ir(builder))
```

---

## File Summary

| File | Purpose |
|------|---------|
| `language/tlx/types.py` | Python placeholder layout classes |
| `language/tlx/__init__.py` | Exports placeholder layout classes |
| `dialect/include/IR/TLXAttrDefs.td` | TableGen definitions for placeholder attributes |
| `dialect/triton_tlx.cc` | C++ builder methods for creating placeholder attributes |
| `dialect/lib/Transforms/ResolvePlaceholderLayouts.cpp` | Resolution pass implementation |
| `dialect/include/Transforms/Passes.td` | Pass declaration |
| `nvidia/backend/compiler.py` | Pipeline integration |
