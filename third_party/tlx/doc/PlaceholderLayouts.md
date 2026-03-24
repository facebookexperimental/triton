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

Right now we have only implemented the placeholder layouts for TMEM dependent layouts, which is the requirement for Flash Attention Backwards.

---

## Overview

The placeholder layout system consists of three components:

1. **Placeholder Layout Attributes**: MLIR attributes that carry shape and type information but defer concrete layout decisions
2. **Python Encoding Classes**: Frontend classes that generate placeholder layout attributes during lowering
3. **Resolution Pass**: A C++ pass that replaces placeholder layouts with concrete layouts after inlining

---

## Placeholder Layout Types

We define one placeholder layout types, organized by memory space and use case:

| Placeholder Type | Memory Space | Resolves To |
|------------------|--------------|-------------|
| `DummyRegisterLayoutAttr` | Registers | `BlockedEncodingAttr` |


### IR Examples

**Before resolution:**
```mlir
// Register tensor with placeholder layout
%0 = tlx.require_layout %arg : tensor<128x64xf16, #tlx.dummy_register_layout<[128, 64], f16>>
```

**After resolution:**
```mlir
// Register resolved to Blocked encoding
%0 = tlx.require_layout %arg : tensor<128x64xf16, #ttg.blocked<...>>
```

---

## Python Frontend Classes

The following Python classes generate placeholder layouts during lowering:

### DummyRegisterLayoutEncoding
```python
class DummyRegisterLayoutEncoding(layout_encoding):
    def __init__(self, shape: List[int], element_type: tl.dtype):
        self.shape = shape
        self.element_type = element_type
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
| `DummyRegisterLayoutAttr` | `resolveRegisterLayout()` | shape, numWarps, threadsPerWarp, numCTAs |

The resolution functions use `ttg::lookupNumWarps()` and similar utilities to obtain the correct context-dependent values after inlining.

---

## TableGen Definitions

The placeholder layout attributes are defined in `TLXAttrDefs.td`:

```tablegen
def TLX_DummyRegisterLayoutAttr : TLX_Attr<"DummyRegisterLayout", []> {
  llet parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "bool":$tmemCompatible
  );
}
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
