---
name: port-tlx-to-plugin
description: >
  Guide for porting TLX ops (local_alloc, alloc_barriers, async_dot, etc.)
  from the in-tree TLX dialect to out-of-tree Triton extension plugin custom ops.
  Use when migrating TLX functionality to the plugin system defined in
  triton/include/triton/Tools/PluginUtils.h and triton/examples/plugins/.
---

# Porting TLX Ops to Triton Extension Plugins

## Overview

TLX ops are currently implemented in-tree across three layers:

| Layer | Location | Role |
|---|---|---|
| **TableGen** | `third_party/tlx/dialect/include/IR/TLXOps.td` | MLIR op definitions |
| **C++ pybind** | `third_party/tlx/dialect/triton_tlx.cc` | `TritonOpBuilder` method bindings (`create_*`) |
| **Python DSL** | `third_party/tlx/language/tlx/` | User-facing API (`tlx.local_alloc(...)`) |

The Triton plugin system (`triton/include/triton/Tools/PluginUtils.h`) provides
three extension mechanisms that can replace these in-tree layers:

1. **Custom Ops** — `tritonAddPluginCustomOp`: inject op creation via `TritonOpBuilder`
2. **Dialect Plugins** — `tritonGetDialectPluginInfo`: register a full MLIR dialect
3. **Pass Plugins** — `tritonAddPluginPass`: insert custom passes into the pipeline

---

## Plugin API Surface

The plugin is a shared library (`.so`) that exports C functions. The relevant
entry points are defined in `triton/include/triton/Tools/PluginUtils.h`:

```cpp
// Required: enumerate what this plugin provides
TRITON_PLUGIN_API tritonEnumeratePluginCustomOps(uint32_t *count, const char **handles);
TRITON_PLUGIN_API tritonEnumeratePluginDialects(uint32_t *count, const char **names);
TRITON_PLUGIN_API tritonEnumeratePluginPasses(uint32_t *count, const char **names);

// Custom op creation — called from Python via the code generator
TRITON_PLUGIN_API tritonAddPluginCustomOp(const char *handle,
                                          TritonOpBuilder &self,
                                          std::vector<mlir::Value> &operands);

// Dialect registration — called to register types/ops with MLIRContext
TRITON_PLUGIN_API_TYPE(DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char *name);

// Pass insertion — called to add a pass to the PassManager
TRITON_PLUGIN_API tritonAddPluginPass(mlir::PassManager *pm, const char *name,
                                      const std::vector<std::string> &args);
TRITON_PLUGIN_API tritonRegisterPluginPass(const char *name);
```

The plugin is loaded at runtime via `TRITON_PASS_PLUGIN_PATH=path/to/lib.so`.

---

## Architecture Decision: Custom Op vs Dialect Plugin

Choose the approach based on what the TLX op needs:

### Use Custom Ops (simpler) when:
- The op maps directly to existing Triton/TritonGPU/NVGPU ops
- No new MLIR types are needed
- The op is essentially a convenience wrapper (e.g., `alloc_barriers` creates
  a `ttg::LocalAllocOp` + barrier init)

### Use Dialect Plugin (full dialect) when:
- The op requires new MLIR op definitions (TableGen `.td` files)
- New MLIR types are needed (e.g., `!tlx.storage_alias_spec`, `!tlx.reuse_group`)
- Custom verifiers, folders, or canonicalization patterns are needed
- The op needs a dedicated lowering pass to LLVM

### Mixed approach:
Many TLX ops will need **both** — a dialect plugin for the MLIR ops/types, and
custom ops to wire them into the Python code generator.

---

## Step-by-Step: Porting via Custom Ops

This is the simpler path. Suitable for ops like `alloc_barriers`, `local_load`,
`local_store`, `barrier_arrive`, `barrier_wait`, etc. that compose existing
Triton ops.

### Step 1: Identify the pybind implementation

Find the `create_*` method in `third_party/tlx/dialect/triton_tlx.cc`. Example
for `alloc_barriers`:

```cpp
// triton_tlx.cc line ~279
.def("create_alloc_barriers",
     [](TritonOpBuilder &self, int numBarriers, int arriveCount,
        Attribute layout) -> mlir::Value {
       // ... creates LocalAllocOp + BarrierInitOp ...
     })
```

### Step 2: Create the plugin custom op implementation

```cpp
// TLXPlugin.cpp
#include "triton/Tools/PluginUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
// ... other includes as needed ...

// Custom op: alloc_barriers
static TritonPluginResult createAllocBarriers(
    const char *handle, TritonOpBuilder &self,
    std::vector<mlir::Value> &operands) {
  // operands encoding: operands[0] = dst (output slot),
  //                     operands[1..N] = inputs
  // Reproduce the logic from triton_tlx.cc create_alloc_barriers
  // ...
  return TP_SUCCESS;
}

// Registry
static const char *ALLOC_BARRIERS_OP = "tlx_alloc_barriers";
static std::unordered_map<std::string,
    TritonPluginResult(*)(const char*, TritonOpBuilder&, std::vector<mlir::Value>&)>
    customOpMap = {
  {ALLOC_BARRIERS_OP, createAllocBarriers},
};
static std::vector<const char*> customOpNames = {ALLOC_BARRIERS_OP};

TRITON_PLUGIN_API
tritonEnumeratePluginCustomOps(uint32_t *count, const char **handles) {
  if (!count) return TP_GENERIC_FAILURE;
  *count = customOpNames.size();
  if (!handles) return TP_SUCCESS;
  for (unsigned i = 0; i < customOpNames.size(); ++i)
    handles[i] = customOpNames[i];
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonAddPluginCustomOp(const char *handle, TritonOpBuilder &self,
                        std::vector<mlir::Value> &operands) {
  auto it = customOpMap.find(handle);
  if (it == customOpMap.end()) return TP_GENERIC_FAILURE;
  return it->second(handle, self, operands);
}
```

### Step 3: Operand convention

The custom op interface passes operands as a flat `std::vector<mlir::Value>`.
By convention in the existing example (`DialectPluginDialect.cpp:120-128`):

```cpp
// operands[0] = destination (output value, set by the custom op)
// operands[1..N] = input values from Python
::mlir::Value &dst = operands[0];
::mlir::Value &src = operands[1];
dst = self.create<SomeOp>(src, ...);
operands[0] = dst;  // Return result to Python
```

### Step 4: Update the Python DSL

Replace calls to `_semantic.builder.create_*` with calls through the plugin
custom op system. The Python side invokes custom ops through the plugin
infrastructure.

---

## Step-by-Step: Porting via Dialect Plugin

This is needed for ops with custom types (e.g., `storage_alias_spec`,
`reuse_group`) or ops requiring TableGen definitions (e.g., `require_layout`,
`local_alias`).

### Step 1: Create the dialect plugin directory structure

Follow the example at `triton/examples/plugins/DialectPlugins/DialectPlugin/`:

```
MyTLXPlugin/
  CMakeLists.txt
  include/
    CMakeLists.txt
    MyTLXPlugin/
      CMakeLists.txt          # TableGen rules
      MyTLXPluginDialect.td   # Dialect definition
      MyTLXPluginDialect.h
      MyTLXPluginOps.td       # Op definitions (port from TLXOps.td)
      MyTLXPluginOps.h
      MyTLXPluginTypes.td     # Type definitions (port from TLXTypes.td)
      MyTLXPluginTypes.h
      MyTLXPluginPasses.td    # Lowering pass definitions
      MyTLXPluginPasses.h
  lib/
    CMakeLists.txt
    MyTLXPlugin/
      CMakeLists.txt
      MyTLXPluginDialect.cpp  # Dialect init + plugin API exports
      MyTLXPluginOps.cpp
      MyTLXPluginTypes.cpp
      MyTLXPluginPasses.cpp   # Lowering patterns
```

### Step 2: Port the TableGen definitions

Copy from `TLXOps.td`, adapting the dialect reference. Example for
`require_layout`:

```tablegen
// MyTLXPluginOps.td
class MyTLX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyTLXPlugin_Dialect, mnemonic, traits>;

def MyTLX_RequireLayoutOp : MyTLX_Op<"require_layout",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType, Pure]> {
  let summary = "require specific layout for a local memory buffer";
  let arguments = (ins TTG_TensorOrMemDesc:$src);
  let results = (outs TTG_TensorOrMemDesc:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasFolder = 1;
}
```

### Step 3: Port type definitions

For types like `!tlx.storage_alias_spec` and `!tlx.reuse_group`, port from
`TLXTypes.td` and `TLXAttrDefs.td`.

### Step 4: Implement the plugin API exports

In the dialect `.cpp` file, export all required C functions. Follow the pattern
from `DialectPlugins/DialectPlugin/lib/DialectPlugin/DialectPluginDialect.cpp`:

```cpp
// Register dialect
TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *count, const char **names) {
  *count = 1;
  if (!names) return TP_SUCCESS;
  names[0] = "MyTLXPlugin";
  return TP_SUCCESS;
}

TRITON_PLUGIN_API_TYPE(DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char *name) {
  return {MLIR_PLUGIN_API_VERSION, "MyTLXPlugin", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<MyTLXPluginDialect>();
            registerMyTLXPasses();
          }};
}

// Register custom ops that create dialect ops
TRITON_PLUGIN_API
tritonEnumeratePluginCustomOps(uint32_t *count, const char **handles) { ... }

TRITON_PLUGIN_API
tritonAddPluginCustomOp(const char *handle, TritonOpBuilder &self,
                        std::vector<mlir::Value> &operands) { ... }

// Register lowering passes
TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *count, const char **names) { ... }

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *name,
                    const std::vector<std::string> &args) { ... }

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *name) { ... }
```

### Step 5: Create the lowering pass

Port patterns from `third_party/tlx/dialect/lib/Transforms/` to the plugin.
Follow the example in `DialectPluginPasses.cpp` which shows
`ConvertOpToLLVMPattern` usage.

### Step 6: CMakeLists.txt

```cmake
# TableGen (in include/MyTLXPlugin/CMakeLists.txt)
add_mlir_dialect(MyTLXPluginOps mytlx)
set(LLVM_TARGET_DEFINITIONS MyTLXPluginOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
# ... types, passes tablegen ...

# Library (in lib/MyTLXPlugin/CMakeLists.txt)
add_mlir_dialect_library(MLIRMyTLXPlugin
    MyTLXPluginDialect.cpp
    MyTLXPluginOps.cpp
    MyTLXPluginTypes.cpp
    MyTLXPluginPasses.cpp
    SHARED
    DEPENDS TritonIR TritonTableGen ...
    LINK_LIBS PUBLIC MLIRPass LLVMSupport ...)
target_compile_options(MLIRMyTLXPlugin PRIVATE -fvisibility=hidden)
```

---

## TLX Ops Porting Classification

### Custom Ops only (compose existing Triton ops)

| TLX Op | Python DSL | Underlying Triton ops |
|---|---|---|
| `alloc_barriers` | `barrier.py` | `LocalAllocOp` + barrier init |
| `barrier_arrive` | `barrier.py` | `ttng::BarrierArriveOp` |
| `barrier_wait` | `barrier.py` | `ttng::BarrierWaitOp` |
| `barrier_expect_bytes` | `barrier.py` | `ttng::BarrierExpectOp` |
| `named_barrier_arrive` | `barrier.py` | `ttng::NamedBarrierArriveOp` |
| `named_barrier_wait` | `barrier.py` | `ttng::NamedBarrierWaitOp` |
| `cluster_barrier` | `barrier.py` | `ttng::ClusterBarrierOp` |
| `local_alloc` | `mem_ops.py` | `ttg::LocalAllocOp` / `ttng::TMEMAllocOp` |
| `local_load` | `mem_ops.py` | `ttg::LocalLoadOp` |
| `local_store` | `mem_ops.py` | `ttg::LocalStoreOp` |
| `local_view` | `mem_ops.py` | `ttg::MemDescSubviewOp` |
| `local_slice` | `mem_ops.py` | `ttg::MemDescSubsliceOp` |
| `local_trans` | `mem_ops.py` | `ttg::MemDescTransOp` |
| `async_load` | `mem_ops.py` | `ttg::AsyncCopyGlobalToLocalOp` |
| `async_store` | `mem_ops.py` | `ttg::AsyncCopyLocalToGlobalOp` |
| `fence` | `mem_ops.py` | `ttng::FenceAsyncSharedOp` |
| `fence_async_shared` | `mem_ops.py` | `ttng::FenceAsyncSharedOp` |
| `async_descriptor_load` | `mem_ops.py` | `ttng::AsyncTMACopyGlobalToLocalOp` |
| `async_descriptor_store` | `mem_ops.py` | `ttng::AsyncTMACopyLocalToGlobalOp` |
| `async_dot` | `mma_ops.py` | `ttng::WarpGroupDotOp` / `ttng::TCGen5MMAOp` |
| `async_dot_wait` | `mma_ops.py` | `ttng::WarpGroupDotWaitOp` / `ttng::TCGen5MMAWaitOp` |
| `tmem_copy` | `mem_ops.py` | `ttng::TMEMCopyOp` |

### Dialect Plugin required (custom TLX MLIR types/ops)

| TLX Op | Why dialect needed |
|---|---|
| `storage_alias_spec` | Custom type `!tlx.storage_alias_spec` |
| `storage_alias_local_alloc` | References `!tlx.storage_alias_spec` type |
| `reuse_group` | Custom type `!tlx.reuse_group` |
| `set_buffer_overlap` | Uses both custom types |
| `require_layout` | Custom TLX op with folder |
| `release_layout` | Custom TLX op |
| `local_alias` | Custom TLX op |

### Also needs lowering passes ported

| TLX Transform Pass | Source |
|---|---|
| `StorageAliasAllocation` | `lib/Transforms/StorageAliasAllocation.cpp` |
| `StorageAliasSizeDefinition` | `lib/Transforms/StorageAliasSizeDefinition.cpp` |
| `StorageAliasLowering` | `lib/Transforms/StorageAliasLowering.cpp` |
| `BufferOffsetCalculation` | `lib/Transforms/BufferOffsetCalculation.cpp` |
| `PropagateLayout` | `lib/Transforms/PropagateLayout.cpp` |
| `InsertRequireLayout` | `lib/Transforms/InsertRequireLayout.cpp` |
| `ResolvePlaceholderLayouts` | `lib/Transforms/ResolvePlaceholderLayouts.cpp` |
| `RewriteLocalAlias` | `lib/Transforms/RewriteLocalAlias.cpp` |
| `Fixup` | `lib/Transforms/Fixup.cpp` |

---

## Loading and Testing the Plugin

### Build
```bash
# Build with shared libs so the plugin can link against libtriton
export LLVM_BUILD_SHARED_LIBS=1
pip install -e . --no-build-isolation
```

### Load at runtime
```bash
TRITON_PASS_PLUGIN_PATH=/path/to/libMyTLXPlugin.so python my_kernel.py
```

### Insert passes via the pipeline hook
```python
from triton import knobs

def inspect_stages_hook(self=None, stages=None, options=None,
                        language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    def make_ttgir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttgir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        # Add plugin lowering pass for TLX dialect ops
        passes.plugin.add_plugin(pm)
        pm.run(mod, 'tlx_lowering')
        return mod

    stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(
        src, metadata, options, capability)
    return get_key(), get_hash()

knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
```

### Correctness validation
```bash
pytest third_party/tlx/tutorials/testing/test_correctness.py
```

---

## Common Pitfalls

1. **Visibility**: All `TRITON_PLUGIN_API` functions must have default visibility.
   The library itself is compiled with `-fvisibility=hidden`, and the macro
   adds `__attribute__((visibility("default")))`.

2. **TritonOpBuilder::create**: The plugin has direct access to the same
   `TritonOpBuilder` the code generator uses. Call `self.create<OpTy>(...)` to
   emit ops. The builder tracks location automatically.

3. **Operand marshalling**: Custom ops receive a flat `vector<mlir::Value>`.
   You must define and document a convention for each op (which index is
   input vs output). The existing example uses `operands[0]` as the return slot.

4. **Dialect dependencies**: If your plugin creates ops from TritonGPU or
   TritonNvidiaGPU dialects, ensure those are loaded in the MLIRContext.
   The Triton runtime loads them by default, but verify in your plugin init.

5. **Linking**: Link against `MLIRPass`, `LLVMSupport`, and use
   `"$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"` for macOS.
   The plugin resolves symbols from the host `libtriton.so` at load time.

6. **Pass ordering**: When inserting lowering passes via the hook, ensure
   TLX dialect ops are lowered *before* the standard TritonGPU-to-LLVM
   conversion, which will reject unknown ops.
