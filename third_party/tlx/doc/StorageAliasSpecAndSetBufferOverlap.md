# TLX `storage_alias_spec` and `set_buffer_overlap` Design

**Author:** Nick Riasanovsky
**Updated:** 2026-02-12

---

## Background

In Blackwell kernels there is often a need to share buffer memory between multiple allocations to allow sufficiently large block sizes for performance. Previously, this was done through the `local_alloc` API via the `reuse` parameter, which accepted an existing `buffered_tensor`. This approach required manual memory management — users had to calculate buffer counts, padding, and offsets themselves — which led to several problems:

1. **Error-prone indexing**: Users had to specify an exact number of buffers to get sufficient isolation. When anything changed (e.g. data type, block size), the number of buffers and their overlap relationships changed. Users had to manually update all index calculations, which was a source of subtle bugs.
2. **Implicit primary ownership**: The original `reuse` API made one allocation the "primary owner" of the buffer. All other allocations had to be smaller, creating asymmetry and requiring careful ordering.
3. **Autotuning limitations**: Due to issue 1, it can be difficult to exhaustively autotune, likely leaving performance on the table.

### Motivating Example

In Flash Attention, `qk_tiles` and `p_tiles` need to share the same underlying memory. With the old API, the user had to manually compute the correct number of buffers for `p_tiles` based on the data type ratio (e.g., `NUM_BUFFERS_QK * 2` for BF16 because `sizeof(float32) / sizeof(bfloat16) == 2`). If the data type changed to FP8, the multiplier would change to 4, and all downstream index logic would need to be updated.

---

## Frontend API

The user-facing API — `storage_alias_spec`, `local_alloc(reuse=...)`,
`reuse_group`, `set_buffer_overlap` — and the Flash Attention usage example
are documented in
[TLX &rsaquo; Buffer Reuse](../../../docs/tlx.md#buffer-reuse).
That is the reference; this document covers the background above and the
IR, pass pipeline, and design rationale below.

---

## IR Operations

The frontend lowers the Python API into four MLIR operations defined in the TLX dialect.

### `tlx.storage_alias_spec`

Creates a storage alias specification. Does not allocate memory itself; it defines a logical grouping for buffer sharing.

**Arguments:** `storage` (smem or tmem), optional `buffer_size_bytes`, optional `buffer_shape` (set by compiler).

**Result:** `!tlx.storage_alias_spec<storage[, size]>`

**Defined in:** `dialect/include/IR/TLXOps.td` (`TLX_StorageAliasSpecOp`)

### `tlx.storage_alias_local_alloc`

An intermediate allocation operation produced when `local_alloc` is called with a `storage_alias_spec`. It references the spec and produces a `!ttg.memdesc` result. After the storage alias lowering pass, this operation is replaced with a `tlx.local_alias` pointing to a standard allocation.

**Arguments:** `storage_alias` (`!tlx.storage_alias_spec`)

**Result:** `!ttg.memdesc<...>`

**Defined in:** `dialect/include/IR/TLXOps.td` (`TLX_StorageAliasLocalAllocOp`)

### `tlx.reuse_group`

Creates a reuse group tree node. Accepts a variadic list of elements (either `!ttg.memdesc` or `!tlx.reuse_group`) and produces a `!tlx.reuse_group<kind>` result.

**Arguments:** `elements` (variadic), `group_kind` (shared or distinct), `group_size` (default 1).

**Result:** `!tlx.reuse_group<kind>`

**Defined in:** `dialect/include/IR/TLXOps.td` (`TLX_ReuseGroupOp`)

### `set_buffer_overlap`

Links a `storage_alias_spec` to its overlap definition (a `reuse_group`). This operation is consumed and erased during the buffer offset calculation pass.

**Arguments:** `storage_alias_spec`, `overlap_def` (`!tlx.reuse_group`)

**Defined in:** `dialect/include/IR/TLXOps.td` (`TLX_SetBufferOverlapOp`)

### `tlx.local_alias`

Creates an alias of a local memory buffer with a different view (shape, element type, or encoding). Produced during the storage alias allocation pass when lowering `StorageAliasLocalAllocOp`. This is the final form — each `local_alias` points to the single backing allocation created for the `storage_alias_spec`.

**Defined in:** `dialect/include/IR/TLXOps.td` (`TLX_LocalAliasOp`)

### Types

Two custom MLIR types support the operations:

- **`!tlx.storage_alias_spec<storage[, size]>`**: Carries the storage kind and optional explicit size. Defined in `dialect/include/IR/TLXTypes.td`.
- **`!tlx.reuse_group<kind>`**: Carries the group kind (shared or distinct). Defined in `dialect/include/IR/TLXTypes.td`.

---

## Compiler Pass Pipeline

The storage alias lowering is orchestrated by a single combined pass (`TLXStorageAliasLoweringPass`) that executes three steps sequentially. The ordering is critical: size definition must precede offset calculation, and offset calculation must precede allocation materialization (because materialization erases the ops that the earlier steps depend on).

### Step 1: Storage Alias Size Definition

**Purpose:** Compute or validate the buffer size for each `storage_alias_spec`.

**Logic:**
- Collects all `StorageAliasLocalAllocOp` operations and groups them by their referenced `storage_alias_spec`.
- For **SMEM**: If a `SetBufferOverlapOp` exists, the reuse group tree is walked to compute the size per buffer. The tree semantics are: `shared` → max of children (multiplied by `group_size`), `distinct` → sum of children. Otherwise, the size is the maximum across all referencing allocations.
- For **TMEM**: Computes a 2D shape (blockM × blockN) based on the maximum dimensions across all users, with scaling for element size relative to i32 (4 bytes). blockM is constrained to 64 or 128 for TMEM hardware requirements.
- If `buffer_size_bytes` was explicitly set by the user, validates that it is large enough. Otherwise, sets it to the computed value.
- Sets the `buffer_shape` attribute on the `StorageAliasSpecOp` for use by subsequent passes.

**Defined in:** `dialect/lib/Transforms/StorageAliasSizeDefinition.cpp`

### Step 2: Buffer Offset Calculation

**Purpose:** Compute the memory offset for each allocation based on the reuse group tree defined by `set_buffer_overlap`.

**Logic:**
- Collects all `SetBufferOverlapOp` operations.
- For each, recursively walks the reuse group tree starting at offset 0:
  - **`shared`**: All children start at the same offset. The `bytesBetweenBufferGroups` is divided by `group_size` for subtiling, and the effective `group_size` is multiplied down to children.
  - **`distinct`**: Children are placed sequentially — each child's offset is the previous child's offset plus its size. Validates that the total does not exceed available space.
- Produces an `offsetMap` mapping each `StorageAliasLocalAllocOp` result to a tuple of `(buffer_offset, bytes_between_buffer_groups, group_size)`.
- Erases the `SetBufferOverlapOp` and cleans up unused `ReuseGroupOp` operations.

**Defined in:** `dialect/lib/Transforms/BufferOffsetCalculation.cpp`

### Step 3: Storage Alias Allocation

**Purpose:** Materialize the actual memory allocations and replace intermediate ops with standard TritonGPU IR.

**Logic:**
1. **Create backing allocations**: For each `StorageAliasSpecOp`, creates a single `LocalAllocOp` (SMEM, 1D byte buffer) or `TMEMAllocOp` (TMEM, 2D i32 buffer) with the computed shape.
2. **Replace intermediate ops**: Each `StorageAliasLocalAllocOp` is replaced with a `LocalAliasOp` pointing to the backing allocation. If offset information exists from Step 2, the alias type's shape may be expanded to accommodate the offset/scale transformations.
3. **Rewrite index operations**: When an allocation has non-trivial offsets (from `set_buffer_overlap`), all `MemDescIndexOp` users are rewritten with the transformation: `newIndex = scaleFactor * originalIndex + offsetSlots + (originalIndex % groupSize)`. This correctly maps logical buffer indices to physical positions in the expanded buffer, accounting for both offset placement and subtiling.
4. **Clean up**: Erases all `StorageAliasSpecOp` operations.

The pass also handles propagation through `MemDescReinterpretOp`, nested `LocalAliasOp`, and `WarpSpecializeOp` captures (updating block argument types in partition regions when the aliased type changes).

**Defined in:** `dialect/lib/Transforms/StorageAliasAllocation.cpp`

### Orchestration

**Defined in:** `dialect/lib/Transforms/StorageAliasLowering.cpp`

The `TLXStorageAliasLoweringPass` calls the three steps in order, failing the pass if any step returns an error.

---

## Compiler Safety Guarantees

A key goal of this design is to produce **static compilation errors** when the overlap scheme cannot be achieved, rather than silently generating incorrect kernels:

- **Size validation**: If `buffer_size_bytes` is explicitly specified and is too small for the computed requirements, the compiler emits an error.
- **Distinct group overflow**: If the children of a `distinct` group require more space than is available within `bytesBetweenBufferGroups`, the compiler emits an error.
- **Offset alignment**: If `buffer_offset` or `bytes_between_buffer_groups` is not a multiple of the per-buffer allocation size, the compiler emits an error.
- **Duplicate overlap definitions**: If `set_buffer_overlap` is called more than once on the same spec, the compiler emits an error.
- **Unused specs**: If a `storage_alias_spec` has no referencing allocations, the compiler emits a warning.

---

## User Fallback Mechanisms

To ensure users always have an escape hatch when the higher-level API is insufficient:

1. **Explicit `buffer_size_bytes`**: The user can specify a size larger than what the compiler would compute, allowing for custom padding or more complex sharing schemes beyond what the reuse group tree can express.
2. **No `set_buffer_overlap`**: If a `storage_alias_spec` is used without calling `set_buffer_overlap`, all allocations start at offset 0 with no inter-allocation padding. The user can then use buffer count manipulation for manual layout control.

---

## File Summary

| File | Role |
|------|------|
| `language/tlx/mem_ops.py` | `storage_alias_spec()` builtin function and updated `local_alloc()` |
| `language/tlx/types.py` | `storage_alias_spec` class, `storage_alias_spec_type`, `reuse_group` class, `reuse_group_type` enum, `reuse_group_ir_type` |
| `language/tlx/__init__.py` | Public exports for the API |
| `dialect/include/IR/TLXOps.td` | MLIR op definitions: `StorageAliasSpecOp`, `StorageAliasLocalAllocOp`, `ReuseGroupOp`, `SetBufferOverlapOp`, `LocalAliasOp` |
| `dialect/include/IR/TLXTypes.td` | MLIR type definitions: `StorageAliasSpecType`, `ReuseGroupType`, `StorageKindAttr`, `ReuseGroupKindAttr` |
| `dialect/triton_tlx.cc` | Python-to-IR bindings: `create_storage_alias_spec()`, `create_set_buffer_overlap()`, `create_reuse_group()` |
| `dialect/lib/Transforms/StorageAliasSizeDefinition.cpp` | Pass Step 1: Compute/validate buffer sizes |
| `dialect/lib/Transforms/BufferOffsetCalculation.cpp` | Pass Step 2: Compute offsets from reuse group tree |
| `dialect/lib/Transforms/StorageAliasAllocation.cpp` | Pass Step 3: Materialize allocations, replace ops, rewrite indices |
| `dialect/lib/Transforms/StorageAliasLowering.cpp` | Combined pass orchestration |
| `test/TLX/buffer-offset-calculation.mlir` | MLIR-level tests for the offset calculation pass |
| `python/test/unit/language/test_tlx_storage_alias.py` | Python unit tests for the storage alias frontend API and end-to-end compilation |
| `tutorials/blackwell_fa_ws_pipelined_persistent.py` | Real-world usage example in Flash Attention |

---

## Future Work

While not covered in the original work, there are several additional opportunities for improvement.

### Eliminating `set_buffer_overlap`

We can modify the code implementation to eliminate the need for applying the method for each spec. Fundamentally
the presence of a `reuse_group` is enough to enforce a relationship and the compiler could just collect the "largest"
reuse for enforcement. This will allow us to eliminate compiler changes and simplify user code.

### Under-Utilization Warning

Currently we don't offer the user insights if they are unnecessarily buffer sharing. For example, with HEAD-DIM=64
in FA a user might opt not to share all of QK and P, alpha, l, and m since there are 64 columns of leftover TMEM.
We could write a compiler pass that suggests either removing sharing with P or (alpha/l/m) to maximize available
TMEM.

### BufferedTensor Reuse Deprecation

We should deprecate the old use of `reuse` in `local_alloc` and require `storage_alias_spec` for clearer ownership
semantics as sizes change. This will require ensuring the `storage_alias_spec` implementation is well tested across
many kernels.

### Explicit Buffer Lowering (no reindexing)

Right now we don't lower directly to LLVM with an updated base pointer/stride due to potential implications on linear layouts.
However, this fundamentally makes some reuses impossible to represent and may cause CUDA core utilization that could otherwise be
avoided during the reindexing.

If we encounter cases where we cannot represent the reuse we should consider the explicit lowering approach and investigate if
there is actually a real linear layout concern with multi-buffering.

#### Moving Layout Alignment

With an explicit buffer offset, additional alignment becomes available. For example, it's possible that one
layout which would be optimal for Buffer A requires 256 byte alignment and is shared with Buffer B that
desires a 128 byte alignment. Currently the only way this could be achieved is if the single allocation is
256 byte aligned, which may not always be possible. However, in theory you could just have A start 128 later
than the original offset. Additionally if there is an external requirement (e.g. TMA requires 128 byte alignment)
and the buffer size is less than the alignment, explicit padding in the lowering would be needed to maintain the
128 byte alignment.

It is unclear how critical this is at this time, but this is an avenue of analysis that becomes available once
we have the lowering capability.

### Reuse Groups for Kernel Fusion

In the abstract kernel fusion case, it's likely that greater buffer reuse will be necessary, potentially in the extreme
requiring allocation of a single buffer and then aliasing it entirely. In that situation, it's possible a kernel
will have buffers with differing liveness (e.g. live in for-loop 1 but not for-loop 2).

While in theory this may be expressible as a very complicated reuse group, we may want to explore allowing
`reuse_group` to be applied multiple times and then require that they either have distinct liveness ranges
or that any buffers used in both have their conditions fixed across both groups (e.g. anchors).

### Synchronization Analysis

This is very difficult and most likely not sufficient to capture all bugs, but it may
be possible to perform static analysis across many more synchronization issues with
the implicit "metadata" information from reuse groups. Here is the high-level logic
with a simple example: Imagine we have a reuse group that marks A and B as shared.
Then based on the compiler guarantees we know that it is never safe to access A[i]
without a guarantee B[i] is no longer live.

Now this is still very difficult because the code is warp specialized, making it more
challenging to determine the dependency graph, and the boundaries are barriers, which
may be possible to fuse together. However, the reuse groups could act as the first of many "metadata infusing operations" which collectively
may make this possible.
