# Porting TLX from triton-tlx-amd-meta to TLXPlugin — Concrete Plan

## Source: triton-tlx-amd-meta (Commits 243261c, 55e1aff)

TLX adds ~27,000 lines across three layers:
- **C++ pybind**: `third_party/tlx/dialect/triton_tlx.cc` (875 lines, 50+ `create_*`/`make_*` methods)
- **Python DSL**: `third_party/tlx/language/tlx/` (7 files: mem_ops.py, barrier.py, mma_ops.py, types.py, utility.py, warp_ops.py, async_task_utils.py)
- **TLX MLIR dialect**: `third_party/tlx/dialect/` (TableGen ops, types, transforms — needed only for storage_alias/reuse_group/layout ops)
- **Upstream patches**: Changes to `ir.cc`, `ir.h`, `semantic.py`, `code_generator.py`, `compiler.py`, `knobs.py`, `autotuner.py`, `cache.py`, `setup.py`
- **Compiler extension**: `third_party/tlx/language/tlx/compiler/` (code_generator.py for async_tasks `with` dispatch)

---

## What's Already Ported (Working)

### C++ Plugin (`TLXLocalAllocPlugin.cpp`)
6 custom ops registered: `tlx_local_alloc`, `tlx_local_alloc_tmem`, `tlx_local_view`, `tlx_local_store`, `tlx_local_load`, `tlx_alloc_barriers`

### C++ Pass Plugin (`TLXConversionPatterns.cpp`)
Custom `ConvertTritonToTritonGPU` pass (`tlx_convert_triton_to_tritongpu`) that adds legalization for `ttg::LocalStoreOp`, `ttg::LocalLoadOp`, `ttg::AsyncCopyGlobalToLocalOp`.

### Python DSL (`python/tlx_plugin/`)
- `mem_ops.py`: `local_alloc`, `local_view`, `local_store`, `local_load` (SMEM only)
- `barrier.py`: `alloc_barriers`, `alloc_warp_barrier`
- `types.py`: `buffered_tensor`, `mbarrier`, layout encodings, `storage_kind`
- `utility.py`: `dtype_of`
- `custom_stages.py`: Pipeline hook for AMD (runs plugin conversion, delegates to `self.make_ttgir()`) and NVIDIA (wraps `make_ttir`)
- `__init__.py`: Re-exports + registers `custom_stages.inspect_stages_hook`

### Integration
- Symlink: `python/triton/language/extra/tlx` → `examples/plugins/TLXPlugin/python/tlx`
- `setup.sh` creates the symlink
- `TRITON_PASS_PLUGIN_PATH=python/triton/plugins/libTLXMemOpsPlugin.so` loads the plugin
- `amd-gemm-pipelined.py` runs successfully on gfx950

---

## Porting Plan — Ordered by Priority

### Phase 1: Core Memory Ops (Expand existing)

Already working for basic SMEM. Expand to cover the full `mem_ops.py` API.

#### 1a. `local_trans` — Transpose a memdesc
- **In-tree pybind**: `create_memdesc_trans(src, order)` → `MemDescTransOp`
- **Plugin custom op**: `tlx_local_trans` — operands: `[result, src, *order_dims]`
- **Python**: Add to `mem_ops.py`, return `buffered_tensor.make_permute(handle, dims)`

#### 1b. `local_slice` / `subslice` — Subslice a memdesc
- **In-tree pybind**: `create_memdesc_subslice(src, offsets, sizes, strides)` → `MemDescSubsliceOp`
- **Plugin custom op**: `tlx_local_slice` — operands: `[result, src, *offsets, *sizes, *strides]`
- **Python**: Add to `mem_ops.py`

#### 1c. `local_reinterpret` — Reinterpret memdesc element type
- **In-tree pybind**: `create_memdesc_reinterpret(src, newElemType, newShape)` → Constructs new MemDescType
- **Plugin custom op**: `tlx_local_reinterpret`
- **Python**: Add to `mem_ops.py`

#### 1d. `local_alloc` with TMEM support
- **In-tree pybind**: `create_tmem_alloc(shape, elemType, layout)` → `TMEMAllocOp`
- **Plugin custom op**: `tlx_local_alloc_tmem` (already registered but needs Python wiring)
- **Python**: Extend `local_alloc()` to dispatch to `tlx_local_alloc_tmem` when `storage == tmem`
- **Note**: Requires Blackwell (sm_100+). TMEM layout encodings need `make_tensor_memory_encoding_attr`.

### Phase 2: Async Operations

#### 2a. `async_load` — Async copy global to local
- **In-tree pybind**: `create_async_load(src_ptrs, dst_memdesc, mask, other, cache, eviction, is_volatile)` → `AsyncCopyGlobalToLocalOp`
- **Plugin custom op**: `tlx_async_load`
- **Python**: Add to `mem_ops.py`

#### 2b. `async_store` — Async copy local to global
- **In-tree pybind**: `create_async_store(src_memdesc, dst_ptrs, mask)` → `AsyncCopyLocalToGlobalOp`
- **Plugin custom op**: `tlx_async_store`

#### 2c. `async_load_commit_group` / `async_load_wait_group`
- **In-tree pybind**: `create_async_commit_group()` / `create_async_wait(num)`
- **Plugin custom ops**: `tlx_async_commit_group`, `tlx_async_wait`

#### 2d. `fence` / `fence_async_shared`
- **In-tree pybind**: `create_fence_async_shared(has_async_proxy)` → `FenceAsyncSharedOp`
- **Plugin custom op**: `tlx_fence_async_shared`

### Phase 3: Barrier Operations (Expand existing)

#### 3a. `barrier_arrive`, `barrier_wait`, `barrier_expect_bytes`
- **In-tree pybind**: `create_barrier_arrive(bar, count)`, `create_barrier_wait(bar, phase, pred)`, `create_barrier_expect(bar, size, pred)`
- **Plugin custom ops**: `tlx_barrier_arrive`, `tlx_barrier_wait`, `tlx_barrier_expect`
- **Python**: Add to `barrier.py`
- **Note**: `barrier_arrive` has remote_cta_rank support that depends on `remote_view`

#### 3b. `named_barrier_arrive` / `named_barrier_wait`
- **In-tree pybind**: `create_named_barrier_arrive/wait(bar, count)`
- **Plugin custom ops**: `tlx_named_barrier_arrive`, `tlx_named_barrier_wait`

#### 3c. `cluster_barrier`
- **In-tree pybind**: `create_cluster_barrier()` → `ClusterBarrierOp`
- **Plugin custom op**: `tlx_cluster_barrier`

### Phase 4: MMA Operations (NVIDIA-specific)

#### 4a. `async_dot` — Warp-group / tcgen05 MMA
- **In-tree pybind**: `create_warp_group_dot(A, B, C, precision, max_imprecise, isAsync)` and `create_tcgen5_dot(A, B, C, use_acc, pred, two_ctas, barriers, isAsync)`
- **Dependencies**: `require_layout` (TLX dialect op), `make_nv_mma_encoding_attr`, `make_dot_operand_encoding_attr`
- **Complexity**: HIGH — needs `require_layout` and `release_layout` ops which are TLX dialect ops
- **Approach**: Either (a) register `require_layout`/`release_layout` as custom ops that create `ttg.convert_layout`, or (b) use a TLX Dialect Plugin for these ops

#### 4b. `async_dot_wait`
- **In-tree pybind**: `create_warp_group_dot_wait(inputs, pendings)`

#### 4c. `async_dot_scaled` — Blackwell scaled MMA
- **In-tree pybind**: `create_tcgen5_dot_scaled(A, B, C, A_scale, B_scale, A_type, B_type, ...)`

#### 4d. `tcgen05_commit`
- **In-tree pybind**: `create_tcgen05_commit(mbar, pred)`

### Phase 5: TMA / Descriptor Operations

#### 5a. `async_descriptor_load` / `async_descriptor_store`
- **In-tree pybind**: `create_async_TMA_load/store(...)` → `AsyncTMACopyGlobalToLocalOp`
- **Dependencies**: Tensor descriptors, barriers
- **Plugin custom ops**: `tlx_async_tma_load`, `tlx_async_tma_store`

#### 5b. `make_tensor_descriptor` / `allocate_tensor_descriptor`
- **In-tree pybind**: `create_make_tensor_descriptor(...)`, `create_global_scratch_alloc(...)`
- **Python types**: `tensor_descriptor_ptr`, `tensor_descriptor_ptr_type`

### Phase 6: TLX Dialect Ops (Requires Dialect Plugin)

These ops use custom TLX MLIR types (`!tlx.storage_alias_spec`, `!tlx.reuse_group`) and cannot be implemented as pure custom ops.

#### 6a. `storage_alias_spec` / `storage_alias_local_alloc` / `set_buffer_overlap`
- **MLIR ops**: `TLX_StorageAliasSpecOp`, `TLX_StorageAliasLocalAllocOp`, `TLX_SetBufferOverlapOp`
- **MLIR types**: `!tlx.storage_alias_spec<smem>`, `!tlx.reuse_group<shared|distinct>`
- **Lowering passes needed**: `StorageAliasAllocation`, `StorageAliasSizeDefinition`, `StorageAliasLowering`, `BufferOffsetCalculation`

#### 6b. `reuse_group`
- **MLIR op**: `TLX_ReuseGroupOp`
- **In-tree pybind**: `create_reuse_group(elements, group_kind, group_size)`

#### 6c. `require_layout` / `release_layout` / `local_alias`
- **MLIR ops**: `TLX_RequireLayoutOp` (with folder), `TLX_ReleaseLayoutOp`, `TLX_LocalAliasOp`
- **Lowering passes**: `PropagateLayout`, `InsertRequireLayout`, `ResolvePlaceholderLayouts`, `RewriteLocalAlias`, `Fixup`

### Phase 7: Warp Specialization / Async Tasks

#### 7a. `async_task` / `async_tasks` — Warp specialization context manager
- **Python**: `compiler/code_generator.py` — complex `visit_With` dispatch, `visit_withAsyncTasks`
- **In-tree pybind**: `create_warp_specialize_op(...)`, `create_warp_yield_op()`, `create_warp_return_op()`
- **Upstream dependency**: Patches to `code_generator.py` (`WITH_DISPATCH`, `visit_With` rewrite, `used_vars` tracking, `enter_sub_region` changes)
- **Complexity**: VERY HIGH — requires upstream `code_generator.py` changes

### Phase 8: Utility & Misc

#### 8a. `remote_view` / `remote_shmem_store` / `async_remote_shmem_store`
- **In-tree pybind**: `create_map_to_remote_buffer(src, cta_rank)`, `create_remote_store(...)`, `create_async_remote_store(...)`

#### 8b. Utility functions
- `cluster_cta_rank`, `cluster_size_1d`, `thread_id` — `create_cluster_cta_rank()`, etc.
- `clock64` — `create_clock64()`
- `stoch_round` — `create_cvt_rs(src, dst_type, rbits)` (Blackwell only)
- `size_of`, `get_fp8_format_name` — pure Python, no IR

#### 8c. `tmem_copy`
- **In-tree pybind**: `create_tmem_copy(src, dst)` → `TMEMCopyOp`

---

## Upstream Triton Dependencies

These are changes TLX makes to upstream Triton code. For the plugin approach,
some can be avoided, others need workarounds:

### Can be avoided (plugin handles these differently):
- `ir.cc`/`ir.h`: `getBuilderClass()` — needed for in-tree pybind `.def()` on builder. Plugin uses `tritonAddPluginCustomOp` instead.
- `cache.py`: TLX path hashing — plugin uses `inspect_stages_hook` key/hash for cache invalidation.
- `setup.py`: TLX build integration — plugin builds separately via CMakeLists.txt.

### Need workarounds in the plugin:
- `semantic.py` `dot_precheck()` refactor — TLX `async_dot` calls `dot_precheck()` which doesn't exist in upstream. **Workaround**: Duplicate the validation logic in the plugin's `mma_ops.py`.
- `semantic.py` `tl.tensor(...)` vs `self.tensor(...)` — TLX changes `load()` to use `tl.tensor()` directly. Plugin code already does this.
- `code_generator.py` `WITH_DISPATCH` + `visit_With` rewrite — needed for `async_tasks`. **Workaround**: For Phase 7, either (a) submit upstream PR for the dispatch mechanism, or (b) monkey-patch `CodeGenerator.visit_With` in the plugin's `__init__.py`.
- `code_generator.py` `used_vars` tracking — needed for warp specialization capture analysis. Same upstream dependency as above.
- `knobs.py` `use_meta_ws`, `dump_ttgir_to_tlx`, `use_ptx_loc` — Optional knobs, can be skipped or added to plugin config.

### Upstream ops added by TLX commits (in `TritonGPUOps.td`, `TritonNvidiaGPUOps.td`):
- `ttg::MemDescSubsliceOp` (51 lines added to TritonGPUOps.td)
- `ttng::AsyncRemoteShmemStoreOp`, `ttng::WarpSpecializeOp`, `ttng::WarpPartitionOp`, `ttng::WarpYieldOp`, `ttng::WarpReturnOp`, `ttng::PruneUnusedBarriersPass` (193 lines to TritonNvidiaGPUOps.td)
- These ops are likely already in upstream Triton (since they're in TritonGPU/TritonNvidiaGPU, not TLX dialect). Verify before porting.

---

## Key Architecture Decisions

### 1. Custom Op vs Dialect Plugin boundary
- **Custom Ops**: Everything that composes existing `ttg::` / `ttng::` ops (Phases 1-5, 8)
- **Dialect Plugin**: Only for TLX-specific MLIR types and ops (Phase 6): `storage_alias_spec`, `reuse_group`, `require_layout`, `release_layout`, `local_alias`

### 2. Pybind method mapping
In-tree TLX uses `builder.create_*()` pybind methods. Plugin uses `builder.tlx_*(args_list)` custom op calls. Each custom op receives a flat `vector<mlir::Value>` with convention:
- `operands[0]` = result slot (set by C++ plugin, returned to Python)
- `operands[1..N]` = input values

For make_*/get_* helpers (layout attrs, types), these are builder methods that don't create ops — they create MLIR attributes/types. Plugin can either:
- (a) Encode layout params as i32 constants in operands (current approach for `local_alloc`)
- (b) Add custom op helpers that return attributes as attached metadata

### 3. Pipeline hook architecture
The `custom_stages.py` hook is the right pattern:
- **AMD**: Replace `stages["ttgir"]` — run plugin conversion first, then `self.make_ttgir()`
- **NVIDIA**: Replace `stages["ttir"]` — run `self.make_ttir()` then plugin conversion
- **Dialect Plugin passes**: Add to the hook after conversion, before LLVM lowering

### 4. Python DSL approach
Keep the existing `@tl.builtin` + `_semantic.builder.tlx_*()` pattern. Each new op:
1. Add C++ custom op in `TLXLocalAllocPlugin.cpp`
2. Add Python wrapper in appropriate `tlx_plugin/*.py` file
3. Export from `__init__.py`

---

## File Mapping Reference

| In-tree TLX | Plugin equivalent |
|---|---|
| `triton_tlx.cc` `create_*()` | `TLXLocalAllocPlugin.cpp` custom ops |
| `triton_tlx.cc` `make_*()` | Encoded in operands or separate custom ops |
| `language/tlx/mem_ops.py` | `python/tlx_plugin/mem_ops.py` |
| `language/tlx/barrier.py` | `python/tlx_plugin/barrier.py` |
| `language/tlx/mma_ops.py` | `python/tlx_plugin/mma_ops.py` (new) |
| `language/tlx/types.py` | `python/tlx_plugin/types.py` |
| `language/tlx/utility.py` | `python/tlx_plugin/utility.py` |
| `language/tlx/warp_ops.py` | `python/tlx_plugin/warp_ops.py` (new) |
| `language/tlx/compiler/` | `python/tlx_plugin/compiler/` (new, Phase 7) |
| `dialect/include/IR/TLXOps.td` | Dialect Plugin TableGen (Phase 6) |
| `dialect/lib/Transforms/*.cpp` | Dialect Plugin passes (Phase 6) |
| Backend `compiler.py` patches | `python/tlx_plugin/custom_stages.py` |

---

## In-tree Pybind Methods → Plugin Custom Op Mapping

Full list of `create_*` methods in `triton_tlx.cc` and their plugin equivalents:

### Already ported:
| Pybind method | Plugin custom op | Status |
|---|---|---|
| N/A (composite) | `tlx_local_alloc` | Done |
| N/A (composite) | `tlx_local_alloc_tmem` | Registered, no Python |
| `create_memdesc_subview` | `tlx_local_view` | Done |
| `create_local_store` | `tlx_local_store` | Done |
| `create_local_load` | `tlx_local_load` | Done |
| `create_alloc_barriers` | `tlx_alloc_barriers` | Done |

### To port (by phase):
| Pybind method | Plugin custom op | Phase |
|---|---|---|
| `create_memdesc_trans` | `tlx_local_trans` | 1a |
| `create_memdesc_subslice` | `tlx_local_slice` | 1b |
| `create_memdesc_reinterpret` | `tlx_local_reinterpret` | 1c |
| `create_async_load` | `tlx_async_load` | 2a |
| `create_async_store` | `tlx_async_store` | 2b |
| `create_async_commit_group` | `tlx_async_commit_group` | 2c |
| `create_async_wait` | `tlx_async_wait` | 2c |
| `create_fence_async_shared` | `tlx_fence_async_shared` | 2d |
| `create_barrier_arrive` | `tlx_barrier_arrive` | 3a |
| `create_warp_barrier_arrive` | `tlx_warp_barrier_arrive` | 3a |
| `create_barrier_wait` | `tlx_barrier_wait` | 3a |
| `create_barrier_expect` | `tlx_barrier_expect` | 3a |
| `create_named_barrier_arrive` | `tlx_named_barrier_arrive` | 3b |
| `create_named_barrier_wait` | `tlx_named_barrier_wait` | 3b |
| `create_cluster_barrier` | `tlx_cluster_barrier` | 3c |
| `create_warp_group_dot` | `tlx_warp_group_dot` | 4a |
| `create_tcgen5_dot` | `tlx_tcgen5_dot` | 4a |
| `create_require_layout` | `tlx_require_layout` | 4a/6c |
| `create_release_layout` | `tlx_release_layout` | 4a/6c |
| `create_warp_group_dot_wait` | `tlx_dot_wait` | 4b |
| `create_tcgen5_dot_scaled` | `tlx_tcgen5_dot_scaled` | 4c |
| `create_tcgen05_commit` | `tlx_tcgen05_commit` | 4d |
| `create_async_TMA_load` | `tlx_async_tma_load` | 5a |
| `create_async_TMA_store` | `tlx_async_tma_store` | 5a |
| `create_async_TMA_prefetch` | `tlx_async_tma_prefetch` | 5a |
| `create_async_TMA_store_wait` | `tlx_async_tma_store_wait` | 5a |
| `create_make_tensor_descriptor` | `tlx_make_tensor_desc` | 5b |
| `create_global_scratch_alloc` | `tlx_global_scratch_alloc` | 5b |
| `create_storage_alias_spec` | Dialect Plugin | 6a |
| `create_reuse_group` | Dialect Plugin | 6b |
| `create_set_buffer_overlap` | Dialect Plugin | 6a |
| `create_warp_specialize_op` | `tlx_warp_specialize` | 7a |
| `create_warp_yield_op` | `tlx_warp_yield` | 7a |
| `create_warp_return_op` | `tlx_warp_return` | 7a |
| `create_map_to_remote_buffer` | `tlx_remote_view` | 8a |
| `create_remote_store` | `tlx_remote_store` | 8a |
| `create_async_remote_store` | `tlx_async_remote_store` | 8a |
| `create_tmem_copy` | `tlx_tmem_copy` | 8c |
| `create_tmem_load` | `tlx_tmem_load` | 8c |
| `create_tmem_store` | `tlx_tmem_store` | 8c |
| `create_tmem_subslice` | `tlx_tmem_subslice` | 8c |
| `create_clock64` | `tlx_clock64` | 8b |
| `create_thread_id` | `tlx_thread_id` | 8b |
| `create_cvt_rs` | `tlx_cvt_rs` | 8b |
| `create_cluster_cta_rank` | `tlx_cluster_cta_rank` | 8b |
| `create_cluster_size_1d` | `tlx_cluster_size_1d` | 8b |

### Builder helpers (make_*/get_*) — not ops, create MLIR attributes:
| Pybind method | Plugin approach |
|---|---|
| `make_swizzled_shared_encoding_attr` | Encode params in operands |
| `make_tensor_memory_encoding_attr` | Encode params in operands |
| `make_tensor_memory_scales_encoding_attr` | Encode params in operands |
| `make_nv_mma_shared_encoding_attr` | Encode params in operands |
| `make_nv_mma_encoding_attr` | Encode params in operands |
| `make_dot_operand_encoding_attr` | Encode params in operands |
| `make_dummy_register_layout_attr` | Encode params in operands |
| `make_dummy_tmem_layout_attr` | Encode params in operands |
| `get_memdesc_type` | Encode params in operands |
| `get_storage_alias_spec_type` | Dialect Plugin |
