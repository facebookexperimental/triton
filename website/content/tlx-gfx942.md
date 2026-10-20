# TLX on gfx942 (AMD MI300 / CDNA3)

MI300X: CDNA3 MFMA, per-CU LDS, explicit buffer loads. Unlike gfx950,
async global-to-LDS copies are not profitable here, so kernels stage
through registers with direct buffer loads plus explicit LDS. Tags follow
[tlx.md](tlx.html); a trailing `?` marks unconfirmed availability.

## Supported `tlx.ops`

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `mm` | direct_load | `mm/gfx942.py:mm` | float16 | — | no | Direct-load GEMM: register-staged buffer loads + explicit LDS, hand-pinned MFMA layouts; intra-CTA K-split for short-M; N-edge panels share one A tile. No descriptor alignment constraints. |
| `addmm` | fused_gemm | `addmm/gfx942.py:addmm` | float16 | — | no | Thin fused-bias wrapper over the `mm` direct-load path: one kernel does GEMM+bias. |
| `mm_torchtlx` | inductor_gfx942_mm | `inductor/gfx942_torch:mm` | float16 | — | no | Inductor template entry so the gfx942 GEMM participates in PT2 template selection, autotuning, and fusion. |
| `addmm_torchtlx` | inductor_gfx942_addmm | `addmm/gfx942_torch:addmm` | float16 | — | no | Inductor template entry for fused GEMM+bias on gfx942. |

## Supported extension builtins (`tlx.*`)

Same cross-backend set as [gfx950](tlx-gfx950.html), minus the
gfx950-only items below.

### Scalars, types, diagnostics

| Builtin | Purpose |
|---|---|
| `assume_uniform` | Assert scalar wave-uniform (`v_readfirstlane`) |
| `zeros` | Zero tensor, optional explicit register layout |
| `dtype_of` / `size_of` | Element type / byte size (host-side) |
| `get_fp8_format_name` | `e5m2`/`e4m3` for scaled-MMA format args |
| `clock64` | 64-bit hardware timer read |
| `assert_same_layout` / `dump_layout` | Layout assert / compile-time layout log |

### Local memory

| Builtin | Purpose |
|---|---|
| `local_alloc` / `local_view` / `local_slice` | Multi-buffer alloc, index subview, same-rank slice |
| `local_store` / `local_load` | Registers to/from LDS buffer |
| `local_gather` / `local_scatter` | Indexed gather/scatter along an axis |
| `storage_alias_spec` | Sized/unsized shared-buffer alias spec |
| `buffered_tensor` / `buffered_tensor_type` | Memory-backed buffer value and its type |

### AMD buffer ops

Register-staged loads are the primary path on this target
(`mm/gfx942.py` uses `buffer_load` for all operand traffic).

| Builtin | Purpose |
|---|---|
| `buffer_load` / `buffer_store` | Global access via scalar base + i32 offsets |
| `buffer_atomic_add`? | Buffer atomic add; unconfirmed on gfx942 |

### Layout control

| Builtin | Purpose |
|---|---|
| `local_trans` / `local_reshape` / `local_reinterpret` | Metadata-only descriptor transforms |
| `require_layout` / `release_layout` | Pin / release register layouts |
| `amd_mfma_layout` | MFMA distributed layout, versions 1-4 |
| `dot_operand_layout` | Explicit MFMA dot-operand view |
| `require_amd_wmma_layout` | Pin tensor to explicit WMMA layout |
| `slice_layout` / `layout` | Rank-reduced view / user Shape:Stride layout |
| `layout_encoding` / `shared_layout_encoding` | Base layout classes |
| `swizzled_shared_layout_encoding` / `swizzled_layout` | Swizzled LDS layouts |
| `padded_shared_layout_encoding` | Padded LDS layout (bank-conflict relief) |
| `shared_linear_layout_encoding` | Explicit linear SMEM mapping |
| `DummyRegisterLayoutEncoding` | Placeholder layout resolved post-inline |
| `storage_kind` | Buffer storage selector enum |
| `reuse_group` / `reuse_group_type` / `reuse_group_ir_type` | Buffer overlap trees for aliasing |
| `storage_alias_spec_type` / `storage_alias_spec_type_class` | Alias-spec MLIR types |

### Compute and register control

| Builtin | Purpose |
|---|---|
| `dot_scaled` | Scaled dot with optional tile schedule |
| `amd_scheduled_mfma` | Native MFMA chains (persistent needs VGPR on gfx942) |
| `amd_mfma_commit` | Wave-local MFMA completion boundary |
| `amd_register_resident` | Constrain packed register groups (AGPR/VGPR) |
| `amd_register_class_anchor` | Per-register tied allocator anchor |

### Synchronization

| Builtin | Purpose |
|---|---|
| `alloc_barriers` / `alloc_warp_barrier` | Allocate SMEM / warp barriers |
| `barrier_wait` / `barrier_arrive` | Phase wait / arrival (`arrive_count` must be 1) |
| `cluster_barrier` | Sync all threads in all cluster CTAs |
| `cond_barrier` | Conditional `s_barrier` phase-shift primitive |
| `workgroup_barrier` | Full workgroup barrier + LDS fence |
| `mbarrier` / `mbarrier_type` | Barrier values and types |
| `async_token` | Tracks async-op completion |
| `allocate_tensor_descriptor` | 128B-aligned descriptor scratch (host-side alloc) |

### Scheduling and threads

| Builtin | Purpose |
|---|---|
| `amd_sched_barrier` | Compiler scheduling marker (no sync/fence) |
| `amd_iglp_opt` | IGLP interleave hint |
| `warp_predicate` | Execute body under per-lane predicate |
| `cluster_cta_rank` / `cluster_size_1d` | CTA rank / cluster size |
| `thread_id` / `num_warps` | Thread id / kernel warp count |
| `warp_all` / `warp_any` | Lane votes |

## Not on this target

- `async_load`, `async_load_commit_group`, `async_load_wait_group`:
  async copies require gfx950.
- `buffer_load_to_local`: direct-to-LDS path requires gfx950; use
  register-staged `buffer_load` + `local_store` instead.
- `warp_pipeline_stage`: CDNA4+ cluster scheduling; not available here.
- `extract_slice`, `rematerialized_range`: used by gfx950 kernels;
  unconfirmed here.
- TDM descriptor ops (`async_amd_descriptor_*`) are gfx1250+.
