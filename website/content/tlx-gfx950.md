# TLX on gfx950 (AMD MI350 / CDNA4)

MI350X/MI355X: 256 CUs in 8 XCDs (32 CUs each), 160 KB LDS per CU,
CDNA4 MFMA with 32-row tiles, direct HBM-to-LDS async copies
(`buffer_load_to_lds`), in-LDS transpose reads, and inter-wave scheduling
control. Tags follow [tlx.md](tlx.html): `gfx950` means confirmed on this
target.

## Supported `tlx.ops`

One blessed implementation per op from `ops/_catalog.py`. Dtypes use bare
torch names (`float16` covers FP16/BF16); `Requires` lists hardware
capabilities; `Backward` is catalog `supports_backward`.

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `mm` | heuristic | `mm/gfx950.py:mm` | float16 | — | no | GEMM dispatching over LocalSplitU, persistent, register-resident, and direct-to-LDS paths under a macro-tile cost model; 8-wave inter-wave pipelines, non-power-of-two tiles, Stream-K. |
| `grouped_gemm` | heuristic | `grouped_gemm/gfx950.py:grouped_gemm` | — | — | no | One persistent kernel over all groups: 256x256 tiles as four 128x128 quadrants, double-buffered LDS half-tiles, 8-wave staged loop; fully ragged M/N/K; host cost model picks engines. |
| `grouped_gemm_mxfp8` | persistent_warp_pipeline_mxfp8 | `grouped_gemm_mxfp8/gfx950.py:grouped_gemm_mxfp8` | — | — | no | Grouped GEMM for E4M3 + E8M0 microscales on the FP16 quadrant pipeline with CDNA4 scaled MFMA; scale slices ship direct-to-LDS; device-resident group scan. |
| `mm_torchtlx` | inductor_gfx950_mm | `inductor/gfx950_torch:mm` | float16 | — | no | Inductor template entry so the gfx950 GEMM participates in PT2 template selection, autotuning, and fusion. |
| `addmm_torchtlx` | inductor_gfx950_addmm | `addmm/gfx950_torch:addmm` | float16 | — | no | Inductor template entry for fused GEMM+bias on gfx950. |
| `bmm_torchtlx` | inductor_gfx950_bmm | `bmm/gfx950_torch:bmm` | float16 | — | no | Inductor template entry for batched GEMM on gfx950. |
| `flash_attn` | rotated_cluster_pipeline | `flash_attn/gfx950.py:flash_attn` | bfloat16 | — | yes | Flash attention forward: rotated 4-cluster `s_setprio` schedule, LDS-resident KV, lazy rescale + exp2 math, short-N and persistent variants; square MHA D64/D128 differentiable path. |
| `prepare_flash_attn_varlen_backward` | host_validated | `flash_attn_varlen/gfx950_bwd.py:prepare_varlen_backward` | — | — | no | Host-side validation of `cu_seqlens`/offsets producing the plan the prepared varlen backward consumes. |
| `flash_attn_varlen_backward` | prepared | `flash_attn_varlen/gfx950_bwd.py:fa_varlen_backward` | bfloat16 | — | no | Packed-varlen attention backward with on-device compact schedule builds, dynamic owner queues, AGPR/VGPR lifetime control, scheduled MFMA. |
| `flash_attn_mxfp8` | mfma_mxfp8 | `flash_attn_mxfp8/gfx950.py:flash_attn_mxfp8` | bfloat16 | — | no | Forward-only MXFP8 attention: E4M3 Q/K/V + E8M0 scales, both MFMAs scaled; quantize then attend; persistent ping-pong staging. |
| `flash_attn_varlen` | mfma_interleaved | `flash_attn_varlen/gfx950.py:flash_attn_varlen` | float16 | — | no | Packed varlen forward, inference-only: 64-row query tiles, XCD round-robin schedule, small-tile latency regime. |
| `hstu_attn_dev` | tlx | `hstu_attn/gfx950.py:hstu_attn` | float16 | — | yes | RecSys HSTU attention: softmax+SiLU dual heads with TW/PW bias, DUALWAVE padded double-buffers, buffer atomics, XCD-whole-sequence schedule. |
| `hstu_cross_attn` | tlx_split_kv | `hstu_cross_attn/gfx950.py:hstu_cross_attn` | float16 | — | yes | HSTU cross attention: shared-KV single-fetch forward, split-KV + merge kernels, split-softmax backward, retained/small-shape paths. |
| `kda_paged_prefill` | tlx | `kda/gfx950_prefill.py:kda_paged_prefill` | bfloat16 | — | no | Chunk-parallel Delta Attention prefill: 64-token chunks, MFMA causal products, 16x16 solves merged into a 64x64 WY inverse, inter-chunk state scan. |
| `kda_recurrent_decode` | tlx | `kda/gfx950_decode.py:kda_recurrent_decode` | bfloat16 | — | no | Recurrent Delta Attention decode: one packed token per row into an FP32 state pool with independent R/W indices. |

## Supported extension builtins (`tlx.*`)

### Scalars, types, diagnostics

| Builtin | Purpose |
|---|---|
| `assume_uniform` | Assert scalar wave-uniform (`v_readfirstlane`) |
| `zeros` | Zero tensor, optional explicit register layout |
| `dtype_of` / `size_of` | Element type / byte size (host-side) |
| `get_fp8_format_name` | `e5m2`/`e4m3` for scaled-MMA format args |
| `clock64` | 64-bit hardware timer read |
| `assert_same_layout` | Compile-time assert two final layouts match |
| `dump_layout` | Compile-time log of resolved layout |

### Local memory

| Builtin | Purpose |
|---|---|
| `local_alloc` | Allocate `num` LDS buffers, with reuse/layout control |
| `local_view` | Subview of buffer array at index (`buf[i]`) |
| `local_slice` | Same-rank subview; runtime offsets allowed |
| `local_store` | Registers into LDS buffer |
| `local_load` | LDS buffer into registers, with token/layout options |
| `local_gather` / `local_scatter` | Indexed gather/scatter along an axis |
| `storage_alias_spec` | Sized/unsized shared-buffer alias spec |
| `buffered_tensor` / `buffered_tensor_type` | Memory-backed buffer value and its type |

### Async copy (HBM to LDS)

Lowers to `buffer_load_to_lds`: data moves directly into LDS without
register staging.

| Builtin | Purpose |
|---|---|
| `async_load` | Async global-to-LDS copy, returns token |
| `async_load_commit_group` | Bundle outstanding loads into a group |
| `async_load_wait_group` | Wait until at most N groups remain in flight |

### AMD buffer ops

| Builtin | Purpose |
|---|---|
| `buffer_load` / `buffer_store` | Global access via scalar base + i32 offsets |
| `buffer_atomic_add` | Buffer atomic add preserving scalar descriptor |
| `buffer_load_to_local` | Raw global-to-LDS copy primitive behind `async_load` |
| `extract_slice` | Aligned register slice, no cross-thread movement |
| `rematerialized_range` | CSE-proof range recomputed at use; cuts live ranges |

### Layout control

| Builtin | Purpose |
|---|---|
| `local_trans` | Permute descriptor dims (zero data movement) |
| `local_reshape` | Reshape descriptor without moving data |
| `local_reinterpret` | Reinterpret dtype/shape/layout, incl. CDNA4 transpose-read |
| `require_layout` / `release_layout` | Pin / release register layouts |
| `amd_mfma_layout` | MFMA distributed layout, versions 1-4 |
| `dot_operand_layout` | Explicit MFMA dot-operand view |
| `require_amd_wmma_layout` | Pin tensor to explicit WMMA layout |
| `slice_layout` | Rank-reduced view of a distributed parent |
| `layout` | User register layout as Shape:Stride |
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
| `amd_scheduled_mfma` | Native MFMA fragment chains with residency control |
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
| `warp_pipeline_stage` | Inter/intra-wave stage marker → `s_setprio` biasing |
| `amd_sched_barrier` | Compiler scheduling marker (no sync/fence) |
| `amd_iglp_opt` | IGLP interleave hint |
| `warp_predicate` | Execute body under per-lane predicate |
| `cluster_cta_rank` / `cluster_size_1d` | CTA rank / cluster size |
| `thread_id` / `num_warps` | Thread id / kernel warp count |
| `warp_all` / `warp_any` | Lane votes |

## Notes

- `buffer_load_to_local` needs 32/128-bit width plus an aligned mask;
  f16/bf16 `buffer_atomic_add` needs width 2 plus a pair-uniform mask.
- `amd_scheduled_mfma` distinguishes transient vs persistent accumulator
  chains; persistent dV stays in AGPRs while dK stays in VGPRs in the
  tuned backward paths.
- TDM descriptor ops (`async_amd_descriptor_*`) are gfx1250+, not gfx950;
  see [TLX on gfx1250](tlx-gfx1250.html).
