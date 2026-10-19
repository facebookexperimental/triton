# TLX on gfx1250 (AMD RDNA4 / MI450 generation)

The distinguishing TLX feature on this target is TDM (Tensor Descriptor
Machine): hardware descriptors with fused multi-member loads, multicast,
and prefetch. Hardware tags follow [TLX](tlx.html).

## Supported `tlx.ops`

One blessed implementation per op from `ops/_catalog.py`, with the columns
of the [gfx950 table](tlx-gfx950.html).

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `flash_attn_mxfp8` | wmma_tdm_mxfp8 | `flash_attn_mxfp8/gfx1250.py:flash_attn_mxfp8` | bfloat16 | — | no | MXFP8 attention forward, D128: E4M3 Q/K/V + E8M0 scales, both WMMAs scaled; TDM loads into LDS, softmax and WMMA stages ping-ponged across warp groups. |

The tutorial `third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/` also
exercises fused TDM loads/stores, TDM prefetch, and WMMA in grouped GEMM.
See its [README](third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/README.md).

## Supported extension builtins (`tlx.*`)

The entries below distinguish existing target-specific tests from source
implementation. **Implemented** means the frontend and AMD lowering provide
that operation; it does not claim that every dtype, layout, or scheduling
mode has been executed on gfx1250. Runtime coverage remains limited to the
cases exercised by the linked tests.

### TDM descriptors

| Builtin | Purpose |
|---|---|
| `update_tensor_descriptor` | Reposition/rebound an AMD TDM descriptor |
| `async_amd_descriptor_load` | Async TDM global-to-local load |
| `async_amd_descriptor_load_fused` | One fused TDM load for 2-4 members |
| `async_amd_descriptor_store` | Async TDM local-to-global store |
| `amd_descriptor_prefetch_tensor` | L2 prefetch hint for a descriptor tile |
| `async_amd_descriptor_wait` | Wait for outstanding TDM operations |

These operations have gfx1250-specific compiler and runtime tests in the
[TDM tests](python/test/unit/language/test_tlx_amd_gfx1250.py) and
[grouped-GEMM tests](third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py).
The [target feature checks](third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp)
enable TDM for the gfx1250 family, not arbitrary newer AMD architectures.

### Memory, compute, and synchronization

| Builtin | Support and restrictions |
|---|---|
| `local_alloc`, `local_view`, `local_slice`, `local_store`, `local_load` | Shared-memory paths implemented; TMEM is NVIDIA-only |
| `local_gather`, `local_scatter` | Portable shared-memory lowering; partitioned-shared layouts are not supported |
| `local_trans`, `local_reshape`, `local_reinterpret`, `require_layout`, `release_layout` | Layout/view transformations implemented; the selected layout must be valid for its consumer |
| `async_load`, `async_load_commit_group`, `async_load_wait_group` | Non-bulk AMD copy path implemented; direct-to-LDS transfers require a legal width/layout, otherwise pointer-based copies can fall back to synchronous loads |
| `require_amd_wmma_layout`, `dot_operand_layout`, `dot_scaled` | WMMA-v3 compute path; use a WMMA parent for dot-operand layouts, not a CDNA MFMA parent |
| `alloc_barriers`, `alloc_warp_barrier`, `barrier_wait`, `barrier_arrive` | LDS barriers; AMD arrival count must be 1 and NVIDIA `scope="cluster"` barriers are not supported |
| `cluster_barrier`, `cluster_cta_rank`, `cluster_size_1d` | Physical multi-CTA cluster support; single-CTA launches fold the cluster operations |
| `workgroup_barrier` | Workgroup synchronization used by the TDM tests |
| `warp_pipeline_stage` | `inter_wave` TDM token-wait pipeline covered by gfx1250 tests; other modes need separate runtime validation |
| `amd_sched_barrier` | Compiler scheduling marker, not a hardware synchronization or memory fence |

The [code-generation tests](python/test/unit/language/test_tlx_codegen.py)
include gfx1250 shared-memory gather/scatter, WMMA, TDM, multicast, and grouped-GEMM compilation. The
[TDM tests](python/test/unit/language/test_tlx_amd_gfx1250.py) also exercise
barrier arrival/wait and inter-wave token waits. Shared-memory gather/scatter
use the [portable lowering](lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp).
Copy and wait handling are implemented in the
[AMD memory lowering](third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp).

### Other implemented AMD operations

| Builtin | Implementation and limits |
|---|---|
| `buffer_load`, `buffer_store` | AMD resource-descriptor global access; see [global memory access](global-memory.html) |
| `buffer_atomic_add` | FP16/BF16/FP32/FP64/int32/int64 lowering; 16-bit atomics require contiguous pairs and a pair-uniform mask |
| `extract_slice`, `rematerialized_range` | AMD slice/range lowering, not gfx950-only APIs |
| `amd_register_resident`, `amd_register_class_anchor` | Register-allocation annotations; use VGPR for gfx1250, without assuming CDNA AGPR behavior |
| `cond_barrier` | Conditional workgroup barrier; callers must obey its wave-participation contract |
| `amd_iglp_opt` | AMD compiler scheduling hint; no performance guarantee |
| `warp_predicate` | AMD EXEC-mask lowering; consumers involving dots or shuffles require wave-uniform participation |
| `warp_all`, `warp_any` | AMD ballot-based lane votes |

Compile-only [availability tests](python/test/unit/language/test_tlx_codegen.py)
cover all six listed buffer-atomic dtypes on gfx1250. This group has source
implementations, but existing gfx1250 runtime tests do not establish coverage
of every direct API above. The relevant evidence
is the [buffer/atomic lowering](third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp),
[AMD memory lowering](third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp),
[warp-predicate lowering](third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp),
and [thread/timer lowering](third_party/amd/lib/TritonAMDGPUToLLVM/SPMDOpToLLVM.cpp).

### Portable values and types

The following helpers construct values, types, layouts, or metadata through
portable TLX paths: `assume_uniform`, `zeros`, `dtype_of`, `size_of`,
`get_fp8_format_name`, `assert_same_layout`, `dump_layout`,
`storage_alias_spec`, `allocate_tensor_descriptor`, `thread_id`, `num_warps`,
`layout_encoding`, `shared_layout_encoding`,
`swizzled_shared_layout_encoding`, `swizzled_layout`,
`padded_shared_layout_encoding`, `shared_linear_layout_encoding`,
`slice_layout`, `DummyRegisterLayoutEncoding`, `layout`, `storage_kind`,
`reuse_group`, `reuse_group_type`, `reuse_group_ir_type`,
`storage_alias_spec_type`, `storage_alias_spec_type_class`,
`buffered_tensor`, `buffered_tensor_type`, `mbarrier`, `mbarrier_type`,
and `async_token`.

Constructing a type does not enable an unsupported storage kind or matrix
instruction. In particular, `storage_kind.tmem` is not an AMD allocation
mode. See [memory](buffers.html) and [layout control](layouts.html).

## Unsupported operations

- `buffer_load_to_local` is restricted to CDNA3/CDNA4 by the target verifier. It is rejected on gfx1250; use TDM descriptor loads or the non-bulk pointer-based copy path.
- `amd_scheduled_mfma` and `amd_mfma_commit` require gfx942/gfx950 MFMA versions. They are rejected on gfx1250, which uses WMMA-v3.
- `amd_mfma_layout` can construct a CDNA layout object, but is not the gfx1250 matrix-compute layout. Use `require_amd_wmma_layout` instead.
- `clock64` currently emits `llvm.amdgcn.s.memtime`, which gfx1250 does not support. The frontend symbol exists, but LLVM cannot select that timer instruction for this target; see the [timer lowering](third_party/amd/lib/TritonAMDGPUToLLVM/SPMDOpToLLVM.cpp).
- NVIDIA TMEM operations and `async_load(bulk=True)` are not gfx1250 copy/compute paths.

The CDNA-only restrictions are enforced by the
[AMD verifier](third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp) and
[matrix-instruction selection](third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp).

## Notes

- WMMA queuing is a compiler launch option (`enable_wmma_queuing`), not a `tlx.amd_set_wave_sched_mode` builtin. The AMD backend sets the wave-scheduler-mode register; see the [compiler options](third_party/amd/backend/compiler.py).
- Promoting a gfx1250 tutorial into `tlx.ops` means adding a catalog entry with `arch="gfx1250"`; see [high-performance kernels](tlx-ops.html).
- Fused TDM loads require disjoint warp masks and support at most 5 multicast recipients. `async_amd_descriptor_wait` takes pending counts or tokens, not both.
