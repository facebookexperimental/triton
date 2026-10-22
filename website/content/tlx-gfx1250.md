# TLX on gfx1250 (AMD RDNA4 / MI450 generation)

Next-generation AMD target. The distinguishing TLX feature is TDM
(Tensor Descriptor Machine): hardware descriptors with fused multi-member
loads, multicast, and prefetch, mirroring what TMA provides on NVIDIA.
Tags follow [tlx.md](tlx.html); a trailing `?` marks availability that
still needs verification — most of this page is `?` by design.

## Supported `tlx.ops`

No `tlx.ops` catalog entries target gfx1250 yet. The closest coverage is
the tutorial `third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/`, which
exercises fused TDM loads/stores, TDM prefetch, `amd_sched_barrier`, and
`amd_set_wave_sched_mode` in a grouped-GEMM experiment (see its README).

## Supported extension builtins (`tlx.*`)

Status is `yes` only where the TDM tutorials or the builtin definition
confirm it; everything else AMD is `?`.

### TDM descriptors (gfx1250+)

| Builtin | Status | Purpose |
|---|---|---|
| `update_tensor_descriptor` | yes | Reposition/rebound an AMD TDM descriptor |
| `async_amd_descriptor_load` | yes | Async TDM global-to-local load |
| `async_amd_descriptor_load_fused` | yes | One fused TDM load for 2-4 members |
| `async_amd_descriptor_store` | yes | Async TDM local-to-global store |
| `amd_descriptor_prefetch_tensor` | yes | L2 prefetch hint for a descriptor tile |
| `async_amd_descriptor_wait` | yes | Wait for outstanding TDM ops |

### Confirmed carryover

| Builtin | Status | Purpose |
|---|---|---|
| `amd_set_wave_sched_mode` | yes | Write wave-scheduler-mode register field |
| `amd_sched_barrier` | yes | Compiler scheduling marker |
| `cluster_barrier` | yes | Sync all threads in all cluster CTAs |

### Cross-backend (expected yes, portability TBD)

Local memory, async copy, layout control, barriers, ids, and votes work
through the portable TLX paths; confirm per kernel. Full list: `assume_uniform`,
`zeros`, `dtype_of`, `size_of`, `get_fp8_format_name`, `clock64`,
`assert_same_layout`, `dump_layout`, `storage_alias_spec`, `local_alloc`,
`local_view`, `local_slice`, `local_store`, `local_load`, `local_gather`,
`local_scatter`, `async_load`, `async_load_commit_group`,
`async_load_wait_group`, `local_trans`, `local_reshape`, `local_reinterpret`,
`require_layout`, `release_layout`, `dot_scaled`,
`allocate_tensor_descriptor`, `alloc_barriers`, `alloc_warp_barrier`,
`barrier_wait`, `barrier_arrive`, `cluster_cta_rank`, `cluster_size_1d`,
`thread_id`, `num_warps`, `warp_all`, `warp_any`, `layout_encoding`,
`shared_layout_encoding`, `swizzled_shared_layout_encoding`,
`swizzled_layout`, `padded_shared_layout_encoding`,
`shared_linear_layout_encoding`, `slice_layout`,
`DummyRegisterLayoutEncoding`, `layout`, `storage_kind`, `reuse_group`,
`reuse_group_type`, `reuse_group_ir_type`, `storage_alias_spec_type`,
`storage_alias_spec_type_class`, `buffered_tensor`, `buffered_tensor_type`,
`mbarrier`, `mbarrier_type`, `async_token`. See
[TLX on gfx950](tlx-gfx950.html) for what each does.

### AMD builtins (`?` — unconfirmed)

`buffer_load`, `buffer_store`, `buffer_atomic_add`, `buffer_load_to_local`,
`extract_slice`, `rematerialized_range`, `require_amd_wmma_layout`,
`amd_register_resident`, `amd_register_class_anchor`, `amd_scheduled_mfma`,
`amd_mfma_commit`, `amd_mfma_layout`, `dot_operand_layout`, `cond_barrier`,
`workgroup_barrier`, `amd_iglp_opt`, `warp_pipeline_stage`,
`warp_predicate`.

## Notes

- Promoting a gfx1250 tutorial into `tlx.ops` means adding a catalog entry
  with `arch="gfx1250"`; see [tlx-ops](tlx-ops.html) for the process.
- Fused TDM loads require disjoint warp masks and support at most 5
  multicast recipients; `async_amd_descriptor_wait` takes pendings xor
  tokens, not both.
