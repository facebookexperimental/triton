# TLX on sm90 (NVIDIA Hopper)

H100-class GPUs: TMA descriptors, `wgmma` async MMA, thread-block
clusters with distributed shared memory, warp specialization with
`async_task` regions, and mbarrier pipelines. No TMEM, no `tcgen05`,
no CLC — see [TLX on sm100](tlx-sm100.html) for the Blackwell
superset. Tags follow [tlx.md](tlx.html).

## Supported `tlx.ops`

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `mm` | ws_cooperative | `mm/sm90.py:mm` | float16 | tma | no | Cooperative-persistent GEMM: TMA producer warp-group feeds `wgmma` consumers through mbarriers. Full 8-config space beats the heuristic on tall shapes. |
| `addmm` | tma_ws | `addmm/sm90.py:addmm` | float16 | — | no | Persistent WS, non-persistent one-CTA-per-tile, and direct-load GEMM+bias family; named barriers sync consumer warp-groups. |
| `mm_torchtlx` | inductor_hopper_gemm_ws | `inductor/sm100_torch:mm` | float16 | tma | no | Inductor template entry for the Hopper WS GEMM. |
| `flash_attn` | ws_pipelined_pingpong | `flash_attn/sm90.py:flash_attn` | float16 | tma | yes | FA3-style attention: persistent WS with K-ahead TMA prefetch ping-pong plus a non-persistent variant; FA2 math on TMA+`wgmma` backward. |

## Supported extension builtins (`tlx.*`)

The sm100 set minus the Blackwell-only items (TMEM, `tcgen05`, CLC,
TMA gather/scatter, `stoch_round`, `warp_redux`, packed PTX).

### Scalars, types, diagnostics

| Builtin | Purpose |
|---|---|
| `assume_uniform` | Scalar uniformity assert (no-op on NV) |
| `zeros` | Zero tensor, optional explicit register layout |
| `dtype_of` / `size_of` | Element type / byte size (host-side) |
| `get_fp8_format_name` | `e5m2`/`e4m3` for scaled-MMA format args |
| `clock64` | 64-bit hardware timer read |
| `assert_same_layout` / `dump_layout` | Layout assert / compile-time layout log |

### Local memory

| Builtin | Purpose |
|---|---|
| `local_alloc` | Allocate `num` SMEM buffers |
| `local_view` / `local_slice` | Buffer subviews |
| `local_store` / `local_load` | Registers to/from SMEM buffer |
| `local_gather` / `local_scatter` | Indexed gather/scatter along an axis |
| `storage_alias_spec` | Sized/unsized shared-buffer alias spec |
| `buffered_tensor` / `buffered_tensor_type` | Memory-backed buffer value and its type |

### Async copy and descriptors (TMA)

| Builtin | Purpose |
|---|---|
| `async_load` / `async_load_commit_group` / `async_load_wait_group` | `cp.async` global-to-SMEM pipeline |
| `async_store` | Async SMEM-to-global bulk copy |
| `async_descriptor_load` | TMA global-to-SMEM load tracked by barrier |
| `async_descriptor_store` / `async_descriptor_store_wait` | TMA store + wait |
| `async_descriptor_prefetch_tensor` / `prefetch` | L2 prefetch hints |
| `allocate_tensor_descriptor` | 128B-aligned descriptor scratch |
| `make_tensor_descriptor` / `reinterpret_tensor_descriptor` | Create / reinterpret TMA descriptors |
| `tensor_descriptor_ptr` / `tensor_descriptor_ptr_type` | Descriptor pointer values and types |

### Layout control

| Builtin | Purpose |
|---|---|
| `local_trans` / `local_reshape` / `local_reinterpret` | Metadata-only descriptor transforms |
| `require_layout` / `release_layout` | Pin / release register layouts |
| `dot_scaled` | Scaled dot with optional tile schedule |
| `nv_mma_layout` | Hopper WGMMA accumulator/register layout |
| `nv_mma_shared_layout_encoding` | Hopper MMA SMEM operand encoding |
| `slice_layout` / `layout` | Rank-reduced view / user Shape:Stride layout |
| `layout_encoding` / `shared_layout_encoding` | Base layout classes |
| `swizzled_shared_layout_encoding` / `swizzled_layout` | Swizzled SMEM layouts |
| `padded_shared_layout_encoding` | Padded SMEM layout |
| `shared_linear_layout_encoding` | Explicit linear SMEM mapping |
| `DummyRegisterLayoutEncoding` | Placeholder layout resolved post-inline |
| `storage_kind` | Buffer storage selector enum |
| `reuse_group` / `reuse_group_type` / `reuse_group_ir_type` | Buffer overlap trees |
| `storage_alias_spec_type` / `storage_alias_spec_type_class` | Alias-spec MLIR types |

### Async compute

| Builtin | Purpose |
|---|---|
| `async_dot` | Warp-group async MMA (`wgmma`; register-A allowed) |
| `async_dot_wait` | Wait for prior async dots |

### Synchronization

| Builtin | Purpose |
|---|---|
| `alloc_barriers` / `alloc_warp_barrier` | SMEM / warp / cluster barriers |
| `barrier_expect_bytes` / `barrier_wait` / `barrier_arrive` | Expect-tx, wait, arrive (multicast capable) |
| `remote_view` | Map barrier into another cluster CTA |
| `remote_shmem_store` / `async_remote_shmem_store` / `async_remote_shmem_copy` | Cross-CTA SMEM access |
| `named_barrier_wait` / `named_barrier_arrive` | Hardware named barriers (ids 3-15) |
| `cluster_barrier` / `fence_mbarrier_init_cluster` | Cluster sync / init fence |
| `fence` / `fence_async_shared` | Memory and proxy fences |
| `mbarrier` / `mbarrier_type` / `async_token` | Barrier values, types, completion tokens |

### Warp specialization, votes, clusters

| Builtin | Purpose |
|---|---|
| `async_task` / `async_tasks` | Warp-specialized task regions and grouping |
| `async_task_replica_id` | Replica id inside an async region |
| `warp_all` / `warp_any` | Lane votes |
| `vote_ballot_sync` | Warp ballot bitmask |
| `cluster_cta_rank` / `cluster_size_1d` | CTA rank / cluster size |
| `thread_id` / `num_warps` | Thread id / kernel warp count |

## Notes

- TMA descriptors need 16-byte-aligned row strides; the `mm` catalog
  entry enforces this via `accepts` and declines the rest.
- WGMMA accumulator layouts are Hopper-specific (`nv_mma_layout`
  asserts version 3.0).
