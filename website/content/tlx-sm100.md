# TLX on sm100 (NVIDIA Blackwell)

B200-class GPUs: TMA descriptors with multicast, `tcgen05` async MMA with
TMEM accumulators and scales, 2-CTA clusters, Cluster Launch Control (CLC),
warp specialization with `async_task` regions, and microscaling (FP4/FP8)
datatypes. Tags follow [tlx.md](tlx.html).

## Supported `tlx.ops`

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `mm` | ws | `mm/sm100.py:mm` | float16 | tma, tmem | no | Persistent warp-specialized GEMM: TMA producers, `tcgen05` consumers, TMEM-lived accumulators, CLC launch control, TMA epilogue stores, split-K reduce. |
| `mm_bias_grad` | ws_wgrad_bias | `mm_bias_grad/sm100.py:mm_bias_grad` | — | tma, tmem | no | Horizontally fused GEMM weight-gradient + bias-grad rowsum sharing `dout` tiles; TMEM-drain epilogue; fused split-K+bias reduce. |
| `addmm` | tma_ws | `addmm/sm100.py:addmm` | float16 | — | no | Persistent WS GEMM+bias with TMA stores, `tcgen05` commit tracking, CLC-coordinated producers; direct-load baseline included. |
| `grouped_gemm` | ws | `grouped_gemm/sm100.py:grouped_gemm` | — | tma, tmem | no | One persistent WS kernel over all groups with per-group TMA descriptors built on-device. |
| `mm_mxfp8` | ws_mxfp8 | `mm_mxfp8/sm100.py:mm_mxfp8` | — | tma, tmem | no | MXFP8 GEMM: B-resident overlap, scaled `tcgen05` MMA, TMEM scales via `tmem_copy`, LDS aliasing across phases. |
| `grouped_gemm_mxfp8` | ws_persistent_mxfp8 | `grouped_gemm_mxfp8/sm100.py:grouped_gemm_mxfp8` | — | tma, tmem | no | Persistent grouped MXFP8 pipeline with explicit fetch/wait/signal tile protocol; scaled MMA throughout. |
| `mm_torchtlx` | inductor_blackwell_gemm_ws | `inductor/sm100_torch:mm` | float16 | tma, tmem | no | Inductor template entry for the Blackwell WS GEMM. |
| `flash_attn` | ws_pipelined_persistent | `flash_attn/sm100.py:flash_attn` | float16 | tma, tmem | yes | FA4-style attention: TMEM accumulators, 1CTA/2CTA MMA paths, TMA multicast, CLC, ballot-built causal masks. |
| `flash_attn_mxfp8` | ws_pipelined_persistent_mxfp8 | `flash_attn_mxfp8/sm100.py:flash_attn_mxfp8` | bfloat16 | tma, tmem | yes | MXFP8 attention: scaled `tcgen05` MMA both dots, dual-scale 32x32 Q layout, TMEM scale handling. |
| `hstu_attn_dev` | ws | `hstu_attn/sm100.py:hstu_attn` | float16 | tma, tmem | yes | HSTU attention: WS forward with softmax/SiLU/correction inner loops + biases; backward recomputes QK/probs per activation path. |
| `kimi_delta_attention` | ws | `kda/sm100.py:kimi_delta_attention` | float16 | tma, tmem | yes | Chunkwise Delta Attention forward (chunk matmuls + WY inverse) and VJP-scan backward with TMEM/TMA staging. |

## Supported extension builtins (`tlx.*`)

### Scalars, types, diagnostics

| Builtin | Purpose |
|---|---|
| `assume_uniform` | Scalar uniformity assert (no-op on NV) |
| `zeros` | Zero tensor, optional explicit register layout |
| `dtype_of` / `size_of` | Element type / byte size (host-side) |
| `get_fp8_format_name` | `e5m2`/`e4m3` for scaled-MMA format args |
| `clock64` | 64-bit hardware timer read |
| `assert_same_layout` / `dump_layout` | Layout assert / compile-time layout log |
| `stoch_round` | Stochastic-round FP32 to FP8/BF16/F16 (`cvt.rs`) |

### Local and tensor memory

| Builtin | Purpose |
|---|---|
| `local_alloc` | Allocate `num` SMEM/TMEM buffers |
| `local_view` / `local_slice` / `subslice` | Buffer subviews (TMEM delegates to `subslice`) |
| `local_store` / `local_load` | Registers to/from local/TMEM buffer |
| `local_gather` / `local_scatter` | Indexed gather/scatter along an axis |
| `tmem_copy` | Async SMEM-to-TMEM copy incl. packed scales |
| `tmem_shift` | Shift TMEM rows down one, async |
| `storage_alias_spec` | Sized/unsized shared-buffer alias spec |
| `buffered_tensor` / `buffered_tensor_type` | Memory-backed buffer value and its type |

### Async copy and descriptors (TMA)

| Builtin | Purpose |
|---|---|
| `async_load` / `async_load_commit_group` / `async_load_wait_group` | `cp.async` global-to-SMEM pipeline |
| `async_store` | Async SMEM-to-global bulk copy |
| `async_descriptor_load` | TMA global-to-SMEM load tracked by barrier |
| `async_descriptor_gather` / `async_descriptor_scatter` | TMA row gather/scatter |
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
| `slice_layout` / `layout` | Rank-reduced view / user Shape:Stride layout |
| `layout_encoding` / `shared_layout_encoding` | Base layout classes |
| `swizzled_shared_layout_encoding` / `swizzled_layout` | Swizzled SMEM layouts |
| `padded_shared_layout_encoding` | Padded SMEM layout |
| `shared_linear_layout_encoding` | Explicit linear SMEM mapping |
| `DummyRegisterLayoutEncoding` | Placeholder layout resolved post-inline |
| `tensor_memory_layout_encoding` / `make_tensor_memory_layout` | TMEM layouts |
| `tensor_memory_scales_layout_encoding` / `make_tensor_memory_scales_layout` | TMEM scale layouts |
| `TMemCTAMode` | TMEM CTA-mode enum |
| `storage_kind` | Buffer storage selector (smem/tmem/smemCluster) |
| `reuse_group` / `reuse_group_type` / `reuse_group_ir_type` | Buffer overlap trees |
| `storage_alias_spec_type` / `storage_alias_spec_type_class` | Alias-spec MLIR types |

### Async compute

| Builtin | Purpose |
|---|---|
| `async_dot` | Warp-group async MMA (`tcgen05`) |
| `async_dot_scaled` | Async scaled MMA with FP4/FP8 scales |
| `async_dot_wait` | Wait for prior async dots |
| `tcgen05_commit` | Track `tcgen05` ops on a dedicated barrier |

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
| `warp_all` / `warp_any` / `warp_redux` | Votes and warp reductions (`redux.sync`) |
| `vote_ballot_sync` | Warp ballot bitmask |
| `clc_create_context` / `clc_producer` / `clc_consumer` | Cluster Launch Control pipeline |
| `clc_response` / `clc_response_type` / `CLCPipelineContext` | CLC response values and aggregate state |
| `cluster_cta_rank` / `cluster_size_1d` | CTA rank / cluster size |
| `thread_id` / `num_warps` | Thread id / kernel warp count |

## Notes

- TMA descriptors need 16-byte-aligned row strides and a trailing
  stride of 1; the `mm`/`flash_attn` catalog entries enforce this via
  `accepts`.
- `packed_add4/sub4/mul4/fma4` target sm103+ PTX and are unconfirmed
  here (`?`).
- `nv_mma_layout` / `nv_mma_shared_layout_encoding` are Hopper WGMMA
  layouts; unconfirmed on Blackwell (`?`).
