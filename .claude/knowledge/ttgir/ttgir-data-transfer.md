# TTGIR Data Transfer Ops

All ops that move data between memory levels.

## Op Taxonomy

| Direction | Op | Mechanism | Min CC |
|---|---|---|---|
| Global → SMEM | `ttg.async_copy_global_to_local` | `cp.async` (per-thread ptrs) | SM80 |
| Global → SMEM | `ttng.async_tma_copy_global_to_local` | TMA bulk (descriptor-based) | SM90 |
| Global → SMEM | `ttng.async_tma_gather` | TMA gather (per-row x-offsets) | SM90 |
| Global → L2 | `ttng.async_tma_prefetch` | TMA prefetch hint (no SMEM) | SM90 |
| SMEM → Global | `ttng.async_tma_copy_local_to_global` | TMA bulk | SM90 |
| SMEM → Global | `ttng.async_tma_reduce` | TMA atomic reduction | SM90 |
| SMEM → Global | `ttng.async_tma_scatter` | TMA scatter (per-row offsets) | SM90 |
| SMEM → Global | `ttng.async_store` | `cp.async.bulk` (non-TMA) | SM90 |
| Reg → SMEM | `ttg.local_alloc` (with src) | Copy on alloc | all |
| Reg → SMEM | `ttg.local_store` | Store to existing buffer | all |
| SMEM → Reg | `ttg.local_load` | Load from SMEM | all |
| SMEM dealloc | `ttg.local_dealloc` | Optional; compiler infers if omitted | all |
| Reg → Remote SMEM | `ttg.remote_shmem_store` | Cluster store (sync) | SM90 |
| Reg → Remote SMEM | `ttg.async_remote_shmem_store` | Cluster store (async, mbarrier) | SM90 |
| SMEM → TMEM | `ttng.tmem_copy` | `tcgen05.cp` | SM100 |
| Reg → TMEM | `ttng.tmem_alloc` (with src) | Copy on alloc | SM100 |
| Reg → TMEM | `ttng.tmem_store` | Store to existing TMEM | SM100 |
| TMEM → Reg | `ttng.tmem_load` | Load from TMEM | SM100 |
| Global alloc | `ttg.global_scratch_alloc` | Returns `!tt.ptr<i8>` | all |

## Completion Tracking

| Op | Tracking Mechanism |
|---|---|
| `async_copy_global_to_local` | Async token → `async_commit_group` / `async_wait` |
| `async_tma_copy_global_to_local` | mbarrier (arrive + wait_barrier) |
| `async_tma_copy_local_to_global` | Optional async token (for SMEM reuse) |
| `async_tma_prefetch` | None (hint only) |
| `async_remote_shmem_store` | mbarrier |
| `tmem_copy` | Optional mbarrier; ordered w.r.t. `tc_gen5_mma` |
| `async_store` | Commit/wait groups |

## Key Relationships

- **TMA ops** require a `!tt.tensordesc` created by `ttng.tensormap_create` or
  `ttng.reinterpret_tensor_descriptor` (see memory-layout doc).
- **TMA multicast**: `async_tma_copy_global_to_local` supports a
  `multicastTargets` bitmask for writing to multiple CTAs in a cluster.
- **Proxy fence**: A `ttng.fence_async_shared` is required between
  `local_store` (generic proxy) and subsequent TMA/wgmma reads (async proxy)
  to the same SMEM buffer.
- **TMEM ops** are Blackwell-only. `tmem_copy` (SMEM→TMEM) is used for MMA
  scale factors; `tmem_load`/`tmem_store` move data between TMEM and registers.
