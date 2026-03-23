# TTGIR Synchronization Ops

Barriers, fences, waits, and other synchronization primitives.

## Op Taxonomy

### mbarriers (SMEM-allocated, 8 bytes each)

| Op | Purpose | PTX |
|---|---|---|
| `ttng.init_barrier` | Initialize with arrival count | `mbarrier.init` |
| `ttng.inval_barrier` | Invalidate for storage reuse | `mbarrier.inval` |
| `ttng.barrier_expect` | Declare expected byte count (for TMA/tcgen05) | `mbarrier.arrive.expect_tx` |
| `ttng.arrive_barrier` | Arrive, decrement pending count | `mbarrier.arrive` |
| `ttng.wait_barrier` | Wait for phase completion | `mbarrier.try_wait.parity` |
| `ttng.async_copy_mbarrier_arrive` | Arrive when prior cp.async ops complete | bridges cp.async → mbarrier |

### Named Barriers (hardware indices 0-15, no SMEM needed)

| Op | Purpose |
|---|---|
| `ttng.arrive_barrier_named` | Arrive on hardware barrier index |
| `ttng.wait_barrier_named` | Wait for N threads to arrive |

Used for lightweight warp-level sync (e.g., ping-pong scheduling in warp
specialization). Only 16 available per SM.

### TCGen5 Commit (Blackwell)

`ttng.tc_gen5_commit`: Commits all prior async tcgen05 ops (MMA + tmem_copy)
to an mbarrier. Sequential ordering: commit A before commit B guarantees
arrive A before arrive B, even if B's group is empty. Optional 2-CTA mode.

### Async Copy Groups (cp.async, SM80+)

| Op | Purpose |
|---|---|
| `ttg.async_commit_group` | Commit pending cp.async ops, return token |
| `ttg.async_wait` | Wait until N or fewer groups outstanding |

### TMA Store Waits

| Op | Purpose |
|---|---|
| `ttng.async_tma_store_wait` | Wait for TMA stores to finish reading SMEM (`pendings` count) |
| `ttng.async_tma_store_token_wait` | Token-based wait for specific TMA store; can arrive on barriers |

### Fences

| Op | Purpose | Min CC |
|---|---|---|
| `ttng.fence_async_shared` | Proxy fence between generic-proxy writes and async-proxy reads | SM90 |
| `ttng.fence` | GPU or system-scope memory fence | SM70 |

### Cluster Sync

| Op | Purpose |
|---|---|
| `ttng.cluster_arrive` | Signal CTA reached sync point (optional `relaxed`) |
| `ttng.cluster_wait` | Block until all CTAs in cluster have arrived |

### Warp-Level

`ttng.vote_ballot_sync`: Warp ballot — collect predicate from each thread,
return 32-bit mask. Pure op.

## Synchronization Patterns

### TMA Load + mbarrier
```
init_barrier %bar, 1
barrier_expect %bar, <bytes>
async_tma_copy_global_to_local %desc [...] %dst, %bar, %pred
wait_barrier %bar, %phase
// SMEM data now available
```

### Blackwell MMA + mbarrier
```
tc_gen5_mma %a, %b, %d, %useD, %pred barriers(%bar : %bar_pred)
tc_gen5_commit %bar
wait_barrier %bar, %phase
// TMEM result now available
```

### cp.async Group Wait
```
%t1 = async_copy_global_to_local ...
%t2 = async_copy_global_to_local ...
%group = async_commit_group tokens %t1, %t2
async_wait %group {num = 0}
// SMEM data now available
```

### Proxy Fence Requirement
```
local_store %tensor, %buf          // generic proxy write to SMEM
fence_async_shared                 // required fence
warp_group_dot %a, %buf, ...      // async proxy read from SMEM
```
Without the fence, the async engine (TMA/wgmma/tcgen05) may read stale data.
