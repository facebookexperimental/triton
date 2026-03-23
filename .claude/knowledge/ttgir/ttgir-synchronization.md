# TTGIR Synchronization Ops

Barriers, fences, waits, and other synchronization primitives.

## mbarriers (Shared Memory Barriers)

mbarriers are allocated in SMEM (8 bytes each). They support both arrival
counting and byte-count tracking (for TMA/tcgen05 operations).

### `ttng.init_barrier`
Initialize an mbarrier with an arrival `count`. Must be called before any
arrive/wait. Lowers to `mbarrier.init.shared::cta.b64`.

```mlir
ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

### `ttng.inval_barrier`
Invalidate an mbarrier so its memory can be reused. Required by PTX spec
before any reuse of mbarrier storage.

```mlir
ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
```

### `ttng.barrier_expect`
Signal an mbarrier that `size` bytes are expected to arrive (from TMA or
tcgen05 operations). The associated wait will block until all expected
bytes arrive.

```mlir
ttng.barrier_expect %bar, 32768, %pred : !ttg.memdesc<...>
```

### `ttng.arrive_barrier`
Arrive on an mbarrier, decrementing the pending arrival count by `count`.
Optional predicate. Optional `perThread` flag (default: per-warp).

```mlir
ttng.arrive_barrier %bar, 1 : !ttg.memdesc<...>
ttng.arrive_barrier %bar, 1, %pred : !ttg.memdesc<...>
```

### `ttng.wait_barrier`
Wait until mbarrier completes its current phase. Blocks using
`mbarrier.try_wait.parity`. Takes a `phase` value to distinguish
ping/pong phases. Optional predicate and memory deps.

```mlir
ttng.wait_barrier %bar, %phase : !ttg.memdesc<...>
ttng.wait_barrier %bar, %phase, %pred deps %buf : !ttg.memdesc<...>, ...
```

### `ttng.async_copy_mbarrier_arrive`
Arrive on an mbarrier once all previously issued `cp.async` copies complete.
Bridges the cp.async completion mechanism with mbarrier tracking.

```mlir
ttng.async_copy_mbarrier_arrive %bar : !ttg.memdesc<...>
```

## Named Barriers (Hardware Indices)

Named barriers use hardware barrier indices (0-15). No SMEM allocation
needed. Used for warp-level synchronization (e.g., ping-pong scheduling).

### `ttng.arrive_barrier_named`
Arrive on a named barrier. `bar` is the barrier index (i32), `numThreads`
is the expected arrival count.

```mlir
ttng.arrive_barrier_named %bar_idx, %num_threads : i32, i32
```

### `ttng.wait_barrier_named`
Wait on a named barrier until `numThreads` threads have arrived.

```mlir
ttng.wait_barrier_named %bar_idx, %num_threads : i32, i32
```

## TCGen5 Commit (Blackwell)

### `ttng.tc_gen5_commit`
Commits all prior async tcgen05 operations (MMA + tmem_copy) to an mbarrier.
When the operations complete, the barrier arrival count is decremented by 1.

Commit groups are sequential: if commit A is issued before commit B, the
arrive on A is guaranteed to happen before the arrive on B, even if B's
group is empty.

Optional 2-CTA mode (`two_ctas`).

```mlir
ttng.tc_gen5_commit %bar : !ttg.memdesc<...>
ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<...>
```

## Async Copy Groups (cp.async)

### `ttg.async_commit_group`
Commit a group of pending cp.async operations. Returns a token.

```mlir
%token = ttg.async_commit_group
```

### `ttg.async_wait`
Wait until there are `num` or fewer outstanding async copy groups.

```mlir
%token = ttg.async_wait {num = 0}
```

## TMA Store Waits

### `ttng.async_tma_store_wait`
Wait until TMA stores have finished reading from SMEM. `pendings` specifies
how many outstanding stores are allowed. Must complete before SMEM can be
overwritten.

```mlir
ttng.async_tma_store_wait {pendings = 0}
```

### `ttng.async_tma_store_token_wait`
Token-based wait for a specific TMA store to finish reading SMEM. After
completion, optionally arrives on barriers (used by warp specialization for
consumer release). Can also carry deferred NVWS tokens.

```mlir
ttng.async_tma_store_token_wait %token barriers(%bar : %pred) : ...
```

## Fences

### `ttng.fence_async_shared`
Proxy fence for async shared memory operations
(`fence.proxy.async.shared`). Required between generic-proxy writes
(e.g., `local_store`) and async-proxy reads (e.g., TMA, wgmma) to the
same SMEM buffer. `bCluster` controls scope. SM90+.

```mlir
ttng.fence_async_shared {bCluster = false}
```

### `ttng.fence`
GPU or system-scope memory fence. The `scope` attribute specifies the
fence scope (e.g., "gpu", "sys"). SM70+.

```mlir
ttng.fence {scope = "gpu"}
```

## Cluster Synchronization

### `ttng.cluster_arrive`
Cluster-level arrive. Signals that this CTA has reached a synchronization
point. `relaxed` attribute controls ordering guarantees.

```mlir
ttng.cluster_arrive {relaxed = false}
```

### `ttng.cluster_wait`
Cluster-level wait. Blocks until all CTAs in the cluster have arrived.

```mlir
ttng.cluster_wait
```

## Warp-Level Sync

### `ttng.vote_ballot_sync`
Warp-level vote ballot. Collects a predicate from each thread in the warp
and returns a 32-bit mask. When `pred` is a tensor, each thread contributes
the OR of its owned elements. Pure operation. SM70+.

```mlir
%mask = ttng.vote_ballot_sync %membermask, %pred : i1 -> i32
```

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
