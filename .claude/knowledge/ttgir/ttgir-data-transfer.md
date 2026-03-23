# TTGIR Data Transfer Ops

All ops that move data between memory levels. Organized by source → destination.

## Global → SMEM

### `ttg.async_copy_global_to_local`
Per-thread pointer async copy from global to shared memory. Uses `cp.async`.
When `useBulk=true`, accepts a scalar pointer and uses bulk copy mode.
Optionally tracked by an mbarrier. Returns an async token for group-based
waiting. SM80+.

```mlir
%token = ttg.async_copy_global_to_local %src, %dst
    barrier %bar : !ttg.memdesc<...>
    : tensor<128x64x!tt.ptr<f16>> -> !ttg.memdesc<128x64xf16, #shared, #smem>
```

### `ttng.async_tma_copy_global_to_local`
TMA descriptor-based bulk async copy from global to shared memory. Requires
an mbarrier for completion tracking. Supports multicast to multiple CTAs in
a cluster via `multicastTargets` bitmask. SM90+.

```mlir
ttng.async_tma_copy_global_to_local %desc[%x, %y] %dst, %barrier, %pred
    : !tt.tensordesc<tensor<128x64xf16>>, !ttg.memdesc<...> -> !ttg.memdesc<...>
```

### `ttng.async_tma_gather`
TMA gather: copies independently-indexed rows from a global memory matrix to
SMEM. Each row is addressed by its own x-offset. Tracked by mbarrier. SM90+.

```mlir
ttng.async_tma_gather %desc[%x_offsets, %y_offset] %dst, %barrier, %pred
    : ...
```

## Global → L2 Cache

### `ttng.async_tma_prefetch`
TMA-based prefetch hint. Brings data into L2 cache without copying to SMEM.
Issues `cp.async.bulk.prefetch.tensor`. Performance hint only — no mbarrier
needed. SM90+.

```mlir
ttng.async_tma_prefetch %desc[%x, %y], %pred : !tt.tensordesc<...>
```

## SMEM → Global

### `ttng.async_tma_copy_local_to_global`
TMA descriptor-based async copy from SMEM to global. Optional token result
for tracking when the TMA engine finishes reading SMEM (so the buffer can
be reused). SM90+.

```mlir
%token = ttng.async_tma_copy_local_to_global %desc[%x, %y] %src
    : !tt.tensordesc<...>, !ttg.memdesc<...> -> !ttg.async_token
```

### `ttng.async_tma_reduce`
TMA async atomic reduction from SMEM to global memory. Supports add, min,
max, etc. Atomicity is per-element with relaxed semantics. SM90+.

```mlir
ttng.async_tma_reduce add, %desc[%x, %y] %src : ...
```

### `ttng.async_tma_scatter`
TMA scatter: writes independently-indexed rows from SMEM to global memory.
Inverse of `async_tma_gather`. SM90+.

```mlir
ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %src : ...
```

### `ttng.async_store`
Non-TMA bulk async copy from SMEM to global via
`cp.async.bulk.global.shared::cta`. Completion tracked by commit/wait
groups. Predicate (threadIdx.x == 0) auto-generated in lowering. SM90+.

```mlir
ttng.async_store %src, %dst, %size : !ttg.memdesc<...>, !tt.ptr<i8>
```

## Registers ↔ SMEM

### `ttg.local_alloc`
Allocates shared memory. With optional `src` tensor, copies register data
into the new buffer (Registers → SMEM). Without `src`, returns an
uninitialized mutable buffer.

```mlir
// With init (Registers → SMEM):
%buf = ttg.local_alloc %tensor : (tensor<128x64xf16, #blocked>)
    -> !ttg.memdesc<128x64xf16, #shared, #smem>
// Without init (alloc only):
%buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
```

### `ttg.local_store`
Stores a distributed register tensor into a pre-allocated SMEM buffer.
Registers → SMEM.

```mlir
ttg.local_store %tensor, %buf : tensor<128x64xf16, #blocked>
    -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
```

### `ttg.local_load`
Loads from SMEM into a distributed register tensor. SMEM → Registers.
Optional async token for dependency tracking.

```mlir
%tensor = ttg.local_load %buf : !ttg.memdesc<128x64xf16, #shared, #smem>
    -> tensor<128x64xf16, #dot_op>
```

### `ttg.local_dealloc`
Deallocates a shared memory buffer. Optional — if omitted, the compiler
infers deallocation at the first post-dominating point of all uses.

```mlir
ttg.local_dealloc %buf : !ttg.memdesc<128x64xf16, #shared, #smem>
```

## Registers → Remote SMEM (Cluster)

### `ttg.remote_shmem_store`
Stores a register tensor into another CTA's shared memory within a cluster.
`ctaRank` identifies the target CTA (0 to cluster_size-1).

```mlir
ttg.remote_shmem_store %tensor, rank %cta_rank, %dst
    : tensor<...> -> !ttg.memdesc<...>
```

### `ttg.async_remote_shmem_store`
Async version with mbarrier completion signaling. Uses PTX
`st.async.shared::cluster.mbarrier::complete_tx::bytes`. SM90+.

```mlir
ttg.async_remote_shmem_store %tensor, rank %cta_rank, %dst barrier %bar
    : tensor<...> -> !ttg.memdesc<...> barrier_ty !ttg.memdesc<...>
```

## SMEM → TMEM (Blackwell only)

### `ttng.tmem_copy`
Async copy from shared memory to tensor memory via `tcgen05.cp`. Supports
both default TMEM layout and scales layout. Completion tracked via optional
mbarrier. When used with `tc_gen5_mma`, ordering is guaranteed
(cp then mma). SM100+.

```mlir
ttng.tmem_copy %smem_src, %tmem_dst, %barrier
    : !ttg.memdesc<...>, !ttg.memdesc<..., #tmem>, !ttg.memdesc<...>
```

## Registers ↔ TMEM (Blackwell only)

### `ttng.tmem_alloc`
Allocates tensor memory. With optional `src` tensor, copies register data
into the new buffer. Without `src`, returns uninitialized TMEM. SM100+.

```mlir
%tmem = ttng.tmem_alloc %tensor : (tensor<128x128xf32, #blocked>)
    -> !ttg.memdesc<128x128xf32, #tmem_encoding, #tensor_memory>
```

### `ttng.tmem_store`
Stores a register tensor into TMEM. Requires a predicate. Optional token
for aliasing/modref tracking. Registers → TMEM. SM100+.

```mlir
ttng.tmem_store %tensor, %tmem_buf, %pred
    : tensor<128x128xf32, #blocked> -> !ttg.memdesc<...>
```

### `ttng.tmem_load`
Loads from TMEM into a register tensor. TMEM → Registers. Optional token.
Layout of result is restricted. SM100+.

```mlir
%tensor = ttng.tmem_load %tmem_buf
    : !ttg.memdesc<128x128xf32, #tmem_encoding, #tensor_memory>
    -> tensor<128x128xf32, #blocked>
```

## Global Memory Allocation

### `ttg.global_scratch_alloc`
Allocates a global memory buffer private to the current program instance.
Returns a pointer. Used for cross-CTA communication or large temporaries
that don't fit in SMEM.

```mlir
%ptr = ttg.global_scratch_alloc {nbytes = 4096, alignment = 128} : !tt.ptr<i8>
```
