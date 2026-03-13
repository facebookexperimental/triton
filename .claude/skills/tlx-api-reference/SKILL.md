---
name: tlx-api-reference
description: >
  TLX DSL API reference for low-level GPU primitives. Use when writing or
  modifying TLX kernel code that uses barriers (mbarrier, named barriers),
  memory allocation (local_alloc, SMEM, TMEM), TMA operations, warp
  specialization (async_tasks, async_task), CLC (cluster launch control),
  or wgmma instructions. Covers Hopper and Blackwell hardware differences.
---

# TLX API Quick Reference

## Warp Specialization

| Function | Description | Arch |
|---|---|---|
| `tlx.async_tasks()` | Context manager wrapping all async task regions | Both |
| `tlx.async_task([task_ids])` | Assign code to specific task IDs (e.g., `[0]` = producer, `[1,2]` = consumers) | Both |
| `tlx.async_task(num_warps=N, num_regs=R)` | Explicit warp/register allocation for a task | Both |
| `tlx.async_task("default", num_regs=R)` | Default task for code outside explicit tasks | Both |
| `tlx.async_task_replica_id()` | Returns replica ID inside an async region | Both |

### Warp specialization skeleton

```python
with tlx.async_tasks():
    with tlx.async_task([0]):       # Producer
        # TMA loads
    with tlx.async_task([1, 2]):    # Consumers
        # MMA compute
```

## Memory Barriers

### mbarrier (shared-memory allocated)

| Function | Description | Arch |
|---|---|---|
| `tlx.alloc_barriers(num_barriers, arrive_count=1)` | Allocate SMEM barriers and initialize with arrive count | Both |
| `tlx.barrier_expect_bytes(bar, bytes, pred=None)` | Set expected transaction byte count on barrier | Both |
| `tlx.barrier_wait(bar, phase, pred=None)` | Wait until barrier phase flips (LOCAL mbarrier only) | Both |
| `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)` | Signal arrival at barrier | Both |
| `tlx.cluster_barrier()` | Full cluster-wide synchronization barrier | Both |

**arrive_count rules:**
- Implicit arrive from `barrier_expect_bytes`: use `arrive_count=1`
- `barrier_arrive` inside `tlx.async_task`: `arrive_count` = number of warp groups
- `barrier_arrive` outside `tlx.async_task`: `arrive_count=1` (only tid==0 arrives)

### Named barriers (hardware-allocated, indices 0–15)

| Function | Description | Arch |
|---|---|---|
| `tlx.named_barrier_wait(bar_id, num_threads)` | Wait until num_threads arrive at bar_id | NVIDIA |
| `tlx.named_barrier_arrive(bar_id, num_threads)` | Signal arrival at bar_id | NVIDIA |

`num_threads` must be a multiple of 32 (warp size). Typically `num_warp_groups * warps_per_group * 32`.

Used for PingPong scheduling to prevent tensor core contention between consumer warp groups.

## Memory Operations

### SMEM / TMEM allocation

| Function | Description | Arch |
|---|---|---|
| `tlx.local_alloc(shape, dtype, num, storage=smem, reuse=None, layout=None)` | Allocate buffered tensor in SMEM or TMEM | Both (TMEM: Blackwell) |
| `tlx.storage_alias_spec(storage=smem, buffer_size_bytes=None)` | Define shared buffer region for multiple `local_alloc` calls via `reuse` | Both |
| `tlx.local_view(buf, index)` | Get view of a single buffer from a multi-buffered tensor | Both |
| `tlx.local_slice(buf, start, end)` | Slice a sub-range of a buffered tensor | Both |
| `tlx.subslice(tensor, dim, start, size)` | Subslice a tensor along a dimension | Both |
| `tlx.local_load(buf)` | Load from SMEM/TMEM buffer into registers | Both |
| `tlx.local_store(val, buf)` | Store from registers into SMEM/TMEM buffer | Both |
| `tlx.local_trans(buf)` | Transpose a shared memory buffer | Both |
| `tlx.local_reinterpret(buf, dtype)` | Reinterpret buffer with a different dtype | Both |
| `tlx.remote_view(buf, remote_cta_rank)` | Get view of buffer in a remote CTA's SMEM | Both |
| `tlx.remote_shmem_store(val, buf)` | Store to remote CTA's shared memory | Both |
| `tlx.async_remote_shmem_store(val, buf)` | Async store to remote CTA's shared memory | Both |
| `tlx.tmem_copy(src, dst)` | Copy between TMEM buffers | Blackwell |
| `tlx.fence_async_shared()` | Memory fence for async shared memory operations | Both |

**Storage kinds:** `tlx.storage_kind.smem`, `tlx.storage_kind.tmem` (Blackwell), `tlx.storage_kind.smemCluster`

### TMA (Tensor Memory Accelerator)

| Function | Description | Arch |
|---|---|---|
| `tlx.make_tensor_descriptor(ptr, shape, strides, block_shape)` | Create TMA descriptor from pointer (host-side) | Hopper+ |
| `tlx.allocate_tensor_descriptor(ptr, shape, strides, block_shape, swizzle_mode)` | Allocate and fill TMA descriptor in SMEM | Hopper+ |
| `tlx.reinterpret_tensor_descriptor(desc, dtype)` | Reinterpret TMA descriptor with different dtype | Hopper+ |
| `tlx.async_descriptor_load(desc, indices, barrier=None)` | Async TMA load from global → SMEM, tracked by barrier | Hopper+ |
| `tlx.async_descriptor_store(desc, val, indices)` | Async TMA store from registers → global | Hopper+ |
| `tlx.async_descriptor_store_wait()` | Wait for all pending TMA stores to complete | Hopper+ |
| `tlx.async_load(ptr, buf, barrier)` | Async bulk copy global → SMEM (cp.async) | Hopper+ |
| `tlx.async_load_commit_group()` | Commit async load group | Hopper+ |
| `tlx.async_load_wait_group(n)` | Wait for async load groups (n pending allowed) | Hopper+ |

## Matrix Multiply (MMA)

| Function | Description | Arch |
|---|---|---|
| `tlx.async_dot(A, B, acc=None, use_acc=None, mBarriers=[], two_ctas=False)` | Warp-group MMA: D = A @ B + C. Maps to wgmma (Hopper) or tcgen05.mma (Blackwell) | Both |
| `tlx.async_dot_scaled(A, B, acc, A_scale, A_format, B_scale, B_format, ...)` | Scaled MMA with FP8 inputs: D = (A*scale_A) @ (B*scale_B) + D | Blackwell |
| `tlx.async_dot_wait(pendings, inp)` | Wait for N pending async dot operations to complete | Both |
| `tlx.tcgen05_commit(mBarrier, two_ctas=False)` | Make mbarrier track completion of prior tcgen05 ops. Use a SEPARATE mbarrier from async_dot | Blackwell |

**Minimum tile sizes for async_dot:** M ≥ 64, K ≥ 16, N ≥ 32

**input_precision options:** `tf32`, `tf32x3`, `ieee`

## CLC (Cluster Launch Control) — Blackwell only

| Function | Description |
|---|---|
| `tlx.clc_create_context(num_consumers, num_stages=1)` | Create CLC pipeline context (allocates barriers + response buffers) |
| `tlx.clc_producer(context, p_producer, multi_ctas=False, k=0)` | Issue CLC try_cancel request from CTA 0 |
| `tlx.clc_consumer(context, p_consumer, multi_ctas=False, k=0, return_3d=False)` | Decode cluster CTA ID from CLC response, signal completion. Returns `tile_id_x` by default, or `(tile_id_x, tile_id_y, tile_id_z)` if `return_3d=True`. Returns `-1` or `(-1, -1, -1)` if no work. |

For 2-CTA mode: set `multi_ctas=True` (uses "arrive remote, wait local" pattern).

## Utility

| Function | Description | Arch |
|---|---|---|
| `tlx.cluster_cta_rank()` | Unique CTA ID within a cluster (all dims) | Both |
| `tlx.thread_id(axis)` | Thread ID along axis 0, 1, or 2 | Both |
| `tlx.dtype_of(tensor_or_desc)` | Get element type of tensor or tensor descriptor | Both |
| `tlx.size_of(dtype)` | Size of dtype in bytes | Both |
| `tlx.get_fp8_format_name(dtype)` | Get FP8 format string ("e5m2" or "e4m3") for scaled MMA | Both |
| `tlx.clock64()` | 64-bit hardware clock value (for timing) | Both |
| `tlx.stoch_round(src, dst_ty, rand_bits)` | Hardware stochastic rounding FP32 → FP8/BF16/F16 | Blackwell |

## Common patterns

### Producer-consumer with mbarrier (pipelined GEMM)

```python
bars_full = tlx.alloc_barriers(num_stages, arrive_count=1)   # TMA arrives implicitly
bars_empty = tlx.alloc_barriers(num_stages, arrive_count=num_consumers)

# Producer: TMA load → signal full
tlx.barrier_expect_bytes(bar_full, nbytes)
tlx.async_descriptor_load(desc, indices, barrier=bar_full)

# Consumer: wait full → MMA → signal empty
tlx.barrier_wait(bar_full, phase)
tlx.async_dot(A, B, acc)
tlx.barrier_arrive(bar_empty)
```

### PingPong with named barriers

```python
# Consumer 0 waits for Consumer 1, then issues MMA
tlx.named_barrier_wait(9, 256)   # 256 = 2 warp groups * 4 warps * 32 threads
qk = tlx.async_dot(q, k)
tlx.named_barrier_arrive(10, 256)

# Consumer 1 waits for Consumer 0's MMA to finish
tlx.named_barrier_arrive(9, 256)
tlx.named_barrier_wait(10, 256)
qk = tlx.async_dot(q, k)
```

## Deep-dive docs

- API reference: `third_party/tlx/README.md`
- Barriers: `third_party/tlx/doc/tlx_barriers.md`
- Placeholder layouts: `third_party/tlx/doc/PlaceholderLayouts.md`
- Storage alias design: `third_party/tlx/doc/storage_alias_spec_design.md`
