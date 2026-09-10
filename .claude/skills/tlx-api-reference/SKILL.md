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
| `tlx.alloc_barriers(num_barriers, arrive_count=1)` | Allocate ordinary SMEM barriers. Software arrivals use leader-based lowering, which may synchronize participating threads before the leader arrives. | Both |
| `tlx.alloc_warp_barrier(num_barriers, num_warps=1, num_arrivals=1)` | Allocate SMEM barriers whose software arrivals are performed independently by every participating thread. The initialized count is `num_warps * 32 * num_arrivals`. | NVIDIA |
| `tlx.barrier_expect_bytes(bar, bytes, pred=None)` | Set expected transaction byte count on barrier | Both |
| `tlx.barrier_wait(bar, phase, pred=None)` | Wait until barrier phase flips (LOCAL mbarrier only) | Both |
| `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)` | Signal arrival at barrier. `remote_cta_rank` signals a barrier in a remote CTA — **only valid when ctas_per_cga > 1**, causes "Unexpected buffer remote view in 1cta mode" otherwise. Guard with `if USE_2CTA:` when kernel supports both modes. | Both |
| `tlx.cluster_barrier()` | Full cluster-wide synchronization barrier | Both |

**Ordinary barrier count rules:**
- `alloc_barriers(..., arrive_count=N)` initializes the number of logical arrivals
  required to complete a phase. Transaction bytes registered with
  `barrier_expect_bytes` are an additional completion condition; do not infer the
  software arrival topology from the transaction byte count.
- For TMA full/data-ready barriers where one task registers the transaction, use
  `arrive_count=1` unless the surrounding protocol explicitly requires additional
  software arrivals.
- For local software barriers shared by replicated `tlx.async_task` consumers, the
  ordinary count is typically the number of consumer replicas that arrive once per
  phase.
- For cross-CTA software arrivals using `remote_cta_rank`, derive the count from the
  exact CTA protocol. Do not assume the local-task rule applies.

### Warp-barrier semantics and safe conversion

`alloc_warp_barrier` changes the physical arrival protocol, not merely the spelling
of the allocation. With an ordinary barrier, TLX selects a leader to perform the
arrival and may synchronize the participating threads first. With a warp barrier,
every participating thread performs its own arrival, avoiding that leader-path
synchronization but issuing more mbarrier arrival operations.

The initialized count is:

```text
expected arrivals per phase = num_warps * 32 * num_arrivals
```

Here `num_warps` is the number of warps in each participating task replica, and
`num_arrivals` is the number of such all-thread arrival events targeting the same
barrier in one phase. For example, two replicated four-warp consumers that each
release one shared buffer slot use `num_warps=4, num_arrivals=2`, for 256 arrivals.
A consumer-owned slot released by one four-warp replica uses
`num_warps=4, num_arrivals=1`, for 128 arrivals. Keep the corresponding
`barrier_arrive` calls at their original final-use points and normally use the
default unit arrival count.

Before converting an ordinary barrier, classify its role and arrival source:

| Barrier role | Warp-barrier candidate? | Rule |
|---|---|---|
| Local software empty/reuse notification | Yes, after proving the topology | Every expected lane must arrive exactly once per declared arrival event, after its final read of the protected storage. |
| Local software data-ready notification | Sometimes | Safe only when readiness is produced by the same statically known all-thread topology; benchmark because per-thread arrivals are not universally faster. |
| TMA transaction/full barrier | No | Keep an ordinary barrier so transaction completion remains tracked through `barrier_expect_bytes` and the TMA operation. |
| MMA/tensor-core completion barrier | No automatic conversion | Preserve the completion mechanism required by the MMA API. |
| Named scheduling barrier | No | Preserve the named-barrier ID, participant count, direction, and phase protocol. |
| Remote, multicast, or cross-CTA barrier | No automatic conversion | Keep the established protocol unless backend support and the complete cluster-wide arrival topology are explicitly proven. |
| Divergently predicated arrival | No | A missing lane leaves the phase incomplete; use a warp barrier only when every counted lane is guaranteed to execute the required arrivals. |

Safe-conversion checklist:

1. Identify the protected buffer and confirm the barrier is signaled by explicit
   software `barrier_arrive` calls rather than TMA, MMA, multicast, or remote
   completion.
2. Enumerate every task replica, warp, lane, predicate, and arrival call that targets
   the barrier during one phase.
3. Verify `num_warps * 32 * num_arrivals` equals the exact number of unit arrivals.
4. Prove every counted lane reaches the arrival after its final access to the buffer,
   including prologue, tail, and persistent-loop iterations.
5. Preserve buffer indices, phase calculations, wait sites, and arrival sites; change
   only the allocator for the first A/B experiment.
6. Keep full/TMA barriers ordinary even when the paired empty/reuse barriers are
   converted.
7. Run correctness and benchmark both forms. Warp barriers trade additional
   per-thread arrivals for removal of leader-path synchronization, so conversion is
   an optimization candidate, not a universal rule.

### Named barriers (hardware-allocated, indices 0–15)

| Function | Description | Arch |
|---|---|---|
| `tlx.named_barrier_wait(bar_id, num_threads)` | Wait until the total participant count reaches bar_id | NVIDIA |
| `tlx.named_barrier_arrive(bar_id, num_threads)` | Signal arrival at bar_id using the total participant count | NVIDIA |

`num_threads` is the total number of threads required to flip the barrier phase:
`num_waiting_threads + num_arriving_threads`. Wait and Arrive calls for the same
barrier phase must use the same value. The count must be a multiple of 32 (warp
size); it is typically `num_warp_groups * warps_per_group * 32`.

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

**Pair-CTA MMA (two_ctas=True):** M must be 128 per CTA.

## Multi-CTA (Cluster) Kernels

`ctas_per_cga=(N,1,1)` in triton.Config sets the cluster size. The grid
specifies **total CTAs**; hardware divides by ctas_per_cga to get the number
of clusters. E.g., grid=(2,1,1) with ctas_per_cga=(2,1,1) = 1 cluster of
2 CTAs.

### 2-CTA tile scheduling in attention backward

Each CTA in a cluster gets its own `program_id` and its own tile_id from
CLC. Two CTAs in a cluster naturally get consecutive tiles (pid 0, pid 1).
**No special tile scheduling is needed for 2-CTA** — `start_n = pid` works
as-is. Grid size and `n_tile_num` do NOT change between 1-CTA and 2-CTA.

Think of 2-CTA as two independent 1-CTAs that handle their own K/V tiles
and share Q/dO via multicast. For L2 efficiency, they process consecutive
N-blocks.

### 2-CTA MMA semantics (`two_ctas=True`)

- **A operand** (TMEM): per-CTA, each CTA has different data
- **B operand** (SMEM): split across CTAs and combined by hardware via multicast
- **Output** (TMEM): split across CTAs along the M dimension, written to both CTAs
- Leader MMA writes to both leader TMEM and peer TMEM

### 2-CTA barrier patterns

- TMA loads with `two_ctas=True`: only leader calls `barrier_expect_bytes`
  (guarded by `if is_leader:`). Use `arrive_count=1`.
- Software arrives (`barrier_arrive` with `remote_cta_rank=0`): both CTAs
  arrive on leader's barrier. Use `arrive_count=NUM_CTAS`.
- MMA `mBarriers` with `two_ctas=True`: hardware signals when input reads
  complete. The TMEM output write may still be in-flight.


**input_precision options:** `tf32`, `tf32x3`, `ieee`

## CLC (Cluster Launch Control) — Blackwell only

| Function | Description |
|---|---|
| `tlx.clc_create_context(num_consumers, num_stages=1)` | Create CLC pipeline context (allocates barriers + response buffers) |
| `tlx.clc_producer(context, p_producer, multi_ctas=False, k=0)` | Issue CLC try_cancel request from CTA 0 |
| `tlx.clc_consumer(context, p_consumer, multi_ctas=False, k=0, return_3d=False)` | Decode tile ID from CLC response, signal completion. Returns tile_id or -1. With `return_3d=True`, returns `(ctaIdX, ctaIdY, ctaIdZ)` tuple. |

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
# Full/data-ready barriers track TMA transaction completion and remain ordinary.
bars_full = tlx.alloc_barriers(num_stages, arrive_count=1)

# Ordinary software-release form: one logical arrival per consumer replica.
bars_empty = tlx.alloc_barriers(
    num_stages,
    arrive_count=num_consumers,
)

# Optional optimized form when each consumer replica has four warps and every lane
# is statically guaranteed to execute one unit arrival after its final buffer read.
bars_empty_warp = tlx.alloc_warp_barrier(
    num_barriers=num_stages,
    num_warps=4,
    num_arrivals=num_consumers,
)

# Producer: wait for reuse, then start TMA load tracked by the ordinary full barrier.
tlx.barrier_wait(bar_empty, empty_phase)
tlx.barrier_expect_bytes(bar_full, nbytes)
tlx.async_descriptor_load(desc, indices, barrier=bar_full)

# Consumer: wait for TMA completion, consume the buffer, then release it.
tlx.barrier_wait(bar_full, full_phase)
acc = tlx.async_dot(A, B, acc)
acc = tlx.async_dot_wait(0, acc)  # Prove the final buffer read completed.
tlx.barrier_arrive(bar_empty)
```

When evaluating the optimized form, substitute `bars_empty_warp` for `bars_empty` at
both the producer wait and consumer arrival sites. Do not convert `bars_full`.

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
