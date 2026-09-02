## Barriers

> For the concepts behind these calls — asynchronous barrier semantics, the
> mbarrier phase model, named vs memory barriers, and producer-consumer
> patterns — see
> [Barrier Support in TLX](third_party/tlx/doc/tlx_barriers.md).

- `barriers = tlx.alloc_barriers(num_barriers, arrive_count=1)` **[sm90+]**

    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_count`: The number of threads that need to arrive at the barrier before it can be released.

- `tlx.barrier_wait(bar, phase)` **[sm90+]**

    Wait until the mbarrier phase completes

- `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)` **[sm90+]**

    Perform the arrive operation on an mbarrier. If `remote_cta_rank` is
    provided, signals the barrier in the specified remote CTA's shared memory
    (useful for multi-CTA synchronization). On AMD this requires
    `arrive_count == 1`.

- `tlx.named_barrier_wait(bar_id, num_threads)` **[sm90+]**

    Wait until `num_threads` total threads have reached the specified named
    mbarrier phase.

- `tlx.named_barrier_arrive(bar_id, num_threads)` **[sm90+]**

    Signal arrival at a named mbarrier.

    For both APIs, `num_threads` is the total number of threads required to flip
    the barrier phase: `num_waiting_threads + num_arriving_threads`. Wait and
    Arrive calls for the same barrier phase must use the same value.

- `tlx.barrier_expect_bytes(bar, bytes)` **[sm90+]**

  Signal a barrier of an expected number of bytes to be copied.

## Scheduling barriers

- `tlx.amd_sched_barrier(mask=0)` **[amd]**

    Prevents selected AMD machine-instruction classes from crossing a source
    boundary. It is a scheduling marker, not a workgroup barrier or memory
    fence.

## Memory fences

- `tlx.fence(scope)` **[sm90+]** issues a memory fence. The `scope` argument is required:

  | Scope | PTX | Description |
  |-------|-----|-------------|
  | `"gpu"` | `fence.acq_rel.gpu` | Device-scope fence. Orders prior global/shared memory writes to be visible to all GPU threads. |
  | `"sys"` | `fence.acq_rel.sys` | System-scope fence. Like `"gpu"` but also visible to the host CPU. |
  | `"async_shared"` | `fence.proxy.async.shared::cta` | Proxy fence for async shared memory. Required between `local_store` and a subsequent TMA store (`async_descriptor_store`) to the same shared memory. |

  Example:
  ```python
  tlx.local_store(smem_buf, data)
  tlx.fence("async_shared")
  tlx.async_descriptor_store(desc, smem_buf, offsets)
  ```

- `tlx.fence_mbarrier_init_cluster(scope)` **[sm90+]** issues a memory fence to make mbarrier init visible to cluster.

  Example:
  ```python
  bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
  tlx.fence_mbarrier_init_cluster()
  tlx.cluster_barrier()

  # now bars is ready for cross CTA use
  tlx.barrier_arrive(bar=bars[0], remote_cta_rank=1)
  ```
