### Barrier operations

- `barriers = tlx.alloc_barrier(num_barriers, arrive_count=1)` **[Hopper+]**

    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.

- `tlx.barrier_wait(bar, phase)` **[Hopper+]**

    Wait until the mbarrier phase completes

- `tlx.barrier_arrive(bar, arrive_count=1)` **[Hopper+]**

    Perform the arrive operation on an mbarrier

- `tlx.named_barrier_wait(bar_id, num_threads)` **[Hopper+]**

    Wait until `num_threads` total threads have reached the specified named mbarrier phase.

- `tlx.named_barrier_arrive(bar_id, num_threads)` **[Hopper+]**

    Signal arrival at a named mbarrier.

    For both APIs, `num_threads` is the total number of threads required to flip
    the barrier phase: `num_waiting_threads + num_arriving_threads`. Wait and
    Arrive calls for the same barrier phase must use the same value.

- `tlx.barrier_expect_bytes(bar, bytes)` **[Hopper+]**

  Signal a barrier of an expected number of bytes to be copied.

- `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)` **[Hopper+]**

    Perform the arrive operation on an mbarrier. If `remote_cta_rank` is provided, signals the barrier in the specified remote CTA's shared memory (useful for multi-CTA synchronization).

- `tlx.amd_sched_barrier(mask=0)`  **[AMD]**

    Prevents selected AMD machine-instruction classes from crossing a source
    boundary. It is a scheduling marker, not a workgroup barrier or memory
    fence.

### Memory Fences

- `tlx.fence(scope)` **[Hopper+]** issues a memory fence. The `scope` argument is required:

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

- `tlx.fence_mbarrier_init_cluster(scope)` **[Hopper+]** issues a memory fence to make mbarrier init visible to cluster.

  Example:
  ```python
  bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
  tlx.fence_mbarrier_init_cluster()
  tlx.cluster_barrier()

  # now bars is ready for cross CTA use
  tlx.barrier_arrive(bar=bars[0], remote_cta_rank=1)
  ```

### Cluster Launch Control (CLC)

CLC (Cluster Launch Control) is a Blackwell-specific feature **[Blackwell]** that enables **dynamic persistent kernel** execution with efficient work stealing across thread blocks. It allows CTAs to dynamically acquire tile IDs from a hardware-managed work queue, enabling load balancing without explicit inter-CTA communication.

#### CLC API

- `context = tlx.clc_create_context(num_consumers=num_consumers)` **[Blackwell]**

    Create a CLC pipeline context with the specified number of stages and expected consumer count.

    **Parameters:**
    - `num_consumers`: Number of consumers that will signal completion per tile (typically 3 async tasks × num_CTAs)

- `tlx.clc_producer(context, p_producer=phase, multi_ctas=False)` **[Blackwell]**

    Issue a CLC try_cancel request to acquire a new tile ID.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase (0 or 1, alternates each iteration)
    - `multi_ctas`: Set to `True` for 2-CTA mode (cluster of 2 CTAs). When enabled, `pred_cta0` is computed internally from `cluster_cta_rank()`.

- `tile_id = tlx.clc_consumer(context, p_consumer=phase, multi_ctas=False, k=0, return_3d=False)` **[Blackwell]**

    Decode the tile ID from a CLC response and signal completion.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase
    - `multi_ctas`: Set to `True` for 2-CTA mode. When enabled, `pred_cta0` is computed internally.
    - `return_3d`: Set to `True` to return `(ctaIdX, ctaIdY, ctaIdZ)` tuple instead of scalar tile_id.

    **Returns:** The tile ID (already offset by `cluster_cta_rank()` for unique tile assignments), or -1 if no work available. With `return_3d=True`, returns `(ctaIdX, ctaIdY, ctaIdZ)` tuple.

#### How CLC Works

CLC uses hardware-assisted work stealing via the PTX instruction:
```
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
```

The `.multicast::cluster::all` qualifier means the response is **asynchronously written to all CTAs** in the cluster. This enables efficient multi-CTA execution where all CTAs in a cluster receive the same base tile ID.

#### CLC Synchronization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLC Producer (clc_producer)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. WAIT:   barrier_wait(bar_empty)      ← Wait for consumers   │
│  2. EXPECT: barrier_expect_bytes(bar_full, 16)                  │
│  3. ISSUE:  clc_issue(response, bar_full) ← Hardware request    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    [Hardware processes CLC]
                    [Multicasts response to all CTAs]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CLC Consumer (clc_consumer)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. WAIT:   barrier_wait(bar_full)       ← Wait for response    │
│  2. QUERY:  tile_id = clc_query(response) ← Extract tile ID     │
│  3. SIGNAL: barrier_arrive(bar_empty)    ← Release producer     │
└─────────────────────────────────────────────────────────────────┘
```

#### Multi-CTA Mode (2-CTA Clusters)

In multi-CTA mode (`multi_ctas=True`), multiple CTAs in a cluster work together on adjacent tiles. The key constraint is: **you can arrive at a remote mbarrier, but you cannot wait on a remote mbarrier** (per NVIDIA specification).

##### Key Principle: "Arrive Remote, Wait Local"

| Operation | Local mbarrier | Remote mbarrier |
|-----------|----------------|-----------------|
| `barrier_wait` | ✅ Allowed | ❌ Undefined behavior |
| `barrier_arrive` | ✅ Allowed | ✅ Allowed (via `remote_cta_rank`) |

##### Example: Multi-CTA GEMM with CLC

```python
@triton.jit
def matmul_kernel(..., PAIR_CTA: tl.constexpr):
    # Create CLC context: 6 consumers for 2-CTA mode (3 tasks × 2 CTAs)
    clc_context = tlx.clc_create_context(num_consumers= 6 if PAIR_CTA else 3)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # Epilogue consumer
            clc_phase_producer = 1
            clc_phase_consumer = 0
            tile_id = start_pid

            while tile_id != -1:
                # Producer: acquire next tile
                tlx.clc_producer(clc_context, p_producer=clc_phase_producer, multi_ctas=PAIR_CTA)
                clc_phase_producer ^= 1

                # ... process tile ...

                # Consumer: get tile ID and signal completion
                tile_id = tlx.clc_consumer(clc_context, p_consumer=clc_phase_consumer, multi_ctas=PAIR_CTA)
                clc_phase_consumer ^= 1
        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            clc_phase_consumer = 0
            tile_id = start_pid

            while tile_id != -1:
                # ... process tile ...

                # Consumer: get tile ID and signal completion
                tile_id = tlx.clc_consumer(clc_context, p_consumer=clc_phase_consumer, multi_ctas=PAIR_CTA)
                clc_phase_consumer ^= 1
        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            clc_phase_consumer = 0
            tile_id = start_pid

            while tile_id != -1:
                # ... process tile ...

                # Consumer: get tile ID and signal completion
                tile_id = tlx.clc_consumer(clc_context, p_consumer=clc_phase_consumer, multi_ctas=PAIR_CTA)
                clc_phase_consumer ^= 1

```

Examples: how mbarriers are communicated in warp specialization
```
    phase = 0
    with tlx.async_tasks():
        with tlx.async_task("default"):

            tlx.barrier_wait(bar=b1, phase=phase ^ 1)

            # Placeholder block to do something

            tlx.barrier_arrive(bar=b0)  # Release

        with tlx.async_task(num_warps=4):

            tlx.barrier_wait(bar=b0, phase=phase)  # Wait

            # Some arith ops TODO. add WS
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            z = x * x
            tl.store(z_ptr + offsets, z, mask=mask)

            tlx.barrier_arrive(bar=b0)  # Wait
```
