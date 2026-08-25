- `tlx.cluster_cta_rank()` **[sm90+]**

  Returns the rank (unique ID) of the current CTA within the cluster.

TLX supports CUDA Thread Block Clustering (available on SM90+ Hopper/Blackwell GPUs) through the `ctas_per_cga` parameter. This provides explicit control over cluster dimensions for multi-CTA cooperative kernels.

#### Usage

Pass `ctas_per_cga` as a tuple when launching a kernel:

```python
kernel[(grid_x, grid_y)](
    ...,
    ctas_per_cga=(2, 1, 1),  # 2x1x1 cluster of CTAs
    **kwargs
)
```

#### Using ctas_per_cga with Autotune

You can specify `ctas_per_cga` in `triton.Config` for autotuning:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=4,
            ctas_per_cga=(2, 1, 1),  # 2x1x1 cluster
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            ctas_per_cga=(1, 1, 1),  # No clustering
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    ...
```


#### TLX vs Triton Semantics

TLX uses **CUDA-native cluster semantics** which differs from Triton's approach:

| Aspect | Triton's way (`num_ctas`) | TLX way (`ctas_per_cga`) |
|--------|---------------------------|--------------------------|
| Grid interpretation | Grid × cluster_dims = total CTAs | Grid = total CTAs |
| Cluster definition | Multiplicative | Regrouping |
| `num_ctas` value | `product(cluster_dims)` | Always 1 |
| `launch_cluster` | Can be False (enabled by `num_ctas != 1`) | Always True |


## Cluster Launch Control (CLC)

CLC (Cluster Launch Control) is a Blackwell-specific feature **[sm100]** that enables **dynamic persistent kernel** execution with efficient work stealing across thread blocks. It allows CTAs to dynamically acquire tile IDs from a hardware-managed work queue, enabling load balancing without explicit inter-CTA communication.

### CLC API

- `context = tlx.clc_create_context(num_consumers=num_consumers)` **[sm100]**

    Create a CLC pipeline context with the specified number of stages and expected consumer count.

    **Parameters:**
    - `num_consumers`: Number of consumers that will signal completion per tile (typically 3 async tasks × num_CTAs)

- `tlx.clc_producer(context, p_producer=phase, multi_ctas=True)` **[sm100]**

    Issue a CLC try_cancel request to acquire a new tile ID.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase (0 or 1, alternates each iteration)
    - `multi_ctas`: Enables cluster-aware synchronization by default when the compilation options specify multiple CTAs. For a `(1, 1, 1)` cluster, the frontend emits the local single-CTA path. Set to `False` to request the local-only path explicitly.

- `tile_id = tlx.clc_consumer(context, p_consumer=phase, multi_ctas=True, k=0, return_3d=False)` **[sm100]**

    Decode the tile ID from a CLC response and signal completion.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase
    - `multi_ctas`: Enables cluster-aware synchronization by default when the compilation options specify multiple CTAs. For a `(1, 1, 1)` cluster, the frontend emits a local barrier arrival. Set to `False` to request the local-only path explicitly.
    - `return_3d`: Set to `True` to return `(ctaIdX, ctaIdY, ctaIdZ)` tuple instead of scalar tile_id.

    **Returns:** The tile ID (offset by `cluster_cta_rank()` for unique tile assignments), or -1 if no work is available. With `return_3d=True`, returns `(ctaIdX, ctaIdY, ctaIdZ)`.

### How CLC Works

CLC uses hardware-assisted work stealing via the PTX instruction:
```
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
```

The `.multicast::cluster::all` qualifier means the response is **asynchronously written to all CTAs** in the cluster. This enables efficient multi-CTA execution where all CTAs in a cluster receive the same base tile ID.

### CLC Synchronization Flow

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

### Multi-CTA Mode (2-CTA Clusters)

In multi-CTA mode (`multi_ctas=True`), multiple CTAs in a cluster work together on adjacent tiles. The key constraint is: **you can arrive at a remote mbarrier, but you cannot wait on a remote mbarrier** (per NVIDIA specification).

#### Key Principle: "Arrive Remote, Wait Local"

| Operation | Local mbarrier | Remote mbarrier |
|-----------|----------------|-----------------|
| `barrier_wait` | ✅ Allowed | ❌ Undefined behavior |
| `barrier_arrive` | ✅ Allowed | ✅ Allowed (via `remote_cta_rank`) |

#### Example: Multi-CTA GEMM with CLC

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
