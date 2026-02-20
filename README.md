# TLX - Triton Low-level Language Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It offers intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, computation, and asynchronous control flow. TLX is designed for expert users pushing Triton closer to the metal.

Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:

- Hardware-specific intrinsics (e.g., wgmma, async_copy, barrier)
- Shared and local memory allocation
- Instruction-level scheduling and control
- Cross-warpgroup synchronization


While this approach places more responsibility on the user, it reduces the compiler's role as a performance bottleneck. Although it may introduce divergence across hardware platforms, it empowers users to perform deeper, architecture-specific optimizations without relying solely on compiler heuristics.


## The DSL Extension

### Local buffer operations

- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS)`

    Allocate `NUM_BUFFERS` buffers in local memory per thread block, each of size size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, tlx.storage_kind.tmem)`

    Allocate `NUM_BUFFERS` of buffers in the tensor memory per thread block, each with size size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, reuse=other_buffers)`

    Alias this allocation to an existing `buffered_tensor` so multiple logical buffers reuse the same underlying local storage (SMEM or TMEM) without reallocation.


- `buffer = tlx.local_view(buffers, buffer_idx)` or `buffer = buffers[buffer_idx]`

    Return a subview of the buffer indexed by `buffer_idx` from `buffers`. Both the explicit `local_view()` call and the indexing syntax `[]` are supported.


- `distributed_tensor = tlx.local_load(buffer, optional_token)`

    Loads the buffer from local memory or tensor memory into a distributed tensor.


- `tlx.local_store(buffer, distributed_tensor)`

    Store a distributed tensor into a buffer in local memory or tensor memory.

- `buffer = tlx.local_trans(buffer, dims)`

    Permutes the dimensions of a tensor.

- `buffer = tlx.local_slice(buffer, offsets=[m, n], shapes=[M, N])`

    Slice a `M x N` tensor at a `m x n` offset.

### Remote buffer operations

- `buffer = tlx.remote_view(buffer, remote_cta_rank)`

  Return a remote view of the `buffer` living in another CTA in the same cluster with ID `remote_cta_rank`. NOTE: for
  now we only support barrier as `buffer`, not general SMEM.

- `tlx.remote_shmem_store(dst, src, remote_cta_rank)`

  Store a distributed tensor into a buffer in the remote shared memory of a cluster (synchronous).

  **Parameters:**
  - `dst`: The destination buffer in local shared memory (will be internally mapped to the remote CTA)
  - `src`: The source distributed tensor to store
  - `remote_cta_rank`: The rank (unique ID) of the remote CTA within the cluster

  **Example:**
  ```python
  # Allocate shared memory buffer
  buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)

  # Store to remote CTA's shared memory (synchronous)
  tlx.remote_shmem_store(buffer[0], src_tensor, remote_cta_rank=1)
  ```

### Async memory access


- `tlx.async_descriptor_load(memdesc, buffer, [offsets], barrier, cache_modifier, eviction_policy, is_volatile)`

   Load a chunk of data from global memory into a local memory buffer. The global address, strides, and buffer size are defined by the memory descriptor. A barrier object is provided and signaled upon completion of the operation.


- `tlx.async_descriptor_store(memdesc, buffer, [offsets])`

   Store a chunk of data from local memory into global memory buffer. The global address, strides, and buffer size are defined by the memory descriptor.


- `tlx.async_remote_shmem_store(dst, src, remote_cta_rank, barrier)`

   Store a distributed tensor into a buffer in the remote shared memory of a cluster asynchronously. Signals the provided mbarrier when the store completes.

   **Parameters:**
   - `dst`: The destination buffer in local shared memory (will be internally mapped to the remote CTA)
   - `src`: The source distributed tensor to store
   - `remote_cta_rank`: The rank (unique ID) of the remote CTA within the cluster
   - `barrier`: mbarrier to signal when the store completes

   **Example:**
   ```python
   # Allocate shared memory buffer and barrier
   buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)
   barrier = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

   # Store to remote CTA's shared memory
   tlx.async_remote_shmem_store(buffer[0], src_tensor, remote_cta_rank=1, barrier=barrier[0])
   ```

- `desc_ptrs = tlx.allocate_tensor_descriptor(num)`

   Allocates global memory for tensor descriptor storage with built-in parameters (nbytes=128, alignment=128 per descriptor).
   Returns a `tensor_descriptor_ptr` with 128-byte stride semantics that supports indexing.

   **Parameters:**
   - `num`: Number of tensor descriptors to allocate (must be a constexpr)

   **Returns:**
   - A `tensor_descriptor_ptr` where indexing (e.g., `desc_ptrs[0]`, `desc_ptrs[1]`) advances by 128 bytes per index

   **Example:**
   ```python
   # Allocate storage for 4 tensor descriptors
   desc_ptrs = tlx.allocate_tensor_descriptor(num=4)

   # Access individual descriptors using indexing
   desc_ptr_0 = desc_ptrs[0]  # First descriptor
   desc_ptr_1 = desc_ptrs[1]  # Second descriptor (128 bytes offset)
   ```

- `tlx.make_tensor_descriptor(desc_ptr, base, shape, strides, block_shape, padding_option)`

   Create a TMA (Tensor Memory Accelerator) descriptor for efficient asynchronous data movement on Hopper and Blackwell GPUs.

   **Parameters:**
   - `desc_ptr` (optional): Tensor descriptor pointer from `allocate_tensor_descriptor()`. Pass `None` for automatic allocation.
   - `base`: Base pointer to the tensor in global memory
   - `shape`: List of tensor dimensions (dynamic, runtime values)
   - `strides`: List of tensor strides (dynamic, runtime values)
   - `block_shape`: Shape of the block to be loaded/stored (compile-time constants)
   - `padding_option`: Padding option for out-of-bounds accesses (default: "zero")

   **Example:**
   ```python
   # Create a 2D tensor descriptor with automatic scratch allocation
   desc = tlx.make_tensor_descriptor(
       desc_ptr=None,  # Compiler allocates scratch memory automatically
       base=tensor_ptr,
       shape=[M, N],
       strides=[N, tl.constexpr(1)],
       block_shape=[64, 64],
   )

   # Or with explicit descriptor allocation for advanced use cases (e.g., pipelining)
   desc_ptrs = tlx.allocate_tensor_descriptor(num=2)

   # Create descriptor at index 0
   tlx.make_tensor_descriptor(
       desc_ptr=desc_ptrs[0],
       base=tensor_ptr,
       shape=[M, N],
       strides=[N, tl.constexpr(1)],
       block_shape=[64, 64],
   )

   # Reinterpret the descriptor for TMA operations
   desc = tlx.reinterpret_tensor_descriptor(
       desc_ptr=desc_ptrs[0],
       block_shape=[64, 64],
       dtype=tl.float16,
   )

   # Use with async TMA operations
   tlx.async_descriptor_load(desc, buffer, offsets=[m_offset, n_offset], barrier=mbar)
   ```

- `desc = tlx.reinterpret_tensor_descriptor(desc_ptr, block_shape, dtype)`

   Reinterpret a tensor descriptor pointer as a TMA-backed tensor descriptor object.

   **Parameters:**
   - `desc_ptr`: A `tensor_descriptor_ptr` pointing to the TMA descriptor (from `allocate_tensor_descriptor`)
   - `block_shape`: Shape of the block to be loaded/stored (compile-time constants)
   - `dtype`: Data type of the tensor elements

   **Example:**
   ```python
   # Allocate and create descriptor
   desc_ptrs = tlx.allocate_tensor_descriptor(num=2)
   tlx.make_tensor_descriptor(desc_ptr=desc_ptrs[0], base=a_ptr, shape=[M, K], strides=[K, 1], block_shape=[128, 64])

   # Reinterpret for use with TMA
   a_desc = tlx.reinterpret_tensor_descriptor(desc_ptr=desc_ptrs[0], block_shape=[128, 64], dtype=tl.float16)
   tlx.async_descriptor_load(a_desc, buffer, offsets=[offs_m, offs_k], barrier=mbar)
   ```

- `tlx.async_load(tensor_ptr, buffer, optional_mask, optional_other, cache_modifier, eviction_policy, is_volatile)`

   Load a chunk of data from global memory into a local memory buffer asynchronously.

   The operation returns a token object which can be used to track the completion of the operation.


- `tlx.async_load_commit_group(tokens)`

   Commits all prior initiated but uncommitted async_load ops an async group. Optionally, each token represents a tracked async load operation.

- `tlx.async_load_wait_group(pendings, tokens)`

   Wait for completion of prior asynchronous copy operations. The `pendings` argument indicates the number of in-flight operations not completed.
   Optionally, each token represents a tracked async commit group operation.


### Async tensor core operations

- `acc = tlx.async_dot(a[i], b[i], acc)`
- `acc = tlx.async_dot(a_reg, b[i], acc)`
- `acc[i] = tlx.async_dot(a[i], b[i], acc[i], barrier)`
- `acc[i] = tlx.async_dot_scaled(a[i], b[i], acc[i], a_scale[i], a_format, b_scale[i], b_format, use_acc, two_ctas, mBarriers)`

    **Parameters:**
    - `a[i]`: A tile in shared memory (FP8 format)
    - `b[i]`: B tile in shared memory (FP8 format)
    - `acc[i]`: Accumulator tile in tensor memory (TMEM)
    - `a_scale[i]`: Per-block scaling factors for A (E8M0 format in SMEM)
    - `a_format`: FP8 format string for A: `"e4m3"`, `"e5m2"`, or `"e2m1"`
    - `b_scale[i]`: Per-block scaling factors for B (E8M0 format in SMEM)
    - `b_format`: FP8 format string for B: `"e4m3"`, `"e5m2"`, or `"e2m1"`
    - `use_acc`: If `True`, compute D = A@B + D; if `False`, compute D = A@B
    - `two_ctas`: If `True`, enables 2-CTA collective MMA (generates `tcgen05.mma.cta_group::2`)
    - `mBarriers`: Optional list of mbarriers for MMA completion signaling

    **2-CTA Scaled MMA:** When `two_ctas=True`, the scaled MMA operates across two CTAs in a cluster. Key considerations:
    - **B data is split**: Each CTA loads half of B (`BLOCK_N // 2`)
    - **B scale is NOT split**: Both CTAs need the full B scale for correct MMA computation
    - **CTA synchronization**: Use "Arrive Remote, Wait Local" pattern before MMA
    - **MMA predication**: Compiler auto-generates predicate so only CTA 0 issues the MMA

    **Example: 2-CTA Scaled MMA**
    ```python
    # B data split across CTAs, but B scale is full
    desc_b = tl.make_tensor_descriptor(b_ptr, ..., block_shape=[BLOCK_K, BLOCK_N // 2])
    desc_b_scale = tl.make_tensor_descriptor(b_scale_ptr, ..., block_shape=[BLOCK_N // 128, ...])  # Full scale

    # Load B with CTA offset, B scale without offset
    tlx.async_descriptor_load(desc_b, b_tile[0], [0, cluster_cta_rank * BLOCK_N // 2], bar_b)
    tlx.async_descriptor_load(desc_b_scale, b_scale_tile[0], [0, 0, 0, 0], bar_b_scale)  # Full B scale

    # CTA sync: "Arrive Remote, Wait Local"
    tlx.barrier_arrive(cta_bars[0], 1, remote_cta_rank=0)
    tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

    # 2-CTA scaled MMA with mBarriers for completion tracking
    tlx.async_dot_scaled(
        a_tile[0], b_tile[0], c_tile[0],
        a_scale_tile[0], "e4m3",
        b_scale_tile[0], "e4m3",
        use_acc=False,
        two_ctas=True,
        mBarriers=[mma_done_bar],
    )
    tlx.barrier_wait(mma_done_bar, tl.constexpr(0))
    ```

    **Alternative: Using tcgen05_commit for MMA completion**
    ```python
    # Issue MMA without mBarriers
    tlx.async_dot_scaled(..., two_ctas=True)

    # Use tcgen05_commit to track all prior MMA ops
    tlx.tcgen05_commit(mma_done_bar, two_ctas=True)
    tlx.barrier_wait(mma_done_bar, tl.constexpr(0))
    ```

    **TMEM-backed MX Scales:**

    For scaled MMA operations on Blackwell GPUs, scales can be stored in Tensor Memory (TMEM) for efficient access. TLX provides automatic layout resolution for TMEM scale buffers.

    *Allocating TMEM Scale Buffers:*

    When allocating TMEM buffers for uint8/int8 types (used for MX scales), TLX uses a placeholder layout (`DummyTMEMLayoutAttr`) that gets automatically resolved to `TensorMemoryScalesEncodingAttr` during compilation when the buffer is used with `async_dot_scaled`.

    ```python
    # Allocate TMEM buffers for scales (layout is automatically resolved)
    a_scale_tmem = tlx.local_alloc((128, 8), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
    b_scale_tmem = tlx.local_alloc((256, 4), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
    ```

    *Copying Scales from SMEM to TMEM:*

    Use `tlx.tmem_copy` to efficiently transfer scale data from shared memory to tensor memory:

    ```python
    # Copy scales from SMEM to TMEM (asynchronous, uses tcgen05.cp instruction)
    tlx.tmem_copy(a_scale_smem, a_scale_tmem)
    tlx.tmem_copy(b_scale_smem, b_scale_tmem)
    ```

    *Using TMEM Scales with Scaled MMA:*

    ```python
    # TMEM scales are automatically detected and used with the correct layout
    tlx.async_dot_scaled(
        a_smem, b_smem, acc_tmem,
        A_scale=a_scale_tmem, A_format="e4m3",
        B_scale=b_scale_tmem, B_format="e4m3",
        use_acc=True,
        mBarriers=[mma_bar],
    )
    ```

    *Complete Example: TMEM-backed Scaled GEMM:*

    ```python
    @triton.jit
    def scaled_gemm_kernel(...):
        # Allocate TMEM for accumulator and scales
        acc = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, num=1, storage=tlx.storage_kind.tmem)
        a_scale_tmem = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32), tl.uint8, num=1, storage=tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32), tl.uint8, num=1, storage=tlx.storage_kind.tmem)

        # Load scales from global memory to SMEM
        tlx.async_descriptor_load(a_scale_desc, a_scale_smem, [...], barrier=bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_smem, [...], barrier=bar)
        tlx.barrier_wait(bar, phase)

        # Copy scales from SMEM to TMEM
        tlx.tmem_copy(a_scale_smem[0], a_scale_tmem[0])
        tlx.tmem_copy(b_scale_smem[0], b_scale_tmem[0])

        # Perform scaled MMA with TMEM scales
        tlx.async_dot_scaled(
            a_smem[0], b_smem[0], acc[0],
            A_scale=a_scale_tmem[0], A_format="e4m3",
            B_scale=b_scale_tmem[0], B_format="e4m3",
            use_acc=False,
        )
    ```

    **Note:** Multibuffering is automatically cancelled for scale buffers since TMEM scales don't support multibuffering. 3D allocations (1×M×K) are automatically flattened to 2D (M×K).

- `acc = tlx.async_dot_wait(pendings, acc)`

    Wait for completion of prior asynchronous dot operations. The pendings argument indicates the number of in-flight operations not completed.

    Example:
    ```python
    acc = tlx.async_dot(a_smem, b_smem)
    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
    tl.store(C_ptrs, acc)
    ```

### Barrier operations

- `barriers = tlx.alloc_barrier(num_barriers, arrive_count=1)`

    Allocates buffer in shared memory and initialize mbarriers with arrive_counts.

    Input:
    - `num_barriers`: The number of barriers to allocate.
    - `arrive_counts`: The number of threads that need to arrive at the barrier before it can be released.

- `tlx.barrier_wait(bar, phase)`

    Wait until the mbarrier phase completes

- `tlx.barrier_arrive(bar, arrive_count=1)`

    Perform the arrive operation on an mbarrier

- `tlx.named_barrier_wait(bar_id, num_threads)`

    Wait until `num_threads` threads have reached the specified named mbarrier phase.

- `tlx.named_barrier_arrive(bar_id, num_threads)`

    Signal arrival at a named mbarrier with the given thread count.

- `tlx.barrier_expect_bytes(bar, bytes)`

  Signal a barrier of an expected number of bytes to be copied.

- `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)`

    Perform the arrive operation on an mbarrier. If `remote_cta_rank` is provided, signals the barrier in the specified remote CTA's shared memory (useful for multi-CTA synchronization).

### Cluster Launch Control (CLC)

CLC (Cluster Launch Control) is a Blackwell-specific feature that enables **dynamic persistent kernel** execution with efficient work stealing across thread blocks. It allows CTAs to dynamically acquire tile IDs from a hardware-managed work queue, enabling load balancing without explicit inter-CTA communication.

#### CLC API

- `context = tlx.clc_create_context(num_consumers=num_consumers)`

    Create a CLC pipeline context with the specified number of stages and expected consumer count.

    **Parameters:**
    - `num_consumers`: Number of consumers that will signal completion per tile (typically 3 async tasks × num_CTAs)

- `tlx.clc_producer(context, p_producer=phase, multi_ctas=False)`

    Issue a CLC try_cancel request to acquire a new tile ID.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase (0 or 1, alternates each iteration)
    - `multi_ctas`: Set to `True` for 2-CTA mode (cluster of 2 CTAs). When enabled, `pred_cta0` is computed internally from `cluster_cta_rank()`.

- `tile_id = tlx.clc_consumer(context, p_consumer=phase, multi_ctas=False, return_3d=False)`

    Decode the cluster CTA ID from a CLC response and signal completion.

    **Parameters:**
    - `context`: CLC pipeline context from `clc_create_context`
    - `phase`: Current barrier phase
    - `multi_ctas`: Set to `True` for 2-CTA mode. When enabled, `pred_cta0` is computed internally.
    - `return_3d`: Set to `True` to return all three dimensions as a tuple.

    **Returns:**
    - If `return_3d=False` (default): Returns `tile_id_x` only (backwards compatible). The x dimension is automatically offset by `cluster_cta_rank()` for unique tile assignments. Returns `-1` if no work available.
    - If `return_3d=True`: Returns a tuple of `(tile_id_x, tile_id_y, tile_id_z)` representing all three dimensions. Returns `(-1, -1, -1)` if no work available.

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


### Warp Specialization operations

- `tlx.async_tasks` and `tlx.async_task`

```
    with tlx.async_tasks
        with tlx.async_task("default")
            ...
        with tlx.async_task(num_warps=4)
            ...
```
`tlx.async_tasks` opens a multi-tasking region where independent asynchronous tasks can be declared. Each task executes in parallel using a dedicated subset of warps within the thread block.

`tlx.async_task("default")` defines the default task, also known as the trunk. It uses the available warps not explicitly reserved by other tasks.

`tlx.async_task(num_warps=4)` defines a warp-specialized asynchronous task that explicitly reserves 4 warps in addition to those used by the trunk task.

#### async_task Parameters

| Parameter | Description |
|-----------|-------------|
| `"default"` | First positional argument to mark this as the default/trunk task |
| `num_warps` | Number of warps to reserve for this task |
| `num_regs` | Number of registers per thread (optional, for register allocation tuning) |
| `replicate` | Number of replicas for this task (default: 1). Creates multiple copies of the task region |
| `warp_group_start_id` | Starting warp ID for this task (optional). Allows explicit control over warp assignment |

#### Explicit Warp Assignment with warp_group_start_id

By default, the compiler automatically assigns warp IDs to each task. However, you can use `warp_group_start_id` to explicitly specify which warps each task should use. This is useful for:
- Fine-grained control over warp-to-task mapping
- Ensuring specific hardware resource allocation
- Advanced optimization scenarios

**Example:**
```python
with tlx.async_tasks():
    with tlx.async_task("default"):  # Uses warps 0-3 (from num_warps=4 kernel param)
        # Producer task
        ...
    with tlx.async_task(num_warps=2, warp_group_start_id=4, replicate=2):
        # Two replicas, each using 2 warps
        # Replica 0: warps 4-5
        # Replica 1: warps 6-7
        ...
    with tlx.async_task(num_warps=1, warp_group_start_id=8):
        # Consumer task using warp 8
        ...
```

**Validation Rules:**
- Warp ranges must not overlap between tasks
- Non-default tasks must not overlap with the default region (warps 0 to kernel's `num_warps`)
- When using `warp_group_start_id`, it must be specified for ALL non-default tasks or NONE

### CUDA Thread Block Clustering

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


### Other operations

- `tlx.cluster_cta_rank()`

  Returns the rank (unique ID) of the current CTA within the cluster.

- `tlx.thread_id(axis)`

    Returns the id of the current thread instance along the given `axis`.

- `tlx.dtype_of(v)`

    Returns the dtype of a tensor or tensor descriptor.

- `tlx.size_of(dtype)`

    Returns the size in bytes of a given Triton dtype. This is useful for dynamically computing memory sizes based on dtype, especially in barrier synchronization code.

    Example:
    ```python
    # Instead of hardcoding size values
    tlx.barrier_expect_bytes(barrier, 2 * BLOCK_M * BLOCK_K)  # Assumes float16

    # Use size_of for dtype-aware computation
    tlx.barrier_expect_bytes(barrier,
                           tlx.size_of(tlx.dtype_of(desc)) * BLOCK_M * BLOCK_K)
    ```

- `tlx.clock64()`

    Returns the current 64-bit hardware clock value. E.g,
    ```
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    ```

- `tlx.stoch_round(src, dst_dtype, rand_bits)`

    Performs hardware-accelerated stochastic rounding for FP32→FP8/BF16/F16 conversions on Blackwell GPUs (compute capability ≥ 100). Uses PTX `cvt.rs.satfinite` instructions for probabilistic rounding.

    **Why Use Stochastic Rounding:**
    - Reduces bias in low-precision training/inference by randomly rounding up or down
    - Improves numerical accuracy compared to deterministic rounding (e.g., round-to-nearest-even)
    - Particularly beneficial when accumulating many small updates in FP8/FP16

    **Performance Characteristics:**
    - Hardware-accelerated: Uses native Blackwell instructions (cvt.rs.satfinite)
    - Minimal overhead: Similar throughput to deterministic rounding
    - Memory bandwidth: Requires additional random bits (uint32 per element)

    Parameters:
    - `src`: Source FP32 tensor
    - `dst_dtype`: Destination dtype (FP8 E5M2, FP8 E4M3FN, BF16, or FP16)
    - `rand_bits`: Random bits (uint32 tensor) for entropy, same shape as src
      - **Important:** Use `n_rounds=7` with `tl.randint4x()` for sufficient entropy
      - Fewer rounds may result in biased rounding behavior
      - Different seeds produce different rounding decisions for better statistical properties

    Example:
    ```python
        # Generate random bits for entropy
        # n_rounds=7 provides sufficient randomness for unbiased stochastic rounding
        offsets = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, offsets, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)

        # Apply stochastic rounding
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits)
    ```

- `tlx.vote_ballot_sync(mask, pred)`

    Collects a predicate from each thread in the warp and returns a 32-bit
    mask where each bit represents the predicate value from the corresponding
    lane. Only threads specified by `mask` participate in the vote.
    ```
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
    ```

## Kernels Implemented with TLX

### GEMM kernels
[Pipelined GEMM on Hopper](third_party/tlx/tutorials/hopper_gemm_pipelined_test.py)

[Warp-specialized GEMM on Hopper](third_party/tlx/tutorials/hopper_gemm_ws_test.py)

[Warp-specialized GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_ws.py)

[Grouped GEMM on Blackwell](third_party/tlx/tutorials/blackwell_grouped_gemm_test.py)

[Pipelined GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_pipelined.py)

[CLC GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_clc.py)

[2-CTA GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_2cta.py)

### Attention kernels

[Warp-specialized pipelined persistent FA fwd/bwd on Blackwell](third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent_test.py)

[Warp-Specialized computation-pipelined pingpong FA fwd on Hopper](third_party/tlx/tutorials/hopper_fa_ws_pipelined_pingpong_test.py)




## Build and install TLX from source

```
git clone https://github.com/facebookexperimental/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Run the tutorials after the build finishes, e.g,
```
python third_party/tlx/tutorials/hopper_fa_ws_pipelined_pingpong_test.py
```

To run Blackwell GEMM tutorial kernels, you can use the following command:

## Change 2: One correctness test script

`[TLX_VERSION=<kernel_name>] pytest third_party/tlx/tutorials/testing/test_correctness.py`

By default only one autotune config will be used by correctness test.

## Change 3: One performance test script for each op {gemm, matmul} x {hopper, blackwell}

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|pipelined|clc|2cta}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]`

## More reading materials

[Barrier Support in TLX](third_party/tlx/doc/tlx_barriers.md  )

[TLX talk in 2025 Triton Developer Conference](third_party/tlx/doc/TLX-triton-conference.pdf)

[TLX talk in 2026 GPU Mode](third_party/tlx/doc/PerformanceOptimizationWithTLX.pdf)
