# TLX - Triton Low-level Language Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It offers intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, computation, and asynchronous control flow. TLX is designed for expert users pushing Triton closer to the metal.

Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:

- Hardware-specific intrinsics (e.g., wgmma, async_copy, barrier)
- Shared and local memory allocation
- Instruction-level scheduling and control
- Cross-warpgroup synchronization


While this approach places more responsibility on the user, it reduces the compiler's role as a performance bottleneck. Although it may introduce divergence across hardware platforms, it empowers users to perform deeper, architecture-specific optimizations without relying solely on compiler heuristics.


## Gluon support

[Gluon](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon)
is Triton's experimental, lower-level layout-aware frontend. It lives under
`python/triton/experimental/gluon/` and is **upstream-synced**: the sources here are
imported from upstream and any local edit is overwritten on the next sync. Do not do
Gluon feature work, bug fixes, or debugging in this fork — send those upstream.

Gluon is **not a first-class supported DSL** for this repo (TLX remains the focus).
However, because fbtriton also serves as a secondary Triton for the community, we
maintain **fundamental Gluon CI coverage** so that our fork does not silently break
the Gluon frontend:

- **What runs.** A curated, critical subset of the Gluon *frontend* tests in
  `python/test/gluon/test_frontend.py`. These are compile-only (Python DSL → TTGIR)
  and target-agnostic — a single run exercises Ampere/Hopper/Blackwell **and** AMD
  (CDNA/RDNA) codegen frontends via a mock `GPUTarget`, so no matching physical GPU is
  required. This is the least-flaky, highest-signal slice of the Gluon suite.
- **Where.** Nightly + per-PR on both accelerators, via the `b200-gluon-test`
  (`.github/workflows/b200.yml`) and `mi350-gluon-test` (`.github/workflows/mi350.yml`)
  jobs. The curated set is selected with a pytest `-k` filter in those workflows.
- **Frontend status.** After the fixes below, `test_frontend.py` is **176 / 186**
  passing. The curated CI `-k` set is a subset of those 176 (green on every target).
- **Not covered (yet).** The GPU-execution Gluon suites (`test_core.py`,
  `test_lowerings.py`, `test_consan.py`, `test_fpsan.py`,
  `test_layout_format_view.py`) require a live device, are arch-gated / flakier, and
  currently have widespread failures on this fork (see gaps below); they are out of
  scope for this minimal signal.

### Fixed as part of enabling this CI

Fork-side (non-Gluon) fixes — the upstream-synced Gluon frontend was **not** modified:

- **`tl.reduce` / `reduction_ordering`** (`python/triton/language/core.py`): forwards the
  FB-local `reduction_ordering` kwarg only to semantics that accept it, so Gluon
  reductions work (`GluonSemantic.reduction` doesn't take it).
- **`dot()` `allow_tf32`** (`python/triton/language/semantic.py`): the FB-local (TLX)
  rewrite made `allow_tf32` a *required positional* arg, breaking the upstream Gluon
  callers that omit it (`gluon/language/amd/_ops.py`, AMD wmma/mfma/warp_pipeline). Now
  defaulted (`dot_precheck` already treats `None` as unspecified) — purely additive.
- **`test_core.py` collection** (allowed test edit): a bad cherry-pick (`#8480`) left a
  stray duplicate of the `test_descriptor_shape` body dangling inside the
  scatter/gather test, an `IndentationError` that made the whole file uncollectable on
  `main` and here. Removed the stray lines and restored the intended output assertion.
- **Frontend goldens regenerated** (allowed test edit, with caveat): `EXPECTTEST_ACCEPT=1`
  refreshed the `assert_expected_inline` strings that drifted from our fork's IR (mostly
  `reduction_ordering` mangling + version skew). These live in the upstream-synced
  `test_frontend.py` and **will be overwritten on the next sync** — that's expected.

### Known gaps / TODO(gluon-ci)

Not simple frontend fixes; tracked rather than resolved here:

- **Frontend, per-target goldens (~9)** — `test_nv_tma_descriptor_load/store`,
  `test_amd_mfma`, `test_amd_wmma_scaled_scalar`, `test_amd_warp_pipeline` emit IR that
  legitimately differs per parametrized target (Hopper vs Blackwell; CDNA vs RDNA), so a
  single inline golden cannot satisfy all params. Needs per-target goldens or an
  upstream test-structure change.
- **`create_lds_barrier_wait` binding mismatch** — `test_amd_mbarrier` /
  `test_amd_tdm_load_mbarrier`: a C++ pybind signature mismatch needing a binding/rebuild.
- **Execution suites — `distributed_type` divergence** — `test_core.py` and
  `test_fpsan.py` fail en masse with `expected ... to be a distributed_type but got
  <..., fp32>` from `ttgl` broadcast/convert_layout, a fork-vs-upstream Gluon type-system
  divergence (Gluon internals are upstream-synced, so the fix belongs upstream). Plus
  `test_consan.py` (expected-CUDA-failure / MLIR PassManager errors, inherently flaky)
  and `test_lowerings.py` (`test_reduce_layouts` is a very large parametrize sweep).
- **`test_layout_format_view.py::test_format_view_kernel`** — `'block_type' object has
  no attribute 'layout'` (Gluon type API skew); the other 15 tests pass.


## The DSL Extension

> **Hardware availability tags.** Each op below is tagged with the targets it runs on:
> `**[Hopper+]**` = NVIDIA Hopper and newer; `**[MI300+]**` = AMD MI300 (CDNA3) and MI350 (CDNA4);
> `**[MI350]**` = AMD MI350 (CDNA4) only. A trailing `?` (e.g. `**[MI300+?]**`) marks AMD availability
> that has **not been confirmed yet** and needs verification.
> Note: async copies (`async_load` and its commit/wait groups) and the AMD buffer ops require **MI350** —
> they are **not** available on MI300. `barrier_arrive` on AMD requires `arrive_count == 1`.

### Local buffer operations

- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS)` **[Hopper+, MI300+]**

    Allocate `NUM_BUFFERS` buffers in local memory per thread block, each of the specified size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, tlx.storage_kind.tmem)` **[Blackwell]**

    Allocate `NUM_BUFFERS` of buffers in the tensor memory per thread block, each with size size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, reuse=other_buffers)` **[Hopper+, MI300+]**

    Alias this allocation to an existing `buffered_tensor` so multiple logical buffers reuse the same underlying local storage (SMEM or TMEM) without reallocation.


- `buffer = tlx.local_view(buffers, buffer_idx)` or `buffer = buffers[buffer_idx]` **[Hopper+, MI300+]**

    Return a subview of the buffer indexed by `buffer_idx` from `buffers`. Both the explicit `local_view()` call and the indexing syntax `[]` are supported.


- `distributed_tensor = tlx.local_load(buffer, optional_token)` **[Hopper+, MI300+]**

    Loads the buffer from local memory or tensor memory into a distributed tensor.


- `tlx.local_store(buffer, distributed_tensor)` **[Hopper+, MI300+]**

    Store a distributed tensor into a buffer in local memory or tensor memory.

- `distributed_tensor = tlx.local_gather(src, indices, axis, optional_token)` **[Hopper+, MI300+?]**

    Gather elements from shared memory along a specified axis using an indices tensor. The output shape matches the indices shape, and elements are gathered from `src` at positions specified by `indices` along the given `axis`.

- `tlx.local_scatter(dst, src, indices, axis, optional_token)` **[Hopper+, MI300+?]**

    Scatter elements to shared memory along a specified axis using an indices tensor. Elements from `src` are written to `dst` at positions specified by `indices` along the given `axis`.

- `buffer = tlx.local_trans(buffer, dims)` **[Hopper+, MI300+]**

    Permutes the dimensions of a tensor.

- `buffer = tlx.local_slice(buffer, offsets=[m, n], shapes=[M, N])` **[Hopper+, MI300+]**

    Slice a `M x N` tensor at a `m x n` offset.

#### Buffer Reuse

TLX provides you the ability to reuse the same allocated buffer across multiple disjoint steps in your kernel. This is
useful to allow additional pipelining when you may not have enough isolated SMEM or TMEM.

- `tlx.storage_alias_spec(storage=storage_kind)` **[Hopper+, MI300+]**

    Defines a buffer that you will want to share across multiple aliases. The storage
    can be either SMEM or TMEM. To use this in an allocation you should provide the spec in the `reuse`
    argument for `local_alloc`. Here is the example from the FA kernel.

```
# Create the storage alias spec for all shared buffers. Cannot be directly
# indexed.
qk_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)

# Allocate all buffers referencing the same spec
qk_tiles = tlx.local_alloc(
    (BLOCK_M_SPLIT, BLOCK_N), qk_dtype, NUM_MMA_GROUPS,
    tlx.storage_kind.tmem, reuse=qk_storage_alias,
)
p_tiles = tlx.local_alloc(
    (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES), tlx.dtype_of(desc_v),
    NUM_MMA_GROUPS * NUM_MMA_SLICES, tlx.storage_kind.tmem,
    reuse=qk_storage_alias,
)
alpha_tiles = tlx.local_alloc(
    (BLOCK_M_SPLIT, 1), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
    tlx.storage_kind.tmem, reuse=qk_storage_alias,
)
l_tiles = tlx.local_alloc(
    (BLOCK_M_SPLIT, 1), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
    tlx.storage_kind.tmem, reuse=qk_storage_alias,
)
m_tiles = tlx.local_alloc(
    (BLOCK_M_SPLIT, 1), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
    tlx.storage_kind.tmem, reuse=qk_storage_alias,
)
```

- `tlx.reuse_group(*tensors, group_type=REUSE_TYPE, group_size=SUBTILE_SIZE)` **[Hopper+, MI300+]**

    A reuse group expresses how you intend to access the shared buffer.
    There are two types: Shared or Distinct. A shared buffer wants to occupy the same memory
    and each index should not be accessed at the same time. A distinct buffer will be accessible
    at the same index at the same time. The compiler will isolate buffer locations and potentially
    expand the buffer allocation to enforce this guarantee, which is helpful with buffers of unequal
    sizes.

    The group_size is used to enable subtiling a buffer. This ensures that for every 1 index
    of a buffer that SUBTILE_SIZE indices of this other buffer/group can be accessed.  Reuse groups
    can be nested to allow expressing more complex relationships. Currently a reuse group
    is not applied unless you assign it to a buffer with `spec.set_buffer_overlap`.

    Here is the example implementation for Flash Attention. In this kernel as the comment suggests,
    QK is shared with P, l, m, and alpha, and P is potentially subtiling.

```
# Define the buffer overlap strategy:
#   QK : |                                                   BLK_M/2 * BLOCK_N * fp32                         |
#   P:   |  BLK_M/(2*SLICES) * fp16| BLK_M/(2*SLICES) * fp16|...
# Alpha:                                                        |BLK_M/2*1*fp32|
#   l  :                                                                        |BLK_M/2*1*fp32|
#   m  :                                                                                       |BLK_M/2*1*fp32|
qk_storage_alias.set_buffer_overlap(
    tlx.reuse_group(
        qk_tiles,
        tlx.reuse_group(
            tlx.reuse_group(p_tiles, group_size=NUM_MMA_SLICES),
            alpha_tiles, l_tiles, m_tiles,
            group_type=tlx.reuse_group_type.distinct,
        ),
        group_type=tlx.reuse_group_type.shared,
    )
)
```

**Compiler Pipeline Inspection Steps**
To introspect the pipeline `add_stages`, before running your kernels, simply set
the add_stages_inspection_hook like so:

```python
def inspect_stages(_self, stages, options, language, capability):
    # inspect or modify add_stages here
triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```
Examples of how to use this for out of tree plugin passes is [here](lib/Plugins/README.md)

Binary wheels are available for CPython 3.10-3.14.

### Remote buffer operations

- `buffer = tlx.remote_view(buffer, remote_cta_rank)` **[Hopper+]**

  Return a remote view of the `buffer` living in another CTA in the same cluster with ID `remote_cta_rank`. NOTE: for
  now we only support barrier as `buffer`, not general SMEM.

- `tlx.remote_shmem_store(dst, src, remote_cta_rank)` **[Hopper+]**

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


- `tlx.async_descriptor_load(desc, buffer, offsets, barrier, pred=None, cache_modifier="", eviction_policy="", multicast_targets=[])` **[Hopper+]**

   Load a chunk of data from global memory into a local memory buffer using TMA. The global address, strides, and buffer size are defined by the tensor descriptor. A barrier object is provided and signaled upon completion of the operation.

   **Parameters:**
   - `desc`: Tensor descriptor for the source
   - `buffer`: Destination buffer in shared memory
   - `offsets`: List of offsets for each dimension
   - `barrier`: mbarrier to signal upon completion
   - `pred`: Optional predicate to guard the load
   - `cache_modifier`: Cache modifier hint (e.g., `""`, `"evict_first"`)
   - `eviction_policy`: L2 cache eviction policy (`""`, `"evict_first"`, `"evict_last"`)
   - `multicast_targets`: Optional list of multicast targets for cluster-wide loads

- `tlx.async_descriptor_prefetch_tensor(memdesc, [offsets], pred, eviction_policy)` **[Hopper+]**

   Hint hardware to load a chunk of data from global memory into a L2 cache to prepare for upcoming `async_descriptor_load` operations.

- `tlx.async_descriptor_store(desc, source, offsets, eviction_policy="", store_reduce="")` **[Hopper+]**

   Store a chunk of data from shared memory into global memory using TMA. The global address, strides, and buffer size are defined by the tensor descriptor.

   Supports optional atomic reduction (`store_reduce`) and L2 cache eviction hints (`eviction_policy`). Both regular stores and atomic reduce stores support cache eviction policies.

   **Parameters:**
   - `desc`: Tensor descriptor for the destination
   - `source`: Source buffer in shared memory
   - `offsets`: List of offsets for each dimension
   - `eviction_policy`: L2 cache eviction policy (`""`, `"evict_first"`, `"evict_last"`)
   - `store_reduce`: Atomic reduction kind (`""`, `"add"`, `"min"`, `"max"`, `"and"`, `"or"`, `"xor"`)

   **Example:**
   ```python
   # Regular TMA store with L2 evict_first hint
   tlx.async_descriptor_store(desc_c, c_buf[0], [offs_m, offs_n], eviction_policy="evict_first")

   # TMA atomic reduce-add with L2 evict_first hint
   tlx.async_descriptor_store(desc_c, c_buf[0], [offs_m, offs_n],
                              eviction_policy="evict_first", store_reduce="add")
   ```


- `tlx.async_remote_shmem_store(dst, src, remote_cta_rank, barrier)` **[Hopper+]**

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
- `tlx.remote_shmem_copy(dst, src, remote_cta_rank)` **[Hopper+]**

  Store a local shared memory buffer into a buffer in the remote shared memory of a cluster asynchronously.

  **Parameters:**
  - `dst`: The destination buffer in local shared memory (will be internally mapped to the remote CTA)
  - `src`: The source distributed tensor to store
  - `remote_cta_rank`: The rank (unique ID) of the remote CTA within the cluster
  - `barrier`: mbarrier to signal when the store completes (will be internally mapped to the remote CTA)

  **Example:**
  ```python
  # Allocate shared memory buffer
  buffer0 = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)
  buffer1 = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float16, 1)
  barrier = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

  # Copy to remote CTA's shared memory
  tlx.remote_shmem_store(buffer0[0], buffer1[0], remote_cta_rank=1, barrier=barrier[0])
  ```

- `desc_ptrs = tlx.allocate_tensor_descriptor(num)` **[Hopper+]**

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

- `tlx.make_tensor_descriptor(desc_ptr, base, shape, strides, block_shape, padding_option)` **[Hopper+]**

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

- `desc = tlx.reinterpret_tensor_descriptor(desc_ptr, block_shape, dtype)` **[Hopper+, MI300+]**

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

- `tlx.async_load(tensor_ptr, buffer, optional_mask, optional_other, cache_modifier, eviction_policy, is_volatile)` **[Hopper+, MI350]**

   Load a chunk of data from global memory into a local memory buffer asynchronously.

   The operation returns a token object which can be used to track the completion of the operation.


- `tlx.async_load_commit_group(tokens)` **[Hopper+, MI350]**

   Commits all prior initiated but uncommitted async_load ops an async group. Optionally, each token represents a tracked async load operation.

- `tlx.async_load_wait_group(pendings, tokens)` **[Hopper+, MI350]**

   Wait for completion of prior asynchronous copy operations. The `pendings` argument indicates the number of in-flight operations not completed.
   Optionally, each token represents a tracked async commit group operation.


### Async tensor core operations

- `acc = tlx.async_dot(a[i], b[i], acc)` **[Hopper+]**
- `acc = tlx.async_dot(a_reg, b[i], acc)` **[Hopper]**
- `acc[i] = tlx.async_dot(a[i], b[i], acc[i], barrier)` **[Blackwell]**
- `acc[i] = tlx.async_dot_scaled(a[i], b[i], acc[i], a_scale[i], a_format, b_scale[i], b_format, use_acc, two_ctas, mBarriers)` **[Blackwell]**

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

    Use `tlx.tmem_copy` **[Blackwell]** to efficiently transfer scale data from shared memory to tensor memory:

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

- `acc = tlx.async_dot_wait(pendings, acc)` **[Hopper+]**

    Wait for completion of prior asynchronous dot operations. The pendings argument indicates the number of in-flight operations not completed.

    Example:
    ```python
    acc = tlx.async_dot(a_smem, b_smem)
    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
    tl.store(C_ptrs, acc)
    ```

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

    Wait until `num_threads` threads have reached the specified named mbarrier phase.

- `tlx.named_barrier_arrive(bar_id, num_threads)` **[Hopper+]**

    Signal arrival at a named mbarrier with the given thread count.

- `tlx.barrier_expect_bytes(bar, bytes)` **[Hopper+]**

  Signal a barrier of an expected number of bytes to be copied.

- `tlx.barrier_arrive(bar, arrive_count=1, remote_cta_rank=None)` **[Hopper+]**

    Perform the arrive operation on an mbarrier. If `remote_cta_rank` is provided, signals the barrier in the specified remote CTA's shared memory (useful for multi-CTA synchronization).

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


### Warp Specialization operations

- `tlx.async_tasks` and `tlx.async_task` **[Hopper+]**

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
| `num_regs` | Number of registers per thread (optional, for register allocation tuning). When provided, it must be divisible by 8. |
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

- `tlx.cluster_cta_rank()` **[Hopper+]**

  Returns the rank (unique ID) of the current CTA within the cluster.

- `tlx.thread_id(axis)` **[Hopper+]**

    Returns the id of the current thread instance along the given `axis`.

- `tlx.dtype_of(v)` **[Hopper+]**

    Returns the dtype of a tensor or tensor descriptor.

- `tlx.size_of(dtype)` **[Hopper+]**

    Returns the size in bytes of a given Triton dtype. This is useful for dynamically computing memory sizes based on dtype, especially in barrier synchronization code.

    Example:
    ```python
    # Instead of hardcoding size values
    tlx.barrier_expect_bytes(barrier, 2 * BLOCK_M * BLOCK_K)  # Assumes float16

    # Use size_of for dtype-aware computation
    tlx.barrier_expect_bytes(barrier,
                           tlx.size_of(tlx.dtype_of(desc)) * BLOCK_M * BLOCK_K)
    ```

- `tlx.clock64()` **[Hopper+]**

    Returns the current 64-bit hardware clock value. E.g,
    ```
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    ```

- `tlx.stoch_round(src, dst_dtype, rand_bits)` **[Blackwell]**

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

- `tlx.vote_ballot_sync(mask, pred)` **[Hopper+]**

    Collects a predicate from each thread in the warp and returns a 32-bit
    mask where each bit represents the predicate value from the corresponding
    lane. Only threads specified by `mask` participate in the vote.
    ```
        ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
    ```

- `tlx.prefetch(pointer, level="L2", mask=None, tensormap=False)` **[Hopper+]** issues a non-blocking prefetch hint for pointer-based scattered/gather loads. This complements `tlx.async_descriptor_prefetch_tensor` (which works on TMA tensor descriptors) by supporting raw pointer tensors.
  Additionally, if `tensormap` is specified to `True`, the API instead does a prefetch of tensor map object (TMA descriptor) and ignores other parameters other than `pointer`.

  | Level | PTX | Description |
  |-------|-----|-------------|
  | `"L1"` | `prefetch.global.L1` | Prefetch into L1 and L2 cache |
  | `"L2"` | `prefetch.global.L2` | Prefetch into L2 cache only (default) |

  Example:
  ```python
  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  tlx.prefetch(input_ptr + offsets, level="L2", mask=mask)
  x = tl.load(input_ptr + offsets, mask=mask)

  ...
  # desc_in can be host side descriptor or device side like this:
  desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
  tlx.prefetch(desc_in, tensormap=True)
  ```

- `tlx.dump_layout(x)` **[Hopper+, MI300+]**

    Compile-time diagnostic that prints the resolved layout of a value to the
    compiler log. `x` may be a register tensor or a shared/tensor-memory buffer
    (memdesc). It emits **no** device code and returns nothing — this is a
    static, host-side diagnostic, distinct from the runtime `tl.device_print` /
    `tl.print`. The op is rendered at the end of the TTGIR pipeline, so the
    printed layout reflects all compiler optimizations, then it is erased.

    The layout is printed in CuTe (CUTLASS) `Shape:Stride` notation (`_N` marks
    a static integer):
    - Register tensors → a thread-value (TV) layout
      `((thread...),(value...)):((thread...),(value...))`, where the thread
      group comes from the hardware lane/warp/block dims and the value group
      from the per-thread registers (stride `_0` denotes a broadcast).
    - Shared/tensor-memory buffers → a single strided layout, e.g. `_64:_1`.
    - Swizzled shared buffers → `Swizzle<B,M,S> o (base):(stride)`.

    In all cases the layout maps a coordinate to the **logical tensor's
    row-major element index** (its codomain): for a register tensor a
    `(thread, value)` coordinate → the logical element index it holds, and for
    a buffer an offset → the buffer element index. The strides are offsets in
    that logical index space, not physical byte/bank addresses.

    Layouts that are not representable as a CuTe layout fall back to the raw
    linear-layout string.

    Example:
    ```python
    x = tl.load(x_ptr + offs)          # register tensor
    tlx.dump_layout(x)                  # -> // cute: ((_32,_2,_2),_1):((_1,_32,_0),_0)

    buf = tlx.local_alloc((BLOCK,), tl.float32, 1)
    v = tlx.local_view(buf, 0)
    tlx.dump_layout(v)                  # -> // cute: _64:_1
    ```


## Buffer Operations (AMD)

> **[MI350]** — available on AMD MI350 (CDNA4) only; not available on MI300.

Buffer operations access global memory via a scalar base pointer and a tensor of i32 element offsets, rather than a tensor of pointers. This maps directly to AMD's hardware buffer instructions, which use a resource descriptor and byte offsets, enabling the hardware to do out-of-bounds checking and cache optimization.

### `tlx.buffer_load`

Load a tensor of values from global memory.

```python
result = tlx.buffer_load(ptr, offsets, mask=None, other=None, cache=None)
```

| Argument | Type | Description |
|----------|------|-------------|
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not loaded. |
| `other` | tensor or scalar, optional | Value used for masked-out elements (where `mask[i]` is `False`). |
| `cache` | str, optional | Cache modifier (e.g. `".ca"`, `".cg"`). |

**Returns**: A tensor with the same shape as `offsets` and element type matching the pointee type of `ptr`.

Lowers to `amdg.buffer_load`, which is eventually lowered to `rocdl.raw.ptr.buffer.load`.

### `tlx.buffer_store`

Store a tensor of values to global memory.

```python
tlx.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)
```

| Argument | Type | Description |
|----------|------|-------------|
| `stored_value` | tensor | Values to write. |
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not written. |
| `cache` | str, optional | Cache modifier. |

**Returns**: Nothing.

Lowers to `amdg.buffer_store`, which is eventually lowered to `rocdl.raw.ptr.buffer.store`.

### `tlx.buffer_load_to_local`

Async load from global memory directly into shared (local) memory, bypassing registers. This is useful for producer warps that prefetch data into shared memory for other warps to consume.

```python
token = tlx.buffer_load_to_local(dest, ptr, offsets, mask=None, other=None, cache_modifier="")
```

| Argument | Type | Description |
|----------|------|-------------|
| `dest` | `tlx.buffered_tensor` | Destination slice in shared memory. |
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not loaded. |
| `other` | tensor or scalar, optional | Value used for masked-out elements. |
| `cache_modifier` | str, optional | Cache modifier string (default `""`). |

**Returns**: A `tlx.async_token` that can be used with `tlx.async_load_wait_group()` to synchronize on the completion of the transfer.

Lowers to `amdg.buffer_load_to_local`, which is eventually lowered to `rocdl.raw.ptr.buffer.load.async.lds` — a single hardware instruction that moves data from global memory to LDS without going through VGPRs.

### Example

```python
import triton.language.extra.tlx as tlx

@triton.jit
def kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = offsets < BLOCK_SIZE

    # Load from global memory using buffer semantics
    data = tlx.buffer_load(src_ptr, offsets, mask=mask, other=0.0)

    # Store to global memory using buffer semantics
    tlx.buffer_store(data, dst_ptr, offsets, mask=mask)
```

For the async global-to-shared variant, see the warp-pipeline GEMM example (`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`).

## AMD TDM Descriptor Loads

`tlx.async_amd_descriptor_load(desc, result, offsets, pred=None)` issues an AMD
TDM descriptor load from global memory to a TLX local buffer. It is available on
TDM-capable AMD targets (`gfx1250+`) and should be synchronized with
`tlx.async_amd_descriptor_wait`.

`tlx.async_amd_descriptor_load_group(descs, results, offsets, warp_masks,
preds=None)` groups multiple AMD TDM descriptor loads behind one static hardware
TDM instruction. Each list entry is one arm:

| Argument | Description |
|----------|-------------|
| `descs[i]` | Tensor descriptor for arm `i`. |
| `results[i]` | Local buffer or local view receiving arm `i`. |
| `offsets[i]` | Offset list for arm `i`; all arms must have the same rank. |
| `warp_masks[i]` | Bitmask selecting the waves that use arm `i`. |
| `preds[i]` | Optional predicate for arm `i`; defaults to true. |

The warp masks must be non-empty, disjoint, axis-aligned, and cover all waves in
the CTA exactly once. The grouped operation currently requires one CTA, the same
rank and element bitwidth for every arm, one shared cache modifier, and shared
layouts supported by AMD TDM lowering. This is useful for kernels where
different wave groups load different inputs, such as A/B GEMM tiles or A/B plus
scale tiles in MXFP GEMM, while keeping the assembly to one TDM instruction per
load group.

Example:
```python
a_tok = tlx.async_amd_descriptor_load_group(
    [a_desc, b_desc],
    [tlx.local_view(a_buf, slot), tlx.local_view(b_buf, slot)],
    [[off_m, k * BLOCK_K], [k * BLOCK_K, off_n]],
    [0b0011, 0b1100],
)
tlx.async_amd_descriptor_wait(0, [a_tok])
```

## Warp Pipeline (AMD)

> **[MI350]** — AMD MI350 (CDNA4); not available on MI300.

`tlx.warp_pipeline_stage(label, *, priority=None)` is a context manager that marks explicit pipeline stage boundaries inside a loop. The compiler partitions the loop body at these boundaries and inserts conditional barriers so that one warp group executes one stage ahead of the other, overlapping memory latency with compute.

**This is an explicit partitioning marker, not an automatic optimization.** Correctness depends on the user's buffering and synchronization structure. In particular:
- Use multi-buffered shared memory (typically triple buffering with `NUM_BUFFERS=3`) to prevent data races between warp groups accessing the same buffer.
- Use explicit `tlx.async_load_wait_group()` to ensure data is ready before consumption.
- Handle prologue (prefetch) and epilogue (drain) around the main loop.

See the gfx1250 warp-pipeline GEMM example (`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`) for the full pattern.

| Parameter | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Stage name for diagnostics (e.g. `"load"`, `"compute"`) |
| `priority` | `int` (0-3), optional | Hardware scheduling hint, maps to `s_setprio`. Higher = more urgent. |

Auto software pipelining is automatically disabled on loops that contain warp pipeline stages.

Example (simplified — see gfx1250 example for production pattern):
```python
import triton.language.extra.tlx as tlx

@triton.jit
def gemm_kernel(..., BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr):
    buf_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS)
    buf_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS)

    # Prologue: prefetch NUM_BUFFERS-1 tiles into shared memory
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        tlx.async_load(a_ptrs, tlx.local_view(buf_A, i), mask=...)
        tlx.async_load(b_ptrs, tlx.local_view(buf_B, i), mask=...)
        tlx.async_load_commit_group()
        a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    tlx.async_load_wait_group(NUM_BUFFERS - 2)

    # Main loop with warp pipelining
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(NUM_BUFFERS - 1, K_ITERS):
        consumer = (k - (NUM_BUFFERS - 1)) % NUM_BUFFERS
        producer = k % NUM_BUFFERS
        with tlx.warp_pipeline_stage("lds_load", priority=1):
            a_tile = tlx.local_load(tlx.local_view(buf_A, consumer))
            b_tile = tlx.local_load(tlx.local_view(buf_B, consumer))
        tlx.async_load_wait_group(0)
        with tlx.warp_pipeline_stage("compute_and_load", priority=0):
            tlx.async_load(a_ptrs, tlx.local_view(buf_A, producer), mask=...)
            tlx.async_load_commit_group()
            acc = tl.dot(a_tile, b_tile, acc)

    # Epilogue: drain remaining buffers
    ...
```

## Scaled Dot (AMD)

`tlx.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, *, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32, tiles_per_warp=None)`
is a thin wrapper around `tl.dot_scaled`. Without `tiles_per_warp` it is exactly
equivalent to `tl.dot_scaled` — pass it only when you need the AMD-specific
WMMA scheduling hint described below.

### `tiles_per_warp` — what it controls

| Concept | Controlled by | What it means |
|---------|---------------|---------------|
| **Warp distribution** | `warpsPerCTA` (chosen automatically by `AccelerateAMDMatmul::planWarps`) | How the total result tile is split *across* warps along M/N. |
| **Per-warp tiling** | `tiles_per_warp` (this hint) | How many `instrShape`-sized WMMA tiles *each warp* covers contiguously before the layout repeats. |

So `tiles_per_warp=[2, 2]` does **not** mean "distribute 4 tiles across 4
warps." It means *each* warp emits a 2×2 block of WMMA instruction tiles,
holding the corresponding 2×2 accumulator registers. Concretely, for a
`tt.dot_scaled` lowered to gfx1250 WMMA (`instrShape = [16, 16, K]`),
4 warps, `warpsPerCTA = [2, 2]`:

| `tiles_per_warp` | Per-warp coverage (M × N) | Per-CTA coverage before repeat (M × N) |
|------------------|---------------------------|----------------------------------------|
| `[1, 1]` (default) | `16 × 16` | `32 × 32` |
| `[2, 2]`           | `32 × 32` | `64 × 64` |

For a `256 × 256` result, `[1, 1]` repeats the layout `8 × 8` times,
`[2, 2]` repeats it `4 × 4`. Larger `tiles_per_warp` gives each warp more
contiguous accumulator state (better register reuse for preshuffled
MXFP scales, fewer warp-level reductions), at the cost of more registers
per warp.

Together, `instrShape`, `warpsPerCTA`, and `tiles_per_warp` define the M/N
extent of one CTA-level WMMA layout period:
`period[d] = instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]`. If the result
tile is larger than this period, the period repeats. The K entry of
`instrShape` is the per-instruction reduction depth and is handled separately
from this M/N tiling.

`tiles_per_warp` is validated by `AccelerateAMDMatmul`: it must have one
entry per result-tile dim, each entry must be positive, and
`instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]` must fit in the result
tile shape.

### Example

```python
import triton.language.extra.tlx as tlx

acc = tlx.dot_scaled(
    a, a_scale, "e5m2",
    b, b_scale, "e5m2",
    acc,
    tiles_per_warp=[2, 2],   # pack 2x2 WMMA tiles per warp for preshuffled MXFP
)
```

### Mechanism (for IR-level users)

The wrapper attaches `amdg.wmma_tiles_per_warp = array<i32: m, n>` on the
resulting `tt.dot_scaled` op. `ScaledBlockedToScaledWMMAF8F6F4` reads the
attribute and substitutes `m, n` for the default `1, 1` when building the
WMMA encoding. Setting the attribute directly on a `tt.dot[_scaled]` op
in MLIR has the same effect; the wrapper just spares Python kernels from
hand-poking attributes.

Currently consumed only by the scaled-WMMA pattern (gfx1250). Regular
`tt.dot` WMMA and the MFMA patterns do not read it.

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

### AMD kernels (gfx950 / CDNA4)

[LDS-pipelined GEMM](third_party/tlx/tutorials/amd_gemm_pipelined.py)

[Warp-pipelined GEMM](third_party/tlx/tutorials/amd_gemm_warp_pipeline.py)

[Async-DMA Flash Attention fwd — simple / prefetch](third_party/tlx/tutorials/amd_fa_pipelined.py)

[Persistent Flash Attention fwd — XCD zig-zag, cross-attention / decode](third_party/tlx/tutorials/amd_fa_persistent.py)

[Rotated 4-cluster Flash Attention fwd](third_party/tlx/tutorials/amd_fa_cluster.py)

[Fused addmm + GLU (Gated Linear Unit: out = x + x*y, x = A@B + bias)](third_party/tlx/tutorials/amd_addmm_glu.py)

[IKBO Flash Attention (In-Kernel Broadcast Optimization, candidate/user broadcast)](third_party/tlx/tutorials/ikbo/ikbo_fa_triton.py)

[IKBO LCE (logit cross-entropy over candidate/user embeddings — not attention)](third_party/tlx/tutorials/ikbo/ikbo_lce_triton.py)

### AMD kernels (gfx1250)

[TDM-pipelined GEMM](third_party/tlx/tutorials/amd_tdm_gemm_pipelined.py)

[MXFP TDM-pipelined GEMM](third_party/tlx/tutorials/amd_mxfp_gemm_tdm_pipelined.py)




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

All kernels — Hopper, Blackwell, and AMD — share this one file; each test is
arch-gated with `@pytest.mark.skipif`, so on any given GPU only the relevant
cases run and the rest skip. To run just the AMD/IKBO cases:

`pytest third_party/tlx/tutorials/testing/test_correctness.py -k "amd or ikbo"`

(on gfx950 the gfx1250-only GEMM cases skip automatically).

## Change 3: One performance test script per op × arch (Hopper, Blackwell, AMD)

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|pipelined|clc|2cta}]`

`third_party/tlx/denoise.sh third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_persistent|ws_pipelined|ws_pipelined_persistent|clc}]`

`denoise.sh` wraps AMD runs too (it applies NUMA pinning and runs the
benchmark; the GPU clock/power lock is NVIDIA-only and is simply skipped on AMD).
gfx950 / CDNA4:

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_gemm_perf.py [--version {warp_pipeline|pipelined}]`

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_fa_perf.py [--version {simple|prefetch|persistent|cluster}]`

Without `--version`, AMD FA perf runs `simple`, `prefetch`, and `persistent`; select
`cluster` explicitly because it is D=128-only.

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_addmm_glu_perf.py [--version {tlx_baseline|tlx_simple_async|tlx_optimized_async|tlx_optimized|tlx_persistent}]`

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_fa_perf.py`  (IKBO Flash Attention)

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_lce_perf.py`  (IKBO LCE — distinct op, not attention)

gfx1250:

`third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_mxfp_gemm_perf.py [--transpose-b]`

## TLX-AMD CI

AMD tutorial kernels are exercised by `.github/workflows/mi350.yml` on a gfx950
(MI350 / CDNA4) runner, mirroring the H100 job in `.github/workflows/h100.yml`:

- **`mi350-tlx-test`** — TLX unit tests (`python/test/unit/language/test_tlx_*.py`)
  plus the tutorial correctness suite
  (`third_party/tlx/tutorials/testing/test_correctness.py`). AMD and IKBO cases
  run; Hopper/Blackwell and gfx1250 cases auto-skip via the arch gates.
- **`mi350-meta-triton-test`** — TritonBench performance coverage (the AMD perf
  scripts above are for local runs; perf-regression tracking lives in TritonBench).

Both run on push, PR, and the nightly schedule; nightly failures are filed as
issues via `report-nightly-failure.yml`.

## More reading materials

[Barrier Support in TLX](third_party/tlx/doc/tlx_barriers.md  )

[TLX talk in 2025 Triton Developer Conference](third_party/tlx/doc/TLX-triton-conference.pdf)

[TLX talk in 2026 GPU Mode](third_party/tlx/doc/PerformanceOptimizationWithTLX.pdf)

[TLX paper](https://arxiv.org/abs/2605.10905)
