## Local buffers

- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS)` **[sm90+, gfx942+]**

    Allocate `NUM_BUFFERS` buffers in local memory per thread block, each of the specified size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, tlx.storage_kind.tmem)` **[sm100]**

    Allocate `NUM_BUFFERS` of buffers in the tensor memory per thread block, each with size size. The memory layout is inferred from its consumers.


- `buffers = tlx.local_alloc(shape, dtype, NUM_BUFFERS, reuse=other_buffers)` **[sm90+, gfx942+]**

    Alias this allocation to an existing `buffered_tensor` so multiple logical buffers reuse the same underlying local storage (SMEM or TMEM) without reallocation.


- `buffer = tlx.local_view(buffers, buffer_idx)` or `buffer = buffers[buffer_idx]` **[sm90+, gfx942+]**

    Return a subview of the buffer indexed by `buffer_idx` from `buffers`. Both the explicit `local_view()` call and the indexing syntax `[]` are supported.


- `distributed_tensor = tlx.local_load(buffer, token=None, layout=None, relaxed=False, rematerialize_coordinates=False, rematerialize_coordinates_group=None)` **[sm90+, gfx942+]**

    Loads the buffer from local or tensor memory. `layout` pins a requested
    register layout. On AMD, the rematerialization options start fresh address
    coordinate live ranges either per load or for a named group of nearby
    loads; the two options are mutually exclusive.


- `tlx.local_store(buffer, distributed_tensor)` **[sm90+, gfx942+]**

    Store a distributed tensor into a buffer in local memory or tensor memory.

- `distributed_tensor = tlx.local_gather(src, indices, axis, optional_token)` **[sm90+, gfx942+?]**

    Gather elements from shared memory along a specified axis using an indices tensor. The output shape matches the indices shape, and elements are gathered from `src` at positions specified by `indices` along the given `axis`.

- `tlx.local_scatter(dst, src, indices, axis, optional_token)` **[sm90+, gfx942+?]**

    Scatter elements to shared memory along a specified axis using an indices tensor. Elements from `src` are written to `dst` at positions specified by `indices` along the given `axis`.

- `buffer = tlx.local_trans(buffer, dims)` **[sm90+, gfx942+]**

    Permutes the dimensions of a tensor.

- `buffer = tlx.local_slice(buffer, offset=[m, n], shape=[M, N])` **[sm90+, gfx942+]**

    Slice a tensor at the given logical offset. On gfx942+, SMEM offsets may be
    runtime scalar i32 tensors; `shape` remains constexpr. The caller must keep
    a runtime-offset view within the allocation and satisfy the same
    tile-alignment contract as a static slice; violating either condition is
    undefined behavior.


### Buffer Reuse

TLX provides you the ability to reuse the same allocated buffer across multiple disjoint steps in your kernel. This is
useful to allow additional pipelining when you may not have enough isolated SMEM or TMEM.

> For why this API replaced the older manual `reuse` parameter, and the design
> behind it, see
> [`storage_alias_spec` and `set_buffer_overlap`](third_party/tlx/doc/StorageAliasSpecAndSetBufferOverlap.md).

- `tlx.storage_alias_spec(storage=storage_kind, buffer_size_bytes=None)` **[sm90+, gfx942+]**

    Defines a buffer that you will want to share across multiple aliases. The storage
    can be either SMEM or TMEM (`smemCluster` is not supported). To use this in an
    allocation you should provide the spec in the `reuse`
    argument for `local_alloc`; that argument accepts either a `storage_alias_spec`
    or, for legacy callers, a `buffered_tensor`. `buffer_size_bytes` is an optional
    compile-time size — if omitted, the compiler uses the maximum across all
    referencing allocations. Here is the example from the FA kernel.

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

- `tlx.reuse_group(*tensors, group_type=REUSE_TYPE, group_size=SUBTILE_SIZE)` **[sm90+, gfx942+]**

    A reuse group expresses how you intend to access the shared buffer.
    There are two types: Shared or Distinct. A shared buffer wants to occupy the same memory
    and each index should not be accessed at the same time. A distinct buffer will be accessible
    at the same index at the same time. The compiler will isolate buffer locations and potentially
    expand the buffer allocation to enforce this guarantee, which is helpful with buffers of unequal
    sizes.

    The group_size is used to enable subtiling a buffer. This ensures that for every 1 index
    of a buffer that SUBTILE_SIZE indices of this other buffer/group can be accessed.  Reuse groups
    can be nested to allow expressing more complex relationships. Currently a reuse group
    is not applied unless you assign it to a buffer with `spec.set_buffer_overlap`,
    whose leaf nodes must all be `buffered_tensor`s allocated from that same spec.

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


## Remote buffers

- `buffer = tlx.remote_view(buffer, remote_cta_rank)` **[sm90+]**

  Return a remote view of the `buffer` living in another CTA in the same cluster with ID `remote_cta_rank`. NOTE: for
  now we only support barrier as `buffer`, not general SMEM.

- `tlx.remote_shmem_store(dst, src, remote_cta_rank)` **[sm90+]**

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
