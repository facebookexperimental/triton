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
