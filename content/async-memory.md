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
