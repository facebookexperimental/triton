- `tlx.async_descriptor_load(desc, buffer, offsets, barrier, pred=None, cache_modifier="", eviction_policy="", multicast_targets=[])` **[sm90+]**

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

- `tlx.async_descriptor_prefetch_tensor(memdesc, [offsets], pred, eviction_policy)` **[sm90+]**

   Hint hardware to load a chunk of data from global memory into a L2 cache to prepare for upcoming `async_descriptor_load` operations.

- `tlx.async_descriptor_store(desc, source, offsets, eviction_policy="", store_reduce="")` **[sm90+]**

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


- `tlx.async_remote_shmem_store(dst, src, remote_cta_rank, barrier)` **[sm90+]**

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
- `tlx.remote_shmem_copy(dst, src, remote_cta_rank)` **[sm90+]**

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

- `desc_ptrs = tlx.allocate_tensor_descriptor(num)` **[sm90+]**

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

- `tlx.make_tensor_descriptor(desc_ptr, base, shape, strides, block_shape, padding_option)` **[sm90+]**

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

- `desc = tlx.reinterpret_tensor_descriptor(desc_ptr, block_shape, dtype)` **[sm90+, gfx942+]**

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

- `tlx.async_load(tensor_ptr, buffer, optional_mask, optional_other, cache_modifier, eviction_policy, is_volatile)` **[sm90+, gfx950]**

   Load a chunk of data from global memory into a local memory buffer asynchronously.

   The operation returns a token object which can be used to track the completion of the operation.

   **MI350X caveat:** When `mask` is provided, callers should also provide `other`—typically `0.0` for numerical kernels. Elements for which `mask=True` are copied from global memory into the destination local buffer. For elements where `mask=False`, `other` is written only if it is provided. Otherwise, the corresponding LDS locations retain unspecified, potentially stale contents from an earlier use of the buffer.

   For example, in FlashAttention, when the sequence length is not a multiple of `BLOCK_N`, the final `V` tile contains masked-out rows. If `other` is omitted, those rows may contain unspecified values, including bit patterns representing NaN or infinity. These values can propagate through the subsequent matrix multiplication and produce incorrect output. Use `other=0.0` for such padded tiles.


- `tlx.async_load_commit_group(tokens)` **[sm90+, gfx950]**

   Commits all prior initiated but uncommitted async_load ops an async group. Optionally, each token represents a tracked async load operation.

- `tlx.async_load_wait_group(pendings, tokens)` **[sm90+, gfx950]**

   Wait for completion of prior asynchronous copy operations. The `pendings` argument indicates the number of in-flight operations not completed.
   Optionally, each token represents a tracked async commit group operation.


## `tlx.buffer_load_to_local`

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

**Requirements.** The direct-to-LDS copy has the following hardware constraints:
- **Vector width.** Each thread's load must reach a supported direct-to-LDS width (**32 or 128 bits**). If it can only be vectorized to a smaller width, it cannot be lowered.
- **Provable pointer/offset alignment.** The compiler must be able to *prove* the alignment that vector width needs.
- **Mask alignment.** If `mask` is given it must be aligned to the vector width: each group of (vector width) consecutive mask values must be identical. The copy transfers each lane's whole vector in one transaction, so a mask whose `True`/`False` boundary cannot be proven vector-aligned (e.g. `offs < K` for a runtime `K`) forces per-element vectorization and cannot lower.


## TDM descriptor loads

`tlx.update_tensor_descriptor(desc, add_offsets=None, set_bounds=None,
pred=None, clamp_bounds=False)` produces a positioned descriptor SSA value.
`add_offsets` advances the tile position without changing bounds;
`set_bounds` rewrites absolute bounds; and `pred` replaces the inherited
predicate. Use `clamp_bounds=True` with `add_offsets` to derive the remaining
OOB extent of an advanced tile.

`tlx.async_amd_descriptor_load(desc, result, offsets=None, pred=None,
clamp_bounds=True)` issues an AMD TDM descriptor load from global memory to a
TLX local buffer. If `offsets` is omitted, `desc` is used as already positioned
and its predicate is preserved. The analogous store accepts `offsets=None` for
the same reason. Both operations are available on TDM-capable AMD targets
(`gfx1250+`) and should be synchronized with
`tlx.async_amd_descriptor_wait`.

`tlx.async_amd_descriptor_load_fused(members, cache_modifier="")` fuses two to
four AMD TDM descriptor loads behind one static hardware instruction. Each
member is a `(positioned_desc, destination, warp_used_hint)` tuple. Position
descriptors with `tlx.update_tensor_descriptor` before issuing the fused load.
The hints must be non-empty, pairwise disjoint, and legal axis-aligned subsets
of the CTA's waves; they do not need to cover every wave. Members must have the
same descriptor rank but may have different shapes and element types. All
members share one cache modifier.

Example:
```python
a_load_desc = tlx.update_tensor_descriptor(
    a_desc, add_offsets=[off_m, k * BLOCK_K], pred=True, clamp_bounds=True)
b_load_desc = tlx.update_tensor_descriptor(
    b_desc, add_offsets=[k * BLOCK_K, off_n], pred=True, clamp_bounds=True)
a_tok = tlx.async_amd_descriptor_load_fused([
    (a_load_desc, tlx.local_view(a_buf, slot), 0b0011),
    (b_load_desc, tlx.local_view(b_buf, slot), 0b1100),
])
tlx.async_amd_descriptor_wait(0, [a_tok])
```
