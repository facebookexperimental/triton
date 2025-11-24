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

- `buffer = tlx.local_view(buffers, buffer_idx)`

    Return a subview of the buffer indexed by `buffer_idx` from `buffers`.


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

### Async memory access


- `tlx.async_descriptor_load(memdesc, buffer, [offsets], barrier, cache_modifier, eviction_policy, is_volatile)`

   Load a chunk of data from global memory into a local memory buffer. The global address, strides, and buffer size are defined by the memory descriptor. A barrier object is provided and signaled upon completion of the operation.


- `tlx.async_descriptor_store(memdesc, buffer, [offsets])`

   Store a chunk of data from local memory into global memory buffer. The global address, strides, and buffer size are defined by the memory descriptor.


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
- `acc[i] = tlx.async_dot_scaled(a[i], b[i], acc[i], a_scale[i], b_scale[i])`
- `acc = tlx.async_dot_wait(pendings, acc)`

    Wait for completion of prior asynchronous dot operations. The pendings argument indicates the number of in-flight operations not completed.

Examples
```
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
        with tlx.asycn_task(default)
            ...
        with tlx.asycn_task(num_warps = 4)
            ...
```
`tlx.async_tasks` opens a multi-tasking region where independent asynchronous tasks can be declared. Each task executes in parallel using a dedicated subset of warps within the thread block..

`tlx.async_task(default)` defines the default task, also known as the trunk. It uses the available warps not explicitly reserved by other tasks. .

`tlx.async_task(num_warps=4)` defines a warp-specialized asynchronous task that explicitly reserves 4 warps in addition to those used by the trunk task..

### Other operations

- `tlx.cluster_cta_rank()`

  Returns the rank (unique ID) of the current CTA within the cluster.

- `tlx.thread_id(axis)`

    Returns the id of the current thread instance along the given `axis`.

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

    Parameters:
    - `src`: Source FP32 tensor
    - `dst_dtype`: Destination dtype (FP8 E5M2, FP8 E4M3FN, BF16, or FP16)
    - `rand_bits`: Random bits (uint32 tensor) for entropy, same shape as src

    Example:
    ```python
        # Generate random bits for entropy
        offsets = tl.arange(0, BLOCK_SIZE // 4)
        r0, r1, r2, r3 = tl.randint4x(seed, offsets, n_rounds=7)
        rbits = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(x.shape)

        # Apply stochastic rounding
        y = tlx.stoch_round(x, tlx.dtype_of(y_ptr), rbits)
    ```


## Kernels Implemented with TLX

### GEMM kernels
[Pipelined GEMM on Hopper](third_party/tlx/tutorials/hopper-gemm-pipelined_test.py)

[Pipelined GEMM on Blackwell](third_party/tlx/tutorials/blackwell-gemm-pipelined.py)

[Warp-specialized GEMM on Hopper](third_party/tlx/tutorials/hopper-gemm-ws_test.py)

[Warp-specialized GEMM on Blackwell](third_party/tlx/tutorials/blackwell-gemm-ws.py)

### Attention kernels

[Warp-specialized FA fwd on Blackwell](third_party/tlx/tutorials/blackwell-fa-ws_test.py)

[Warp-specialized pipelined FA fwd on Blackwell](third_party/tlx/tutorials/blackwell-fa-ws-pipelined_test.py)

[Warp-specialized FA fwd on Hopper](third_party/tlx/tutorials/hopper-fa-ws_test.py)

[Warp-Specialized computation-pipelined FA fwd on Hopper](third_party/tlx/tutorials/hopper-fa-ws-pipelined_test.py)

[Warp-Specialized computation-pipelined pingpong FA fwd on Hopper](third_party/tlx/tutorials/hopper-fa-ws-pipelined-pingpong_test.py)

[Warp-Specialized computation-pipelined pingpong HSTU fwd on Hopper](https://github.com/meta-recsys/generative-recommenders/blob/bcb3aeea0f7b48faa9ea8d0d0337a055897618ec/generative_recommenders/ops/triton/triton_hstu_attention.py#L1262)




### Pipelined GEMM on NVIDIA Hopper

```
@triton.jit
def matmul_kernel_pipelined_hopper(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                                   stride_bk, stride_bn,  #
                                   stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                   BLOCK_SIZE_K: tl.constexpr,  #
                                   GROUP_SIZE_M: tl.constexpr,  #
                                   NUM_STAGES: tl.constexpr  #
                                   ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offset computation
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # allocate NUM_STAGES buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    # prefetch (pipelining) for NUM_STAGES - 1 buffers
    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        token_a = tlx.async_load(a_ptrs, buffers_A[i], mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, buffers_B[i], mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        tlx.async_load_commit_group([token_a, token_b])

    # main K loop
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Disable auto-pipelining with num_stages=0
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=0):
        # identify the buffer index for the current iteration
        buf = k % NUM_STAGES

        # wait for buffers to be ready
        tlx.async_load_wait_group(NUM_STAGES - 2)

        # do the mma
        acc = tlx.async_dot(buffers_A[buf], buffers_B[buf], acc)

        # prefetch for i-th iteration, i.e, NUM_STAGES - 1 ahead
        i = k + NUM_STAGES - 1
        # wait for the previous MMA using this buffer to complete
        acc = tlx.async_dot_wait(NUM_STAGES - 1, acc)
        # prefetch
        token_a = tlx.async_load(a_ptrs, buffers_A[i % NUM_STAGES], mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, buffers_B[i % NUM_STAGES], mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        tlx.async_load_commit_group([token_a, token_b])
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # wait for last mma to complete
    acc = tlx.async_dot_wait(0, acc)
    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```


### Warp-Specialized GEMM on NVIDIA Blackwell

```
@triton.jit
def matmul_kernel_tma_ws_blackwell(a_desc, b_desc, c_desc, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                                   BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                                   GROUP_SIZE_M: tl.constexpr,  #
                                   NUM_SMEM_BUFFERS: tl.constexpr,  #
                                   NUM_TMEM_BUFFERS: tl.constexpr,  #
                                   NUM_SMS: tl.constexpr,  #
                                   EPILOGUE_SUBTILE: tl.constexpr,  #
                                   ):
    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_SMEM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_SMEM_BUFFERS)
    # use multiple TMEM buffers to overlap MMA and epilogue
    tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # producer, TMA load
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            load_phase = 0  # the current phase of TMA load
            # we virtually "flatten" the two layer loop as if we're performing tma loads on
            # one big list of data
            processed_k_iters = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for previous phase(round) of dot for this buf
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)
                    # buffer is now ready to be used again
                    offs_k = k * BLOCK_SIZE_K
                    tlx.barrier_expect_bytes(smem_full_bars[buf],
                                             2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)  # float16
                    tlx.async_descriptor_load(a_desc, buffers_A[buf], [offs_am, offs_k], smem_full_bars[buf])
                    tlx.async_descriptor_load(b_desc, buffers_B[buf], [offs_k, offs_bn], smem_full_bars[buf])
                    # flip phase at the end of a round
                    load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)
                processed_k_iters += k_tiles
        with tlx.async_task(num_warps=1, num_regs=232):  # MMA consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            dot_phase = 0  # the current phase of dot op
            tmem_write_phase = 1  # sync between epilogue consumer and MMA consumer
            cur_tmem_buf = 0

            processed_k_iters = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                # wait epilogue consumer to be done with the buffer before reusing it
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_write_phase = tmem_write_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                # now iterate along K to compute result for the block
                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    # wait for current phase(round) of load for this buf
                    tlx.barrier_wait(smem_full_bars[buf], dot_phase)
                    # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                    tlx.async_dot(buffers_A[buf], buffers_B[buf], tmem_buffers[cur_tmem_buf], use_acc=k > 0,
                                  mBarriers=[smem_empty_bars[buf]], out_dtype=tl.float32)
                    # flip phase at the end of a round
                    dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

                # wait for last mma to complete
                last_buf = (processed_k_iters + k_tiles - 1) % NUM_SMEM_BUFFERS
                # in case phase was flipped, we should use the phase value when dot op was issued
                last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                # done filling this buffer, signal epilogue consumer
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                # possibly enter next iteration (next tile) without waiting for epilogue
                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                processed_k_iters += k_tiles

        with tlx.async_task(num_warps=4, num_regs=232):  # epilogue consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            tmem_read_phase = 0
            cur_tmem_buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_read_phase = tmem_read_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                # load the result from TMEM to registers
                acc_tmem = tmem_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE:
                    # We load/store the result half by half to reduce SMEM pressure
                    acc_tmem_subslice1 = tlx.subslice(acc_tmem, 0, BLOCK_SIZE_N // 2)
                    result = tlx.local_load(acc_tmem_subslice1)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn], c)

                    acc_tmem_subslice2 = tlx.subslice(acc_tmem, BLOCK_SIZE_N // 2, BLOCK_SIZE_N // 2)
                    result = tlx.local_load(acc_tmem_subslice2)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c)
                else:
                    result = tlx.local_load(acc_tmem)
                    c = result.to(tl.float16)
                    c_desc.store([offs_am, offs_bn], c)

                # done storing this buffer, signal MMA consumer to resume writing to it
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
```

## Build and install TLX from source

```
git clone https://github.com/facebookexperimental/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Run the tutorials after the build finishes, e.g,
```
python third_party/tlx/tutorials/hopper-fa-ws-pipelined-pingpong_test.py
```

## More reading materials

[Barrier Support in TLX](third_party/tlx/doc/tlx_barriers.md  )

[TLX talk in 2025 Triton Developer Conference](third_party/tlx/doc/TLX-triton-conference.pdf)
