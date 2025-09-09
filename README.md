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


- `tlx.thread_id(axis)`

    Returns the id of the current thread instance along the given `axis`.





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
<<<<<<< HEAD
=======

<details>
<summary>
Alternatively, follow these steps to build LLVM from source manually.
</summary>

1. Find the version of LLVM that Triton builds against.  Check
`cmake/llvm-hash.txt` to see the current version. For example, if it says:
       49af6502c6dcb4a7f7520178bd14df396f78240c.

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run:

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables:

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e .

</details>

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e .` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- The build system creates a `compile_commands.json` file under the Triton repo
  directory. This file is used by VSCode IntelliSense and clangd to provide
  code completion and other features for C++ code.

  If IntelliSense does not work, you can try the following steps:

    - Do a local build. Run command `pip install -e .`.
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find ./build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`.
    - In VSCode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe:

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Configuration knobs**

See [`python/triton/knobs.py`](python/triton/knobs.py) for the full list of configuration knobs. You can set those knobs directly in python or use environment variables to control them. Below are some of the environment variables you can specify (see `knobs.py` for the full list):

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`.
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.
  - An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.

- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).
  - When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.
- `TRITON_FRONT_END_DEBUGGING=1` disables exception wrapping when an error occurs in the compiler frontend, allowing the full stack trace to be seen.
- `TRITON_DISABLE_LINE_INFO=1` removes all line information from the module.
- `PTXAS_OPTIONS` passes additional command-line options to the PTX assembler `ptxas` (only on NVIDIA).

> [!NOTE]
> Some of these environment variables don't have a knob in `knobs.py`-- those are only relevant to the C++ layer(s), hence they don't exist in the python layer.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```


# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers).

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers, please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md).
>>>>>>> 9c3563d04 ([BACKEND][NVIDIA] Update ptxas_options knobs default value (#8121))
