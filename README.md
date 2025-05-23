# TLX - Triton Low-level Language Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It offers intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, computation, and asynchronous control flow. TLX is designed for expert users pushing Triton closer to the metal.

Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:

- Hardware-specific intrinsics (e.g., wgmma, async_copy, barrier)
- Shared and local memory allocation
- Instruction-level scheduling and control
- Cross-warpgroup synchronization


While this approach places more responsibility on the user, it reduces the compiler's role as a performance bottleneck. Although it may introduce divergence across hardware platforms, it empowers users to perform deeper, architecture-specific optimizations without relying solely on compiler heuristics.


## Preview of the DSL Extension

### Local buffer allocation

- `buffers = tlx.local_alloc(shape, dtype, num_buffers)`

    Allocate number_buffers buffers in local memory per thread block, each of size size. The memory layout is inferred from its consumers..


- `buffers = tlx.tmem_alloc(size, NUM_STAGES)`

    Allocate number_buffers of buffers in the tensor memory per thread block, each with size size.

### Async memory access

- `buffer = tlx.async_descriptor_load(memdesc, [offsets], barrier)`

   Load a chunk of data from global memory into a local memory buffer. The global address, strides, and buffer size are defined by the memory descriptor. A barrier object is provided and signaled upon completion of the operation.



### Async tensor core operations

- `acc = tlx.async_dot(a[i], b[i], acc)`
- `acc = tlx.async_dot(a_reg, b[i], acc)`
- `acc[i] = tlx.async_dot(a[i], b[i], acc[i], barrier)`


### Barrier operations

- `barriers = tlx.alloc_barrier(num_barriers, expected_count)`

    Allocate a specified number of barrier objects in the local memory.

- `tlx.barrier_wait(bar, phase)`

- `tlx.barrier_arrive(bar)`

- `tlx.barrier_expect_bytes(bar, bytes)`

Signal a barrier of an expected number of bytes to be copied.


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





## Preview of Kernels Implemented with TLX

### Warp-Specialized GEMM on NVIDIA Hopper

```
@triton.jit
def matmul_kernel_tma_ws_cooperative_hopper(
   a_desc, b_desc, c_desc, M, N, K,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
   NUM_STAGES: tl.constexpr, NUM_WARPS: tl.constexpr, dtype: tl.constexpr
):
    # allocate NUM_STAGES buffers
    dtype_size = tlx.sizeof(dtype)
    a = tlx.local_alloc(BLOCK_M // 2 * BLOCK_K * dtype_size, NUM_STAGES * 2)
    b = tlx.local_alloc(BLOCK_N * BLOCK_K * dtype_size, NUM_STAGES)

    # allocate barriers
    barEmptyA = tlx.alloc_barrier(NUM_STAGES * 2, NUM_WARPS * 4)
    barFullA = tlx.alloc_barrier(NUM_STAGES * 2, NUM_WARPS * 4)
    barEmptyB = tlx.alloc_barrier(NUM_STAGES, 128)
    barFullB = tlx.alloc_barrier(NUM_STAGES, 128)

    with tlx.async_tasks():
        # producer group
        with tlx.async_task(num_warps = 4, registers=40)
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            pid_m = pid // num_pid_m
            pid_n = pid % num_pid_n
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            phase = 1
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # locate the buffer index that current iteration should access
                buf = k % NUM_STAGES
                offs_k = k * BLOCK_K

                # wait for the A buffer to be released by the consumer 1
                tlx.barrier_wait(barEmptyA[buf], phase)
                tlx.barrier_expect_bytes(barFullA[buf], BLOCK_M // 2 * BLOCK_K * dtype_size)
                a[buf] = tlx.async_descriptor_load(a_desc, [offs_am, offs_k], barFullA[buf])

                # wait for the B buffer to be released by both consumers
                tlx.barrier_wait(barEmptyB[buf], phase)
                tlx.barrier_expect_bytes(barFullB[buf], BLOCK_N * BLOCK_K * dtype_size)
                b[buf] = tlx.async_descriptor_load(b_desc, [offs_bn, offs_k], barFullB[buf])

                # wait for the A buffer to be released by the consumer 2
                buf2 = buf + NUM_STAGES
                tlx.barrier_wait(barEmptyA[buf2], phase)
                tlx.barrier_expect_bytes(barFullA[buf2], BLOCK_M // 2 * BLOCK_K * dtype_size)
                a[buf2] = tlx.async_descriptor_load(a_desc, [offs_am + BLOCK_M // 2, offs_k], barFullA[buf2])

                # buffers in a row share the same phase
                phase = (buf < NUM_STAGES - 1) ? phase : phase ^ 1

        # Two consumer groups
        with tlx.async_task(num_warps = 4, registers=232, replicate=2)
            phase = 0
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # locate the buffer index that current iteration should access
                buf = k % NUM_STAGES
                phase = (buf > 0) ? phase : phase ^ 1
                bufA = buf + NUM_STAGES * duplicate
                # wait for the buffer to be produced by the consumer
                tlx.barrier_wait(barFullA[bufA], phase)
                tlx.barrier_wait(barFullB[buf], phase)
                acc = tlx.async_dot(a[bufA], b[buf], acc)
                # wait for current mma to complete
                tlx.async_dot_wait(0)
                # release buffers
                tlx.barrier_arrive(barEmptyA[bufA])
                tlx.barrier_arrive(barEmptyB[buf])
            c = acc.to(dtype)
            tlx.async_descriptor_store(c_desc, [offs_am + duplicate * BLOCK_M // 2, offs_bn], c)
```


### Warp-Specialized GEMM on NVIDIA Blackwell

```
@triton.jit
def matmul_kernel_tma_ws_blackwell(
    a_desc, b_desc, c_desc, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, NUM_WARPS: tl.constexpr, dtype: tl.constexpr
):
    # allocate NUM_STAGES buffers
    dtype_size = tlx.sizeof(dtype)
    a = tlx.local_alloc(BLOCK_M * BLOCK_K * dtype_size, NUM_STAGES)
    b = tlx.local_alloc(BLOCK_N * BLOCK_K * dtype_size, NUM_STAGES)
    acc = tlx.tmem_alloc(BLOCK_M * BLOCK_N, 1)

    # allocate barriers
    barSmemFull = tlx.alloc_barrier(NUM_STAGES, NUM_WARPS * 4)
    barSmemEmpty = tlx.alloc_barrier(NUM_STAGES, 32)
    barTmemFull = tlx.alloc_barrier(1, 32)

    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            pid_m = pid // num_pid_m
            pid_n = pid % num_pid_n
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            phase = 0
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # locate the buffer index that current iteration should access
                buf = k % NUM_STAGES
                offs_k = k * BLOCK_K
                # wait for the buffer to be released by the consumer
                tlx.barrier_wait(barSmemEmpty[buf], phase ^ 1)
                tlx.barrier_expect_bytes(barSmemFull[buf], (BLOCK_M + BLOCK_N) * BLOCK_K) * dtype_size)
                a[buf] = tlx.async_descriptor_load(a_desc, [offs_am, offs_k], barSmemFull[buf])
                b[buf] = tlx.async_descriptor_load(b_desc, [offs_bn, offs_k], barSmemFull[buf])
                # buffers in a row share the same phase
                phase = (buf < NUM_STAGES - 1) ? phase : phase ^ 1

        # mma group
        with tlx.async_task(num_warps = 1)
            phase = 0
            buf = 0
            last_phase = 0
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                # locate the buffer index that current iteration should access
                buf = k % NUM_STAGES
                # wait for the buffer to be produced by the consumer
                tlx.barrier_wait(barSmemFull[buf], phase)
                # release buffers on completion by setting barSmemEmpty
                acc[0] = tlx.async_dot(a[buf], b[buf].T, acc[0], barSmemEmpty[buf])
                last_phase = phase
                phase = (buf < NUM_STAGES - 1) ? phase : phase ^ 1
            # wait for the last mma to complete
            tlx.barrier_wait(barSmemEmpty[buf], last_phase)
            tlx.barrier_arrive(barTmemFull[0])

        # epilog group
        with tlx.async_task(num_warps = 4)
            phase = 0
            tl.barrier_wait(barTmemFull[0], phase)
            c = tlx.tmem_load(acc[0])
            c = c.to(dtype)
            tlx.async_descriptor_store(c_desc, [offs_am, offs_bn], c)
```
