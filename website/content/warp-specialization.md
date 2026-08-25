- `tlx.async_tasks` and `tlx.async_task` **[sm90+]**

```
    with tlx.async_tasks()
        with tlx.async_task("default")
            ...
        with tlx.async_task(num_warps=4)
            ...
```
`tlx.async_tasks` opens a multi-tasking region where independent asynchronous tasks can be declared. Each task executes in parallel using a dedicated subset of warps within the thread block.

`tlx.async_task("default")` defines the default task, also known as the trunk. It uses the available warps not explicitly reserved by other tasks.

`tlx.async_task(num_warps=4)` defines a warp-specialized asynchronous task that explicitly reserves 4 warps in addition to those used by the trunk task.

### async_tasks Parameters

| Parameter                | Description                                                                                                                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `exclusive`              | Assert this is the only one `tlx.async_tasks` in the kernel for more efficient PTX. Default to False.                                                                                            |
| `no_ending_cluster_sync` | This suppresses compiler generated cluster sync at end of Warp Spec. Should only be used if user guarantees all cross CTA SMEM/TMEM access are done by end of WS default task. Default to False. |
| `mbarrier_try_wait_suspend_ns` | On Blackwell, use the four-operand `mbarrier.try_wait.parity` form with this suspend hint for waits in the kernel. `None` is unspecified, `0` explicitly disables the hint, and positive values enable it. If multiple `async_tasks` regions specify a value, the minimum explicit value is used module-wide. Default to None. |

### async_task Parameters

| Parameter | Description |
|-----------|-------------|
| `"default"` | First positional argument to mark this as the default/trunk task |
| `num_warps` | Number of warps to reserve for this task |
| `num_regs` | Number of registers per thread (optional, for register allocation tuning). It is supported by both default and non-default tasks and must be divisible by 8. |
| `replicate` | Number of replicas for this task (default: 1). Creates multiple copies of the task region |
| `warp_group_start_id` | Starting warp ID for this task (optional). Allows explicit control over warp assignment |

### Default Task Register Budget

Register budgets are specified per thread. When the default task does not set
`num_regs`, register allocation keeps the original donation model: non-default
warp groups receive their requested budgets, and the default task receives the
remaining registers.

Setting `num_regs` on the default task selects a fixed budget instead:

```python
with tlx.async_tasks():
    with tlx.async_task("default", num_regs=80):
        ...
    with tlx.async_task(num_warps=4, num_regs=24):
        ...
```

In this example, the default and non-default tasks receive 80 and 24 registers
per thread, respectively. The default task does not absorb unused registers.
Non-default warp groups without an explicit budget evenly share the remaining
register pool. If every task has a fixed budget, any surplus is left unused.
The compiler may raise a request to the hardware or instrumentation safety
minimum when required.

### Explicit Warp Assignment with warp_group_start_id

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

## Warp pipeline

> **[gfx950]** — AMD MI350 (CDNA4); not available on MI300.

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
