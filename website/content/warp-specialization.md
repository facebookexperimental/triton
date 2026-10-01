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

> **[amd]** — AMD targets only. Intra-wave scheduling is currently tuned and
> validated on gfx950.

```python
tlx.warp_pipeline_stage(
    label=None,
    *,
    scope="inter_wave",
    priority=None,
    pair=0,
    auto_interleave=False,
    cover_policy="balanced",
)
```

`tlx.warp_pipeline_stage` is a context manager for two distinct AMD pipeline
mechanisms. The default `scope="inter_wave"` preserves the original behavior:
the compiler partitions a loop into stages and phase-shifts two warp groups.
`scope="intra_wave"` instead defines a bounded instruction-scheduling window
inside each wave; it does not split or phase-shift warp groups.

| Parameter | Type | Description |
|-----------|------|-------------|
| `label` | `str`, optional | Stage name used in diagnostics. Defaults to `"cluster"` for inter-wave stages and `"stage"` for intra-wave stages. |
| `scope` | `"inter_wave"` or `"intra_wave"` | Selects warp-group phase shifting or instruction scheduling within each wave. Defaults to `"inter_wave"`. |
| `priority` | `int` (0-3), optional | Inter-wave-only hardware priority, lowered to `s_setprio`. Higher values are more urgent. |
| `pair` | non-negative `int` | Intra-wave-only source relationship identifier. Two adjacent explicit regions form a window only when their identifiers match. The identifier may be reused after a window and is not the machine scheduling-group ID. Defaults to `0`. |
| `auto_interleave` | `bool` | Intra-wave only. Lets the compiler partition and interleave independent memory and MFMA chunks in a paired window or one mixed region. Defaults to `False`. |
| `cover_policy` | `"balanced"` or `"proportional"` | Intra-wave automatic-interleaving policy. `"balanced"` distributes surplus MFMA cover evenly; `"proportional"` preserves the target issue-cost ratio. Non-default policies require `auto_interleave=True`. |

### Inter-wave pipeline

Inter-wave stages split the loop body at source boundaries and insert
conditional barriers so that one warp group executes one stage ahead of the
other. This overlaps memory latency with compute but makes the source
responsible for the complete pipeline protocol:

- Use multi-buffered shared memory, typically triple buffering, to avoid races between warp groups.
- Use explicit `tlx.async_load_wait_group()` calls before consuming data.
- Implement the prologue, steady state, and epilogue drain.

`priority` is valid only in this scope. A non-default `pair`,
`auto_interleave=True`, or a non-default `cover_policy` is rejected. See the
gfx1250 warp-pipeline GEMM example
(`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`) for
the full pattern.

Auto software pipelining is automatically disabled on loops that contain warp pipeline stages.

Example (simplified):
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

### Intra-wave scheduling

Intra-wave stages keep every wave on the same control path. The compiler
counts the machine instructions represented by the marked TTGIR operations and
emits AMD scheduling-group constraints so independent DS/VMEM instructions can
be issued between MFMAs. These constraints are instruction-scheduler
directives, not runtime synchronization or memory fences.

The explicit form uses two adjacent regions. They must use the same `pair`, be
dataflow-independent, and contain one memory-only region and one MFMA-only
region. Adjacency selects the candidate regions; `pair` is a fail-closed source
contract that detects accidental mismatches. It is not required to be globally
unique and can be reused by consecutive statically unrolled windows.

```python
with tlx.warp_pipeline_stage("compute", scope="intra_wave", pair=0):
    acc = tl.dot(a_tile, b_tile, acc)
with tlx.warp_pipeline_stage("future_load", scope="intra_wave", pair=0):
    future = tl.load(future_ptrs)
```

With `auto_interleave=True`, one region may contain both independent streams.
The compiler derives memory and MFMA chunks from SSA dependencies, moves the
chunks into a stable interleaving before register allocation, and distributes
MFMA cover according to `cover_policy`:

```python
with tlx.warp_pipeline_stage(
    "load_and_compute",
    scope="intra_wave",
    auto_interleave=True,
    cover_policy="proportional",
):
    future = tl.load(future_ptrs)
    acc = tl.dot(a_tile, b_tile, acc)
```

Automatic single-region interleaving accepts read-only memory operations.
Memory writes, asynchronous commits, volatile loads, control flow, unknown
side effects, or dataflow between the memory and MFMA streams are rejected.
The marked region is the complete scheduling window: the compiler does not
borrow operations from outside it.

Intra-wave scheduling does not make buffer reuse safe or data ready. The
kernel must still provide the correct buffering, asynchronous waits, and reuse
boundaries. `priority` is rejected in this scope.
