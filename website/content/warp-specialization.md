### Warp Specialization operations

- `tlx.async_tasks` and `tlx.async_task` **[Hopper+]**

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

#### async_tasks Parameters

| Parameter                | Description                                                                                                                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `exclusive`              | Assert this is the only one `tlx.async_tasks` in the kernel for more efficient PTX. Default to False.                                                                                            |
| `no_ending_cluster_sync` | This suppresses compiler generated cluster sync at end of Warp Spec. Should only be used if user guarantees all cross CTA SMEM/TMEM access are done by end of WS default task. Default to False. |
| `mbarrier_try_wait_suspend_ns` | On Blackwell, use the four-operand `mbarrier.try_wait.parity` form with this suspend hint for waits in the kernel. `None` is unspecified, `0` explicitly disables the hint, and positive values enable it. If multiple `async_tasks` regions specify a value, the minimum explicit value is used module-wide. Default to None. |

#### async_task Parameters

| Parameter | Description |
|-----------|-------------|
| `"default"` | First positional argument to mark this as the default/trunk task |
| `num_warps` | Number of warps to reserve for this task |
| `num_regs` | Number of registers per thread (optional, for register allocation tuning). It is supported by both default and non-default tasks and must be divisible by 8. |
| `replicate` | Number of replicas for this task (default: 1). Creates multiple copies of the task region |
| `warp_group_start_id` | Starting warp ID for this task (optional). Allows explicit control over warp assignment |

#### Default Task Register Budget

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
