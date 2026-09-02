Compiler and core-language capabilities this fork adds on top of upstream
Triton. These need no TLX code — they apply to ordinary Triton kernels.

## AutoWS — automatic warp specialization

AutoWS is a compiler optimization that partitions a kernel's operations into
specialized warp groups — typically a **producer** group that handles memory
loads and a **consumer** group that handles computation (MMA/tensor core ops).
By assigning different hardware resources to each group, warp specialization
enables overlap of memory transfers, CUDA core work, and tensor core work,
improving SM utilization.

Where [TLX](tlx.html) has you write warp specialization by hand with
`tlx.async_tasks`, AutoWS asks the compiler to do it. The two are separate
paths: the TLX unit tests and the AutoWS suites are deliberately kept apart.

### Enabling it

Mark a loop with the `warp_specialize` attribute:

```python
for k in tl.range(0, K, BLOCK_K, warp_specialize=True):
    ...
```

It is also accepted on `tl.condition` for while loops:

```python
while tl.condition(c, warp_specialize=True, num_stages=3):
    ...
```

Per `tl.range`'s own documentation, this enables automatic warp specialization
on the loop: the compiler will attempt to partition memory, MMA, and vector
operations in the loop into separate async partitions. **This increases the
total number of warps the kernel requires.**

The marker is a *request*, not a decision — if the compiler's partition
scheduling assigns no partitions, the kernel compiles without warp
specialization rather than failing.

To select the Meta warp-specialization backend:

```bash
export TRITON_USE_META_WS=1
```

or, scoped from Python:

```python
triton.knobs.nvidia.use_meta_ws = True
```

### Hardware

Requires sm_90 or newer — Hopper (`sm90`) and Blackwell (`sm100`). On Blackwell,
task assignments come from the `PartitionSchedulingMeta` pass rather than the
Hopper `doTaskPartition` path.

### Tuning knobs

| Knob | Env var | Purpose |
|------|---------|---------|
| `knobs.nvidia.use_meta_ws` | `TRITON_USE_META_WS` | Select the Meta WS backend |
| `knobs.nvidia.ws_tile_prefetch_depth` | `TRITON_WS_TILE_PREFETCH_DEPTH` | Buffers for the dynamic-persistent tile-id broadcast channel (1 = single-stage) |
| `knobs.nvidia.ws_mem_plan_topk` | `TRITON_WS_MEM_PLAN_TOPK` | Enumerate the top-K TMEM column packings |
| `knobs.nvidia.ws_mem_plan_pick` | `TRITON_WS_MEM_PLAN_PICK` | Which ranked packing to apply (0 = cost/occupancy-best) |
| `knobs.nvidia.ws_mem_plan_topk_dump` | `TRITON_WS_MEM_PLAN_TOPK_DUMP` | Write ranked packings as JSON |

`minRegAutoWS` / `maxRegAutoWS` (compilation options, not env vars) control the
per-thread register budgets AutoWS assigns to non-tensor and tensor partitions.
If either is provided from the Python frontend, its value must be divisible by 8
so the emitted register allocation matches backend warp-group granularity.

Prefer the knobs over the raw `TRITON_WS_MEM_PLAN_*` environment variables so
tests can scope them with `knobs.nvidia.scope()` without leaking global env
state across a batch run.

### Tests

```bash
pytest python/test/unit/language/test_autows_addmm.py
pytest python/test/unit/language/test_autows_auto_tma.py
pytest python/test/unit/language/test_autows_quantized_matmul.py
pytest python/test/unit/language/test_warp_specialization.py
pytest python/test/unit/language/test_tutorial09_warp_specialization.py
```

Compiler-side coverage is in the WarpSpecialization lit suite (no GPU required);
see [CI](ci.html).

### Design documentation

The pass pipeline and its sub-passes are documented in detail under
[`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/`](third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/).
Start with `Overview.md`, which lays out the pipeline:

```
doTaskPartition          (Hopper only; skipped on Blackwell)
  → doTaskIdPropagate
  → doDynamicTileBroadcast
  → doDataPartition
  → doPingPongPrep       (optional)
  → doConvertDescriptorLoadsToNVWS
  → doBufferAllocation
  → doMemoryPlanner
  → doCodePartition
  → doPingPongSync       (optional)
  → doTokenLowering
  → doLoopSchedulePreprocessing + scheduleLoops
```

Frequently useful entry points from that directory:

| Topic | Document |
|-------|----------|
| Pipeline overview | `Overview.md` |
| Partition assignment | `PartitionSchedulingMeta.md` |
| Buffer allocation | `BufferAllocation.md`, `SmemAllocationDesign.md` |
| Code partitioning | `CodePartition.md`, `CodeSpecialization.md` |
| Barriers | `BarrierInsertion.md`, `BarrierConstraints.md`, `BarrierFusion.md` |
| TMEM | `TMEMAllocationHeuristics.md`, `TMEMInterleave.md` |
| Debugging wrong results or hangs | `DebuggingAccuracyAndDeadlocks.md` |

Additional design notes live in [`docs/design/`](docs/design/), including
`2cta-autoWS-sync.md` and `llm-autows.md`.

## Reduction ordering

Triton's default reduction (`tl.sum`, `tl.reduce`) uses layout-dependent
accumulation order, so changing `num_warps` or `BLOCK_SIZE` can change the
floating-point result. For workloads that require bitwise reproducibility —
deterministic training, numerical debugging, regression testing — the
`reduction_ordering` parameter requests a deterministic accumulation order that
is independent of thread layout.

```python
# Sum with deterministic ordering
z = tl.sum(x, axis=1, reduction_ordering=tl.ReductionOrdering.INNER_TREE)

# Custom combine function with deterministic ordering
z = tl.reduce(x, axis=1, combine_fn=my_fn,
              reduction_ordering=tl.ReductionOrdering.INNER_TREE)

# Default (no ordering guarantee, best performance)
z = tl.sum(x, axis=1)  # equivalent to ReductionOrdering.UNORDERED
```

Given the same logical input data and reduction ordering, the result is bitwise
identical regardless of `num_warps`, memory layout, or other compilation
parameters.

`ReductionOrdering` objects cannot be used directly inside JIT-compiled code
(they are Python objects without a Triton type), so pass them as `tl.constexpr`
kernel parameters:

```python
@triton.jit
def kernel(X, Z, ORDERING: tl.constexpr):
    x = tl.load(X + tl.arange(0, 1024))
    z = tl.sum(x, axis=0, reduction_ordering=ORDERING)
    tl.store(Z, z)

kernel[(1,)](x, z, ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=4)
```

Full design, ordering semantics, and the frontend-to-C++ plumbing are in
[Reduction Ordering in Triton](third_party/tlx/doc/reduction_ordering.md).
