"""Curated TLX guide content that supplements the repository README."""

GUIDE_CONTENT = {
    "overview": r"""
## Architecture: the MIMW programming model

TLX uses a **Multiple Instructions, Multiple Warps (MIMW)** model. A warp group is an explicit execution agent: one group can move data, another can issue tensor-core work, and another can run an epilogue. The groups communicate through shared or tensor memory and synchronize with hardware barriers.

This fills the gap between CUDA's thread-level SIMT model and Triton's program-level blocked model. Regular tensor expressions still look like Triton; orchestration becomes explicit only where hardware-aware control matters.

| Layer | Primary unit | Best fit |
|---|---|---|
| CUDA / PTX | Thread and warp | Maximum control, with manual scheduling and synchronization |
| Triton | Program instance / block | Productive blocked tensor programs and compiler-managed layouts |
| TLX | Cooperating warp groups | Explicit pipelines, async engines, tensor memory, and cluster execution |

### Orchestration surfaces

- **Warp groups:** `tlx.async_tasks()` and `tlx.async_task()` define producer, consumer, and epilogue roles.
- **Synchronization:** mbarriers express phase-based dependencies between warp groups and asynchronous engines.
- **Memory:** local buffers, TMA-style descriptor operations, and Blackwell tensor memory keep data close to compute.
- **Clusters:** CTA rank, cluster barriers, and remote shared-memory operations enable multi-CTA cooperation.
- **Scheduling:** Cluster Launch Control supports hardware-assisted persistent work distribution on Blackwell.

### Why this level of abstraction?

Modern GPUs increasingly depend on features that sit between individual threads and whole thread blocks: warp specialization, asynchronous data movement, tensor-memory accumulators, and clustered execution. TLX exposes those features without requiring a kernel to be written directly in CUDA or PTX, while keeping layouts compiler-inferred unless an expert needs to override them.
""",
    "getting-started": r"""
# Getting started

TLX is imported alongside Triton's standard language module. Start with ordinary Triton indexing and tensor operations, then introduce TLX only for the parts that need explicit warp-group, memory, or synchronization control.

```python
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
```

## Quick reference

| Item | Repository location |
|---|---|
| TLX implementation | [third_party/tlx/language/tlx](third_party/tlx/language/tlx/) |
| Exported API | [third_party/tlx/language/tlx/__init__.py](third_party/tlx/language/tlx/__init__.py) |
| Tutorials | [third_party/tlx/tutorials](third_party/tlx/tutorials/) |
| Design notes | [third_party/tlx/doc](third_party/tlx/doc/) |
| Correctness suite | [third_party/tlx/tutorials/testing/test_correctness.py](third_party/tlx/tutorials/testing/test_correctness.py) |

Build instructions and test commands live on the [Build, install, and test](install-and-test.html) page.

## Choose a first tutorial

| Target | Suggested starting point |
|---|---|
| Simple warp specialization | [vector-add2.py](third_party/tlx/tutorials/vector-add2.py) |
| NVIDIA Hopper | [hopper_fa_ws_pipelined.py](third_party/tlx/tutorials/hopper_fa_ws_pipelined.py) |
| NVIDIA Blackwell | [blackwell_fa_ws_pipelined.py](third_party/tlx/tutorials/blackwell_fa_ws_pipelined.py) |
| AMD CDNA | [amd_fa_pipelined.py](third_party/tlx/tutorials/amd_fa_pipelined.py) |

## Minimal warp-specialized example

The starter tutorial computes two independent additions in parallel. The default task uses the kernel's normal warps; the second task reserves four additional warps.

```python
@triton.jit
def add2_kernel(x, y, out_xy, a, b, out_ab, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    with tlx.async_tasks():
        with tlx.async_task("default"):
            lhs = tl.load(x + offsets, mask=mask)
            rhs = tl.load(y + offsets, mask=mask)
            tl.store(out_xy + offsets, lhs + rhs, mask=mask)

        with tlx.async_task(num_warps=4):
            lhs = tl.load(a + offsets, mask=mask)
            rhs = tl.load(b + offsets, mask=mask)
            tl.store(out_ab + offsets, lhs + rhs, mask=mask)
```

The tasks are independent here. Pipelined GEMM and attention kernels add shared buffers and mbarriers so producer and consumer groups can safely reuse storage across iterations.

## A practical learning path

- Run `vector-add2.py` to see task creation without synchronization.
- Study [local and remote buffers](buffers.html) and [async memory access](async-memory.html).
- Add phase-based handoffs using the [barrier and CLC](synchronization.html) APIs.
- Compare Hopper, Blackwell, and AMD tutorial implementations of the same operation.
- Use the [performance optimization](performance-optimization.html) checklist only after correctness is established.
""",
    "hardware-support": r"""
# Hardware support

TLX presents one low-level programming model across NVIDIA and AMD GPUs, but the available mechanisms reflect each architecture. Portable kernel structure is encouraged; hardware-specific fast paths remain explicit.

| Target | Core TLX mechanisms | Representative tutorials |
|---|---|---|
| NVIDIA Hopper (H100) | TMA descriptor loads/stores, WGMMA, mbarriers, warp specialization, CTA clusters | [Hopper pipelined FA](third_party/tlx/tutorials/hopper_fa_ws_pipelined.py) |
| NVIDIA Blackwell (B200 / GB200) | Hopper features plus tensor memory, fifth-generation async MMA, CLC, paired-CTA execution, DSMEM | [Blackwell CLC GEMM](third_party/tlx/tutorials/blackwell_gemm_clc.py) |
| AMD gfx942 (MI300) | Local buffers, buffer operations, MFMA-oriented kernels, explicit LDS management | [gfx942 GEMM](third_party/tlx/tutorials/amd_gemm_gfx942.py) |
| AMD gfx950 (MI350 / MI355) | Direct-to-LDS async loads, warp pipelining, padded LDS layouts, TDM and persistent kernels | [AMD warp-pipeline GEMM](third_party/tlx/tutorials/amd_gemm_warp_pipeline.py) |

## NVIDIA Blackwell

Blackwell is the broadest TLX target. Tensor Memory (TMEM) holds accumulator tiles outside the normal register file, while `tlx.async_dot` issues non-blocking tensor-core operations and can signal mbarriers on completion. Cluster Launch Control provides hardware-managed tile acquisition for persistent kernels, including paired-CTA execution.

Key mechanisms include:

- `tlx.local_alloc(..., tlx.storage_kind.tmem)` for tensor-memory allocations.
- `tlx.async_dot()` and `tlx.async_dot_scaled()` for asynchronous MMA.
- `tlx.clc_create_context()` for dynamic persistent scheduling.
- `tlx.async_remote_shmem_store()` for cross-CTA shared-memory exchange.
- `ctas_per_cga` for CUDA-native thread-block clustering.

## NVIDIA Hopper

Hopper kernels typically use a TMA producer warp group, a WGMMA consumer group, and an optional epilogue group. Multiple shared-memory stages overlap global-memory movement with tensor-core execution. The same mbarrier phase model governs both explicit warp-group handoffs and TMA completion.

The Hopper tutorials are the clearest place to learn producer/consumer pipelines before adding Blackwell TMEM or CLC.

## AMD CDNA

AMD support uses the same local-buffer and task concepts with architecture-native operations:

- `tlx.buffer_load()` and `tlx.buffer_store()` expose scalar-base-plus-offset buffer addressing.
- `tlx.buffer_load_to_local()` and async loads move data directly into LDS where supported.
- `tlx.warp_pipeline_stage()` marks explicit load and compute stages and maps priority hints to `s_setprio`.
- `tlx.padded_shared_layout_encoding` describes LDS padding used to avoid bank conflicts.

gfx950 provides the fullest AMD pipeline support. gfx942 kernels use a smaller subset and more explicit LDS staging. Support for newer CDNA generations continues to build on these interfaces.

## Portability guidance

- Keep indexing, masks, and tensor math in standard Triton when possible.
- Isolate TMEM, CLC, direct-to-LDS, and layout-specific code behind compile-time target choices.
- Share launch geometry and correctness references across targets, but tune tile sizes and stage counts independently.
- Treat hardware availability tags in the API reference as the source of truth for individual operations.
""",
    "performance-optimization": r"""
# Performance optimization

Correctness comes first. Once a kernel is stable, TLX performance work is mainly about overlapping independent engines, keeping operands on chip, and assigning enough work to each warp group to hide latency without exhausting registers or shared memory.

## Warp-specialized producer/consumer pipelines

The highest-leverage transformation is to separate memory movement from tensor-core execution. A producer group fills local buffers while a consumer group computes on a previous stage.

```python
with tlx.async_tasks():
    with tlx.async_task("default"):
        for k in range(K_TILES):
            slot = k % NUM_STAGES
            tlx.barrier_expect_bytes(load_bars[slot], TILE_BYTES)
            tlx.async_descriptor_load(a_desc, a_smem[slot], offsets, load_bars[slot])

    with tlx.async_task(num_warps=1):
        for k in range(K_TILES):
            slot = k % NUM_STAGES
            tlx.barrier_wait(load_bars[slot], phase)
            acc = tlx.async_dot(a_smem[slot], b_smem[slot], acc)
```

Balance the groups: an under-provisioned producer starves compute, while too many producer warps reduce occupancy and register budget for the consumer.

## Multi-stage buffering

Double or triple buffering lets one stage compute while another loads. The useful depth depends on load latency, tile size, shared-memory capacity, and the number of in-flight operations supported by the target.

```python
buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), dtype, NUM_STAGES)
slot = iteration % NUM_STAGES
tile = tlx.local_view(buffers, slot)
```

More stages are not automatically better: they consume shared memory and may reduce occupancy. Tune stage count together with tile size and warp allocation.

## Fusion and layout management

Fuse operations when an intermediate would otherwise round-trip through global memory. TLX normally infers local layouts from consumers, so a fused pipeline can combine attention, normalization, activation, or projection stages without manually reconciling every layout.

Use explicit layouts only when measurement shows that inference cannot express the required hardware mapping or avoid a bank conflict.

## Persistent scheduling

Persistent kernels amortize launch and scheduling overhead by keeping CTAs resident and acquiring multiple work tiles. Blackwell CLC provides hardware-assisted work acquisition; other targets can use software tile schedulers.

Persistence is most useful for short or irregular work where launch overhead and load imbalance are visible. It can hurt when a large regular grid already saturates the GPU.

## AMD warp pipelining

On gfx950, explicit warp-pipeline stages guide instruction arbitration between memory-heavy and MFMA-heavy work:

```python
for tile in tl.range(0, K_ITERS):
    with tlx.warp_pipeline_stage("mfma", priority=0):
        acc = tl.dot(a_tile, b_tile, acc)

    with tlx.warp_pipeline_stage("memory", priority=1):
        token = tlx.async_load(ptrs, next_buffer)
        tlx.async_load_commit_group([token])
```

The marker does not create a correct pipeline by itself. Buffer rotation, waits, and producer/consumer ownership must already be valid.

## Autotuning dimensions

- `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`: work and reuse per CTA.
- `NUM_STAGES`: latency hiding versus shared-memory footprint.
- `num_warps`: execution resources assigned to the default task.
- `num_warps` and `num_regs` on async tasks: per-role resource allocation.
- `ctas_per_cga`: cluster geometry for cooperative kernels.
- Target-specific layouts and instruction shapes.

## Optimization checklist

- Profile before changing the schedule; classify compute, memory, or latency limits.
- Confirm tensor-core instructions are selected for the intended shapes and dtypes.
- Check occupancy, register spills, shared-memory use, and barrier stalls together.
- Tune one structural choice at a time and retain a correctness reference for every configuration.
- Re-measure full workloads: a faster kernel can still lose end-to-end through launch count, synchronization, or fusion boundaries.

Reported end-to-end examples are summarized in [Production case studies](production-case-studies.html).
""",
    "debugging": r"""
# Debugging performance and numerics

Performance and numerical failures often have different root causes, but the fastest workflow is the same: create one reproducible configuration, inspect the earliest stage where behavior diverges, and change only one variable at a time.

## Performance debugging workflow

- **Profile first.** Decide whether the kernel is memory-bound, tensor-core-bound, occupancy-limited, or latency-bound.
- **Read compiler diagnostics.** Failed pipelining, unavailable tensor-core shapes, register spills, uncoalesced accesses, and failed vectorization often identify the limiting transformation.
- **Inspect every compilation stage.** Compare TTIR, TTGIR, LLVM IR, and PTX or AMDGPU output rather than reasoning only from Python source.
- **Diff known-good and regressed builds.** Hold shapes and launch metadata constant while comparing generated IR and selected autotune configurations.
- **Inspect runtime scheduling.** Warp traces reveal barrier stalls, load/compute imbalance, deadlocks, and instruction issue gaps that static IR cannot.

### Diagnostic tools

| Tool | Use |
|---|---|
| TritonParse | Capture and compare compilation and launch events; extract standalone reproductions and bisect regressions |
| Triton-MPP | Classify bottlenecks and collect occupancy, tensor-core, TMA, barrier, and per-operation latency data |
| CUTracer | Inspect SASS-level execution, reconstruct TMA/MMA activity, and investigate deadlocks or races |
| Compute Sanitizer | Check memory access, initialization, synchronization, and shared-memory races on a concrete kernel run |

### Common performance symptoms

| Symptom | Likely cause | First checks |
|---|---|---|
| Low tensor-core utilization | Unsupported dot shape/dtype or gaps between MMA issues | Inspect lowered MMA instructions and producer readiness |
| Register spills | Tiles or live ranges are too large | Reduce tile size, shorten live ranges, or rebalance registers across tasks |
| Memory bandwidth saturation | Too little reuse or compute overlap | Increase useful fusion or pipeline overlap; verify coalescing |
| Barrier stalls | Producer/consumer imbalance or incorrect stage cadence | Compare arrive/wait phases and time spent by each task |
| Low occupancy | Shared-memory, register, or cluster footprint is too large | Reduce stages, buffers, registers, or cluster size |
| Launch overhead dominates | Work is short or fragmented | Fuse stages or consider persistent scheduling |
| AMD LDS bank conflicts | Incompatible access pattern and layout | Try padded or swizzled LDS layouts and re-measure |

### Useful compiler controls

```bash
# Dump Triton/MLIR compilation stages
MLIR_ENABLE_DUMP=1 python reproduce.py

# Run through the interpreter when the kernel subset is supported
TRITON_INTERPRET=1 python reproduce.py

# Show autotuning decisions
TRITON_PRINT_AUTOTUNING=1 python reproduce.py

# Enable additional Triton diagnostics
TRITON_DEBUG=1 python reproduce.py
```

## Numerical debugging workflow

### Preserve computation order

Non-linear operations are sensitive to scale placement. Scaling after an activation is not equivalent to scaling its input.

```python
# Wrong: activation sees the unscaled range
scores = gelu(tl.dot(q, k_t)) * qk_scale

# Correct: scale before the non-linearity
scores = gelu(tl.dot(q, k_t) * qk_scale)
```

### Synchronize asynchronous consumers

An epilogue must not read an accumulator or shared buffer until the final producing operation has completed.

```python
with tlx.async_task(num_warps=1):
    tlx.barrier_wait(dot_done, phase)
    result = tlx.local_load(accumulator)
```

### Treat low-precision formats explicitly

- Accumulate FP8, MXFP8, BF16, and FP16 dot products in FP32 unless the algorithm deliberately chooses otherwise.
- Verify scale expansion and reshaping independently from the dot product.
- Track expected ranges before and after quantization, activation, and normalization.
- Use different random seeds when validating stochastic rounding.

### Expect reduction-order differences

Floating-point addition is not associative. Changing tile shape, stage count, or the number of cooperating warp groups can alter reduction order without indicating a correctness bug. Evaluate error against an algorithm-appropriate tolerance and check whether it grows with problem size.

### Isolate the first bad value

- Compare TLX against a known-correct PyTorch or Triton reference.
- Validate stages separately: loads, dot products, scaling, activation, reductions, then stores.
- Print or store selected intermediate tiles rather than only the final output.
- If only one autotune configuration fails, inspect boundary masks, buffer rotation, and barrier phases for that configuration.
- Repeat the same launch to distinguish deterministic arithmetic differences from races or uninitialized data.

```python
torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)

error = (output - reference).abs()
print("max abs:", error.max().item())
print("mean abs:", error.mean().item())
```

Tolerance is workload-specific. Typical starting points are around `1e-5` for FP32, `1e-3` to `1e-2` for FP16/BF16, and wider for FP8-family inputs, always with FP32 reference accumulation where practical.
""",
    "production-case-studies": r"""
# Production case studies

These reported deployments illustrate where TLX's explicit scheduling model has mattered beyond an isolated microbenchmark.

## GEM model training on Blackwell

Recommendation-system foundation models combine smaller matrix shapes, frequent normalization, and heterogeneous modules that make dense hardware utilization difficult.

Reported results on B200 include:

- Dense model FLOP utilization increasing from 17.5% to 37.5%, with a 1.6× improvement over the H100 generation.
- More than ten TLX kernels covering grouped or jagged attention, rotary embedding, group index selection, and normalization.
- More than 20% end-to-end QPS improvement while remaining numerically equivalent for the deployment target.

The practical lesson was not just peak throughput: the team could express hardware-aware kernels in roughly 2.6K lines where an equivalent deeply templated implementation could exceed 10K lines, preserving iteration speed during model development.

## In-kernel broadcast optimization on Hopper

RecSys inference can replicate user features across many candidates, wasting memory bandwidth and launch overhead. The IKBO implementation fused broadcast, compression, and attention work into persistent warp-specialized kernels.

Reported highlights include:

- A 4× speedup for the linear-compression component.
- 621 BF16 TFLOP/s for the Flash Attention component, 2.4× over its baseline.
- Up to 70% latency reduction for the optimized inference path.

Producer and consumer warp groups use ping-pong buffers so memory movement, tensor-core work, and epilogue processing remain overlapped inside one resident kernel.

## Distributed grouped GEMM for LLM training

Expert-parallel mixture-of-experts layers need grouped matrix multiplication while tokens and weights move across nodes. A TLX implementation integrates computation and communication to reduce exposed end-to-end latency.

The kernel has been reported running continuously across more than 56K GB200 GPUs for large-scale LLM pretraining. The central benefit is operational: a Python-level kernel remains inspectable and tunable while still exposing the cluster, memory, and warp-group controls required at that scale.

## What transfers to other workloads

- Optimize the full pipeline rather than one operator in isolation.
- Fuse when intermediate global-memory traffic or launch overhead dominates.
- Use persistent scheduling for irregular or short tiles, not as a default for every grid.
- Preserve a high-level numerical reference and production-shaped benchmark throughout development.
- Prefer the smallest amount of hardware-specific control that closes the measured performance gap.
""",
}
