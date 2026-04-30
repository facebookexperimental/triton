# TLX - Triton Low-level Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It provides intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, compute, and asynchronous control flow, designed for power users pushing Triton to the metal.

## Memory Fences

`tlx.fence(scope)` issues a memory fence. The `scope` argument is required:

| Scope | PTX | Description |
|-------|-----|-------------|
| `"gpu"` | `fence.acq_rel.gpu` | Device-scope fence. Orders prior global/shared memory writes to be visible to all GPU threads. |
| `"sys"` | `fence.acq_rel.sys` | System-scope fence. Like `"gpu"` but also visible to the host CPU. |
| `"async_shared"` | `fence.proxy.async.shared::cta` | Proxy fence for async shared memory. Required between `local_store` and a subsequent TMA store (`async_descriptor_store`) to the same shared memory. |

Example:
```python
tlx.local_store(smem_buf, data)
tlx.fence("async_shared")
tlx.async_descriptor_store(desc, smem_buf, offsets)
```

## Warp Pipeline (AMD)

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
