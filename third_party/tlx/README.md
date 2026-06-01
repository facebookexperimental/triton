# TLX - Triton Low-level Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It provides intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, compute, and asynchronous control flow, designed for power users pushing Triton to the metal.

## Buffer Operations (AMD)

Buffer operations access global memory via a scalar base pointer and a tensor of i32 element offsets, rather than a tensor of pointers. This maps directly to AMD's hardware buffer instructions, which use a resource descriptor and byte offsets, enabling the hardware to do out-of-bounds checking and cache optimization.

### `tlx.buffer_load`

Load a tensor of values from global memory.

```python
result = tlx.buffer_load(ptr, offsets, mask=None, other=None, cache=None)
```

| Argument | Type | Description |
|----------|------|-------------|
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not loaded. |
| `other` | tensor or scalar, optional | Value used for masked-out elements (where `mask[i]` is `False`). |
| `cache` | str, optional | Cache modifier (e.g. `".ca"`, `".cg"`). |

**Returns**: A tensor with the same shape as `offsets` and element type matching the pointee type of `ptr`.

Lowers to `amdg.buffer_load`, which is eventually lowered to `rocdl.raw.ptr.buffer.load`.

### `tlx.buffer_store`

Store a tensor of values to global memory.

```python
tlx.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)
```

| Argument | Type | Description |
|----------|------|-------------|
| `stored_value` | tensor | Values to write. |
| `ptr` | scalar pointer | Base address in global memory. |
| `offsets` | i32 tensor | Per-element byte offsets from `ptr`. |
| `mask` | bool tensor, optional | When `mask[i]` is `False`, the element is not written. |
| `cache` | str, optional | Cache modifier. |

**Returns**: Nothing.

Lowers to `amdg.buffer_store`, which is eventually lowered to `rocdl.raw.ptr.buffer.store`.

### `tlx.buffer_load_to_local`

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

### Example

```python
import triton.language.extra.tlx as tlx

@triton.jit
def kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int32)
    mask = offsets < BLOCK_SIZE

    # Load from global memory using buffer semantics
    data = tlx.buffer_load(src_ptr, offsets, mask=mask, other=0.0)

    # Store to global memory using buffer semantics
    tlx.buffer_store(data, dst_ptr, offsets, mask=mask)
```

For the async global-to-shared variant, see the warp-pipeline GEMM example (`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`).

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

## AMD TDM Descriptor Loads

`tlx.async_amd_descriptor_load(desc, result, offsets, pred=None)` issues an AMD
TDM descriptor load from global memory to a TLX local buffer. It is available on
TDM-capable AMD targets (`gfx1250+`) and should be synchronized with
`tlx.async_amd_descriptor_wait`.

`tlx.async_amd_descriptor_load_group(descs, results, offsets, warp_masks,
preds=None)` groups multiple AMD TDM descriptor loads behind one static hardware
TDM instruction. Each list entry is one arm:

| Argument | Description |
|----------|-------------|
| `descs[i]` | Tensor descriptor for arm `i`. |
| `results[i]` | Local buffer or local view receiving arm `i`. |
| `offsets[i]` | Offset list for arm `i`; all arms must have the same rank. |
| `warp_masks[i]` | Bitmask selecting the waves that use arm `i`. |
| `preds[i]` | Optional predicate for arm `i`; defaults to true. |

The warp masks must be non-empty, disjoint, axis-aligned, and cover all waves in
the CTA exactly once. The grouped operation currently requires one CTA, the same
rank and element bitwidth for every arm, one shared cache modifier, and shared
layouts supported by AMD TDM lowering. This is useful for kernels where
different wave groups load different inputs, such as A/B GEMM tiles or A/B plus
scale tiles in MXFP GEMM, while keeping the assembly to one TDM instruction per
load group.

Example:
```python
a_tok = tlx.async_amd_descriptor_load_group(
    [a_desc, b_desc],
    [tlx.local_view(a_buf, slot), tlx.local_view(b_buf, slot)],
    [[off_m, k * BLOCK_K], [k * BLOCK_K, off_n]],
    [0b0011, 0b1100],
)
tlx.async_amd_descriptor_wait(0, [a_tok])
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
