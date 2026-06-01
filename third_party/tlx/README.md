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

## Scaled Dot (AMD)

`tlx.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, *, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32, tiles_per_warp=None)`
is a thin wrapper around `tl.dot_scaled`. Without `tiles_per_warp` it is exactly
equivalent to `tl.dot_scaled` — pass it only when you need the AMD-specific
WMMA scheduling hint described below.

### `tiles_per_warp` — what it controls

| Concept | Controlled by | What it means |
|---------|---------------|---------------|
| **Warp distribution** | `warpsPerCTA` (chosen automatically by `AccelerateAMDMatmul::planWarps`) | How the total result tile is split *across* warps along M/N. |
| **Per-warp tiling** | `tiles_per_warp` (this hint) | How many `instrShape`-sized WMMA tiles *each warp* covers contiguously before the layout repeats. |

So `tiles_per_warp=[2, 2]` does **not** mean "distribute 4 tiles across 4
warps." It means *each* warp emits a 2×2 block of WMMA instruction tiles,
holding the corresponding 2×2 accumulator registers. Concretely, for a
`tt.dot_scaled` lowered to gfx1250 WMMA (`instrShape = [16, 16, K]`),
4 warps, `warpsPerCTA = [2, 2]`:

| `tiles_per_warp` | Per-warp coverage (M × N) | Per-CTA coverage before repeat (M × N) |
|------------------|---------------------------|----------------------------------------|
| `[1, 1]` (default) | `16 × 16` | `32 × 32` |
| `[2, 2]`           | `32 × 32` | `64 × 64` |

For a `256 × 256` result, `[1, 1]` repeats the layout `8 × 8` times,
`[2, 2]` repeats it `4 × 4`. Larger `tiles_per_warp` gives each warp more
contiguous accumulator state (better register reuse for preshuffled
MXFP scales, fewer warp-level reductions), at the cost of more registers
per warp.

Together, `instrShape`, `warpsPerCTA`, and `tiles_per_warp` define the M/N
extent of one CTA-level WMMA layout period:
`period[d] = instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]`. If the result
tile is larger than this period, the period repeats. The K entry of
`instrShape` is the per-instruction reduction depth and is handled separately
from this M/N tiling.

`tiles_per_warp` is validated by `AccelerateAMDMatmul`: it must have one
entry per result-tile dim, each entry must be positive, and
`instrShape[d] * warpsPerCTA[d] * tiles_per_warp[d]` must fit in the result
tile shape.

### Example

```python
import triton.language.extra.tlx as tlx

acc = tlx.dot_scaled(
    a, a_scale, "e5m2",
    b, b_scale, "e5m2",
    acc,
    tiles_per_warp=[2, 2],   # pack 2x2 WMMA tiles per warp for preshuffled MXFP
)
```

### Mechanism (for IR-level users)

The wrapper attaches `amdg.wmma_tiles_per_warp = array<i32: m, n>` on the
resulting `tt.dot_scaled` op. `ScaledBlockedToScaledWMMAF8F6F4` reads the
attribute and substitutes `m, n` for the default `1, 1` when building the
WMMA encoding. Setting the attribute directly on a `tt.dot[_scaled]` op
in MLIR has the same effect; the wrapper just spares Python kernels from
hand-poking attributes.

Currently consumed only by the scaled-WMMA pattern (gfx1250). Regular
`tt.dot` WMMA and the MFMA patterns do not read it.
