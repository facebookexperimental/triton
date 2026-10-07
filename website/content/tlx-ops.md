Production-ready kernels promoted from the TLX tutorials into the FBTriton op library.

## What tlx.ops is

- Promote performant and prod-impactful ops from TLX tutorials into the FBTriton op library.
- Catalog existing OSS TLX kernels and pick the best kernel per (arch, op).
- Enforce the highest quality bar for both code change and CI.
- Demote `tutorials/` to a playground scratch with no CI coverage.

## Interface

```python
from triton.tlx.ops import mm as tlx_mm

out = tlx_mm(a, b)
```

`tlx.ops` infers the architecture from the input tensors and dispatches to the
matching registered implementation.

## Structure

```
third_party/tlx/
    ops/                    -> (symlink) python/triton/tlx/ops
        kernels/
            mm/          sm90.py  sm100.py  gfx942.py  gfx950.py
            addmm/       gfx942.py
            bmm/         gfx942.py
            ...
```

## Current availability

Implemented today:

```
mm/sm100.py
flash_attn/sm100.py
flash_attn/gfx950.py
flash_attn_mxfp8/sm100.py
flash_attn_mxfp8/gfx950.py
hstu_attn/sm100.py
kda/sm100.py
```

Current kernel selections for the rest of the catalog:

```
kernels/
    mm/                   sm90.py  sm100.py  gfx942.py  gfx950.py
    addmm/                gfx942.py
    bmm/                  gfx942.py
    flash_attn/           sm90.py  sm100.py  gfx950.py  (fwd + bwd)
    flash_attn_mxfp8/     sm100.py  gfx950.py           (gfx950 forward only)
    hstu_attn/            sm100.py  gfx942.py
                          _util.py  _stubs.py  _reference.py
    kda/                  sm100.py
    gdpa/                 sm100.py  gfx950.py
    grouped_gemm/         sm100.py  gfx950.py
    grouped_gemm_mxfp8/   sm100.py  gfx950.py
    addmm_glu/            gfx950.py
    paged_decode/         gfx950.py
    ikbo_fa/              gfx950.py
    ikbo_lce/             gfx950.py
    multi_cta_layernorm/  sm100.py
    cross_attention/      sm100.py
    bmm_shared_a/         gfx950.py
```

## MXFP8 grouped GEMM

```python
out = triton.tlx.ops.grouped_gemm_mxfp8(
    x,
    x_scale,
    w,
    w_scale,
    split_sizes,
    out=None,
    num_sms=None,
    sf_layout="natural",
)
```

The exact public signature is
`(x, x_scale, w, w_scale, split_sizes, *, out=None, num_sms=None,
sf_layout='natural')`. This is a forward-only operation for SM100 (compute
capability 10.0) and gfx950 (MI350/MI355X): E4M3 inputs and E8M0 scale factors
produce a contiguous BF16 result. It has no backward implementation or fallback
on other architectures.

### Data and group layout

- `x` is contiguous E4M3 `[GM, K]` data, packed in expert order.
- `w` is contiguous E4M3 `[G, N, K]` or packed `[G * N, K]` data. For rank 2,
  `G` comes from `split_sizes.shape[0]`, and `w.shape[0]` must be divisible by
  `G`.
- `split_sizes` is contiguous int32 `[G]` on the same CUDA device. Values may be
  zero, must be nonnegative, every exclusive prefix must be divisible by 128,
  and the final sum must equal `GM`. Thus zero-sized experts are supported in
  any position, but a nonzero expert may start only at a 128-row boundary.
- `GM` and `K` must be positive and divisible by 128; `G` and `N` must be
  positive. TMA base pointers and row byte strides must be 16-byte aligned,
  which additionally requires `N` to be divisible by 8 for the BF16 output.
- Every input must be contiguous and on `x.device`.

The split values remain on device so dispatch does not introduce a host read or
synchronization. The kernel checks nonnegativity, aligned prefixes, and the final
sum. Violating one of those requirements triggers a device trap; callers that
need to observe the error immediately must synchronize the CUDA stream.

### Scale-factor layouts

Both scale tensors are contiguous E8M0. `sf_layout` accepts exactly these values:

- `"natural"` (default): `x_scale` has exact shape `[GM, K // 32]`.
  With rank-3 `w`, `w_scale` is `[G, N, K // 32]`; with rank-2 `w`, it is
  `[G * N, K // 32]`.
- `"cublas_blocked"`: both scales are rank-2 opaque byte tensors in the cuBLAS
  128-row by 4-scale-column atom layout. Let
  `SK = 4 * ceil((K // 32) / 4)`. `x_scale` must contain exactly
  `ceil(GM / 128) * 128 * SK` bytes, and `w_scale` exactly
  `G * ceil(N / 128) * 128 * SK` bytes. Tensor shape is otherwise opaque; byte
  order, not a semantic row/column interpretation, defines this layout.

### Output and scheduling

The result shape is `[GM, N]`. If `out` is supplied, it must be a contiguous,
non-overlapping BF16 tensor of that exact shape on `x.device`, and the operation
returns the same tensor object. `num_sms=None` launches one persistent CTA per SM
on the input device; an explicit `num_sms` must be an integer from 1 through the
device SM count.

The SM100 implementation uses fixed, shape-selected 1CTA and 2CTA configurations
with a dynamic atomic tile scheduler; it does not autotune. Large, sufficiently
saturated N/K shapes use a cooperative 2CTA cluster, while other shapes retain
the 1CTA path. Each launch owns a counter initialized to zero; that reset is part
of CUDA graph capture and is replayed before the kernel, so repeated graph
replays traverse all tiles.

The gfx950 implementation launches one persistent workgroup per CU (`num_sms`
counts CUs) with an XCD-aware static tile stride, so it needs no counter and is
graph-safe. When 256x256 tiles fill the device it uses the 2x2-quadrant
`warp_pipeline_stage` engine of the FP16 gfx950 grouped GEMM, with each
half-tile's scales copied direct-to-LDS alongside its data. Smaller problems
use a compiler-pipelined 128-row tile. It does not autotune.
