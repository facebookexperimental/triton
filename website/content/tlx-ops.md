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
mm_mxfp8/sm100.py
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
    mm_mxfp8/             sm100.py
    addmm/                gfx942.py
    bmm/                  gfx942.py
    flash_attn/           sm90.py  sm100.py  gfx950.py  (fwd + bwd)
    flash_attn_mxfp8/     sm100.py  gfx950.py           (fwd + bwd)
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

### gfx950 MXFP8 attention backward

`flash_attn_mxfp8` accepts contiguous, same-device BF16 Q/K/V with identical
`[batch, heads, sequence, 128]` shapes and a positive sequence length divisible
by 256. Each head must contain fewer than `2**31` elements. Both causal and
noncausal backward return BF16 gradients, accumulate in FP32, and avoid atomics.
Training uses hardware exp2 and saves the matching base-2 logsumexp.

Q/K use E4M3 payloads with E8M0 scales per 32×32 block, as in current Blackwell.
The general backward requantizes operands for their reduction axes and uses
32×32 dS quantization. Separate dQ and dK/dV owners recompute QK and dP.

For `B=4, H=32, D=128`, the specialized path supports
`N=1024/2048/4096/8192, sm_scale=0.5` with either causal flag, plus
`N=8192, sm_scale=1.3` noncausal. It shares square32 Q/K/dO payloads between
reduction orientations, fuses backward preparation, and materializes FP8 dS
once for the dQ consumer. It requires whole, aligned contiguous storage.
Temporary workspace per call is `128*N*N + 128*(N//32)**2 + 34816*N` bytes,
up to **8 GiB + 280 MiB at N8192**, excluding gradients and saved inputs.
Causal calls allocate the same dense workspace. Other configurations use the
general implementation; allocation or kernel failures are not silently retried.
These are quantized training recipes, not BF16-equivalent gradients.

#### Explicit FP8-input backward boundaries

The gfx950 backend also provides internal entrypoints for the specialized
shape/scale domain above. These do not change the BF16-only public
`tlx.ops.flash_attn_mxfp8` interface or add differentiation through quantization.

- **Complete FP8-input backward:** `launch_backward_shared_square_mxfp8`
  accepts prequantized E4M3 Q/K/V and uint8 E8M0 scales, BF16 dO/O, and the
  matching FP32 base-2 LSE. It includes dO quantization, FP32
  `Delta = sum(O * dO)`, K-scale preparation, and both gradient kernels.
- **Preparation:** `prepare_backward_shared_square_mxfp8` produces the
  backward dO payload/scales, Delta, and K-scale layout. Delta uses the
  original BF16 dO, not a dequantized FP8 approximation.
- **Prequantized core:** `launch_backward_shared_square_mxfp8_core` consumes
  those prepared tensors and launches the dK/dV and dQ kernels. Its timing
  excludes preparation and must be labeled separately from complete backward.

Q/K and dO use square32 scales repeated across each group of 32 sequence rows.
Backward V instead uses **feature32** scales, one E8M0 byte per 32 features
of each token. `gfx950_quant.quantize_mxfp8(v)` produces this representation
from a higher-precision V. Forward's `gfx950.quantize_mxfp8_v(v)` uses
**sequence32** scales and is not interchangeable: changing quantization axes
is not a scale transpose. The FP8-input entrypoints do not silently convert V.

Prequantized inputs and prepared tensors must belong to the same forward and
backward invocation. Callers own scale values and that provenance; checks are
metadata-only. Benchmark reports must state whether Q/K/V quantization is
upstream work or included in the measured workflow. Moving dO preparation
outside the timer is a core-only measurement, not an end-to-end speedup.

## MXFP8 GEMM

```python
out = triton.tlx.ops.mm_mxfp8(a, a_scale, b, b_scale, out=None, sf_layout="natural", space="full")
```

Computes `a @ b.T` from contiguous E4M3 `a` `[M, K]` and `b` `[N, K]` with one
E8M0 scale per 32 K values, returning contiguous BF16 `[M, N]`. `M`, `N` and
`K` must be multiples of 128. This is a forward-only SM100 operation promoted
from `tutorials/blackwell_gemm_ws_mxfp8.py`.

- `sf_layout="natural"`: `a_scale` is `[M, K // 32]` and `b_scale` is
  `[N, K // 32]`; they are packed into the blocked layout on every call.
- `sf_layout="cublas_blocked"`: scales are already in the cuBLAS 128x4 atom
  layout, e.g. torchao's `MXTensor.to_mx(..., is_swizzled_scales=True).scale`,
  and must contain exactly `M * K // 32` and `N * K // 32` bytes.
- `space="full"` (default) autotunes over 1-/2-CTA, split-K, overlapped-
  accumulator and pipeline-depth configs; the first call per shape compiles and
  benchmarks the pruned space. When all of B fits in shared memory next to at
  least four A stages (small N and K), it also tries a B-resident family: B is
  loaded once per CTA, each tile computes a whole 128 x N row so A is read from
  DRAM once, and the epilogue uses async TMA stores. `space="heuristic"`
  launches one shape-picked config without autotuning.
- A supplied `out` must be contiguous, non-overlapping BF16 `[M, N]` on
  `a.device` and is returned by identity.

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
