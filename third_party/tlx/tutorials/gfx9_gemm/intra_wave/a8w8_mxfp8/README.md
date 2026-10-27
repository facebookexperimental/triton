# gfx950 regular MXFP8 GEMM

This tutorial computes `A @ B.T` for contiguous square E4M3 matrices of size
1024, 2048, 4096, 8192, or 16384 with E8M0 per-32 scales and BF16 output.
It is a **regular** GEMM: no expert/group index or grouped scheduler is used.

```python
from triton.language.extra.tlx.tutorials.gfx9_gemm.intra_wave.a8w8_mxfp8 import (
    pack_regular_mxfp8_scales,
    regular_mxfp8_square,
)

packed_a, packed_b = pack_regular_mxfp8_scales(a_scales, b_scales)
regular_mxfp8_square(a, b, packed_a, packed_b, out=out)
```

`a_scales` and `b_scales` have natural shape `[size, size // 32]` and E8M0
or `uint8` dtype. Packing rearranges these bytes into K-major blocked atoms;
it does not change the numerical scales. Use `pack_regular_mxfp8_scale` to pack
and cache the weight scales independently when weights are static. The
cached-scale measurements below pre-pack both scales before timing the GEMM.

The 1K path uses 64x64 output tiles, 4 waves, and 3 compiler pipeline stages;
2K uses 64x128, 4 waves, and 2 stages. The larger path uses 256x256 output
tiles, padded/double-buffered LDS for data and scales, and 8 waves. AGPR
allocation is disabled on that path to avoid scratch spills on MI350X.

The regular entrypoints and launch grid contain no group routing. The gfx950
tile bodies live in `tlx.ops.kernels.mxfp8_tiles_gfx950` and are shared with
grouped GEMM; the regular path does not import its private helpers. This is a
tutorial API with a prepacked-scale contract, not a public `tlx.ops` API.

## Regular GEMM comparison on MI350X

All paths compute the same `A @ B.T` with E4M3 operands, per-row/per-32 E8M0
scales, and BF16 output. Both AITER columns invoke **regular**, not grouped,
GEMMs from AITER source `457b1151809a`. Its Triton path uses raw scales and
row-major inputs; its gfx950 FlyDSL path uses raw scales and a 16x16-preshuffled
weight matrix. For these square shapes, FlyDSL selected untuned heuristic tiles.
Numbers are median microseconds per operation; TFLOPS is `2 * size**3 / time`.

| M=N=K | TLX, cached scales | TLX, pack both + GEMM | AITER Triton, raw scales | AITER FlyDSL, preshuffled B |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 6.13 µs / 350 TFLOPS | 15.47 µs / 139 TFLOPS | 7.44 µs / 289 TFLOPS | 5.94 µs / 362 TFLOPS |
| 2K | 19.79 µs / 868 TFLOPS | 29.76 µs / 577 TFLOPS | 46.59 µs / 369 TFLOPS | 21.94 µs / 783 TFLOPS |
| 4K | 55.80 µs / 2463 TFLOPS | 74.17 µs / 1853 TFLOPS | 358.40 µs / 384 TFLOPS | 128.66 µs / 1068 TFLOPS |
| 8K | 453.75 µs / 2423 TFLOPS | 467.63 µs / 2351 TFLOPS | 2804.52 µs / 392 TFLOPS | 923.95 µs / 1190 TFLOPS |
| 16K | 3815.74 µs / 2305 TFLOPS | 3844.23 µs / 2288 TFLOPS | 25220.40 µs / 349 TFLOPS | 8350.88 µs / 1053 TFLOPS |

Seven graph-replay samples, 12 iterations/sample, two operations/replay;
preallocated outputs. The TLX cached-scales column measures only GEMM, while
its second column includes packing both scales on every call. AITER FlyDSL
measures GEMM only: the B preshuffle is done once before timing, and its
E8M0 scales stay in row-major order. The current AITER FlyDSL runner was called
directly because its public dispatcher tried to build an unavailable HIP
helper in this container. There are no tuned square-shape rows, so the runner
uses the dispatcher's heuristic fallback.

The 1K/2K outputs were fully checked against an independent FP32 dequantized
reference; a fixed-seed 128x128 sample was checked at 4K-16K, including after
graph replay. All reported timings passed (`rtol=0.03`, `atol=0.5`).
Measurements used physical GPU 7 with the default AUTO clock policy.

Clocks were **not** fixed: AMD documents that performance-determinism mode is
unsupported on MI350, and this host rejected a manual maximum GFXCLK limit.
We restored AUTO after checking that control. The table reports interleaved
medians, not fixed-clock peak performance. See [AMD SMI performance-level
support](https://rocm.docs.amd.com/projects/amdsmi/en/docs-10.1.0/conceptual/perf-determinism.html).

Environment: MI350X, TLX commit `8dc562b7a2`, PyTorch
2.16.0a0+rocm10.2.0a20261012, AITER source `457b1151809a`, FlyDSL 0.3.4.1.
Development benchmark scripts are not part of this tutorial; the comparison
conditions and results are recorded above.
