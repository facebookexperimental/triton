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
scales, and BF16 output. This uses the **regular**, not grouped, GEMM entrypoint
from Primus-Turbo FlyDSL. The AITER column invokes its regular group-32 Triton
kernel (package-default configuration; no tuned rows for these square shapes).
Numbers are median microseconds per operation; TFLOPS is `2 * size**3 / time`.

| M=N=K | TLX, cached scales | TLX, pack both + GEMM | AITER Triton, raw scales | Primus FlyDSL, raw scales |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 7.19 µs / 299 TFLOPS | 16.92 µs / 127 TFLOPS | 8.02 µs / 268 TFLOPS | 17.96 µs / 120 TFLOPS |
| 2K | 19.98 µs / 860 TFLOPS | 30.80 µs / 558 TFLOPS | 47.03 µs / 365 TFLOPS | 27.46 µs / 626 TFLOPS |
| 4K | 60.91 µs / 2257 TFLOPS | 75.49 µs / 1821 TFLOPS | 356.60 µs / 385 TFLOPS | 53.39 µs / 2574 TFLOPS |
| 8K | 452.97 µs / 2427 TFLOPS | 469.35 µs / 2343 TFLOPS | 2799.88 µs / 393 TFLOPS | 429.12 µs / 2562 TFLOPS |
| 16K | 3806.39 µs / 2311 TFLOPS | 3852.94 µs / 2283 TFLOPS | 25258.56 µs / 348 TFLOPS | N/A (non-reproducible output) |

Seven graph-replay samples, 12 iterations/sample, two operations/replay;
preallocated outputs. The TLX cached-scales column measures only GEMM, while
its second column includes packing both scales on every call. Primus's regular
call includes its fused preshuffle + GEMM; first-call autotuning is excluded.
The 1K/2K outputs were fully checked against an independent FP32 dequantized
reference; a fixed-seed 128x128 sample was checked at 4K-16K, including after
graph replay. All reported timings passed (`rtol=0.03`, `atol=0.5`). The 1K-8K
Primus full outputs also matched exactly across three identical cold-cache
runs. Measurements used physical GPU 7 with the default AUTO clock policy.

The 16K Primus autotuner found no reproducible configuration. Direct testing
confirmed that identical cold runs of its chosen regular-GEMM kernel changed
the full output; six alternate configurations also changed output. That timing
is excluded rather than publishing an invalid comparison. Sampled numerical
closeness alone would not have detected this.

Clocks were **not** fixed: AMD documents that performance-determinism mode is
unsupported on MI350, and this host rejected a manual maximum GFXCLK limit.
We restored AUTO after checking that control. The table reports interleaved
medians, not fixed-clock peak performance. See [AMD SMI performance-level
support](https://rocm.docs.amd.com/projects/amdsmi/en/docs-10.1.0/conceptual/perf-determinism.html).

Environment: MI350X, TLX 3.9, PyTorch 2.16.0a0+rocm10.2.0a20261012,
amd-aiter 0.1.24 (source `2ebbe6275b2a`), Primus-Turbo 0.0.0.dev1 (source
`ca1fdd673172`), FlyDSL 0.3.4.1. Development benchmark scripts are not part
of this tutorial; the comparison conditions and results are recorded above.
