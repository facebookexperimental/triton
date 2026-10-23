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
kernel-only benchmark pre-packs both scales before timing:

```bash
python3 third_party/tlx/tutorials/gfx9_gemm/intra_wave/a8w8_mxfp8/bench_upstream.py \
    --size 1024 --size 2048 --size 4096 --size 8192 --size 16384
```

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
| 1K | 7.08 µs / 303 TFLOPS | 16.69 µs / 129 TFLOPS | 7.79 µs / 276 TFLOPS | 17.88 µs / 120 TFLOPS |
| 2K | 19.81 µs / 867 TFLOPS | 30.54 µs / 563 TFLOPS | 46.58 µs / 369 TFLOPS | 26.87 µs / 639 TFLOPS |
| 4K | 61.98 µs / 2217 TFLOPS | 77.27 µs / 1779 TFLOPS | 357.27 µs / 385 TFLOPS | 56.15 µs / 2448 TFLOPS |
| 8K | 465.33 µs / 2363 TFLOPS | 476.91 µs / 2306 TFLOPS | 2802.61 µs / 392 TFLOPS | 439.93 µs / 2499 TFLOPS |
| 16K | 3889.17 µs / 2262 TFLOPS | 3928.49 µs / 2239 TFLOPS | 25169.60 µs / 349 TFLOPS | 3606.50 µs / 2439 TFLOPS* |

Seven graph-replay samples, 12 iterations/sample, two operations/replay;
preallocated outputs. The TLX cached-scales column measures only GEMM, while
its second column includes packing both scales on every call. Primus's regular
call includes its fused preshuffle + GEMM; first-call autotuning is excluded.
The 1K/2K outputs were fully checked against an independent FP32 dequantized
reference; a fixed-seed 128x128 sample was checked at 4K-16K, including after
graph replay. All passed (`rtol=0.03`, `atol=0.5`). Clocks were **not** fixed.

The 16K Primus autotuner warned that it found no reproducible configuration and
ranked candidates by time alone. Its 16K number passed the sampled correctness
check in this run, but should be treated as provisional.

Environment: MI350X, TLX 3.9, PyTorch 2.16.0a0+rocm10.2.0a20261012,
amd-aiter 0.1.24 (source `2ebbe6275b2a`), Primus-Turbo 0.0.0.dev1 (source
`ca1fdd673172`), FlyDSL 0.3.4.1. To rerun with all three installed:

```bash
python3 third_party/tlx/tutorials/gfx9_gemm/intra_wave/a8w8_mxfp8/bench_compare_regular.py \
    --size 1024 --size 2048 --size 4096 --size 8192 --size 16384 \
    --samples 7 --iterations 12 --unroll 2
```
