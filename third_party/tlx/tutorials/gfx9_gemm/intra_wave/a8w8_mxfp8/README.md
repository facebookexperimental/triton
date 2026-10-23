# gfx950 regular MXFP8 GEMM (work in progress)

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

The tile bodies currently call private helpers from upstream
`tlx.ops.grouped_gemm_mxfp8` to reuse its gfx950 pipeline. The regular
entrypoints and launch grid are distinct, but separating the tile bodies and
settling the public scale-layout contract remain follow-up work. Do not compare
kernel-only numbers with timings that include scale packing.
