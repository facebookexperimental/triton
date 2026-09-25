# Fused weighted-LayerNorm backward GEMM

This OSS-only benchmark reproduces the two symmetric `T290084482` regions:

```text
dy = BF16(gradient[2097152, 1024] @ projection_weight.T[1024, 256])
xhat = (x - mean) * rstd
wdy = gamma * FP32(dy)
dx = rstd * (wdy - mean(wdy) - xhat * mean(xhat * wdy))
final = BF16(FP32(residual) + FP32(BF16(dx)))
dweight = BF16(sum(FP32(dy) * xhat, rows))
dbias = BF16(sum(FP32(dy), rows))
```

No Buck or fbsource imports are required. The standalone reference and original
fused Triton seed are in `bwd_weighted_layernorm_gemm.py`; the tuned TLX kernel
and benchmark driver are in `bwd_weighted_layernorm_gemm_tlx.py`.

## Winning TLX schedule

- `BM64 / BN128 / BK64`
- two-CTA cluster split across the 256 output columns
- one resident wave (152 CTAs on GB200) with a static grid-stride loop
- six SMEM stages and two TMEM buffers
- eight epilogue warps
- two FP32 row scalars exchanged through DSMEM
- dweight/dbias accumulated in registers and atomically finalized once per
  persistent CTA

Splitting CTAs across N is intentional. The tested pair-CTA GEMM schedule is
faster for bare GEMM because it shares B, but each CTA then owns all 256
columns and its LayerNorm epilogue becomes the bottleneck. The selected
schedule duplicates A traffic while giving the normalization twice as many
CTAs and half-width tiles.

Run on a free Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_weighted_layernorm_gemm_tlx.py \
  --verification-seeds 0 1 2
```

GB200 locked-clock medians (20 samples, three repetitions):

| Implementation | Medians (ms) | Best (ms) |
|---|---:|---:|
| optimized OSS unfused | 2.333, 2.332, 2.332, 2.333, 2.357 | 2.332 |
| original fused Triton seed | 7.716, 7.724, 7.724, 7.715, 7.720 | 7.715 |
| tuned fused TLX | 2.462, 2.475, 2.469, 2.474, 2.467 | 2.462 |

The tuned kernel is 3.13x faster than the fused seed and is 5.6% slower than
the optimized unfused composition. Across seeds 0/1/2, final relative-L2
is `5.18e-6 / 4.90e-6 / 5.36e-6`; dweight and dbias match after their BF16
output cast.

Static persistence wins because every row tile has identical work and the
stride preserves the two-CTA N split. CLC measures about `2.53-2.59 ms`; a
non-persistent 65,536-CTA launch is about `8.58 ms` because it also flushes
dweight/dbias atomics per tile.

NCU reports 168 registers/thread, about 154 KB dynamic SMEM, 18.5% achieved
occupancy, and zero local-memory load/store traffic. The kernel therefore has
no register spills, and reducing the register cap cannot create another
resident block because both registers and SMEM limit occupancy. Compared with
the CLC version under NCU, the static schedule reduces instrumented duration
from `3.93` to `3.70 ms`, raises tensor-pipe activity from `40.1%` to `42.6%`,
and lowers long-scoreboard stalls from `22.0%` to `20.7%`. Barrier stalls remain
about `23.7%`.

Rejected local experiments include 4/5/7/8 SMEM stages, BK32/BK128, BM128,
one or three TMEM buffers, epilogue subtiles, lower register caps, a separate
statistics task, cached gamma, TMA-staged x/residual inputs, and A multicast.
The A-multicast path was correct but its per-stage cross-CTA readiness barrier
offset the saved traffic (`2.524-2.532 ms`). The remaining gap is in the
LayerNorm DSMEM rendezvous and column-gradient reductions rather than the bare
GEMM, which is about `1.04 ms` for the pair-CTA TLX implementation.
