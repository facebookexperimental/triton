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
| optimized OSS unfused | 2.325, 2.333, 2.386 | 2.325 |
| original fused Triton seed | 7.720, 7.713, 7.721 | 7.713 |
| tuned fused TLX | 2.585, 2.578, 2.577 | 2.577 |

The tuned kernel is 2.99x faster than the fused seed but remains 10.8% slower
than the optimized unfused composition. Across seeds 0/1/2, final relative-L2
is `5.18e-6 / 4.90e-6 / 5.36e-6`; dweight and dbias match after their BF16
output cast.

The remaining gap is not the GEMM. A bare pair-CTA TLX GEMM is about 1.04 ms
at this shape. The fused schedule pays for the LayerNorm row exchange and
column-gradient reductions, while the N-split GEMM also reloads A in both
CTAs. A likely next step is an A-multicast N-split schedule, provided both CTA
MMAs can consume the multicast operand with a correct per-CTA completion
protocol.
