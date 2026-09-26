# GEMM with weighted RMSNorm and SiLU

This OSS benchmark reproduces `T289757910`: BF16
`[5120,8192] @ [8192,512]`, followed by weighted RMSNorm over the complete
512-element row and SiLU. The integrated benchmark preserves the raw BF16 GEMM
output, the normalized/activated BF16 output, and FP32 rstd.

The promoted TLX kernel evaluates the GEMM once. A two-CTA cluster splits the
512 output columns into two N=256 tiles while each CTA walks the complete K
dimension with an FP32 TMEM accumulator. The epilogue saves the raw BF16 GEMM
output, exchanges FP32 sum-of-squares partials through distributed shared
memory, computes the common rstd, and applies the weight and SiLU.

## Promoted GB200 result

GEO selected `BM128/BN256/BK64`, a two-CTA N split, six A buffers, four B
buffers, a four-deep TMA-store staging ring, separate one-warp A and B
producers, a one-warp MMA task, and an eight-warp epilogue. On idle GPU 0 with
locked 2062 MHz clocks, the authoritative profiler-device-time run measured:

| Implementation | Best median |
|---|---:|
| frozen fused TLX | `81.632 us` |
| promoted fused TLX | `48.452 us` |
| `torch.compile` reference | `35.784 us` |

The promoted kernel is `1.686x` faster than the frozen fused denominator, but
is still `10.3%` slower than the task's historical `43.92 us` unfused result
and `35.4%` slower than the same-run `torch.compile` reference.

The standalone OSS event-timed harness, which includes host dispatch gaps,
measured best-of-five medians of `107.408 us` unfused and `104.640 us` fused
under the same clock lock. The GEO numbers sum GPU kernel time and are the
tuning contract; the event numbers represent end-to-end API latency.

Both the authoritative GEO check and the promoted-file check pass. The GEO
contract uses unscaled A and reports a seed-42 maximum output error of one BF16
step (`1.5625e-2`) within its `atol=rtol=1e-2` rule. In the standalone scaled
seeds 0, 1, and 2, projection is bit exact and maximum absolute errors are
`1.53e-5` for FP32 rstd and `2.44e-4` for BF16 output. The GEMM accumulator
remains FP32. The only algorithmic approximation is the FP32 SiLU sigmoid
implemented with approximate `exp2`/division; no accumulator precision was
reduced.

NCU reports 168 registers/thread, 230.79 KiB dynamic shared memory, 19.46%
occupancy, and a 0.53-wave launch. The N split fixes the BM64 tensor-core
throughput loss while preserving the full sequential K reduction. The staged
TMA epilogue avoids the much slower direct store from the TMEM layout.

Rejected short-run candidates include:

| Candidate | Best median |
|---|---:|
| four-CTA N split / BN128 | `65.9 us` |
| BK128 | `55.3 us` |
| BK32 | `64.4 us` |
| four launch warps | `51.8 us` |
| split-K 2 | `79.4 us`, rejected: changes projection low bits |

Run the locked benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:third_party/tlx/tutorials \
  third_party/tlx/denoise.sh \
  /home/mren/.conda/envs/metamain/bin/python \
  third_party/tlx/tutorials/fwd_rmsnorm_silu_gemm_tlx.py \
  --warmup 10 --samples 50 --reps 5 --verification-seeds 0 1 2
```

The prior fbsource output/rstd-only reproduction measured `78.736 -> 118.960
us`; its original recorded result was `43.92 -> 114.21 us`. Those older runs
used different harnesses and should not be mixed with GEO's profiler timing.
The promoted kernel preserves the stricter integrated ABI, including raw
`buf88`/projection, rather than optimizing only the final output and rstd.
