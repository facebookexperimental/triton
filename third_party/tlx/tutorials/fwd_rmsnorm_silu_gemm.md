# GEMM with weighted RMSNorm and SiLU

This OSS benchmark reproduces `T289757910`: BF16
`[5120,8192] @ [8192,512]`, followed by weighted RMSNorm over the complete
512-element row and SiLU. The integrated benchmark preserves the raw BF16 GEMM
output, the normalized/activated BF16 output, and FP32 rstd.

The TLX candidate evaluates the GEMM once. It reuses each A tile across two
N=256 FP32 TMEM accumulators. Two epilogue tasks unload the accumulator halves,
save the raw BF16 GEMM output, exchange FP32 sum-of-squares partials through
shared memory, then apply the common rstd, weight, and exact `tl.sigmoid` SiLU.

## GB200 result

The best schedule found is `BM64/BN256/BK64`, four A buffers, two B buffers per
N group, two epilogue tasks, and eight default-task warps. On idle GPU 0 with
locked clocks, `warmup=10`, `samples=50`, and five alternating repetitions:

| Implementation | Best median |
|---|---:|
| OSS unfused cuBLAS + RMSNorm/SiLU | `102.368 us` |
| integrated Triton seed | `126.816 us` |
| `tlx.ops.mm` heuristic, GEMM only | `182.912 us` |
| fused TLX | `134.976 us` |

The TLX candidate is `32.608 us` slower than unfused (`0.7584x`) and
`8.160 us` slower than the integrated Triton seed. It is therefore not a
promotion candidate.

Across seeds 0, 1, and 2, the saved BF16 GEMM output is bit exact. Maximum
absolute error is `1.53e-5` for FP32 rstd and `2.44e-4` for the BF16 activated
output. Reduction reassociation is allowed; GEMM accumulation remains FP32 and
the exact sigmoid path is retained.

PTXAS reports 96 registers per thread, a 32-byte stack frame, 32 bytes of spill
stores, and 68 bytes of spill loads. The kernel is not meaningfully constrained
by register pressure. Its complete-row tile produces only 80 CTAs on a 152-SM
GB200. Splitting N across a cluster restores CTA occupancy but loses to duplicated
A traffic and cluster/reduction synchronization.

Rejected short-run candidates include:

| Candidate | Best median |
|---|---:|
| one serial epilogue task | `137.712 us` |
| BM128 | `142.912 us` |
| BK128 with A2/B1 buffering | `137.840 us` |
| two-CTA N split | `190.512 us` |
| four-CTA N split | `229.120 us` |
| 96-register second epilogue cap | `136.224 us` |

Run the locked benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:third_party/tlx/tutorials \
  third_party/tlx/denoise.sh \
  /home/mren/.conda/envs/metamain/bin/python \
  third_party/tlx/tutorials/fwd_rmsnorm_silu_gemm_tlx.py \
  --warmup 10 --samples 50 --reps 5 --verification-seeds 0 1 2
```

A fresh fbsource run of the official output/rstd-only benchmark measures
`78.736 -> 118.960 us` (`0.6619x`); its earlier recorded result was
`43.92 -> 114.21 us`. The OSS baseline is `23.632 us` slower than the current
fbsource baseline, while the fused implementations are much closer. The OSS
candidate deliberately exercises the stricter integrated ABI by also saving
and checking raw `buf88`, so its `134.976 us` is not a like-for-like replacement
for the official fused timing. Neither comparison currently yields a fused win.
