# Skinny GEMM three-layout fusion

This OSS benchmark reproduces `T289757878` without Buck or fbsource imports:

```text
C[M,64] = A[M,112] @ B[64,112].T
layout0 = C
layout1 = C
layout2 = C.reshape(5120,256,64).permute(0,2,1).contiguous()
M = 1310720
```

The numerical contract is BF16 inputs, FP32 accumulation, and BF16 outputs.
Reduction reassociation is allowed. Correctness uses `atol=rtol=0.02` and also
reports max absolute error, relative-L2, and exact equality for every output.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/fwd_skinny_gemm_multi_layout_tlx.py \
  --warmup 10 --samples 50 --reps 5 \
  --verification-seeds 0 1 2
```

## Winning schedule

The winner retains a row-major accumulator because two of the three outputs are
row-major. It computes two K=64 MMA steps (the second is masked at K=112), casts
the completed FP32 accumulator once, and writes all three destinations. The
static-persistent grid uses one CTA per GB200 SM, `BLOCK_ROWS=256`, eight warps,
and three compiler stages.

Locked GB200 best-of-five medians (`warmup=10`, `samples=50`) are:

| Comparison | Unfused/other | Tuned fused | Result |
|---|---:|---:|---:|
| OSS paired benchmark | `821.088 us` | `160.704 us` | `5.11x` |
| Original fbsource unfused baseline | `174.690 us` | `160.704 us` | `1.09x` |
| Original fbsource fused seed | `782.350 us` | `160.704 us` | `4.87x` |

The full paired ranges are `821.088–823.104 us` unfused and
`160.704–163.104 us` fused. The tuned kernel, original seed, and TLX variant
are bit exact on all three outputs for seeds 0/1/2, although exactness is not a
requirement.

The OSS unfused path is again distorted by its scalar-indexed layout kernel:
the standalone GEMM is about `98.528 us`, while the three-output copy is about
`731.472 us`. The original full-forward attribution was `75.04 us` GEMM plus
`99.54 us` materialization. The tuned fused result still clears the original
standalone baseline, but matching-fbsource and full-forward validation are
required before promotion.

## TLX comparison and profiling

The TLX implementation uses TMA for A, two fixed K=64 B tiles in SMEM, two
FP32 TMEM buffers, and a direct three-store epilogue. Its best configuration is
`BM128/BN64/BK64`, two M groups, one CTA per SM, eight producer warps at 104
registers, and four epilogue warps at 96 registers. It measures `164.096 us`
best-of-five, narrowly behind the simpler Triton implementation.

PTXAS reports 154 registers/thread and no spills for the winner. NCU reports
155.67 KiB dynamic shared memory, one resident block per SM, 12.5% theoretical
and 12.46% achieved occupancy, 52.92% memory throughput, 46.30% DRAM
throughput, and 19.48% compute throughput. Register caps of 128 or below
regress performance, and shared memory independently limits residency.

Output-major accumulation is slightly slower because it favors only the
grouped output while making both row-major stores unfavorable. High-level TMA
stores are correct but regress to about `192 us`; smaller/larger row tiles,
16-warp schedules, non-persistent grids, and five pipeline stages also lose or
exceed the shared-memory limit.
