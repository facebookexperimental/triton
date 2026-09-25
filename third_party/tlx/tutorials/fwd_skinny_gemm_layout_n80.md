# N=80 skinny GEMM direct-layout fusion

This OSS benchmark reproduces `T289757895` without Buck or fbsource imports:

```text
C[M,80] = A[M,64] @ B[80,64].T
output = C.reshape(5120,256,80).permute(0,2,1).contiguous()
M = 1310720
```

The numerical contract is BF16 inputs, FP32 accumulation, and BF16 output.
Reduction reassociation is allowed. Correctness uses `atol=rtol=0.02` and also
reports max absolute error, relative-L2, and exact equality.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/fwd_skinny_gemm_layout_n80_tlx.py \
  --warmup 10 --samples 50 --reps 5 \
  --verification-seeds 0 1 2
```

## Winning schedule

The winning kernel computes the transposed-equivalent contraction so that the
accumulator already has the physical destination orientation. Each persistent
CTA covers 512 rows and splits the 80 output columns into exact 64- and
16-column MMA/store phases. This avoids the 48 dead columns in a padded N=128
tile while reusing the loaded A tile and cached B values. The grid contains one
CTA per GB200 SM, with eight warps and three compiler stages.

Locked GB200 best-of-five medians (`warmup=10`, `samples=50`) are:

| Comparison | Unfused/other | Tuned fused | Result |
|---|---:|---:|---:|
| OSS paired benchmark | `994.336 us` | `107.392 us` | `9.26x` |
| Original fbsource unfused baseline | `135.550 us` | `107.392 us` | `1.26x` |
| Original fbsource fused seed | `332.240 us` | `107.392 us` | `3.09x` |

The full five-repetition OSS ranges are `994.336–995.712 us` unfused and
`107.392–109.504 us` fused. The tuned kernel, original seed, and TLX variant
are bit exact for seeds 0/1/2, although exactness is not required.

The OSS unfused result is not representative of the original environment. Its
standalone GEMM is about `91.488 us`, while its scalar-indexed layout copy is
about `916.944 us`; the original full-forward attribution was `66.46 us` GEMM
plus `72.22 us` layout. Therefore the relevant evidence is that the fused
candidate also clears the original 135.55-us standalone baseline. Matching
fbsource and full-forward validation are still required before promotion.

## TLX comparison and profiling

The TLX implementation uses TMA for A, one cached B tile, FP32 TMEM
accumulators, and a direct grouped-layout epilogue. Its best configuration uses
`BM64/BN128/BK64`, four M groups, one A buffer, one TMEM buffer, eight producer
warps at 104 registers, and eight epilogue warps at 96 registers. Because the
TLX MMA requires N at least 32 and this schedule uses BN128, it computes and
loads an N=128 accumulator for 80 useful columns. It measures `109.504 us`
best-of-five when paired with unfused. A direct head-to-head run gives
`100.864 us` for the split Triton kernel and `103.360 us` for TLX.

PTXAS reports 228 registers/thread and no spills for the winning Triton kernel.
NCU reports 206.86 KiB dynamic shared memory, one resident block per SM, 12.5%
theoretical and 12.46% achieved occupancy, 50.04% memory throughput, 31.32%
DRAM throughput, and 24.35% compute throughput. Register caps from 96 down to
64 cause severe spilling and regress latency to roughly `0.5–0.7 ms`; shared
memory also independently limits residency to one block.

Rejected schedules include padded N=128 (`~105 us` in repeated-kernel sweeps),
N=16/N=32 chunking (`~95–120 us`), smaller row tiles (`~91–100 us` at their
best), 16-warp schedules, and 1024-row tiles. A three-way 32+32+16 variant was
discarded after exposing incorrect output at the 512-row tile.
