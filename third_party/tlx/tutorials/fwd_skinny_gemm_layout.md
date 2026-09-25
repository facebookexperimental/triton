# Skinny GEMM direct-layout fusion

This OSS benchmark reproduces `T289757859` without Buck or fbsource imports:

```text
C[M,16] = A[M,64] @ B[16,64].T
output = C.reshape(5120,256,16).permute(0,2,1).contiguous()
M = 1310720
```

The numerical contract is BF16 inputs, FP32 accumulation, and BF16 output.
Reduction reassociation is allowed. Correctness uses `atol=rtol=0.02` and also
reports max absolute error, relative-L2, and exact equality.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/fwd_skinny_gemm_layout_tlx.py \
  --warmup 10 --samples 50 --reps 5 \
  --verification-seeds 0 1 2
```

## Winning schedule

The winning fused kernel changes the GEMM orientation to match the physical
destination. It computes one complete output group as
`B[16,64] @ A_tile[256,64].T`, producing the physical `[16,256]` tile directly.
No accumulator transpose or layout-copy launch is needed.

The GB200 configuration uses `BLOCK_ROWS=256`, eight warps, two compiler
stages, `maxnreg=64`, and a persistent grid of four programs per SM. Locked
GB200 best-of-five medians (`warmup=10`, `samples=50`) are:

| Implementation | Time |
|---|---:|
| cuBLAS GEMM + Triton layout copy | `157.888 us` |
| fused output-major kernel | `65.632 us` |

All five fused repetition medians (`65.632`–`71.488 us`) beat all five unfused
medians (`157.888`–`179.552 us`). The speedup is `2.41x`. Outputs are bit exact
for seeds 0/1/2, although exactness is not required.

This OSS unfused baseline is slower than the original fbsource measurement
(`59.58 us`), so the production/full-forward win still requires confirmation
in the matching fbsource environment.

NCU reports 64 registers/thread, 51.20 KiB dynamic shared memory, four resident
blocks per SM, 50% theoretical occupancy, 41.5% achieved occupancy, 62.6%
memory throughput, 49.0% DRAM throughput, and 32.0% compute throughput.

## TLX comparison and rejected schedules

The TLX implementation uses TMA for A, a cached computed-B tile, TMEM FP32
accumulators, and a direct grouped-layout epilogue. Because Blackwell TLX MMA
requires `N >= 32`, it pads the logical width 16 to 32. Its best tested schedule
uses four `BM128` groups, one A slot, two TMEM slots, and eight producer warps;
it measures `66.672 us` best-of-five and is also bit exact. It is competitive
but cannot beat the output-major Triton formulation's exact-width MMA.

- The original persistent `BM128/BN256` seed measures about `125.5 us` in this
  OSS environment.
- Non-persistent narrow/output-major kernels reach roughly `61`–`65 us` in
  short sweeps but are less consistent than the selected persistent route.
- One to four programs per SM were tested. Four programs per SM plus
  `maxnreg=64` raises theoretical occupancy from 37.5% to 50%.
- One versus two TMEM slots, one versus two A slots, `BM64/BM128`, two to eight
  epilogue warps, and one to eight producer warps did not beat the final fused
  schedule.
