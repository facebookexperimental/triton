# D=8472 weighted RMSNorm backward

This OSS benchmark reproduces the `T289881412` reduction-only fusion:

```text
x       [5120,8472], physical row stride 8512, BF16
dy      [5120,8472], dense, BF16
weight  [8472], BF16
rstd    [5120], FP32
  -> dx [5120,8472], BF16
  -> dweight [8472], BF16
```

This is neither a GEMM prologue nor a GEMM epilogue. It is a multi-pass
RMSNorm-backward rewrite that computes a row reduction for `dx` and a column
reduction for `dweight`.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_rmsnorm_d8472.py \
  --warmup 10 --samples 50 --reps 5 \
  --verification-seeds 0 1 2
```

## Winning schedule

The source kernel assigns eight rows to a CTA and materializes the entire
8472-column row as a padded 16384-column program tensor. The earlier candidate
splits columns across CTAs, writes nine row-dot partials, and launches another
kernel to finish `dx`.

The winner instead assigns four rows to each of 1280 CTAs. Each CTA traverses
the row in five 2048-column chunks, retains four FP32 row-dot accumulators in
registers, and then traverses the row again to emit BF16 `dx` and one FP32
dweight partial. A second kernel reduces the 1280 dweight partials. Factoring
the row-invariant `rstd` outside the first reduction removes one FP32 multiply
per element. Reduction reassociation is allowed; accumulator precision remains
FP32 throughout.

Locked GB200 best-of-five medians (`warmup=10`, `samples=50`) are:

| Variant | Latency | Relative result |
|---|---:|---:|
| OSS source + finalizer | `2185.344 us` | `1.00x` |
| prior fbsource column-split candidate | `195.500 us` | winner is `1.84x` faster |
| tuned row-owned two-pass kernel | `106.464 us` | `20.53x` OSS speedup |

Across seeds 0/1/2, `dx` has maximum absolute error `0.0078125` and maximum
relative-L2 `2.21e-6`; dweight has maximum absolute error `0.5` and maximum
relative-L2 `7.32e-5`. These differences come from permitted FP32 reduction
reassociation. There is no approximation and no accumulator-precision change.

PTXAS reports 79 registers/thread with no spills for the main kernel and 32
registers/thread with no spills for the finalizer. NCU reports 8.19 KiB dynamic
shared memory, 37.5% theoretical and 34.52% achieved occupancy, 38.14% DRAM
throughput, and 47.84% SM throughput for the main kernel. A 64-register cap
spills and regresses. The finalizer reaches 81.91% achieved occupancy.

The column-split three-kernel version remains as a diagnostic implementation;
its best tested configuration is about `120 us`. A row-owned `BLOCK_N=4`,
`BLOCK_D=2048`, eight-warp schedule wins by avoiding the global row-partial
buffer and one launch. TLX task decomposition is not useful here because there
is no asynchronous GEMM pipeline to overlap and each CTA's second pass depends
on its completed row reduction.

The historical integrated candidate improved full backward from `41146.38 us`
to `38870.32 us`. The new static winner still needs matching-fbsource and full
backward validation before promotion.
