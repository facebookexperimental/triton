# Batched GEMM-concat fusion

This OSS benchmark reproduces `T289757919` without Buck or fbsource imports:

```text
projection[5120,32,64] = A[5120,32,256] @ B[5120,256,64]
output[:,0:2048] = projection.flatten(1)
output[:,2048:6144] = features[:,16384:20480]
output[:,6144:8472] = extra[:,0:2328]
```

The same values are written to compact `[5120,8472]` and padded-stride
`[5120,8512][:,:8472]` outputs. This is an epilogue fusion: the concat consumes
the BMM result and independent feature slices; it does not transform either
GEMM input.

The numerical contract is BF16 inputs, FP32 GEMM accumulation, and BF16
outputs. Reduction reassociation is allowed. Correctness uses
`atol=rtol=0.02` and reports max-absolute, relative-L2, and exact equality.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:third_party/tlx/tutorials \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/fwd_batched_gemm_concat_tlx.py \
  --warmup 10 --samples 50 --reps 5 \
  --verification-seeds 0 1 2
```

## Winning schedule

One CTA owns one batch. It first issues the three full 2048-element concat
copies and the final 280-element tail, then computes the exact `32x64` GEMM in
two K=128 steps and writes the BF16 projection directly to both destinations.
The winning launch uses eight warps and two compiler stages. Keeping the copy
in the same CTA is faster than separate copy CTAs because the latter inherit
the GEMM branch's register allocation and add scheduling overhead.

Locked GB200 best-of-five medians (`warmup=10`, `samples=50`) are:

| Comparison | Latency | Relative result |
|---|---:|---:|
| OSS unfused BMM + concat | `195.136 us` | `1.00x` |
| tuned fused Triton | `98.880 us` | `1.97x` |
| persistent TLX/TMA comparison | `124.224 us` | `1.57x` |
| original fused seed, OSS port | `373.248 us` | `0.52x` |
| historical fbsource unfused | `100.460 us` | tuned fused is `1.6%` faster |
| historical fbsource fused seed | `500.160 us` | tuned fused is `5.06x` faster |

The original fbsource attribution was `48.11 us` BMM plus `50.53 us` concat.
In this OSS checkout the independently measured components are `59.968 us`
and `152.784 us`; the scalar-indexed OSS concat is anomalously slow, so the
cross-environment comparison against `100.460 us` is more informative than
the headline OSS speedup. Matching-fbsource and full-forward validation remain
required before production promotion.

All seed, tuned Triton, and TLX results are bit exact for both destinations on
seeds 0/1/2, although exactness is not required by the contract.

## TLX comparison and profiling

The TLX version uses persistent CTAs, four staged TMA operand buffers, a single
TMEM accumulator, one producer warp, one MMA warp, and an eight-warp epilogue.
Its epilogue copies the independent feature slices while the producer/MMA tasks
compute the projection. Blackwell MMAv5 requires M=64 or M=128, however, so the
logical M=32 contraction must use a masked M=64 tile. The doubled A traffic and
MMA work make TLX slower than the exact-width Triton implementation despite the
overlap.

PTXAS reports 50 registers/thread and no spills for the Triton winner. NCU
reports 24.58 KiB dynamic shared memory, 50% theoretical occupancy, 47.29%
achieved occupancy, 64.20% DRAM throughput, and 22.11% SM throughput. This is
primarily a bandwidth/shape-efficiency problem, not register spilling.
