# SwiGLU reconstruction GEMM

This OSS-only benchmark reproduces `T289901422` without Buck or fbsource
imports:

```text
hidden = BF16(SiLU(gate[K,N]) * up[K,N])
dweight = down_gradient[K,M].T @ hidden[K,N]
(M,N,K) = (4096,8192,5120)
```

The unfused baseline materializes `hidden` with Triton and calls cuBLAS. The
TLX kernel computes the FP32 SwiGLU expression in its B-operand producer,
rounds it once to BF16 in shared memory, and performs an FP32-accumulating
GEMM without writing `hidden` to global memory.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_swiglu_gemm_tlx.py \
  --verification-seeds 0 1 2 \
  --json /tmp/t289901422.json
```

## Best exact schedule

The best tested exact schedule uses `BM128/BN128/BK64`, four M accumulator
groups, one A slot, three computed-B slots, sixteen producer warps, and one
four-warp epilogue. One reconstructed B tile feeds four MMAs before being
released.

Locked GB200 best-of-five medians (`warmup=5`, `samples=20`):

| Implementation | Time | Relative to unfused |
|---|---:|---:|
| explicit reconstruction + cuBLAS | `272.320 us` | `1.000x` |
| fused Triton seed | `798.368 us` | `0.341x` |
| fused TLX | `813.760 us` | `0.335x` |

The TLX result is bit exact for seeds 0/1/2. A plain `tlx.ops.mm` diagnostic
measures `254.714 us` versus `230.817 us` for cuBLAS, showing that the remaining
gap comes from repeated exact sigmoid reconstruction rather than the base TLX
GEMM.

NCU reports 40 registers/thread, 119.11 KiB dynamic shared memory, one resident
block per SM, 37.5% theoretical occupancy, and 27.6% compute throughput. The
kernel is latency/prologue limited rather than bandwidth saturated.

## Rejected experiments

- One, two, four, and eight M-group schedules measured approximately `2.72`,
  `1.56`, `0.81`, and `0.88 ms`, respectively. Four groups are the best balance
  between reconstruction reuse, MMA issue count, and grid size.
- Reducing A from two buffers to one and increasing B from two to three buffers
  improved the exact path from about `0.835` to `0.814 ms`.
- A two-CTA collaborative MMA is exact but measures `0.884 ms`.
- GEO's `tanh.approx` sigmoid pattern measures `0.525 ms`, but changes the
  result (`relative_l2=4.55e-4`, `max_abs=0.015625`) and is rejected by the
  bit-exact contract.
- Replacing full division with a Newton-refined approximate reciprocal can be
  made bit exact across seeds 0/1/2, but regresses to about `0.91 ms`.
- A DSMEM broadcast prototype was rejected after its synchronization protocol
  failed to complete; it is not exposed by the benchmark driver.

The exact fusion remains substantially slower than materialization plus
cuBLAS. A competitive follow-up needs cross-CTA sharing of reconstructed B
without serializing MMAs, or a contract that explicitly permits the faster
approximate sigmoid.
