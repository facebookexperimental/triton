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

The numerical contract permits GEMM reduction reassociation. The default
`direct` variant uses `tl.sigmoid`; the optional `geo` variant uses GEO's
`0.5 * tanh.approx(0.5 * x) + 0.5` sigmoid. Both retain FP32 pointwise math,
the BF16 pre-dot rounding boundary, FP32 accumulation, and BF16 output. A
change to accumulator precision is outside this contract and must be reported
explicitly.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_swiglu_gemm_tlx.py \
  --variant geo \
  --verification-seeds 0 1 2 \
  --json /tmp/t289901422.json
```

Correctness is gated by `relative_l2 <= 5e-4` and `max_abs <= 0.03125`.
Exact equality is reported as a diagnostic, not required.

## GEO-style approximate schedule

The fastest tested fused schedule uses the GEO packed-FP32 `tanh.approx`
sigmoid with `BM128/BN128/BK128`, four M accumulator groups, one A slot, two
computed-B slots, sixteen producer warps, and an eight-subtile epilogue.

Locked GB200 best-of-five medians (`warmup=5`, `samples=20`):

| Implementation | Time | Relative to unfused |
|---|---:|---:|
| explicit reconstruction + cuBLAS | `272.256 us` | `1.000x` |
| fused Triton seed | `798.336 us` | `0.341x` |
| fused TLX, GEO sigmoid | `495.360 us` | `0.550x` |

Across seeds 0/1/2, relative-L2 is `4.54e-4` to `4.61e-4` and maximum
absolute error is `0.015625`. The GEO variant is 39.1% faster than the best
exact TLX result, but remains 82.0% slower than materialization plus cuBLAS.
The native 2-CTA GEO variant measures `539.904 us` and is slower.

NCU reports 70 registers/thread, 201.01 KiB dynamic shared memory, one resident
block per SM, 37.5% achieved occupancy, 48.3% compute throughput, and 17.3%
DRAM throughput. Both registers and shared memory limit residency to one block.

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
- The initial GEO schedule at `BK64/B3` measured `502.592 us`; increasing the
  K tile to 128 and reducing the computed-B ring to two slots improved it to
  `495.360 us`.
- Reducing the GEO producer to eight warps regresses to about `0.77 ms`; it
  still needs sixteen warps to cover the pointwise prologue.
- Replacing full division with a Newton-refined approximate reciprocal can be
  made bit exact across seeds 0/1/2, but regresses to about `0.91 ms`.
- A DSMEM broadcast prototype was rejected after its synchronization protocol
  failed to complete; it is not exposed by the benchmark driver.

Even with approximate sigmoid, the fusion remains substantially slower than
materialization plus cuBLAS. A competitive follow-up needs cross-CTA sharing
of reconstructed B without serializing MMAs.
