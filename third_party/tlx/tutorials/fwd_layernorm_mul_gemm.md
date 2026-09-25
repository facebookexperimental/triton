# LayerNorm/mul GEMM prologue

This OSS-only benchmark reproduces the two symmetric forward regions in
`T290058050` without Buck or fbsource imports:

```text
mean, rstd = LayerNormStats(X[2097152, 256])
A = BF16(concat(SiLU(U), (LayerNorm(X) * gamma + beta) * U))
projection = BF16(FP32(A @ W[512, 256]) + FP32(residual))
```

Mean and rstd remain live outputs. The concatenated 2 GiB BF16 activation is a
semantic rounding boundary, but the tuned kernel constructs each tile directly
in shared memory instead of materializing it globally.

The standalone reference and original fused Triton seed are in
`fwd_layernorm_mul_gemm.py`. The TLX kernels and benchmark driver are in
`fwd_layernorm_mul_gemm_tlx.py`.

## Winning schedule

- statistics: 16 rows/program, four warps
- projection: `BM64 / BN256 / BK128`
- one static persistent CTA per GB200 SM
- separate two-entry A and B shared-memory rings
- one TMEM accumulator
- eight-warps/128-register computed-A producer, one-warp MMA task, four-warp
  residual epilogue

The producer issues each weight TMA load before computing the corresponding
BF16 A tile, overlapping the descriptor load with SiLU or affine-LayerNorm/mul
work. Separate A/B barriers make asymmetric-buffer experiments possible, but
two buffers for each operand are the measured optimum.

Run on a free Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/fwd_layernorm_mul_gemm_tlx.py \
  --verification-seeds 0 1 2
```

GB200 locked-clock medians (20 samples, five repetitions):

| Implementation | Medians (ms) | Best (ms) |
|---|---:|---:|
| materialized activation + PyTorch/cuBLAS | 1.754, 1.757, 1.757, 1.755, 1.755 | 1.754 |
| OSS port of the original fused Triton seed | 5.012, 5.012, 5.011, 5.012, 5.012 | 5.011 |
| tuned fused TLX | 1.448, 1.447, 1.447, 1.447, 1.446 | 1.446 |

The TLX path is 3.47x faster than the OSS fused seed and 17.6% faster than the
unfused materialization plus addmm path. The initial `BK64`, five-SMEM-stage,
two-TMEM-buffer TLX schedule measured about 1.98 ms.

Across target-shape seeds 0/1/2, projection relative-L2 against an explicit
FP32-accumulating BF16-boundary reference is `5.89e-6 / 6.01e-6 / 5.23e-6`;
mean and rstd are within `1.2e-8` relative-L2. Relative-L2 against the original
fused seed is at most `1.78e-5` for projection and `3.8e-8` for statistics. The
benchmark keeps the actual BF16-output `torch.addmm` path for
timing, while its accuracy reference requests FP32 GEMM output explicitly;
this avoids the OSS PyTorch build's reduced-precision BF16 reduction mode and
matches the task's FP32-accumulation contract.

## Profiling

NCU reports 128 registers/thread, 172.31 KiB dynamic shared memory, one resident
CTA per SM, and 25.0% achieved occupancy. The kernel has about 5.28 MB of local
loads and 1.12 MB of local stores over the complete 2M-row launch. Lowering the
producer register allocation to 96 increases spills and regresses latency to
about 2.14 ms; raising it to 144 also regresses to about 1.87 ms because it
starves the other partitions. A de-unrolled producer reduces live ranges but
regresses to about 1.73 ms. The small spill volume is therefore preferable to
the tested register-allocation alternatives.

The instrumented kernel reaches 45.5% SM throughput and 36.2% memory throughput.
Long-scoreboard dependencies account for about 45.1% of issue latency, which is
consistent with the computed prologue's direct X/U loads. Shared memory and
registers both limit the kernel to one block per SM.

## Rejected variants

- `BM128` (`~3.48 ms`) loses parallelism and makes its software producer too
  large; Blackwell TMEM does not support `BM32` for this MMA.
- `BK64` (`~1.98 ms`) pays twice as many barrier and TMA issue steps; `BK256`
  (`~1.88 ms`) leaves only one shared-memory stage.
- One or three-plus A stages, one B stage, and two TMEM accumulators all regress.
- Two resident waves with the shallow one-B-buffer schedule measure about
  `1.89 ms`; occupancy does not compensate for serialized B loading.
- Keeping U/gamma/beta with `evict_last` is slightly slower (`~1.46 ms`).
- The GEO-inspired single-X-read variant is correct but measures about
  `1.78 ms`. Its full-K A ring and extra normalizer task pay off when A is reused
  over several N tiles; this shape has exactly one `BN256` output tile.

The remaining optimization target is the computed producer's global-load
latency. A materially different design would need to reduce X/U traffic or
publish one computed A tile to additional consumers without introducing the
single-buffer B serialization seen here.
