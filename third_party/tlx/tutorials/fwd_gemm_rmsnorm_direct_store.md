# GEMM, residual, RMSNorm, and direct stores

This OSS benchmark reproduces the fused region in `T289775012`. Two symmetric
BF16 `(5120, 4096, 8192)` GEMMs produce tiles viewed as rows of width 256. Each
branch then adds a residual with an explicit BF16 boundary, computes weighted
RMSNorm with an FP32 sum-of-squares and rstd, saves the BF16 post-add tensor and
FP32 rstd, and writes normalized BF16 values directly to two destination
partitions.

The TLX kernel is a genuine epilogue fusion: each branch is one kernel launch
containing TMA operand loads, FP32 tensor-core accumulation, residual addition,
the RMSNorm reduction, and both destination stores. The two independent model
branches remain two launches.

## Winning GB200 schedule

The selected schedule uses `BM128/BN256/BK64`, four shared-memory operand
buffers, one FP32 TMEM accumulator, and the uncapped eight-warp default
epilogue. The BF16 post-add tile is staged in shared memory for the second
normalization pass, reusing the B-operand allocation after GEMM.

On idle GPU 0 with locked clocks, `warmup=10`, `samples=50`, and five
alternating repetitions, the best medians were:

| Implementation | Complete two-branch latency |
|---|---:|
| OSS unfused (2 cuBLAS GEMMs + 5 Triton launches) | `725.712 us` |
| fused TLX | `699.984 us` |

The fused path saves `25.728 us` (`1.0368x`). Every fused repetition was below
the paired unfused repetition. Across seeds 0, 1, and 2, saved post-add outputs
are bit exact. The maximum rstd error is `2.29e-5`; the maximum normalized BF16
destination error is `4.88e-4`. These differences come from the allowed FP32
reduction reassociation; accumulator precision is unchanged.

PTXAS reports 168 registers per thread, a 144-byte stack frame, 144 bytes of
spill stores, and 156 bytes of spill loads. Capping the epilogue at 160
registers regresses to `791 us`, so the tiny spill is preferable.

Rejected short-run candidates (best of three medians) include:

| Change | Latency |
|---|---:|
| three operand buffers | `764.048 us` |
| `BK128`, two operand buffers | `778.336 us` |
| `BM64` | `952.352 us` |
| two column epilogue tasks with FP32 row-sum exchange | `853.392 us` |
| one epilogue in 128-column subtiles | `705.776 us` |
| one epilogue in 64-column subtiles | `713.056 us` |

Run the locked benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:third_party/tlx/tutorials \
  third_party/tlx/denoise.sh \
  /home/mren/.conda/envs/metamain/bin/python \
  third_party/tlx/tutorials/fwd_gemm_rmsnorm_direct_store_tlx.py \
  --warmup 10 --samples 50 --reps 5 --verification-seeds 0 1 2
```

The original fbsource standalone recorded `739.87 us` unfused and `812.99 us`
for its AutoWS candidate. The separate materialized-SwiGLU hybrid had already
shown this TLX boundary winning (`872.86 -> 838.29 us` including identical
SwiGLU launches); this OSS port isolates and confirms that win. Matching
fbsource full-forward validation remains the promotion gate.
