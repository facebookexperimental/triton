# Standalone buf157 backward benchmark

This is an OSS-only port of the `buf157` RLLayer backward fusion benchmark. It
does not require Buck or `generative_recommenders`; it uses only PyTorch and
the Triton checkout under test.

Run the exact `(M, N, K) = (512, 256, 2,097,152)` case from the repository root:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 \
  python third_party/tlx/tutorials/bwd_layernorm_mul_dropout_buf157.py \
  --warmup 5 --samples 20 --reps 3 --verification-seeds 0 1 2 \
  --json /tmp/rllayer_bwd_layernorm_mul_dropout_buf157.json
```

The exact case retains roughly 11 GiB before accuracy-check temporaries; a
three-seed run can peak well above 24 GiB and is intended for a large-memory
NVIDIA GPU. For a quick compile and correctness smoke test, use the smallest
schedule-compatible row count:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 \
  python third_party/tlx/tutorials/bwd_layernorm_mul_dropout_buf157.py \
  --rows 8192 --check-only
```

The unfused path runs the specialized LayerNorm/mul backward with `compute_y`
enabled and calls `torch.mm` on the materialized `Y`. The fused path disables
the `Y` store and reconstructs `concat(SiLU(U), LayerNorm(X) * U)` inside the
split-K dW GEMM prologue. Both paths retain and compare `dx`, `du`, LayerNorm
`dweight`, `dbias`, and the projection dW result.

## TLX kernel

The TLX version starts from the tuned FP32-workspace Blackwell GEMM and
replaces its A-operand TMA load with the fused reconstructed-Y prologue:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 \
  third_party/tlx/denoise.sh python \
  third_party/tlx/tutorials/bwd_layernorm_mul_dropout_buf157_tlx.py \
  --warmup 5 --samples 20 --reps 3 --verification-seeds 0 1 2 \
  --json /tmp/rllayer_bwd_layernorm_mul_dropout_buf157_tlx_fp32.json
```

The tuned GB200 configuration is BM256/BN256/BK64, two CTAs, four SMEM
stages, one TMEM buffer, split-K 76, FP32 split workspace, and a 16-warp
software producer. At the exact production shape, the three verification
seeds have projection-dW relative L2 errors of 3.97e-4, 4.19e-4, and 4.40e-4.
The best-of-three median end-to-end times were 2.547 ms for TLX, 3.510 ms for
the original Triton fused kernel, and 2.076 ms for the unfused PyTorch/cuBLAS
path on the tested GB200. Passing `--fp16-workspace` reduced the TLX result to
2.538 ms; its three projection-dW relative L2 errors were 8.53e-4, 8.86e-4,
and 8.27e-4, so FP32 remains the safer default.
