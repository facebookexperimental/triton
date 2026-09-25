# Saved-LayerNorm GEMM fan-out

This OSS-only benchmark reproduces `T290048886` without Buck or fbsource
imports. One saved affine-LayerNorm activation feeds two GEMMs:

```text
H = BF16((X - mean) * rstd * gamma + beta)       # [2097152, 256]
projection = BF16(FP32(H @ W) + bias)            # W: [256, 768]
dW = BF16(FP32(H.T @ gradient))                  # gradient: [2097152, 1024]
```

The unfused reference materializes `H` once. Each fused route instead
reconstructs the BF16 tile in its GEMM prologue. The projection kernel reuses
one normalized A tile across its three output-N tiles. The transposed dW kernel
uses split-K 38 and an FP32 workspace.

Run the exact target on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_saved_layernorm_gemm_tlx.py \
  --verification-seeds 0 1 2 \
  --json /tmp/t290048886.json
```

For a quick compile and accuracy check:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  /path/to/python third_party/tlx/tutorials/bwd_saved_layernorm_gemm_tlx.py \
  --rows 4096 --check-only
```

## GB200 result

Locked-clock best-of-five medians (`warmup=5`, `samples=20`):

| Region | Unfused | Fused seed | Tuned TLX |
|---|---:|---:|---:|
| projection | `1.125 ms` | `2.221 ms` | `1.295 ms` |
| dW | `1.182 ms` | `2.168 ms` | `1.791 ms` |
| complete fan-out | `1.900 ms` | `4.198 ms` | `3.048 ms` |

Projection is bit-exact for seeds 0/1/2. dW relative-L2 is
`5.84e-4 / 5.91e-4 / 5.94e-4`, below the `1e-3` limit.

The tuned TLX kernels substantially improve the original fused seeds, but do
not beat the unfused fan-out. The unfused path pays the BF16 materialization
once and shares it across both GEMMs; the two standalone fused kernels each
repeat the LayerNorm reconstruction. Further work should therefore share one
producer across both GEMM orientations rather than tune either kernel in
isolation.
