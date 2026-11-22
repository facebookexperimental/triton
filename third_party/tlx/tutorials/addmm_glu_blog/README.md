# Optimizing GEMM + Activation on CDNA4 with TLX

These four TLX kernels implement a fused **addmm + GLU** for gfx950 (CDNA4 / MI350):

```
X   = A @ B + bias        # the addmm / projection
out = X + X * Y           # the gate / activation
```

on the following targeted shapes: **M = 1024, N = 21568, fp16, K in {256, 512, 1024}**.

## The four versions

| File | Blog section | What it adds |
|------|--------------|--------------|
| `v1_register_staged.py` | V1 – The Starting Point | Correct baseline; every tile takes the register-staging detour (HBM → regs → LDS → regs → MFMA), serial load/stage/compute. |
| `v2_direct_to_lds.py` | V2 – Direct to LDS, Prefetching, Swizzling, Autotuning | Async direct-to-LDS loads, two-stage warp pipeline, XCD swizzle + grouped tile ordering + `.cs` streaming hints, autotuned tiles. |
| `v3_deep_pipeline_persistent.py` | V3 – A Deeper Pipeline and Persistent Scheduling | NUM_BUFFERS 2 → 3 with combined A/B commit groups so the hot-loop wait relaxes from a full drain to one tile in flight; persistent scheduling. Epilogue still unfused. |
| `v4_fused_epilogue.py` | V4 – Making the Epilogue Nearly Free | Bias folded into the accumulator, Y prefetched into registers under the drain matmuls, and the gate collapsed to a single packed `tl.fma`. Differs from V3 only in the epilogue. |

Each file is self-contained (kernel + `BEST_CONFIG` + a `run(a, b, bias, y)` launcher)
and, when run directly (`python v3_deep_pipeline_persistent.py`), checks itself for
correctness against a PyTorch reference at all three K.

## Benchmarking

`bench.py` runs all four versions and compares them to a **torch.compile**
path and to **rocBLAS** (the matmul alone — no bias, no gate):

```bash
python bench.py                                  # library column = rocBLAS
TORCH_BLAS_PREFER_HIPBLASLT=1 python bench.py    # library column = hipBLASLt
```

It prints TFLOPS (over the GEMM FLOPs) and speedups, and verifies correctness.

## Correctness

Every version is checked with `torch.allclose(out, reference, atol=2e-1, rtol=2e-2)`,
where the reference is `(A@B + bias) + (A@B + bias) * Y` accumulated in fp32 — the
tolerances reflect fp16 accumulation over K up to 1024.
