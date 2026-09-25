# GEMM with RMSNorm/SiLU backward

This OSS-only benchmark reproduces the D=512 `bwd_buf64_rmsnorm` region from
`T289840347` without Buck or fbsource imports:

```text
gradient = BF16(A[5120,2048] @ W.T[2048,512])
dx = RMSNorm/SiLU backward(saved_input, rstd, weight, gradient)
dweight = column reduction(saved_input, rstd, gradient, weight)
```

The TLX kernel preserves the BF16 GEMM output, computes the complete row-local
`dx`, and writes FP32 dweight partials. A second kernel performs only the final
partial reduction and BF16 cast.

Run on an idle Blackwell GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python \
  third_party/tlx/denoise.sh /path/to/python \
  third_party/tlx/tutorials/bwd_rmsnorm_silu_gemm_tlx.py \
  --verification-seeds 0 1 2 \
  --json /tmp/t289840347_buf64.json
```

## Winning schedule and results

The static schedule uses `BM64/BK64`, four A buffers, two B buffers, two N=256
TMEM accumulators, and two parallel eight-warp epilogue tasks. The epilogues
exchange their row-dot partials through shared memory before producing dx.

Locked GB200 best-of-five medians (`warmup=5`, `samples=20`):

| Implementation | Time | Relative to unfused |
|---|---:|---:|
| PyTorch/cuBLAS plus three Triton kernels | `147.344 us` | `1.000x` |
| compact fused Triton seed | `99.136 us` | `1.486x` |
| fused TLX | `123.104 us` | `1.197x` |

Projection, dx, and dweight are bit-exact for seeds 0/1/2. NCU reports 128
registers/thread, 181.11 KiB dynamic shared memory, one resident block, and
30.0% tensor-pipe activity for the TLX kernel. The unchanged final dweight
reduction is included in all timings.
