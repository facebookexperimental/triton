---
name: kernel-perf-testing
description: >
  Run TLX kernel performance benchmarks on Hopper, Blackwell, and AMD
  (gfx950/CDNA4, gfx1250) GPUs. Use when user asks to benchmark, profile, or
  measure performance of any TLX kernel (GEMM, Flash Attention, addmm+GLU, IKBO
  variants). Handles GPU selection, denoise wrapping (NVIDIA only), and version
  flags. Never run unless explicitly asked.
disable-model-invocation: true
---

# Kernel Performance Testing

**Never run performance tests unless the user explicitly asks.**

Perf scripts live in `third_party/tlx/tutorials/testing/test_<arch>_<op>_perf.py`,
one per (op, input-contract, arch). Each takes `[--version ...]` to select a
kernel variant; with no `--version` it runs all variants for that op.

## NVIDIA (Hopper / Blackwell)

### GPU selection protocol

1. Run `nvidia-smi` to check GPU occupancy.
2. Pick the GPU with the lowest memory usage.
3. Set `CUDA_VISIBLE_DEVICES` to that GPU.

Wrap benchmarks with `denoise.sh` (locks clocks/power) for stable results.

### Hopper GPU

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]
```

### Blackwell GPU

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|clc|pipelined|2cta}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_persistent|ws_pipelined|ws_pipelined_persistent|clc}] [--mode fwd|bwd]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_mxfp8_perf.py
```

## AMD (gfx950 / CDNA4, gfx1250)

`denoise.sh` works on AMD too: it runs the wrapped command with NUMA pinning. Its
GPU clock/power lock is implemented with `nvidia-smi`, so on AMD that part is
simply skipped (no clock lock) — expect a bit more run-to-run variance.

Pick a free GPU with `rocm-smi` (check `GPU use (%)` / VRAM). `denoise.sh` reads
`CUDA_VISIBLE_DEVICES`, which torch on ROCm also honors for device selection.

### gfx950 (CDNA4 / MI350-class)

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_gemm_perf.py [--version {warp_pipeline|pipelined}] [--dtype fp16|bf16]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_fa_perf.py [--version {simple|prefetch|persistent}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_addmm_glu_perf.py [--version {tlx_baseline|tlx_simple_async|tlx_optimized_async|tlx_optimized|tlx_persistent}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_fa_perf.py   # IKBO Flash Attention
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_ikbo_lce_perf.py  # IKBO LCE (distinct op, not attention)
```

### gfx1250

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_amd_mxfp_gemm_perf.py [--version {tdm_pipelined}] [--transpose-b]
```

Each script self-gates (`is_hip`, `is_hip_cdna4`, or `is_hip_gfx1250`) and prints
"Skipping benchmarks" on the wrong hardware.

### Other kernels

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/<KERNEL.py>
```

## If tests hang

Run `third_party/tlx/killgpu.sh` to kill GPU processes that have been running too long.

## Interpreting results

- Output reports **TFLOPS** for each problem size and configuration.
- A reference baseline is printed alongside the Triton/TLX results: cuBLAS/rocBLAS
  for GEMM, SDPA for FA, the PyTorch `pytorch_baseline` for addmm+GLU.
- Higher TFLOPS = better. Look for regressions relative to previous runs.
- Check for consistency across runs — high variance suggests noisy measurements (ensure `denoise.sh` is being used).
