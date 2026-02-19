---
name: kernel-perf-testing
description: >
  Run TLX kernel performance benchmarks on Hopper and Blackwell GPUs.
  Use when user asks to benchmark, profile, or measure performance of
  any TLX kernel (GEMM, Flash Attention variants). Handles GPU selection,
  denoise wrapping, and version flags. Never run unless explicitly asked.
disable-model-invocation: true
---

# Kernel Performance Testing

**Never run performance tests unless the user explicitly asks.**

## GPU selection protocol

1. Run `nvidia-smi` to check GPU occupancy.
2. Pick the GPU with the lowest memory usage.
3. Set `CUDA_VISIBLE_DEVICES` to that GPU.

## Benchmark commands

All benchmarks must be wrapped with `denoise.sh` for stable results.

### Hopper GPU

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_gemm_perf.py [--version {ws|pipelined}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_hopper_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]
```

### Blackwell GPU

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_gemm_perf.py [--version {ws|pipelined|clc|2cta}]
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/testing/test_blackwell_fa_perf.py [--version {ws|ws_pipelined|ws_pipelined_pingpong|ws_pipelined_pingpong_persistent}]
```

### Other kernels

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> third_party/tlx/denoise.sh python third_party/tlx/tutorials/<KERNEL.py>
```

## If tests hang

Run `third_party/tlx/killgpu.sh` to kill GPU processes that have been running too long.

## Interpreting results

- Output reports **TFLOPS** for each problem size and configuration.
- Compare against cuBLAS baselines when available (printed alongside Triton results).
- Higher TFLOPS = better. Look for regressions relative to previous runs.
- Check for consistency across runs — high variance suggests noisy measurements (ensure `denoise.sh` is being used).
