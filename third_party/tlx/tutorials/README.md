# TLX Tutorial Kernel Benchmarks

This directory contains runnable TLX tutorial kernels.

The benchmark CLIs print:

```text
execution time: <ms> ms, <tflops> TFLOPS
```

## MXFP gfx1250 GEMM

Default split fp8e4m3 x fp4 sliceMNK config used for TLX/Gluon comparisons:

```bash
python3 third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_test.py \
  -M 2048 -N 1024 -K 8192 \
  -BM 256 -BN 256 -BK 256 \
  --num_buffers 3 \
  --group_size_m 8 \
  --dtype_a float8_e4m3 --dtype_b float4 \
  --scale_preshuffled --with_a_scale \
  --schedule sliceMNK \
  --partial_tdm --tdm_split \
  --l2_prefetch_distance -1 \
  --benchmark_mode eager --benchmark_num_iters 32
```

Use CUDA-graph benchmarking when launch overhead should be removed:

```bash
python3 third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_test.py \
  -M 2048 -N 1024 -K 8192 \
  -BM 256 -BN 256 -BK 256 \
  --num_buffers 3 \
  --group_size_m 8 \
  --dtype_a float8_e4m3 --dtype_b float4 \
  --scale_preshuffled --with_a_scale \
  --schedule sliceMNK \
  --partial_tdm --tdm_split \
  --l2_prefetch_distance -1 \
  --benchmark_mode graph --benchmark_num_iters 32
```

## f16 gfx1250 GEMM

Single-warp-per-SIMD TDM schedule:

```bash
python3 third_party/tlx/tutorials/amd-tdm-gemm-pipelined_test.py \
  --kernel single_warp \
  -M 2048 -N 1024 -K 8192 \
  -BM 256 -BN 256 \
  --num_buffers 2 \
  --transpose_b \
  --l2_prefetch_distance 2 \
  --benchmark_mode eager --benchmark_num_iters 32
```

Simple two-buffer TDM pipeline:

```bash
python3 third_party/tlx/tutorials/amd-tdm-gemm-pipelined_test.py \
  --kernel simple \
  -M 2048 -N 1024 -K 8192 \
  -BM 128 -BN 128 -BK 32 \
  --no-transpose_b \
  --benchmark_mode eager --benchmark_num_iters 32
```

Pass `--benchmark_mode none --check` to run once and print a max absolute
difference against `torch.matmul` instead of timing.
