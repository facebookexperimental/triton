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

## f16 gfx1250 Grouped GEMM

Large-tile default for the packed ragged-M grouped GEMM TDM path:

```text
A layout: [sum(M_g), K]
B layout: [G, N, K]  # transposed/K-contiguous expert weights
C layout: [sum(M_g), N]
group_offsets: [G + 1]

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
GROUP_M = 4
tdm_pipeline_depth = 2
l2_prefetch_distance = 0
num_warps = 4
waves_per_eu = 1
```

This is a full-tile path: every `M_g` must be divisible by `BLOCK_M`,
and `N`/`K` must be divisible by `BLOCK_N`/`BLOCK_K`.  The depth-2 TDM
ring-buffer setting is the recommended starting point for this kernel.

Preferred smaller-M configuration:

```text
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128
GROUP_M = 4
tdm_pipeline_depth = 2
dedicated_c_buffer = true
cross_tile_prefetch = true
num_warps = 4
waves_per_eu = 1
```

The dedicated C buffer keeps the asynchronous output store from blocking reuse
of the A ring. It performed better than the same `128x256` tile with C aliased
onto A and remained competitive with the `256x256` large-tile default, while
using a finer M tile for smaller expert groups.

Cross-tile prefetch peels the final two K-loop iterations and reuses each
released input-ring slot for K0/K1 of the next tile assigned to the same
persistent program. It reduced dispatch time on the compact diagnostic shape.
It currently requires dedicated C staging, a depth-2 ring, and an even number
of `BLOCK_K` iterations. It only helps groups with more tiles than persistent
programs. Each persistent program primes its first tile once in the group
preheader; later tiles are fed by the preceding tile's peeled tail. The
preheader issues K0 before constructing the tile-adjusted descriptors used for
K1, overlapping descriptor setup with K0 movement. The steady loop derives
producer/consumer indices from its canonical K-loop IV instead of carrying two
additional scalar counters.

The mirrored `256x128x128` depth-2 configuration also supports a dedicated C
buffer and performed better than its aliased form, but `128x256x128` was the
better asymmetric seed in local tuning.

Pass `--auto_config` to rank the validated `256x256`, `128x256`, `256x128`,
and `128x128` seeds using:

```text
relative saturated rate * CU utilization * useful/padded FLOPs
```

The selector keeps explicit `-BM/-BN` arguments available for controlled
profiling and unsupported shapes.

Persistent program remapping is independently selectable with
`--xcd_remap none|balanced|chunked`. `balanced` assigns each XCD a contiguous,
evenly sized logical PID range. `chunked` assigns configurable logical PID
chunks per XCD (`--xcd_chunk`, default 2). The default remains `none` until the
mapping is tuned on multi-XCD gfx1250 execution.

Correctness and compile checks:

```bash
pytest -s --tb=short third_party/tlx/tutorials/amd_grouped_gemm_gfx1250_test.py
```

Benchmark run:

```bash
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250_test.py \
  --m_list 512,512,512,512 \
  -N 1024 -K 2048 \
  -BM 256 -BN 256 -BK 128 \
  --group_m 4 \
  --tdm_pipeline_depth 2 \
  --l2_prefetch_distance 0 \
  --benchmark_mode eager --benchmark_num_iters 32
```

Pass `--benchmark_mode none --check` to run once and check against
`torch.matmul`.

Smaller-M dedicated-C run:

```bash
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250_test.py \
  --m_list 512,256 \
  -N 512 -K 2048 \
  --num_programs 4 \
  --auto_config \
  --benchmark_mode none --check
```

Compact full-path diagnostic shape:

```bash
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250_test.py \
  --m_list 512,256 \
  -N 1024 -K 1024 \
  -BM 256 -BN 256 -BK 128 \
  --group_m 4 \
  --tdm_pipeline_depth 2 \
  --l2_prefetch_distance 0 \
  --num_programs 4 \
  --benchmark_mode none --check
```

The two groups contain 8 and 4 output tiles, respectively. With four
persistent programs, every program handles two tiles in the first group and
one tile in the second, with eight K iterations per tile. This exercises the
steady K loop, the in-loop pending-C store, the
per-group final flush, the group transition, and both multi-tile and
single-tile group paths. Since pending C state is reset at each group
boundary, total tile count alone is insufficient: at least one individual
group must contain more tiles than `num_programs`.

Large regular benchmark shape:

```text
G = 16
M_g = 4096 for every group
N = 4096
K = 4096
```

```bash
M_LIST=$(printf '4096,%.0s' {1..15})4096
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250_test.py \
  --m_list "${M_LIST}" \
  -N 4096 -K 4096 \
  -BM 256 -BN 256 -BK 128 \
  --group_m 4 \
  --tdm_pipeline_depth 2 \
  --l2_prefetch_distance 0 \
  --benchmark_mode eager --benchmark_num_iters 32
```

Run the regular-shape sweep, where each tuple is
`(G, M_per_group, N, K)`:

```bash
python3 third_party/tlx/tutorials/bench_grouped_gemm_gfx1250.py
```

The default sweep covers the `G=8/32`, `M_per_group=32768/65536`,
`N=4096/8192`, `K=4096` combinations plus the `16x4096x4096x4096`
reference. Each case runs in a separate process to release its large GPU
allocations before the next case. Use `--dry-run` to inspect commands,
`--csv results.csv` for a summary file, or repeat `--case G,M,N,K` to run a
subset.
