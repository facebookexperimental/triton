# FP16 gfx1250 Grouped GEMM

This directory contains an importable persistent grouped GEMM implementation
for gfx1250, a single-case runner, and a regular-shape benchmark sweep.

## Files

- `grouped_gemm.py`: pointer-table baseline, optimized TDM kernel, launchers,
  and host-side configuration/remapping helpers.
- `run.py`: single-shape correctness and benchmark CLI.
- `bench.py`: multi-shape benchmark runner with process isolation and CSV output.

## Data Layout

The optimized path accepts packed activations, K-contiguous expert weights,
and group offsets:

```text
A: [sum(M_g), K]
B: [G, N, K]
C: [sum(M_g), N]
group_offsets: [G + 1]
```

It is a full-tile path. Every `M_g` must be divisible by `BLOCK_M`, and `N`
and `K` must be divisible by `BLOCK_N` and `BLOCK_K`.

## Configurations

Large-tile default:

```text
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
GROUP_M = 4
tdm_pipeline_depth = 2
l2_prefetch_distance = 0
num_warps = 4
waves_per_eu = 1
```

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
of the A ring. It performed better than aliasing C onto A while retaining a
finer M tile for smaller expert groups.

Cross-tile prefetch peels the final TDM-ring rotation and reuses released input
slots for K0/K1 of the next tile assigned to the same persistent program. It
currently requires dedicated C staging, a depth-2 ring, and an even number of
`BLOCK_K` iterations. It only helps groups with more tiles than persistent
programs.

Each persistent program primes its first tile once in the group preheader.
Later tiles are fed by the preceding tile's peeled tail. The preheader issues
K0 before constructing the tile-adjusted descriptors used for K1, overlapping
descriptor setup with K0 movement.

The mirrored `256x128x128` configuration also supports dedicated C staging,
but `128x256x128` is the preferred asymmetric seed.

Pass `--auto_config` to rank the validated `256x256`, `128x256`, `256x128`,
and `128x128` seeds using:

```text
relative saturated rate * CU utilization * useful/padded FLOPs
```

Persistent program remapping is independently selectable with
`--xcd_remap none|balanced|chunked`. The default is `none`.

## Correctness

Run one shape and compare against `torch.matmul`:

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/run.py \
  --m_list 512,256 \
  -N 512 -K 2048 \
  --num_programs 4 \
  --auto_config \
  --benchmark_mode none --check
```

## Single-Shape Benchmark

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/run.py \
  --m_list 512,512,512,512 \
  -N 1024 -K 2048 \
  -BM 256 -BN 256 -BK 128 \
  --group_m 4 \
  --tdm_pipeline_depth 2 \
  --l2_prefetch_distance 0 \
  --benchmark_mode eager --benchmark_num_iters 32
```

Use `--benchmark_mode graph` to remove launch overhead from the timing.

## Full-Path Diagnostic

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/run.py \
  --m_list 512,256 \
  -N 1024 -K 1024 \
  -BM 256 -BN 256 -BK 128 \
  --group_m 4 \
  --tdm_pipeline_depth 2 \
  --l2_prefetch_distance 0 \
  --num_programs 4 \
  --benchmark_mode none --check
```

The two groups contain both multi-tile and single-tile persistent-program
paths. This exercises the steady K loop, pending C store, per-group final
flush, and group transition.

## Benchmark Sweep

Run the default regular-shape sweep:

```bash
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py
```

Each tuple is `(G, M_per_group, N, K)`. The default cases cover the requested
`G=8/32`, `M_per_group=32768/65536`, `N=4096/8192`, `K=4096`
combinations plus the `16x4096x4096x4096` reference.

Each case runs in a separate process so large GPU allocations are released
before the next case. Useful options:

```bash
# Inspect commands without running them.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py --dry-run

# Write a summary CSV.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --csv grouped_gemm_results.csv

# Run a subset.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --case 16,4096,4096,4096
```
