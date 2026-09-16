# FP16 gfx1250 Grouped GEMM

This directory contains a persistent grouped GEMM implementation for gfx1250,
its correctness and compile tests, and a regular-shape benchmark sweep.

## Files

- `amd_grouped_gemm_gfx1250_test.py`: pointer-table baseline, optimized TDM
  kernel, tests, and single-shape benchmark CLI.
- `bench.py`: multi-shape benchmark runner with process isolation and CSV
  output.

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

This default aliases the C staging tile onto the A ring. The square depth-2
schedule bounds operand lifetimes at each dot and enables CDNA5
`SCHED_MODE[2]` so queued WMMAs overlap independent instructions.

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
of the A ring while retaining a finer M tile for smaller expert groups.

Cross-tile prefetch peels the final TDM-ring rotation and reuses released input
slots for K0/K1 of the next tile assigned to the same persistent program. It
requires a depth-2 ring and an even number of `BLOCK_K` iterations. The
asymmetric tiles require `--dedicated_c_buffer` and prefetch within each
group, so they need more tiles per group than persistent programs.

For `256x256x128` with `--cross_tile_prefetch` and dedicated C staging off,
the within-group hybrid pipelines output through two `32x256` LDS slots.
It splits C into eight row chunks and issues TDM stores, waiting only for
the older store before reusing its slot. The final two stores can overlap
the next tile's entry while the prefetched A/B tiles remain intact. TDM output
avoids per-lane output address calculations, and the small staging slots fit
alongside the input rings. With cross-tile prefetch off, C aliases the A ring.

With `--dedicated_c_buffer`, the square schedule also prefetches across group boundaries,
skipping empty groups. It uses vector stores for C, leaving the two input
rings intact. Keeping one K block per loop and bounding operand lifetimes at
each dot limits register pressure. With L2 prefetch disabled, it folds tile
offsets directly into fused TDM loads to shorten descriptor lifetimes.

The square schedule keeps four upcoming group boundaries in scalar state,
refilling them in the preceding tile's tail. It specializes the group count
known from B's shape to eliminate unused refill state. It also enables CDNA5
`SCHED_MODE[2]` to overlap queued WMMAs with independent instructions.

Run the square cross-tile configuration with:

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py \
  --m_list 2048,2048,2048,2048 -N 1024 -K 2048 \
  -BM 256 -BN 256 -BK 128 --num_programs 32 --group_m 4 \
  --tdm_pipeline_depth 2 --l2_prefetch_distance 0 \
  --dedicated_c_buffer --cross_tile_prefetch --benchmark_mode none --check
```

For example, `M=2048, N=1024` gives 32 square tiles. With 32 persistent
programs, each program handles one tile per group and prefetches its tile
from the following nonempty group. At `M=4096`, each program also prefetches
a second tile within the same group.

Each persistent program primes its first tile in the group preheader.
Later tiles are fed by the preceding tile's peeled tail. The square schedule
also skips priming at later group boundaries. For the asymmetric tiles, the
preheader issues K0 before constructing the tile-adjusted descriptors used for
K1, overlapping descriptor setup with K0 movement.

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

Run the complete grouped GEMM test file:

```bash
pytest -s --tb=short \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py
```

Run one shape and compare against `torch.matmul`:

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py \
  --m_list 512,256 \
  -N 512 -K 2048 \
  --num_programs 4 \
  --auto_config \
  --benchmark_mode none --check
```

## Single-Shape Benchmark

```bash
python3 \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py \
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
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py \
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

The sweep defaults to `256x256x128`, depth 2, alias-C, and cross-tile prefetch
disabled for every shape. Use
`--cross-tile-prefetch` to compare the revised hybrid with TDM output stores.
`--auto-config`
instead lets the kernel's general cost model select the configuration,
including whether to use cross-tile prefetch.

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
