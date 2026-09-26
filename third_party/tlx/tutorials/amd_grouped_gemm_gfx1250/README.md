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
`--xcd_remap none|balanced|chunked`. The standalone kernel script defaults to
`none`; the `bench.py` sweep defaults to `chunked`.

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
combinations plus the `16x4096x4096x4096` reference. The `G=1` cases use
`M=N=K=4096`, `8192`, and `16384` and run ordinary GEMM through the same
kernel and configuration.

The sweep defaults to `256x256x128`, depth 2, and the within-group hybrid with
cross-tile prefetch and TDM output stores enabled for every shape. It uses
four-workgroup input multicast, with cluster synchronization before input
refills. XCD remapping defaults to `chunked`.

Use `--cluster-size 1` for ordinary workgroups. Combine it with
`--no-cross-tile-prefetch` to compare the alias-C schedule, `--xcd-remap none`
to disable remapping, or `--auto-config` to let the kernel's general cost
model select the configuration, including whether to use cross-tile prefetch.

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

# Run a single-group (ordinary GEMM) case.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --case 1,4096,4096,4096
```

## Cluster multicast

`--cluster-size 2` shares B within a workgroup pair. `--cluster-size 4`
shares A and B across a logical two-by-two output region. Each recipient
issues the matching TDM request, allowing hardware to multicast the inputs.
`--no-cluster-multicast` keeps the cluster synchronization but loads each
workgroup's inputs independently, providing a comparison control.

The kernel expresses recipient masks on `tlx.async_amd_descriptor_load_fused`
and uses `tlx.cluster_barrier()` before refilling an input slot. The barrier
first synchronizes local waves, then arrives and waits at the AMD cluster
barrier so a remote refill cannot overwrite data another workgroup still reads.
The masks and barriers lower through the compiler's AMD TDM and synchronization
operations. Compilation uses the ordinary JIT cache and launch path.
Each input mask selects two recipients, within gfx1250's limit of five.

The launch uses `ctas_per_cga=(cluster_size, 1, 1)`, with one independent
program per workgroup and one-CTA tensor layouts. The grid and `num_programs`
count physical workgroups; the HIP launcher handles conversion to cluster
counts and checks grid divisibility.

Clustering requires `256x256x128`, depth two, `group_m=4`, cross-tile prefetch
enabled, dedicated C staging and L2 prefetch disabled, and chunked remapping
with eight logical XCDs and chunk size two. Groups must have equal positive
M divisible by 1024, and N must be divisible by 512. The program count must
be divisible by 16 and divide each group's output tile count. These conditions
keep cluster members on the same loop and group boundaries. Set `--num-programs`
explicitly when the device CU count does not satisfy them. Automatic
configuration selection is unsupported with clustering.

```bash
# Four-workgroup multicast.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --cluster-size 4 --benchmark-mode graph --check

# Matching synchronization-only control.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --cluster-size 4 --no-cluster-multicast --benchmark-mode graph --check

# Ordinary workgroups using the same hybrid pipeline.
python3 third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/bench.py \
  --cluster-size 1 --benchmark-mode graph --check
```

The Python wrapper and standalone script expose `cluster_size` and
`cluster_multicast`, using underscores in CLI flags. They default to
`cluster_size=1` for the general ragged-group path.
