# FP16 gfx1250 Grouped GEMM

Persistent grouped GEMM with packed and pointer-table TDM kernels.

- `amd_grouped_gemm_gfx1250_test.py`: kernels, host interfaces, tests, and
  single-shape benchmark CLI.
- `bench.py`: shape sweep for both kernels, with process isolation and CSV output.

## gfx950-compatible interface

Host call signatures and the setup return structure match
`ops/kernels/grouped_gemm/gfx950.py`:

```python
from triton.language.extra.tlx.tutorials.amd_grouped_gemm_gfx1250 import (
    amd_grouped_gemm_gfx1250_test as gfx1250,
)

outputs = gfx1250.grouped_gemm(group_a, group_b)

# Prepare once, then time or replay only the launch.
args = gfx1250._make_grouped_gemm_args(group_a, group_b, config={"NUM_SM": 8})
a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_count, cfg, outputs = args
gfx1250._perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_count, cfg)
```

| Tensor | Shape | Layout |
|---|---|---|
| `group_a[g]` | `[M_g, K]` | Row-major, strides `(K, 1)` |
| `group_b[g]` | `[K, N]` | Column-major, strides `(1, K)`, e.g. contiguous weights `[N, K].T` |
| `outputs[g]` | `[M_g, N]` | Row-major |

Inputs must be FP16 on one gfx1250 device, with 16-byte-aligned bases and
common positive N/K. Ragged M, zero-row groups, and partial M/N tiles are
supported. The interface is forward-only.

The default matches `bench.py`: 256x256x128 tiles, depth 2, within-group hybrid
prefetch, `C_STAGING_MODE=0`, chunked XCD remapping, four warps, and four-workgroup
multicast when the cluster requirements below hold. Otherwise it uses cluster
size 1 with the same schedule. Explicit `CLUSTER_SIZE` requests must satisfy
the cluster requirements or raise `ValueError`.
The default requires K divisible by 256. Other supported schedules require K divisible
by 128 and enough K blocks to fill the pipeline; K=384 can use
`config={"CROSS_TILE_PREFETCH": False}`. `NUM_SM` caps persistent programs at
the number of output tiles.

Setup validates inputs and tuning before allocating outputs and metadata.
Unsupported requests raise `ValueError`. Config uses common gfx950 keys
(`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `NUM_BUFFERS`) plus TDM options in
`_GROUPED_GEMM_INTERFACE_CONFIG`. Only schedules in
`_GROUPED_GEMM_INTERFACE_SCHEDULES` are accepted. gfx950-only keys such as
`TILE_MODE` and `HAS_K_TAIL` are rejected; use the cfg returned by this setup.

Prepared launches perform no device allocation, readback, or matrix repacking.
Keep inputs, tables, and outputs alive through completion. In-place metadata
updates must preserve the validated restrictions, cluster configuration, and
selected tail specialization; `_perf_fn` trusts table contents.

## Other entry points

`grouped_gemm_tdm_list(group_a, group_b, group_c=None, ...)` exposes the
tutorial's tuning and benchmark options, including caller-provided outputs.
It constructs metadata on each call and requires the same canonical input
strides as the host interface above.

`grouped_gemm_tdm_list_kernel` accepts gfx950's six leading device arguments:

```text
group_a_ptrs, group_b_ptrs, group_c_ptrs: int64 [G]
group_gemm_sizes: int32 [G, 3] containing M_g, N, K
g_lds: int32 [G, 3] containing lda_g, ldb_g, ldc_g
group_size: G
```

Size/stride tables may also be flat `[3*G]`, as emitted by the setup function.
Pass tuning arguments and compile-time `K`; N is read from the size table.
The caller must validate common N/K and pipeline/cluster restrictions. This
entry point also supports padded A/B leading strides divisible by 8 and
arbitrary C leading strides, with 16-byte-aligned bases.

The kernel defaults to `ALLOW_MN_TAILS=True`, preserving tile, pipeline, and
output-staging settings. Descriptor bounds handle TDM tails; direct stores
use masks. Odd N needs no output padding. The square cross-tile path without
L2 prefetch sets M/N input bounds once per tile, outside the steady K loop.
For full M/N tiles and C leading strides divisible by 8,
`ALLOW_MN_TAILS=False` selects the aligned specialization. List wrappers select
it automatically; clustered launches require it.

`grouped_gemm_tdm` is the packed interface:

```text
A: [sum(M_g), K]    B: [G, N, K]    C: [sum(M_g), N]
group_offsets: [G + 1]
```

Every M, N, and K must cover full tiles. The packed and list JIT entry points
share the TDM pipeline, prefetch, and output implementation.

## Schedules

The main schedules use BK=128, depth 2, `group_m=4`, four warps, and
`waves_per_eu=1`:

| Tile | Cross-tile prefetch | Dedicated C option | Output |
|---|---|---|---|
| 256x256 | Off | Off | TDM, staging aliases the A ring |
| 256x256 | Within group | Off | Eight TDM stores through two 32x256 LDS slots |
| 256x256 | Across groups | On | Direct vector stores, preserving input rings |
| 128x256 or 256x128 | Within group | On | TDM through dedicated C staging |

Cross-tile prefetch reuses released ring slots for the next tile's K0/K1.
It requires depth 2 and an even number of K blocks. Within-group prefetch
needs more tiles per group than persistent programs.

`--auto_config` ranks validated 256x256, 128x256, 256x128, and 128x128 seeds
by relative rate × CU utilization × useful/padded FLOPs. Remapping is selectable
with `--xcd_remap none|balanced|chunked`; the standalone script defaults to
`none`, while `bench.py` defaults to `chunked`.

## Tests and benchmarks

From the repository root:

```bash
pytest -s --tb=short \
  third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py
pytest -s --tb=short python/test/unit/language/test_tlx_amd_gfx1250.py -k grouped_gemm
```

Run the following commands from this directory. Check a list-kernel shape:

```bash
python3 amd_grouped_gemm_gfx1250_test.py \
  --kernel list --m_list 257,513 -N 257 -K 512 \
  -BM 256 -BN 256 -BK 128 --num_programs 4 \
  --tdm_pipeline_depth 2 --cross_tile_prefetch --benchmark_mode none --check
```

Use `--benchmark_mode eager` or `graph` and `--benchmark_num_iters 32` to time
a single shape. Graph mode removes launch overhead. Timing excludes metadata
and output allocation.

The default sweep runs packed then list with identical settings for every
shape. It uses the 256x256 hybrid schedule, depth 2, chunked remapping, and
four-workgroup input multicast. Cases are `(G, M_per_group, N, K)`; each runs
in a separate process to release its allocations.

```bash
python3 bench.py --csv grouped_gemm_results.csv
python3 bench.py --dry-run
python3 bench.py --case 16,4096,4096,4096 --kernel list --cluster-size 1
```

`--kernel packed|list` selects one variant; the default runs both and labels
the summary/CSV with a `kernel` column. With `--cluster-size 1`, use
`--no-cross-tile-prefetch` for alias-C or `--auto-config` for tile selection.

Compare automatic wave specialization with the baseline:

```bash
python3 bench.py --case 2,1024,1024,512 --num-programs 16 \
  --wave-specialization all --check --csv wave_specialization.csv
python3 bench.py --wave-specialization on
```

`--wave-specialization off|on|all` defaults to `off`. `all` runs each shape/kernel
combination with specialization disabled and enabled, in separate processes.
The summary and CSV identify the mode. The standalone kernel script accepts
`--wave_specialization off|on`. The packed/list host wrappers expose
`enable_wave_specialization=False`.

## Cluster multicast

Cluster size 2 shares B within a workgroup pair; size 4 shares A/B across a
two-by-two output region. Cluster barriers protect input-slot reuse.
`--no-cluster-multicast` retains synchronization but loads inputs independently.
Program counts refer to physical workgroups.

Clustering requires:

- 256x256x128 tiles, depth 2, `group_m=4`, cross-tile prefetch on, dedicated C
  staging and L2 prefetch off.
- Chunked remapping with eight logical XCDs and chunk size 2.
- Equal positive M divisible by 1024; N divisible by 512.
- Program count divisible by 16 and dividing each group's output tile count.
  Set `--num-programs` explicitly if the device CU count does not satisfy this.
- Automatic configuration selection disabled.

```bash
python3 bench.py --cluster-size 4 --benchmark-mode graph --check
python3 bench.py --cluster-size 4 --no-cluster-multicast --benchmark-mode graph --check
python3 bench.py --cluster-size 1 --benchmark-mode graph --check
```

The standalone kernel script uses underscores in flags (`--cluster_size`,
`--cluster_multicast`, etc.) and defaults to cluster size 1; `bench.py` uses
hyphens.
