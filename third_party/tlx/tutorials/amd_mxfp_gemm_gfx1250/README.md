# gfx1250 MXFP GEMM

This directory contains TLX MXFP GEMM kernels for AMD gfx1250:
`C = (A * a_scale) @ (B * b_scale)^T` with E8M0 block scales (one per 32 K
elements), FP32 accumulation, and FP32 or FP16 output.

- [`amd_mxfp_gemm_tdm_pipelined.py`](amd_mxfp_gemm_tdm_pipelined.py): the
  non-persistent kernel, the config-based `matmul` API, the
  `mxgemm_tdm_pipelined` API, and a single-shape command line.
- The persistent kernel, the device helpers it shares with the non-persistent
  kernel, and `pack_scale` live in
  [`ops/kernels/mm_mxfp8/gfx1250.py`](../../ops/kernels/mm_mxfp8/gfx1250.py),
  which backs `tlx.ops.mm_mxfp8` on gfx1250; the tutorial imports them.
- [`bench.py`](bench.py): a multi-shape benchmark of the persistent kernel
  that runs each case in a separate process.

Two kernels are provided:

- `mxgemm_tdm_pipelined_kernel` follows the Gluon MXFP GEMM example: one
  output tile per program, FP8 or FP4 A and B, optional A scales, baseline
  or sliced K/N/MNK schedules, and none/2way/4way/partial TDM fusion.
- `mxgemm_tdm_persistent_kernel` is a persistent kernel for 256x256 output
  tiles with FP8 E4M3 A and FP8 E4M3 (A8W8) or FP4 E2M1 (A8W4) B. The rest
  of this document describes it.

## Persistent kernel design

**Work split.** Each program walks a sequence of output tiles in group-M
order. Four waves, one per SIMD, split a 256x256 tile 2x2, so each wave
keeps a 128x128 FP32 accumulator in registers and issues
`v_wmma_scale_f32_16x16x128_f8f6f4` for its four 64x64 quadrants. A8W8 uses
K blocks of 128 (BK128); A8W4 uses BK256.

**Input ring.** A, B, and both scale tensors stream into LDS through a ring
of `NUM_BUFFERS` TDM stages. Loads use partial fusion: A data and its
scales are issued by waves 0 and 2, B data and its scales by waves 1 and
3, so each wave issues two TDM instructions per stage. Each K step

1. waits at a workgroup barrier until all waves have read the slot to
   refill,
2. refills that slot with a stage `NUM_BUFFERS` steps ahead,
3. waits for the next stage's TDM transfers (`s_wait_tensorcnt`),
4. waits at a workgroup barrier until the stage is visible to all waves.

Matrix instructions of the current step are scheduled across this
sequence, and operand reads for the next step start as soon as its stage
is visible. Scales are read with the payloads into the WMMA scale
operands.

**Tile transition.** With cross-tile prefetch (A8W8), the last
`NUM_BUFFERS` K steps of a tile refill each freed slot with the next tile's
inputs, so the next tile's data is in flight while the current tile
finishes. The C epilogue converts the accumulators to the output type,
writes them to LDS in 64-row chunks, and stores each chunk with a TDM
store:

- A8W8 stages C through dedicated LDS slots (`OUTPUT_C_SLOTS`). A chunk
  waits only when its slot's previous TDM store is still pending.
- A8W4 stages C in A input stages it no longer reads, keeping the A slot
  pitch so retiring a C slot frees exactly one input stage. Without
  cross-tile prefetch, each tile is primed after its C stores.

**Clusters.** `CLUSTER_SIZE` workgroups form a cluster whose matching TDM
loads are multicast. A grid layout `RxC` maps a cluster onto R M tiles by
C N tiles, so each A tile is loaded once for C workgroups and each B tile
once for R. A cluster barrier every `CLUSTER_BARRIER_INTERVAL` K blocks
keeps the members' requests close enough to merge; each workgroup's own
waits protect its LDS, so the barrier is not needed for correctness.
`XCD_REMAP` permutes whole clusters across XCDs so that clusters on one XCD
share inputs.

**A8W8 options.**

- `STATIC_SLOT_UNROLL` unrolls the three- or four-stage steady loop by the
  ring depth, so every step reads a loop-invariant LDS slot and operand
  addresses leave the loop. Each tile's first step is peeled so its WMMAs
  read zero accumulators as an inline constant. The A and B slots are then
  split into 64-row pieces that alternate between two LDS partitions,
  reducing contention among the waves that read one operand subtile.
- `OVERLAP_PEEL` runs each next tile's first K step inside the current
  tile's C epilogue: after a C quadrant is staged, its registers take that
  quadrant's next-tile WMMAs, which execute while later C chunks wait for
  slots. The next tile's first refill is issued before the last C chunks,
  because TDM operations complete in order.
- Without the static unroll, the three-stage loop reads each operand's
  scale before its payload, so each quadrant waits only for its own LDS
  reads.
- A four-stage ring uses two smaller C slots (64x64 for FP32, 64x128 for
  FP16) and a grouped operand prefetch.
- `MERGED_HANDOFF_BARRIER` (four-stage static loop) replaces the K step's
  two workgroup barriers with one, split around register-only work. Each
  step reads both second halves of its slot during the C00 WMMAs, waits for
  its share of the next stage, signals (`tlx.named_barrier_arrive`, which
  lowers to `s_barrier_signal`), runs the C01 WMMAs, waits
  (`tlx.named_barrier_wait`, `s_barrier_wait`), then refills the slot and
  reads the next stage across the C10 and C11 WMMAs. The one rendezvous
  orders the slot's last reads before its refill and publishes the next
  stage to every wave, so the operand reads are relaxed and Membar adds no
  barriers of its own.
- `STATIC_CLUSTER_STEPS` drops the run-time cluster-rendezvous test from the
  unrolled steps that can never refill a rendezvous stage.
- `SCALE_LAYOUT="wave256"` uses a second preshuffled scale format
  (`pack_scale(x, layout="wave256")`): one row per 256 M/N rows, ordered per
  K128 step as `[row bit 5][row % 32][row bit 7][row bit 6][4 K]`, so each
  lane's 16 bytes are the four scale dwords its wave needs for both 128-row
  quadrants. The merged step reads each operand's scales with one
  `ds_load_b128` and splits the quadrants in registers; the other schedules
  read one quadrant per `ds_load_b64`. The default `block128` format stores
  one row per 128 rows.

**A8W4 options.** `SPLIT_CLUSTER_BARRIER` signals the periodic cluster
rendezvous before independent matrix work and waits just before the
refill. `EARLY_C_STORES` interleaves the final quadrant dots with C stores.

**FP16 output** rounds the FP32 accumulators once in registers before
staging, halving C's LDS, TDM, and memory traffic. C buffers that alias
input slots hold twice the rows at the same pitch.

## Requirements

The persistent kernel requires FP8 E4M3 A with A scales, transposed B
(`[N, K]`), scales preshuffled with `pack_scale` (`wave256` only for A8W8
BK128), partial TDM fusion, and
full 256x256 output tiles with at least `NUM_BUFFERS` K blocks. A8W8 uses
two to four BK128 stages; A8W4 uses three BK256 stages (FP8 B with BK256
uses two). Clustering requires GROUP_SIZE_M 4 or 8, full M groups, and a
program count divisible by 16 that divides the output tile count.
`mxgemm_tdm_pipelined` validates these constraints before launching.

## Benchmark

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4 -M 8192 -N 8192 -K 4096
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

By default both variants run at 8192x8192x4096, 8192x8192x8192,
16384x16384x16384, and 32768x32768x32768, first with FP32 and then with FP16
output, with this configuration:

| Setting | `mx8xmx8` (A8W8) | `mx8xmx4` (A8W4) |
| --- | --- | --- |
| B format, BK | FP8 E4M3, 128 | FP4 E2M1, 256 |
| Input ring | FP32: 3 stages, 4 for K >= 16384; FP16: 4 stages | 3 stages |
| C staging | FP32: 3 dedicated slots, 2 with four stages; FP16: 2 slots | retired A stages |
| Cross-tile prefetch | on | off |
| Static slot unroll | FP32: three stages with K >= 8192; FP16: K <= 16384 | off |
| Merged handoff barrier, static cluster steps | FP16 with K <= 16384 | off |
| Overlap peel | FP32: three stages; FP16: K <= 16384 | off |
| Scale layout | block128 (`--scale-layout wave256` to change) | block128 |
| Split cluster barrier, early C stores | off | on |
| Clusters | 16 workgroups, 4x4 layout, multicast, barrier every 8 K blocks | same |
| Program order | group-M 8, chunked XCD remap (8 XCDs, chunk 2) | same |
| Programs | 256, capped by the tile count | same |

Each option can be overridden (`--help`). Timing repeats eager batches of
`--benchmark-num-iters` milliseconds: batches run for `--warmup-seconds`,
then the median batch over `--duration-seconds` is reported. The device
clock depends on load, so keep the timing settings fixed when comparing
configurations. The summary prints the AMD code generator revision. The
kernels are tuned with the AMDGPU code generator pinned in
`cmake/amd-llvm-info.json`. Builds with `cmake/meta-llvm-info.json` compile
the code generator against the core LLVM instead; set
`TRITON_AMD_CODEGEN_PATH` to the pinned library to use it.

The standalone command line runs one shape, for example the A8W8 default at
8192 cubed:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/amd_mxfp_gemm_tdm_pipelined.py \
  -M 8192 -N 8192 -K 8192 -BK 128 --dtype_b float8_e4m3 --num_buffers 3 --persistent \
  --cluster_size 16 --cluster_layout 4x4 --cluster_barrier_interval 8 --xcd_remap chunked \
  --static_slot_unroll --overlap_peel --output_c_slots 3 --num_programs 256
```

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`;
gfx1250 runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`
and `third_party/tlx/tutorials/testing/test_correctness.py`.
