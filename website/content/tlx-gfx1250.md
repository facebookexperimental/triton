# TLX on gfx1250 (AMD RDNA4 / MI450 generation)

The distinguishing TLX feature on this target is TDM (Tensor Descriptor
Machine): hardware descriptors with fused multi-member loads, multicast,
and prefetch. Hardware tags follow [TLX](tlx.html).

## Supported `tlx.ops`

One blessed implementation per op from `ops/_catalog.py`, with the columns
of the [gfx950 table](tlx-gfx950.html).

| Op | Variant | Impl | Dtypes | Requires | Backward | What it does |
|---|---|---|---|---|---|---|
| `flash_attn_mxfp8` | wmma_tdm_mxfp8 | `flash_attn_mxfp8/gfx1250.py:flash_attn_mxfp8` | bfloat16 | — | no | MXFP8 attention forward, D128: E4M3 Q/K/V + E8M0 scales, both WMMAs scaled; TDM loads into LDS, softmax and WMMA stages ping-ponged across warp groups; persistent non-causal and causal kernels. |
| `grouped_gemm` | persistent_tdm | `grouped_gemm/gfx1250.py:grouped_gemm` | float16 | — | no | Persistent ragged-M grouped GEMM over pointer tables: 256x256x128 tiles, two-stage fused TDM ring with cross-tile prefetch, four-workgroup TDM multicast when the groups allow it. Common N/K; K a multiple of 128 with at least two blocks; M/N tails masked. |
| `mm_mxfp` | persistent_tdm | `mm_mxfp/gfx1250.py:mm_mxfp` | float8_e4m3fn | — | no | Persistent MXFP8 GEMM with BF16 output: 256x256x128 tiles, four-stage fused TDM ring carried across tiles, statically unrolled loop with split workgroup barriers up to K=16384, C staged through LDS for TDM stores, 4x4 multicast clusters. M and N multiples of 256, K a multiple of 128 and at least 256; natural or cuBLAS-blocked scales. |

`grouped_gemm` was promoted from the tutorial
[`amd_grouped_gemm_gfx1250`](third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/README.md).
`mm_mxfp/gfx1250.py` also holds the general MXFP kernel; see
[MXFP GEMM kernels](#mxfp-gemm-kernels).

## MXFP GEMM kernels

[`mm_mxfp/gfx1250.py`](third_party/tlx/ops/kernels/mm_mxfp/gfx1250.py)
holds two MXFP GEMM kernels for `C = (A * a_scale) @ (B * b_scale)^T` with
E8M0 block scales (one per 32 K elements), FP32 accumulation, and FP32 or FP16
output. `mxgemm_tdm_pipelined` launches either one with every schedule option,
and `matmul` is its config-dictionary form.

- `mxgemm_tdm_pipelined_kernel` follows the Gluon MXFP GEMM example: one
  output tile per program, FP8 or FP4 A and B, optional A scales, baseline
  or sliced K/N/MNK schedules, and none/2way/4way/partial TDM fusion. Its
  grouped modes back
  [`grouped_gemm_mxfp/gfx1250.py`](third_party/tlx/ops/kernels/grouped_gemm_mxfp/gfx1250.py).
- `mxgemm_tdm_persistent_kernel` is a persistent kernel for 256x256 output
  tiles with FP8 E4M3 A and FP8 E4M3 (A8W8) or FP4 E2M1 (A8W4) B. It backs
  `mm_mxfp` and is described below. With `NUM_GROUPS > 0` it also runs the
  A8W8 and A8W4 grouped GEMMs whose groups average more than 128 rows; smaller
  groups stream the weights through `mxgemm_grouped_stream_kernel` in
  [`grouped_gemm_mxfp/gfx1250.py`](third_party/tlx/ops/kernels/grouped_gemm_mxfp/gfx1250.py).

`TILES_PER_WARP=(M, N)` (`--tiles_per_warp M N` on the command line) selects
the WMMA tiles per warp, `2 2` by default.

### Persistent kernel design

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
  slots. With three buffers the next tile's first refill is issued before
  the last C chunks, because TDM operations complete in order; with four it
  follows the last C chunk, since the next tile needs that stage only at its
  fourth step.
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

### Requirements

The persistent kernel requires FP8 E4M3 A with A scales, transposed B
(`[N, K]`), scales preshuffled with `pack_scale` (`wave256` only for A8W8
BK128), partial TDM fusion, and
full 256x256 output tiles with at least `NUM_BUFFERS` K blocks. A8W8 uses
two to four BK128 stages; A8W4 uses three BK256 stages (FP8 B with BK256
uses two). Clustering requires GROUP_SIZE_M 4 or 8, full M groups, and a
program count divisible by 16 that divides the output tile count.
`mxgemm_tdm_pipelined` validates these constraints before launching.

### Benchmark

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/ops/kernels/mm_mxfp/bench_gfx1250.py --csv mxfp.csv
python third_party/tlx/ops/kernels/mm_mxfp/bench_gfx1250.py --variant mx8xmx4 -M 8192 -N 8192 -K 4096
python third_party/tlx/ops/kernels/mm_mxfp/bench_gfx1250.py --dry-run
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
python third_party/tlx/ops/kernels/mm_mxfp/bench_gfx1250.py --single \
  -M 8192 -N 8192 -K 8192 -BK 128 --dtype_b float8_e4m3 --num_buffers 3 --persistent \
  --cluster_size 16 --cluster_layout 4x4 --cluster_barrier_interval 8 --xcd_remap chunked \
  --static_slot_unroll --overlap_peel --output_c_slots 3 --num_programs 256
```

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`;
gfx1250 runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`
and `python/test/unit/tlx_ops/test_mm_mxfp_gfx1250.py`.

## Supported extension builtins (`tlx.*`)

The entries below distinguish existing target-specific tests from source
implementation. **Implemented** means the frontend and AMD lowering provide
that operation; it does not claim that every dtype, layout, or scheduling
mode has been executed on gfx1250. Runtime coverage remains limited to the
cases exercised by the linked tests.

### TDM descriptors

| Builtin | Purpose |
|---|---|
| `update_tensor_descriptor` | Reposition/rebound an AMD TDM descriptor |
| `async_amd_descriptor_load` | Async TDM global-to-local load |
| `async_amd_descriptor_load_fused` | One fused TDM load for 2-4 members |
| `async_amd_descriptor_store` | Async TDM local-to-global store |
| `amd_descriptor_prefetch_tensor` | L2 prefetch hint for a descriptor tile |
| `async_amd_descriptor_wait` | Wait for outstanding TDM operations |

These operations have gfx1250-specific compiler and runtime tests in the
[TDM tests](python/test/unit/language/test_tlx_amd_gfx1250.py) and
[grouped-GEMM tests](third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py).
The [target feature checks](third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp)
enable TDM for the gfx1250 family, not arbitrary newer AMD architectures.

### Memory, compute, and synchronization

| Builtin | Support and restrictions |
|---|---|
| `local_alloc`, `local_view`, `local_slice`, `local_store`, `local_load` | Shared-memory paths implemented; TMEM is NVIDIA-only |
| `local_gather`, `local_scatter` | Portable shared-memory lowering; partitioned-shared layouts are not supported |
| `local_trans`, `local_reshape`, `local_reinterpret`, `require_layout`, `release_layout` | Layout/view transformations implemented; the selected layout must be valid for its consumer |
| `async_load`, `async_load_commit_group`, `async_load_wait_group` | Non-bulk AMD copy path implemented; direct-to-LDS transfers require a legal width/layout, otherwise pointer-based copies can fall back to synchronous loads |
| `require_amd_wmma_layout`, `dot_operand_layout`, `dot_scaled` | WMMA-v3 compute path; use a WMMA parent for dot-operand layouts, not a CDNA MFMA parent |
| `alloc_barriers`, `alloc_warp_barrier`, `barrier_wait`, `barrier_arrive` | LDS barriers; AMD arrival count must be 1 and NVIDIA `scope="cluster"` barriers are not supported |
| `cluster_barrier`, `cluster_cta_rank`, `cluster_size_1d` | Physical multi-CTA cluster support; single-CTA launches fold the cluster operations |
| `workgroup_barrier` | Workgroup synchronization used by the TDM tests |
| `warp_pipeline_stage` | `inter_wave` TDM token-wait pipeline covered by gfx1250 tests; other modes need separate runtime validation |
| `amd_sched_barrier` | Compiler scheduling marker, not a hardware synchronization or memory fence |

The [code-generation tests](python/test/unit/language/test_tlx_codegen.py)
include gfx1250 shared-memory gather/scatter, WMMA, TDM, multicast, and grouped-GEMM compilation. The
[TDM tests](python/test/unit/language/test_tlx_amd_gfx1250.py) also exercise
barrier arrival/wait and inter-wave token waits. Shared-memory gather/scatter
use the [portable lowering](lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp).
Copy and wait handling are implemented in the
[AMD memory lowering](third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp).

### Other implemented AMD operations

| Builtin | Implementation and limits |
|---|---|
| `buffer_load`, `buffer_store` | AMD resource-descriptor global access; see [global memory access](global-memory.html) |
| `buffer_atomic_add` | FP16/BF16/FP32/FP64/int32/int64 lowering; 16-bit atomics require contiguous pairs and a pair-uniform mask |
| `amd_scaled_upcast_fp4` | Packed E2M1 (FP4) × E8M0 scales to BF16/F16 with `v_cvt_scale_pk8_*_fp4`, eight values per instruction; each scale covers a multiple of 8 values (32 for MXFP4) |
| `amd_scaled_downcast_fp4` | FP32/BF16/F16 ÷ E8M0 scales packed to E2M1 with `v_cvt_scalef32_pk8_fp4_*`; round to nearest even, saturating at 6 |
| `extract_slice`, `rematerialized_range` | AMD slice/range lowering, not gfx950-only APIs |
| `amd_register_resident`, `amd_register_class_anchor` | Register-allocation annotations; use VGPR for gfx1250, without assuming CDNA AGPR behavior |
| `cond_barrier` | Conditional workgroup barrier; callers must obey its wave-participation contract |
| `amd_iglp_opt` | AMD compiler scheduling hint; no performance guarantee |
| `warp_predicate` | AMD EXEC-mask lowering; consumers involving dots or shuffles require wave-uniform participation |
| `warp_all`, `warp_any` | AMD ballot-based lane votes |

Compile-only [availability tests](python/test/unit/language/test_tlx_codegen.py)
cover all six listed buffer-atomic dtypes on gfx1250. This group has source
implementations, but existing gfx1250 runtime tests do not establish coverage
of every direct API above. The relevant evidence
is the [buffer/atomic lowering](third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp),
[AMD memory lowering](third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp),
[warp-predicate lowering](third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp),
and [thread/timer lowering](third_party/amd/lib/TritonAMDGPUToLLVM/SPMDOpToLLVM.cpp).

### Portable values and types

The following helpers construct values, types, layouts, or metadata through
portable TLX paths: `assume_uniform`, `zeros`, `dtype_of`, `size_of`,
`get_fp8_format_name`, `assert_same_layout`, `dump_layout`,
`storage_alias_spec`, `allocate_tensor_descriptor`, `thread_id`, `num_warps`,
`layout_encoding`, `shared_layout_encoding`,
`swizzled_shared_layout_encoding`, `swizzled_layout`,
`padded_shared_layout_encoding`, `shared_linear_layout_encoding`,
`slice_layout`, `DummyRegisterLayoutEncoding`, `layout`, `storage_kind`,
`reuse_group`, `reuse_group_type`, `reuse_group_ir_type`,
`storage_alias_spec_type`, `storage_alias_spec_type_class`,
`buffered_tensor`, `buffered_tensor_type`, `mbarrier`, `mbarrier_type`,
and `async_token`.

Constructing a type does not enable an unsupported storage kind or matrix
instruction. In particular, `storage_kind.tmem` is not an AMD allocation
mode. See [memory](buffers.html) and [layout control](layouts.html).

## Unsupported operations

- `buffer_load_to_local` is restricted to CDNA3/CDNA4 by the target verifier. It is rejected on gfx1250; use TDM descriptor loads or the non-bulk pointer-based copy path.
- `amd_scheduled_mfma` and `amd_mfma_commit` require gfx942/gfx950 MFMA versions. They are rejected on gfx1250, which uses WMMA-v3.
- `amd_mfma_layout` can construct a CDNA layout object, but is not the gfx1250 matrix-compute layout. Use `require_amd_wmma_layout` instead.
- `clock64` currently emits `llvm.amdgcn.s.memtime`, which gfx1250 does not support. The frontend symbol exists, but LLVM cannot select that timer instruction for this target; see the [timer lowering](third_party/amd/lib/TritonAMDGPUToLLVM/SPMDOpToLLVM.cpp).
- NVIDIA TMEM operations and `async_load(bulk=True)` are not gfx1250 copy/compute paths.

The CDNA-only restrictions are enforced by the
[AMD verifier](third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp) and
[matrix-instruction selection](third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp).

## Notes

- WMMA queuing is a compiler launch option (`enable_wmma_queuing`), not a `tlx.amd_set_wave_sched_mode` builtin. The AMD backend sets the wave-scheduler-mode register; see the [compiler options](third_party/amd/backend/compiler.py).
- Fused TDM loads require disjoint warp masks and support at most 5 multicast recipients. `async_amd_descriptor_wait` takes pending counts or tokens, not both.
