# TTGIR-guided instruction scheduling on AMD

This document describes AMD TTGIR-guided instruction scheduling: how the
compiler derives a machine-scheduling plan from final TTGIR and carries that
plan through lowering to the AMDGPU machine scheduler.

The user-facing capability is **TTGIR instruction scheduling**. The current AMD
backend realization uses `sched_group_barrier`, so the implementation pass is
deliberately named `SchedGroupBarrierScheduler`. Keeping the implementation
name mechanism-specific distinguishes it from Triton's existing dot
decomposition, modulo scheduling, and loop scheduling passes.

The scheduler is default-off and independent of any one kernel. The gfx950
shared-A BMM is the first tuned consumer and the validation case used for the
performance results below.

## Motivation

Useful scheduling decisions require information from two different compiler
levels:

* Final TTGIR still contains tensor layouts, MFMA encodings, vectorization,
  pointer alignment, and the relationship between memory operations and dots.
* The real LDS hazard boundaries are created later by
  `ModuleMembarAnalysis`. Those boundaries define the regions within which the
  AMDGPU machine scheduler is allowed to interleave instructions.

An LLVM IR scheduler sees the second part of this picture but has lost much of
the first. A scheduler that runs only at TTGIR sees the first part but cannot
place constraints at Membar boundaries that do not exist yet.

The design therefore separates **planning** from **materialization**:

```text
final TTGIR
  -> classify relevant operations
  -> predict lowering-aware machine-instruction multiplicities
  -> record the plan as TTGIR/module metadata
  -> run ModuleMembarAnalysis
  -> split blocks into the actual machine-scheduling regions
  -> materialize ordered scheduling groups at those boundaries
  -> run the LLVM AMDGPU machine scheduler
```

No LLVM IR instruction reordering is performed. TTGIR decides what should be
interleaved; AMD lowering places the constraints where the machine scheduler
can realize that decision.

## Goals and non-goals

The current design has four goals:

1. Make the scheduling decision while tensor and layout semantics are present.
2. Express the decision at the actual LDS hazard boundaries.
3. Predict machine-instruction counts from the same information used by
   lowering, rather than treating one TTGIR operation as one instruction.
4. Keep selection cache-keyed and per kernel so it can be autotuned.

The pass does not:

* change data dependencies or reorder TTGIR/LLVM IR operations;
* replace dot decomposition or modulo scheduling;
* infer an arbitrary kernel's best cover ratio in the current version;
* enable itself globally by default;
* claim a machine class whose lowering multiplicity is not modeled.

## Compiler placement

`TritonAMDGPUSchedGroupBarrierSchedulerPass` runs at the end of the AMD TTGIR
pipeline, after transformations that can change instruction classes or widths,
including buffer-op conversion, in-thread transpose, and late layout cleanup.
This placement is important: moving the pass earlier can price an operation
using a layout or access width that no longer exists in the final program.

The pass only attaches metadata. During AMD-to-LLVM conversion,
`ModuleMembarAnalysis` first creates the real `ttg.barrier` operations. The
conversion then calls `materializeDeferredSchedGroupBarriers` before lowering
the remaining Triton operations.

## Scheduling model

The model uses four machine-instruction classes:

| Symbol | Machine class | AMD scheduler mask |
| --- | --- | ---: |
| `M` | MFMA | `1 << 3` |
| `G` | VMEM read | `1 << 5` |
| `R` | DS read | `1 << 8` |
| `W` | DS write | `1 << 9` |

Memory instructions are **anchors**. Independent MFMA instructions are the
latency-hiding **cover** scheduled after each anchor. A region is represented
as an ordered sequence of anchor/cover pairs:

```text
(G128, M x 4), (G64, M x 2), (G32, M x 1),
(R, M x 1), ..., (W, M x write_cover)
```

All pairs in one region use the same scheduler sync ID. This makes their order
one machine-scheduling pipeline rather than a collection of unrelated hints.

## Phase 1: classify and price final TTGIR

Every relevant TTGIR operation receives:

* `ttg.amd.sched_group_barrier.machine_mask`: eventual AMD machine class;
* `ttg.amd.sched_group_barrier.machine_count`: predicted number of machine
  instructions produced by lowering;
* `ttg.amd.sched_group_barrier.mfma_cover`: MFMA cover for each VMEM machine
  instruction, derived from its final access width.

The module records that the plan is enabled and the optional required region
count. The attributes are a deferred plan; they are not scheduling
instructions and do not change program semantics.

### MFMA multiplicity

For a block-level dot, the predicted number of MFMAs is:

```text
ceil(M / (instrM * warpsM))
  * ceil(N / (instrN * warpsN))
  * ceil(logicalK / instrK)
```

`M` and `N` come from the result tensor. The instruction shape and warp
partition come from `AMDMfmaEncodingAttr`. `logicalK` comes from the A operand.
For E2M1 `tt.dot_scaled`, the stored K dimension counts bytes containing two
fp4 values, so the model doubles K before applying the formula.

This is a per-operation calculation. Matching only the block's total MFMA
count is insufficient: two incorrectly priced dots can cancel in the total
while producing unfillable groups at their actual positions.

### DS-read multiplicity

The model obtains the exact elements held by each lane from the operation's
`LinearLayout`, then divides the per-lane bytes by the access width selected by
lowering:

```text
ds_read_count = ceil(bytes_per_lane / ds_access_bytes)
```

The common path uses 16-byte `ds_read_b128`. On gfx950, MFMA B operands use the
transposed 8-byte DS-read path, so those operations are priced at twice the
instruction count of a 16-byte access with the same lane payload.

Using `getTotalElemsPerThread` is intentional. A dot-operand layout can be
replicated across a warp dimension it does not span; byte arithmetic based
only on the logical tensor shape misses that replication.

### DS-write multiplicity

DS-store width is derived by mirroring local-store lowering:

1. Convert the source register type and destination memdesc to linear layouts.
2. Compose the register-to-LDS conversion, including padded encodings.
3. Remove broadcast register dimensions.
4. Ask `largestVectorisation` for the legal store width.
5. Divide the register dimension by that width.

A large logical store can therefore lower to scalar DS writes when padding
breaks physical contiguity. Assuming every store becomes `ds_write_b128`
under-counts this case.

### VMEM-read multiplicity

For buffer loads, the candidate vector width is computed from:

* AxisInfo offset contiguity;
* base-pointer divisibility in elements;
* contiguous elements per thread in the final layout;
* the 128-bit maximum buffer-load width;
* non-power-of-two and blocked-layout clamping.

The contiguity already selected by buffer-op conversion is then retained as an
input to the final width.

The machine count is the per-thread element count divided by the resulting
vector width. Direct-to-LDS and remaining load forms use their final tensor or
memdesc layout to derive an equivalent count.

The same final width selects a small cover pattern. A 128-bit
`buffer_load_dwordx4` receives four MFMAs, a 64-bit
`buffer_load_dwordx2` receives two, and a 32-bit `buffer_load_dword` receives
one. Narrower reads conservatively receive one. In formula form, for the
default calibration:

```text
vmem_cover = max(1, ceil(mfma_per_dwordx4 * min(access_bytes, 16) / 16))
```

This width-aware rule does not invoke Triton's general modulo scheduler. The
128-bit case is the measured performance anchor. Scaling down prevents a
narrower lowering, which already contains more VMEM instructions for the same
payload, from receiving four MFMAs after every instruction.

### Why multiplicity accuracy matters

`sched_group_barrier` counts machine instructions, not IR operations.
Under-counting leaves matching instructions outside the requested pipeline,
where they can form a long cluster. Over-counting asks the backend to fill a
group that does not exist. Both change the realized ISA schedule.

The implementation does not emit a catch-all group. A broad mask can also
match scheduling pseudo-instructions and invalidate the ordered program. New
machine classes must therefore provide a lowering-aware count model before
they are admitted.

## Phase 2: discover real scheduling regions

After `ModuleMembarAnalysis`, each block is split at every `ttg.barrier`:

```text
region 0 | barrier | region 1 | barrier | ... | region N
```

The block is eligible when:

* it contains at least one predicted MFMA and one predicted memory anchor; and
* `required_region_count` is zero, or the block has exactly the requested
  number of regions.

The region-count policy is a generic topology selector, not a kernel-shape
check. It lets an autotuning configuration select a validated steady-state
structure while declining a structurally different tail block. The shared-A
BMM selects four regions; the compiler contains no BMM dimensions.

If a block is accepted, hard `sched_barrier(0)` fences are placed at its start,
between Membar-delimited regions, and at its end. These fences prevent the
machine scheduler from moving an instruction into a neighboring region.

## Phase 3: build an anchor/cover program

For every accepted region, the algorithm walks annotated operations in their
original TTGIR order and computes:

```text
mfmas       = sum(machine_count for M operations)
vmem_reads  = sum(machine_count for G operations)
ds_reads    = sum(machine_count for R operations)
ds_writes   = sum(machine_count for W operations)

fixed_cover = sum(machine_count * mfma_cover for G operations) + ds_reads

write_cover = 1
if ds_writes > 0 and mfmas > fixed_cover:
  write_cover = max(1, floor((mfmas - fixed_cover) / ds_writes))
```

The cover assigned to each anchor instance is:

| Anchor | MFMA cover |
| --- | ---: |
| 128-bit VMEM read (`G`) | 4 |
| 64-bit VMEM read (`G`) | 2 |
| 32-bit or narrower VMEM read (`G`) | 1 |
| DS read (`R`) | 1 |
| DS write (`W`) | `write_cover` |

The algorithm expands each operation by its predicted machine count. For each
anchor instruction it emits one anchor group followed by its MFMA cover group.
If a region has no MFMA work, it emits only the ordered memory groups.

A memory-only prologue has one conservative special case: the first VMEM
operation may move after the first DS-read operation. The move is performed at
whole-operation granularity; the internal machine count of either operation is
not split.

In pseudocode, materialization is:

```text
for block in function:
  regions = split_at_membar_boundaries(block)
  if not eligible(block, regions):
    continue

  emit_hard_fence(block.begin)
  for region in regions:
    anchors, mfmas = collect_in_ttgir_order(region)
    groups = allocate_cover(anchors, mfmas)
    sync_id = fresh_sync_id()
    for anchor, cover in groups:
      emit_group(anchor.mask, count=1, sync_id)
      if cover != 0:
        emit_group(MFMA, count=cover, sync_id)
    emit_hard_fence(region.end)
```

The materialized operations lower through ROCDL to the AMDGPU scheduler. The
final ISA order, not the presence of metadata or ROCDL operations, is the
source of truth for whether the plan was realized.

## Selection and autotuning contract

The scheduler is default-off. A kernel or autotuning configuration enables it
with cache-keyed HIPOptions:

```python
enable_sched_group_barrier_scheduler = True
sched_group_barrier_mfma_per_dwordx4 = 4
sched_group_barrier_required_region_count = 0
```

For compiler experiments, the environment can provide the default enablement:

```bash
TRITON_AMD_TTGIR_SCHEDULE=1
```

An explicit HIPOption takes precedence over the environment. The resolved
value and both tuning parameters are stored in `HIPOptions`, so they
participate in the compilation cache key. An autotuner can therefore compare
enabled/disabled variants, cover ratios, and region policies without reusing a
binary compiled for another schedule.

AMDGPU's generic high-register-pressure strategy can replace an otherwise
valid pre-RA schedule. A pressure-sensitive consumer may also select the
cache-keyed `disable_unclustered_high_rp_reschedule` option. This is not part of
the anchor/cover algorithm; it prevents a later policy from discarding an
accepted plan. Spill size and occupancy must still be validated.

## Performance result

The gfx950 shared-A BMM uses a 256x256 MI16 tile and the four-region policy. For
`B=1024, M=1195, N=256, K=2309`, an eight-round paired warm-cache benchmark
alternated the LLVM-IR and TTGIR variants and bracketed them with unscheduled
measurements to reduce clock and temperature bias.

| Scheduling path | Median latency | vs. no scheduling | vs. best LLVM-IR |
| --- | ---: | ---: | ---: |
| No scheduling (before) | 2249 us | - | - |
| Best LLVM-IR scheduler evaluated | 2070 us | -8.0% | - |
| **TTGIR scheduling (after)** | **1972 us** | **-12.2%** | **-4.7%** |

TTGIR scheduling outperformed every LLVM-IR scheduling configuration evaluated
for this kernel. All three measured variants were bitwise exact and used zero
scratch.

The realized TTGIR-scheduled hot loop contains:

```text
MFMA=128, DS_READ=48, DS_WRITE=8, VMEM_READ=16, hard barriers=3
```

Its class stream starts with `R8 G4 R16`, then interleaves the memory anchors
with MFMA cover groups. The maximum consecutive MFMA run is 10. The final
object uses a zero-byte private segment and 510 VGPRs.

## Validation procedure

A scheduling change is not validated by counting IR hints. Validate the final
object and runtime behavior:

1. Compare every predicted per-operation machine count with disassembly and
   source locations.
2. Compare hot-loop totals and the requested/realized class stream.
3. Check hard-fence placement at every expected region boundary.
4. Check private-segment size, VGPR allocation, and occupancy.
5. Run fixed-seed numerical correctness, including partial and unaligned K.
6. Benchmark enabled and disabled variants in the same process with alternating
   order and bracket samples.
7. Run the AMD TLX regression suite to catch unrelated layout regressions.

For the current stack:

* `test_tlx_amd.py`: 91 passed, 5 skipped;
* focused option, validation, and environment-precedence tests: 4 passed;
* target BMM: bitwise exact against `torch.bmm`, `max_abs=0`;
* Buck targets and the local `libtriton.so` build succeed.

## Current limitations and extension points

The region mechanism is general, but the current cost model covers only MFMA,
VMEM reads, DS reads, and DS writes. It uses a configurable fixed VMEM cover,
a one-MFMA DS-read cover, and an even distribution of residual MFMA work over
DS writes. It does not yet derive these ratios from a device throughput model.

The `required_region_count` gate selects a complete block topology. The current
implementation does not independently accept or reject each region with a
profitability model. Extending admission should preserve the existing
all-or-nothing block boundary contract unless cross-region motion is proven
safe.

Adding a new machine class requires:

1. a stable mapping from final TTGIR to the backend machine class;
2. a lowering-aware per-operation multiplicity model;
3. an anchor or cover policy;
4. disassembly-based validation across the affected layouts and access widths.

Future autotuning can search enablement, VMEM cover, and region topology first.
A later profitability model can replace the fixed cover policy without
changing the TTGIR-planning/post-Membar-materialization architecture.
