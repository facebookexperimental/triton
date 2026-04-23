# [triton] Support 2-CTA in Triton with ctas_per_cga

**Task:** T259718487
**Base Diff:** D96323995 (`[TLX] Add two_cta support to async_descriptor_load`)
**Reference:** D97215251 (Draft OSS PR #1059: compiler automation with `Transform2CTALoads`)
**Author:** Zhijing Li (tissue030)

---

## Table of Contents

1. [Background](#background)
2. [Problem Statement](#problem-statement)
3. [Design](#design)
4. [Implementation](#implementation)
5. [Generated TTGIR Example](#generated-ttgir-example)
6. [Files Changed](#files-changed)
7. [How to Test](#how-to-test)
8. [Known Limitations and Future Work](#known-limitations-and-future-work)
9. [Auto-WS + 2-CTA Integration](#auto-ws--2-cta-integration)

---

## Background

### What is 2-CTA MMA?

On NVIDIA Blackwell (SM100+) GPUs, two CTAs (Cooperative Thread Arrays) in a
cluster can cooperatively execute a single MMA (Matrix Multiply-Accumulate)
instruction via `tcgen05.mma.cta_group::2`. This effectively doubles the tile
size (up to 256x256) and improves tensor core utilization.

In 2-CTA mode:
- **CTA 0** (leader, even-ranked) loads **A** and **half of B** (`B[:, :N/2]`)
- **CTA 1** (odd-ranked) loads the **other half of B** (`B[:, N/2:]`)
- Both CTAs issue `tcgen05.mma.cta_group::2`, which reads A from CTA 0 and B
  from both CTAs' SMEM
- The hardware synchronizes execution across both CTAs

### Two Approaches to 2-CTA

| Approach | `num_ctas` | CTA Layout | B Splitting | Who manages sync? |
|----------|-----------|------------|-------------|-------------------|
| **PlanCTA** (compiler-driven) | 2 | `CTASplitNum=[2,1]` | `splitBOperand()` in AccelerateMatmul | Compiler via CTA layout encoding |
| **ctas_per_cga** (user-driven) | 1 | `CTASplitNum=[1,1]` | `Transform2CTALoads` pass | Compiler via explicit IR transforms |

The reviewer (@manman-ren) directed us to use the **ctas_per_cga approach** because
PlanCTA is unreliable for 2-CTA. With `ctas_per_cga=(2,1,1)`, the user sets the
cluster shape and `num_ctas=1` is set automatically — bypassing PlanCTA entirely.

### Related Diffs

| Diff | Description |
|------|-------------|
| **D96323995** (base) | Adds `two_cta` to `AsyncTMACopyGlobalToLocalOp`; emits `.cta_group::2` on TMA PTX. Demonstrates the `remote_view` pattern for cross-CTA barrier mapping in TLX. |
| **D97215251** (reference) | Draft OSS PR #1059: full compiler automation with 3 passes (`Propagate2CTAAttr`, `Transform2CTALoads`, `Insert2CTASync`). We port `Transform2CTALoads` and reuse our existing `Insert2CTASync`. `Propagate2CTAAttr` is NOT needed since AccelerateMatmul already propagates the attribute. |

### Reference Implementations

- **`fbcode/generative_recommenders/ops/triton/triton_addmm.py`** -- Production
  TLX kernel with manual 2-CTA + WS ("arrive remote, wait local" pattern)
- **`third_party/tlx/tutorials/blackwell_gemm_2cta.py`** -- TLX 2-CTA GEMM tutorial

---

## Problem Statement

When `tl.dot(two_ctas=True)` is used with `ctas_per_cga=(2,1,1)` on Blackwell:
1. The user writes standard Triton code loading **full B** (`BLOCK_K × BLOCK_N`)
2. The compiler must split B loads so each CTA loads only half (`BLOCK_K × BLOCK_N/2`)
3. After pipelining, cross-CTA sync must be inserted before MMA
4. TLX kernels (which also use `ctas_per_cga`) must not be affected

With `ctas_per_cga=(2,1,1)`, `num_ctas=1` is set automatically. This means:
- PlanCTA sees `num_ctas=1` and does nothing (no CTA split)
- AccelerateMatmul's `canUseTwoCTAs()` returns false (`CTASplitNum=[1,1]`)
- `splitBOperand()` cannot work (requires `CTASplitNum=[1,2]` from PlanCTA)

Therefore, a new `Transform2CTALoads` pass is needed to physically transform
B descriptor loads at the IR level.

---

## Design

### User API

The user writes standard Triton code with two additions:
1. `tl.dot(a, b, acc, two_ctas=True)` — opt in to 2-CTA MMA
2. `ctas_per_cga=(2, 1, 1)` — set cluster shape at kernel launch

```python
@triton.jit
def matmul_2cta(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # Standard TMA descriptor loads (full B)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], ..., [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], ..., [BLOCK_K, BLOCK_N])

    for k in range(k_tiles):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])  # Full B — compiler splits this
        acc = tl.dot(a, b, acc, two_ctas=True)

# Launch with ctas_per_cga — bypasses PlanCTA
matmul_2cta[grid](..., ctas_per_cga=(2, 1, 1))
```

The `two_ctas` attribute flows through:

```
tl.dot(a, b, acc, two_ctas=True)    # Python (core.py)
  -> semantic.dot(..., two_ctas=True)  # semantic.py
    -> builder.create_dot(..., true)   # ir.cc
      -> tt.dot ... {two_ctas}         # TTIR DotOp (TritonOps.td)
```

### Pipeline

```
TTIR: tt.dot {two_ctas}
  |
convert_to_ttgpuir            <-- preserves {two_ctas} on DotOp
  |
AccelerateMatmul               <-- getTwoCtas()=true: skips splitBOperand,
  |                                sets two_ctas on TCGen5MMAOp
Transform2CTALoads             <-- NEW: splits B descriptor loads per CTA
  |
schedule_loops + Meta WS       <-- 4 partitions: default/gemm/load/epilogue
  |
pipeline + hoist_tmem_alloc
  |
Insert2CTASync                 <-- "arrive remote, wait local" cross-CTA sync
  |                                (AFTER all WS passes to avoid interference;
  |                                 skips TLX kernels via tlx.has_tlx_ops check)
tma_lowering
```

### AccelerateMatmul: User-Driven 2-CTA Path

When `dotOp.getTwoCtas()` is true, AccelerateMatmul takes the user-driven path:

```cpp
if (dotOp.getTwoCtas()) {
    // User-driven 2-CTA (ctas_per_cga): no splitBOperand needed.
    // Transform2CTALoads will handle B splitting later.
    useTwoCTAs = true;
} else {
    // Compiler-driven 2-CTA (PlanCTA heuristic): split B at IR level.
    useTwoCTAs = canUseTwoCTAs(dotOp);
    if (useTwoCTAs) {
        b = splitBOperand(b, rewriter);
    }
}
// ...
mma.setTwoCtas(useTwoCTAs);  // Propagates to TCGen5MMAOp
```

### Transform2CTALoads Pass

This pass runs before pipelining/WS and transforms B descriptor loads for non-async
(non-TLX) 2-CTA MMAs. For each `TCGen5MMAOp` with `two_ctas=true && !is_async`:

1. **Trace B operand** back through `LocalAllocOp` → `DescriptorLoadOp` → `MakeTensorDescOp`
2. **Clone `MakeTensorDescOp`** with half-width block shape: `[BLOCK_K, BLOCK_N]` → `[BLOCK_K, BLOCK_N/2]`
3. **Insert CTA offset computation**:
   ```
   cta_rank = nvgpu.cluster_id
   cta_mod2 = cta_rank % 2
   offset = cta_mod2 * (BLOCK_N / 2)
   new_offs_n = offs_bn + offset
   ```
4. **Create new `DescriptorLoadOp`** with half-width result type and new descriptor
5. **Create new `LocalAllocOp`** with half-width SMEM memdesc

After this transform:
- CTA 0 loads `B[:, offs_bn : offs_bn + BLOCK_N/2]`
- CTA 1 loads `B[:, offs_bn + BLOCK_N/2 : offs_bn + BLOCK_N]`

### Insert2CTASync: "Arrive Remote, Wait Local" Pattern

Before each 2-CTA MMA, the `Insert2CTASync` pass inserts:

1. `nvgpu.cluster_id` -- get CTA rank
2. `arith.andi %rank, -2` -- compute leader rank (even CTA)
3. `ttng.map_to_remote_buffer` -- map local barrier to leader CTA's SMEM
4. `ttng.arrive_barrier` -- both CTAs arrive on leader's barrier (count=1 each)
5. `ttng.wait_barrier` -- only leader waits (pred = rank % 2 == 0)
6. `ttng.tc_gen5_mma {two_ctas}` -- 2-CTA MMA executes

**TLX guard:** The pass checks for `tlx.has_tlx_ops` module attribute and
returns early for TLX kernels. This is necessary because TLX kernels with
`ctas_per_cga=(2,1,1)` manage their own cross-CTA sync via explicit barrier ops.
Running `Insert2CTASync` on warp-specialized TLX IR causes barrier placement
errors (`"Barrier init outside of the first block in function"`).

---

## Implementation

### Changes Summary

| Component | File | Change |
|-----------|------|--------|
| **Python API** | `core.py` | Add `two_ctas=False` param to `tl.dot()` |
| **Semantic** | `semantic.py` | Thread `two_ctas` through `dot()` to `create_dot()` |
| **IR binding** | `ir.cc` | Add `bool twoCtas` to `create_dot`, set attr on DotOp |
| **DotOp IR** | `TritonOps.td` | Add `UnitAttr:$two_ctas` to DotOp arguments |
| **TTIR→TTGIR** | `TritonToTritonGPUPass.cpp` | Preserve `two_ctas` during DotOp conversion |
| **AccelerateMatmul** | `AccelerateMatmul.cpp` | User-driven path: skip `splitBOperand` when `getTwoCtas()`, propagate `two_ctas` to MMA |
| **Transform loads** | `Transform2CTALoads.cpp` | **NEW**: Split B descriptor loads per CTA (half-width) |
| **Sync pass** | `Insert2CTASync.cpp` | Insert cross-CTA sync; skip TLX via `tlx.has_tlx_ops` |
| **Pass def** | `Passes.td` | Add `NVGPU2CTATransformLoads` + `NVGPUInsert2CTASync` pass definitions |
| **Build** | `CMakeLists.txt` | Add `Transform2CTALoads.cpp`, `Insert2CTASync.cpp` |
| **Pass wrapper** | `triton_nvidia.cc` | Add `add_2cta_transform_loads`, `add_insert_2cta_sync` wrappers |
| **Pipeline** | `compiler.py` | `Transform2CTALoads` for `ctas_per_cga`, `Insert2CTASync` for all cluster (with TLX guard); for Meta WS, `Insert2CTASync` runs after all WS passes |

### Key Design Decisions

1. **`tl.dot(two_ctas=True)` as explicit opt-in**: Prevents backward compat
   issues. Without this, any kernel with `cluster_dims >= 2` + `tl.dot()` would
   silently get 2-CTA behavior.

2. **`ctas_per_cga` bypass of PlanCTA**: Following the reviewer's direction, we
   use `ctas_per_cga=(2,1,1)` which sets `num_ctas=1`, bypassing PlanCTA entirely.
   `Transform2CTALoads` handles B splitting at the IR level instead of relying on
   CTA layout encoding (`CTASplitNum`).

3. **`Transform2CTALoads` from D97215251**: The pass is ported from D97215251's
   design but written fresh following `Insert2CTASync`'s pattern. The key
   transform: clone `MakeTensorDescOp` with half block shape, add CTA-based offset.
   `Propagate2CTAAttr` from D97215251 is NOT needed — AccelerateMatmul already
   propagates `two_ctas` from `DotOp` to `TCGen5MMAOp`.

4. **`Insert2CTASync` with `tlx.has_tlx_ops` guard**: The pass runs for all
   cluster kernels but checks the `tlx.has_tlx_ops` module attribute and returns
   early for TLX kernels. This is needed because both TLX and pure Triton kernels
   use `ctas_per_cga=(2,1,1)`, but TLX manages its own barriers. The per-MMA
   `is_async` check alone was insufficient — warp-specialized TLX IR causes
   barrier placement errors even when the pass finds no ops to transform.

5. **`TritonToTritonGPUPass.cpp` fix**: The `TritonDotPattern` was dropping
   `two_ctas` when recreating `DotOp` during TTIR→TTGIR conversion. Fixed by
   explicitly copying the attribute.

---

## Files Changed

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `hopper/lib/Transforms/Transform2CTALoads.cpp` | ~200 | B descriptor load splitting pass |
| `hopper/lib/Transforms/Insert2CTASync.cpp` | ~200 | Cross-CTA sync pass |
| `test/Hopper/TwoCTA/transform_2cta_loads.mlir` | ~120 | MLIR lit test for load splitting |
| `test/Hopper/TwoCTA/insert_2cta_sync.mlir` | ~125 | MLIR lit test for sync pass |
| `third_party/tlx/tutorials/blackwell-triton-addmm-2cta_test.py` | ~490 | Pure Triton E2E test + auto-WS 2CTA + perf benchmark |

### Modified Files

| File | Change |
|------|--------|
| `include/triton/Dialect/Triton/IR/TritonOps.td` | `UnitAttr:$two_ctas` on DotOp |
| `python/triton/language/core.py` | `two_ctas=False` param on `tl.dot()` |
| `python/triton/language/semantic.py` | Thread `two_ctas` through `dot()` |
| `python/src/ir.cc` | `bool twoCtas` on `create_dot` |
| `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | Preserve `two_ctas` during conversion |
| `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` | User-driven 2-CTA path |
| `hopper/include/Transforms/Passes.td` | `NVGPU2CTATransformLoads` + `NVGPUInsert2CTASync` definitions |
| `hopper/lib/Transforms/CMakeLists.txt` | Add `Transform2CTALoads.cpp`, `Insert2CTASync.cpp` |
| `third_party/nvidia/triton_nvidia.cc` | Add pass wrappers |
| `third_party/nvidia/backend/compiler.py` | Pipeline integration with TLX guard; Meta WS: `Insert2CTASync` after all WS passes |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp` | Skip cluster barrier for WS; cluster0 predicate on `TCGen5CommitOp` lowering |
| `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp` | Propagate `two_ctas` from MMA to `TCGen5CommitOp` (2 locations) |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp` | Exit cluster barrier for auto-WS with cluster-dim-x >= 2 |
| `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp` | TMEM `cta_group::2` when cluster-dim-x >= 2 (not just num-ctas == 2) |
| `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp` | Remove `doInsert2CTASync` call (now handled by separate pass in compiler.py) |

---

## How to Test

### Step 1: Build the Triton Compiler

```bash
buck2 build @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  //third-party/triton/beta/triton:triton-opt
```

### Step 2: Run MLIR Lit Tests (no GPU needed)

```bash
buck2 test @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  '//third-party/triton/beta/triton:lit_test/Hopper/TwoCTA/transform_2cta_loads.mlir' \
  '//third-party/triton/beta/triton:lit_test/Hopper/TwoCTA/insert_2cta_sync.mlir'
```

Expected: `Tests finished: Pass 2. Fail 0.`

### Step 3: Run TLX Regression Test (requires Blackwell GPU)

```bash
buck2 test @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  '//third-party/triton/beta/triton:py_tlx_blackwell_test' \
  -- test_async_dot_blackwell_2cta_tma
```

Expected: `Tests finished: Pass 2. Fail 0.`

### Step 4: Run Pure Triton E2E Test (requires Blackwell GPU)

```bash
buck2 test @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  '//third-party/triton/beta/triton:py_tlx_blackwell_triton-addmm-2cta'
```

Expected: `Tests finished: Pass 11. Fail 0.` (non-WS 2CTA correctness + compilation, auto-WS 2CTA correctness + compilation, perf benchmark)

### Step 5: Run Auto-WS + 2CTA Correctness Only (requires Blackwell GPU)

```bash
buck2 test @fbcode//mode/opt \
  '//third-party/triton/beta/triton:py_tlx_blackwell_triton-addmm-2cta' \
  -- test_matmul_2cta_ws_correctness
```

Expected: `Tests finished: Pass 4. Fail 0.` (128-128-128, 256-256-64, 128-256-128, 256-256-128)

### Debugging Tips

To see the TTGIR output at each pass stage:

```bash
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python your_kernel.py 2>&1 | grep -A1000 "make_ttgir"
```

To run Transform2CTALoads in isolation on an MLIR file:

```bash
buck2 run @fbcode//mode/opt -m ovr_config//triton:beta \
  //third-party/triton/beta/triton:triton-opt -- \
  --nvgpu-2cta-transform-loads your_input.mlir
```

---

## Known Limitations and Future Work

1. **Single barrier for phase tracking**: Uses `phase = (iv - lb) % 2` with one
   barrier. Works for pipeline depth <= 2; may need multi-buffered barriers for
   deeper pipelines.

2. **Multiple MMAs per loop iteration**: Shared single barrier; could cause phase
   conflicts if they overlap. Future: allocate separate barriers per MMA.

3. **TCGen5MMAScaledOp**: Only `TCGen5MMAOp` is handled; the scaled variant is
   not yet supported.

4. **B from function arguments**: `Transform2CTALoads` requires B's descriptor to
   come from `MakeTensorDescOp` (not a function argument). This covers all real
   kernels where `make_tensor_descriptor` is called in the kernel body.

5. **Encoding compatibility**: When halving the N dimension, the blocked encoding
   on the descriptor load result may need adjustment. The pass computes a
   compatible encoding by reducing `threadsPerWarp` in the N dimension.

6. **D96323995's `.cta_group::2` on TMA loads**: D96323995 added `.cta_group::2`
   support for TMA loads, which routes barrier arrivals to the leader CTA in
   hardware. A future optimization could use this on B-operand TMA loads,
   potentially eliminating the explicit cross-CTA sync.

---

## Auto-WS + 2-CTA Integration

### Overview

When `tl.dot(two_ctas=True)` is combined with `warp_specialize=True` and
`TRITON_USE_META_WS=1`, the auto-WS framework (Meta's `WSCodePartition` +
`WSLowerMem`) handles the warp-specialized pipeline while the 2-CTA passes
handle cross-CTA B-splitting and synchronization.

### Pipeline Order (compiler.py)

```
AccelerateMatmul        ← sets two_ctas on TCGen5MMAOp
Transform2CTALoads      ← splits B descriptor loads per CTA
schedule_loops + WS     ← Meta WS partitioning (4 partitions: default/gemm/load/epilogue)
pipeline                ← software pipelining of the K-loop
hoist_tmem_alloc        ← hoists TMEM alloc before WarpSpecializeOp
Insert2CTASync          ← cross-CTA sync (AFTER all WS passes to avoid interference)
```

`Insert2CTASync` runs after all WS-related passes to avoid scheduling/pipeline
interference. The barrier ops won't be reordered or erased by subsequent WS passes.
`getThreadId()` returns relative IDs inside `WarpSpecializeOp` partition regions,
so `InitBarrierOp` and `ArriveBarrierOp` work correctly in the consumer warp group.

### Bugs Fixed for Auto-WS + 2-CTA

**1. Deadlock: Cluster barrier inside WS MMA loop (`MMAv5.cpp`)**

The PTX lowering inserted `ClusterArriveOp` + `ClusterWaitOp` (cluster-wide
`barrier.cluster.arrive/wait`) inside each MMA loop iteration for non-TLX
`cta_group::2` kernels. In WS mode, only the consumer warp group executes this,
but `barrier.cluster` requires ALL threads in the CTA → deadlock.

Fix: Skip `ClusterArriveOp`/`ClusterWaitOp` when inside a `WarpSpecializeOp`
(detected via `getParentOfType<WarpSpecializeOp>()`). The `cluster0` MMA predicate
(`cluster_cta_id & 1 == 0`) is kept unconditional for all `twoCTAs` cases.

**2. Correctness: `TCGen5CommitOp` wrong `cta_group` (`WSCodePartition.cpp`)**

The WS framework's `TCGen5CommitOp` (gemm→epilogue completion barrier) was created
without the `two_ctas` attribute, producing `tcgen05.commit.cta_group::1` in PTX.
But the MMA uses `cta_group::2`. A `cta_group::1` commit does NOT wait for
`cta_group::2` MMA operations, so the epilogue could read TMEM before the MMA
finished writing.

Fix: Propagate `two_ctas` from the MMA op when creating `TCGen5CommitOp`:
```cpp
bool twoCTAs = cast<ttng::TCGen5MMAOp>(mmaOp).getTwoCtas();
builder.createWithAsyncTaskIds<ttng::TCGen5CommitOp>(
    mmaOp->getLoc(), barrier, twoCTAs);
```

**3. Correctness: `TCGen5CommitOp` missing cluster0 predicate (`MMAv5.cpp`)**

After fixing Bug 2, both CTAs issued the `cta_group::2.multicast::cluster`
commit, causing the barrier to get double-arrived (2 arrives instead of 1),
advancing the barrier phase too fast and desynchronizing the pipeline.

Fix: Add `cluster0` predicate to `TCGen5CommitOpConversion` when `twoCTAs=true`:
```cpp
if (twoCTAs) {
    Value clusterCTARank = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
    Value isLeader =
        b.icmp_eq(b.and_(clusterCTARank, b.i32_val(1)), b.i32_val(0));
    pred = b.and_(pred, isLeader);
}
```

**4. TMEM `cta_group::2` for auto-WS (`NVGPUToLLVMPass.cpp`)**

TMEM alloc/dealloc used `cta_group::1` for auto-WS 2CTA because `num-ctas=1`.
Fix: Check `cluster-dim-x >= 2` in addition to `num-ctas`.

**5. Exit cluster barrier for auto-WS (`ConvertWarpSpecializeToLLVM.cpp`)**

The exit cluster barrier (before `return`) was only emitted for TLX kernels.
Fix: Also emit for auto-WS kernels when `cluster-dim-x >= 2`.

### Additional Files Changed for Auto-WS + 2-CTA

| File | Change |
|------|--------|
| `MMAv5.cpp` | Skip cluster barrier for WS; cluster0 predicate on `TCGen5CommitOp` |
| `WSCodePartition.cpp` | Propagate `two_ctas` to `TCGen5CommitOp` (2 locations) |
| `ConvertWarpSpecializeToLLVM.cpp` | Exit cluster barrier for cluster-dim-x >= 2 |
| `NVGPUToLLVMPass.cpp` | TMEM `cta_group::2` for cluster-dim-x >= 2 |
| `compiler.py` | `Insert2CTASync` after all WS passes for Meta WS |
| `WarpSpecialization.cpp` | Remove `doInsert2CTASync` call (handled by separate pass) |

### Test Commands

```bash
# Auto-WS + 2CTA correctness (all 4 sizes)
buck2 test @fbcode//mode/opt //third-party/triton/beta/triton:py_tlx_blackwell_triton-addmm-2cta -- test_matmul_2cta_ws_correctness

# Non-WS 2CTA regression
buck2 test @fbcode//mode/opt //third-party/triton/beta/triton:py_tlx_blackwell_triton-addmm-2cta -- test_matmul_2cta_correctness

# TLX WS+2CTA regression
buck2 test @fbcode//mode/opt //third-party/triton/beta/triton:py_tlx_blackwell_triton-addmm-2cta -- test_tlx_ws_2cta_correctness
```
