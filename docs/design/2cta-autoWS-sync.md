# [Phase2][autoWS] Correctly Support 2-CTA in Triton

**Task:** T259718487
**Base Diff:** D96323995 (`[TLX] Add two_cta support to async_descriptor_load`)
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

### What is autoWS (Auto Warp Specialization)?

Triton's autoWS compiler passes partition a kernel into producer/consumer warps:
- **Producer warps**: Execute TMA loads, signal barriers when data is ready
- **Consumer warps**: Wait on barriers, execute MMA operations

### Related Diffs

| Diff | Description |
|------|-------------|
| **D96323995** (base) | Adds `two_cta` to `AsyncTMACopyGlobalToLocalOp`; emits `.cta_group::2` on TMA PTX |
| **D97215251** (reference) | Draft OSS PR #1059: full compiler automation with 3 passes (skips WS) |

### Reference Implementations

- **`fbcode/generative_recommenders/ops/triton/triton_addmm.py`** -- Production
  TLX kernel with manual 2-CTA + WS ("arrive remote, wait local" pattern)
- **`third_party/tlx/tutorials/blackwell_gemm_2cta.py`** -- TLX 2-CTA GEMM tutorial

---

## Problem Statement

When `tl.dot(two_ctas=True)` is used with `cluster_dims=(2,1,1)` on Blackwell:
1. `PlanCTA` must assign M-split `CTASplitNum={2,1}` (not the default N-split)
2. `AccelerateMatmul` must detect 2-CTA eligibility, split B, set `two_ctas` on MMA
3. After pipelining, cross-CTA sync must be inserted before MMA

The task T259718487 states:
> "slicing of operand B probably needs to happen before memory planner, the
> extra synchronization due to 2cta can be after."

---

## Design

### User API: `tl.dot(two_ctas=True)`

The user explicitly opts in to 2-CTA MMA by passing `two_ctas=True` to `tl.dot()`.
This attribute flows through:

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
PlanCTA                        <-- sees {two_ctas}, forces CTASplitNum={2,1} (M-split)
  |
AccelerateMatmul               <-- canUseTwoCTAs() returns true, splits B, sets two_ctas on MMA
  |
pipeline/WS passes
  |
Insert2CTASync                 <-- inserts "arrive remote, wait local" cross-CTA sync
  |
tma_lowering
```

### PlanCTA Fix

`PlanCTA`'s `getCTATiling` heuristic prefers N-split for typical matmul shapes.
When `two_ctas` is set on the `DotOp`, we override the heuristic:

```cpp
if (dot.getTwoCtas()) {
    splitM = 2;  // M-split for 2-CTA MMA
    splitN = 1;
} else {
    std::tie(splitM, splitN) = getCTATiling(M, N, K, numCTAs);
}
```

### The "Arrive Remote, Wait Local" Pattern

Before each 2-CTA MMA, the `Insert2CTASync` pass inserts:

1. `nvgpu.cluster_id` -- get CTA rank
2. `arith.andi %rank, -2` -- compute leader rank (even CTA)
3. `ttng.map_to_remote_buffer` -- map local barrier to leader CTA's SMEM
4. `ttng.arrive_barrier` -- both CTAs arrive on leader's barrier (count=1 each)
5. `ttng.wait_barrier` -- only leader waits (pred = rank % 2 == 0)
6. `ttng.tc_gen5_mma {two_ctas}` -- 2-CTA MMA executes

---

## Generated TTGIR Example

Running `triton_addmm_2cta.py` with `tl.dot(two_ctas=True)`, `num_ctas=2`,
`cluster_dims=(2,1,1)` produces the following TTGIR. The full output is available
in paste [P2244242392](https://www.internalfb.com/intern/paste/P2244242392/).

Key encodings (from the TTGIR header):

```mlir
// Output + A: M-split {2,1} across 2 CTAs
#blocked = #ttg.blocked<{..., CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], ...}>
#shared  = #ttg.nvmma_shared<{..., CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], ...}>

// B: N-split {1,2} via splitBOperand() -- each CTA loads half of N
#shared1 = #ttg.nvmma_shared<{..., CTAsPerCGA = [1, 2], CTASplitNum = [1, 2], ...}>

// TMEM accumulator: M-split, blockM=64 per CTA (128/2)
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, CTASplitM = 2>
```

The generated function body (annotated):

```mlir
tt.func @test_two_ctas(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
    // --- Data loading ---
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<f16>, #blocked1>                          // Load A
    %3 = ttg.local_alloc %2 : ... -> !ttg.memdesc<128x64xf16, #shared, #smem>         // A -> SMEM
    %5 = tt.load %4 : tensor<64x128x!tt.ptr<f16>, #blocked3>                          // Load B
    %6 = ttg.local_alloc %5 : ... -> !ttg.memdesc<64x128xf16, #shared1, #smem>        // B -> SMEM (N-split)

    // --- TMEM accumulator allocation ---
    %result, %token = ttng.tmem_alloc %7 : ... -> (!ttg.memdesc<128x128xf32, #tmem, ...>, !ttg.async.token)

    // --- Cross-CTA barrier allocation (arrive_count=2) ---
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared2, #smem, mutable>
    %9 = ttg.memdesc_index %8[%c0] : ... -> !ttg.memdesc<1xi64, ...>
    ttng.init_barrier %9, 2 : ...                                                      // 2 arrivals expected

    // --- "Arrive Remote, Wait Local" cross-CTA sync ---
    %11 = nvgpu.cluster_id                                                             // CTA rank
    %c-2_i32 = arith.constant -2 : i32
    %12 = arith.andi %11, %c-2_i32 : i32                                              // leader = rank & ~1
    %13 = ttng.map_to_remote_buffer %10, %12 : ... -> ...#ttng.shared_cluster_memory   // map to leader
    ttng.arrive_barrier %13, 1 : ...                                                   // both CTAs arrive
    %14 = arith.remui %11, %c2_i32 : i32
    %15 = arith.cmpi eq, %14, %c0_i32 : i32                                           // is_leader
    ttng.wait_barrier %13, %c0_i32, %15 : ...                                         // only leader waits

    // --- 2-CTA MMA ---
    %16 = ttng.tc_gen5_mma %3, %6, %result[%token], %true, %true {two_ctas}
      : !ttg.memdesc<128x64xf16, #shared, #smem>,                                     // A: M-split
        !ttg.memdesc<64x128xf16, #shared1, #smem>,                                    // B: N-split
        !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>                // C: CTASplitM=2

    // --- Barrier cleanup ---
    ttng.inval_barrier %17 : ...
    ttg.local_dealloc %8 : ...

    // --- TMEM load + store result ---
    %result_4, %token_5 = ttng.tmem_load %result[%16] : ... -> tensor<128x128xf32, #blocked4>
    tt.store %19, %20 : tensor<128x128x!tt.ptr<f32>, #blocked5>
    tt.return
}
```

Key observations:
- **Output CTA layout**: `CTASplitNum = [2, 1]` (M-split) -- forced by PlanCTA
  because `{two_ctas}` was on the `DotOp`
- **B operand CTA layout**: `CTASplitNum = [1, 2]` (N-split) -- set by
  `splitBOperand()` in AccelerateMatmul
- **TMEM encoding**: `CTASplitM = 2` -- accumulator split along M across CTAs
- **Cross-CTA sync**: `arrive_barrier` + `wait_barrier` with leader predicate --
  inserted by `Insert2CTASync` pass
- **MMA**: `tc_gen5_mma {two_ctas}` -- 2-CTA cooperative MMA

---

## Implementation

### Changes Summary

| Component | File | Change |
|-----------|------|--------|
| **Python API** | `core.py` | Add `two_ctas=False` param to `tl.dot()` |
| **Semantic** | `semantic.py` | Thread `two_ctas` through `dot()` to `create_dot()` |
| **IR binding** | `ir.cc` | Add `bool twoCtas` to `create_dot`, set attr on DotOp |
| **DotOp IR** | `TritonOps.td` | Add `UnitAttr:$two_ctas` to DotOp arguments |
| **TTIR->TTGIR** | `TritonToTritonGPUPass.cpp` | Preserve `two_ctas` during DotOp conversion |
| **PlanCTA** | `PlanCTA.cpp` | Force M-split `{2,1}` when `dot.getTwoCtas()` |
| **Sync pass** | `Insert2CTASync.cpp` | **NEW**: Insert cross-CTA sync before 2-CTA MMA |
| **Pass def** | `Passes.td` | Add `NVGPUInsert2CTASync` pass definition |
| **Build** | `CMakeLists.txt` | Add `Insert2CTASync.cpp` |
| **Pass wrapper** | `triton_nvidia.cc` | Add `add_insert_2cta_sync` wrapper |
| **Pipeline** | `compiler.py` | Add pass after `add_pipeline` for SM100+ with cluster >= 2 |

### Key Design Decisions

1. **`tl.dot(two_ctas=True)` as explicit opt-in**: Prevents backward compat
   issues. Without this, any kernel with `cluster_dims >= 2` + `tl.dot()` would
   silently get 2-CTA behavior.

2. **PlanCTA fix vs separate passes**: Instead of D97215251's 2 post-AccelerateMatmul
   passes (`Propagate2CTAAttr` + `Transform2CTALoads`), we fix PlanCTA to produce
   the correct M-split layout. This lets AccelerateMatmul's existing
   `canUseTwoCTAs()` + `splitBOperand()` work naturally.

3. **Insert2CTASync after pipeline**: Unlike D97215251's `Insert2CTASync` which
   runs pre-pipeline and skips WS, our pass runs post-pipeline and is compatible
   with autoWS.

4. **`TritonToTritonGPUPass.cpp` fix**: The `TritonDotPattern` was dropping
   `two_ctas` when recreating `DotOp` during TTIR->TTGIR conversion. Fixed by
   explicitly copying the attribute.

---

## Files Changed

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `hopper/lib/Transforms/Insert2CTASync.cpp` | 175 | Cross-CTA sync pass |
| `test/Hopper/TwoCTA/insert_2cta_sync.mlir` | 66 | MLIR lit test for sync pass |
| `test/Hopper/TwoCTA/two_ctas_plancta.mlir` | 30 | MLIR lit test for full pipeline |
| `python/test/unit/hopper/test_2cta_sync.py` | 80 | Python compilation test |
| `third_party/tlx/tutorials/triton_addmm_2cta.py` | 120 | E2E test with TMA descriptors |

### Modified Files

| File | Change |
|------|--------|
| `include/triton/Dialect/Triton/IR/TritonOps.td` | `UnitAttr:$two_ctas` on DotOp |
| `python/triton/language/core.py` | `two_ctas=False` param on `tl.dot()` |
| `python/triton/language/semantic.py` | Thread `two_ctas` through `dot()` |
| `python/src/ir.cc` | `bool twoCtas` on `create_dot` |
| `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | Preserve `two_ctas` during conversion |
| `lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp` | Force M-split when `two_ctas` |
| `hopper/include/Transforms/Passes.td` | `NVGPUInsert2CTASync` pass definition |
| `hopper/lib/Transforms/CMakeLists.txt` | Add `Insert2CTASync.cpp` |
| `third_party/nvidia/triton_nvidia.cc` | `add_insert_2cta_sync` wrapper |
| `third_party/nvidia/backend/compiler.py` | Add pass to SM100+ pipeline |

---

## How to Test

### Step 1: Build the Triton Compiler

```bash
buck2 build @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  //third-party/triton/beta/triton:triton-opt
```

### Step 2: Run the MLIR Lit Test (no GPU needed)

```bash
buck2 test @fbcode//mode/opt \
  -m ovr_config//triton:beta \
  '//third-party/triton/beta/triton:lit_test/Hopper/TwoCTA/insert_2cta_sync.mlir'
```

Expected: `Tests finished: Pass 1. Fail 0.`

### Step 3: Verify Full Pipeline via triton-opt

Create the test input IR:

```bash
cat > /tmp/test_pipeline.mlir << 'EOF'
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @test_two_ctas(%a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %2 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %3 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
    %a = tt.load %2 : tensor<128x64x!tt.ptr<f16>>
    %b = tt.load %3 : tensor<64x128x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %cst, inputPrecision = tf32 {two_ctas} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    %4 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    tt.store %4, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}
EOF
```

Run the full pipeline:

```bash
buck2 run @fbcode//mode/opt -m ovr_config//triton:beta \
  //third-party/triton/beta/triton:triton-opt -- \
  --convert-triton-to-tritongpu="target=cuda:100 num-warps=4 threads-per-warp=32 num-ctas=2" \
  --tritongpu-coalesce \
  --triton-nvidia-gpu-plan-cta \
  --tritongpu-accelerate-matmul \
  --nvgpu-insert-2cta-sync \
  /tmp/test_pipeline.mlir
```

Expected output should contain:
- `CTASplitNum = [2, 1]` (M-split from PlanCTA)
- `CTASplitM = 2` (TMEM encoding)
- `tc_gen5_mma ... {two_ctas}` (2-CTA MMA)
- `nvgpu.cluster_id` + `map_to_remote_buffer` + `arrive_barrier` (cross-CTA sync)

### Step 4: Run E2E Python Test (requires Blackwell GPU)


```bash
CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 \
  buck2 run @fbcode//mode/opt \
    -m ovr_config//triton:beta \
    -c fbcode.enable_gpu_sections=true \
    -c fbcode.platform010_cuda_version=12.8 \
    --no-remote-cache \
    '//third-party/triton/beta/triton:py_2cta_sync_blackwell_test' \
    -- third-party/triton/beta/triton/third_party/tlx/tutorials/triton_addmm_2cta.py
```

Expected output:
```
GPU: NVIDIA GB200, SM 10.0
Compiling kernel with num_ctas=2, cluster_dims=(2,1,1)...
Compilation + execution OK

=== IR Check ===
  tc_gen5_mma: OK
  two_ctas: OK
  cluster_id: OK
  map_to_remote_buffer: OK
  arrive_barrier: OK

=== Correctness ===
  max_diff: 0.000000
  result: PASSED

=== TEST PASSED ===
```

### Debugging Tips

To see the TTGIR output at each pass stage:

```bash
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python your_kernel.py 2>&1 | grep -A1000 "make_ttgir"
```

To run a single pass in isolation on an MLIR file:

```bash
buck2 run @fbcode//mode/opt -m ovr_config//triton:beta \
  //third-party/triton/beta/triton:triton-opt -- \
  --nvgpu-insert-2cta-sync your_input.mlir
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

4. **D96323995 optimization**: A future improvement could use `.cta_group::2`
   on B-operand TMA loads for hardware-level barrier routing, potentially
   eliminating the explicit cross-CTA sync.
