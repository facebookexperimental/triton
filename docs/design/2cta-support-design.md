# 2-CTA Support in Core Triton

**Author**: Manman Ren
**Date**: 2026-03-07
**Status**: Design Proposal

## Overview

This document outlines the design for adding 2-CTA (Cooperative Thread Array) support to core Triton via `tl.dot(..., two_ctas=True)`. The goal is to enable users to leverage 2-CTA MMA instructions on Blackwell GPUs (SM100+) without requiring TLX-specific APIs.

## Background

### What is 2-CTA Mode?

2-CTA mode enables two CTAs within a cluster to cooperatively execute a single MMA (Matrix Multiply-Accumulate) instruction. This provides:
- Larger effective tile sizes (up to 256×256)
- Better utilization of Blackwell's tensor cores
- Reduced synchronization overhead for large matrix operations

### Current State

2-CTA is currently supported only through **TLX** (Triton Language eXtensions):
- Tutorial: `/third_party/tlx/tutorials/blackwell_gemm_2cta.py`
- Requires explicit use of TLX APIs for cluster operations, barriers, and memory management

### Goal

Enable 2-CTA support in core Triton with minimal user-facing API changes:
```python
# User simply adds two_ctas=True
c = tl.dot(a, b, two_ctas=True)
```

The Triton compiler will automatically emit the necessary TTGIR operations.

## Key TLX Operations and TTGIR Mapping

| TLX API | Purpose | TTGIR Operation |
|---------|---------|-----------------|
| `tlx.cluster_cta_rank()` | Get CTA rank within cluster (0-N) | `triton::nvgpu::ClusterCTAIdOp` |
| `tlx.barrier_arrive(bar, count, remote_cta_rank)` | Signal mbarrier on remote CTA | `ttng::MapToRemoteBufferOp` + `ttng::ArriveBarrierOp` |
| `tlx.async_dot(..., two_ctas=True)` | 2-CTA matrix multiply | `ttng::TCGen5MMAOp` with `two_ctas=true` |

### TTGIR Operation Details

#### 1. ClusterCTAIdOp
**Location**: `third_party/nvidia/include/Dialect/NVGPU/IR/NVGPUOps.td:128`
```tablegen
def NVGPU_ClusterCTAIdOp : NVGPU_Op<"cluster_id", [Pure]> {
  // Returns unique CTA ID within cluster (1D rank across all dims)
}
```

#### 2. MapToRemoteBufferOp
**Location**: `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:73`
```tablegen
def TTNG_MapToRemoteBufferOp : TTNG_Op<"map_to_remote_buffer", [Pure, MemDescViewTrait]> {
  let summary = "Map shared memory buffer to the corresponding buffer in the target CTA";
  let arguments = (ins TTG_MemDescType:$src, I32:$ctaRank);
  let results = (outs TTG_MemDescType:$result);
}
```

#### 3. ArriveBarrierOp
**Location**: `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:248`
```tablegen
def TTNG_ArriveBarrierOp : TTNG_Op<"arrive_barrier"> {
  let arguments = (ins TTG_MemDescType:$alloc, I32Attr:$count, Optional<I1>:$pred);
}
```

#### 4. TCGen5MMAOp
**Location**: `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:533`
```tablegen
def TTNG_TCGen5MMAOp : TTNG_Op<"tc_gen5_mma", [...]> {
  let arguments = (ins
    TTG_MemDescType:$a,
    TTG_MemDescType:$b,
    TTG_MemDescType:$d,
    ...
    UnitAttr:$two_ctas  // ← 2-CTA flag
  );
}
```

## Implementation Design

### Phase 1: API Extension

**File**: `python/triton/language/core.py`

```python
def dot(a, b, acc=None, ..., two_ctas: constexpr = False):
    """
    Performs matrix multiplication: D = A @ B + C

    Args:
        ...
        two_ctas: If True, enables 2-CTA mode where two CTAs in a cluster
                  cooperatively execute the MMA. Requires:
                  - Blackwell GPU (SM100+)
                  - Cluster size >= 2 (ctas_per_cga >= (2, 1, 1))
    """
```

### Phase 2: Semantic Layer

Thread `two_ctas` parameter through the semantic layer to the IR builder:
- `python/triton/language/semantic.py`
- `lib/Dialect/Triton/IR/Ops.cpp`

### Phase 3: Compiler Passes

#### 3.1 Insert2CTASyncPass

**Purpose**: Auto-insert cross-CTA synchronization for 2-CTA MMA operations.

```cpp
// Pseudocode
void Insert2CTASyncPass::runOnOperation() {
  module.walk([](ttng::TCGen5MMAOp op) {
    if (!op.getTwoCtas()) return;

    // 1. Get cluster CTA rank
    Value ctaRank = builder.create<nvgpu::ClusterCTAIdOp>(loc);

    // 2. Compute leader predicate: cta_rank % 2 == 0
    Value isLeader = builder.create<arith::CmpIOp>(eq, ctaRank % 2, 0);

    // 3. Compute leader rank: cta_rank & ~1
    Value leaderRank = builder.create<arith::AndIOp>(ctaRank, ~1);

    // 4. Map local barrier to leader's barrier
    Value remoteMbar = builder.create<ttng::MapToRemoteBufferOp>(
        localMbar, leaderRank);

    // 5. Arrive on leader's barrier
    builder.create<ttng::ArriveBarrierOp>(remoteMbar, 1);

    // 6. Leader waits for both CTAs
    builder.create<ttng::WaitBarrierOp>(localMbar, phase, isLeader);
  });
}
```

#### 3.2 Transform2CTALoadsPass

**Purpose**: Split B matrix loads across CTAs.

**Transformation**:
- Each CTA loads `BLOCK_N // 2` columns of B
- CTA 0 loads columns `[0, N/2)`, CTA 1 loads columns `[N/2, N)`
- A matrix is shared fully (no split)

### Phase 4: Validation

Add compile-time checks:
- Error if `two_ctas=True` but target is not Blackwell (SM100+)
- Error if `two_ctas=True` but cluster size < 2
- Warning if tile dimensions are not optimal for 2-CTA

## Data Distribution

| Operand | 1-CTA Mode | 2-CTA Mode |
|---------|------------|------------|
| Matrix A (SMEM) | Full: `BLOCK_M × BLOCK_K` | Full (shared): `BLOCK_M × BLOCK_K` |
| Matrix B (SMEM) | Full: `BLOCK_K × BLOCK_N` | Split: `BLOCK_K × (BLOCK_N/2)` per CTA |
| Accumulator (TMEM) | Full: `BLOCK_M × BLOCK_N` | Full: `BLOCK_M × BLOCK_N` |

## Synchronization Pattern: "Arrive Remote, Wait Local"

```
CTA 0 (Leader)                    CTA 1 (Follower)
─────────────────                 ─────────────────
Load A[0:M, 0:K]                  Load A[0:M, 0:K]
Load B[0:K, 0:N/2]                Load B[0:K, N/2:N]
     │                                  │
     │  ←── barrier_arrive ────────────┘
     │
     ▼
Wait on barrier (arrive_count=2)
     │
     ▼
Issue tcgen05.mma.cta_group::2
     │
     ▼
Both CTAs get result in TMEM
```

## Implementation Roadmap

| Step | Component | Effort | Description |
|------|-----------|--------|-------------|
| 1 | `tl.dot` API | Easy | Add `two_ctas` parameter |
| 2 | Semantic layer | Easy | Thread parameter to builder |
| 3 | Backend emission | Easy | `TCGen5MMAOp` already has `two_ctas` attr |
| 4 | Insert2CTASyncPass | Medium | Auto-insert sync operations |
| 5 | Transform2CTALoadsPass | Medium | Split B loads across CTAs |
| 6 | Barrier allocation | Medium | Allocate mbarrier with `arrive_count=2` |
| 7 | Testing | Medium | Unit tests, correctness, benchmarks |

## Example Usage

### User Code (Simple)
```python
@triton.jit
def matmul_2cta(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Standard tile loading
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)

    # 2-CTA dot - compiler handles the rest
    c = tl.dot(a, b, two_ctas=True)

    tl.store(c_ptr + ..., c)

# Launch with cluster
matmul_2cta[grid](a, b, c, M, N, K,
                  BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
                  ctas_per_cga=(2, 1, 1))  # Required for 2-CTA
```

### Compiler Output (TTGIR)
```mlir
func.func @matmul_2cta(...) {
  // Auto-generated: Get CTA rank
  %cta_rank = nvgpu.cluster_id : i32
  %c2 = arith.constant 2 : i32
  %c1_not = arith.constant -2 : i32  // ~1
  %cta_rank_mod2 = arith.remui %cta_rank, %c2 : i32
  %c0 = arith.constant 0 : i32
  %is_leader = arith.cmpi eq, %cta_rank_mod2, %c0 : i1
  %leader_rank = arith.andi %cta_rank, %c1_not : i32

  // Auto-generated: Split B load
  %half_n = arith.constant 64 : i32  // BLOCK_N / 2
  %b_offset = arith.muli %cta_rank_mod2, %half_n : i32
  // ... load B with offset

  // Auto-generated: Cross-CTA sync
  %remote_mbar = ttng.map_to_remote_buffer %local_mbar, %leader_rank
  ttng.arrive_barrier %remote_mbar, 1
  ttng.wait_barrier %local_mbar, %phase, %is_leader

  // 2-CTA MMA
  ttng.tc_gen5_mma %a, %b, %acc {two_ctas} : ...
}
```

## Testing Plan

1. **Unit Tests**: Verify TTGIR generation with `two_ctas=True`
2. **Correctness Tests**: Compare 2-CTA results against reference (PyTorch)
3. **Performance Tests**: Benchmark against 1-CTA and TLX versions
4. **Edge Cases**:
   - Non-Blackwell GPUs (should error)
   - Cluster size 1 (should error)
   - Various tile sizes

## References

- TLX 2-CTA Tutorial: `/third_party/tlx/tutorials/blackwell_gemm_2cta.py`
- TLX Test: `/python/test/unit/language/test_tlx.py:test_async_dot_blackwell_2cta_tma`
- NVIDIA PTX ISA: `tcgen05.mma.cta_group::2` instruction
- TTGIR Ops: `/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`

---

## Appendix A: Pass Pipeline Analysis for 2-CTA Insertion

### Current SM100+ Pass Pipeline (from `compiler.py`)

The pass pipeline for Blackwell (SM100+) in `third_party/nvidia/backend/compiler.py:338-388`:

```python
# Phase 1: Initial IR conversion and layout optimization
passes.ttir.add_convert_to_ttgpuir(pm, ...)      # Convert Triton IR to TTGPU IR
passes.ttgpuir.add_coalesce(pm)                   # Coalesce memory accesses
tlx.tlx_passes.add_tlx_propagate_layout(pm)       # TLX layout propagation
tlx.tlx_passes.add_tlx_rewrite_local_alias(pm)    # TLX local alias rewriting
passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)        # F32 dot tensor core handling
nvidia.passes.ttnvgpuir.add_plan_cta(pm, ...)     # CTA planning

# Phase 2: Layout and matmul optimization
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_thread_locality(pm)
passes.ttgpuir.add_accelerate_matmul(pm)          # ← DOT → MMA conversion
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_dot_operands(pm, ...)
nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)

# Phase 3: Loop optimization
passes.ttir.add_loop_aware_cse(pm)
passes.ttgpuir.add_fuse_nested_loops(pm)
passes.common.add_canonicalizer(pm)
passes.ttir.add_triton_licm(pm)

# Phase 4: TMEM and data partitioning
passes.ttgpuir.add_optimize_accumulator_init(pm)
passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
nvidia.passes.hopper.add_data_partitioning(pm, 1)  # ← Data partition

# Phase 5: Scheduling and warp specialization
passes.ttgpuir.add_assign_latencies(pm, ...)
passes.ttgpuir.add_schedule_loops(pm, ...)
if use_meta_ws:
    nvidia.passes.hopper.add_partition_scheduling_meta(pm)  # Meta partition
    nvidia.passes.hopper.add_hopper_warpspec(pm, ...)       # Warp spec
else:
    passes.ttgpuir.add_warp_specialize(pm, ...)

# Phase 6: Pipelining
passes.ttgpuir.add_pipeline(pm, ...)
passes.ttgpuir.add_combine_tensor_select_and_if(pm)
passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)

# Phase 7: Final optimizations
passes.common.add_canonicalizer(pm)
passes.ttir.add_loop_aware_cse(pm)
passes.ttgpuir.add_prefetch(pm)
passes.ttgpuir.add_optimize_dot_operands(pm, ...)
passes.ttgpuir.add_coalesce_async_copy(pm)
nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
nvidia.passes.ttnvgpuir.add_tma_store_buffer_reuse(pm)
passes.ttgpuir.add_remove_layout_conversions(pm)
nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
passes.ttgpuir.add_reduce_data_duplication(pm)
passes.ttgpuir.add_reorder_instructions(pm)
passes.ttir.add_loop_aware_cse(pm)
passes.common.add_symbol_dce(pm)
passes.ttgpuir.add_optimize_partition_warps(pm)
nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
nvidia.passes.ttnvgpuir.add_lower_mma(pm)          # ← MMA lowering (tcgen05)
```

### Recommended 2-CTA Pass Insertion Points

For 2-CTA support, we need to insert passes at specific points:

#### Option A: Early Transformation (Recommended)

Insert **after `accelerate_matmul`** but **before `optimize_accumulator_init`**:

```python
# Phase 2: Layout and matmul optimization
passes.ttgpuir.add_accelerate_matmul(pm)          # DOT → TCGen5MMAOp
passes.ttgpuir.add_remove_layout_conversions(pm)
passes.ttgpuir.add_optimize_dot_operands(pm, ...)

# ─────────────── INSERT 2-CTA PASSES HERE ───────────────
if opt.two_ctas:
    nvidia.passes.ttnvgpuir.add_detect_2cta_dots(pm)    # NEW: Detect dots with two_ctas=True
    nvidia.passes.ttnvgpuir.add_transform_2cta_loads(pm) # NEW: Split B loads
    nvidia.passes.ttnvgpuir.add_insert_2cta_sync(pm)     # NEW: Insert cross-CTA sync
# ────────────────────────────────────────────────────────

nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
```

**Rationale**:
- `accelerate_matmul` has already converted `tt.dot` → `TCGen5MMAOp`
- The 2-CTA attribute is already on the MMA op
- We can now transform loads and insert sync before scheduling

#### Option B: Late Transformation (After Scheduling)

Insert **after `partition_scheduling_meta`** but **before `pipeline`**:

```python
# Phase 5: Scheduling and warp specialization
nvidia.passes.hopper.add_partition_scheduling_meta(pm)
nvidia.passes.hopper.add_hopper_warpspec(pm, ...)

# ─────────────── INSERT 2-CTA PASSES HERE ───────────────
if opt.two_ctas:
    nvidia.passes.ttnvgpuir.add_2cta_barrier_allocation(pm)  # Allocate arrive_count=2 barriers
    nvidia.passes.ttnvgpuir.add_2cta_sync_insertion(pm)       # Insert cross-CTA sync
# ────────────────────────────────────────────────────────

passes.ttgpuir.add_pipeline(pm, ...)
```

**Rationale**:
- Warp specialization is already done
- Barrier allocation can account for 2-CTA requirements
- Sync insertion happens before pipelining transforms barriers

### Proposed 2-CTA Passes

| Pass | Location | Purpose |
|------|----------|---------|
| `Detect2CTADotsPass` | After `accelerate_matmul` | Find `TCGen5MMAOp` with `two_ctas=True` |
| `Transform2CTALoadsPass` | After detection | Split B matrix loads (each CTA loads half) |
| `Insert2CTASyncPass` | After load transform | Insert `ClusterCTAIdOp`, `MapToRemoteBufferOp`, `ArriveBarrierOp` |
| `Allocate2CTABarriersPass` | After warp spec | Modify barrier `arrive_count` to 2 |

### Implementation Priority

| Priority | Pass | Why |
|----------|------|-----|
| 1 | `Insert2CTASyncPass` | Core synchronization pattern |
| 2 | `Transform2CTALoadsPass` | B matrix split is critical for perf |
| 3 | `Allocate2CTABarriersPass` | Barrier allocation adjustment |
| 4 | `Detect2CTADotsPass` | May be combined with existing logic |

### Interaction with Warp Specialization

When 2-CTA is combined with warp spec:

1. **Each CTA independently** does producer/consumer warp spec
2. **Cross-CTA sync** is orthogonal to within-CTA warp spec
3. The 2-CTA sync happens at **tile boundaries**, not K-loop iterations
4. Insert 2-CTA sync passes **before** warp spec passes

```
┌─────────────────────────────────────────────────────────────┐
│                        Cluster                              │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │       CTA 0         │    │       CTA 1         │        │
│  │  ┌───────┐ ┌─────┐  │    │  ┌───────┐ ┌─────┐  │        │
│  │  │Prod WG│→│Cons │  │    │  │Prod WG│→│Cons │  │        │
│  │  │Load A │ │MMA  │  │    │  │Load A │ │MMA  │  │        │
│  │  │Load B0│ │Store│  │    │  │Load B1│ │Store│  │        │
│  │  └───────┘ └─────┘  │    │  └───────┘ └─────┘  │        │
│  └──────────┬──────────┘    └──────────┬──────────┘        │
│             │    Cross-CTA Sync        │                   │
│             └──────────────────────────┘                   │
│                    arrive_remote()                         │
│                    wait_local()                            │
└─────────────────────────────────────────────────────────────┘
```

### Overview

Meta's warp specialization (autoWS) is a set of compiler passes that automatically partition work across warp groups. This is triggered by:
- `tl.range(..., warp_specialize=True)` on loops
- Environment variables: `TRITON_USE_META_PARTITION=1 TRITON_USE_META_WS=1`

### Key Passes (in order)

| Pass | Purpose |
|------|---------|
| `nvgpu-ws-data-partition` | Partition data across warp groups |
| `tritongpu-assign-latencies` | Assign latency attributes (use-meta-ws=true) |
| `tritongpu-schedule-loops` | Schedule loops with WS annotations |
| `nvgpu-partition-scheduling-meta` | Meta's scheduling partitioner |
| `nvgpu-warp-specialization` | Main WS transformation |
| `tritongpu-pipeline` | Software pipelining |
| `tritongpu-optimize-partition-warps` | Optimize warp partition |

### Auto WS vs TLX Explicit WS

| Aspect | Auto WS | TLX Explicit WS |
|--------|---------|-----------------|
| API | `tl.range(..., warp_specialize=True)` | `tlx.async_tasks()` with explicit tasks |
| Control | Compiler-driven | User-defined task partitioning |
| Kernel Example | `_addmm_fwd_tma_persistent` | `_addmm_fwd_tma_ws_persistent` |
| Flexibility | Lower | Higher |
| Maintenance | Easier | More complex |

### Example: Auto WS Kernel Structure

```python
@triton.jit
def _addmm_fwd_tma_persistent(..., WARP_SPECIALIZE: tl.constexpr):
    # Outer loop with warp spec
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, 
                            flatten=True, 
                            warp_specialize=WARP_SPECIALIZE):
        # Inner K loop with warp spec
        for k in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
            x = x_desc.load([offs_xm, offs_k])
            w = w_desc.load([offs_k, offs_wn])
            accumulator = tl.dot(x, w, accumulator)
        
        # Epilogue
        z_desc.store([offs_xm, offs_wn], z)
```

### Example: TLX Explicit WS Kernel Structure

```python
@triton.jit
def _addmm_fwd_tma_ws_persistent(...):
    # Allocate buffers and barriers explicitly
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), dtype, NUM_BUFFERS)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    
    with tlx.async_tasks():
        # Consumer task (default)
        with tlx.async_task("default"):
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                tlx.barrier_wait(tmem_full_bars[cur_buf], phase)
                acc = tlx.local_load(tmem_buffers[cur_buf])
                z_desc.store([offs_xm, offs_wn], z)
        
        # Producer task
        with tlx.async_task("producer0"):
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                for k in range(0, k_tiles):
                    tlx.async_descriptor_load(x_desc, x_buffers[buf], ...)
                    tlx.barrier_arrive(smem_full_bars[buf], 1)
```

### 2-CTA + Warp Spec Interaction

When combining 2-CTA with warp specialization:

1. **Cluster Setup**: `ctas_per_cga=(2, 1, 1)` creates a 2-CTA cluster
2. **CTA Rank**: Each CTA gets rank 0 or 1 via `tlx.cluster_cta_rank()`
3. **B Matrix Split**: CTA 0 loads B[:, 0:N/2], CTA 1 loads B[:, N/2:N]
4. **Cross-CTA Sync**: Leader CTA waits for both before MMA
5. **Warp Spec within CTA**: Each CTA independently does producer/consumer warp spec

```python
if PAIR_CTA:
    cluster_cta_rank = tlx.cluster_cta_rank()
    pred_cta0 = cluster_cta_rank == 0
    cta_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=2)
    
    # Each CTA loads half of B
    w_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), w_desc.dtype, NUM_BUFFERS)
```

### Current Limitation (Compiler Bug)

As of 2026-03-07, Meta's auto WS with `WARP_SPECIALIZE=True` on `_addmm_fwd_tma_persistent` fails with:

```
Assertion `id && id->size() == 1' failed
  in WarpSpecialization/Partition.cpp:143
Pipeline failed: nvgpu-partition-scheduling-meta
```

This occurs because the `nvgpu-partition-scheduling-meta` pass doesn't properly handle certain loop structures when `flatten=True` is combined with nested `warp_specialize=True` loops.

### Benchmark Results (NVIDIA B200, 4096×4096×4096 FP16)

| Implementation | Time (ms) | TFLOPS | Notes |
|----------------|-----------|--------|-------|
| torch.addmm (cuBLAS) | 0.156 | 879 | Baseline |
| TMA Persistent (WS=False) | 0.576 | 239 | No warp spec |
| TMA Persistent (WS=True) | FAIL | - | Compiler bug |
| TLX WS Persistent | ~0.17 | ~800 | Explicit TLX |
| triton_addmm (top-level) | ~0.17 | ~830 | TLX-based |

### Recommended Approach for 2-CTA

Until the compiler bug is fixed, use **TLX explicit warp specialization** for 2-CTA kernels:

1. Use `tlx.async_tasks()` with explicit task definitions
2. Use `tlx.async_dot(..., two_ctas=True)` for 2-CTA MMA
3. Manually manage barriers with `tlx.alloc_barriers(arrive_count=2)`
4. Split B matrix loads based on `tlx.cluster_cta_rank()`
