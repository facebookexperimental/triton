# Code Partition Barrier Summary — BWD HD64

## Overview

This document summarizes the barrier structure produced by code partitioning
for the Flash Attention BWD persistent kernel with `HEAD_DIM=64`.
Analysis based on `tmp-ttgir/_attn_bwd_persist.ttgir`.

## TMEM Allocations

| Name | Shape | shareGroup | buffer.id | Encoding |
|------|-------|-----------|-----------|----------|
| dpT  | 1×128×128×f32 | 2 | 8 | blockM=128, blockN=128 |
| qkT  | 1×128×128×f32 | 0 | 7 | blockM=128, blockN=128 |
| dv   | 1×128×64×f32  | 1 | 6 | blockM=128, blockN=64  |
| dk   | 1×128×64×f32  | 3 | 5 | blockM=128, blockN=64  |

## SMEM Allocations

| Name | Shape | buffer.id | Notes |
|------|-------|-----------|-------|
| dsT  | 2×128×128×f16 | 0 | double-buffered |
| do   | 2×128×64×f16  | 1 | double-buffered |
| q    | 2×128×64×f16  | 2 | double-buffered |
| v    | 1×128×64×f16  | 3 | single-buffered |
| k    | 1×128×64×f16  | 4 | single-buffered |

## 4 Partitions (Warp Groups)

| Partition | Type | async_task_id | Warps | Role |
|-----------|------|---------------|-------|------|
| default / partition0 | reduction | 0 | 1 | dQ epilogue: tmem_load dQ → scale → TMA atomic_add to global |
| partition1 | gemm | 1 | 1 | All MMA operations: qkT, dpT, dV, dK, dQ |
| partition2 | load | 2 | 8 | TMA loads: k, v, q, do |
| partition3 | computation | 3 | 8 | Softmax, ppT, dsT computation; tmem_load qkT/dpT; tmem_store ppT |

## MMA Operations (all in Task 1 / partition1)

| MMA | Operand D (TMEM) | useAcc | Commit barriers | Source loc |
|-----|-----------------|--------|-----------------|------------|
| qkT MMA | `qkT_49` (memdesc_index of qkT) | `%false` | `%arg25` (1×1, HW commit) | #loc202 |
| dpT MMA | `dpT_57` (memdesc_index of dpT) | `%false` | `%arg32` (2×1, do consumed), `%arg33` (1×1, HW commit) | #loc201 |
| dV MMA  | `dv_78` (memdesc_index of dv)   | loop-carried (true after iter 0) | `%arg29` (1×1, HW commit) | #loc203 |
| dK MMA  | `dk_84` (memdesc_index of dk)   | loop-carried (true after iter 0) | `%arg36` (2×1, q consumed) | #loc204 |
| dQ MMA  | `dq_88` (**tmem_subslice of dpT**, cols 0-63) | `%false` | `%arg37` (2×1, dsT consumed), `%arg38` (1×1, dQ commit for Task 0) | #loc208 |

### dQ Operand D Chain

The dQ MMA's operand D is NOT a separate TMEM allocation. It is derived from
the dpT allocation via:

```
%dpT_86 = tmem_subslice %dpT_9 {N = 0}        → cols 0-63 of dpT (128×128)
%dpT_87 = memdesc_reinterpret %dpT_86          → 1×128×64
%dq_88  = memdesc_index %dpT_87[0]             → 128×64
dQ MMA writes to %dq_88
```

This is safe because of the **transitive dependency chain** (see below).

## Complete Barrier Map

### warp_specialize argument → partition arg mapping

The `warp_specialize` call passes barriers as positional arguments.
All partitions share the same physical barriers via these `%arg` references.

| warp_spec arg | Partition arg | Size | Purpose |
|---|---|---|---|
| `%23` | `%arg22` | 2×1 | q TMA load complete |
| `%26` | `%arg25` | 1×1 | qkT MMA HW commit |
| `%31` | `%arg28` | 2×1 | do TMA load complete |
| `%34` | `%arg29` | 1×1 | dV MMA HW commit |
| `%28` | `%arg32` | 2×1 | dpT MMA commit (do consumed) |
| `%36` | `%arg33` | 1×1 | dpT MMA HW commit |
| `%20` | `%arg36` | 2×1 | dK MMA commit (q consumed) |
| `%38` | `%arg37` | 2×1 | dQ MMA commit #1 (dsT consumed) |
| `%41` | `%arg38` | 1×1 | dQ MMA commit #2 (for Task 0 dQ consumer) |
| `%14` | `%arg39` | 1×1 | dK epilog commit |
| `%16` | `%arg40` | 1×1 | dK epilog commit #2 |
| `%18` | `%arg41` | 1×1 | dV epilog commit |
| `%8`  | `%arg42` | 1×1 | k TMA load gate (outer tile) |
| `%44` | `%arg57` | 1×1 | dQ consumed (by Task 0 → Task 1) |
| `%47` | `%arg58` | 2×1 | dsT ready (Task 3 → Task 1) |
| `%54` | `%arg59` | 1×1 | dpT consumed (Task 3 → Task 1) |
| `%57` | `%arg60` | 1×1 | ppT stored / dV consumed (Task 3 → Task 1) |
| `%62` | `%arg61` | 1×1 | qkT consumed (Task 3 → Task 1) |

## Producer-Consumer Barrier Flows

### Flow 1: qkT (shareGroup 0)

```
Task 1: wait %arg61 (qkT consumed) → qkT MMA → commit %arg25 (HW)
Task 3: wait %arg25 (qkT committed) → tmem_load qkT → arrive %arg61 (qkT consumed)
```

### Flow 2: dpT (shareGroup 2) — most complex

```
Task 1: wait %arg57 (dQ consumed) + wait %arg59 (dpT consumed) → dpT MMA → commit %arg32 (do consumed) + %arg33 (HW)
Task 3: wait %arg33 (dpT committed) → tmem_load dpT → arrive %arg59 (dpT consumed)
Task 2: wait %arg32 (do consumed) → TMA load do
```

### Flow 3: dV (shareGroup 1)

```
Task 0: tmem_store zeros → dV (init, with tmem.start=[7,7])
Task 3: wait %arg29 (dV committed) → tmem_store ppT → arrive %arg60 (ppT ready)
Task 1: wait %arg60 (ppT ready) → dV MMA (useAcc=true) → commit %arg29 (HW)
Task 3 (epilog): wait %arg41 → tmem_load dV → TMA store to global
```

### Flow 4: dK (shareGroup 3)

```
Task 0: tmem_store zeros → dK (init, with tmem.start=[9,9])
Task 1: wait %arg58 (dsT ready) → dK MMA (useAcc=true) → commit %arg36 (q consumed)
Task 2: wait %arg36 (q consumed) → TMA load q
Task 3 (epilog): wait %arg39 → tmem_load dK → TMA store to global
```

### Flow 5: dQ (subslice of dpT, shareGroup 2)

```
Task 1: dQ MMA (after dK MMA in sequential order) → commit %arg37 (dsT consumed) + %arg38 (dQ ready for Task 0)
Task 0: wait %arg38 (dQ committed) → tmem_load dQ (4 subslices × 128×16) → cp.reduce → arrive %arg57 (dQ consumed)
Task 1: wait %arg57 (dQ consumed) → dpT MMA (next iteration)
Task 3: wait %arg37 (dsT consumed) → store next dsT to SMEM
```

### Flow 6: dsT (SMEM, double-buffered)

```
Task 3: wait %arg37 (dsT consumed by prev) → local_store dsT → arrive %arg58 (dsT ready)
Task 1: wait %arg58 (dsT ready) → dK MMA (reads dsT) → dQ MMA (reads dsT)
Task 1: dQ MMA commit → arrive %arg37 (dsT consumed)
```

## Key Insight: dpT/dQ TMEM Sharing Is Safe

The dQ MMA writes to columns 0-63 of the dpT TMEM buffer. This does NOT race
with Task 3's `tmem_load dpT` because of the **transitive dependency chain**:

```
dpT MMA (Task 1)
  → commit %arg33 (dpT HW commit)
    → Task 3 waits %arg33
      → tmem_load dpT (Task 3 CONSUMES dpT)
        → compute dsT = pT * (dpT - Di)
          → local_store dsT to SMEM
            → arrive %arg58 (dsT READY)
              → Task 1 waits %arg58
                → dK MMA (reads dsT from SMEM)
                  → dQ MMA (writes to dpT subslice) ← dpT already consumed!
```

By the time dQ MMA executes, dpT has been consumed by Task 3. No explicit
dpT-consumed barrier is needed before dQ MMA — the dsT barrier (`%arg58`)
provides transitive protection.

## Barrier Initialization

All barriers are initialized with `init_barrier ..., 1` (arrival count = 1).
Barriers are separated by `gpu.barrier` calls to ensure visibility across
warp groups before the `warp_specialize` region begins.

Single-buffered barriers (`1×1`): phase alternates `curr_m & 1`.
Double-buffered barriers (`2×1`): indexed by `tile_idx % 2`.

## SWP (Software Pipelining)

The BWD kernel uses `num_stages=2`. The SWP pass in `SoftwarePipeliner.cpp`:
1. **Skips** when `num_stages=1` (`schedule.getNumStages() == 1 → continue`)
2. For WS loops with `num_stages=2`, generates prolog + peeled epilog
3. Stage 0 of iteration N+1 overlaps with stage 1 of iteration N

**To disable SWP**: set `num_stages=1` in the kernel's `tl.range()` call.
This makes the SWP pass skip the loop entirely — no prolog/epilog generation.

## Open Questions

- Both dK and dQ are corrupted in multi-wave execution while dV is always correct
- Single-wave (≤152 CTAs) is always correct
- The barrier structure appears sound at the TTGIR level
- Need to investigate whether the issue is in SWP (prolog/epilog barrier init)
  or elsewhere (phase calculation, barrier lowering, etc.)
