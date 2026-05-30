# Replacing LLIR SCHED with a TTGIR-level pass — design discussion

> **STATUS (2026-05-30):** This design is **implemented**. See
> [`README.md`](README.md) for the doc index, [`llir_sched_at_ttgir_plan.md`](llir_sched_at_ttgir_plan.md)
> for the phased plan (Phases 0-4 + 6 landed; Phase 5 deferred), and
> [`ttgir_sched_status.md`](ttgir_sched_status.md) for e2e validation
> data. Headline outcome: stand-alone matmul +4.0 % mean (up to +11 %
> at K=8192); FA-fwd runs correctly under the new pass (vs the
> matmul_4waves LLIR pass which crashes on it).

This document summarizes the design discussion around moving the
`TRITON_ENABLE_LLIR_SCHED` pass (currently in
`third_party/amd/lib/TritonAMDGPUToLLVM/LLIRSchedule.cpp`) from
LLVM IR level up to TTGIR level. The motivation is the same as the
broader matmul_4waves story (see `matmul_4waves_summary.md` §1,
`gl_matmul_passes_summary.md`, `llir_dump/v8_metamd/README.md`).

## Why move it?

Current pass is **pattern-matched** to a specific LLIR shape — see the
analysis in `matmul_4waves_summary.md` and the IR-diff in
`llir_dump/v8_metamd/README.md` for the actual reorder it does. Concretely:

| Property | LLIR pass today | What TTGIR could deliver |
|---|---|---|
| Dataflow safety | None — crashes with `Instruction does not dominate all uses!` on FA-fwd (see `llir_dump/fa_fwd/README.md`) | MLIR has a real verifier + typed SSA; reorder can refuse to violate dominance |
| Coverage | Only works on the v10/v8 anchor pattern; crashes / no-ops elsewhere | Could pattern-match TTGIR ops (`ttg.local_load`, `ttg.local_store`, `ttg.async_copy_global_to_local`, `tt.dot`) — portable across LLVM-version intrinsic-name churn |
| Debuggability | Requires LLVM debug builds, intrinsic-name knowledge | `--mlir-print-ir-after-each-pass` works out of the box |
| Reusability across arches | gfx9 + gfx12 intrinsic matching is hand-rolled | Same TTGIR walker works for any AMDGCN target whose lowering produces equivalent ops |
| Suppresses backend misched | Required (and currently disables it) | Could insert `llvm.amdgcn.sched.barrier(0)` at MLIR→LLVM lowering — keeps misched enabled, just barred from crossing the markers |

The big risk: the scheduler operates on **individual MFMA intrinsics** and
**individual `ds_read.tr` vectors**, both of which only exist
after MLIR→LLVM lowering. To move the pass up, the corresponding
fine-grained ops have to exist (or be created) at TTGIR.

## The granularity reality at TTGIR (v8 / v10)

From the actual TTGIR dump
(`llir_dump/v8_metamd/pre_ttgir.mlir`), the inner loop of v8 has:

| op | count | typical shape |
|---|---:|---|
| `tt.dot` | 14 | `tensor<256x64> × tensor<64x128> → tensor<256x128>` (most), plus epilogue dots `<64x64> × <64x128>` |
| `ttg.local_load` | 12 | `<256x64>` (A) and `<64x128>` (B half) |
| `amdg.buffer_load_to_local` | 12 | same shapes |
| `amdg.extract_slice` | 12 | for B-halves / acc-row chunks |
| `ttg.async_commit_group` / `async_wait` | 8 / 6 | async pipeline |
| `scf.for` | 1 | K-loop |

The 256 MFMAs and ~256 `ds_read.tr` vectors per inner iteration
emerge during **TTGIR → LLVM lowering**, driven by the
`AMDMFMALayout(version=4, instr_shape=[16,16,32], warps_per_cta=[2,2])`
attribute on the result tensor. Each `tt.dot tensor<256x64> ×
tensor<64x128>` expands to (256/16) × (128/16) = 128 MFMA tile-calls.

So **a TTGIR-level SCHED needs the kernel decomposed down to MFMA-tile
granularity in TTGIR** before it has anything fine-grained to schedule.

## Two paths: A (kernel-author) or B (compiler pass)

| Path | Description | Cost | Outcome |
|---|---|---|---|
| **A** — Author-side fine slicing in Gluon | Kernel author writes ~128 `gl.amd.cdna3.mfma(a_chunk, b_chunk, acc_chunk)` calls per logical dot at `[16,64] × [64,16]` tile granularity, with matching `extract_slice` on every operand | High — `~250 lines of boilerplate Python per kernel`. Not practical without macro / loop-unroll support in Gluon | Same effect as B at code-gen time, but burdens every kernel author |
| **B** — TTGIR tile-split + schedule pass | Compiler pass takes a `tt.dot tensor<256x64> × tensor<64x128>` and rewrites into 128 `tt.dot tensor<16x64> × tensor<64x16>` with `extract_slice` / `insert_slice` on operands and accumulator. Then reorder + insert `sched.barrier`s | Medium — needs a careful MLIR rewrite + plumb through pipeliner + layout-system | Reusable across kernels; matmul_4waves Gluon source stays at current granularity |

Below is the analysis of path **B** — which existing infrastructure to reuse.

## Existing slicing/decomposition infrastructure in `MetaMain/triton`

Three candidate sources of reusable code, in increasing order of fit:

### 1. `BlockPingpong.cpp::sliceDot` (AMD, K-dim splitter)

`/home/mren/MetaMain/triton/third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp:468`

- Splits a `tt::DotOp` along K into `numSlices` sub-dots.
- `genLocalSlice` (line 421) creates per-slice `ttg::MemDescSubsliceOp` (shmem-side subview) + `ttg::LocalLoadOp` (register-side reload) with the correct `DotOperandEncodingAttr`.
- Chains accumulator via `mapping.map(op.getC(), prevDot->getResult(0))`.
- AMD-flavored; already knows MFMA encoding, kWidth, shmem descriptors.

**Limit**: only splits along K. The accumulator stays full-shape, so the final `op->replaceAllUsesWith(prevDot)` returns a value of the **original `tensor<MxN>` shape** — no user-side rewrite is needed. Users see no change.

### 2. `WGMMAPipeline.cpp::splitRSDot` (Hopper WGMMA K-dim splitter)

`/home/mren/MetaMain/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:312`

- Same idea, cleaner MLIR: `splitLhs` (line 240) uses `tt::ReshapeOp` + `tt::TransOp` + recursive `tt::SplitOp` for tensor operands; `splitRhs` (line 289) uses `MemDescSubsliceOp` for shmem.
- Most idiomatic of the three.

**Limit**: same as BlockPingpong — K-split only, no user-side rewrite.

### 3. `WSDataPartition.cpp` (Hopper warp-spec data partitioner)

`/home/mren/MetaMain/triton/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp` (2036 lines)

Full-blown SSA-walking partitioner. The pieces that matter:

| Piece | Location | What it does |
|---|---|---|
| `DataPartitionScheme` | line 93 | Per-op partition dim, dot operand index, rematerialized ops, skip set, function-arg dims |
| `getBackwardSliceToPartition` | line 291 | Recursively walks producers, propagates partition dim, flips through `TransOp` / `MemDescTransOp`, expands through `ExpandDimsOp`, triggers `rematerializeOp` on conflicts |
| `getForwardSliceToPartition` | line 441 | **Symmetric forward walker — propagates through users**, classifies user ops, detects conflicts |
| `sliceOp` | line 853 + 1443 | Clones op with retyped results (`MemDescType`, `RankedTensorType`, `TensorDescType`, `TensorMemoryEncodingAttr`); special-cases dot operand-side vs result-side dim handling |
| `rewriteRematerializedOps` | line 764 | Duplicates ops that need to live in multiple partition dims |
| `doDeepCleanup` | line 1461 | DCE / cleanup after partitioning |
| Top-level driver | `NVGPUWSDataPartitionPass` (line 2001) | Orchestrates the whole flow |

This is the **only one** of the three with both forward + backward
walkers and full retyping of the user chain.

## Why WSDataPartition is the right fit (not BlockPingpong / WGMMAPipeline)

The K-only splitters preserve the result `tensor<MxN>` shape — the
accumulator is summed across all K-slices and the final value drops in
where the original lived. Users see nothing.

For an MFMA-tile-grain TTGIR pass we **must split along M and N too**
(see "Do we need to split along K?" below). That changes the result
shape from `tensor<256x128>` to many smaller tiles, so users
(`fptrunc → local_store → buffer_store` for write-back, or chained
`tt.dot` for FA-style kernels) must be re-typed and re-cloned.

That's exactly what `WSDataPartition::sliceOp` does and what
BlockPingpong / WGMMAPipeline don't.

## Recommended adaptation path

1. Copy `WSDataPartition.cpp` → new file under
   `third_party/amd/lib/TritonAMDGPUTransforms/` (or shared TritonGPU
   transforms). Strip Hopper-only branches: `WarpGroupDotOp`,
   `TCGen5MMAOp`, `TMEMAllocOp` / `TMEMLoadOp` / `TMEMStoreOp`,
   `TensorMemoryEncodingAttr` rewrite, `AsyncTaskId` plumbing.
2. Replace the `computePartitionScheme` entry point: instead of warp-spec
   heuristics, scan the inner loop for `tt::DotOp`s, read their result
   `AMDMFMAEncodingAttr::getInstrShape()`, and emit one partition round
   per dim to be split.
3. Borrow `genLocalSlice` from `BlockPingpong.cpp:421` for the AMD shmem-
   subview machinery (`MemDescSubsliceOp` + `LocalLoadOp` with the right
   `DotOperandEncodingAttr`).
4. Reuse the rest of WSDataPartition as-is for the SSA mechanics (forward/
   backward walk, retype, remat, cleanup).
5. After splitting, either:
   - **Annotate** the scheduling region (no physical reorder at TTGIR; let
     the existing LLIR pass run as today on the lowered IR), OR
   - Reorder at TTGIR and emit `triton_amdgpu.sched_barrier` markers that
     lower to `llvm.amdgcn.sched.barrier(0)` — so misched stays enabled
     in the LLVM backend but can't cross the markers.

## Do we need to split along K?

Short answer: **No for capturing most of the SCHED win; yes if we want to
fully eliminate the LLIR pass.**

### What "fully" would require

The LLIR pass interleaves **individual MFMA intrinsics** with
**individual `ds_read.tr` vectors** at 1-to-1 grain. To do that at TTGIR
both sides need to exist at that grain:

- **Dots**: split all three dims → `tensor<16x32> × tensor<32x16> → tensor<16x16>` = exactly 1 MFMA per `tt.dot`. **K split needed.**
- **LDS reads**: `ttg.local_load` returns a full operand tile (e.g. `<256, 64>`); individual `ds_read.tr` vectors only appear after lowering. To pair with single MFMAs we'd need to **also** split `local_load` down to per-MFMA-tile vectors — a non-trivial second pass / representation change.

That second item is the bigger lift, and it's the real blocker for moving
SCHED entirely to TTGIR.

### Interleave-slot count for v8/v10 inner loop (BM=256, BN=128 per half, BK=64, MFMA 16x16x32)

| Split scheme | sub-dots per `tt.dot` | MFMAs per sub-dot | Total slots / loop iter (both halves) |
|---|---:|---:|---:|
| No split (today)         | 1   | 256 | 0 |
| K only                   | 2   | 128 | 4 |
| M only                   | 16  | 16  | 32 |
| N only                   | 8   | 32  | 16 |
| **M + N**                | 128 | 2   | **256** ← matches LLIR pass's interleave count |
| M + N + K                | 256 | 1   | 512 ← LLIR-grain fully replicated |

The **M + N split already gives 256 interleave slots per loop iteration**,
which matches the 256 MFMA + ~256 LR vectors the LLIR scheduler operates
on. But each TTGIR slot still corresponds to 2 MFMAs sharing an
accumulator, so the **physical schedule** at AMDGCN level pairs them tightly.
Whether that's a problem is empirical:
- If 2-MFMA pairs sharing an accumulator naturally form back-to-back work
  with shared operand registers, leaving them paired is fine (and even
  beneficial for register reuse).
- If breaking them apart matters, you need K-split too — or keep the LLIR
  pass as a downstream cleanup.

### Recommendation on K-split

**Start with M + N split only.** Reasons:

1. Reuses WSDataPartition framework cleanly — two passes of the existing
   scheme, both on output-side dims, no accumulator-chain complication.
2. Doesn't require breaking `local_load` apart — each existing per-iter
   local_load naturally feeds the 8 (M-row) or 16 (N-col) sub-dots.
3. `kWidth` in `DotOperandEncodingAttr` stays at 8 (matched to BK=64 /
   8-elements-per-lane) — no encoding adjustment.
4. v10/v8 already manually N-split via dual `acc0`/`acc1` — so M-split is
   the only new structural change.
5. The 2-MFMA pair per sub-dot is finished by LLIR SCHED if needed.

**Add K-split later** only if measurement shows the LLIR pass's intra-pair
interleave is irreplaceable AND you're willing to also build per-vector
`local_load` decomposition.

## Final answer

**Yes, it's possible.** The slicing primitives needed for path B are
production-grade in three independent places — WGMMAPipeline (cleanest
SSA), BlockPingpong (closest to AMD MFMA layout), and WSDataPartition
(the only one with full user-side walker + retype). For SCHED specifically,
**WSDataPartition is the right base** because we must split along M/N
(which changes the dot's result shape and therefore requires user-side
rewriting that the K-only splitters don't do).

The recommended split scheme is **M + N only** (skip K), accepting that
the LLIR pass stays on as a downstream cleanup for the intra-pair
1-MFMA-per-LR interleave. Full LLIR replacement requires K-split + a new
`local_load` decomposition pass, which is a much larger investment than
the +17–22 % win the current LLIR pass already delivers on v10/v8.

The downstream-control concern (LLVM-level misched undoing the TTGIR
schedule) is handled by lowering `triton_amdgpu.sched_barrier` markers to
`llvm.amdgcn.sched.barrier(0)` — already an established AMDGPU intrinsic;
no need to disable misched.

## Companion docs

- `matmul_4waves_summary.md` §1 — branch-level overview of the scheduler design
- `gl_matmul_passes_summary.md` — performance numbers for the current LLIR pass on v10/v8
- `llir_dump/v8_metamd/README.md` — actual pre vs post LLIR IR for v8 + SCHED
- `llir_dump/fa_fwd/README.md` — the SSA-dominance failure on FA-fwd that a TTGIR pass would avoid by construction

## Key file references

- `~/AMD/triton/third_party/amd/lib/TritonAMDGPUToLLVM/LLIRSchedule.cpp` — the current pass (1579 lines)
- `~/MetaMain/triton/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp` — partition framework (2036 lines)
- `~/MetaMain/triton/third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp:421-501` — `genLocalSlice` + `sliceDot` (AMD shmem subview machinery)
- `~/MetaMain/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:240-367` — `splitLhs` / `splitRhs` / `splitRSDot` (cleanest MLIR splitter reference)
- `~/AMD/triton/study_matmul/gluon/matmul_kernels/matmul_kernel.py::v10` (line 1532) and `~/MetaMain/.../v8_beyond_hotloop/matmul_kernel.py` — the two reference kernels the pass would target
