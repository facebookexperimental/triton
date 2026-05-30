# Plan: TTGIR-level M + N dot decomposition for SCHED

Concrete, phased implementation plan for the "M + N split only" path
described in `llir_sched_at_ttgir_design.md`. Goal: a new TTGIR-level
AMD pass that decomposes the v8/v10 main-loop `tt.dot` ops into
MFMA-tile-sized sub-dots along M and N (skipping K), then reorders them
with the same 4-MFMA-per-GR / 1-MFMA-per-LR recipe the current LLIR pass
uses. K-split is **explicitly deferred** to a follow-up; the existing
LLIR pass stays on as a downstream cleanup for the intra-K-pair interleave.

## Success criteria (top-level)

- New pass builds, links into the AMD pipeline, and is opt-in behind
  `TRITON_ENABLE_TTGIR_SCHED=1` (mirroring the existing
  `TRITON_ENABLE_LLIR_SCHED` knob).
- Numerical correctness verified on v10 and v8 (`bench.py --version 8`
  reports `Triton and Torch match` for all K ∈ {1024..16384}, all dtypes).
- Performance:
  - With the new pass ON and `TRITON_ENABLE_LLIR_SCHED=0`: should reach
    ≥ 80 % of v8 + SCHED+AS (i.e. ≥ 1054 TF mean at fp16 vs the 1318 TF
    target).
  - With both passes ON (TTGIR pass + LLIR pass downstream): should match
    or exceed today's SCHED+AS numbers (1318 TF mean fp16 on v8).
- Coverage: pass either no-ops or produces correct schedule on every kernel
  that currently breaks the LLIR pass (stand-alone matmul, triton_kernels,
  FA-fwd). Acceptable outcomes: no-op (because no matching pattern) or
  numerically-correct reorder. **Not acceptable: crash or wrong output.**

## Phase 0 — Scaffold (~1 day)

Bring up an empty pass plumbed into the AMD pipeline so the rest is
incremental.

**Files to create:**

```
third_party/amd/lib/TritonAMDGPUTransforms/
├── DotDecomposeAndSchedule.cpp     # new pass body (start ~50 lines, no-op)
└── CMakeLists.txt                  # add the new .cpp
third_party/amd/include/TritonAMDGPUTransforms/
└── Passes.td                       # register `TritonAMDGPUDotDecomposeAndSchedulePass`
third_party/amd/python/
└── triton_amd.cc                   # expose `add_ttgir_sched_pass`
include/triton/Tools/Sys/
└── GetEnv.hpp                      # register TRITON_ENABLE_TTGIR_SCHED env var
third_party/amd/backend/
└── compiler.py                     # gate the pass on the env var
```

**Pass body (initial):**
- Inherits `OperationPass<ModuleOp>` (start with module scope; can scope to
  func later if perf matters).
- Walks `tt.FuncOp → scf.ForOp` and emits a remark `"ttgir-sched: visited N
  forOps with M dot ops"`. That's it — proves the plumbing.

**Tests:**
- `test/TritonGPU/amd/ttgir-sched-noop.mlir` — lit test that runs the new
  pass with `--triton-amdgpu-dot-decompose-and-schedule` on a trivial
  function and asserts the remark fires + IR is unchanged.

**Pass/fail gate:**
- `ninja triton-amdgpu` builds.
- `bench.py --version 8 --K 4096` with `TRITON_ENABLE_TTGIR_SCHED=1` runs
  successfully with the remark in stderr, same TFLOPS as stock.

## Phase 1 — M-split only (~3-5 days)

Implement single-dim partitioning along M, reusing WSDataPartition's
machinery. **No N-split, no schedule reorder yet.** Just verify the
slicer produces correct IR.

**Files to create:**

```
third_party/amd/lib/TritonAMDGPUTransforms/
├── DotPartitionScheme.{h,cpp}      # AMD-flavored DataPartitionScheme
│                                    # — copy of struct from WSDataPartition.cpp:93
│                                    #   stripped of AsyncTaskId + TMEM fields
├── DotSliceWalker.cpp              # adapted getBackwardSliceToPartition
│                                    # + getForwardSliceToPartition from
│                                    # WSDataPartition.cpp:291-573
└── DotSliceOp.cpp                  # adapted sliceOp from
                                     # WSDataPartition.cpp:853-1442
                                     # — strip WarpGroupDotOp, TCGen5MMAOp,
                                     #   TMEMAllocOp; keep tt.DotOp branch
```

**Reused-but-modified from `WSDataPartition.cpp`:**

| Function | Action |
|---|---|
| `DataPartitionScheme` struct | Copy; remove `funcArgPartitionDims` (TMA-only) |
| `getBackwardSliceToPartition` (line 291) | Copy as-is; remove the `WarpGroupDotOp` branch, replace with `tt::DotOp` |
| `getForwardSliceToPartition` (line 441) | Copy as-is; remove warp-spec specific user checks |
| `getSliceToPartition` (line 575) | Copy as-is |
| `rematerializeOp` (line 230) | Copy as-is |
| `rewriteRematerializedOps` (line 764) | Copy as-is |
| `sliceOp` (line 853) | Copy; in the operand-vs-result-dim special-case, replace `WarpGroupDotOp`/`TCGen5MMAOp` with `tt::DotOp`; remove `TensorMemoryEncodingAttr` rewrite |

**New driver in `DotDecomposeAndSchedule.cpp`:**

```cpp
void DotDecomposeAndSchedulePass::runOnOperation() {
  if (!enabledViaEnv()) return;

  getOperation().walk([&](scf::ForOp forOp) {
    if (!isInnerMatMulLoop(forOp)) return;        // pattern guard
    DotPartitionScheme scheme;
    if (!planMSplit(forOp, scheme)) return;       // pick numPartitions = BM / instrM
    if (!getSliceToPartition(scheme)) {
      forOp.emitRemark() << "ttgir-sched: M-split infeasible, skipping";
      return;
    }
    applySliceOp(scheme);                         // clones the SSA chain
  });
}
```

`planMSplit` reads the `tt::DotOp`'s result encoding, extracts
`AMDMFMAEncodingAttr::getInstrShape()`, sets `numPartitions = BLOCK_M /
instrShape[0]` (e.g. 256 / 16 = 16), sets partition dim = 0 (M).

`isInnerMatMulLoop` heuristic: the forOp's body contains ≥ 1 `tt::DotOp`
whose result has an `AMDMFMAEncodingAttr` and operands flow back to
`ttg::LocalLoadOp` / `amdg::buffer_load_to_local`. Refuse on anything else
(this is the coverage guard that lets us no-op cleanly on FA-fwd, etc.).

**Tests:**
- `test/TritonGPU/amd/dot-partition-m-only.mlir` — synthetic 1-dot loop;
  `// CHECK:` that the dot is split into N sub-dots along M, all
  accumulator users are also retyped, and the IR verifies.
- Re-run `bench.py --version 8 --K 4096 --dtype fp16` with
  `TRITON_ENABLE_TTGIR_SCHED=1`. Expected: numerically correct, same or
  slightly worse perf (no schedule reorder yet, just more IR — should be
  within ±5 % of stock).

**Pass/fail gate:**
- M-split sub-dots visible in `--mlir-print-ir-after-all` output.
- v8 numerically correct, perf within ±5 % of stock.
- Re-run lit suite: nothing else regresses.

**Risks:**
- **Layout-system rejection of sub-tile encodings.** The smaller dots have
  result type `tensor<16x128xf32, #mma>` where `#mma` is the same
  `AMDMFMAEncodingAttr`. Verifier may insist the tensor shape match the
  MFMA layout's tile structure exactly. Mitigation: if rejected, introduce
  a smaller MFMA layout `#mma_sliced` and convert via `ttg.convert_layout`.
  Cost: one extra layout, one extra layout-conversion op per sub-dot.
- **Accumulator chaining across iterations.** Today the `tt.dot %a, %b,
  %prev_acc` uses an iter-arg `%prev_acc` produced by `scf.for`. After
  M-split, each sub-dot needs its own iter-arg slice — so `scf.for` itself
  gets new iter-args. WSDataPartition handles this via `YieldOp`
  recognition in `getForwardSliceToPartition`. Mitigation: ensure the
  YieldOp branch is preserved verbatim from WSDataPartition.

## Phase 2 — N-split (~2-3 days)

Compose N-split on top of M-split, reusing the same machinery.

**Implementation:**
- `planNSplit` analogous to `planMSplit`, sets dim = 1, `numPartitions = BLOCK_N
  / instrShape[1]` (e.g. 128 / 16 = 8).
- Two-pass strategy: run the partitioner with `dim = 0` first, then with
  `dim = 1`. The forward walker on the M-partitioned IR will recursively
  apply N-split to each M-slice's sub-dot.
- v10 / v8 already manually do an "N-split into halves" via `acc0`/`acc1`
  + `extract_slice`. **Detect this and skip** — don't re-split the already-
  sliced halves. Add a check: if the parent op is already
  `amdg::ExtractSliceOp` with offsets that span the natural N-half,
  treat the dot as already N-partitioned at N/2 granularity and split
  only into N_half / instrN = 64 / 16 = 4 finer slices.

**Tests:**
- `test/TritonGPU/amd/dot-partition-mn.mlir` — synthetic dot, check M then
  N partition gives M*N sub-dots.
- `test/TritonGPU/amd/dot-partition-n-half-already-sliced.mlir` — input has
  the v10-style dual `acc0`/`acc1` already, check the pass detects and
  splits only the inner 1/8 N.
- E2E: `bench.py --version 8 --K 4096`. Expected: numerically correct, perf
  again ±5 % of stock.

**Pass/fail gate:**
- 128 sub-dots in the inner-loop body for v8 (16 M × 8 N).
- 256 sub-dots for v10 fp16 (16 M × 8 N × 2 dual-accs).
- v8/v10 fp16/bf16/fp8 all numerically correct.

## Phase 3 — Schedule reorder + sched.barrier markers (~3-4 days)

Apply the actual scheduling recipe at TTGIR.

**Files to create:**

```
third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td
   → def SchedBarrierOp : ...        # marker op; lowers to llvm.amdgcn.sched.barrier
third_party/amd/lib/TritonAMDGPUToLLVM/
   → DotDecomposeAndScheduleLowering.cpp  # lower SchedBarrierOp
third_party/amd/lib/TritonAMDGPUTransforms/
   → ScheduleAnchorPattern.cpp      # the recipe (copied from LLIRSchedule.cpp:919)
```

**Recipe transposed to TTGIR:**

| LLIR anchor | TTGIR op |
|---|---|
| `@llvm.amdgcn.raw.ptr.buffer.load.async.lds` | `amdg::buffer_load_to_local` (GR) |
| `@llvm.amdgcn.ds.read.tr*` | `ttg::local_load` (LR) |
| store `addrspace(3)` | `ttg::local_store` (LW) |
| `@llvm.amdgcn.mfma.*` | `tt::DotOp` (after Phase-2 decomposition) |

**Driver pseudocode in `ScheduleAnchorPattern.cpp`:**

```cpp
struct Anchor { Operation *op; Kind kind; };  // GR / LR / LW / MFMA

LogicalResult applyScheduleRecipe(scf::ForOp forOp) {
  SmallVector<Anchor> anchors = collectAnchors(forOp.getBody());
  if (anchors.empty()) return success();

  // Apply the same 4-MFMA-per-GR / 1-MFMA-per-LR template
  // (mirror of LLIRSchedule.cpp:scheduleMFMAWithSpacing)
  reorderInBlock(anchors);
  insertSchedBarriers(anchors);  // amdg::sched_barrier between subregions
  return success();
}
```

The reorder itself walks `Operation::moveBefore` / `moveAfter` — MLIR's
SSA verifier guarantees we can't break dominance (will assert if we try).
That's the **structural safety** advantage over the LLIR pass.

**`amdg.sched_barrier` op lowering:**

```cpp
// DotDecomposeAndScheduleLowering.cpp
struct SchedBarrierOpLowering : public ConvertOpToLLVMPattern<SchedBarrierOp> {
  LogicalResult matchAndRewrite(SchedBarrierOp op, ...) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, /*resultType=*/{}, "llvm.amdgcn.sched.barrier",
        ValueRange{rewriter.create<LLVM::ConstantOp>(loc, 0)});
    return success();
  }
};
```

So we use the established intrinsic and the LLVM AMDGPU backend honors it
without needing to disable misched.

**Tests:**
- `test/TritonGPU/amd/dot-schedule-anchor-pattern.mlir` — input: decomposed
  inner loop with shuffled GR/LR/LW/MFMA order; check that the output has
  the expected recipe pattern (4 MFMA after each GR, 1 after each LR/LW),
  plus `amdg.sched_barrier` markers in the right places.
- `test/TritonAMDGPUToLLVM/sched-barrier-lowering.mlir` — `amdg.sched_barrier`
  → `llvm.amdgcn.sched.barrier(0)`.
- E2E: `bench.py --version 8 --K 4096` with TTGIR pass ON and LLIR pass OFF.
  **Expected: ≥ 80 % of v8 + SCHED+AS (≥ 1126 TF at K=4096).**
- E2E: same with both passes ON. **Expected: ≥ today's SCHED+AS (≥ 1408 TF
  at K=4096).**

**Pass/fail gate:**
- Lit test confirms scheduled IR matches expected recipe.
- v8 perf with TTGIR-only meets ≥ 80 % bar.
- v8 perf with TTGIR + LLIR matches or beats today's SCHED+AS.

**Risks:**
- **The TTGIR reorder is too coarse to leave room for LLIR pass.** If the
  TTGIR pass inserts `sched.barrier` between every sub-dot/LR pair, the
  LLIR pass has nothing left to interleave (the backend won't reorder
  across barriers). Mitigation: emit `sched.barrier` *only* at the
  boundaries of GR-LR-MFMA *regions*, not between every pair — so the
  backend (and the LLIR pass if still enabled) has freedom inside each
  region.
- **`amdg.sched_barrier` op not honored by the existing AMD lowering
  pipeline.** Some AMD-specific opt passes may DCE pure-marker ops with no
  results. Mitigation: mark with `MemoryEffects = ()` so it's not pure /
  not DCE-able; verify on small lit test before integration.

## Phase 4 — Coverage + safety net (~2-3 days)

Run the pass on every kernel that currently breaks LLIR SCHED and validate
the no-op-or-correct guarantee.

**Test matrix:**

| Workload | Expected | Validation |
|---|---|---|
| v10 (`gl_matmul.py`) | TTGIR pass applies, numerically correct, ≥ v10+SCHED perf | `python gl_matmul.py` with TTGIR=1, LLIR=0 |
| v8 (`bench.py --version 8`) | Same as v10 | `bench.py --version 8 --K 4096 --dtype fp16 bf16` |
| Stand-alone autotuned matmul (`_one_run_envcompare.py`) | `isInnerMatMulLoop` returns false → no-op; numerically correct | Run with TTGIR=1; expect same TFLOPS as stock |
| `triton_kernels.matmul` (BM256,BN256,W8) | Same — no-op | `_one_run.py '{"block_m":256,"block_n":256,"block_k":64,"num_warps":8}' 4096` |
| FA-fwd (`bench_triton_fa.py`) | `isInnerMatMulLoop` returns false (chained dots, layout mismatch) → no-op; numerically correct | Run with TTGIR=1 |
| 03-matrix-multiplication tutorial | No-op or apply correctly | Smoke test |

**Acceptance:** every workload either no-ops cleanly (same perf as stock)
or applies and produces numerically-correct + ≥-stock perf. **No crashes,
no wrong output.**

If any case produces wrong output, the `isInnerMatMulLoop` guard is too
permissive; tighten it.

## Phase 5 — Disable LLIR pass on v8/v10 by default (~1 day)

Once Phase 3 + 4 are green, the LLIR pass can stop being the default
gate for these kernels.

**Changes:**
- `compiler.py`: if `TRITON_ENABLE_TTGIR_SCHED=1`, automatically disable
  the LLIR pass even if `TRITON_ENABLE_LLIR_SCHED=1` is also set (avoid
  double-reordering producing degraded schedules).
- Add a knob `TRITON_KEEP_LLIR_CLEANUP=1` (default off) for users who
  want the LLIR pass as a downstream cleanup if measurement shows the
  TTGIR pass alone is missing some win.
- Add an integration check: `compile_v8_default_config` benchmark in CI
  that flags if TTGIR-only perf drops by > 5 %.

## Phase 6 — Cleanup + docs (~1 day)

- Update `~/AMD/triton/claude/matmul_4waves_summary.md` to list the new
  pass in the LLIR-scheduler theme (now superseded for v8/v10).
- Add a `~/AMD/triton/claude/llir_dump/v8_metamd/post_ttgir_sched.ll` IR
  dump alongside the existing `pre_llir.ll` / `post_llir.ll` showing the
  new pass output.
- Update `gl_matmul_passes_summary.md` headline numbers if the TTGIR pass
  delivers an additional win over LLIR pass.
- Cherry-pick `llir_sched_at_ttgir_design.md` + this plan into the same
  branch commit.

## Risk matrix (summary)

| Risk | Probability | Impact | Mitigation |
|---|:-:|:-:|---|
| Layout-system rejection of sub-tile encodings | M | H | Plan B: introduce `#mma_sliced` + ttg.convert_layout shim |
| Accumulator iter-arg explosion in scf.for | L | M | Use WSDataPartition's YieldOp handling verbatim |
| `sched.barrier` markers killed by AMD opt passes | L | M | Mark with side-effects; lit-test the lowering early |
| TTGIR reorder too coarse → can't match LLIR perf | M | M | Keep LLIR pass as opt-in downstream cleanup; phase-3 success bar is ≥ 80 % alone, parity with both on |
| v8 already-N-sliced halves get re-sliced wrong | M | H | Detect via `extract_slice` parent, split only inner factor |
| Pass triggers on workloads it shouldn't | M | H | Conservative `isInnerMatMulLoop` guard; phase-4 test matrix catches every known failure mode |
| Crash on FA-fwd (today's LLIR pass failure mode) | L | H | Guard refuses chained-dot patterns by default; verify in phase 4 |

## Out of scope (deferred)

- **K-split.** Documented as a Phase-7 follow-up only if measurement shows
  the LLIR pass cleanup is irreplaceable. Requires the `local_load`
  decomposition pass.
- **`local_load` per-vector decomposition.** Same Phase-7.
- **FA-fwd support.** The chained-dot pattern needs a different recipe
  (the softmax `extractelement → fmul → exp → MFMA` chain). Could be a
  Phase-8 with the same framework but a different per-recipe code path.
- **Replacing `amdgcnas` (the post-codegen ASM rewriter).** That's a
  separate, independent pass; SCHED-at-TTGIR doesn't touch it.

## Estimated total effort

| Phase | Days |
|---|---:|
| 0 Scaffold | 1 |
| 1 M-split | 3-5 |
| 2 N-split + already-sliced detection | 2-3 |
| 3 Schedule recipe + sched.barrier lowering | 3-4 |
| 4 Coverage matrix + safety net | 2-3 |
| 5 Default-disable LLIR on v8/v10 | 1 |
| 6 Docs + cleanup | 1 |
| **Total** | **13-18 working days** |

Single-developer estimate; assumes the developer is familiar with the
MLIR rewriter API and has built Triton from source. Add ~30 % buffer for
the layout-system risk if it materializes.

## Commit strategy

One commit per phase, following the matmul_4waves convention (C1/C2/C2.5
etc.):

- `D1` `[AMD][TTGIR-SCHED] Scaffold opt-in TRITON_ENABLE_TTGIR_SCHED pass (no-op)`
- `D2` `[AMD][TTGIR-SCHED] M-split via adapted WSDataPartition machinery`
- `D3` `[AMD][TTGIR-SCHED] N-split composed on top of M, detect already-sliced halves`
- `D4` `[AMD][TTGIR-SCHED] Apply 4-MFMA-per-GR scheduling recipe + amdg.sched_barrier`
- `D4.5` `[AMD][TTGIR-SCHED] Lower amdg.sched_barrier to llvm.amdgcn.sched.barrier(0)`
- `D5` `[AMD][TTGIR-SCHED] Coverage matrix: no-op guards for FA / triton_kernels / stand-alone`
- `D6` `[AMD][TTGIR-SCHED] Default-disable LLIR pass when TTGIR-SCHED active on v8/v10`

Each commit independently buildable and testable; Phase-1 onward keeps the
e2e perf within ±5 % of stock until Phase-3 turns on the real win.

## Reference doc tree (for the implementer)

```
~/AMD/triton/claude/
├── matmul_4waves_summary.md           # branch overview; what the LLIR pass does
├── gl_matmul_passes_summary.md        # current perf numbers (target to beat)
├── llir_sched_at_ttgir_design.md      # the design rationale (parent doc)
├── llir_sched_at_ttgir_plan.md        # ← this file
├── llir_dump/
│   ├── v8_metamd/                     # actual pre/post LLIR IR for v8 + SCHED
│   └── fa_fwd/                        # the FA-fwd crash analysis
├── kernels/
│   ├── v10_4wave_matmul.py            # the v10 design target
│   └── v8_beyond_hotloop_matmul.py    # the v8 reference
└── triton_kernels_baseline/
    └── bench_env_compare.py           # the harness for the coverage matrix in Phase 4

~/MetaMain/triton/
├── third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp
│       # the partitioner framework to adapt (Phase 1)
├── third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp
│       # AMD shmem-subview machinery to borrow (Phase 1)
└── lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp
        # cleanest MLIR splitter reference (Phase 1)

~/AMD/triton/third_party/amd/lib/TritonAMDGPUToLLVM/LLIRSchedule.cpp
        # the recipe code to translate to TTGIR (Phase 3)
```
