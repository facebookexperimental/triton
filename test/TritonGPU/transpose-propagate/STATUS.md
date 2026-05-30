# Transpose Propagate Pass — D0–D16 Status (overnight execution complete)

This directory contains the lit tests for the new algebraic transpose
propagation pass (TritonGPUTransposePropagate). Tests are organised by
phase commit, each covering one rule or engine feature.

## Status (post-D16 — overnight execution 2026-05-30)

**D0 – D16 complete and committed.** All 14 lit tests pass. Full
TritonGPU lit suite: 119/130 passed (was 106/117 baseline; +13 net
new tests, no regressions). **20 commits in branch.**

| Phase | Commit  | Status | Coverage |
|-------|---------|--------|----------|
| D0    | `04362271f` | ✅ | Scaffolding: header, cpp stubs, pass reg, lit stub |
| D1    | `f816db167` | ✅ | Plan engine + DFS + conservative-default BoundaryInsert |
| D2    | `03562aaa9` | ✅ | Rule: elementwise (arith + math) |
| D3    | `a21248be0` | ✅ | Rules: ReduceAxisSwap, ExpandDimsSwap, BroadcastSwap |
| D4    | `9af420b1f` | ✅ | Rule: TransElide |
| D5    | `695b7a9ce` | ✅ | Rule: DotFlip (classification) |
| D6    | `6c96f5895` | ✅ | Rule: ConvertLayoutAdjust (classification) |
| D7    | `98959f575` | ✅ | Rule: SharedMemBoundary (explicit classification) |
| D8    | `d8ba73e24` | ✅ | Rule: SCFCarryRetype + iter_arg DFS extension |
| D8.1  | `941eca3d7` | ✅ | STATUS.md initial doc |
| D9    | `5526cdd9d` | ✅ | Commit engine: lazy-root + simple-rule materialisation + rollback |
| D9.1  | `cff30f3d9` | ✅ | STATUS.md update for D9 |
| D10   | `d4e4225dc` | ✅ | Real DotFlip transform (encoding-aware + DotOperand convert insertion) |
| D11   | `79e6681c8` | ✅ | SCFCarryRetype transform (wrap approach) |
| D12   | `3238e6fc7` | ✅ | Dry-run validation wrapper |
| D13   | `8d7d88f7d` | ✅ | Synthetic FA-fwd-shape e2e lit test |
| D14   | `b51f1d001` | ✅ | Multi-root FA-bwd-shape e2e lit test |
| D14.1 | `535547b72` | ✅ | STATUS.md update for D9-D14 |
| D15   | `1552874b0` | ✅ | AMD pipeline wire-in (env-gated) |
| D16   | `934a6fc00` | ✅ | Auto-annotate-first-dot env knob |

## End-to-end usage (post-D16)

The pass is now fully wired and runnable on real kernels. Two env vars:

```bash
# Enable the pass in the AMD pipeline (D15):
export TRITON_TRANSPOSE_PROPAGATE_ENABLE=1

# Optional: auto-annotate the first dot in each FuncOp (D16):
export TRITON_TRANSPOSE_PROPAGATE_AUTOANNOTATE_FIRST_DOT=1

# Optional: verbose remark output (debug):
export TRITON_TRANSPOSE_PROPAGATE_DEBUG=1

# Now run any AMD-compiled kernel:
python python/tutorials/06-fused-attention.py
```

The pass:
1. Walks each FuncOp; if AUTOANNOTATE_FIRST_DOT is set and no dot is
   annotated, annotates the first tt.dot.
2. Plans the propagation closure from each annotated root.
3. Dry-runs the commit on a cloned funcOp + verifies. If verify
   fails, skips commit (no IR mutation).
4. Commits: rewrites IR in-place per the rule registry.
5. Strips the annotation for idempotence.

## Capabilities at D16

### Plan phase
- 9 rule kinds, conservative BoundaryInsert fallback.
- Multi-root per FuncOp.
- Extended DFS through scf.for iter_args.

### Commit phase
- All transform factories real: elementwise (with encoding coercion),
  axis-swap, trans-elide, DotFlip (encoding-aware + DotOperand
  convert insertion), convert-layout, SCFCarryRetype (wrap approach).
- Lazy-root pattern: tt.trans inserted on root.result, original kept.
- Topo-sorted materialisation.
- Boundary back-trans at every site.
- Cleanup-on-failure + success-path dead-trans cleanup.

### Safety
- Dry-run wrapper: clone + verify before real commit.
- Per-transform nullptr detection rolls back the whole commit.

### Integration
- AMD pipeline: env-gated invocation between remove_layout_conversions
  and accelerate_matmul.
- Python binding: `passes.ttgpuir.add_transpose_propagate(pm)`.
- Auto-annotation knob for zero-source-change experiments.

## End-to-end IR example (D13 FA-fwd-shape)

```mlir
// Input (annotated QK):
scf.for ... iter_args(%acc = %init) -> #blocked {
  %qk = tt.dot %q, %k, %zero {tt.transpose_propagate_root} : ... → #blocked
  %scaled = arith.mulf %qk, %sm_scale
  %p_f32 = math.exp2 %scaled
  %p_f16 = arith.truncf %p_f32 to fp16
  %p = ttg.convert_layout %p_f16 → DotOp<0, #blocked>
  %pv = tt.dot %p, %v, %acc
  scf.yield %pv
}

// Output (post-pass):
scf.for ... iter_args(%acc = %init) -> #blocked {
  %1 = tt.trans %acc → #blocked1                       (SCFCarryRetype prep)
  %2 = tt.dot %q, %k, %zero                            (QK root, unchanged)
  %3 = tt.trans %2 → #blocked1                          (lazy root trans)
  %4 = tt.trans %sm_scale → #blocked1
  %5 = arith.mulf %3, %4 → #blocked1                    (transposed mulf)
  %6 = math.exp2 %5 → #blocked1
  %7 = arith.truncf %6 → fp16
  %8 = ttg.convert_layout %7 → DotOp<0, #blocked>       (encoding adjust)
  %9 = tt.trans %v → #linear                            (DotFlip prep)
  %10 = ttg.convert_layout %9 → DotOp<0, #blocked1>     (new A of PV)
  %11 = ttg.convert_layout %8 → DotOp<1, #blocked1>     (new B of PV)
  %12 = tt.dot %10, %11, %1 → #blocked1                 (flipped PV)
  %13 = tt.trans %12 → #blocked                         (back-trans for yield)
  scf.yield %13
}
```

Math correctness preserved by construction. Verifier passes.

## How to verify

```bash
cd /home/mren/MetaMain/triton
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flydsl
export LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/
export PATH=/data/users/mren/miniconda3/bin:$PATH
export PYTHONPATH=/data/users/mren/miniconda3/lib/python3.11/site-packages:$PYTHONPATH

# Build (incremental)
/usr/bin/ninja -C build/cmake.linux-x86_64-cpython-3.11 triton triton-opt

# Run transpose-propagate lit suite (14 tests, all PASS)
/home/mren/OpenSource/llvm-build/bin/llvm-lit -v \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/transpose-propagate/

# Run full TritonGPU suite (119/130 PASS, no regressions)
/home/mren/OpenSource/llvm-build/bin/llvm-lit \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/

# Inspect commit on the FA-fwd-shape e2e test:
BINDIR=$(realpath build/cmake.linux-x86_64-cpython-3.11/bin)
$BINDIR/triton-opt \
  test/TritonGPU/transpose-propagate/d13-fa-fwd-shape.mlir \
  --tritongpu-transpose-propagate
```

## What's NOT yet done (deferred)

| Item | Description |
|---|---|
| Real SCFCarryRetype | Rewrite scf.for signature (transposed iter_args, init values). D11 uses "wrap" with extra tt.trans pair per iter -- correct but perf-wasteful. |
| Real cost model in Score | Currently feasibility-only stub. Should weight DotFlip × MFMA throughput minus boundary trans cost minus register pressure. |
| Validated perf signal on real FA-fwd | D15 + D16 wire it up, but no bench has been run. The user said don't run perf tests without explicit ask. |
| Auto-annotation heuristic beyond "first dot" | More sophisticated: pick dots where the chain has many shared layouts, where the post-flip MFMA shape unlocks a better instruction, etc. |
| HSTU bwd full chain test | The user's FA-bwd example has 5 chained dots with branching; D14 tests multi-root but not the full chain. |

## Architecture summary

```
Driver (TransposePropagatePass.cpp):
  auto-annotate? → Plan → Score → DryRun → Commit
  (D16)            (D1-D8) (stub) (D12)   (D9-D11)
                     ↓                       ↓
                   rule registry          rule transform factories
                   (9 kinds)              + boundary trans
                                          + SCFCarryRetype wrap
                                          + cleanup-on-failure
                                          + annotation strip
```

## Branch

`transpose-propagate-v2` in `/home/mren/MetaMain/triton`. **20 commits ahead of `main`**.
Build via flydsl env + LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/.

## File map

| File | Purpose |
|---|---|
| `include/triton/Dialect/TritonGPU/Transforms/TransposePropagate.h` | Public API: TransposePlan, TransposeRule, plan/score/commit/dryRun |
| `lib/Dialect/TritonGPU/Transforms/TransposePropagate.cpp` | Engine: rule registry, DFS plan, commit pipeline, dry-run |
| `lib/Dialect/TritonGPU/Transforms/TransposePropagatePass.cpp` | Pass driver: auto-annotation, plan→score→dry-run→commit |
| `include/triton/Dialect/TritonGPU/Transforms/Passes.td` | Pass registration |
| `include/triton/Dialect/TritonGPU/IR/Dialect.h` | kTransposePropagateRootAttrName constant |
| `lib/Dialect/TritonGPU/Transforms/CMakeLists.txt` | Build wiring |
| `python/src/passes.cc` | Python binding (`add_transpose_propagate`) |
| `third_party/amd/backend/compiler.py` | AMD pipeline invocation (env-gated) |
| `test/TritonGPU/transpose-propagate/*.mlir` | 14 lit tests + STATUS.md |

| Phase | Commit  | Status | Coverage |
|-------|---------|--------|----------|
| D0    | `04362271f` | ✅ | Scaffolding: header, cpp stubs, pass reg, lit stub |
| D1    | `f816db167` | ✅ | Plan engine + DFS + conservative-default BoundaryInsert |
| D2    | `03562aaa9` | ✅ | Rule: elementwise (arith + math) |
| D3    | `a21248be0` | ✅ | Rules: ReduceAxisSwap, ExpandDimsSwap, BroadcastSwap |
| D4    | `9af420b1f` | ✅ | Rule: TransElide |
| D5    | `695b7a9ce` | ✅ | Rule: DotFlip (classification) |
| D6    | `6c96f5895` | ✅ | Rule: ConvertLayoutAdjust (classification) |
| D7    | `98959f575` | ✅ | Rule: SharedMemBoundary (explicit classification) |
| D8    | `d8ba73e24` | ✅ | Rule: SCFCarryRetype + iter_arg DFS extension |
| D8.1  | `941eca3d7` | ✅ | STATUS.md initial doc |
| D9    | `5526cdd9d` | ✅ | Commit engine: lazy-root + simple-rule materialisation + rollback |
| D9.1  | `cff30f3d9` | ✅ | STATUS.md update for D9 |
| D10   | `d4e4225dc` | ✅ | Real DotFlip transform (encoding-aware + DotOperand convert insertion) |
| D11   | `79e6681c8` | ✅ | SCFCarryRetype transform (wrap approach) |
| D12   | `3238e6fc7` | ✅ | Dry-run validation wrapper |
| D13   | `8d7d88f7d` | ✅ | Synthetic FA-fwd-shape e2e lit test |
| D14   | `b51f1d001` | ✅ | Multi-root FA-bwd-shape e2e lit test |

## Capabilities at D14

### Plan phase (D0–D8)
- 9 rule kinds classified: Rewrite, ReduceAxisSwap, ExpandDimsSwap,
  BroadcastSwap, TransElide, DotFlip, ConvertLayoutAdjust,
  SharedMemBoundary, SCFCarryRetype.
- Conservative-default BoundaryInsert for unrecognised ops.
- Extended DFS for SCFCarryRetype (walks through scf.for iter_args).
- Multiple annotated roots per FuncOp.

### Commit phase (D9–D11)
- Lazy-root pattern: tt.trans inserted on root.result, root op kept.
- Real transforms: elementwise (with auto trans+convert on
  non-in-closure operands), ReduceAxisSwap, ExpandDimsSwap,
  BroadcastSwap, TransElide, DotFlip (encoding-aware), ConvertLayoutAdjust.
- SCFCarryRetype "wrap" approach: trans on iter_arg at body top,
  back-trans before yield. scf.for signature unchanged.
- Boundary back-trans at every (consumer, opIdx) site.
- Topo-sorted materialisation via mlir::topologicalSort.
- Cleanup-on-failure + success-path dead-trans cleanup.
- Annotation stripping for idempotence.

### Safety (D12)
- Dry-run wrapper: clone funcOp into temp module (with original
  module's attributes mirrored), re-plan + commit on clone, verify,
  discard. Only when verify passes does the real commit fire.

### End-to-end tests (D13, D14)
- D13: FA-fwd-shape kernel (scf.for + QK + softmax + PV + SCFCarryRetype).
  Commits cleanly, produces well-formed IR.
- D14: multi-root FA-bwd-shape (two annotated dots, independent
  closures). Validates the multi-root code path.

## End-to-end IR example (D13 FA-fwd-shape)

```mlir
// Input:
scf.for ... iter_args(%acc = %init) -> #blocked {
  %qk = tt.dot %q, %k, %zero {tt.transpose_propagate_root} : ... → #blocked
  %scaled = arith.mulf %qk, %sm_scale
  %p_f32 = math.exp2 %scaled
  %p_f16 = arith.truncf %p_f32 to fp16
  %p = ttg.convert_layout %p_f16 → DotOp<0, #blocked>
  %pv = tt.dot %p, %v, %acc
  scf.yield %pv
}

// Output (D9+D10+D11):
scf.for ... iter_args(%acc = %init) -> #blocked {
  %1 = tt.trans %acc → #blocked1                       (SCFCarryRetype prep)
  %2 = tt.dot %q, %k, %zero                            (QK root, unchanged)
  %3 = tt.trans %2 → #blocked1                          (lazy root trans)
  %4 = tt.trans %sm_scale → #blocked1
  %5 = arith.mulf %3, %4 → #blocked1                    (transposed mulf)
  %6 = math.exp2 %5 → #blocked1
  %7 = arith.truncf %6 → fp16
  %8 = ttg.convert_layout %7 → DotOp<0, #blocked>       (encoding adjust)
  %9 = tt.trans %v → #linear                            (DotFlip prep)
  %10 = ttg.convert_layout %9 → DotOp<0, #blocked1>     (new A of PV)
  %11 = ttg.convert_layout %8 → DotOp<1, #blocked1>     (new B of PV)
  %12 = tt.dot %10, %11, %1 → #blocked1                 (flipped PV)
  %13 = tt.trans %12 → #blocked                         (back-trans for yield)
  scf.yield %13
}
```

Math correctness:
- `trans(mulf(trans(qk), trans(sm))) = mulf(qk, sm)` — softmax math.
- `trans(dot(trans(V), trans(P), trans(acc))) = dot(P, V, acc)` — PV update.
- iter_arg input `acc` (#blocked), output `yield(back-trans)` (#blocked) — type consistent.

## How to verify

```bash
cd /home/mren/MetaMain/triton
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flydsl
export LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/
export PATH=/data/users/mren/miniconda3/bin:$PATH
export PYTHONPATH=/data/users/mren/miniconda3/lib/python3.11/site-packages:$PYTHONPATH

# Build
/usr/bin/ninja -C build/cmake.linux-x86_64-cpython-3.11 triton-opt

# Run transpose-propagate lit suite (14 tests, all PASS)
/home/mren/OpenSource/llvm-build/bin/llvm-lit -v \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/transpose-propagate/

# Run full TritonGPU suite (119/130 PASS, no regressions)
/home/mren/OpenSource/llvm-build/bin/llvm-lit \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/

# Inspect commit output on the FA-fwd-shape test:
BINDIR=$(realpath build/cmake.linux-x86_64-cpython-3.11/bin)
$BINDIR/triton-opt \
  test/TritonGPU/transpose-propagate/d13-fa-fwd-shape.mlir \
  --tritongpu-transpose-propagate
```

## What's NOT yet done (deferred)

| Item | Description |
|---|---|
| Real SCFCarryRetype | Rewrite scf.for signature (transposed iter_args, init values). D11 uses "wrap" approach with extra tt.trans pair per iter, which is correct but wastes perf. |
| Real cost model in Score | Currently feasibility-only stub. Should weight DotFlip × MFMA-instruction-throughput minus boundary trans cost minus register pressure. |
| Real FA-fwd from 06-fused-attention.py | D13 is synthetic. Running the real Python kernel through compile+exec needs more work to handle K/V local_alloc/load patterns, multi-iter-arg loops (m_i, l_i, acc), causal mask, etc. The infrastructure handles these (or rolls back via dry-run), but no validated end-to-end perf number yet. |
| Auto-annotation heuristic | Currently annotation-driven only. A future pass could auto-annotate roots based on chain-dot patterns + target info. |
| AMD pipeline wire-in | Pass exists in the registry but isn't invoked by `third_party/amd/backend/compiler.py`. Adding it (gated on env var) is a few lines once everything else is verified. |

## Architecture summary

```
Annotation  →  Plan (legality)  →  Score (profitability)  →  DryRun (safety)  →  Commit (transform)
    │                │                       │                       │                     │
  tt.dot       forward DFS              feasibility           clone+verify          topo-sort
  +attr        + rule registry          stub (D9-D12)         (D12)                 + factories
                                                                                    + boundary trans
                                                                                    + SCFCarryRetype
                                                                                    + cleanup-on-failure
                                                                                    + annotation strip
```

## Branch

`transpose-propagate-v2` in `/home/mren/MetaMain/triton`. **17 commits ahead of `main`**.
Build via flydsl env + LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/.

## File map

| File | Purpose |
|---|---|
| `include/triton/Dialect/TritonGPU/Transforms/TransposePropagate.h` | Public API: TransposePlan, TransposeRule, rule kinds enum, plan/score/commit/dryRun |
| `lib/Dialect/TritonGPU/Transforms/TransposePropagate.cpp` | Engine: rule registry, DFS plan, commit pipeline, dry-run wrapper |
| `lib/Dialect/TritonGPU/Transforms/TransposePropagatePass.cpp` | Pass driver: walks FuncOps, plan→score→dry-run→commit |
| `include/triton/Dialect/TritonGPU/Transforms/Passes.td` | Pass registration |
| `include/triton/Dialect/TritonGPU/IR/Dialect.h` | kTransposePropagateRootAttrName constant |
| `lib/Dialect/TritonGPU/Transforms/CMakeLists.txt` | Build wiring |
| `test/TritonGPU/transpose-propagate/*.mlir` | 14 lit tests (1 per phase + STATUS.md) |
