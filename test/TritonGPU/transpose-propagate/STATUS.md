# Transpose Propagate Pass — D0–D8 Status

This directory contains the lit tests for the new algebraic transpose
propagation pass (TritonGPUTransposePropagate). Tests are organised by
phase commit, each covering one rule or engine feature.

## Status (post-D8 — overnight execution 2026-05-30)

**D0 – D8 complete and committed.** All 9 lit tests pass. Full TritonGPU
lit suite unchanged (no regressions on the 117-test baseline).

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

## Architecture as of D8

Three pure phases per the master plan:

1. **plan** — `planTransposePropagation(funcOp)` walks every annotated
   `tt.dot` (root), does forward DFS through use chains classifying each
   op via the rule registry. Conservative fallthrough: any op without a
   rule becomes `BoundaryInsert`. Returns a `TransposePlan` with
   `ops`, `boundaryOps`, optionally a `rejected()` marker.
   - Special handling: `SCFCarryRetype` extends the DFS to the parent
     scf.for's iter_arg at the corresponding yield position.
   - `TransElide` / `BoundaryInsert` / `SharedMemBoundary` do not recurse.

2. **score** — stub (always feasible when plan is non-empty). The real
   cost-model belongs in D9+ alongside dry-run validation.

3. **commit** — no-op. **This is the largest remaining piece of work.**

## What's NOT yet implemented (D9 – D12 remaining)

| Phase | Description | Estimated LoC | Complexity |
|-------|-------------|---------------|------------|
| D9    | Dry-run validation: clone FuncOp, simulate commit, verify, discard | ~100 | high (needs commit working first) |
| D10   | Cleanup-on-failure tracking + rollback path | ~50 | medium |
| D11   | FA-fwd end-to-end lit + bench  | ~50 | depends on D5-D8 commits |
| D12   | FA-bwd end-to-end lit (matches user's example) | ~50 | depends on D5-D8 commits |

### The commit-engine gap

D5 (DotFlip), D6 (ConvertLayoutAdjust), D7 (SharedMemBoundary), and D8
(SCFCarryRetype) all have **classification-only** rules. Their
`transform` factories are stubs returning nullptr. The commit engine
that actually invokes the factories and mutates IR is not yet written.

The commit engine needs to:

1. **Topo-sort** plan.ops + the implied "produce transposed versions of
   the iter_arg uses" from the extended DFS.
2. For each plan op, call its rule's `transform` factory with the
   already-transposed operands (looked up from a `seedMap` of
   `original_value -> transposed_value`).
3. For each boundary site `(consumer, opIdx)`, insert a `tt.trans` op
   before the consumer that converts the in-closure value back to
   original orientation.
4. **DotFlip specifically**: insert `tt.trans` on the non-in-closure
   operands (e.g., when DotFlip fires on opIdx=2, opA and opB need
   tt.trans wrapping), then build the new dot with swapped operand
   roles per the identity `out^T = (A·B + C)^T = B^T · A^T + C^T`.
5. **SCFCarryRetype**: rewrite the scf.for op's iter_arg types + init
   value types (with tt.trans on the init), update yield types, and
   handle the loop result's outside-loop users with a boundary trans.
6. Track every created op in a `createdOps` list for cleanup if
   anything fails.

### Why we stopped at D8

D5-D8 commits intentionally landed as classification-only because the
**commit engine itself is a substantial atomic unit of work** (touches
every rule, requires careful scf.for IR mutation, requires the dry-run
infrastructure to be safe to land). Splitting it across multiple D9-D12
commits creates an "always-broken" interim state.

The right next step is either:

* **One large D9 commit** that lands the commit engine wholesale + all
  rule factories realised + dry-run + cleanup-on-failure. ~500-700
  LoC, one atomic logical unit.
* **A simpler "smoke" D9** that handles only the simplest rules
  (Rewrite, ReduceAxisSwap, ExpandDimsSwap, BroadcastSwap, TransElide
  + boundary insertion) and explicitly **doesn't run** when the plan
  contains DotFlip or SCFCarryRetype (those defer to a later commit).
  ~200-300 LoC.

The simpler smoke path is consistent with the incremental-commit
discipline of the plan but doesn't unblock FA-fwd / FA-bwd e2e (since
both kernels require DotFlip).

## How to verify what's built

```bash
cd /home/mren/MetaMain/triton
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flydsl
export LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/
export PATH=/data/users/mren/miniconda3/bin:$PATH
export PYTHONPATH=/data/users/mren/miniconda3/lib/python3.11/site-packages:$PYTHONPATH

# Build
/usr/bin/ninja -C build/cmake.linux-x86_64-cpython-3.11 triton-opt

# Run the transpose-propagate lit suite (9 tests)
/home/mren/OpenSource/llvm-build/bin/llvm-lit -v \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/transpose-propagate/

# Manual remark check on any test:
BINDIR=$(realpath build/cmake.linux-x86_64-cpython-3.11/bin)
TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 $BINDIR/triton-opt \
  test/TritonGPU/transpose-propagate/d8-scf-yield.mlir \
  --tritongpu-transpose-propagate
```

Expected: `transpose-propagate: plan roots=N ops=N boundary=N` remark
emitted per FuncOp with at least one annotated root, no IR mutation
(commit is still a no-op).

## Relationship to ~/AMD/triton C0-C19

C0-C19 in ~/AMD/triton is a learning artifact that implements
*layout-encoding propagation* (warpsPerCTA swap) — a different
abstraction. The user's perf-signal experiment with C19 confirmed
this isn't what FlyDSL does. This pass (in ~/MetaMain/triton) is the
*algebraic transpose* propagation that matches the original master
plan and the FlyDSL transformation.

See `~/.llms/plans/amd_transpose_propagate_v2.plan.md` for the full
plan + rationale for the restart.

## Branch

`transpose-propagate-v2` in `/home/mren/MetaMain/triton`. 9 commits
ahead of `main`.
