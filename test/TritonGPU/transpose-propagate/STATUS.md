# Transpose Propagate Pass — D0–D9 Status

This directory contains the lit tests for the new algebraic transpose
propagation pass (TritonGPUTransposePropagate). Tests are organised by
phase commit, each covering one rule or engine feature.

## Status (post-D9 — overnight execution 2026-05-30)

**D0 – D9 complete and committed.** All 10 lit tests pass. Full
TritonGPU lit suite: 115/126 passed (was 106/117 baseline; +9 net
new tests, no regressions).

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
| **D9**| **`5526cdd9d`** | ✅ | **Commit engine: lazy-root + simple-rule materialisation + rollback** |

## D9 deliverables

**The commit engine fires!** For plans that contain only simple rules
(Rewrite, ReduceAxisSwap, ExpandDimsSwap, BroadcastSwap, TransElide,
ConvertLayoutAdjust, BoundaryInsert), the pass produces well-formed
IR end-to-end:

```mlir
// Input:
%dot = tt.dot %a, %b, %c {tt.transpose_propagate_root} : ... -> tensor<16x16xf32, #blocked>
%p   = arith.truncf %dot : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
tt.return %p

// Output (committed by D9):
%0 = tt.dot %arg0, %arg1, %arg2 : ...     // annotation stripped
%1 = tt.trans %0 → #blocked1              // lazy root trans
%2 = arith.truncf %1 → tensor<16x16xf16, #blocked1>   // rewritten in transposed orientation
%3 = tt.trans %2 → #blocked               // boundary back-trans
tt.return %3                              // original type preserved
```

Math correctness: `trans(truncf(trans(dot))) == truncf(dot)` since
truncf is elementwise (commutes with transpose).

### What D9 commits successfully

- **Lazy root**: each annotated `tt.dot` gets a `tt.trans` inserted
  right after it. Root itself stays unchanged (no algebraic flip).
- **Elementwise / ReduceAxisSwap / ExpandDimsSwap / BroadcastSwap**:
  full transform factories. Operands not in the closure get wrapped
  with `tt.trans` + `ttg.convert_layout` to match the target encoding.
- **TransElide**: `tt.trans` of an in-closure value returns the
  operand verbatim, the trans op is deleted.
- **ConvertLayoutAdjust**: builds a new convert with shape-swapped
  destination type.
- **Boundary back-trans**: every `(consumer, opIdx)` in
  `plan.boundaryOps` gets a `tt.trans` inserted to bring the in-
  closure value back to original orientation.
- **Cleanup-on-failure**: every created op is tracked. If any
  transform returns nullptr (or aborts otherwise), the rollback
  path erases all `use_empty()` new ops in reverse order, leaving
  the IR unchanged.
- **Annotation strip**: `tt.transpose_propagate_root` removed from
  roots after a successful commit (idempotence).

### What D9 cleanly rolls back (transform returns nullptr)

- **DotFlip** (downstream dots): the new dot's result encoding must
  match the new accumulator C's encoding, but A/B need to be
  `DotOperand<...parent=newCEnc>`. Building the convert chain
  consistently is a follow-up commit.
- **SCFCarryRetype**: scf.for signature mutation is non-trivial
  (rewrite iter_arg types, insert trans on init values, update
  yield, handle outside-loop result users). Follow-up commit.
- **SharedMemBoundary**: `transform=nullptr` in the registry; if
  a plan reaches one, rolls back (rare in test patterns).

### What D9 does NOT have (deferred)

- **Real dry-run validation**: a clone-funcOp / verify / discard
  wrapper. Current safety comes from per-transform `nullptr`
  detection + `use_empty()`-guarded rollback. A future commit
  should add a true clone-based dry-run to catch cross-op
  verifier issues (e.g., scf.for body type mismatches) before
  any IR mutation lands.

## How to verify

```bash
cd /home/mren/MetaMain/triton
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flydsl
export LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/
export PATH=/data/users/mren/miniconda3/bin:$PATH
export PYTHONPATH=/data/users/mren/miniconda3/lib/python3.11/site-packages:$PYTHONPATH

# Build
/usr/bin/ninja -C build/cmake.linux-x86_64-cpython-3.11 triton-opt

# Run the transpose-propagate lit suite (10 tests, all PASS)
/home/mren/OpenSource/llvm-build/bin/llvm-lit -v \
  build/cmake.linux-x86_64-cpython-3.11/test/TritonGPU/transpose-propagate/

# Inspect commit output on a simple chain:
BINDIR=$(realpath build/cmake.linux-x86_64-cpython-3.11/bin)
$BINDIR/triton-opt \
  test/TritonGPU/transpose-propagate/d9-commit-simple.mlir \
  --tritongpu-transpose-propagate
```

## Remaining work (D10 – D12)

| Phase | Description | Estimated LoC |
|-------|-------------|---------------|
| D10   | Real DotFlip transform: encoding-aware result + DotOperand convert insertion on swapped A/B | ~150 |
| D11   | Real SCFCarryRetype transform: scf.for signature mutation | ~150 |
| D12   | Dry-run validation wrapper (clone funcOp, verify, discard) | ~80 |
| D13   | FA-fwd end-to-end lit + bench (needs D10 + D11 + D12) | ~50 |
| D14   | FA-bwd e2e lit (HSTU-style, needs D10) | ~50 |

D9 unblocks the simple-rule perf experiments. D10-D14 unblock the
real FA kernel transformations.

## Branch

`transpose-propagate-v2` in `/home/mren/MetaMain/triton`. 12 commits
ahead of `main`. Build via flydsl env + LLVM_BUILD_DIR=/home/mren/OpenSource/llvm-build/.
