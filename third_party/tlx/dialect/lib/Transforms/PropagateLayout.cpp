#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/DataFlowFramework.h"
#define DEBUG_TYPE "tlx-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXPROPAGATELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

class RequireLayoutPattern : public mlir::OpRewritePattern<RequireLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RequireLayoutOp requireLayoutOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(requireLayoutOp.getSrc().getType()))
      return failure();
    auto convertLayoutOp = rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        requireLayoutOp, requireLayoutOp.getType(), requireLayoutOp.getSrc());
    return success();
  }
};

class ReleaseLayoutPattern : public mlir::OpRewritePattern<ReleaseLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReleaseLayoutOp releaseLayoutOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto convertLayoutOp = rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        releaseLayoutOp, releaseLayoutOp.getType(), releaseLayoutOp.getSrc());
    return success();
  }
};

class TlxPropagateLayoutPass
    : public impl::TlxPropagateLayoutBase<TlxPropagateLayoutPass> {
public:
  using impl::TlxPropagateLayoutBase<
      TlxPropagateLayoutPass>::TlxPropagateLayoutBase;

  // Cancel multibuffering for TMEM allocations that will have scales encoding.
  // Scales don't support multibuffering, so 3D shapes (1xMxK) must be flattened
  // to 2D (MxK). This should be called before layout propagation so that the
  // types are already flattened when layouts are applied.
  void cancelMultibufferingForScales(triton::FuncOp funcOp,
                                     DataFlowSolver &solver) {
    DenseMap<Value, ttg::MemDescType> allocsToFlatten;

    // First pass: Collect TMEMAllocOps that will have scales encoding
    // (determined by dataflow analysis) and need multibuffering cancelled
    funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
      auto origType = allocOp.getType();
      auto shape = origType.getShape();
      if (shape.size() != 3 || shape[0] != 1)
        return WalkResult::advance();

      // Check if this alloc will receive scales encoding via layout
      // propagation
      auto *lattice =
          solver.lookupState<LayoutEncodingLattice>(allocOp.getResult());
      if (!lattice || lattice->getValue().isUninitialized())
        return WalkResult::advance();

      auto encoding = lattice->getValue().getLayoutEncoding();
      if (!isa<ttng::TensorMemoryScalesEncodingAttr>(encoding))
        return WalkResult::advance();

      // This is a 3D alloc with scales encoding - needs flattening
      // Cancel multibuffering: flatten 1xMxK to MxK
      SmallVector<int64_t> newShape{shape[1], shape[2]};
      auto newType = ttg::MemDescType::get(
          newShape, origType.getElementType(), origType.getEncoding(),
          origType.getMemorySpace(), origType.getMutableMemory());
      allocsToFlatten[allocOp.getResult()] = newType;
      return WalkResult::advance();
    });

    // Second pass: Update memdesc_index ops that index into flattened allocs
    // These ops should be replaced with the alloc directly since we're
    // canceling multibuffering
    SmallVector<ttg::MemDescIndexOp> indexOpsToRemove;
    for (auto &[allocValue, newType] : allocsToFlatten) {
      for (auto &use : allocValue.getUses()) {
        if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(use.getOwner())) {
          // The index op is accessing the buffering dimension which we're
          // removing. Replace all uses of the index op with the flattened
          // alloc.
          indexOp.getResult().replaceAllUsesWith(allocValue);
          indexOpsToRemove.push_back(indexOp);
        }
      }
    }

    // Erase the index ops
    for (auto indexOp : indexOpsToRemove) {
      indexOp.erase();
    }

    // Third pass: Update the alloc types
    for (auto &[allocValue, newType] : allocsToFlatten) {
      allocValue.setType(newType);
    }
  }

  void runOnFuncOp(triton::FuncOp funcOp) {
    // We can terminate early if we don't have a layout constraint.
    WalkResult walkResult = funcOp.walk([&](mlir::Operation *op) {
      if (auto requireLayoutOp = dyn_cast<tlx::RequireLayoutOp>(op))
        if (isa<gpu::MemDescType>(requireLayoutOp.getType()))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (!walkResult.wasInterrupted())
      return;

    PatternRewriter rewriter(&getContext());
    SymbolTableCollection symbolTable;
    Operation *op = getOperation();
    DataFlowSolver solver;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LayoutBackwardPropagation>(symbolTable);
    solver.load<LayoutForwardPropagation>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Cancel multibuffering for scales allocations before applying layouts.
    // This flattens 3D (1xMxK) to 2D (MxK) and removes memdesc_index ops.
    cancelMultibufferingForScales(funcOp, solver);

    auto getNewMemDescType = [&](ttg::MemDescType origType,
                                 Attribute encoding) {
      return ttg::MemDescType::get(
          origType.getShape(), origType.getElementType(), encoding,
          origType.getMemorySpace(), origType.getMutableMemory());
    };

    funcOp.walk([&](mlir::Operation *op) {
      if (isa<tlx::RequireLayoutOp>(op))
        return WalkResult::advance();

      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        Region *firstRegion = wsOp.getPartitionRegions()[0];
        for (auto [i, blockArg] :
             llvm::enumerate(firstRegion->getArguments())) {
          if (!isa<ttg::MemDescType>(blockArg.getType()))
            continue;
          auto lattice = solver.lookupState<LayoutEncodingLattice>(blockArg);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          if (lattice->getValue().isUninitialized())
            continue;
          for (Region *partitionRegion : wsOp.getPartitionRegions()) {
            if (auto origType =
                    dyn_cast<ttg::MemDescType>(blockArg.getType())) {
              auto newType = getNewMemDescType(
                  origType, lattice->getValue().getLayoutEncoding());
              partitionRegion->getArgument(i).setType(newType);
            }
          }
        }
        return WalkResult::advance();
      }

      for (auto [i, result] : llvm::enumerate(op->getResults())) {
        if (!isa<ttg::MemDescType>(result.getType()))
          continue;
        auto *lattice = solver.lookupState<LayoutEncodingLattice>(result);
        if (!lattice)
          llvm_unreachable("Lattice not found.");
        if (lattice->getValue().isUninitialized())
          continue;
        if (auto origType = dyn_cast<ttg::MemDescType>(result.getType())) {
          auto newType = getNewMemDescType(
              origType, lattice->getValue().getLayoutEncoding());
          op->getResult(i).setType(newType);
        }
      }
      return WalkResult::advance();
    });

    // Verify that no DummyTMEMLayoutAttr remains after layout propagation
    bool hasDummyLayout = false;
    funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
      auto encoding = allocOp.getType().getEncoding();
      if (isa_and_nonnull<DummyTMEMLayoutAttr>(encoding)) {
        allocOp.emitError(
            "DummyTMEMLayoutAttr was not resolved during layout propagation");
        hasDummyLayout = true;
      }
      return WalkResult::advance();
    });
    if (hasDummyLayout)
      return signalPassFailure();

    return;
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RequireLayoutPattern>(context);
    patterns.add<ReleaseLayoutPattern>(context);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
