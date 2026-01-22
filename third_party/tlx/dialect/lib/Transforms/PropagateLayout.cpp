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

  /// Cancel multibuffering for TMEM allocations that will receive scales
  /// encoding.
  ///
  /// Background:
  /// - MX (Microscaling) scales for scaled MMA operations are stored in TMEM
  ///   with TensorMemoryScalesEncodingAttr layout.
  /// - Multibuffering creates 3D allocations with shape (num_buffers, M, K)
  ///   where num_buffers is typically 1 for scales.
  /// - However, TensorMemoryScalesEncodingAttr only supports 2D shapes (M, K).
  ///
  /// This function:
  /// 1. Uses dataflow analysis to predict which allocs will receive scales
  ///    encoding (before the encoding is actually applied).
  /// 2. Flattens 3D shapes (1×M×K) to 2D (M×K) for those allocations.
  /// 3. Removes memdesc_index ops that access the buffering dimension.
  ///
  /// Why this runs BEFORE layout propagation:
  /// - The layout walk updates types based on dataflow analysis results.
  /// - If we flattened types during/after the layout walk, memdesc_index ops
  ///   would still expect 3D input, causing rank mismatch errors.
  /// - By running this first, index ops are removed before type rewriting.
  ///
  /// @param funcOp The function to process.
  /// @param solver The dataflow solver with layout analysis results.
  void cancelMultibufferingForScales(triton::FuncOp funcOp,
                                     DataFlowSolver &solver) {
    DenseMap<Value, ttg::MemDescType> allocsToFlatten;

    // =========================================================================
    // Pass 1: Identify TMEMAllocOps that will have scales encoding
    // =========================================================================
    // Use dataflow analysis to predict the final encoding before it's applied.
    // This allows us to identify scale allocs proactively.
    funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
      auto origType = allocOp.getType();
      auto shape = origType.getShape();

      // Only process 3D allocations with buffering dimension of 1.
      // Shape (1, M, K) indicates single-buffered TMEM allocation.
      if (shape.size() != 3 || shape[0] != 1)
        return WalkResult::advance();

      // Query dataflow analysis for the predicted layout encoding.
      // The solver has already propagated layout constraints through the IR.
      auto *lattice =
          solver.lookupState<LayoutEncodingLattice>(allocOp.getResult());
      if (!lattice || lattice->getValue().isUninitialized())
        return WalkResult::advance();

      auto encoding = lattice->getValue().getLayoutEncoding();

      // Only flatten allocations that will receive TensorMemoryScalesEncoding.
      // Other TMEM allocations (e.g., accumulators) should keep their shape.
      if (!isa<ttng::TensorMemoryScalesEncodingAttr>(encoding))
        return WalkResult::advance();

      // Build the flattened 2D type: (1, M, K) -> (M, K)
      // Preserve the original encoding (DummyTMEMLayoutAttr) since the actual
      // scales encoding will be applied later in the layout walk.
      SmallVector<int64_t> newShape{shape[1], shape[2]};
      auto newType = ttg::MemDescType::get(
          newShape, origType.getElementType(), origType.getEncoding(),
          origType.getMemorySpace(), origType.getMutableMemory());
      allocsToFlatten[allocOp.getResult()] = newType;
      return WalkResult::advance();
    });

    // =========================================================================
    // Pass 2: Remove memdesc_index ops that access the buffering dimension
    // =========================================================================
    // With single-buffering cancelled, index ops like `memdesc_index %alloc[0]`
    // become redundant. Replace their uses with the alloc directly.
    SmallVector<ttg::MemDescIndexOp> indexOpsToRemove;
    for (auto &[allocValue, newType] : allocsToFlatten) {
      for (auto &use : allocValue.getUses()) {
        if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(use.getOwner())) {
          // The index op extracts a 2D slice from 3D: (1, M, K)[0] -> (M, K)
          // After flattening, the alloc is already 2D, so replace all uses
          // of the index result with the flattened alloc value.
          indexOp.getResult().replaceAllUsesWith(allocValue);
          indexOpsToRemove.push_back(indexOp);
        }
      }
    }

    // Erase the now-unused index ops.
    for (auto indexOp : indexOpsToRemove) {
      indexOp.erase();
    }

    // =========================================================================
    // Pass 3: Update alloc types to the flattened 2D shape
    // =========================================================================
    // This must happen after index ops are removed to avoid type mismatches.
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

    // =========================================================================
    // Cancel multibuffering for scales BEFORE applying layouts
    // =========================================================================
    // This must run before the layout walk because:
    // 1. The layout walk updates types based on dataflow analysis results.
    // 2. If we tried to flatten types during/after the layout walk,
    //    memdesc_index ops would still reference 3D types, causing errors.
    // 3. By running this first, we remove index ops and flatten alloc types
    //    before any type rewriting occurs.
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

    // =========================================================================
    // Verification: Ensure all DummyTMEMLayoutAttr have been resolved
    // =========================================================================
    // DummyTMEMLayoutAttr is a placeholder for TMEM allocations that should
    // receive a proper layout (e.g., TensorMemoryScalesEncodingAttr) during
    // layout propagation. If any remain, it indicates a bug in the analysis.
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
