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
