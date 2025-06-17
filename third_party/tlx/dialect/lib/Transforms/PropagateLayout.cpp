#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/DataFlowFramework.h"
#define DEBUG_TYPE "triton-gpu-taskid-propagate"
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

    PatternRewriter rewriter(&getContext());
    SymbolTableCollection symbolTable;
    Operation *op = getOperation();
    DataFlowSolver solver;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LayoutPropagation>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    getOperation()->walk([&](triton::FuncOp funcOp) {
      WalkResult walkResult = funcOp.walk([&](mlir::Operation *op) {
        for (auto [i, result] : llvm::enumerate(op->getResults())) {
          auto *lattice = solver.lookupState<LayoutEncodingLattice>(result);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          if (lattice->getValue().isUninitialized())
            continue;
          if (auto origType = dyn_cast<RankedTensorType>(result.getType())) {
            auto newType = RankedTensorType::get(
                origType.getShape(), origType.getElementType(),
                lattice->getValue().getLayoutEncoding());
            op->getResult(i).setType(newType);
            if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
              auto valAttr = cast<DenseElementsAttr>(constantOp.getValueAttr());
              auto valType = cast<ShapedType>(valAttr.getType());
              // This reshape hack adds the new encoding to the value
              // attribute.
              valAttr = valAttr.reshape(newType);
              op->setAttr("value", valAttr);
              break;
            }
          } else if (auto origType =
                         dyn_cast<gpu::MemDescType>(result.getType())) {
            auto newType = gpu::MemDescType::get(
                origType.getShape(), origType.getElementType(),
                lattice->getValue().getLayoutEncoding(),
                origType.getMemorySpace(), origType.getMutableMemory());
            op->getResult(i).setType(newType);
          }
        }
        return WalkResult::advance();
      });
    });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReleaseLayoutPattern>(context);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();

    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
