#include "IR/Dialect.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include "mlir/Transforms/Passes.h"


#define DEBUG_TYPE "tlx-amd-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace tlx = ::mlir::triton::tlx;

namespace mlir {
namespace amdpipeliner {
  std::optional<ttg::SwizzledSharedEncodingAttr>
  getSharedEncIfAllUsersAreDotEnc(Value loadedValue);
}

namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequiredLayout\n");
  WalkResult result = m.walk([&](tt::DotOp dotOp) -> WalkResult {
    SetVector<Operation *> backwardSet;
    BackwardSliceOptions options;
    options.inclusive = false;
    options.omitUsesFromAbove = false;
    if (failed(mlir::getBackwardSlice(dotOp.getOperation(), &backwardSet, options))) {
      return WalkResult::interrupt();
    }
    LLVM_DEBUG({
          llvm::dbgs() << "DotOp\n";
          dotOp.dump();
    });
    for (Operation *op : backwardSet) {
      if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "LocalLoadOp\n";
          localLoadOp.dump();
        });
        // Get the shared encoding for this local load op based on the dot op
        auto encoding = mlir::amdpipeliner::getSharedEncIfAllUsersAreDotEnc(localLoadOp->getResult(0)).value_or(nullptr);
        if (encoding) {
          LLVM_DEBUG({
            llvm::dbgs() << "SwizzledSharedEncodingAttr\n";
            encoding.dump();
          });
          builder.setInsertionPoint(localLoadOp);
          auto encodingAttr = mlir::cast<Attribute>(encoding);
          auto loadMemDescTy = op->getOperands()[0];
          if(auto type = dyn_cast<ttg::MemDescType>(loadMemDescTy.getType())) {
            auto newType = ttg::MemDescType::get(
                type.getShape(), type.getElementType(), encodingAttr,
                type.getMemorySpace(), type.getMutableMemory());
            auto converLayoutOp = builder.create<tlx::RequireLayoutOp>(op->getLoc(), newType, loadMemDescTy);
            localLoadOp->setOperand(0, converLayoutOp.getResult());
          }
        } else {
          localLoadOp->emitError("Cannot find appropriate shared encoding for local load op");
          return WalkResult::interrupt(); 
        }
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
        return failure();
  }
  return success();
}

struct TLXInsertRequireLayoutPass
    : public impl::TLXInsertRequireLayoutBase<TLXInsertRequireLayoutPass> {
public:
  using impl::TLXInsertRequireLayoutBase<
      TLXInsertRequireLayoutPass>::TLXInsertRequireLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if(failed(tlx::insertRequireLayout(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
