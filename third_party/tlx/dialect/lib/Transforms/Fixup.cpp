#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::tlx {

#define GEN_PASS_DEF_TRITONTLXFIXUP
#include "tlx/dialect/include/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tlx-fixup"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class TritonTLXFixupPass : public impl::TritonTLXFixupBase<TritonTLXFixupPass> {
  using impl::TritonTLXFixupBase<TritonTLXFixupPass>::TritonTLXFixupBase;

public:
  // validate the module and error early for unsupported cases
  LogicalResult verifyModule(ModuleOp &mod) {
    // ws should not capture RankedTensorType
    ttg::WarpSpecializeOp *invalidWSOp = nullptr;
    auto result = mod.walk([&](ttg::WarpSpecializeOp op) {
      for (auto argType : op.getOperandTypes()) {
        if (isa<RankedTensorType>(argType)) {
          invalidWSOp = &op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return invalidWSOp->emitError()
             << "WarpSpecializeOp should not capture "
                "RankedTensorType. Try moving tensor "
                "computation into specific async task.";
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (failed(verifyModule(mod))) {
      return signalPassFailure();
    }

    // First check if there is any TLX related op in the module. If not, do
    // nothing.
    auto tlxDialectName = TLXDialect::getDialectNamespace();
    WalkResult result = mod.walk([&](Operation *op) {
      // Ops directly in TLX Dialect
      if (op->getDialect()->getNamespace() == tlxDialectName) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    auto hasTLXOps = result.wasInterrupted();

    auto hasExplicitLocalMemAccess =
        mod.walk([&](Operation *op) {
             // Ops that should not be in TTIR unless introduced by TLX
             if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp,
                     ttng::AsyncTMACopyGlobalToLocalOp,
                     ttng::AsyncTMACopyLocalToGlobalOp, ttng::TMEMAllocOp,
                     ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TCGen5MMAOp>(
                     op)) {
               return WalkResult::interrupt();
             }
             return WalkResult::advance();
           })
            .wasInterrupted();

    auto hasWarpSpecOps = mod.walk([&](Operation *op) {
                               if (isa<ttg::WarpSpecializeOp, ttg::WarpYieldOp,
                                       ttg::WarpReturnOp>(op)) {
                                 return WalkResult::interrupt();
                               }
                               return WalkResult::advance();
                             })
                              .wasInterrupted();

    if (!hasTLXOps && !hasExplicitLocalMemAccess && !hasWarpSpecOps) {
      return;
    }

    // Attach metadata to the module.
    Builder b(&getContext());
    mod->setAttr(ttg::AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(ttg::AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(ttg::AttrTargetName, b.getStringAttr(this->target.getValue()));
    if (hasTLXOps)
      mod->setAttr(AttrHasTLXOpsName, b.getBoolAttr(true));
    if (hasExplicitLocalMemAccess)
      mod->setAttr(AttrHasExplicitLocalMemAccessName, b.getBoolAttr(true));
    if (hasWarpSpecOps)
      mod->setAttr(AttrHasWarpSpecOpsName, b.getBoolAttr(true));
  }
};

} // namespace mlir::triton::tlx
