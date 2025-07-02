#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

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
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // First check if there is any TLX related op in the module. If not, do
    // nothing.
    auto tlxDialectName = TLXDialect::getDialectNamespace();
    WalkResult result = mod.walk([&](Operation *op) {
      // Ops directly in TLX Dialect
      if (op->getDialect()->getNamespace() == tlxDialectName) {
        return WalkResult::interrupt();
      }
      // Ops that should not be in TTIR unless introduced by TLX
      if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::AsyncCommitGroupOp,
              ttg::AsyncWaitOp, ttg::MemDescTransOp, ttng::FenceAsyncSharedOp,
              ttng::WarpGroupDotOp, ttng::WarpGroupDotWaitOp,
              ttng::TensorDescToTMAPtrOp, ttng::AsyncTMACopyGlobalToLocalOp,
              ttng::AsyncTMACopyLocalToGlobalOp, ttng::TMEMAllocOp,
              ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TCGen5MMAOp>(op)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!result.wasInterrupted()) {
      // No TLX op found, do nothing.
      return;
    }

    // Attach metadata to the module.
    Builder b(&getContext());
    mod->setAttr(ttg::AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(ttg::AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(ttg::AttrTargetName, b.getStringAttr(this->target.getValue()));
    mod->setAttr(AttrHasTLXOpsName, b.getBoolAttr(true));
  }
};

} // namespace mlir::triton::tlx
