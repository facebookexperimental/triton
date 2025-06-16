#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tlx/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::tlx {

#define GEN_PASS_DEF_TRITONTLXATTACHMETADATA
#include "tlx/dialect/include/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tlx-attach-metadata"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class TritonTLXAttachMetadataPass
    : public impl::TritonTLXAttachMetadataBase<TritonTLXAttachMetadataPass> {
  using impl::TritonTLXAttachMetadataBase<
      TritonTLXAttachMetadataPass>::TritonTLXAttachMetadataBase;

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    Builder b(&getContext());
    mod->setAttr(ttg::AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(ttg::AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(ttg::AttrTargetName, b.getStringAttr(this->target.getValue()));
  }
};

} // namespace mlir::triton::tlx
