#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSSTRIPPARTITIONATTRSOUTSIDEWS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isWarpSpecializeLoop(Operation *op) {
  auto forOp = dyn_cast<scf::ForOp>(op);
  return forOp && forOp->hasAttr(kWarpSpecializeAttrName);
}

static bool isNestedInWarpSpecializeLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isWarpSpecializeLoop(parent))
      return true;
  }
  return false;
}

class NVWSStripPartitionAttrsOutsideWS
    : public mlir::triton::impl::NVWSStripPartitionAttrsOutsideWSBase<
          NVWSStripPartitionAttrsOutsideWS> {
public:
  using mlir::triton::impl::NVWSStripPartitionAttrsOutsideWSBase<
      NVWSStripPartitionAttrsOutsideWS>::NVWSStripPartitionAttrsOutsideWSBase;

  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (isWarpSpecializeLoop(op) || isNestedInWarpSpecializeLoop(op))
        return;

      op->removeAttr(kPartitionAttrName);
      op->removeAttr(kPartitionOutputsAttrName);
      op->removeAttr(kPartitionStagesAttrName);
      op->removeAttr(kWarpSpecializeTagAttrName);
    });
  }
};

} // namespace
