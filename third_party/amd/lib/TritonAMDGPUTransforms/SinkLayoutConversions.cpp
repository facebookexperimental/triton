#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSINKLAYOUTCONVERSIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return the first user in the same block of the given op. If the user is in a
// nested block then return the op owning the block. Return nullptr if not
// existing.
static Operation *getFirstUseInSameBlock(Operation *op) {
  SmallVector<Operation *> usersInSameBlock;
  for (auto user : op->getUsers()) {
    if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
      usersInSameBlock.push_back(ancestor);
  }
  auto minOpIt =
      llvm::min_element(usersInSameBlock, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
  return minOpIt != usersInSameBlock.end() ? *minOpIt : nullptr;
}

// Sink conversion after the last dealloc but before the first use in its block.
// This helps to avoid unnecessary shared memory allocation.
static void sinkLayoutConversions(triton::FuncOp funcOp) {
  SmallVector<ttg::ConvertLayoutOp> convertOps;
  funcOp.walk([&](ttg::ConvertLayoutOp op) { convertOps.push_back(op); });
  for (auto op : convertOps) {
    Operation *user = getFirstUseInSameBlock(op);
    for (auto it = Block::iterator(op), ie = op->getBlock()->end();
         it != ie && &*it != user; ++it)
      if (isa<ttg::LocalDeallocOp>(&*it))
        op->moveAfter(&*it);
  }
}

// Consecutive cond_barriers in a block open and close a window in which the
// two warp groups are phase-shifted (e.g. around a ping-pong loop). A
// conversion that exchanges data through shared memory inserts a workgroup
// barrier; inside the window that barrier pairs with a different barrier in
// the other group, so it may read data the other group has not written. Move
// conversions out of the window, after its closing cond_barrier, when no use
// precedes it.
static void sinkLayoutConversionsPastCondBarriers(triton::FuncOp funcOp) {
  funcOp.walk([](Block *block) {
    SmallVector<triton::amdgpu::CondBarrierOp> barriers;
    for (auto barrier : block->getOps<triton::amdgpu::CondBarrierOp>())
      barriers.push_back(barrier);
    for (size_t i = 0; i + 1 < barriers.size(); i += 2) {
      Operation *open = barriers[i], *close = barriers[i + 1];
      SmallVector<ttg::ConvertLayoutOp> convertOps;
      for (auto it = std::next(Block::iterator(open)),
                ie = Block::iterator(close);
           it != ie; ++it)
        if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(&*it))
          convertOps.push_back(convert);
      for (auto convert : llvm::reverse(convertOps)) {
        Operation *user = getFirstUseInSameBlock(convert);
        if (!user || close->isBeforeInBlock(user))
          convert->moveAfter(close);
      }
    }
  });
}

} // namespace

struct TritonAMDGPUSinkLayoutConversionsPass
    : public impl::TritonAMDGPUSinkLayoutConversionsBase<
          TritonAMDGPUSinkLayoutConversionsPass> {

  void runOnOperation() override {
    sinkLayoutConversions(getOperation());
    sinkLayoutConversionsPastCondBarriers(getOperation());
  }
};

} // namespace mlir
