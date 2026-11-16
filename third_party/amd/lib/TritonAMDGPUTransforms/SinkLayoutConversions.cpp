#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
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

// Return true if `a` and `b` are provably logical complements: one is
// `arith.xori` of the other with true, or both are `arith.cmpi` over the same
// operands with inverted predicates (the canonical form of `~(x >= y)`).
static bool isComplement(Value a, Value b) {
  auto isNotOf = [](Value x, Value y) {
    auto xorOp = x.getDefiningOp<arith::XOrIOp>();
    if (!xorOp)
      return false;
    auto isTrue = [](Value v) {
      APInt c;
      return matchPattern(v, m_ConstantInt(&c)) && c.isAllOnes();
    };
    return (xorOp.getLhs() == y && isTrue(xorOp.getRhs())) ||
           (xorOp.getRhs() == y && isTrue(xorOp.getLhs()));
  };
  if (isNotOf(a, b) || isNotOf(b, a))
    return true;
  auto cmpA = a.getDefiningOp<arith::CmpIOp>();
  auto cmpB = b.getDefiningOp<arith::CmpIOp>();
  return cmpA && cmpB && cmpA.getLhs() == cmpB.getLhs() &&
         cmpA.getRhs() == cmpB.getRhs() &&
         cmpB.getPredicate() == arith::invertPredicate(cmpA.getPredicate());
}

// A cond_barrier(p) followed in the same block by cond_barrier(~p) opens and
// closes a window in which the two warp groups are phase-shifted (e.g. around
// a ping-pong loop). A conversion that exchanges data through shared memory
// inserts a workgroup barrier; inside the window that barrier pairs with a
// different barrier in the other group, so it may read data the other group
// has not written. Move conversions out of the window, after its closing
// cond_barrier, when no use precedes it. Pairing is a caller contract the IR
// does not enforce, so a window is only recognized when the closing predicate
// is provably the complement of the opening one; any other cond_barrier is
// left alone and does not shift the pairing of later ones.
static void sinkLayoutConversionsPastCondBarriers(triton::FuncOp funcOp) {
  funcOp.walk([](Block *block) {
    SmallVector<triton::amdgpu::CondBarrierOp> barriers;
    for (auto barrier : block->getOps<triton::amdgpu::CondBarrierOp>())
      barriers.push_back(barrier);
    for (size_t i = 0; i + 1 < barriers.size();) {
      if (!isComplement(barriers[i].getPred(), barriers[i + 1].getPred())) {
        ++i;
        continue;
      }
      Operation *open = barriers[i], *close = barriers[i + 1];
      i += 2;
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
