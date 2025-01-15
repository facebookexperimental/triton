#include <memory>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-peeling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

scf::ForOp peelFirstIteration(scf::ForOp forOp) {
  // Extract the first iteration outside the loop.
  OpBuilderWithAsyncTaskIds builder(forOp);

  // Map block arguments to loop initial values.
  IRMapping mapping;
  mapping.map(forOp.getBody()->getArguments()[0], forOp.getLowerBound());
  for (unsigned i = 1; i < forOp.getBody()->getArguments().size(); ++i) {
    mapping.map(forOp.getBody()->getArguments()[i], forOp.getInitArgs()[i - 1]);
    LLVM_DEBUG({
      LDBG("Mapping ");
      forOp.getBody()->getArguments()[i].dump();
      LDBG(" to ");
      forOp.getInitArgs()[i - 1].dump();
      LDBG("\n");
    });
  }

  // Clone the operations in the loop body for the first iteration.
  SmallVector<Value> peeledResults;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      for (auto result : yieldOp->getOperands()) {
        peeledResults.push_back(mapping.lookup(result));
      }
    } else {
      auto newOp = builder.cloneWithAsyncTaskIds(op, mapping);
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
        mapping.map(op.getResult(i), newOp->getResult(i));
      }
    }
  }

  // Adjust the original loop to become the remainder loop.
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value newLb = builder.create<arith::AddIOp>(forOp->getLoc(), lb, step);
  assert(peeledResults.size() == forOp.getNumResults() &&
         "peeled results size mismatch");
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      forOp->getLoc(), newLb, forOp.getUpperBound(), step, peeledResults);
  newForOp->setAttrs(forOp->getAttrs());
  newForOp.getRegion().takeBody(forOp.getRegion());
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  // Erase the original loop.
  forOp.erase();
  return newForOp;
}

static bool isConstantZeroTensor(Value v) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  auto splat = mlir::dyn_cast<SplatElementsAttr>(constOp.getValue());
  if (!splat)
    return false;
  return splat.getSplatValue<FloatAttr>().getValue().convertToFloat() == 0.0f;
}

// Check if the loop must be run at least once.
bool loopMustBeRunAtLeastOnce(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound();
  auto ub = forOp.getUpperBound();
  auto step = forOp.getStep();
  auto lbInt = getConstantIntValue(lb);
  auto ubInt = getConstantIntValue(ub);
  auto stepInt = getConstantIntValue(step);

  // Peeling is not needed if there is one or less iteration.
  if (lbInt && ubInt && stepInt)
    return ceil(float(*ubInt - *lbInt) / *stepInt) > 1;

  // Check if there is an assume that says the loop is not empty.
  if (!lbInt || !ubInt) {
    // Get the block containing the ForOp.
    Block *block = forOp->getBlock();
    // Iterate over operations in the block before the ForOp.
    for (auto it = Block::iterator(forOp); it != block->begin(); --it) {
      if (auto assumeOp = dyn_cast<LLVM::AssumeOp>(it)) {
        LLVM_DEBUG({
          LDBG("Found AssumeOp prior to ForOp:\n");
          assumeOp->dump();
        });
        auto truth = assumeOp->getOperand(0);
        if (auto cmpOp = truth.getDefiningOp<arith::CmpIOp>()) {
          switch (cmpOp.getPredicate()) {
          case arith::CmpIPredicate::sgt:
            if (cmpOp.getLhs() == ub && cmpOp.getRhs() == lb) {
              return true;
            }
          case arith::CmpIPredicate::slt:
            if (cmpOp.getLhs() == lb && cmpOp.getRhs() == lb) {
              return true;
            }
          default:
            break;
          }
        }
      }
    }
  }

  return false;
}

bool shouldPeel(scf::ForOp forOp) {
  SmallVector<Operation *> dotOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.hasTrait<OpTrait::DotLike>())
      dotOps.push_back(&op);
  }

  bool hasZeroAccDotOp = false;
  for (Operation *dotOp : dotOps) {
    auto acc = dotOp->getOperand(2);
    if (auto arg = dyn_cast<BlockArgument>(acc)) {
      assert(arg.getOwner() == forOp.getBody());
      if (isConstantZeroTensor(forOp.getInitArgs()[arg.getArgNumber() - 1])) {
        hasZeroAccDotOp = true;
        break;
      }
    }
  }

  if (!hasZeroAccDotOp)
    return false;
  return loopMustBeRunAtLeastOnce(forOp);
}

#define GEN_PASS_DEF_TRITONMATMULLOOPPEELING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonMatmulLoopPeelingPass
    : public impl::TritonMatmulLoopPeelingBase<TritonMatmulLoopPeelingPass> {

public:
  TritonMatmulLoopPeelingPass() = default;
  TritonMatmulLoopPeelingPass(const TritonMatmulLoopPeelingPass &) {}
  void runOnOperation() override {
    LDBG("Loop peeling pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (auto loop : loops) {
      if (shouldPeel(loop))
        (void)peelFirstIteration(loop);
    }
    LLVM_DEBUG({
      LDBG("After loop peeling");
      getOperation()->dump();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
