#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

void getBackwardSliceWithWS(Value target,
                            SetVector<Operation *> *backwardSlice) {
  SetVector<Value> worklist;
  worklist.insert(target);

  BackwardSliceOptions options;
  options.omitUsesFromAbove = false;
  options.omitBlockArguments = true;
  options.inclusive = true;

  while (!worklist.empty()) {
    Value nextTarget = worklist.back();
    worklist.pop_back();

    if (auto arg = dyn_cast<BlockArgument>(nextTarget)) {
      if (auto wsPartitionOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(
              arg.getOwner()->getParentOp())) {
        auto argIndex = arg.getArgNumber();
        auto wsOp = wsPartitionOp->getParentOfType<ttg::WarpSpecializeOp>();
        // map to WSOp's operand at the same index
        nextTarget = wsOp.getOperand(argIndex);
      } else {
        // ttg::WarpSpecializeOp's default region just captures
        // from trunk so no need to special handle the defining block args.
        // We should omit block args for other block structures like scf.For,
        // and the captures would still be handled automatically
        continue;
      }
    }

    SetVector<Operation *> ops;
    if (failed(getBackwardSlice(nextTarget, &ops, options))) {
      llvm_unreachable("getBackwardSlice failed");
    }

    for (auto op : ops) {
      if (backwardSlice->insert(op)) {
        for (auto operand : op->getOperands()) {
          if (isa<BlockArgument>(operand)) {
            worklist.insert(operand);
          }
        }
      }
    }
  }
}

} // namespace mlir::triton::nvidia_gpu
