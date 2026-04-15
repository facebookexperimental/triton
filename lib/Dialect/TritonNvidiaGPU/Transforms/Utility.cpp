#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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
      auto *block = arg.getOwner();
      if (auto wsPartitionOp =
              dyn_cast<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
          wsPartitionOp && block->isEntryBlock()) {
        auto argIndex = arg.getArgNumber();
        auto wsOp = wsPartitionOp->getParentOfType<ttg::WarpSpecializeOp>();
        // map to WSOp's operand at the same index
        nextTarget = wsOp.getOperand(argIndex);
      } else if (block->isEntryBlock()) {
        // Entry block arg of a region-based control flow op (e.g. scf.for,
        // scf.while). Use RegionBranchOpInterface to find all values that
        // can flow into this block arg position.
        if (auto regionBranch =
                dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
          // Find a RegionSuccessor for our region by querying all possible
          // branch points (parent op + each region). We must check all
          // because some regions are only reachable from sibling regions,
          // not from the parent (e.g. scf.while's "after" region).
          // The RegionSuccessor's inputs may be a subset of block args
          // (e.g. scf.for drops the induction variable), so we find the
          // arg's index within those inputs.
          auto *parentOp = block->getParentOp();
          auto *region = block->getParent();

          // Helper to search successors from a given branch point.
          auto searchSuccessors = [&](auto getSuccessorsFn) -> bool {
            SmallVector<RegionSuccessor> successors;
            getSuccessorsFn(successors);
            for (auto &successor : successors) {
              if (successor.getSuccessor() != region)
                continue;
              auto inputs = successor.getSuccessorInputs();
              for (auto [i, input] : llvm::enumerate(inputs)) {
                if (input == arg) {
                  SmallVector<Value> predValues;
                  regionBranch.getPredecessorValues(successor, i, predValues);
                  for (auto val : predValues)
                    worklist.insert(val);
                  return true;
                }
              }
            }
            return false;
          };

          // Check from parent op first, then from each sibling region.
          // Some regions are only reachable from siblings, not from the
          // parent (e.g. scf.while's "after" region).
          if (!searchSuccessors([&](auto &s) {
                regionBranch.getSuccessorRegions(RegionBranchPoint::parent(),
                                                 s);
              })) {
            for (auto &r : parentOp->getRegions()) {
              if (searchSuccessors(
                      [&](auto &s) { regionBranch.getSuccessorRegions(r, s); }))
                break;
            }
          }
        }
        continue;
      } else {
        // Non-entry CF block arg: trace through predecessor terminators.
        auto argIdx = arg.getArgNumber();
        for (auto *pred : block->getPredecessors()) {
          auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branchOp)
            continue;
          for (unsigned i = 0, e = branchOp->getNumSuccessors(); i < e; ++i) {
            if (branchOp->getSuccessor(i) != block)
              continue;
            auto succOperands = branchOp.getSuccessorOperands(i);
            // SuccessorOperands splits block args into "produced" (implicit,
            // defined by the block) and "forwarded" (explicit operands from
            // the branch). For standard cf/scf ops produced count is always
            // 0, but we handle the general case for correctness.
            auto produced = succOperands.getProducedOperandCount();
            if (argIdx >= produced) {
              auto forwarded = succOperands.getForwardedOperands();
              worklist.insert(forwarded[argIdx - produced]);
            }
          }
        }
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
        // If this op is a RegionBranchOpInterface, trace its results back
        // through region terminators to find the values that produce them
        // (e.g. scf.while results come from scf.condition operands).
        // NOTE: This traces all results, not just those used in the slice,
        // so it may over-approximate. This is safe (conservative) but could
        // be tightened to only trace results used by ops in the slice.
        if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(op)) {
          RegionSuccessor parentSuccessor(op, op->getResults());
          for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
            SmallVector<Value> predValues;
            regionBranch.getPredecessorValues(parentSuccessor, i, predValues);
            for (auto val : predValues)
              worklist.insert(val);
          }
        }
      }
    }
  }
}

} // namespace mlir::triton::nvidia_gpu
