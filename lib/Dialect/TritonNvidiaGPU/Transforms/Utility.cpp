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
  // Exclude RegionBranchOpInterface ops from getBackwardSlice so it doesn't
  // pull in all their operands indiscriminately. We handle them precisely
  // in the operand loop, tracing only the specific result index.
  options.filter = [](Operation *op) {
    return !isa<RegionBranchOpInterface>(op);
  };

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
          } else if (auto opResult = dyn_cast<OpResult>(operand)) {
            if (auto regionBranch =
                    dyn_cast<RegionBranchOpInterface>(opResult.getOwner())) {
              // Trace through region terminators at this specific result
              // index (e.g. scf.for result #1 → init #1 and yield #1).
              auto *branchOp = regionBranch.getOperation();
              RegionSuccessor parentSuccessor(branchOp, branchOp->getResults());
              SmallVector<Value> predValues;
              regionBranch.getPredecessorValues(
                  parentSuccessor, opResult.getResultNumber(), predValues);
              for (auto val : predValues)
                worklist.insert(val);
              // The filter excluded this op from getBackwardSlice, so add
              // it and its control operands (e.g. lb, ub, step for scf.for)
              // manually. Skip entry/init operands — they're already traced
              // precisely by getPredecessorValues at the right index.
              if (backwardSlice->insert(branchOp)) {
                unsigned entryBegin = 0, entryEnd = 0;
                SmallVector<RegionSuccessor> successors;
                regionBranch.getSuccessorRegions(RegionBranchPoint::parent(),
                                                 successors);
                for (auto &succ : successors) {
                  if (succ.getSuccessor()) {
                    auto entryOps =
                        regionBranch.getEntrySuccessorOperands(succ);
                    if (!entryOps.empty()) {
                      entryBegin = entryOps.getBeginOperandIndex();
                      entryEnd = entryBegin + entryOps.size();
                    }
                    break;
                  }
                }
                for (auto [idx, branchOperand] :
                     llvm::enumerate(branchOp->getOperands())) {
                  if (idx >= entryBegin && idx < entryEnd)
                    continue;
                  worklist.insert(branchOperand);
                }
              }
            }
            // Non-RegionBranchOpInterface results are already in ops
            // via getBackwardSlice — no action needed.
          }
        }
      }
    }
  }
}

} // namespace mlir::triton::nvidia_gpu
