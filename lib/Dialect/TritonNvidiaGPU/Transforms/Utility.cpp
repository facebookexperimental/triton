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

    // --- Handle block arguments ---
    if (auto arg = dyn_cast<BlockArgument>(nextTarget)) {
      auto *block = arg.getOwner();
      if (auto wsPartitionOp =
              dyn_cast<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
          wsPartitionOp && block->isEntryBlock()) {
        // WS partition entry block arg → map to WarpSpecializeOp operand.
        auto wsOp = wsPartitionOp->getParentOfType<ttg::WarpSpecializeOp>();
        worklist.insert(wsOp.getOperand(arg.getArgNumber()));
      } else if (block->isEntryBlock()) {
        // Entry block arg of a region-based control flow op (e.g. scf.for,
        // scf.while). Use RegionBranchOpInterface to find all values that
        // can flow into this block arg position.
        if (auto regionBranch =
                dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
          // Find a RegionSuccessor for our region by querying all possible
          // branch points (parent op + each region). We must check all
          // because some regions are only reachable from siblings, not
          // from the parent (e.g. scf.while's "after" region).
          auto *parentOp = block->getParentOp();
          auto *region = block->getParent();

          // Helper: search successors from a given branch point for our
          // region, then call getPredecessorValues at the matching index.
          auto searchSuccessors = [&](auto getSuccessorsFn) -> bool {
            SmallVector<RegionSuccessor> successors;
            getSuccessorsFn(successors);
            for (auto &successor : successors) {
              if (successor.getSuccessor() != region)
                continue;
              // Successor inputs may be a subset of block args (e.g.
              // scf.for drops the IV), so match by identity, not index.
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
      } else {
        // Non-entry CF block arg: trace through predecessor terminators.
        auto argIdx = arg.getArgNumber();
        for (auto *pred : block->getPredecessors()) {
          // terminator can be e.g. cf.br/cf.cond_br
          auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branchOp)
            continue;
          for (unsigned i = 0, e = branchOp->getNumSuccessors(); i < e; ++i) {
            if (branchOp->getSuccessor(i) != block)
              continue;
            auto succOperands = branchOp.getSuccessorOperands(i);
            auto produced = succOperands.getProducedOperandCount();
            if (argIdx >= produced) {
              auto forwarded = succOperands.getForwardedOperands();
              worklist.insert(forwarded[argIdx - produced]);
            }
          }
        }
      }
      continue; // end of --- Handle block arguments ---
    }

    // If nextTarget is a result of a RegionBranchOpInterface, handle it
    // directly — the filter would exclude it from getBackwardSlice.
    if (auto opResult = dyn_cast<OpResult>(nextTarget)) {
      if (auto regionBranch =
              dyn_cast<RegionBranchOpInterface>(opResult.getOwner())) {
        auto *branchOp = regionBranch.getOperation();
        RegionSuccessor parentSuccessor(branchOp, branchOp->getResults());
        SmallVector<Value> predValues;
        regionBranch.getPredecessorValues(
            parentSuccessor, opResult.getResultNumber(), predValues);
        for (auto val : predValues)
          worklist.insert(val);
        // Also add control operands (e.g. lb, ub, step for scf.for).
        // These aren't predecessor values — they don't flow through
        // results — but results depend on them, and no op inside the
        // region uses them, so omitUsesFromAbove won't trace them.
        // Skip entry/init operands; those are traced precisely above.
        if (backwardSlice->insert(branchOp)) { // first time seeing this op
          // Find the init operand range to skip (already traced precisely).
          // Default empty range = skip nothing = all operands are control.
          unsigned entryBegin = 0, entryEnd = 0;
          SmallVector<RegionSuccessor> successors;
          regionBranch.getSuccessorRegions( // which regions can parent enter?
              RegionBranchPoint::parent(), successors);
          for (auto &succ : successors) {
            if (!succ.getSuccessor()) // skip parent-to-parent successor
              continue;
            // operands forwarded into this region (e.g. initArgs for scf.for)
            auto entryOps = regionBranch.getEntrySuccessorOperands(succ);
            if (!entryOps.empty()) {
              entryBegin = entryOps.getBeginOperandIndex();
              entryEnd = entryBegin + entryOps.size();
            }
            break; // one region successor is enough to determine the range
          }
          for (auto [idx, branchOperand] :
               llvm::enumerate(branchOp->getOperands())) {
            if (idx >= entryBegin && idx < entryEnd)
              continue; // init arg — traced by getPredecessorValues above
            worklist.insert(branchOperand); // control operand (lb, ub, step)
          }
        }
        continue; // end of handling OpResult of RegionBranchOpInterface
      }
    }

    SetVector<Operation *> ops;
    if (failed(getBackwardSlice(nextTarget, &ops, options))) {
      llvm_unreachable("getBackwardSlice failed");
    }

    for (auto op : ops) {
      if (backwardSlice->insert(op)) {
        for (auto operand : op->getOperands()) {
          // Block args and RegionBranchOpInterface results are not in ops
          // (omitBlockArguments / filter). Add them to the worklist so the
          // top-of-loop handlers trace them precisely.
          if (isa<BlockArgument>(operand) ||
              (isa<OpResult>(operand) &&
               isa<RegionBranchOpInterface>(operand.getDefiningOp())))
            worklist.insert(operand);
        }
      }
    }
  }
}

} // namespace mlir::triton::nvidia_gpu
