#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Analysis/TaskIdPropagation.h"

#define DEBUG_TYPE "task-id-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// TaskId
//===----------------------------------------------------------------------===//

void TaskId::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  if (isUnknown()) {
    os << "<UNKNOWN>";
    return;
  }
  return getTaskIds().print(os);
}

TaskId TaskId::join(const TaskId &lhs, const TaskId &rhs) {return TaskId::getUnknownTaskId();}

TaskId TaskId::meet(const TaskId &lhs, const TaskId &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return TaskId::getUnknownTaskId();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;

  auto lhsTasks = lhs.getTaskIds().getValues<AsyncTaskId>();
  auto rhsTasks = rhs.getTaskIds().getValues<AsyncTaskId>();
  // Meet the task ids by merging and deduplicating them
  SmallVector<AsyncTaskId> result(lhsTasks.begin(), lhsTasks.end());
  result.insert(result.end(), rhsTasks.begin(), rhsTasks.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  auto i32Ty = IntegerType::get(lhs.getTaskIds().getContext(), 32);
  auto size = static_cast<int64_t>(result.size());
  auto vecTy = VectorType::get(size, i32Ty);
  auto mergedAndDedupedTaskIds =
      TaskId(DenseIntElementsAttr::get(vecTy, ArrayRef<AsyncTaskId>(result)));
  return mergedAndDedupedTaskIds;
}

//===----------------------------------------------------------------------===//
// TaskIdBackwardPropagation
//===----------------------------------------------------------------------===//

void TaskIdBackwardPropagation::propagateToYield(
    scf::YieldOp yieldOp, SmallVector<TaskId> &lattices) {
  for (auto [lattice, yieldOperand] :
       llvm::zip_equal(lattices, yieldOp->getOperands())) {
    auto yieldLattice = getLatticeElement(yieldOperand);
    ChangeResult changed = yieldLattice->meet(lattice);
    propagateIfChanged(yieldLattice, changed);
  }
}

void TaskIdBackwardPropagation::propagateToParent(Operation *op,
                                                  const TaskId &taskId) {
  auto parentOp = op->getParentOp();
  while (parentOp && !isa<triton::FuncOp>(parentOp)) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // Propagate to the control operands of the for op.
      for (auto controlOperand :
           forOp.getOperands().take_front(forOp.getNumControlOperands())) {
        auto controlLattice = getLatticeElement(controlOperand);
        ChangeResult changed = controlLattice->meet(taskId);
        propagateIfChanged(controlLattice, changed);
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      auto cond = ifOp.getCondition();
      auto condLattice = getLatticeElement(cond);
      ChangeResult changed = condLattice->meet(taskId);
      propagateIfChanged(condLattice, changed);
    } else {
      if (!isa<triton::FuncOp>(parentOp))
        llvm_unreachable("Other parent ops are not supported.");
    }
    parentOp = parentOp->getParentOp();
  }
}

LogicalResult TaskIdBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TaskIdLattice *> operands,
    ArrayRef<const TaskIdLattice *> results) {
  // Already annotated
  // TODO(Arda): Replace the following with getAsyncTaskIds when we no longer
  // need to dump the task ids into the IR.
  auto taskIdAttr = op->getAttrOfType<DenseIntElementsAttr>("async_task_id");
  if (taskIdAttr) {
    const auto annotated = TaskId(taskIdAttr);
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(annotated);
      propagateIfChanged(operandLattice, changed);
    }
    // Propagate to the parent ops such as control flows
    propagateToParent(op, annotated);
    return success();
  }
  // If it is not annotated by the user, propagate from results to the
  // operands
  for (const auto resultLattice : results) {
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
    }
  }

  for (const auto resultLattice : results) {
    auto lattice = resultLattice;
    propagateToParent(op, lattice->getValue());
  }
  return success();
}

void TaskIdBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  auto defOp = operand.getOwner();
  assert(isa<scf::IfOp>(defOp) || isa<scf::ForOp>(defOp));

  SmallVector<TaskId> lattices(defOp->getNumResults(),
                               TaskId::getUninitialized());
  for (auto [i, result] : llvm::enumerate(defOp->getResults())) {
    auto resultLattice = getLatticeElement(result);
    // Wait for all the results to be initialized.
    if (resultLattice->getValue().isUninitialized())
      return;
    lattices[i] =
        resultLattice->getValue().meet(lattices[i], resultLattice->getValue());
  }

  // Propagate to the yield ops
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    propagateToYield(yieldOp, lattices);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto thenYieldOp =
        ifOp.thenYield(); // Then yield always exists even if it is implicit
    auto elseYieldOp =
        !ifOp.getElseRegion().empty() ? ifOp.elseYield() : nullptr;
    for (auto yieldOp : ArrayRef{thenYieldOp, elseYieldOp}) {
      if (!yieldOp)
        continue;
      propagateToYield(yieldOp, lattices);
    }
  } else {
    llvm_unreachable("Unknown branch operation");
  }
  return;

  // TODO(Arda): Address what happens when loop is annotated
}

void TaskIdBackwardPropagation::visitCallOperand(OpOperand &operand) {
  llvm_unreachable(
      "Should not have any call operands in the IR after inlining.");
}

void TaskIdBackwardPropagation::setToExitState(TaskIdLattice *lattice) {}

} // namespace mlir::triton
