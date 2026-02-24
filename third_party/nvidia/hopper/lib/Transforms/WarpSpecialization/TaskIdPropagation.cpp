#include "TaskIdPropagation.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "task-id-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::triton::gpu {

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

TaskId TaskId::join(const TaskId &lhs, const TaskId &rhs) {
  return TaskId::getUnknownTaskId();
}

TaskId TaskId::meet(const TaskId &lhs, const TaskId &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return TaskId::getUnknownTaskId();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;

  auto context = lhs.getTaskIds().getContext();
  auto lhsTasks = lhs.getTaskIds().asArrayRef();
  auto rhsTasks = rhs.getTaskIds().asArrayRef();
  // Meet the task ids by merging and deduplicating them
  SmallVector<AsyncTaskId> result(lhsTasks.begin(), lhsTasks.end());
  result.insert(result.end(), rhsTasks.begin(), rhsTasks.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  auto mergedAndDedupedTaskIds =
      TaskId(DenseI32ArrayAttr::get(context, ArrayRef<AsyncTaskId>(result)));
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

void TaskIdBackwardPropagation::propagateToTerminator(
    Operation *op, ArrayRef<const TaskIdLattice *> &lattices) {
  for (auto [lattice, terminatorOperand] :
       llvm::zip_equal(lattices, op->getOperands())) {
    auto terminatorLattice = getLatticeElement(terminatorOperand);
    ChangeResult changed = terminatorLattice->meet(lattice->getValue());
    propagateIfChanged(terminatorLattice, changed);
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
      if (!isa<triton::FuncOp, triton::ReduceOp>(parentOp))
        llvm_unreachable("Other parent ops are not supported.");
    }
    parentOp = parentOp->getParentOp();
  }
}

LogicalResult TaskIdBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TaskIdLattice *> operands,
    ArrayRef<const TaskIdLattice *> results) {
  // TODO(Arda): Replace the following with getAsyncTaskIds when we no longer
  // need to dump the task ids into the IR.
  auto taskIdAttr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id");

  // An op is a non-anchor (allows backward propagation to flow through) only
  // if it is a scalar arithmetic/math op. These ops compute shared addresses
  // or indices used across tasks and need the union of consumer task IDs.
  // All other annotated ops (Triton ops, tensor ops, control flow) are anchors
  // whose task IDs define the computation partition and must not be overridden.
  bool isScalarArithOrMath =
      isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) &&
      llvm::none_of(op->getResultTypes(),
                    [](Type t) { return isa<RankedTensorType>(t); });
  bool isAnchor = taskIdAttr && !isScalarArithOrMath;

  if (isAnchor) {
    const auto annotated = TaskId(taskIdAttr);
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(annotated);
      propagateIfChanged(operandLattice, changed);
    }
    propagateToParent(op, annotated);

    if (op->getNumRegions() == 1) {
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
        propagateToTerminator(reduceOp.getCombineOp().front().getTerminator(),
                              results);
      }
    }

    return success();
  }

  // Non-anchor: propagate from results to operands (standard backward flow).
  for (const auto resultLattice : results) {
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
    }
  }

  for (const auto resultLattice : results)
    propagateToParent(op, resultLattice->getValue());

  // For non-anchor ops with existing annotations, also propagate the
  // annotation backward so it contributes to operand lattices.
  if (taskIdAttr) {
    const auto annotated = TaskId(taskIdAttr);
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(annotated);
      propagateIfChanged(operandLattice, changed);
    }
    propagateToParent(op, annotated);
  }

  if (op->getNumRegions() == 1) {
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      propagateToTerminator(reduceOp.getCombineOp().front().getTerminator(),
                            results);
    }
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
    propagateToYield(ifOp.thenYield(), lattices);
    if (!ifOp.getElseRegion().empty())
      propagateToYield(ifOp.elseYield(), lattices);
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

} // namespace mlir::triton::gpu
