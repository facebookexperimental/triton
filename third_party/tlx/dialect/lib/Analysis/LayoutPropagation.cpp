#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "tlx/dialect/include/Analysis/LayoutPropagation.h"

#define DEBUG_TYPE "layout-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::triton::tlx {

//===----------------------------------------------------------------------===//
// LayoutEncoding
//===----------------------------------------------------------------------===//
void LayoutEncoding::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  if (isUnknown()) {
    os << "<UNKNOWN>";
    return;
  }
  return getLayoutEncoding().print(os);
}

LayoutEncoding LayoutEncoding::join(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  return LayoutEncoding::getUnknownLayout();
}

LayoutEncoding LayoutEncoding::meet(const LayoutEncoding &lhs,
                                    const LayoutEncoding &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return LayoutEncoding::getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
  llvm_unreachable("Conflicting layouts");
}

//===----------------------------------------------------------------------===//
// LayoutBackwardPropagation
//===----------------------------------------------------------------------===//

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  if (auto requireLayoutOp = dyn_cast<triton::tlx::RequireLayoutOp>(op)) {
    Attribute layout = requireLayoutOp.getType().getEncoding();
    const auto layoutLattice = LayoutEncoding(layout);
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(layoutLattice);
      propagateIfChanged(operandLattice, changed);
    }
    return success();
  }

  auto isScalar = [](Type type) { return type.isIntOrIndexOrFloat(); };
  // Propagate from results to the operands
  for (const auto resultLattice : results) {
    for (auto [i, operandLattice] : llvm::enumerate(operands)) {
      // Don't propagate if the operand is a scalar
      if (isScalar(op->getOpOperand(i).get().getType()))
        continue;
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &operand) {}

void LayoutBackwardPropagation::visitCallOperand(OpOperand &operand) {
  llvm_unreachable(
      "Should not have any call operands in the IR after inlining.");
}

void LayoutBackwardPropagation::setToExitState(LayoutEncodingLattice *lattice) {
}

//===----------------------------------------------------------------------===//
// LayoutForwardPropagation
//===----------------------------------------------------------------------===//

LogicalResult LayoutForwardPropagation::visitOperation(
    Operation *op, ArrayRef<const LayoutEncodingLattice *> operands,
    ArrayRef<LayoutEncodingLattice *> results) {

  if (!isa<triton::gpu::MemDescSubviewOp>(op))
    return success();

  auto memDescSubviewOp = cast<triton::gpu::MemDescSubviewOp>(op);
  for (const auto operandLattice : operands) {
    for (auto [i, resultLattice] : llvm::enumerate(results)) {
      ChangeResult changed = resultLattice->meet(operandLattice->getValue());
      propagateIfChanged(resultLattice, changed);
    }
  }
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *lattice) {
}

} // namespace mlir::triton::tlx
