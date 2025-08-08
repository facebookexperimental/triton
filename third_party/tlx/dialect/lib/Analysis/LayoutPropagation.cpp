#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "tlx/dialect/include/Analysis/LayoutPropagation.h"

#define DEBUG_TYPE "tlx-layout-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::dataflow;
namespace ttg = ::mlir::triton::gpu;

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

LogicalResult LayoutBackwardPropagation::visitRegionInReverse(Operation *op) {
  for (Region &region : llvm::reverse(op->getRegions())) {
    for (Block &block : llvm::reverse(region)) {
      for (Operation &nestedOp : llvm::reverse(block)) {
        SmallVector<LayoutEncodingLattice *> operands;
        for (auto operand : nestedOp.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<const LayoutEncodingLattice *> results;
        for (const Value result : nestedOp.getResults())
          results.push_back(getLatticeElement(result));
        auto visitResult = visitOperation(&nestedOp, operands, results);
        if (failed(visitResult))
          return visitResult;
      }
    }
  }
  return success();
}

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  LDBG("Visiting operation " << *op << "\n");
  if (isa<tlx::ReleaseLayoutOp>(op))
    return success();

  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegionInReverse(op);

  // Transpose op needs to be handled specially. When flowing backwards through
  // it, we need to update the layout encoding.
  if (auto memDescTransOp = dyn_cast<ttg::MemDescTransOp>(op)) {
    auto resultLattice = results[0];
    if (auto mmaEncoding = dyn_cast<ttg::NVMMASharedEncodingAttr>(
            resultLattice->getValue().getLayoutEncoding())) {
      SmallVector<unsigned, 4> newOrder;
      llvm::transform(memDescTransOp.getOrder(), std::back_inserter(newOrder),
                      [](int32_t x) { return static_cast<unsigned>(x); });
      auto newMmaEncoding = ttg::NVMMASharedEncodingAttr::get(
          mmaEncoding.getContext(),
          memDescTransOp.getSrc().getType().getShape(), newOrder,
          mmaEncoding.getCTALayout(),
          memDescTransOp.getSrc().getType().getElementType(),
          mmaEncoding.getFp4Padded());
      const auto updatedResultLayoutEncoding = LayoutEncoding(newMmaEncoding);
      auto operandLattice = operands[0];
      ChangeResult changed = operandLattice->meet(updatedResultLayoutEncoding);
      propagateIfChanged(operandLattice, changed);
    }
    return success();
  }

  if (auto requireLayoutOp = dyn_cast<triton::tlx::RequireLayoutOp>(op)) {
    // Skip the layout propagation for registers. require_layout ops on tensor
    // types will be rewritten into convert_layout ops, and following passes
    // will handle them.
    if (isa<RankedTensorType>(requireLayoutOp.getType()))
      return success();
    Attribute layout = requireLayoutOp.getType().getEncoding();
    const auto layoutLattice = LayoutEncoding(layout);
    for (auto [operandLattice, operand] :
         llvm::zip_equal(operands, requireLayoutOp->getOperands())) {
      ChangeResult changed = operandLattice->meet(layoutLattice);
      propagateIfChanged(operandLattice, changed);
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (auto warpSpecializePartitionsOp =
                op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
          auto warpSpecializeOp = warpSpecializePartitionsOp.getParentOp();
          auto blockArgumentLattice = getLatticeElement(
              warpSpecializeOp.getExplicitCaptures()[arg.getArgNumber()]);
          ChangeResult changed = blockArgumentLattice->meet(layoutLattice);
          propagateIfChanged(blockArgumentLattice, changed);
          // Propagate to all the partition regions
          for (Region *partitionRegion :
               warpSpecializeOp.getPartitionRegions()) {
            auto blockArgumentLattice = getLatticeElement(
                partitionRegion->getArgument(arg.getArgNumber()));
            ChangeResult changed = blockArgumentLattice->meet(layoutLattice);
            propagateIfChanged(blockArgumentLattice, changed);
          }
        }
      }
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
      if (auto arg = dyn_cast<BlockArgument>(op->getOpOperand(i).get())) {
        if (auto warpSpecializePartitionsOp =
                op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
          auto warpSpecializeOp = warpSpecializePartitionsOp.getParentOp();
          auto blockArgumentLattice = getLatticeElement(
              warpSpecializeOp.getExplicitCaptures()[arg.getArgNumber()]);
          ChangeResult changed =
              blockArgumentLattice->meet(resultLattice->getValue());
          propagateIfChanged(blockArgumentLattice, changed);
          // Propagate to all the partition regions
          for (Region *partitionRegion :
               warpSpecializeOp.getPartitionRegions()) {
            auto blockArgumentLattice = getLatticeElement(
                partitionRegion->getArgument(arg.getArgNumber()));
            ChangeResult changed =
                blockArgumentLattice->meet(resultLattice->getValue());
            propagateIfChanged(blockArgumentLattice, changed);
          }
        }
      }
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  auto branchOp = operand.getOwner();
  LDBG("Visiting branch op " << *branchOp << "\n");
  if (isa<ttg::WarpSpecializeOp>(branchOp)) {
    auto unused = visitRegionInReverse(branchOp);
    (void)unused;
  }
}

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

  auto isScalar = [](Type type) { return type.isIntOrIndexOrFloat(); };
  auto memDescSubviewOp = cast<triton::gpu::MemDescSubviewOp>(op);
  for (const auto [operandIdx, operandLattice] : llvm::enumerate(operands)) {
    if (isScalar(op->getOperand(operandIdx).getType()))
      continue;
    for (auto resultLattice : results) {
      ChangeResult changed = resultLattice->meet(operandLattice->getValue());
      propagateIfChanged(resultLattice, changed);
    }
  }
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *lattice) {
}

} // namespace mlir::triton::tlx
