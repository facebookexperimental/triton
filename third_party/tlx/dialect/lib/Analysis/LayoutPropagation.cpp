#include "IR/Dialect.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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
namespace ttng = ::mlir::triton::nvidia_gpu;

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
  assert(lhs == rhs && "Conflicting layouts");
  return lhs;
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

void LayoutBackwardPropagation::visitWarpSpecRegionArgs(
    Operation *op, Value opnd, const LayoutEncoding &resultEncoding) {
  if (auto arg = dyn_cast<BlockArgument>(opnd)) {
    if (auto warpSpecializePartitionsOp =
            op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
      auto warpSpecializeOp = warpSpecializePartitionsOp.getParentOp();
      auto blockArgumentLattice = getLatticeElement(
          warpSpecializeOp.getExplicitCaptures()[arg.getArgNumber()]);
      ChangeResult changed = blockArgumentLattice->meet(resultEncoding);
      propagateIfChanged(blockArgumentLattice, changed);
      // Propagate to all the partition regions
      for (Region *partitionRegion : warpSpecializeOp.getPartitionRegions()) {
        auto blockArgumentLattice =
            getLatticeElement(partitionRegion->getArgument(arg.getArgNumber()));
        ChangeResult changed = blockArgumentLattice->meet(resultEncoding);
        propagateIfChanged(blockArgumentLattice, changed);
      }
    }
  }
}

LogicalResult LayoutBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutEncodingLattice *> operands,
    ArrayRef<const LayoutEncodingLattice *> results) {
  LDBG("Visiting operation " << *op << "\n");
  if (isa<tlx::ReleaseLayoutOp, tlx::LocalAliasOp>(op))
    return success();

  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegionInReverse(op);

  // Handle LocalLoadOp to infer smem layout from result encoding
  if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
    LDBG("LocalLoadOp: processing " << *op);

    // Get the result tensor type
    auto resultType = dyn_cast<RankedTensorType>(localLoadOp.getType());
    if (!resultType) {
      LDBG("LocalLoadOp: result is not RankedTensorType, skipping");
      return success();
    }

    auto resultEncoding = resultType.getEncoding();
    if (!resultEncoding) {
      LDBG("LocalLoadOp: result has no encoding, skipping");
      return success();
    }
    LDBG("LocalLoadOp: result encoding = " << resultEncoding);

    // Get the source MemDescType
    auto srcType = dyn_cast<ttg::MemDescType>(localLoadOp.getSrc().getType());
    if (!srcType) {
      LDBG("LocalLoadOp: source is not MemDescType, skipping");
      return success();
    }
    LDBG("LocalLoadOp: source type = " << srcType);

    // Compute the shared memory encoding based on the result encoding.
    // For blocked encoding, use the contiguity and order from the result type
    // to create a vectorized shared encoding.
    Attribute sharedEncoding;
    if (isa<ttg::BlockedEncodingAttr>(resultEncoding)) {
      LDBG("LocalLoadOp: result has BlockedEncodingAttr");

      auto ctaLayout = ttg::getCTALayout(srcType.getEncoding());
      auto order = ttg::getOrderForMemory(resultType);
      unsigned bitWidth = srcType.getElementTypeBitWidth();

      LDBG("LocalLoadOp: order = [" << order[0] << ", " << order[1]
                                    << "], bitWidth = " << bitWidth);

      // Get the contiguity per thread from the result encoding.
      // The vector size is the contiguity in the fastest-varying dimension.
      auto contigPerThread = ttg::getContigPerThread(resultType);
      LDBG("LocalLoadOp: contigPerThread = [" << contigPerThread[0] << ", "
                                              << contigPerThread[1] << "]");

      unsigned vecSize = contigPerThread[order[0]];
      // Limit vector size to 128 bits (max vectorized load size).
      vecSize = std::min<unsigned>(vecSize, 128 / bitWidth);
      vecSize = std::max<unsigned>(vecSize, 1);

      LDBG("LocalLoadOp: vecSize = " << vecSize);

      // Compute swizzling parameters to avoid bank conflicts.
      // Following the pattern from WMMA/MFMA shared encoding computation.
      constexpr unsigned numBanks = 32;
      constexpr unsigned bankBitWidth = 32;
      int64_t innerDimLength = srcType.getShape()[order[0]];
      int elemsPerOneBanksRow = (numBanks * bankBitWidth) / bitWidth;
      int perPhase = std::max<int>(1, elemsPerOneBanksRow / innerDimLength);
      // maxPhase determines how many phases before the pattern repeats.
      int maxPhase = std::max<int>(1, innerDimLength / vecSize / perPhase);

      LDBG("LocalLoadOp: innerDimLength = "
           << innerDimLength
           << ", elemsPerOneBanksRow = " << elemsPerOneBanksRow
           << ", perPhase = " << perPhase << ", maxPhase = " << maxPhase);

      sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
          op->getContext(), vecSize, perPhase, maxPhase, order, ctaLayout);

      LDBG("LocalLoadOp: computed sharedEncoding = " << sharedEncoding);
    } else {
      LDBG("LocalLoadOp: result encoding is not BlockedEncodingAttr, skipping");
    }

    if (sharedEncoding) {
      const auto layoutEncoding = LayoutEncoding(sharedEncoding);
      auto operandLattice = operands[0];
      ChangeResult changed = operandLattice->meet(layoutEncoding);
      LDBG("LocalLoadOp: propagating layout, changed = "
           << (changed == ChangeResult::Change ? "true" : "false"));
      propagateIfChanged(operandLattice, changed);
      visitWarpSpecRegionArgs(op, localLoadOp.getSrc(), layoutEncoding);
    }
    return success();
  }

  // Transpose op needs to be handled specially. When flowing backwards through
  // it, we need to update the layout encoding.
  if (auto memDescTransOp = dyn_cast<ttg::MemDescTransOp>(op)) {
    auto resultLattice = results[0];
    LayoutEncoding resultLayoutEncoding = resultLattice->getValue();
    if (!resultLayoutEncoding.isUninitialized()) {
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
        ChangeResult changed =
            operandLattice->meet(updatedResultLayoutEncoding);
        propagateIfChanged(operandLattice, changed);
        visitWarpSpecRegionArgs(op, memDescTransOp.getSrc(),
                                updatedResultLayoutEncoding);
      }
    }
    return success();
  }

  // Similar to MemDescTransOp, we need to specially handle TMEMSubSliceOp
  if (auto tmemSliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    // Slice resultLayoutEncoding
    auto resultLattice = results[0];
    LayoutEncoding resultLayoutEncoding = resultLattice->getValue();
    if (!resultLayoutEncoding.isUninitialized()) {
      if (auto tmemEncoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
              resultLattice->getValue().getLayoutEncoding())) {
        auto srcTy = cast<ttg::MemDescType>(tmemSliceOp.getSrc().getType());
        auto srcEncoding =
            dyn_cast<ttng::TensorMemoryEncodingAttr>(srcTy.getEncoding());
        auto newTmemEncoding = ttng::TensorMemoryEncodingAttr::get(
            tmemEncoding.getContext(), srcEncoding.getBlockM(),
            srcEncoding.getBlockN(), tmemEncoding.getUnpacked(),
            tmemEncoding.getCTASplitM(), tmemEncoding.getCTASplitN());
        const auto updatedResultLayoutEncoding =
            LayoutEncoding(newTmemEncoding);
        auto operandLattice = operands[0];
        ChangeResult changed =
            operandLattice->meet(updatedResultLayoutEncoding);
        propagateIfChanged(operandLattice, changed);
        visitWarpSpecRegionArgs(op, tmemSliceOp.getSrc(),
                                updatedResultLayoutEncoding);
      }
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
      visitWarpSpecRegionArgs(op, operand, layoutLattice);
    }
    return success();
  }

  // Propagate from results to the operands
  for (const auto resultLattice : results) {
    for (auto [i, operandLattice] : llvm::enumerate(operands)) {
      // Only propagate for memdesc types
      if (!isa<ttg::MemDescType>(op->getOpOperand(i).get().getType()))
        continue;
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
      visitWarpSpecRegionArgs(op, op->getOpOperand(i).get(),
                              resultLattice->getValue());
    }
  }
  return success();
}

void LayoutBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  auto branchOp = operand.getOwner();
  LDBG("Backward visiting branch op " << *branchOp << "\n");
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
  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegion(op);

  if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp, ttng::TMEMSubSliceOp,
           ttg::LocalAllocOp, ttng::TMEMAllocOp>(op))
    return success();

  for (const auto [operandIdx, operandLattice] : llvm::enumerate(operands)) {
    if (!isa<ttg::MemDescType>(op->getOperand(operandIdx).getType()))
      continue;
    LayoutEncoding operandLayoutEncoding = operandLattice->getValue();

    // Slice operandLayoutEncoding
    if (auto sliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
      if (!operandLayoutEncoding.isUninitialized()) {
        auto dstTy = cast<ttg::MemDescType>(sliceOp.getType());
        auto dstEncoding =
            dyn_cast<ttng::TensorMemoryEncodingAttr>(dstTy.getEncoding());
        auto encoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
            operandLayoutEncoding.getLayoutEncoding());
        auto newEncoding = ttng::TensorMemoryEncodingAttr::get(
            op->getContext(), dstEncoding.getBlockM(), dstEncoding.getBlockN(),
            encoding.getUnpacked(), encoding.getCTASplitM(),
            encoding.getCTASplitN());
        operandLayoutEncoding = LayoutEncoding(newEncoding);
      }
    }

    for (auto resultLattice : results) {
      ChangeResult changed = resultLattice->meet(operandLayoutEncoding);
      propagateIfChanged(resultLattice, changed);
    }
  }

  for (const auto [resultIdx, resultLattice] : llvm::enumerate(results)) {
    if (failed(visitWarpSpecRegionArgs(op, op->getResult(resultIdx),
                                       resultLattice->getValue())))
      return failure();
  }

  return success();
}

LogicalResult LayoutForwardPropagation::visitWarpSpecRegionArgs(
    Operation *op, Value result, const LayoutEncoding &resultEncoding) {
  // For all use of the result, propagate the resultEncoding to the
  // corresponding warp spec region arg if it is a captured arg.
  for (auto &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned idx = use.getOperandNumber();
      // Propagate to the i-th argument of every partition region
      // Propagate to all the partition regions
      for (Region *partitionRegion : wsOp.getPartitionRegions()) {
        auto blockArgumentLattice =
            getLatticeElement(partitionRegion->getArgument(idx));
        ChangeResult changed = blockArgumentLattice->meet(resultEncoding);
        propagateIfChanged(blockArgumentLattice, changed);
      }
      if (failed(visitRegion(wsOp)))
        return failure();
    }
  }

  return success();
}

LogicalResult LayoutForwardPropagation::visitRegion(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        SmallVector<const LayoutEncodingLattice *> operands;
        for (const auto operand : nestedOp.getOperands())
          operands.push_back(getLatticeElement(operand));
        SmallVector<LayoutEncodingLattice *> results;
        for (Value result : nestedOp.getResults())
          results.push_back(getLatticeElement(result));
        auto visitResult = visitOperation(&nestedOp, operands, results);
        if (failed(visitResult))
          return visitResult;
      }
    }
  }
  return success();
}

void LayoutForwardPropagation::setToEntryState(LayoutEncodingLattice *lattice) {
}

} // namespace mlir::triton::tlx
