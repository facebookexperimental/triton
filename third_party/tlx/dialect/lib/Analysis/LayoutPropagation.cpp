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
  // Forward merges should stay conservative: distinct concrete layouts widen to
  // unknown instead of asserting so region joins can fall back cleanly.
  if (lhs.isUnknown() || rhs.isUnknown())
    return LayoutEncoding::getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
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
  LDBG("Conflicting memdesc layouts " << lhs << " vs " << rhs
                                      << "; widening to unknown");
  return LayoutEncoding::getUnknownLayout();
}

static bool isValidPermutation(ArrayRef<int32_t> order, unsigned rank) {
  if (order.size() != rank)
    return false;

  SmallVector<char> seen(rank, 0);
  for (int32_t dim : order) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = 1;
  }
  return true;
}

static bool
isEffectivelyUnswizzledShared(ttg::SwizzledSharedEncodingAttr encoding) {
  return encoding.getVec() == 1 && encoding.getPerPhase() == 1 &&
         encoding.getMaxPhase() == 1;
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
      auto blockArgumentLattice =
          getLatticeElement(warpSpecializeOp.getPartitionOp()
                                .getExplicitCaptures()[arg.getArgNumber()]);
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
  if (isa<tlx::ReleaseLayoutOp>(op))
    return success();

  if (isa<RegionBranchOpInterface, ttg::WarpSpecializePartitionsOp>(op))
    return visitRegionInReverse(op);

  // Transpose op needs to be handled specially. When flowing backwards through
  // it, we need to update the layout encoding.
  if (auto memDescTransOp = dyn_cast<ttg::MemDescTransOp>(op)) {
    auto resultLattice = results[0];
    LayoutEncoding resultLayoutEncoding = resultLattice->getValue();
    if (!resultLayoutEncoding.isUninitialized()) {
      Attribute resultEnc = resultLattice->getValue().getLayoutEncoding();
      SmallVector<unsigned, 4> newOrder;
      llvm::transform(memDescTransOp.getOrder(), std::back_inserter(newOrder),
                      [](int32_t x) { return static_cast<unsigned>(x); });
      Attribute srcEncoding;
      if (auto mmaEncoding =
              dyn_cast<ttg::NVMMASharedEncodingAttr>(resultEnc)) {
        srcEncoding = ttg::NVMMASharedEncodingAttr::get(
            mmaEncoding.getContext(),
            memDescTransOp.getSrc().getType().getShape(), newOrder,
            mmaEncoding.getCGALayout(),
            memDescTransOp.getSrc().getType().getElementType(),
            mmaEncoding.getFp4Padded());
      } else if (auto swizzledEncoding =
                     dyn_cast<ttg::SwizzledSharedEncodingAttr>(resultEnc)) {
        auto srcType =
            cast<ttg::MemDescType>(memDescTransOp.getSrc().getType());
        unsigned rank = srcType.getRank();
        auto transOrder = memDescTransOp.getOrder();
        if (!isValidPermutation(transOrder, rank) ||
            swizzledEncoding.getOrder().size() != rank) {
          memDescTransOp.emitOpError(
              "swizzled_shared backward propagation through memdesc_trans "
              "requires a valid transpose permutation");
          return failure();
        }
        if (!isEffectivelyUnswizzledShared(swizzledEncoding)) {
          memDescTransOp.emitOpError(
              "swizzled_shared backward propagation through memdesc_trans "
              "only supports effectively unswizzled encodings");
          return failure();
        }

        // For effectively unswizzled shared layouts, inverting the transpose
        // only needs to update the iteration order.
        SmallVector<unsigned> invOrder(rank);
        for (unsigned i = 0; i < rank; ++i)
          invOrder[transOrder[i]] = i;
        auto encOrder = swizzledEncoding.getOrder();
        SmallVector<unsigned> permutedOrder(rank);
        for (unsigned i = 0; i < rank; ++i)
          permutedOrder[i] = invOrder[encOrder[i]];
        srcEncoding = ttg::SwizzledSharedEncodingAttr::get(
            swizzledEncoding.getContext(), swizzledEncoding.getVec(),
            swizzledEncoding.getPerPhase(), swizzledEncoding.getMaxPhase(),
            permutedOrder, swizzledEncoding.getCGALayout());
      }
      if (srcEncoding) {
        const auto updatedResultLayoutEncoding = LayoutEncoding(srcEncoding);
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

  // TMEMSubSliceOp preserves the source tile shape and only refines the
  // column-stride/CTA-split details on the 2D slice view. The verifier already
  // guarantees tensor-memory encodings on both source and result.
  if (auto tmemSliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
    auto resultLattice = results[0];
    LayoutEncoding resultLayoutEncoding = resultLattice->getValue();
    if (!resultLayoutEncoding.isUninitialized() &&
        !resultLayoutEncoding.isUnknown()) {
      if (auto tmemEncoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
              resultLattice->getValue().getLayoutEncoding())) {
        auto srcTy = cast<ttg::MemDescType>(tmemSliceOp.getSrc().getType());
        auto srcEncoding =
            dyn_cast<ttng::TensorMemoryEncodingAttr>(srcTy.getEncoding());
        if (!srcEncoding)
          return tmemSliceOp.emitOpError(
              "expected tensor memory source encoding while propagating "
              "through tmem_subslice");
        auto newTmemEncoding = ttng::TensorMemoryEncodingAttr::get(
            tmemEncoding.getContext(), srcEncoding.getBlockM(),
            srcEncoding.getBlockN(), tmemEncoding.getColStride(),
            tmemEncoding.getCGALayout(), tmemEncoding.getTwoCTAs());
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

  // Handle TMEMCopyOp: when destination has TensorMemoryScalesEncodingAttr,
  // the source shared memory must be unswizzled. Propagate this constraint.
  if (auto tmemCopyOp = dyn_cast<ttng::TMEMCopyOp>(op)) {
    // Check the lattice encoding for the destination. The lattice may have
    // TensorMemoryScalesEncodingAttr propagated from downstream operations
    // (e.g., RequireLayoutOp). If the IR already has the encoding, the source
    // should already be correctly set up.
    auto dstLattice = operands[1];
    auto dstLatticeEncoding = dstLattice->getValue();
    if (!dstLatticeEncoding.isUninitialized() &&
        isa<ttng::TensorMemoryScalesEncodingAttr>(
            dstLatticeEncoding.getLayoutEncoding())) {
      // Source must be unswizzled for scales copy.
      // Create an unswizzled encoding requirement for the source.
      auto srcType = cast<ttg::MemDescType>(tmemCopyOp.getSrc().getType());
      auto ctx = srcType.getContext();

      // Build unswizzled NVMMASharedEncodingAttr with default CTA layout
      auto ctaLayout =
          ttg::CGAEncodingAttr::get1CTALayout(ctx, srcType.getRank());
      auto unswizzledEncoding = ttg::NVMMASharedEncodingAttr::get(
          ctx,
          /*swizzlingByteWidth=*/0,
          /*transposed=*/false,
          srcType.getElementType().getIntOrFloatBitWidth(),
          /*fp4Padded=*/false, ctaLayout);
      const auto unswizzledLayoutEncoding = LayoutEncoding(unswizzledEncoding);
      auto operandLattice = operands[0];
      ChangeResult changed = operandLattice->meet(unswizzledLayoutEncoding);
      propagateIfChanged(operandLattice, changed);
      visitWarpSpecRegionArgs(op, tmemCopyOp.getSrc(),
                              unswizzledLayoutEncoding);
      return success();
    }
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
// TensorLayout
//===----------------------------------------------------------------------===//

void TensorLayout::print(raw_ostream &os) const {
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

TensorLayout TensorLayout::join(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  return meet(lhs, rhs);
}

TensorLayout TensorLayout::meet(const TensorLayout &lhs,
                                const TensorLayout &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return TensorLayout::getUnknownLayout();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
  return TensorLayout::getUnknownLayout();
}

static bool isTrackedTensorValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool isAllowedTensorLayoutUser(Operation *op, unsigned operandIndex) {
  // This mirrors InsertRequireLayout's pre-materialization policy. Before the
  // insert pass runs, dot operands flow through convert_layout and transparent
  // region carriers. After convert_layout is rewritten into explicit
  // tlx.require_layout anchors, tensor propagation treats those anchors plus
  // the same transparent carriers as the legal local_load-to-dot path.
  if (auto requireLayoutOp = dyn_cast<RequireLayoutOp>(op)) {
    if (!isa<RankedTensorType>(requireLayoutOp.getType()) || operandIndex != 0)
      return false;
    return isSupportedDotConstraintEncoding(
        cast<RankedTensorType>(requireLayoutOp.getType()).getEncoding());
  }

  return isTransparentLayoutCarrierOp(op);
}

static bool canRewriteTensorResult(Operation *op) {
  return isa<ttg::LocalLoadOp, RegionBranchOpInterface>(op);
}

//===----------------------------------------------------------------------===//
// TensorBackwardPropagation
//===----------------------------------------------------------------------===//

LogicalResult TensorBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TensorLayoutLattice *> operands,
    ArrayRef<const TensorLayoutLattice *> results) {
  LDBG("Visiting tensor operation " << *op << "\n");

  if (auto requireLayoutOp = dyn_cast<RequireLayoutOp>(op)) {
    if (!isa<RankedTensorType>(requireLayoutOp.getType()))
      return success();

    Attribute layout = requireLayoutOp.getType().getEncoding();
    if (!isSupportedDotConstraintEncoding(layout))
      return success();

    const auto layoutLattice = TensorLayout(layout);
    for (auto [operandLattice, operand] :
         llvm::zip_equal(operands, requireLayoutOp->getOperands())) {
      if (!isTrackedTensorValue(operand))
        continue;
      ChangeResult changed = operandLattice->meet(layoutLattice);
      propagateIfChanged(operandLattice, changed);
    }
    return success();
  }

  if (isa<ReleaseLayoutOp>(op))
    return success();

  // If a tracked tensor value is used by an unsupported operation, rewriting
  // the producer chain is no longer legal for that entire component.
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!isTrackedTensorValue(operand))
      continue;
    if (isAllowedTensorLayoutUser(op, index))
      continue;

    TensorLayout operandState = operands[index]->getValue();
    if (operandState.isUninitialized())
      continue;

    LDBG("Marking tensor layout unknown due to unsupported user "
         << op->getName() << " on operand #" << index);
    ChangeResult changed =
        operands[index]->meet(TensorLayout::getUnknownLayout());
    propagateIfChanged(operands[index], changed);
  }

  // Only a narrow set of tensor-producing operations can absorb a propagated
  // layout directly. Everything else falls back to a local convert.
  if (!canRewriteTensorResult(op)) {
    for (Value result : op->getResults()) {
      if (!isTrackedTensorValue(result))
        continue;

      auto *resultLattice = getLatticeElement(result);
      TensorLayout resultState = resultLattice->getValue();
      if (resultState.isUninitialized())
        continue;

      LDBG("Keeping explicit tensor layout conversion because producer "
           << op->getName() << " cannot be retagged directly");
      ChangeResult changed =
          resultLattice->meet(TensorLayout::getUnknownLayout());
      propagateIfChanged(resultLattice, changed);
    }
  }

  return success();
}

void TensorBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  if (!isTrackedTensorValue(operand.get()))
    return;

  Operation *owner = operand.getOwner();
  if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(owner))
    return;

  auto *lattice = getLatticeElement(operand.get());
  TensorLayout state = lattice->getValue();
  if (state.isUninitialized())
    return;

  ChangeResult changed = lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::visitCallOperand(OpOperand &operand) {
  if (!isTrackedTensorValue(operand.get()))
    return;

  auto *lattice = getLatticeElement(operand.get());
  TensorLayout state = lattice->getValue();
  if (state.isUninitialized())
    return;

  ChangeResult changed = lattice->meet(TensorLayout::getUnknownLayout());
  propagateIfChanged(lattice, changed);
}

void TensorBackwardPropagation::setToExitState(TensorLayoutLattice *lattice) {}

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

    // Unknown layouts do not provide enough information to refine a TMEM slice
    // result, so only splice concrete tensor-memory encodings through.
    if (auto sliceOp = dyn_cast<ttng::TMEMSubSliceOp>(op)) {
      if (!operandLayoutEncoding.isUninitialized() &&
          !operandLayoutEncoding.isUnknown()) {
        auto dstTy = cast<ttg::MemDescType>(sliceOp.getType());
        auto dstEncoding =
            dyn_cast<ttng::TensorMemoryEncodingAttr>(dstTy.getEncoding());
        auto encoding = dyn_cast<ttng::TensorMemoryEncodingAttr>(
            operandLayoutEncoding.getLayoutEncoding());
        if (!encoding)
          return sliceOp.emitOpError(
              "expected tensor memory layout while propagating through "
              "tmem_subslice");
        auto newEncoding = ttng::TensorMemoryEncodingAttr::get(
            op->getContext(), dstEncoding.getBlockM(), dstEncoding.getBlockN(),
            encoding.getColStride(), encoding.getCGALayout(),
            encoding.getTwoCTAs());
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
