#include "Analysis/AMDGPUAllocation.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"

namespace mlir::triton::AMD {

static unsigned getSwizzledCvtScratchInBytes(const LinearLayout &srcLayout,
                                             const LinearLayout &dstLayout,
                                             unsigned bitwidth,
                                             TargetInfoBase &targetInfo) {
  int numBanks = targetInfo.getSharedMemoryBanks();
  auto vecBitwidth =
      triton::gpu::getVecBitwidthLdSt(srcLayout, dstLayout, bitwidth);
  auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
  unsigned elems = getNumScratchElemsSwizzledCvt(srcLayout, dstLayout, bitwidth,
                                                 numBanks, srcTile, dstTile);
  return elems * bitwidth / 8;
}

unsigned getConvertLayoutScratchInBytes(gpu::ConvertLayoutOp op,
                                        TargetInfoBase &targetInfo) {
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  // A transpose moves its odd tiles one at a time on the pow2 path.
  auto plan = getMixedRadixConversionPlan(srcTy, dstTy);
  if (plan && plan->transfer == MixedRadixConversionTransfer::Shared) {
    auto kReg = StringAttr::get(op.getContext(), "register");
    if (auto tiles = getMixedRadixTransposeTiles(
            gpu::toLinearLayout(srcTy).removeZeroBasesAlongDim(kReg),
            gpu::toLinearLayout(dstTy).removeZeroBasesAlongDim(kReg),
            getBitwidth(srcTy)))
      return getSwizzledCvtScratchInBytes(tiles->sourceTile,
                                          tiles->destinationTile,
                                          getBitwidth(srcTy), targetInfo);
  }
  if (auto bytes = getMixedRadixConversionScratchBytes(op))
    return *bytes;
  if (!cvtNeedsSharedMemory(op))
    return 0;
  return getSwizzledCvtScratchInBytes(gpu::toLinearLayout(srcTy),
                                      gpu::toLinearLayout(dstTy),
                                      getBitwidth(srcTy), targetInfo);
}

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op,
                                            TargetInfoBase &targetInfo) {

  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    ReduceOpHelper::GetNumScratchElemsFn AMDGetNumScratchElemsFn =
        [&targetInfo](const triton::LinearLayout &src,
                      const triton::LinearLayout &dst, unsigned bitwidth) {
          int numBanks = targetInfo.getSharedMemoryBanks();
          auto vecBitwidth =
              triton::gpu::getVecBitwidthLdSt(src, dst, bitwidth);
          auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
          return getNumScratchElemsSwizzledCvt(src, dst, bitwidth, numBanks,
                                               srcTile, dstTile);
        };
    return ReduceOpHelper(reduceOp).getScratchSizeInBytes(
        AMDGetNumScratchElemsFn);
  }

  if (auto cvtLayout = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
    return getConvertLayoutScratchInBytes(cvtLayout, targetInfo);
  }

  if (auto ws = dyn_cast<mlir::triton::gpu::WarpSpecializeOp>(op)) {
    uint64_t captureSize = 0;
    // Tightly pack the captures in memory.
    for (Type type : ws.getPartitionOp().getOperandTypes()) {
      if (auto descType = dyn_cast<TensorDescType>(type))
        captureSize +=
            mlir::triton::amdgpu::getTensorDescNumDwords(descType) * 4;
      else
        captureSize += mlir::triton::gpu::getSharedMemorySize(type);
    }
    // ConSan adds captures after allocation; reserve space pre-computed by the
    // common TritonInstrumentPrepareConSanCaptures pass.
    if (auto extra = ws->getAttrOfType<IntegerAttr>(
            mlir::triton::instrument::kConSanExtraCaptureBytesAttr))
      captureSize += extra.getInt();
    return captureSize;
  }

  if (isa<amdgpu::BufferAtomicCASOp, amdgpu::BufferAtomicRMWOp>(op))
    return getAtomicResultScratchSize(op->getResult(0));

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::AMD
