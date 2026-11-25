#include "LDSEncoding.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

namespace mlir {

// On targets without direct-to-LDS scatter support, a direct-to-LDS copy
// requires enough data per CTA for each lane to write at least 32 bits into
// LDS. The source layout itself may still be suboptimal here; CoalesceAsyncCopy
// can rewrite it before the final direct-to-LDS lowering checks coalescing
// legality.
static bool
hasEnoughCTABytesForDirectToLds(tt::LoadOp loadOp,
                                const tt::AMD::TargetInfo &targetInfo) {
  if (targetInfo.supportsDirectToLdsScatter())
    return true;

  auto srcTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!srcTy)
    return true;

  int64_t ctaTileBits = mlir::product(ttg::getShapePerCTA(srcTy)) *
                        getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
  return ctaTileBits >= 32 * targetInfo.getWarpSize();
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::AMD::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads.
  if (!triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis))
    return false;

  using tt::amdgpu::ISAFamily;
  bool hasDirectToLdsPath =
      llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4, ISAFamily::CDNA5},
                         targetInfo.getISAFamily());

  // On targets without scatter support, reject CTA tiles that are too small to
  // provide one 32-bit direct-to-LDS write per lane. Layout-specific coalescing
  // is checked later, after CoalesceAsyncCopy has a chance to rewrite the src.
  if (hasDirectToLdsPath &&
      !hasEnoughCTABytesForDirectToLds(loadOp, targetInfo))
    return false;

  if (sharedEnc && hasDirectToLdsPath)
    return canLoadDirectToLdsInPipeline(
        cast<RankedTensorType>(loadOp.getPtr().getType()), sharedEnc,
        targetInfo);

  return true;
}

std::optional<ttg::SharedEncodingTrait> getSharedEncIfAllUsersAreDotEnc(
    Operation *loadOp, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const TargetFeatures &targetFeatures, const tt::AMD::TargetInfo &targetInfo,
    bool useAsyncCopy) {
  assert(loadOp);
  Value loadedValue = loadOp->getResult(0);
  llvm::SmallVector<ttg::SharedEncodingTrait> sharedEncs;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SharedEncodingTrait tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingTrait>(memDesc.getEncoding());
      // If the immediate user is ttg::LocalAllocOp, likely it's created in
      // TritonAMDGPUOptimizeDotOperands. We should just respect it.
      if (!isa<ttg::LocalAllocOp>(user) &&
          !getSharedEncIfAllUsersAreDotEnc(user, axisInfoAnalysis,
                                           targetFeatures, targetInfo,
                                           useAsyncCopy)) {
        return std::nullopt;
      }
      LDBG("Deduced shared encoding candidate from memDesc: " << tempAttr);
      sharedEncs.push_back(tempAttr);
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto cgaLayout = ttg::getCGALayout(srcTy.getEncoding());

      auto order = getLoadSharedOrder(srcTy, targetInfo, useAsyncCopy);

      unsigned bitWidth = getIntOrFloatOrPtrBitWidth(srcTy.getElementType());

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        // Determine if we can use padded layouts and fallback to swizzled
        // layouts if not
        bool canUseAsyncCopy = false;
        if (useAsyncCopy && isa<tt::LoadOp>(loadOp)) {
          // We pass numBuffers=2 because we assume the schedule will not
          // determine a single buffer (which does not work with AsyncCopy)
          canUseAsyncCopy = canBeConvertedToAsyncLoad(
              2, cast<tt::LoadOp>(loadOp), {}, axisInfoAnalysis, targetInfo);
        }
        tempAttr = getDotOperandSharedEncoding(targetFeatures, srcTy, order,
                                               dotOpEnc, canUseAsyncCopy);
        LDBG("Deduced shared encoding candidate from dot layout: " << tempAttr);
        sharedEncs.push_back(tempAttr);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto mfmaEnc = getDotEncoding<ttg::AMDMfmaEncodingAttr>(
                userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = mfmaEnc.composeSharedLayoutForOperand(
              cgaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from mfma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        } else if (auto dotEnc = getDotEncoding<ttg::AMDWmmaEncodingAttr>(
                       userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = dotEnc.composeSharedLayoutForOperand(
              cgaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from wmma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        }
      }
    }
  }

  auto equalSharedEncIgnoreVec = [](ttg::SwizzledSharedEncodingAttr a,
                                    ttg::SwizzledSharedEncodingAttr b) {
    if (!a || !b)
      return false;
    return (a.getPerPhase() == b.getPerPhase() &&
            a.getMaxPhase() == b.getMaxPhase() &&
            a.getOrder() == b.getOrder() &&
            a.getCGALayout() == b.getCGALayout());
  };
  if (sharedEncs.empty() || !sharedEncs.front())
    return std::nullopt;
  auto maxVecSharedEnc = sharedEncs.front();

  // TODO add support for padded layouts. Right now they will use a separate
  // allocation
  for (auto sharedEnc : llvm::drop_begin(sharedEncs, 1)) {
    auto maybeSwizzShared =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(sharedEnc);
    auto maybeSwizzMaxVec =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(maxVecSharedEnc);

    if (!equalSharedEncIgnoreVec(maybeSwizzShared, maybeSwizzMaxVec)) {
      LDBG("Incompatible shared encodings");
      return std::nullopt;
    }
    if (maybeSwizzShared.getVec() > maybeSwizzMaxVec.getVec()) {
      maxVecSharedEnc = sharedEnc;
    }
  }

  LDBG("Deduced shared encoding: " << maxVecSharedEnc);

  return maxVecSharedEnc;
}

} // namespace mlir
