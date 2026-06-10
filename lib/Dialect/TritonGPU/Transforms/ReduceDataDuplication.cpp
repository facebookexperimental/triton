#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      bool isNpot = hasNpotShape(dstType) || hasNpotShape(srcType);
      // Guard NPOT shapes before cvtNeedsSharedMemory, which calls
      // toLinearLayout and may crash on encodings that don't handle NPOT.
      if (isNpot && !npotSafeForLinearLayout(dstDotOp.getParent()))
        return;
      if (!cvtNeedsSharedMemory(srcType, dstType))
        return;
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto CGALayout = triton::gpu::getCGALayout(srcEncoding);
      Attribute sharedEnc;
      auto parentMma =
          dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(dstDotOp.getParent());
      bool isMMAv3Plus = parentMma && parentMma.getVersionMajor() >= 3;
      // CORRECTNESS (AMD): getNVIDIAComputeCapability asserts a `cuda:` target
      // and aborts (dev) / reads `cc` uninitialized (opt, UB) on HIP modules.
      // This pass runs in the AMD pipeline, so only query compute capability on
      // the NVIDIA-MMA branch where it is actually used.
      bool isNpotNvidiaMMAv3Plus =
          isNpot && isMMAv3Plus && getNVIDIAComputeCapability(mod) >= 90;
      if (isNpotNvidiaMMAv3Plus) {
        sharedEnc = triton::gpu::NVMMASharedEncodingAttr::get(
            mod.getContext(), srcType.getShape(), order, CGALayout,
            srcType.getElementType(), /*fp4Padded=*/false);
      } else if (isNpot) {
        // For NPOT shapes with MMAv2 (or other safe encodings), use
        // SwizzledSharedEncodingAttr which handles NPOT via the standard path.
        // combineCtaCgaWithShape applies modular reduction.
        sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
            mod.getContext(), dstDotOp, srcType.getShape(), order, CGALayout,
            srcType.getElementType());
      } else {
        sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
            mod.getContext(), dstDotOp, srcType.getShape(), order, CGALayout,
            srcType.getElementType());
      }
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(), sharedEnc,
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
