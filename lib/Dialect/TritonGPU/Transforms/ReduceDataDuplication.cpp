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
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> WalkResult {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return WalkResult::advance();
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return WalkResult::advance();
      if (!cvtNeedsSharedMemory(cvtOp))
        return WalkResult::advance();
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto sharedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
          mod.getContext(), dstDotOp, srcType.getShape(), order,
          triton::gpu::getCGALayout(srcEncoding), srcType.getElementType());
      if (!isLegalMemDescLayoutShape(dstType.getShape(), sharedEncoding)) {
        cvtOp.emitError("shared-memory staging of an NPOT dot operand is not "
                        "yet supported (a non-leading dim is not a power of "
                        "2)");
        signalPassFailure();
        return WalkResult::interrupt();
      }
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(), sharedEncoding,
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
      return WalkResult::advance();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
