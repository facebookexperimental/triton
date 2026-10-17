#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::isPermutationMatrixLayout;
using ::mlir::triton::gpu::toLinearLayout;

namespace mlir::triton::AMD {
LogicalResult convertAMDFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult
convertScaledMFMAGroup(triton::amdgpu::ScaledMfmaGroupOp op,
                       triton::amdgpu::ScaledMfmaGroupOp::Adaptor adaptor,
                       const LLVMTypeConverter *typeConverter,
                       ConversionPatternRewriter &rewriter,
                       bool loadReadyOperands);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::AMD

namespace {
// Only these data operations can preserve the short entry proof.
// Do not infer readiness from purity, layouts, or user attributes.
bool isScaledGroupDataForwardingOp(Operation *op) {
  namespace ttg = mlir::triton::gpu;
  if (isa<arith::ExtUIOp, arith::TruncIOp, arith::ShRUIOp, arith::AndIOp,
          arith::AddIOp, arith::MulIOp, arith::OrIOp, arith::CmpIOp>(op)) {
    for (Type type : op->getOperandTypes())
      if (!isa<IntegerType, IndexType>(getElementTypeOrSelf(type)))
        return false;
    for (Type type : op->getResultTypes())
      if (!isa<IntegerType, IndexType>(getElementTypeOrSelf(type)))
        return false;
    return true;
  }
  if (isa<arith::SelectOp, arith::BitcastOp, triton::BitcastOp, triton::SplatOp,
          triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
          triton::TransOp, ttg::ConvertLayoutOp, ttg::RequireLayoutOp,
          ttg::ReleaseLayoutOp>(op))
    return true;
  // The AMD backend does not link the TLX dialect, so match TLX ops by name.
  StringRef name = op->getName().getStringRef();
  return name == "tlx.require_layout" || name == "tlx.release_layout";
}

bool hasLoadReadyScaledGroupInputs(triton::amdgpu::ScaledMfmaGroupOp op) {
  SmallVector<Value> pending{op.getA(), op.getAScale(), op.getB(),
                             op.getBScale()};
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    // Block arguments and control-flow results do not prove input readiness.
    if (!producer)
      return false;
    if (isa<arith::ConstantOp, triton::MakeRangeOp>(producer))
      continue;
    if (auto load = dyn_cast<triton::LoadOp>(producer)) {
      // Masked lanes can forward an unproved fallback value.
      if (load.getMask() || load.getOther())
        return false;
      continue;
    }
    if (isa<triton::gpu::LocalLoadOp>(producer))
      continue;
    if (!isScaledGroupDataForwardingOp(producer))
      return false;
    // Check the condition and both branches of every select.
    pending.append(producer->operand_begin(), producer->operand_end());
  }
  return true;
}

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    auto dType = op.getD().getType();
    auto dEncoding = dType.getEncoding();

    // WMMA is the only path that has been validated against non-permutation
    // (e.g. swizzled-warp) encodings, so handle it up front and then enforce
    // the permutation-matrix invariant for all other paths.
    if (isa<AMDWmmaEncodingAttr>(dEncoding))
      return AMD::convertWMMA(op, adaptor, getTypeConverter(), rewriter);

    if (!isPermutationMatrixLayout(toLinearLayout(dType.getShape(), dEncoding)))
      return rewriter.notifyMatchFailure(op,
                                         "Non-WMMA DotOp result encoding must "
                                         "have a permutation-matrix linear "
                                         "layout");

    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return AMD::convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }

    if (isa<BlockedEncodingAttr>(dEncoding))
      return AMD::convertAMDFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct ScaledMfmaGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ScaledMfmaGroupOp> {
  ScaledMfmaGroupOpConversion(
      LLVMTypeConverter &typeConverter, const AMD::TargetInfo &targetInfo,
      const llvm::DenseSet<Operation *> &loadReadyGroups,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo),
        loadReadyGroups(loadReadyGroups) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::ScaledMfmaGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo.getArch() != "gfx950")
      return op.emitOpError("is supported only on gfx950");
    return AMD::convertScaledMFMAGroup(
        op, adaptor, getTypeConverter(), rewriter,
        loadReadyGroups.contains(op.getOperation()));
  }

private:
  const AMD::TargetInfo &targetInfo;
  const llvm::DenseSet<Operation *> &loadReadyGroups;
};

struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dType = op.getD().getType();
    auto dEncoding = dType.getEncoding();

    if (isa<AMDWmmaEncodingAttr>(dEncoding))
      return AMD::convertScaledWMMA(op, adaptor, getTypeConverter(), rewriter);

    if (!isPermutationMatrixLayout(toLinearLayout(dType.getShape(), dEncoding)))
      return rewriter.notifyMatchFailure(
          op, "non-WMMA dot encoding must have a permutation-matrix linear "
              "layout");

    if (isa<AMDMfmaEncodingAttr>(dEncoding))
      return AMD::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotScaleOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

namespace mlir::triton::AMD {
llvm::DenseSet<Operation *> collectLoadReadyScaledMfmaGroups(ModuleOp mod) {
  llvm::DenseSet<Operation *> loadReadyGroups;
  mod.walk([&](triton::amdgpu::ScaledMfmaGroupOp op) {
    if (hasLoadReadyScaledGroupInputs(op))
      loadReadyGroups.insert(op.getOperation());
  });
  return loadReadyGroups;
}

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfo &targetInfo,
    const llvm::DenseSet<Operation *> &loadReadyGroups,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, benefit);
  patterns.add<ScaledMfmaGroupOpConversion>(typeConverter, targetInfo,
                                            loadReadyGroups, benefit);
}
} // namespace mlir::triton::AMD
