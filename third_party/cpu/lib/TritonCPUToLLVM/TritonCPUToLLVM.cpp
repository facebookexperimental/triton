#include "TritonCPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "PatternTritonCPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONCPUTOLLVM
#include "TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::CPU;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addIllegalDialect<NVVM::NVVMDialect>();
  }
};

// Copied from the CPU fork.
struct CPUBarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using BarrierOp = mlir::gpu::BarrierOp;
  explicit CPUBarrierOpConversion(LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<BarrierOp>(typeConverter) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just make it a no-op for now
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTritonCPUToLLVM
    : public triton::impl::ConvertTritonCPUToLLVMBase<ConvertTritonCPUToLLVM> {
  using ConvertTritonCPUToLLVMBase::ConvertTritonCPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  ConvertTritonCPUToLLVM(const std::set<std::string> &cpuFeatures)
      : ConvertTritonCPUToLLVMBase({cpuFeatures}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    assert(mod->getAttrOfType<StringAttr>("ttg.target").getValue() == "cpu");

    triton::gpu::enableCPUMode();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TargetInfo targetInfo;
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    assert(threadsPerWarp == 1 && "Only support 1 thread per warp for now");
    assert(numWarps == 1 && "Only support 1 warp for now");
    assert(numCTAs == 1 && "Must be single CTAs");

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                                    numWarps, targetInfo,
                                                    patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, targetInfo, benefit);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    patterns.add<CPUBarrierOpConversion>(typeConverter);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::CPU::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonCPUToLLVMPass(const std::set<std::string> &cpuFeatures) {
  return std::make_unique<ConvertTritonCPUToLLVM>(cpuFeatures);
}

} // namespace triton
} // namespace mlir
