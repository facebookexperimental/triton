#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

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
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();
    if (failed(maybeInsertClusterSync(mod))) {
      return signalPassFailure();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
        typeConverter, patterns, patternBenefitNvidiaTensorCoreSubviewPattern);
    mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo,
                                                    patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        targetInfo, benefit);
    populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                  computeCapability,
                                  patternBenefitClampOptimizedPattern);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                      computeCapability, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit,
                                    targetInfo);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                      benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
        typeConverter, patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter,
                                                           patterns, benefit);
    mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                        benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately to avoid breaking analysis.
    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    const SmallVector<int> clusterIds =
        triton::gpu::TritonGPUDialect::getClusterDims(mod);
    bool isClustered = ((clusterIds[0] * clusterIds[1] * clusterIds[2]) > 1);
    if (numCTAs == 1 && !tlx::tlxEnablePairedMMA(mod) && !isClustered) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }

  void getBackwardSliceWithWS(Value target,
                              SetVector<Operation *> *backwardSlice) {
    SetVector<Value> worklist;
    worklist.insert(target);

    BackwardSliceOptions options;
    options.omitUsesFromAbove = false;
    options.omitBlockArguments = true;
    options.inclusive = true;

    while (!worklist.empty()) {
      Value nextTarget = worklist.back();
      worklist.pop_back();

      if (auto arg = dyn_cast<BlockArgument>(nextTarget)) {
        if (auto wsPartitionOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(
                arg.getOwner()->getParentOp())) {
          auto argIndex = arg.getArgNumber();
          auto wsOp = wsPartitionOp->getParentOfType<ttg::WarpSpecializeOp>();
          // map to WSOp's operand at the same index
          nextTarget = wsOp.getOperand(argIndex);
        } else {
          // ttg::WarpSpecializeOp's default region just captures
          // from trunk so no need to special handle the defining block args.
          // We should omit block args for other block structures like scf.For,
          // and the captures would still be handled automatically
          continue;
        }
      }

      SetVector<Operation *> ops;
      if (failed(getBackwardSlice(nextTarget, &ops, options))) {
        llvm_unreachable("getBackwardSlice failed");
      }

      for (auto op : ops) {
        if (backwardSlice->insert(op)) {
          for (auto operand : op->getOperands()) {
            if (isa<BlockArgument>(operand)) {
              worklist.insert(operand);
            }
          }
        }
      }
    }
  }

  LogicalResult
  ensureEarlyRemoteBarInit(ModuleOp &mod,
                           SetVector<Operation *> &remoteBarInitOps) {
    triton::FuncOp funcOp = nullptr;
    mod.walk([&](triton::FuncOp op) {
      if (triton::isKernel(op)) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    assert(funcOp && "Expecting to find a kernel func but got none.");
    for (auto op : remoteBarInitOps) {
      if (op->getBlock() != &funcOp.front()) {
        op->emitError() << "Barrier init outside of the first block in "
                           "function is not supported for remote barriers";
        return failure();
      }
    }

    return success();
  }

  // If we're doing TLX 2cta paired MMA, insert cluster sync properly to
  // bootstrap remote bars
  LogicalResult maybeInsertClusterSync(ModuleOp &mod) {
    if (!tlx::tlxEnablePairedMMA(mod)) {
      return success();
    }

    bool hasMapaOp = false;
    SetVector<Operation *> barAllocOps;
    // Find all bar alloc op in the back slice of mapa ops
    mod.walk([&](ttng::MapToRemoteBufferOp mapaOp) {
      hasMapaOp = true;
      SetVector<Operation *> ops;
      getBackwardSliceWithWS(mapaOp.getResult(), &ops);

      for (auto op : ops) {
        if (isa<ttg::LocalAllocOp>(op)) {
          barAllocOps.insert(op);
        }
      }
    });

    // If there's no mapa, it's not possible to access remote barrier so
    // skipping
    if (!hasMapaOp) {
      return success();
    }

    assert(!barAllocOps.empty() &&
           "Failed to find bar alloc op for remote bar");

    // Find the init op for remote barriers
    SetVector<Operation *> remoteBarInitOps;
    mod.walk([&](ttng::InitBarrierOp barInitOp) {
      SetVector<Operation *> ops;
      getBackwardSliceWithWS(barInitOp.getAlloc(), &ops);
      if (llvm::any_of(
              ops, [&](Operation *op) { return barAllocOps.contains(op); })) {
        // barInitOp is for remote bar
        remoteBarInitOps.insert(barInitOp);
      }
    });

    assert(!remoteBarInitOps.empty() &&
           "Failed to find bar init op for remote bar");

    // Enforcing front end for 2cta kernels:
    // All remote barrier init ops need to happen at the first block of
    // function. This is to make 2cta cluster sync insertion easier for WarpSpec
    // case. If in the future there's a need to really alloc/init barriers after
    // a WS op, we can seek to relax this limitation and fix cluster sync
    // insertions.
    if (failed(ensureEarlyRemoteBarInit(mod, remoteBarInitOps))) {
      return failure();
    }

    // Follow the program order and identify the last bar init op.
    // This is based on the assumption that all bar init happens at the first
    // block of the kernel func op, as we currently enforce earlier in this
    // pass. If that assumption changes, we should revisit this heuristic here.
    ttng::InitBarrierOp lastBarInitOp;
    auto firstBlock = remoteBarInitOps.front()->getBlock();
    for (auto it = firstBlock->rbegin(), e = firstBlock->rend(); it != e;
         ++it) {
      if (remoteBarInitOps.contains(&*it)) {
        lastBarInitOp = cast<ttng::InitBarrierOp>(*it);
        break;
      }
    }

    OpBuilder builder(lastBarInitOp);
    builder.setInsertionPointAfter(lastBarInitOp);
    // need to insert cluster arrive and wait to prevent CTA_X from arriving
    // CTA_Y's bar before CTA_Y inits it, as shown in ptx doc examples:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
    builder.create<ttng::ClusterArriveOp>(lastBarInitOp.getLoc(),
                                          /*relaxed*/ false);
    builder.create<ttng::ClusterWaitOp>(lastBarInitOp.getLoc());

    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability,
                                                  ptxVersion);
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after) {
  // Multiple init barriers on the same allocation would usually not happen but
  // that allows us to avoid barriers between multiple subslice of an array of
  // mbarriers. This is still correct even if the inits happen on the same
  // allocation.
  if (isa<triton::nvidia_gpu::InitBarrierOp>(before) &&
      isa<triton::nvidia_gpu::InitBarrierOp>(after))
    return true;

  if (isa<triton::nvidia_gpu::InvalBarrierOp>(before) &&
      isa<triton::nvidia_gpu::InvalBarrierOp>(after))
    return true;

  //  We can't have a warp get ahead when we have a chain of mbarrier wait so we
  //  need a barrier in between two WaitBarrierOp.
  if (isa<triton::nvidia_gpu::WaitBarrierOp>(before) &&
      isa<triton::nvidia_gpu::WaitBarrierOp>(after))
    return false;

  // Even though WaitBarrierOp, AsyncTMACopyGlobalToLocalOp and
  // AsyncTMACopyGlobalToLocalOp read and write to the mbarrier allocation it is
  // valid for them to happen in different order on different threads, therefore
  // we don't need a barrier between those operations.
  if (isa<triton::nvidia_gpu::WaitBarrierOp,
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::BarrierExpectOp>(before) &&
      isa<triton::nvidia_gpu::WaitBarrierOp,
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::BarrierExpectOp>(after))
    return true;

  // A mbarrier wait is released only when the whole operations is done,
  // therefore any thread can access the memory after the barrier even if some
  // threads haven't reached the mbarrier wait.
  if (isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::WaitBarrierOp>(before) &&
      !isa<triton::nvidia_gpu::InvalBarrierOp>(after))
    return true;

  return false;
}

} // namespace triton
} // namespace mlir
