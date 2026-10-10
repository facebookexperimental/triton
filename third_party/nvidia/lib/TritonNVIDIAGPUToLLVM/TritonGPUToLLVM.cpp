#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"

#include "Allocation.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NamedBarrier.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

class NvidiaLLVMConversionTarget : public ConversionTarget {
public:
  explicit NvidiaLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    // This base is shared by scoped partial conversions, so only list IR that
    // is valid throughout every phase of this pass.
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public NvidiaLLVMConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : NvidiaLLVMConversionTarget(ctx) {
    // CF is lowered after the axis-info-dependent patterns have finished.
    addLegalDialect<cf::ControlFlowDialect>();
    // The custom NVGPU dialect is LLVM-level IR lowered by a subsequent pass.
    addLegalDialect<triton::nvgpu::NVGPUDialect>();

    // Leave extension dialects (e.g., proton) unclassified so partial
    // conversion can preserve them for their own downstream lowering passes.
    addIllegalDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                      triton::nvidia_gpu::TritonNvidiaGPUDialect,
                      triton::instrument::TritonInstrumentDialect,
                      mlir::gpu::GPUDialect>();

    // Warp specialization and warp ID are lowered by subsequent passes.
    addLegalOp<triton::gpu::WarpIdOp, triton::gpu::WarpSpecializeOp,
               triton::gpu::WarpYieldOp,
               triton::gpu::WarpSpecializePartitionsOp,
               triton::gpu::WarpReturnOp>();
    // TCGen5GlobalAllocOp survives until NVGPUToLLVM, despite its dialect
    // otherwise being illegal at this stage.
    addLegalOp<triton::nvidia_gpu::TCGen5GlobalAllocOp>();
  }
};

void createSharedMemoryGlobal(ModuleOp mod, LLVMTypeConverter &typeConverter) {
  OpBuilder builder(mod.getBodyRegion());
  Type elemTy = typeConverter.convertType(builder.getIntegerType(8));
  // A zero-sized array with external linkage represents dynamic shared memory.
  // Request 16-byte alignment because 4xi32 is the widest supported access.
  auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
  LLVM::GlobalOp::create(
      builder, mod.getLoc(), arrayTy, /*isConstant=*/false,
      LLVM::Linkage::External, "global_smem", /*value=*/Attribute(),
      /*alignment=*/16, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
}

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion,
                         bool enableConcurrencySanitizer)
      : ConvertTritonGPUToLLVMBase(
            {computeCapability, ptxVersion, enableConcurrencySanitizer}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion,
                         bool enableConcurrencySanitizer,
                         bool enableTreeReduction)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion,
                                    enableConcurrencySanitizer,
                                    enableTreeReduction}) {}

  void runOnOperation() override;

private:
  LogicalResult prepareModule(ModuleOp mod, TargetInfo &targetInfo);
  LogicalResult lowerFunctions(ModuleOp mod, LLVMTypeConverter &typeConverter,
                               TargetInfo &targetInfo);
  void populateConversionPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  TargetInfo &targetInfo);
  LogicalResult lowerTritonGPUOps(ModuleOp mod,
                                  LLVMTypeConverter &typeConverter,
                                  TargetInfo &targetInfo);
  LogicalResult lowerControlFlow(ModuleOp mod,
                                 LLVMTypeConverter &typeConverter);
  void finalizeModule(ModuleOp mod);

  // Push entry-block mbarrier init ops early enough to preserve the
  // mbarrier-init -> cluster-sync -> TMEM-alloc sequence. Barrier init ops in
  // WS partition bodies are local pipeline barriers; they do not participate in
  // this kernel-entry cluster synchronization point.
  LogicalResult ensureEarlyEntryBarInit(ModuleOp &mod,
                                        SetVector<Operation *> &barInitOps) {
    triton::FuncOp funcOp = nullptr;
    mod.walk([&](triton::FuncOp op) {
      if (triton::isKernel(op)) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    assert(funcOp && "Expecting to find a kernel func but got none.");
    Block *entryBlock = &funcOp.front();

    // Move all mbar init ops (and their deps) to the beginning of the block
    SetVector<Operation *> opsToMove;
    SmallVector<Operation *> worklist(barInitOps.begin(), barInitOps.end());
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      if (!opsToMove.insert(op))
        continue;
      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == entryBlock)
            worklist.push_back(defOp);
        }
      }
    }
    SmallVector<Operation *> opsInBlockOrder;
    for (auto &op : *entryBlock) {
      if (opsToMove.contains(&op))
        opsInBlockOrder.push_back(&op);
    }
    Operation *insertPt = nullptr;
    for (auto *op : opsInBlockOrder) {
      if (!insertPt)
        op->moveBefore(entryBlock, entryBlock->begin());
      else
        op->moveAfter(insertPt);
      insertPt = op;
    }

    // Check the block again to make sure all entry-block mbar init ops are
    // earlier than the first entry-block TMEM allocation op.
    Operation *firstTMEMAlloc = nullptr;
    for (auto &op : *entryBlock) {
      if (isa<ttng::TCGen5GlobalAllocOp, ttng::TMEMAllocOp>(&op)) {
        firstTMEMAlloc = &op;
        break;
      }
    }
    if (firstTMEMAlloc) {
      for (auto *op : barInitOps) {
        if (!op->isBeforeInBlock(firstTMEMAlloc)) {
          op->emitError() << "Barrier init is not before TMEM allocation. "
                             "Cannot insert cluster sync between them.";
          return failure();
        }
      }
    }

    return success();
  }

  // Return the operand or result Value of a given op if the Value is used for
  // cross CTA mbarrier arrival. This function assumes the kernel has cluster
  // size larger than 1.
  std::optional<SetVector<Value>> getRemoteBarrier(Operation *op) {
    if (auto mapaOp = llvm::dyn_cast<ttng::MapToRemoteBufferOp>(op)) {
      // plain cross CTA mbarrier arrive and cross CTA DSMEM store/copy need
      // mapa to map mbarrier addr explicitly
      llvm::SetVector<Value> bars;
      bars.insert(mapaOp.getResult());
      return bars;
    } else if (auto tmaLoadOp =
                   llvm::dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      // If it's a TMA load with multicast, the mbar signal is multicasted too
      if (tmaLoadOp.getMulticastTargets()) {
        llvm::SetVector<Value> bars;
        bars.insert(tmaLoadOp.getBarrier());
        return bars;
      }
    } else if (auto asyncCLCTryCancelOp =
                   llvm::dyn_cast<ttng::AsyncCLCTryCancelOp>(op)) {
      // If it's AsyncCLCTryCancelOp, the signal will be broadcasted to other
      // CTAs only when .multicast::cluster::all is specified, which is true now
      // no matter what cluster size is. Since we're assuming cluster size > 1,
      // we should consider the barrier here as remote barrier.
      llvm::SetVector<Value> bars;
      bars.insert(asyncCLCTryCancelOp.getMbarAlloc());
      return bars;
    } else if (auto clcTryCancelOp = llvm::dyn_cast<ttng::CLCTryCancelOp>(op)) {
      // The core Triton CLC scheduler also multicasts its response and
      // completion signal when an explicit physical cluster is configured.
      // All CTAs must finish initializing their local completion barriers
      // before the lead CTA can issue the request.
      llvm::SetVector<Value> bars;
      bars.insert(clcTryCancelOp.getMbarrier());
      return bars;
    } else if (auto tcgen5CommitOp = llvm::dyn_cast<ttng::TCGen5CommitOp>(op)) {
      // As of now, there're only three sources to have a tcgen05.commit
      // instruction:
      // 1. Front end supplied a TCGen5CommitOp directly
      // 2. When lowering gen5 TMEMCopy to llvm, compiler inserts inline ptx
      // 3. When lowering gen5 MMA to llvm, compiler inserts inline ptx
      // And the eventual tcgen05.commit has .multicast::cluster to broadcast
      // mbar signals to multiple CTAs only under 2cta mode.
      // https://github.com/facebookexperimental/triton/blob/70d488dc45ca7e75432b0352cb9dd07b602a82cf/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp#L327
      // Although it's valid
      // to have .multicast::cluster for 1cta mode too, there's currently no
      // support for it.

      // Cases 1 and 2 will read module attribute for 2cta mode, case 3 will
      // read module attr or op arg for 2cta mode, which are equivalent since
      // all tcgen05 ops have to be consistent with module attr on this.

      // Case 1: explicit TCGen5CommitOp from front end or earlier passes
      if (!tcgen5CommitOp.getDescs().empty()) {
        llvm::SetVector<Value> bars;
        bars.insert(tcgen5CommitOp.getBarrier());
        return bars;
      }
    } else if (llvm::isa<ttng::MMAv5OpInterface>(op)) {
      // case 3 for gen5 commit: a commit inline ptx will be generated for each
      // barrier on the gen5 MMA op. If the mod is in 2cta mode, the commit op
      // can multicast bar signals.
      if (tlx::tlxEnablePairedMMA(op)) {
        llvm::SetVector<Value> bars;
        // TODO: move getBarriers() into MMAv5OpInterface to simplify this
        if (auto mma = llvm::dyn_cast<ttng::TCGen5MMAOp>(op)) {
          for (auto bar : mma.getBarriers()) {
            bars.insert(bar);
          }
        } else {
          // "assert" it's a scaled MMA op so that we crash explicitly if new
          // MMAv5OpInterface is added
          auto scaledMMA = llvm::cast<ttng::TCGen5MMAScaledOp>(op);
          for (auto bar : scaledMMA.getBarriers()) {
            bars.insert(bar);
          }
        }
        return bars;
      }
    }

    return std::nullopt;
  }

  // If the kernel is clustered, insert cluster sync properly to
  // bootstrap remote bars or tmem
  LogicalResult maybeInsertClusterSync(ModuleOp &mod) {
    if (!triton::gpu::isPhysicalCluster(mod)) {
      return success();
    }

    bool hasRemoteBar = false;
    bool hasMulticastArrive = false;
    // Find if we have a remote bar or multicast arrive.
    mod.walk([&](Operation *op) {
      if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
          arrive && arrive.isMulticast()) {
        hasMulticastArrive = true;
        return WalkResult::interrupt();
      }
      SetVector<Operation *> ops;
      auto remoteBar = getRemoteBarrier(op);
      if (remoteBar.has_value()) {
        hasRemoteBar = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // If we have remote mbar/SMEM access, a multicast arrive, or 2cta TMEM
    // allocation, we need a cluster sync after mbar init and before use.
    bool shouldInsert =
        hasRemoteBar || hasMulticastArrive || tlx::tlxEnablePairedMMA(mod);
    if (!shouldInsert) {
      return success();
    }

    // Find the kernel entry block and collect entry-block barrier inits.
    // Only entry-block barriers need the cluster fence+sync for cross-CTA
    // visibility. Barriers inside WS partition bodies are local pipeline
    // barriers that don't participate in cross-CTA communication.
    triton::FuncOp funcOp = nullptr;
    mod.walk([&](triton::FuncOp op) {
      if (triton::isKernel(op)) {
        funcOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    assert(funcOp && "Expecting to find a kernel func.");
    Block *entryBlock = &funcOp.front();

    SetVector<Operation *> entryBarInitOps;
    for (auto &op : *entryBlock) {
      if (isa<ttng::InitBarrierOp>(op))
        entryBarInitOps.insert(&op);
    }

    if (entryBarInitOps.empty())
      return success();

    if (failed(ensureEarlyEntryBarInit(mod, entryBarInitOps)))
      return failure();

    ttng::InitBarrierOp lastBarInitOp;
    for (auto it = entryBlock->rbegin(), e = entryBlock->rend(); it != e;
         ++it) {
      if (entryBarInitOps.contains(&*it)) {
        lastBarInitOp = cast<ttng::InitBarrierOp>(*it);
        break;
      }
    }

    OpBuilder builder(lastBarInitOp);
    builder.setInsertionPointAfter(lastBarInitOp);
    // need to insert fence to make mbar init visible to cluster
    ttng::FenceMBarrierInitReleaseClusterOp::create(builder,
                                                    lastBarInitOp.getLoc());
    // need to insert cluster arrive and wait to prevent CTA_X from arriving
    // CTA_Y's bar before CTA_Y inits it, as shown in ptx doc examples:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
    ttng::ClusterArriveOp::create(builder, lastBarInitOp.getLoc(),
                                  /*relaxed*/ true);
    ttng::ClusterWaitOp::create(builder, lastBarInitOp.getLoc());
    // mark mod attr so that WS lowering is aware of this cluster sync point
    tlx::setClusterSyncKernelInitOnMod(mod, true);
    return success();
  }
};

void ConvertTritonGPUToLLVM::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp mod = getOperation();
  TargetInfo targetInfo(computeCapability, ptxVersion);

  // These analyses and transformations require high-level shared-memory ops,
  // so they must all run before dialect conversion starts.
  if (failed(prepareModule(mod, targetInfo))) {
    signalPassFailure();
    return;
  }

  mlir::LowerToLLVMOptions option(context);
  option.overrideIndexBitwidth(32);
  TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

  if (failed(lowerFunctions(mod, typeConverter, targetInfo))) {
    signalPassFailure();
    return;
  }

  // The shared-memory global must exist before call and return conversion so
  // those patterns can resolve each function's shared-memory base address.
  createSharedMemoryGlobal(mod, typeConverter);

  // CF was kept while ModuleAxisInfoAnalysis was in use.
  // Lower it after all axis-info-dependent patterns have finished.
  if (failed(lowerTritonGPUOps(mod, typeConverter, targetInfo)) ||
      failed(lowerControlFlow(mod, typeConverter))) {
    signalPassFailure();
    return;
  }

  finalizeModule(mod);
}

LogicalResult ConvertTritonGPUToLLVM::prepareModule(ModuleOp mod,
                                                    TargetInfo &targetInfo) {
  // Assign warp-specialization barrier IDs before shared-memory allocation.
  mlir::triton::nvidia_gpu::NamedBarrierIdAllocator barrierAllocator(mod);
  if (failed(mlir::triton::nvidia_gpu::ensureWarpSpecializeBarrierIds(
          mod, barrierAllocator)))
    return failure();

  ModuleAllocation allocation(
      mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
               targetInfo));
  mlir::triton::nvidia_gpu::runClusterBarrierInsertion(allocation,
                                                       computeCapability);
  if (failed(mlir::triton::nvidia_gpu::runCrossCTAMBarrierInitSyncInsertion(
          allocation, computeCapability)))
    return failure();

  ModuleMembarAnalysis membarPass(allocation, canSkipBarSync);
  membarPass.run();
  if (failed(maybeInsertClusterSync(mod)))
    return failure();

  if (enableConcurrencySanitizer) {
    auto hooks = mlir::triton::instrument::createConSanHooks("nvidia");
    if (!hooks) {
      mod.emitError("no ConSan hooks registered for nvidia");
      return failure();
    }
    if (failed(mlir::triton::instrument::runConcurrencySanitizer(mod, *hooks)))
      return failure();

    // Normalize instrumentation-generated IR before allocation and lowering.
    mlir::PassManager cleanupPm(mod.getContext());
    cleanupPm.addPass(mlir::triton::gluon::createGluonCanonicalize());
    cleanupPm.addPass(mlir::createCSEPass());
    if (failed(cleanupPm.run(mod)))
      return failure();
  }

  mlir::triton::nvidia_gpu::runClusterBarrierMbarAllocator(mod);
  mod.walk([&](triton::gpu::GlobalScratchAllocOp) -> WalkResult {
    mlir::triton::gpu::runGlobalScratchMemoryAllocation(mod);
    return WalkResult::interrupt();
  });

  return success();
}

LogicalResult ConvertTritonGPUToLLVM::lowerFunctions(
    ModuleOp mod, LLVMTypeConverter &typeConverter, TargetInfo &targetInfo) {
  NvidiaLLVMConversionTarget target(*mod.getContext());
  target.addIllegalOp<triton::FuncOp>();
  RewritePatternSet patterns(mod.getContext());
  mlir::triton::populateFuncOpConversionPattern(
      typeConverter, patterns, targetInfo, patternBenefitDefault);
  return applyPartialConversion(mod, target, std::move(patterns));
}

void ConvertTritonGPUToLLVM::populateConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, TargetInfo &targetInfo) {
  const int benefit = patternBenefitPrioritizeOverLLVMConversions;
  mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
      typeConverter, patterns, patternBenefitNvidiaTensorCoreSubviewPattern);
  mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo,
                                                  patterns, benefit);
  populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                              benefit);
  populateElementwiseOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      computeCapability, targetInfo, benefit);
  populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                computeCapability,
                                patternBenefitClampOptimizedPattern);
  populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                    computeCapability, patterns,
                                    axisInfoAnalysis, benefit);
  mlir::triton::populateReduceOpToLLVMPatternsWithOptions(
      typeConverter, patterns, targetInfo, benefit, enableTreeReduction);
  mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                             targetInfo, benefit);
  mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
  populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit, targetInfo);
  populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit,
                                   targetInfo);
  mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                  targetInfo, benefit);
  mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                             targetInfo, benefit);
  mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);
  mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                    benefit);
  mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                            benefit);
  // TODO(thomas): this should probably be done in a separate step to not
  // interfere with our own lowering of arith ops. Add arith/math's patterns
  // to help convert scalar expression to LLVM.
  mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
  mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
  mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
  mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
  mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(typeConverter,
                                                            patterns, benefit);
  mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                 patterns, benefit);
  mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(
      typeConverter, patterns, benefit, targetInfo);
  mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                      benefit);
  mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo);
  mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);
  mlir::triton::populateGSanToLLVMPatterns(typeConverter, patterns,
                                           axisInfoAnalysis, targetInfo);
}

LogicalResult ConvertTritonGPUToLLVM::lowerTritonGPUOps(
    ModuleOp mod, LLVMTypeConverter &typeConverter, TargetInfo &targetInfo) {
  ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
  RewritePatternSet patterns(mod.getContext());
  populateConversionPatterns(typeConverter, patterns, axisInfoAnalysis,
                             targetInfo);
  TritonLLVMConversionTarget target(*mod.getContext());
  return applyPartialConversion(mod, target, std::move(patterns));
}

LogicalResult
ConvertTritonGPUToLLVM::lowerControlFlow(ModuleOp mod,
                                         LLVMTypeConverter &typeConverter) {
  MLIRContext *context = mod.getContext();
  NvidiaLLVMConversionTarget target(*context);
  target.addIllegalDialect<cf::ControlFlowDialect>();
  RewritePatternSet patterns(context);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  return applyPartialConversion(mod, target, std::move(patterns));
}

void ConvertTritonGPUToLLVM::finalizeModule(ModuleOp mod) {
  // Fold CTAId when there is only one CTA, under either cluster model.
  if (triton::gpu::lookupPhysicalNumCTAs(mod) == 1) {
    mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
      OpBuilder builder(id);
      Value zero = LLVM::createConstantI32(id->getLoc(), builder, 0);
      id.replaceAllUsesWith(zero);
    });
  }

  // elect.sync with membermask=-1 returns the same predicate in a convergence
  // region, but CSE will not deduplicate it because it has side effects.
  mod.walk([](Block *block) {
    NVVM::ElectSyncOp first = nullptr;
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto elect = dyn_cast<NVVM::ElectSyncOp>(&op)) {
        if (!first)
          first = elect;
        else {
          elect.replaceAllUsesWith(first.getResult());
          elect.erase();
        }
      }
    }
  });

  fixUpLoopAnnotation(mod);
  // Ensure warp-group code is isolated from above.
  makeAllWarpGroupsIsolatedFromAbove(mod);
}

} // anonymous namespace

namespace mlir::triton {

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

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion,
                                 bool enableConcurrencySanitizer) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability, ptxVersion,
                                                  enableConcurrencySanitizer);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion,
                                 bool enableConcurrencySanitizer,
                                 bool enableTreeReduction) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability, ptxVersion,
                                                  enableConcurrencySanitizer,
                                                  enableTreeReduction);
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after,
                            bool /*beforeIsRead*/, bool /*afterIsRead*/,
                            Allocation * /*allocation*/) {
  // These mbarrier ops are single threaded, so are always synchronized wrt.
  // each other.
  if (isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          before) &&
      isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::BarrierExpectOp>(
          after))
    return true;

  // wait_barrier will never run ahead of the load it's waiting on
  if (isa<ttng::TMALoadLikeOpInterface>(before) &&
      isa<ttng::WaitBarrierOp>(after))
    return true;

  // Identical same-width commutative atomics can be freely reordered.
  auto beforeAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(before);
  auto afterAtomic = dyn_cast<triton::gpu::LocalAtomicScatterRMWOp>(after);
  return beforeAtomic && afterAtomic && beforeAtomic.isCommutative() &&
         afterAtomic.isCommutative() &&
         beforeAtomic.getAtomicRmwOp() == afterAtomic.getAtomicRmwOp() &&
         beforeAtomic.getDst().getType().getElementType() ==
             afterAtomic.getDst().getType().getElementType();
}

} // namespace mlir::triton
