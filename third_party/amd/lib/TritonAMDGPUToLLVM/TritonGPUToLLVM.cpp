#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "TritonAMDGPUTransforms/SchedulingAttributes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONAMDGPUTOLLVM
#define GEN_PASS_DEF_FINALIZESCHEDULEDMFMAOPERANDS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// Materialize the schedule-group program at real LDS hazard boundaries. The
// TTGIR pass provides the scheduling decision and exact machine counts; this
// stage only gains the boundaries after ModuleMembarAnalysis.
static void materializeDeferredSchedGroupBarriers(ModuleOp mod) {
  if (!mod->hasAttr("ttg.amd.sched_group_barrier.enabled"))
    return;

  unsigned requiredRegionCount = 0;
  if (auto attr = mod->getAttrOfType<IntegerAttr>(
          "ttg.amd.sched_group_barrier.required_region_count"))
    requiredRegionCount = static_cast<unsigned>(attr.getInt());

  unsigned nextSyncId = 1;
  mod.walk([&](triton::FuncOp func) {
    for (Block &block : func.getBody()) {
      // Explicit intra-wave regions already carry their complete user-chosen
      // scheduling-group program. Do not overlay the automatic whole-block
      // planner when both features are enabled for the same kernel.
      if (llvm::any_of(block.without_terminator(), [](Operation &op) {
            return isa<ROCDL::SchedGroupBarrier>(op) &&
                   op.hasAttr(triton::AMD::kIntraWavePipelineWindowAttr);
          }))
        continue;

      SmallVector<Operation *> boundaries;
      for (Operation &op : block.without_terminator())
        if (isa<triton::gpu::BarrierOp>(op))
          boundaries.push_back(&op);

      SmallVector<SmallVector<Operation *>, 4> regions(1);
      unsigned totalMFMA = 0;
      unsigned totalAnchors = 0;
      for (Operation &op : block.without_terminator()) {
        if (isa<triton::gpu::BarrierOp>(op)) {
          regions.emplace_back();
          continue;
        }
        auto mask = op.getAttrOfType<IntegerAttr>(
            "ttg.amd.sched_group_barrier.machine_mask");
        auto count = op.getAttrOfType<IntegerAttr>(
            "ttg.amd.sched_group_barrier.machine_count");
        if (!mask || !count)
          continue;
        regions.back().push_back(&op);
        if (mask.getInt() == (1 << 3))
          totalMFMA += static_cast<unsigned>(count.getInt());
        else
          totalAnchors += static_cast<unsigned>(count.getInt());
      }
      if (totalMFMA == 0 || totalAnchors == 0 ||
          (requiredRegionCount && regions.size() != requiredRegionCount))
        continue;
      boundaries.push_back(block.getTerminator());

      Location loc = block.front().getLoc();
      OpBuilder startBuilder(&block.front());
      ROCDL::SchedBarrier::create(startBuilder, loc,
                                  ROCDL::SchedGroupMask::none);

      for (auto [regionIndex, pair] :
           llvm::enumerate(llvm::zip(regions, boundaries))) {
        auto &[ops, boundary] = pair;
        struct Anchor {
          int32_t mask;
          unsigned count;
          unsigned mfmaCover;
        };
        SmallVector<Anchor> anchors;
        unsigned mfmas = 0;
        unsigned numWrites = 0;
        for (Operation *op : ops) {
          auto mask = op->getAttrOfType<IntegerAttr>(
              "ttg.amd.sched_group_barrier.machine_mask");
          auto count = op->getAttrOfType<IntegerAttr>(
              "ttg.amd.sched_group_barrier.machine_count");
          unsigned n = static_cast<unsigned>(count.getInt());
          if (mask.getInt() == (1 << 3)) {
            mfmas += n;
            continue;
          }
          unsigned mfmaCover = 0;
          if (auto attr = op->getAttrOfType<IntegerAttr>(
                  "ttg.amd.sched_group_barrier.mfma_cover"))
            mfmaCover = static_cast<unsigned>(attr.getInt());
          anchors.push_back(
              {static_cast<int32_t>(mask.getInt()), n, mfmaCover});
          if (mask.getInt() == (1 << 9))
            numWrites += n;
        }

        // In a memory-only prologue, issue the first global load after the
        // first local-load operation instead of after every LDS read.
        if (mfmas == 0 && anchors.size() > 2) {
          auto firstGlobal = llvm::find_if(
              anchors, [](const Anchor &a) { return a.mask == (1 << 5); });
          if (firstGlobal != anchors.end() &&
              firstGlobal != anchors.begin() + 1) {
            Anchor global = *firstGlobal;
            anchors.erase(firstGlobal);
            anchors.insert(anchors.begin() + 1, global);
          }
        }

        unsigned fixedCover = 0;
        for (const Anchor &anchor : anchors) {
          if (anchor.mask == (1 << 5))
            fixedCover += anchor.mfmaCover * anchor.count;
          else if (anchor.mask == (1 << 8))
            fixedCover += anchor.count;
        }
        unsigned writeCover = 1;
        if (numWrites && mfmas > fixedCover)
          writeCover = std::max(1u, (mfmas - fixedCover) / numWrites);

        struct Group {
          int32_t mask;
          unsigned cover;
        };
        SmallVector<Group> groups;
        for (const Anchor &anchor : anchors) {
          unsigned cover = anchor.mask == (1 << 5)   ? anchor.mfmaCover
                           : anchor.mask == (1 << 8) ? 1
                           : anchor.mask == (1 << 9) ? writeCover
                                                     : 0;
          for (unsigned i = 0; i < anchor.count; ++i)
            groups.push_back({anchor.mask, mfmas ? cover : 0});
        }

        OpBuilder builder(boundary);
        unsigned syncId = nextSyncId++;
        for (const Group &group : groups) {
          ROCDL::SchedGroupBarrier::create(
              builder, loc, static_cast<ROCDL::SchedGroupMask>(group.mask), 1,
              syncId);
          if (group.cover)
            ROCDL::SchedGroupBarrier::create(builder, loc,
                                             ROCDL::SchedGroupMask::mfma_wmma,
                                             group.cover, syncId);
        }
        if (regionIndex + 1 < regions.size())
          ROCDL::SchedBarrier::create(builder, loc,
                                      ROCDL::SchedGroupMask::none);
      }

      OpBuilder endBuilder(block.getTerminator());
      ROCDL::SchedBarrier::create(endBuilder, loc, ROCDL::SchedGroupMask::none);
    }
  });
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<triton::instrument::TritonInstrumentDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
    // Predicated regions are lowered after their bodies have been converted
    // to LLVM by TritonAMDGPUConvertWarpSpecializeToLLVM.
    addLegalOp<triton::gpu::WarpPredicateOp>();
    addLegalOp<triton::gpu::PredicateYieldOp>();
    // These have no lowering after this pass, so a pattern that bails out
    // must fail the conversion instead of leaving the op behind. The rest of
    // the dialect stays unmarked: some of it is lowered later.
    addIllegalOp<triton::amdgpu::ScheduledMfmaOp>();
    addIllegalOp<triton::amdgpu::MfmaCommitOp>();
  }
};

// TLX layout propagation is allowed to create captured or yielded
// convert_layout operations temporarily while it reconciles a predicate
// body's native layout with its carried values. By LLVM lowering, every such
// unsafe conversion must have moved outside the region. Non-wave-uniform EXEC
// forbids any cross-lane transfer; wave-uniform EXEC permits warp shuffles but
// not conversions that lower through CTA shared memory and a barrier.
static LogicalResult validateFinalWarpPredicateLayouts(ModuleOp mod) {
  WalkResult result = mod.walk([&](triton::gpu::WarpPredicateOp predicateOp) {
    bool waveUniform = predicateOp.getWaveUniform().value_or(false);
    triton::gpu::ConvertLayoutOp unsafeConvert;
    predicateOp.getRegion().walk([&](Operation *nested) {
      if (nested != predicateOp.getOperation() &&
          isa<triton::gpu::WarpPredicateOp>(nested))
        return WalkResult::skip();
      auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(nested);
      if (!convert)
        return WalkResult::advance();
      auto srcType = cast<RankedTensorType>(convert.getSrc().getType());
      auto dstType = cast<RankedTensorType>(convert.getType());
      if (triton::gpu::toLinearLayout(srcType) ==
          triton::gpu::toLinearLayout(dstType))
        return WalkResult::advance();
      if (waveUniform && (triton::gpu::lookupNumWarps(predicateOp) == 1 ||
                          !cvtNeedsSharedMemory(srcType, dstType)))
        return WalkResult::advance();
      unsafeConvert = convert;
      return WalkResult::interrupt();
    });

    if (!unsafeConvert)
      return WalkResult::advance();
    if (waveUniform)
      predicateOp.emitError(
          "wave-uniform body still contains a shared-memory layout conversion");
    else
      predicateOp.emitError(
          "non-wave-uniform body still contains cross-lane layout conversion");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

// Only order operands forwarded entirely from LDS reads. In particular, a
// computed or register-resident operand must not acquire this ordering merely
// because part of its backward slice contains a load.
static bool isLocalLoadDerived(Value operand) {
  DenseSet<Value> visited;
  DenseMap<Value, SmallVector<Value>> dependents;
  SmallVector<Value> worklist{operand};
  SmallVector<Value> loadRoots;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    if (!isa<RankedTensorType>(value.getType()))
      return false;
    if (value.getDefiningOp<triton::gpu::LocalLoadOp>()) {
      // The descriptor, indices, and synchronization token do not contribute
      // tensor elements, so the local load itself is the terminal root.
      loadRoots.push_back(value);
      continue;
    }

    SmallVector<Value> sources;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *block = arg.getOwner();
      // Region entry arguments need their own forwarding semantics. The AMD
      // pipeline lowers SCF to CFG before this pass; unsupported regions and
      // function arguments conservatively retain pure operand pins.
      if (block->isEntryBlock() || block->hasNoPredecessors())
        return false;
      for (auto pred = block->pred_begin(), end = block->pred_end();
           pred != end; ++pred) {
        auto branch = dyn_cast<BranchOpInterface>((*pred)->getTerminator());
        if (!branch)
          return false;
        // Inspect every edge, including distinct edges from the same block.
        SuccessorOperands incoming =
            branch.getSuccessorOperands(pred.getSuccessorIndex());
        unsigned index = arg.getArgNumber();
        if (index >= incoming.size() || incoming.isOperandProduced(index))
          return false;
        sources.push_back(incoming[index]);
      }
    } else {
      Operation *def = value.getDefiningOp();
      if (!def ||
          !isa<triton::gpu::ConvertLayoutOp, triton::ReshapeOp, triton::TransOp,
               triton::ExpandDimsOp, triton::BroadcastOp, triton::SplitOp,
               triton::amdgpu::ExtractSliceOp>(def))
        return false;
      // These operations only rearrange, duplicate, or select source elements.
      sources.push_back(def->getOperand(0));
    }
    for (Value source : sources) {
      dependents[source].push_back(value);
      worklist.push_back(source);
    }
  }

  // A visited cycle alone is not proof of a load origin. Remove every value
  // reachable forward from a load; any remaining forwarding cycle is rootless.
  // This accepts loop-carried values only when all their sources are supported
  // and each value reaches a load through an initialization or backedge.
  worklist = std::move(loadRoots);
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.erase(value))
      continue;
    auto it = dependents.find(value);
    if (it != dependents.end())
      llvm::append_range(worklist, it->second);
  }
  return visited.empty();
}

struct ConvertTritonAMDGPUToLLVM
    : public triton::impl::ConvertTritonAMDGPUToLLVMBase<
          ConvertTritonAMDGPUToLLVM> {
  explicit ConvertTritonAMDGPUToLLVM(StringRef gfxArch, bool ftz) {
    this->gfxArch = gfxArch.str();
    this->ftz = ftz;
  }

  ConvertTritonAMDGPUToLLVM(StringRef gfxArch, bool ftz,
                            bool enableTreeReduction) {
    this->gfxArch = gfxArch.str();
    this->ftz = ftz;
    this->enableTreeReduction = enableTreeReduction;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::amdgpu::TritonAMDGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    if (failed(validateFinalWarpPredicateLayouts(mod)))
      return signalPassFailure();

    AMD::TargetInfo targetInfo(this->gfxArch.getValue());
    if (targetInfo.getISAFamily() == triton::amdgpu::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->gfxArch.getValue() << "'";
      return signalPassFailure();
    }

    // Function/CFG conversion can replace block arguments with materialization
    // casts, so classify the original tensor graph before the first conversion.
    AMD::ScheduledMfmaLoweringState scheduledMfmaState;
    mod.walk([&](triton::amdgpu::ScheduledMfmaOp op) {
      if (op.getAccumulatorRole() == "persistent" &&
          op.getResidentOperand() == "none" && isLocalLoadDerived(op.getA()) &&
          isLocalLoadDerived(op.getB()))
        scheduledMfmaState.operandOrderEligibleOps.insert(op.getOperation());
    });

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonAMDGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    // Allocate shared memory and set barrier
    auto allocationFn = [&targetInfo](Operation *op) {
      return AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };
    ModuleAllocation allocation(mod, allocationFn,
                                targetInfo.getSharedMemoryPartitionSize());

    if (targetInfo.requiresAliasInfoForAsyncOps())
      AMD::annotateLocalLoadsSyncedViaAsyncWait(mod);

    ModuleMembarAnalysis membarPass(allocation, AMD::membarFilter,
                                    AMD::getWarpLocalScratchSync());
    membarPass.run();
    materializeDeferredSchedGroupBarriers(mod);

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::AMD::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for AMD specific patterns higher so they apply before common
    // patterns
    int AMDBenefit = commonBenefit + 1;
    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

    AMD::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, AMDBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    AMD::populateDotOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                     AMDBenefit);
    AMD::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                             axisInfoAnalysis, allocation,
                                             targetInfo, AMDBenefit);
    AMD::populateBarrierOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                         axisInfoAnalysis, AMDBenefit);
    AMD::populateFpCastOpToLLVMPatterns(typeConverter, patterns, ftz,
                                        axisInfoAnalysis, allocation,
                                        targetInfo, AMDBenefit);
    AMD::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, AMDBenefit);
    AMD::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    AMD::populateBarrierOpToLLVMPatterns(typeConverter, patterns, AMDBenefit);
    AMD::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                            AMDBenefit);

    mlir::triton::populateReduceOpToLLVMPatternsWithOptions(
        typeConverter, patterns, targetInfo, commonBenefit,
        enableTreeReduction);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    AMD::populateHistogramOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                           AMDBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateGatherOpToLLVMPatterns,
                      commonBenefit);

    auto coordinateGroups = std::make_shared<DistributedCoordinateGroups>();
    AMD::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        AMDBenefit, coordinateGroups,
                                        scheduledMfmaState);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, commonBenefit,
                                                 std::move(coordinateGroups));
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    AMD::populateSPMDOpToLLVMPattern(typeConverter, patterns, AMDBenefit);

    mlir::triton::AMD::populateTritonAMDGPUToLLVMPatterns(
        typeConverter, patterns, targetInfo, AMDBenefit);
    mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, AMDBenefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::AMD::populateWarpIdOpToLLVMPattern(typeConverter, targetInfo,
                                                     patterns, commonBenefit);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->gfxArch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->gfxArch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo);
    mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    AMD::finalizeScheduledMfmaLowering(scheduledMfmaState);
    AMD::adjustModeRegister(mod, targetInfo);
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
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem",
        /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

struct FinalizeScheduledMfmaOperands
    : public triton::impl::FinalizeScheduledMfmaOperandsBase<
          FinalizeScheduledMfmaOperands> {
  void runOnOperation() override {
    SmallVector<LLVM::InlineAsmOp> operandPins;
    WalkResult result = getOperation().walk([&](Operation *op) {
      Attribute marker =
          op->getAttr(triton::AMD::kScheduledMfmaOperandPinAttrName);
      if (!marker)
        return WalkResult::advance();

      auto pin = dyn_cast<LLVM::InlineAsmOp>(op);
      if (!isa<UnitAttr>(marker) || !pin || pin.getHasSideEffects() ||
          !pin.getAsmString().empty() || pin->getNumOperands() != 1 ||
          pin->getNumResults() != 1 ||
          (pin.getConstraints() != "=v,0" && pin.getConstraints() != "=a,0")) {
        op->emitError("expected a pure empty tied register-class pin marked "
                      "with a unit attribute");
        return WalkResult::interrupt();
      }
      auto type = dyn_cast<VectorType>(pin->getResult(0).getType());
      if (!type || type.getRank() != 1 || type.isScalable() ||
          !type.getElementType().isInteger(32) ||
          pin->getOperand(0).getType() != type) {
        op->emitError("expected matching fixed vector<i32> operand and result "
                      "types for a scheduled MFMA operand pin");
        return WalkResult::interrupt();
      }
      operandPins.push_back(pin);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Validate every marker before mutation. Removing it also makes this pass
    // idempotent without treating unrelated side-effecting asm as operand pins.
    for (LLVM::InlineAsmOp pin : operandPins) {
      pin.setHasSideEffects(true);
      pin->removeAttr(triton::AMD::kScheduledMfmaOperandPinAttrName);
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef gfxArch, bool ftz) {
  return std::make_unique<ConvertTritonAMDGPUToLLVM>(gfxArch, ftz);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef gfxArch, bool ftz,
                                    bool enableTreeReduction) {
  return std::make_unique<ConvertTritonAMDGPUToLLVM>(gfxArch, ftz,
                                                     enableTreeReduction);
}

std::unique_ptr<OperationPass<ModuleOp>>
createFinalizeScheduledMfmaOperandsPass() {
  return std::make_unique<FinalizeScheduledMfmaOperands>();
}

} // namespace mlir::triton
