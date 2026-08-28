#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionLoopPeeling.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/Dump.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static void pipelineWgmma(ModuleOp moduleOp, unsigned numStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    if (getNumStagesOrDefault(forOp, numStages) > 1)
      mlir::triton::asyncLaunchDots(forOp);
  }
}

static bool hasMMAv5WaitsInLastStage(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int maxStage = schedule.getNumStages() - 1;
  bool hasMMAv5 = false;
  bool hasWaitInLastStage = false;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<triton::nvidia_gpu::WaitBarrierOp>(op) &&
        schedule[&op].first == maxStage) {
      hasWaitInLastStage = true;
    }
    if (isa<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
      hasMMAv5 = true;
    }
  }
  return hasMMAv5 && hasWaitInLastStage;
}

static std::optional<StringRef>
getWarpSpecializedPartitionType(scf::ForOp forOp) {
  auto wsOp = forOp->getParentOfType<triton::gpu::WarpSpecializeOp>();
  if (!wsOp)
    return std::nullopt;
  auto typesAttr = wsOp->getAttrOfType<ArrayAttr>(kPartitionTypesAttrName);
  if (!typesAttr)
    return std::nullopt;

  SmallVector<StringRef> partitionTypes;
  for (Attribute attr : typesAttr) {
    auto type = dyn_cast<StringAttr>(attr);
    if (!type)
      return std::nullopt;
    partitionTypes.push_back(type.getValue());
  }

  Region *loopRegion = forOp->getParentRegion();
  if (wsOp.getDefaultRegion().isAncestor(loopRegion))
    return partitionTypes.empty() ? std::nullopt
                                  : std::optional(partitionTypes.front());

  auto partitionRegions = wsOp.getPartitionRegions();
  if (partitionTypes.size() < partitionRegions.size())
    return std::nullopt;
  size_t typeOffset = partitionTypes.size() - partitionRegions.size();
  for (auto [idx, partitionRegion] : llvm::enumerate(partitionRegions)) {
    if (partitionRegion->isAncestor(loopRegion))
      return partitionTypes[idx + typeOffset];
  }
  return std::nullopt;
}

static bool containsMMA(scf::ForOp forOp) {
  return forOp
      ->walk([](Operation *op) {
        return isa<triton::nvidia_gpu::MMAv5OpInterface,
                   triton::DotOpInterface>(op)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

static bool containsMMAv5(scf::ForOp forOp) {
  return forOp
      ->walk([](triton::nvidia_gpu::MMAv5OpInterface) {
        return WalkResult::interrupt();
      })
      .wasInterrupted();
}

static bool needsCustomMetaWSEpiloguePeeling(scf::ForOp forOp) {
  Operation *scope = forOp.getOperation();
  if (auto wsOp = forOp->getParentOfType<triton::gpu::WarpSpecializeOp>())
    scope = wsOp.getOperation();

  // The generic pipeline expander predicates peeled epilogue operations.
  // Peer gathers cannot be predicated, and predicating a TMA reduction in one
  // partition independently of its sibling GEMM partitions breaks their
  // iteration lockstep. Keep the legacy PredicateStage-based peeling for the
  // entire warp-specialize operation when either operation is present.
  return scope
      ->walk([&](Operation *op) {
        if (isa<triton::nvidia_gpu::TwoCTAPeerGatherOp,
                triton::nvidia_gpu::AsyncTMAReduceOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static void expandLoops(ModuleOp moduleOp) {
  DenseSet<MaskOp> peeledMaskOps;
  auto processPeeledEpilogueOp = [&](RewriterBase &rewriter, Operation *op,
                                     bool isEpilogue) -> Operation * {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (auto predOp = dyn_cast<triton::gpu::PredicateStageOp>(op)) {
      if (isEpilogue) {
        // Return false for the predicate of the peeled iteration
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 0);
      }
      if (predOp.getStage() == predOp.getMaxStage() - 1) {
        return mlir::arith::ConstantIntOp::create(
            rewriter, predOp.getLoc(), predOp.getResult().getType(), 1);
      }
      return triton::emitPredicateForStage(
                 rewriter, predOp.getIv(), predOp.getUb(), predOp.getStep(),
                 predOp.getMaxStage(), predOp.getStage())
          .getDefiningOp();
    }
    if (auto maskOp = dyn_cast<triton::gpu::MaskOp>(op)) {
      if (isEpilogue) {
        peeledMaskOps.insert(maskOp);
      }
    }
    return op;
  };

  SmallVector<scf::ForOp> loops;
  bool hasWarpSpec = false;
  moduleOp->walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
      hasWarpSpec = true;
    loops.push_back(forOp);
  });
  auto metaWS = triton::tools::getBoolEnv("TRITON_USE_META_WS");
  // The partition-type filter below exists to prune the loops that the extra
  // ScheduleLoops re-run re-staged. That re-run only happens on the 2-CTA path
  // (see CUDABackend.make_ttgir). On the 1-CTA path every partition loop still
  // carries its own valid post-WS schedule, so filtering there drops the
  // epilogue / epilogue_store loops and miscompiles the kernel.
  for (scf::ForOp forOp : loops) {
    if (metaWS && triton::gpu::isPhysicalCluster(forOp) &&
        forOp->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
      std::optional<StringRef> partitionType =
          getWarpSpecializedPartitionType(forOp);
      if (!partitionType)
        continue;
      if (*partitionType == "gemm") {
        if (!containsMMA(forOp))
          continue;
      } else if (*partitionType != "load") {
        // Load-worker loops carry their own software-pipeline schedule and
        // must be expanded to materialize the TMA prologue.
        continue;
      }
    }
    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      continue;
    }
    // Skip pipelining when we have a single stage.
    if (metaWS && schedule.getNumStages() == 1) {
      continue;
    }

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    if (metaWS && containsMMAv5(forOp)) {
      unsigned maxMMAStage = 0;
      for (auto &[op, stage] : finalSchedule)
        if (isa<triton::nvidia_gpu::MMAv5OpInterface>(op))
          maxMMAStage = std::max(maxMMAStage, stage);
      for (auto &[op, stage] : finalSchedule) {
        auto wait = dyn_cast<triton::nvidia_gpu::WaitBarrierOp>(op);
        if (wait && !wait.getDeps().empty() && stage > maxMMAStage)
          stage = maxMMAStage;
      }
    }
    triton::PipeliningOption options;
    bool useCustomMetaWSEpilogue =
        metaWS && needsCustomMetaWSEpiloguePeeling(forOp);
    options.supportDynamicLoops = true;
    options.peelEpilogue = metaWS && !useCustomMetaWSEpilogue;
    options.predicateFn = wrapInMaskOp;
    options.getScheduleFn =
        [&](scf::ForOp forOp,
            std::vector<std::pair<Operation *, unsigned>> &schedule) {
          schedule = finalSchedule;
        };

    // Testing feature: allow for unresolved predicate stage ops
    // in the loop body.
    bool keepPredicateStage = forOp->hasAttr("__test_keep_predicate_stage");

    // FB Change: Enable epilogue peeling for warp specialized loops
    // This may not be fully working but seems to work based on FA testing.
    bool customEpiloguePeeling =
        hasMMAv5WaitsInLastStage(forOp, schedule) &&
        !forOp->getParentOfType<triton::gpu::WarpSpecializeOp>() &&
        !keepPredicateStage; // do not peel if we are testing the stage
                             // predication
    if (metaWS)
      customEpiloguePeeling = useCustomMetaWSEpilogue;

    if (keepPredicateStage || customEpiloguePeeling) {
      options.emitPredicateStageFn =
          [](RewriterBase &rewriter, Value inductionVar, Value upperBound,
             Value step, uint64_t maxStage, uint64_t stage) {
            return triton::gpu::PredicateStageOp::create(
                rewriter, inductionVar.getLoc(), inductionVar, upperBound, step,
                maxStage, stage);
          };
    }
    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        triton::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp)) {
      continue;
    }
    forOp = *newForOp;
    if (customEpiloguePeeling) {
      mlir::triton::peelLoopEpilogue(forOp, processPeeledEpilogueOp);
    }

    // Prune all the statically dead mask ops in the epilogue. This is a
    // hack, ideally we should do it for all the mask ops, but it is incorrect
    // if we have speculatively executed async cp operations that will store to
    // shmem even if the mask is false.
    for (auto maskOp : peeledMaskOps) {
      rewriter.setInsertionPoint(maskOp);
      if (isConstantIntValue(maskOp.getPred(), 0)) {
        SmallVector<Value> results;
        for (auto result : maskOp->getResults()) {
          auto poisonOp = mlir::ub::PoisonOp::create(rewriter, maskOp->getLoc(),
                                                     result.getType());
          results.push_back(poisonOp);
        }
        maskOp->replaceAllUsesWith(results);
        maskOp->erase();
      }
    }
    peeledMaskOps.clear();
  }
  assert(moduleOp.getOps<triton::gpu::PredicateStageOp>().empty() &&
         "PredicateStageOp should be resolved after the pipeline expansion");
  assert(verify(moduleOp).succeeded());
  resolveMaskOp(moduleOp);
}

struct PipelinePass : public impl::TritonGPUPipelineBase<PipelinePass> {

  using impl::TritonGPUPipelineBase<PipelinePass>::TritonGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (moduleOp->hasAttr(kSkipGenericPipelineAttrName))
      return;

    // Transform the loop by introducing async operations to prepare it for
    // pipeline expansion.
    lowerLoops(moduleOp);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: LowerLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Code partitioning has already produced physical warp-specialize regions.
    // Peel only after lowerLoops has consumed the original schedule: moving a
    // scheduled use into a prologue earlier would invalidate its def-use stage
    // analysis.
    peelPartitionLoops(moduleOp);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: "
             "PartitionLoopPeeling\n"
          << moduleOp << "\n\n\n";
    }

    // Apply the pipeline expansion.
    expandLoops(moduleOp);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// SoftwarePipeliner internal IR Dump After: "
             "ExpandLoops\n"
          << moduleOp << "\n\n\n";
    }

    // Cleanup the IR from the pipeline attributes.
    removePipeliningAttributes(moduleOp);

    pipelineWgmma(moduleOp, numStages);

    // schedule the waits
    mlir::triton::updateWaits(getOperation());

    // Clean up arithmetic before applying the next level of pipelining to
    // simplify the IR.
    auto arithDialect =
        getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();

    {
      auto metaWS = triton::tools::getBoolEnv("TRITON_USE_META_WS");
      SmallVector<scf::ForOp> loops;
      bool hasWarpSpec = false;
      getOperation()->walk([&](scf::ForOp forOp) {
        // Bail out for loops with num_stage <= 1.
        if (getNumStagesOrDefault(forOp, numStages) > 1)
          loops.push_back(forOp);
        if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
          hasWarpSpec = true;
      });

      // With Meta's warpspec, we are handling this in AutoWS.
      if (!metaWS || !hasWarpSpec)
        for (scf::ForOp forOp : loops) {
          mlir::triton::pipelineTMAStores(forOp);
        }
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
