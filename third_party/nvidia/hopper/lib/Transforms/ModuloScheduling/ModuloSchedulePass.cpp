// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass A: Modulo Schedule Pass
//
// Builds a DDG from scf.for loop bodies, computes MinII, runs Rau's iterative
// modulo scheduling, and annotates ops with loop.stage and loop.cluster
// attributes for downstream pipelining passes.

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "nvgpu-modulo-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

// ============================================================================
// Emit loop.stage / loop.cluster attributes from modulo schedule
// ============================================================================

static void emitScheduleAttributes(scf::ForOp loop,
                                   const ttg::DataDependenceGraph &ddg,
                                   const ttg::ModuloScheduleResult &schedule) {
  const int II = schedule.II;
  const int maxStage = schedule.getMaxStage();
  auto ctx = loop.getContext();

  // Step 2.5: Compute per-stage cluster IDs from modulo cycles.
  // Ops in the same stage are ordered by cycle: lower cycle → lower cluster ID.
  // This preserves the modulo schedule's within-stage ordering for downstream
  // pipelining, instead of relying on IR program order.
  llvm::DenseMap<int, SmallVector<int>> stageToCycles;
  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    int stage = it->second / II;
    stageToCycles[stage].push_back(it->second);
  }
  // Deduplicate and sort cycles per stage to assign dense cluster IDs.
  llvm::DenseMap<int, llvm::DenseMap<int, int>> stageAndCycleToCluster;
  for (auto &[stage, cycles] : stageToCycles) {
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());
    for (int i = 0, e = cycles.size(); i < e; ++i)
      stageAndCycleToCluster[stage][cycles[i]] = i;
  }

  for (const auto &node : ddg.getNodes()) {
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    // For multi-stage super-nodes (prologue/kloop/epilogue sharing the same
    // Operation*), only write attrs from the node registered in opToIdx
    // (the epilogue) to avoid overwrites.
    auto opIt = ddg.getOpToIdx().find(node.op);
    if (opIt != ddg.getOpToIdx().end() && opIt->second != node.idx)
      continue;
    int stage = it->second / II;
    int cycle = it->second;
    int clusterId = stageAndCycleToCluster[stage][cycle];
    node.op->setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), stage));
    node.op->setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), clusterId));
  }

  // Ensure ALL ops in the loop body have loop.stage/loop.cluster attrs.
  // Downstream passes assert every op is in the schedule.
  for (auto &op : loop.getBody()->without_terminator()) {
    if (!op.hasAttr(tt::kLoopStageAttrName))
      op.setAttr(tt::kLoopStageAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
    if (!op.hasAttr(tt::kLoopClusterAttrName))
      op.setAttr(tt::kLoopClusterAttrName,
                 IntegerAttr::get(IntegerType::get(ctx, 32), 0));
  }

  LDBG("Emitted schedule: II=" << II << " maxStage=" << maxStage);

  loop->setAttr("tt.modulo_ii",
                IntegerAttr::get(IntegerType::get(ctx, 32), II));
  loop->setAttr(tt::kScheduledMaxStageAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));
}

// ============================================================================
// Pass A: Modulo Scheduling
// ============================================================================

/// The main pass.
struct ModuloSchedulePass
    : public PassWrapper<ModuloSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloSchedulePass)

  StringRef getArgument() const override { return "nvgpu-modulo-schedule"; }

  StringRef getDescription() const override {
    return "Modulo scheduling for warp specialization (Pass A)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    // Find innermost loops with TMA loads or MMA ops.
    SmallVector<scf::ForOp> innerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (hasInnerLoop)
        return;
      bool hasTMALoad = false;
      bool hasMMAv5 = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
          hasTMALoad = true;
        if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
          hasTMALoad = true;
        if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
          hasMMAv5 = true;
      });
      if (!hasTMALoad && !hasMMAv5)
        return;
      innerLoops.push_back(loop);
    });

    LDBG("Found " << innerLoops.size() << " innermost loop(s)");

    for (auto innerLoop : innerLoops) {
      // Build DDG for this inner loop.
      auto ddg = ttg::DataDependenceGraph::build(innerLoop, model);
      if (ddg.getNumNodes() == 0)
        continue;

      LDBG("DDG: " << ddg.getNumNodes() << " nodes, "
                    << ddg.getEdges().size() << " edges");

      // Run Rau's modulo scheduling.
      auto schedResult = ttg::runModuloScheduling(ddg);
      if (failed(schedResult)) {
        LDBG("Scheduling FAILED");
        continue;
      }

      LDBG("Schedule: II=" << schedResult->II
                           << " ResMII=" << ddg.computeResMII()
                           << " RecMII=" << ddg.computeRecMII()
                           << " maxStage=" << schedResult->getMaxStage());

      // Emit loop.stage / loop.cluster on all ops.
      emitScheduleAttributes(innerLoop, ddg, *schedResult);
    }

    // Step 2: Schedule outer loops (persistent kernels).
    SmallVector<scf::ForOp> outerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](scf::ForOp) { hasInnerLoop = true; });
      if (!hasInnerLoop)
        return;
      if (loop->getParentOfType<scf::ForOp>())
        return;
      outerLoops.push_back(loop);
    });

    LDBG("Found " << outerLoops.size() << " outer loop(s)");

    for (auto outerLoop : outerLoops) {
      auto outerDDG = ttg::DataDependenceGraph::build(outerLoop, model);
      if (outerDDG.getNumNodes() == 0)
        continue;

      LDBG("Outer DDG: " << outerDDG.getNumNodes() << " nodes, "
                          << outerDDG.getEdges().size() << " edges");

      auto outerSched = ttg::runModuloScheduling(outerDDG);
      if (failed(outerSched)) {
        LDBG("Outer scheduling FAILED");
        continue;
      }

      LDBG("Outer schedule: II=" << outerSched->II
                                 << " ResMII=" << outerDDG.computeResMII()
                                 << " RecMII=" << outerDDG.computeRecMII()
                                 << " maxStage=" << outerSched->getMaxStage());

      emitScheduleAttributes(outerLoop, outerDDG, *outerSched);
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}

void registerNVGPUModuloSchedule() {
  PassRegistration<ModuloSchedulePass>();
}
} // namespace mlir
