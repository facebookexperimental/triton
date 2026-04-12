// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Pass B: Schedule Integration + Modulo Partition Scheduling
//
// Two responsibilities:
// 1. Configure IR attributes so downstream passes use the modulo schedule.
// 2. Assign WS partitions (ttg.partition) using DDG pipe classification
//    and utilization analysis. Supports nested loops via bottom-up traversal.
//    Replaces PartitionScheduling for modulo-scheduled kernels.

#include "DataDependenceGraph.h"
#include "LatencyModel.h"
#include "ModuloReservationTable.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvgpu-modulo-ws-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {

// ============================================================================
// Modulo Partition Scheduling — utilization-driven warp group assignment
// ============================================================================

// Pipelines with utilization > this threshold get dedicated warp groups.
// 30% is chosen empirically: below this, the pipeline is idle most of the
// time and doesn't benefit from a dedicated warp group.
constexpr double kUtilizationThreshold = 0.3;

/// Partition a loop's ops into warp groups based on DDG pipe classification.
/// Returns number of partitions created, or 0 if not applicable.
static int partitionLoopByUtilization(scf::ForOp loop,
                                      const ttg::LatencyModel &model,
                                      bool isWSLoop = false) {
  // Read II from tt.modulo_ii if already set by Pass A, otherwise
  // build DDG and schedule to compute it.
  int II = 0;
  if (auto iiAttr = loop->getAttrOfType<IntegerAttr>("tt.modulo_ii"))
    II = iiAttr.getInt();

  auto ddg = ttg::DataDependenceGraph::build(loop, model);
  if (II <= 0) {
    auto schedResult = ttg::runModuloScheduling(ddg);
    if (failed(schedResult))
      return 0;
    II = schedResult->II;
  }
  if (II <= 0)
    return 0;

  LDBG("Building partition for loop with "
       << std::distance(loop.getBody()->begin(), loop.getBody()->end())
       << " ops, II=" << II);

  // Compute per-pipeline utilization.
  llvm::DenseMap<ttg::HWPipeline, int> pipeLoad;
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;
    pipeLoad[node.pipeline] += node.selfLatency;
  }

  // Determine which pipelines get their own warp group.
  SmallVector<ttg::HWPipeline> ownGroup;
  SmallVector<ttg::HWPipeline> mergeGroup;
  for (auto pipe : {ttg::HWPipeline::MEM, ttg::HWPipeline::TC,
                    ttg::HWPipeline::CUDA, ttg::HWPipeline::SFU}) {
    int load = pipeLoad.lookup(pipe);
    if (load == 0)
      continue;
    double util = static_cast<double>(load) / II;
    if (util > kUtilizationThreshold)
      ownGroup.push_back(pipe);
    else
      mergeGroup.push_back(pipe);
  }

  // MEM always gets its own group (TMA producer needs dedicated warp).
  // Remove from mergeGroup if it was placed there by the threshold check.
  if (!llvm::is_contained(ownGroup, ttg::HWPipeline::MEM) &&
      pipeLoad.lookup(ttg::HWPipeline::MEM) > 0) {
    ownGroup.insert(ownGroup.begin(), ttg::HWPipeline::MEM);
    llvm::erase(mergeGroup, ttg::HWPipeline::MEM);
  }

  if (ownGroup.size() < 2)
    return 0; // Need at least 2 groups for WS.

  // Build pipe → partition ID mapping.
  llvm::DenseMap<ttg::HWPipeline, int> pipeToPartition;
  int nextId = 0;
  for (auto pipe : ownGroup)
    pipeToPartition[pipe] = nextId++;
  int defaultPartId = -1;
  if (!mergeGroup.empty()) {
    defaultPartId = nextId++;
    for (auto pipe : mergeGroup)
      pipeToPartition[pipe] = defaultPartId;
  }
  int numPartitions = nextId;

  // All-partitions list for shared/scalar ops.
  SmallVector<int> allParts;
  for (int i = 0; i < numPartitions; i++)
    allParts.push_back(i);

  LLVM_DEBUG({
    DBGS() << numPartitions << " groups (II=" << II << "): ";
    for (auto pipe : ownGroup)
      llvm::dbgs() << ttg::getPipelineName(pipe) << "=" << pipeToPartition[pipe]
                   << " ";
    if (!mergeGroup.empty())
      llvm::dbgs() << "default=" << defaultPartId;
    llvm::dbgs() << "\n";
  });

  // Step 1: Seed assignment — DDG-classified ops get their specific partition.
  // Skip ops with regions (scf.for, scf.if) — their child ops may get different
  // partitions, and the verifier requires parent partitions to be a superset of
  // all children. These ops get allParts in Step 3 instead.
  llvm::DenseMap<Operation *, int> opPartitionMap;
  for (const auto &node : ddg.getNodes()) {
    if (node.op->getNumRegions() > 0)
      continue; // Skip ForOps, IfOps — handled later.
    auto it = pipeToPartition.find(node.pipeline);
    if (it != pipeToPartition.end()) {
      opPartitionMap[node.op] = it->second;
      ttg::setPartition(node.op, ArrayRef<int>{it->second});
    }
  }

  // Step 2: Propagate partitions through use-def chains.
  // For unassigned ops, inherit partition from users (demand-driven).
  // Iterate until convergence.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &op : loop.getBody()->without_terminator()) {
      if (isa<scf::ForOp>(op) || opPartitionMap.count(&op))
        continue;
      // Collect partitions from all users within this loop body.
      SetVector<int> userParts;
      for (auto *user : op.getUsers()) {
        // Find the ancestor op in the loop body block.
        Operation *ancestor = loop.getBody()->findAncestorOpInBlock(*user);
        if (!ancestor)
          continue;
        auto uit = opPartitionMap.find(ancestor);
        if (uit != opPartitionMap.end())
          userParts.insert(uit->second);
      }
      if (userParts.size() == 1) {
        int part = *userParts.begin();
        opPartitionMap[&op] = part;
        ttg::setPartition(&op, ArrayRef<int>{part});
        changed = true;
      }
    }
  }

  // Step 2.5: TMEM consistency — TMEMStoreOp and TMEMLoadOp sharing a
  // TMEMAllocOp must be in the same partition. PartitionScheduling asserts
  // this.
  loop.walk([&](ttng::TMEMAllocOp allocOp) {
    std::optional<int> simtPartition;
    for (auto *user : allocOp->getUsers()) {
      if (isa<ttng::TMEMLoadOp>(user)) {
        auto uit = opPartitionMap.find(user);
        if (uit != opPartitionMap.end())
          simtPartition = uit->second;
      }
    }
    if (!simtPartition) {
      for (auto *user : allocOp->getUsers()) {
        if (isa<ttng::TMEMStoreOp>(user)) {
          auto uit = opPartitionMap.find(user);
          if (uit != opPartitionMap.end())
            simtPartition = uit->second;
        }
      }
    }
    if (!simtPartition)
      return WalkResult::advance();
    for (auto *user : allocOp->getUsers()) {
      if (isa<ttng::TMEMStoreOp, ttng::TMEMLoadOp>(user)) {
        opPartitionMap[user] = *simtPartition;
        ttg::setPartition(user, ArrayRef<int>{*simtPartition});
      }
    }
    return WalkResult::advance();
  });

  // Step 3: Remaining unassigned ops → allParts. Walk recursively to cover
  // ops inside scf.if regions (flattened persistent kernels have tile-boundary
  // conditionals). Skip inner ForOps (handled by inner loop processing).
  loop.walk([&](Operation *op) {
    if (isa<scf::ForOp>(op) && op != loop.getOperation())
      return WalkResult::skip(); // Don't recurse into inner ForOps.
    if (!ttg::hasPartition(op))
      ttg::setPartition(op, allParts);
    return WalkResult::advance();
  });

  // Inner ForOps: set partition on the ForOp itself via raw setAttr (don't
  // propagate to region terminators — body ops are handled by inner loop
  // processing). The ForOp gets allParts since both MEM and TC run inside it.
  Builder b(loop.getContext());
  {
    auto sorted = llvm::to_vector(allParts);
    llvm::sort(sorted);
    for (auto &op : loop.getBody()->without_terminator()) {
      if (isa<scf::ForOp>(op))
        op.setAttr(ttg::kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
    }
  }

  // Set ttg.partition on the WS loop itself (required by verifier if
  // ttg.partition.outputs is set). Use raw setAttr to avoid propagating.
  if (isWSLoop) {
    auto sorted = llvm::to_vector(allParts);
    llvm::sort(sorted);
    loop->setAttr(ttg::kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
  }

  // Yield → all partitions.
  ttg::setPartition(cast<scf::YieldOp>(loop.getBody()->getTerminator()),
                    allParts);

  // Only serialize WS metadata on the actual WS loop (not inner K-loops).
  // PartitionSet::fromLoop reads these attrs and will get confused if inner
  // loops have them too.
  if (isWSLoop) {
    Builder b(loop.getContext());
    SmallVector<Attribute> stages;
    for (int i = 0; i < numPartitions; i++) {
      int stage = 0;
      // TC partition gets stage 1 (consumer, pipelined after MEM producer).
      for (auto pipe : ownGroup)
        if (pipeToPartition[pipe] == i && pipe == ttg::HWPipeline::TC)
          stage = 1;
      stages.push_back(b.getI32IntegerAttr(stage));
    }
    loop->setAttr(ttg::kPartitionStagesAttrName, b.getArrayAttr(stages));
    ttg::setWarpSpecializeTag(loop, 0);

    // Set partition outputs — for now all results go to all partitions.
    SmallVector<SetVector<int>> outputParts;
    for (unsigned i = 0; i < loop.getNumResults(); i++) {
      SetVector<int> ids;
      for (int p : allParts)
        ids.insert(p);
      outputParts.push_back(ids);
    }
    ttg::setPartitionOutputs(loop, outputParts);
  }

  return numPartitions;
}

/// Bottom-up partition scheduling for nested WS loops.
/// Inner loops are partitioned first with specific per-op partitions,
/// then the outer WS loop. For flattened loops (no inner loops), skip
/// partition assignment and let PartitionScheduling handle it.
static void moduloPartitionScheduling(scf::ForOp wsLoop,
                                      const ttg::LatencyModel &model) {
  // Collect inner loops (deepest first).
  SmallVector<scf::ForOp> innerLoops;
  wsLoop.getBody()->walk([&](scf::ForOp inner) {
    if (inner != wsLoop)
      innerLoops.push_back(inner);
  });

  // Flattened case: no inner loops. The WS loop IS the only loop.
  // Skip our partition assignment — PartitionScheduling's getInitialPartitions
  // already handles flattened loops with DescriptorLoadOp/MMA pattern matching.
  // Our contribution is the modulo schedule (loop.stage/loop.cluster).
  if (innerLoops.empty()) {
    LDBG("Flattened WS loop — skipping partition, using PartitionScheduling "
         "default");
    return;
  }

  // Partition inner loops bottom-up.
  for (auto inner : llvm::reverse(innerLoops)) {
    int n = partitionLoopByUtilization(inner, model, /*isWSLoop=*/false);
    if (n > 0)
      LDBG("Inner loop: " << n << " groups");
  }

  // Partition the outer WS loop itself.
  int n = partitionLoopByUtilization(wsLoop, model, /*isWSLoop=*/true);
  if (n > 0)
    LDBG("Outer WS loop: " << n << " groups");
}

// ============================================================================
// processScheduledLoop — existing Pass B logic (schedule integration)
// ============================================================================

static void processScheduledLoop(scf::ForOp loop) {
  auto ctx = loop.getContext();
  bool isWS = loop->hasAttr(tt::kWarpSpecializeAttrName);

  // Read num_stages if already set by Pass A Step 3 (computeBufferDepths).
  int numStages = 0;
  if (auto ns = loop->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName))
    numStages = ns.getInt();

  if (isWS || loop->hasAttr("tt.modulo_ii")) {
    // WS loops or modulo-scheduled loops: keep loop.stage/loop.cluster attrs.
    // For modulo-scheduled non-WS loops, the schedule must survive to
    // downstream ScheduleLoops (which skips them via tt.modulo_ii check).
    int maxStage = 0;
    for (auto &op : loop.getBody()->without_terminator()) {
      if (auto stageAttr =
              op.getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName))
        maxStage = std::max(maxStage, (int)stageAttr.getInt());
    }

    // Derive num_stages from the schedule when Pass A Step 3 found no
    // LocalAllocOp (e.g. outer tile loops of persistent kernels where
    // SMEM buffers are allocated outside the loop).
    if (numStages == 0 && maxStage > 0) {
      numStages = maxStage + 1;
      LDBG("Derived num_stages=" << numStages << " from maxStage=" << maxStage);
    }

    if (numStages > 0) {
      loop->setAttr(tt::kNumStagesAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), numStages));
      // scheduled_max_stage reflects the actual schedule, not buffer depth.
      loop->setAttr(tt::kScheduledMaxStageAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), maxStage));
      LDBG("Set num_stages=" << numStages
                             << " scheduled_max_stage=" << maxStage);
    }
    LDBG("Modulo/WS loop: kept loop.stage/loop.cluster");
  } else {
    // Strip schedule attrs from direct children only — don't recurse
    // into nested scf::ForOp regions (they have their own schedules).
    for (auto &op : loop.getBody()->without_terminator()) {
      if (isa<scf::ForOp>(op))
        continue;
      op.removeAttr(tt::kLoopStageAttrName);
      op.removeAttr(tt::kLoopClusterAttrName);
    }
    LDBG("Non-WS loop: stripped loop.stage/loop.cluster");
  }
  // Keep tt.modulo_ii on the loop so downstream ScheduleLoops (inside AutoWS)
  // knows to skip re-scheduling this loop and its partition clones.
}

struct ModuloWSPartitionPass
    : public PassWrapper<ModuloWSPartitionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloWSPartitionPass)

  StringRef getArgument() const override { return "nvgpu-modulo-ws-partition"; }

  StringRef getDescription() const override {
    return "Schedule integration for warp specialization (Pass B)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    // Step 1: Modulo partition scheduling for WS loops (bottom-up).
    moduleOp.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(tt::kWarpSpecializeAttrName))
        moduloPartitionScheduling(loop, model);
    });

    // Step 2: Schedule integration (existing Pass B logic).
    moduleOp.walk([&](scf::ForOp loop) {
      bool hasMMAv5 = false;
      bool hasTMALoad = false;
      bool hasSchedule = false;
      // Only check direct children of the loop body — don't recurse into
      // nested scf::ForOp regions. Otherwise a non-scheduled outer loop
      // containing a scheduled inner loop would match, and processScheduledLoop
      // would strip the inner loop's schedule attrs in pre-order traversal.
      for (auto &op : loop.getBody()->without_terminator()) {
        if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
          hasMMAv5 = true;
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp,
                ttng::AsyncTMACopyGlobalToLocalOp>(op))
          hasTMALoad = true;
        if (op.hasAttr(tt::kLoopStageAttrName))
          hasSchedule = true;
        if (hasSchedule && (hasMMAv5 || hasTMALoad))
          break;
      }
      if (!hasSchedule || (!hasMMAv5 && !hasTMALoad))
        return;

      processScheduledLoop(loop);
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloWSPartition() {
  return std::make_unique<ModuloWSPartitionPass>();
}

void registerNVGPUModuloWSPartition() {
  PassRegistration<ModuloWSPartitionPass>();
}
} // namespace mlir
