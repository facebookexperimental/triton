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
    // Emit raw cycle for downstream buffer depth computation (Step 3).
    node.op->setAttr("tt.modulo_cycle",
                     IntegerAttr::get(IntegerType::get(ctx, 32), cycle));
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
// Step 3: Derive per-resource buffer depths from modulo schedule
// ============================================================================

// Blackwell sm_100 SMEM budget (reserve some for barriers/scratch).
constexpr int kSmemBudgetBytes = 228 * 1024;

static int getMemDescSizeBytes(ttg::MemDescType memDescType) {
  int numElements = 1;
  for (auto dim : memDescType.getShape())
    numElements *= dim;
  return numElements * memDescType.getElementType().getIntOrFloatBitWidth() / 8;
}

/// Compute per-resource buffer depths from the modulo schedule.
///
/// For each local_alloc (SMEM buffer) in the loop body:
///   1. Find its producer's cycle (tt.modulo_cycle on the load)
///   2. Find the last consumer's end cycle (cycle + latency)
///   3. lifetime = last_consumer_end - producer_cycle
///   4. num_buffers = floor(lifetime / II) + 1
///
/// Then check SMEM budget: if total exceeds limit, reduce depths.
/// Returns the max buffer depth, or 0 if no buffers found.
static int computeBufferDepths(scf::ForOp loop,
                               const ttg::LatencyModel &model) {
  auto ctx = loop.getContext();
  auto iiAttr = loop->getAttrOfType<IntegerAttr>("tt.modulo_ii");
  if (!iiAttr)
    return 0;
  int II = iiAttr.getInt();
  if (II <= 0)
    return 0;

  struct BufferInfo {
    Operation *allocOp;
    int sizeBytes;
    int numBuffers;
  };
  SmallVector<BufferInfo> buffers;

  for (auto &op : loop.getBody()->without_terminator()) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !alloc.getSrc())
      continue;

    auto memDescType = dyn_cast<ttg::MemDescType>(alloc.getType());
    if (!memDescType)
      continue;

    // Find producer cycle from the source op.
    auto *producer = alloc.getSrc().getDefiningOp();
    if (!producer)
      continue;
    auto prodCycleAttr =
        producer->getAttrOfType<IntegerAttr>("tt.modulo_cycle");
    if (!prodCycleAttr)
      continue;
    int prodCycle = prodCycleAttr.getInt();

    // Find last consumer end cycle.
    int lastConsumerEnd = prodCycle;
    for (auto *user : alloc->getUsers()) {
      auto userCycleAttr =
          user->getAttrOfType<IntegerAttr>("tt.modulo_cycle");
      if (!userCycleAttr)
        continue;
      int userCycle = userCycleAttr.getInt();
      auto info = model.getLatency(user);
      lastConsumerEnd = std::max(lastConsumerEnd, userCycle + info.latency);
    }

    int lifetime = lastConsumerEnd - prodCycle;
    int numBuffers = std::max(lifetime / II + 1, 1);
    int sizeBytes = getMemDescSizeBytes(memDescType);
    buffers.push_back({alloc, sizeBytes, numBuffers});

    LDBG("Buffer: producer_cycle=" << prodCycle
                                   << " last_consumer_end=" << lastConsumerEnd
                                   << " lifetime=" << lifetime << " II=" << II
                                   << " -> num_buffers=" << numBuffers
                                   << " (" << sizeBytes << " bytes)");
  }

  if (buffers.empty())
    return 0;

  // SMEM budget check: reduce depths if total exceeds limit.
  auto computeTotalSmem = [&]() {
    int total = 0;
    for (auto &b : buffers)
      total += b.sizeBytes * b.numBuffers;
    return total;
  };

  while (computeTotalSmem() > kSmemBudgetBytes) {
    int worstIdx = 0, worstCost = 0;
    for (int i = 0; i < (int)buffers.size(); ++i) {
      int cost = buffers[i].sizeBytes * buffers[i].numBuffers;
      if (cost > worstCost) {
        worstCost = cost;
        worstIdx = i;
      }
    }
    if (buffers[worstIdx].numBuffers <= 1)
      break;
    buffers[worstIdx].numBuffers--;
    LDBG("Reduced buffer depth for SMEM budget");
  }

  // Emit tt.num_buffers on each local_alloc.
  int maxNumBuffers = 1;
  for (auto &b : buffers) {
    b.allocOp->setAttr("tt.num_buffers",
                       IntegerAttr::get(IntegerType::get(ctx, 32), b.numBuffers));
    maxNumBuffers = std::max(maxNumBuffers, b.numBuffers);
  }

  LDBG("Buffer depths: max=" << maxNumBuffers
                              << " totalSmem=" << computeTotalSmem() << "B");
  return maxNumBuffers;
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

      // Step 3: Derive SMEM buffer depths from cycle-level lifetimes.
      // Only set tt.num_stages if the user didn't specify one (no
      // tt.num_stages attr on the loop already).
      if (!innerLoop->hasAttr(tt::kNumStagesAttrName)) {
        int numStages = computeBufferDepths(innerLoop, model);
        if (numStages > 0) {
          auto ctx = innerLoop.getContext();
          innerLoop->setAttr(
              tt::kNumStagesAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), numStages));
          LDBG("Set tt.num_stages=" << numStages << " from modulo schedule");
        }
      }

      // Clean up tt.modulo_cycle — internal attr used only to pass cycle
      // info from emitScheduleAttributes to computeBufferDepths.
      for (auto &op : innerLoop.getBody()->without_terminator())
        op.removeAttr("tt.modulo_cycle");
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
