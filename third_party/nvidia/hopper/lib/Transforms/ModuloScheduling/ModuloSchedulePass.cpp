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
#include "ModuloScheduleGraph.h"

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

/// Emit tt.autows annotations on MMA ops from the modulo schedule.
/// These survive through the WS pass (which preserves discardable attrs on
/// MMA ops) and are read by scheduleKeyOpsAnnotation() inside the WS pass's
/// internal scheduleLoops call.
///
/// Format: {"stage": "N", "order": "M"} as a JSON string attribute.
/// "stage" = which SWP pipeline stage the MMA should be in.
/// "order" = relative ordering within the stage (cluster ID).
static void emitMMAAnnotations(scf::ForOp loop,
                               const ttg::DataDependenceGraph &ddg,
                               const ttg::ModuloScheduleResult &schedule) {
  const int II = schedule.II;
  auto ctx = loop.getContext();

  // Compute MMA stages from transitive MMA dependency count.
  //
  // For each MMA, walk backward through distance-0 DDG edges and count
  // how many other MMA nodes are transitively reachable. This captures
  // the data flow structure:
  //   - MMAs depending on 0-1 other MMAs → stage 0 (can be prefetched)
  //   - MMAs depending on 2+ other MMAs → stage 1 (gated on multiple
  //     prior results, natural pipeline boundary)
  //
  // Example: FA backward has 5 MMAs:
  //   qkT (0 MMA deps) → stage 0
  //   dpT (0 MMA deps) → stage 0
  //   dv  (1 MMA dep: qkT) → stage 0
  //   dq  (2 MMA deps: qkT, dpT via dsT) → stage 1
  //   dk  (2 MMA deps: qkT, dpT via dsT) → stage 1
  llvm::DenseSet<unsigned> mmaNodes;
  for (const auto &node : ddg.getNodes()) {
    if (isa<ttng::MMAv5OpInterface>(node.op) || isa<tt::DotOp>(node.op))
      mmaNodes.insert(node.idx);
  }

  // For each MMA, compute transitive MMA predecessors via backward BFS
  // through distance-0 edges only.
  llvm::DenseMap<unsigned, int> mmaStage;
  for (unsigned mmaIdx : mmaNodes) {
    llvm::DenseSet<unsigned> visited;
    llvm::SmallVector<unsigned> worklist;
    worklist.push_back(mmaIdx);
    visited.insert(mmaIdx);

    int mmaPredCount = 0;
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      for (const auto *edge : ddg.getInEdges(cur)) {
        if (edge->distance > 0)
          continue; // skip loop-carried edges
        if (!visited.insert(edge->srcIdx).second)
          continue;
        if (mmaNodes.count(edge->srcIdx))
          mmaPredCount++;
        worklist.push_back(edge->srcIdx);
      }
    }

    // 0-1 MMA predecessors → stage 0 (prefetchable)
    // 2+  MMA predecessors → stage 1 (pipeline boundary)
    mmaStage[mmaIdx] = (mmaPredCount >= 2) ? 1 : 0;
    LDBG("MMA node " << mmaIdx << ": " << mmaPredCount
                     << " transitive MMA predecessors → stage "
                     << mmaStage[mmaIdx]);
  }

  // Collect MMA ops with their stage and cycle, then assign dense cluster IDs.
  struct MMAInfo {
    unsigned nodeIdx;
    Operation *op;
    int stage;
    int cycle;
  };
  llvm::SmallVector<MMAInfo> mmas;

  for (const auto &node : ddg.getNodes()) {
    if (!isa<ttng::MMAv5OpInterface>(node.op) && !isa<tt::DotOp>(node.op))
      continue;
    auto it = schedule.nodeToCycle.find(node.idx);
    if (it == schedule.nodeToCycle.end())
      continue;
    auto stageIt = mmaStage.find(node.idx);
    int stage = stageIt != mmaStage.end() ? stageIt->second : 0;
    mmas.push_back({node.idx, node.op, stage, it->second});
  }

  // Skip annotation if all MMAs are in the same stage — the dependency
  // analysis found no multi-MMA fan-in, so annotations won't help and
  // may break the downstream pipeliner (e.g., GEMM with 1 dot tiled
  // into 4 MMAs, or FA FWD with 2 dots tiled into 4+ MMAs).
  {
    llvm::DenseSet<int> stages;
    for (auto &mma : mmas)
      stages.insert(mma.stage);
    if (stages.size() <= 1) {
      LDBG("Skipping MMA annotations: all " << mmas.size()
                                            << " MMAs in same stage");
      return;
    }
  }

  // Assign order (cluster) within each stage based on MMA dependency depth.
  // MMAs that are independent within the same stage get the same order,
  // matching the hand-tuned convention (e.g., dpT and dv both at order 2,
  // dq and dk both at order 1).
  //
  // Depth = number of same-stage MMA predecessors in the DDG.
  // This groups independent MMAs into the same cluster.
  llvm::DenseMap<unsigned, int> mmaDepthInStage;
  for (auto &mma : mmas) {
    int depth = 0;
    for (auto &other : mmas) {
      if (other.stage != mma.stage || other.nodeIdx == mma.nodeIdx)
        continue;
      // Check if 'other' is a transitive predecessor of 'mma' (distance-0).
      llvm::DenseSet<unsigned> visited;
      llvm::SmallVector<unsigned> worklist;
      worklist.push_back(mma.nodeIdx);
      visited.insert(mma.nodeIdx);
      bool found = false;
      while (!worklist.empty() && !found) {
        unsigned cur = worklist.pop_back_val();
        for (const auto *edge : ddg.getInEdges(cur)) {
          if (edge->distance > 0)
            continue;
          if (edge->srcIdx == other.nodeIdx) {
            found = true;
            break;
          }
          if (visited.insert(edge->srcIdx).second)
            worklist.push_back(edge->srcIdx);
        }
      }
      if (found)
        depth++;
    }
    mmaDepthInStage[mma.nodeIdx] = depth;
  }

  for (auto &mma : mmas) {
    int cluster = mmaDepthInStage[mma.nodeIdx];
    std::string json = "{\"stage\": \"" + std::to_string(mma.stage) +
                       "\", \"order\": \"" + std::to_string(cluster) + "\"}";
    mma.op->setAttr("tt.autows", StringAttr::get(ctx, json));

    LDBG("MMA annotation: stage=" << mma.stage << " order=" << cluster << " on "
                                  << *mma.op);
  }

  if (!mmas.empty())
    LDBG("Emitted tt.autows on " << mmas.size() << " MMA ops");
}

// ============================================================================
// Step 3: Derive per-resource buffer depths from modulo schedule
// ============================================================================

// Blackwell sm_100 SMEM budget (reserve some for barriers/scratch).
constexpr int kSmemBudgetBytes = 228 * 1024;

// Fallback trip count used when the loop bounds are not constant.
// Marked via `tripCountIsEstimated = true` so callers can detect it.
// Chosen as a small but non-trivial number (e.g., a 256x256 GEMM with
// BLOCK_K=64 has K_TILES=4) — keeps stage/buffer-depth heuristics from
// degenerating, but downstream code that needs a precise trip count
// must check the `tripCountIsEstimated` flag.
constexpr int kEstimatedTripCount = 4;

// computeBufferDepths removed — buffer allocation is now done via
// allocateBuffersForLoop on the ScheduleGraph (stage-diff based).

// ============================================================================
// Phase 0d: Build ScheduleGraph from DDG + Schedule
// ============================================================================

static ttg::ScheduleNode
convertDDGNode(const ttg::DDGNode &ddgNode, unsigned nodeId,
               const ttg::ModuloScheduleResult &sched) {
  ttg::ScheduleNode sn;
  sn.id = nodeId;
  sn.op = ddgNode.op;
  sn.pipeline = ddgNode.pipeline;
  sn.latency = ddgNode.latency;
  sn.selfLatency = ddgNode.selfLatency;

  auto cycleIt = sched.nodeToCycle.find(ddgNode.idx);
  if (cycleIt != sched.nodeToCycle.end()) {
    sn.cycle = cycleIt->second;
    sn.stage = cycleIt->second / sched.II;
  }

  if (ddgNode.isSuperNode) {
    sn.prologueLatency = ddgNode.prologueLatency;
  }
  return sn;
}

/// Step 2.5: Compute dense cluster IDs within each stage.
/// Ops in the same stage are sorted by cycle; same cycle → same cluster,
/// different cycle → different cluster (lower cycle = lower cluster ID).
static void computeClusterIds(ttg::ScheduleLoop &loop) {
  // Group node indices by stage
  llvm::DenseMap<int, SmallVector<unsigned>> stageToNodes;
  for (auto &node : loop.nodes) {
    stageToNodes[node.stage].push_back(node.id);
  }

  for (auto &[stage, nodeIds] : stageToNodes) {
    // Collect unique cycles in this stage, sorted
    SmallVector<int> cycles;
    for (unsigned nid : nodeIds)
      cycles.push_back(loop.nodes[nid].cycle);
    llvm::sort(cycles);
    cycles.erase(llvm::unique(cycles), cycles.end());

    // Build cycle → dense cluster ID map
    llvm::DenseMap<int, int> cycleToCluster;
    for (int i = 0, e = cycles.size(); i < e; ++i)
      cycleToCluster[cycles[i]] = i;

    // Assign cluster IDs
    for (unsigned nid : nodeIds)
      loop.nodes[nid].cluster = cycleToCluster[loop.nodes[nid].cycle];
  }
}

/// Build a child ScheduleLoop for an inner scf.for loop (super-node).
static unsigned buildChildScheduleLoop(scf::ForOp innerLoop,
                                       ttg::ScheduleGraph &graph,
                                       const ttg::LatencyModel &model) {
  auto innerDDG = ttg::DataDependenceGraph::build(innerLoop, model);
  unsigned loopId = graph.addLoop(innerLoop);
  auto &schedLoop = graph.getLoop(loopId);

  if (innerDDG.getNumNodes() == 0)
    return loopId;

  auto innerSched = ttg::runModuloScheduling(innerDDG);
  if (failed(innerSched))
    return loopId;

  schedLoop.II = innerSched->II;
  schedLoop.maxStage = innerSched->getMaxStage();

  int tcStart = innerSched->II;
  for (const auto &node : innerDDG.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC) {
      auto it = innerSched->nodeToCycle.find(node.idx);
      if (it != innerSched->nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  schedLoop.prologueLatency = tcStart;

  schedLoop.tripCount = kEstimatedTripCount;
  schedLoop.tripCountIsEstimated = true;
  {
    auto lb = innerLoop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
    auto ub = innerLoop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
    auto step = innerLoop.getStep().getDefiningOp<arith::ConstantIntOp>();
    if (lb && ub && step && step.value() > 0) {
      int64_t tc = (ub.value() - lb.value() + step.value() - 1) / step.value();
      if (tc > 0) {
        schedLoop.tripCount = static_cast<int>(tc);
        schedLoop.tripCountIsEstimated = false;
      }
    }
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (const auto &ddgNode : innerDDG.getNodes()) {
    unsigned nodeId = schedLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = nodeId;
    auto sn = convertDDGNode(ddgNode, nodeId, *innerSched);

    if (ddgNode.isSuperNode) {
      if (auto nestedLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildScheduleLoop(nestedLoop, graph, model);
        sn.childPipelineId = childId;
      }
    }

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = nodeId;
  }

  for (const auto &ddgEdge : innerDDG.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::ScheduleEdge se;
    se.srcId = srcIt->second;
    se.dstId = dstIt->second;
    se.latency = ddgEdge.latency;
    se.distance = ddgEdge.distance;
    schedLoop.edges.push_back(se);
  }

  // Step 2.5: compute cluster IDs
  computeClusterIds(schedLoop);

  return loopId;
}

/// Build the top-level ScheduleLoop for a scheduled loop.
static unsigned buildScheduleLoop(scf::ForOp loop,
                                  const ttg::DataDependenceGraph &ddg,
                                  const ttg::ModuloScheduleResult &sched,
                                  ttg::ScheduleGraph &graph,
                                  const ttg::LatencyModel &model) {
  unsigned loopId = graph.addLoop(loop);
  auto &schedLoop = graph.getLoop(loopId);
  schedLoop.II = sched.II;
  schedLoop.maxStage = sched.getMaxStage();

  int tcStart = sched.II;
  for (const auto &node : ddg.getNodes()) {
    if (node.pipeline == ttg::HWPipeline::TC || node.isSuperNode) {
      auto it = sched.nodeToCycle.find(node.idx);
      if (it != sched.nodeToCycle.end())
        tcStart = std::min(tcStart, it->second);
    }
  }
  schedLoop.prologueLatency = tcStart;

  // Extract trip count
  schedLoop.tripCount = kEstimatedTripCount;
  schedLoop.tripCountIsEstimated = true;
  {
    auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
    auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
    auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
    if (lb && ub && step && step.value() > 0) {
      int64_t tc = (ub.value() - lb.value() + step.value() - 1) / step.value();
      if (tc > 0) {
        schedLoop.tripCount = static_cast<int>(tc);
        schedLoop.tripCountIsEstimated = false;
      }
    }
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (const auto &ddgNode : ddg.getNodes()) {
    unsigned nodeId = schedLoop.nodes.size();
    ddgToPipe[ddgNode.idx] = nodeId;
    auto sn = convertDDGNode(ddgNode, nodeId, sched);

    if (ddgNode.isSuperNode) {
      if (auto innerLoop = dyn_cast<scf::ForOp>(ddgNode.op)) {
        unsigned childId = buildChildScheduleLoop(innerLoop, graph, model);
        sn.childPipelineId = childId;
        // Do NOT overwrite sn.prologueLatency: it was copied from
        // ddgNode.prologueLatency by convertDDGNode and represents the
        // latency the parent scheduler actually used for this super-node's
        // edge model. The child's recomputed prologueLatency belongs to
        // the child ScheduleLoop's own metadata; for empty/unscheduled
        // children it's 0 and would underestimate the super-node here.
      }
    }

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = nodeId;
  }

  for (const auto &ddgEdge : ddg.getEdges()) {
    auto srcIt = ddgToPipe.find(ddgEdge.srcIdx);
    auto dstIt = ddgToPipe.find(ddgEdge.dstIdx);
    if (srcIt == ddgToPipe.end() || dstIt == ddgToPipe.end())
      continue;
    ttg::ScheduleEdge se;
    se.srcId = srcIt->second;
    se.dstId = dstIt->second;
    se.latency = ddgEdge.latency;
    se.distance = ddgEdge.distance;
    schedLoop.edges.push_back(se);
  }

  // Step 2.5: compute cluster IDs
  computeClusterIds(schedLoop);

  return loopId;
}

/// Top-level: build a ScheduleGraph from DDG + schedule result.
/// Phase 0 only (DDG→nodes/edges). Buffer allocation is a separate step.
static ttg::ScheduleGraph
buildScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                   const ttg::ModuloScheduleResult &sched,
                   const ttg::LatencyModel &model) {
  ttg::ScheduleGraph graph;
  buildScheduleLoop(loop, ddg, sched, graph, model);
  return graph;
}

// ============================================================================
// Pass A: Modulo Scheduling
// ============================================================================

/// The main pass.
struct ModuloSchedulePass
    : public PassWrapper<ModuloSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuloSchedulePass)

  ModuloSchedulePass() = default;
  ModuloSchedulePass(const ModuloSchedulePass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "nvgpu-modulo-schedule"; }

  StringRef getDescription() const override {
    return "Modulo scheduling for warp specialization (Pass A)";
  }

  // Test-only knob: when set, dump the ScheduleGraph to llvm::errs()
  // unconditionally. Used by lit tests in opt builds, where `-debug-only`
  // is unavailable because LLVM_DEBUG is compiled out.
  Option<bool> printScheduleGraph{
      *this, "print-schedule-graph",
      llvm::cl::desc("Dump the ScheduleGraph to stderr unconditionally "
                     "(test-only; bypasses LLVM_DEBUG)"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    // Find loops that directly contain MMA ops in their body (not nested).
    // Unlike the original innermost-loop filter, this handles deeply nested
    // kernels like FA backward where MMAs are at depth 3-4 with epilogue
    // loops nested deeper.
    SmallVector<scf::ForOp> innerLoops;
    moduleOp.walk([&](scf::ForOp loop) {
      // Check direct children of loop body for MMA/load ops.
      bool hasTMALoad = false;
      bool hasMMAv5 = false;
      bool hasExistingAnnotation = false;
      for (auto &op : loop.getBody()->without_terminator()) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(&op))
          hasTMALoad = true;
        if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(&op))
          hasTMALoad = true;
        if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(&op)) {
          hasMMAv5 = true;
          if (op.hasAttr("tt.autows"))
            hasExistingAnnotation = true;
        }
      }
      if (!hasTMALoad && !hasMMAv5)
        return;
      // Skip loops that already have hand-tuned tt.autows annotations
      // from Python attrs=. These are set at the Python level and
      // propagated through accelerateMatmul. Re-annotating would
      // override the hand-tuned schedule.
      if (hasExistingAnnotation) {
        LDBG("Skipping loop with existing tt.autows annotations");
        return;
      }
      innerLoops.push_back(loop);
    });

    LDBG("Found " << innerLoops.size() << " innermost loop(s)");

    for (auto innerLoop : innerLoops) {
      // Build DDG for this inner loop.
      auto ddg = ttg::DataDependenceGraph::build(innerLoop, model);
      if (ddg.getNumNodes() == 0)
        continue;

      LDBG("DDG: " << ddg.getNumNodes() << " nodes, " << ddg.getEdges().size()
                   << " edges");

      // Run modulo scheduling.
      // Count key ops for diagnostics.
      int nMEM = 0, nTC = 0, nCUDA = 0, nSFU = 0, nNONE = 0;
      for (const auto &node : ddg.getNodes()) {
        switch (node.pipeline) {
        case ttg::HWPipeline::MEM:
          nMEM++;
          break;
        case ttg::HWPipeline::TC:
          nTC++;
          break;
        case ttg::HWPipeline::CUDA:
          nCUDA++;
          break;
        case ttg::HWPipeline::SFU:
          nSFU++;
          break;
        case ttg::HWPipeline::NONE:
          nNONE++;
          break;
        }
      }
      LDBG("Running scheduling on "
           << ddg.getNumNodes() << " nodes (MEM=" << nMEM << " TC=" << nTC
           << " CUDA=" << nCUDA << " SFU=" << nSFU << " NONE=" << nNONE << ")");
      auto schedResult = ttg::runModuloScheduling(ddg);
      if (failed(schedResult)) {
        LDBG("Scheduling FAILED");
        continue;
      }
      LDBG("Scheduling SUCCESS: II=" << schedResult->II);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PASS-A] Schedule: II=" << schedResult->II << " ResMII="
                 << ddg.computeResMII() << " RecMII=" << ddg.computeRecMII()
                 << " maxStage=" << schedResult->getMaxStage() << "\n");

      // Log per-node schedule.
      LLVM_DEBUG({
        for (const auto &node : ddg.getNodes()) {
          auto it = schedResult->nodeToCycle.find(node.idx);
          if (it == schedResult->nodeToCycle.end())
            continue;
          int cycle = it->second;
          int stage = cycle / schedResult->II;
          llvm::dbgs() << "[PASS-A]   N" << node.idx
                       << "  cycle=" << cycle << "  stage=" << stage
                       << "  " << ttg::getPipelineName(node.pipeline)
                       << "  selfLat=" << node.selfLatency << "  ";
          node.op->print(
              llvm::dbgs(),
              OpPrintingFlags().skipRegions().elideLargeElementsAttrs());
          llvm::dbgs() << "\n";
        }
      });

      // Emit tt.autows annotations on MMA ops instead of loop.stage attrs.
      // tt.autows survives through the WS pass (which preserves discardable
      // attrs on MMA ops) and is read by scheduleKeyOpsAnnotation() inside
      // the WS pass's internal scheduleLoops call.
      //
      // We don't emit loop.stage/loop.cluster here because:
      // 1. The WS pass's scheduleLoops overwrites them anyway
      // 2. Their presence sets stageAssigned=true which disables
      //    annotation-based scheduling in scheduleLoops
      //
      // emitMMAAnnotations internally skips annotation when all MMAs end
      // up in the same stage (no multi-stage partition found).
      emitMMAAnnotations(innerLoop, ddg, *schedResult);

      // Emit tt.num_stages so downstream pipelining recognises this loop
      // as scheduled. Even single-stage (maxStage=0) loops need the attr
      // present — without it, they're treated as unpipelined and skip
      // latency/buffering behaviour.
      if (!innerLoop->hasAttr(tt::kNumStagesAttrName)) {
        int numStages = schedResult->getMaxStage() + 1;
        auto ctx = innerLoop.getContext();
        innerLoop->setAttr(
            tt::kNumStagesAttrName,
            IntegerAttr::get(IntegerType::get(ctx, 32), numStages));
      }

      // Build ScheduleGraph for analysis/debug.
      auto pipelineGraph =
          buildScheduleGraph(innerLoop, ddg, *schedResult, model);

      LLVM_DEBUG({
        llvm::dbgs()
            << "[PASS-A] === Inner Loop ScheduleGraph ===\n";
        pipelineGraph.dump();
      });
      if (printScheduleGraph) {
        llvm::errs() << "[PASS-A] === Inner Loop ScheduleGraph ===\n";
        pipelineGraph.dump(llvm::errs());
      }

      // Clean up tt.modulo_cycle — internal attr, not needed downstream.
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

      // Log per-node outer DDG schedule.
      LLVM_DEBUG({
        for (const auto &node : outerDDG.getNodes()) {
          auto it = outerSched->nodeToCycle.find(node.idx);
          if (it == outerSched->nodeToCycle.end())
            continue;
          int cycle = it->second;
          int stage = cycle / outerSched->II;
          llvm::dbgs() << "[PASS-A]   N" << node.idx
                       << "  cycle=" << cycle << "  stage=" << stage
                       << "  " << ttg::getPipelineName(node.pipeline)
                       << "  selfLat=" << node.selfLatency << "  "
                       << node.op->getName().getStringRef() << "\n";
        }
      });

      auto outerGraph =
          buildScheduleGraph(outerLoop, outerDDG, *outerSched, model);

      LLVM_DEBUG({
        llvm::dbgs()
            << "[PASS-A] === Outer Loop ScheduleGraph (BEFORE expand) ===\n";
        outerGraph.dump();
      });
      if (printScheduleGraph) {
        llvm::errs()
            << "[PASS-A] === Outer Loop ScheduleGraph (BEFORE expand) ===\n";
        outerGraph.dump(llvm::errs());
      }

      // Emit outer loop schedule attrs for downstream passes.
      emitScheduleAttributes(outerLoop, outerDDG, *outerSched);

      // Clean up tt.modulo_cycle — internal attr, not needed downstream.
      for (auto &op : outerLoop.getBody()->without_terminator())
        op.removeAttr("tt.modulo_cycle");
    }

  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}

void registerNVGPUModuloSchedule() { PassRegistration<ModuloSchedulePass>(); }
} // namespace mlir
