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

#include <limits>

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

// Fallback trip count when the loop bounds aren't constant-foldable.
// Used so kernel_time_cost can give a finite (rather than div-by-zero)
// answer for cost-based depth reduction.
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

// ============================================================================
// Phase 1: Buffer Allocation
// ============================================================================

static ttg::MemoryKind classifyMemoryKind(Operation *op) {
  if (isa<ttng::TMEMAllocOp>(op))
    return ttg::MemoryKind::TMEM;
  // Both local_alloc (pre-lowering) and async_tma_copy (post-lowering)
  // produce SMEM buffers that need multi-buffering.
  if (isa<ttg::LocalAllocOp, ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return ttg::MemoryKind::SMEM;
  // TMA stores need an SMEM staging buffer — the TMA engine reads from
  // SMEM, not registers. The buffer is allocated during TMA lowering but
  // must be accounted for in the SMEM budget here.
  if (isa<tt::DescriptorStoreOp, ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return ttg::MemoryKind::SMEM;
  return ttg::MemoryKind::Register;
}

static void extractBufferShape(Operation *op, ttg::ScheduleBuffer &buf) {
  Type resultType;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    resultType = alloc.getType();
  else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op))
    resultType = tmemAlloc.getType();
  else if (auto tmaCopy = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    resultType = tmaCopy.getResult().getType();
  else if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op))
    resultType = storeOp.getSrc().getType();
  else if (op->getNumResults() > 0)
    resultType = op->getResult(0).getType();

  auto extractFromShapedType = [&](llvm::ArrayRef<int64_t> shape, Type elemTy) {
    for (auto dim : shape) {
      if (dim <= 0 || ShapedType::isDynamic(dim))
        return;
    }
    if (!elemTy.isIntOrFloat())
      return;
    for (auto dim : shape)
      buf.shape.push_back(dim);
    buf.elementBitWidth = elemTy.getIntOrFloatBitWidth();
  };

  if (auto memDesc = dyn_cast_or_null<ttg::MemDescType>(resultType)) {
    extractFromShapedType(memDesc.getShape(), memDesc.getElementType());
  } else if (auto tensorType = dyn_cast_or_null<RankedTensorType>(resultType)) {
    extractFromShapedType(tensorType.getShape(), tensorType.getElementType());
  }
}

/// Step 3: Compute buffer count from cycle-level lifetime.
///
/// Design doc formula:
///   lifetime(R) = lastConsumerEnd - producerStart
///   num_buffers(R) = floor(lifetime(R) / II) + 1
///
/// For loop-carried edges (distance > 0), the consumer in iteration i+d
/// effectively ends at: consumerEnd + d * II (in absolute time).
/// This is equivalent to adding d * II to the lifetime.
static unsigned computeBufferCount(const ttg::ScheduleLoop &loop,
                                   unsigned producerNodeId) {
  const auto &producer = loop.getNode(producerNodeId);
  int prodCycle = producer.cycle;
  int II = loop.II;
  if (II <= 0)
    return 1;

  // Find the latest consumer end cycle among direct successors.
  // The DDG has edges from this producer to every op that reads its
  // result, so walking outgoing edges covers all consumers.
  int lastConsumerEnd = prodCycle;
  for (const auto &edge : loop.edges) {
    if (edge.srcId != producerNodeId)
      continue;
    const auto &consumer = loop.getNode(edge.dstId);
    // Consumer hold time: use selfLatency (pipeline occupancy) when
    // available, falling back to latency (result-ready time). This
    // matches computeBufferLifetimes so that count and lifetime are
    // computed consistently.
    int hold = consumer.selfLatency ? consumer.selfLatency : consumer.latency;
    int consumerEnd = consumer.cycle + hold + edge.distance * II;
    lastConsumerEnd = std::max(lastConsumerEnd, consumerEnd);
  }

  int lifetime = lastConsumerEnd - prodCycle;
  int numBuffers = lifetime / II + 1;
  return static_cast<unsigned>(std::max(numBuffers, 1));
}

static void allocateBuffersForLoop(ttg::ScheduleLoop &loop) {
  llvm::SmallVector<unsigned, 4> dataBufferIds;
  for (auto &node : loop.nodes) {
    if (!node.op)
      continue;

    auto kind = classifyMemoryKind(node.op);
    if (kind == ttg::MemoryKind::Register)
      continue;

    unsigned bufId = loop.buffers.size();
    ttg::ScheduleBuffer buf;
    buf.id = bufId;
    buf.kind = kind;
    buf.defOp = node.op;
    extractBufferShape(node.op, buf);

    buf.count = computeBufferCount(loop, node.id);

    loop.buffers.push_back(buf);
    node.producesBuffer = bufId;

    if (buf.count > 1)
      dataBufferIds.push_back(bufId);

    llvm::DenseSet<unsigned> markedConsumers;
    for (const auto &edge : loop.edges) {
      if (edge.srcId == node.id && markedConsumers.insert(edge.dstId).second)
        loop.nodes[edge.dstId].consumesBuffers.push_back(bufId);
    }
  }

  // Equalize co-consumed buffer depths: buffers that feed the same
  // consumer op (e.g., A and B tiles both feeding MMA) must have the
  // same depth. Otherwise the shallower buffer limits the pipeline
  // depth and the deeper buffer wastes SMEM.
  //
  // Walk upstream from each node to collect all SMEM buffers it
  // transitively consumes (through NONE-pipeline intermediaries like
  // memdesc_trans), then equalize their depths.
  for (const auto &node : loop.nodes) {
    // Only equalize for pipeline ops that consume multiple buffers.
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;

    // Collect all SMEM buffers reachable upstream through edges.
    llvm::SmallVector<unsigned> upstreamBufs;
    llvm::SmallVector<unsigned> worklist;
    llvm::DenseSet<unsigned> visited;
    worklist.push_back(node.id);
    visited.insert(node.id);
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      const auto &curNode = loop.nodes[cur];
      // If this node produces an SMEM buffer, collect it.
      if (curNode.producesBuffer != UINT_MAX &&
          curNode.producesBuffer < loop.buffers.size() &&
          loop.buffers[curNode.producesBuffer].kind ==
              ttg::MemoryKind::SMEM)
        upstreamBufs.push_back(curNode.producesBuffer);
      // Walk upstream through predecessors (NONE-pipeline only, to
      // avoid crossing pipeline boundaries).
      for (const auto &edge : loop.edges) {
        if (edge.dstId != cur || edge.distance > 0)
          continue;
        const auto &pred = loop.nodes[edge.srcId];
        if (pred.pipeline != ttg::HWPipeline::NONE &&
            pred.pipeline != ttg::HWPipeline::MEM)
          continue;
        if (visited.insert(edge.srcId).second)
          worklist.push_back(edge.srcId);
      }
    }

    if (upstreamBufs.size() <= 1)
      continue;

    unsigned maxDepth = 0;
    for (unsigned bufId : upstreamBufs)
      maxDepth = std::max(maxDepth, loop.buffers[bufId].count);
    for (unsigned bufId : upstreamBufs) {
      if (loop.buffers[bufId].count != maxDepth) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Step3] Equalized buf" << bufId << " depth from "
                   << loop.buffers[bufId].count << " to " << maxDepth
                   << " (co-consumed by "
                   << node.op->getName().getStringRef() << ")\n");
        loop.buffers[bufId].count = maxDepth;
      }
    }
  }

  for (unsigned dataBufId : dataBufferIds) {
    unsigned barId = loop.buffers.size();
    ttg::ScheduleBuffer bar;
    bar.id = barId;
    bar.kind = ttg::MemoryKind::BARRIER;
    bar.count = loop.buffers[dataBufId].count;
    bar.defOp = loop.buffers[dataBufId].defOp;
    bar.pairedBufferId = dataBufId;
    loop.buffers[dataBufId].pairedBufferId = barId;
    loop.buffers.push_back(bar);
  }
}

// ============================================================================
// Step 4.6: Global Memory Budget Check and Reduction
// ============================================================================

// Blackwell sm_100 TMEM budget. Logical capacity is 128 lanes × 512 cols ×
// 4 bytes/col = 256KB.
constexpr int64_t kTmemBudgetBytes = 128 * 512 * 4;

// Forward decl — defined under Step 4.5 below; called by reduceBuffersForBudget
// to refresh PhysicalBuffer sizes after a depth reduction.
static void buildPhysicalBuffers(ttg::ScheduleLoop &loop);

/// Compute total SMEM/TMEM usage. Buffers in the same merge group share
/// a physical allocation sized to the largest member at the deepest
/// count, so we charge each group exactly once via its PhysicalBuffer.
/// Unmerged data buffers and all BARRIER buffers (always SMEM) are
/// charged individually.
static int64_t computeTotalMemory(const ttg::ScheduleLoop &loop,
                                  ttg::MemoryKind targetKind) {
  int64_t total = 0;

  // Charge each materialized physical buffer once.
  for (const auto &pb : loop.physicalBuffers) {
    bool isTarget = (pb.kind == targetKind);
    if (targetKind == ttg::MemoryKind::SMEM &&
        pb.kind == ttg::MemoryKind::BARRIER)
      isTarget = true;
    if (isTarget)
      total += pb.totalBytes();
  }

  // Charge unmerged buffers (mergeGroupId == UINT_MAX) directly.
  for (const auto &buf : loop.buffers) {
    if (buf.mergeGroupId != UINT_MAX)
      continue;
    bool isTarget = (buf.kind == targetKind);
    if (targetKind == ttg::MemoryKind::SMEM &&
        buf.kind == ttg::MemoryKind::BARRIER)
      isTarget = true;
    if (!isTarget)
      continue;
    if (buf.kind != ttg::MemoryKind::BARRIER &&
        (buf.shape.empty() || buf.elementBitWidth == 0))
      continue;
    total += buf.totalBytes();
  }
  return total;
}

static int64_t computeTotalSmem(const ttg::ScheduleLoop &loop) {
  return computeTotalMemory(loop, ttg::MemoryKind::SMEM);
}
static int64_t computeTotalTmem(const ttg::ScheduleLoop &loop) {
  return computeTotalMemory(loop, ttg::MemoryKind::TMEM);
}

/// Compute the buffer lifetime (in cycles) for a given producer node.
static int computeBufferLifetime(const ttg::ScheduleLoop &loop,
                                 unsigned producerNodeId) {
  const auto &producer = loop.getNode(producerNodeId);
  int prodCycle = producer.cycle;
  int lastConsumerEnd = prodCycle;
  for (const auto &edge : loop.edges) {
    if (edge.srcId != producerNodeId)
      continue;
    const auto &consumer = loop.getNode(edge.dstId);
    int holdTime = std::max(consumer.selfLatency, consumer.latency);
    int end = consumer.cycle + holdTime + edge.distance * loop.II;
    lastConsumerEnd = std::max(lastConsumerEnd, end);
  }
  return lastConsumerEnd - prodCycle;
}

/// Cost (design doc §1437-1477): kernel time increase per byte saved by
/// reducing this buffer's depth by 1. Lower = greedily reduce first.
///
/// new_lifetime_bound = (count - 1) × II. If lifetime exceeds it, the
/// producer must stall and effective II grows; otherwise depth reduction
/// is free of latency impact (ii_increase = 0).
///
/// time_increase = ii_increase × tripCount  (loop region)
///               = ii_increase             (non-loop region — single pass)
/// cost          = time_increase / size_bytes_saved
static double kernelTimeCost(const ttg::ScheduleLoop &loop,
                             const ttg::ScheduleBuffer &buf) {
  if (buf.count <= 1 || buf.kind == ttg::MemoryKind::BARRIER)
    return std::numeric_limits<double>::infinity();
  if (loop.II <= 0)
    return std::numeric_limits<double>::infinity();
  int lifetime = buf.liveEnd - buf.liveStart;
  int newCount = static_cast<int>(buf.count) - 1;
  int newLifetimeBound = newCount * loop.II;
  int iiIncrease = 0;
  if (lifetime > newLifetimeBound) {
    int newII = (lifetime + newCount - 1) / newCount;
    iiIncrease = newII - loop.II;
  }
  int tc = loop.tripCount > 0 ? loop.tripCount : 1;
  double timeIncrease = static_cast<double>(iiIncrease) * tc;
  int64_t saved = buf.sizeBytes();
  if (saved <= 0)
    return std::numeric_limits<double>::infinity();
  return timeIncrease / static_cast<double>(saved);
}

/// Build co-consumed buffer groups: buffers that transitively feed the
/// same pipeline op must have the same depth.
static llvm::SmallVector<llvm::SmallVector<unsigned>>
buildCoConsumedGroups(const ttg::ScheduleLoop &loop) {
  // Map each SMEM buffer to a group ID via union-find.
  llvm::DenseMap<unsigned, unsigned> bufToGroup;
  unsigned nextGroup = 0;

  for (const auto &node : loop.nodes) {
    if (node.pipeline == ttg::HWPipeline::NONE)
      continue;
    // Walk upstream to collect all SMEM buffers feeding this node.
    llvm::SmallVector<unsigned> upstreamBufs;
    llvm::SmallVector<unsigned> worklist = {node.id};
    llvm::DenseSet<unsigned> visited = {node.id};
    while (!worklist.empty()) {
      unsigned cur = worklist.pop_back_val();
      const auto &curNode = loop.nodes[cur];
      if (curNode.producesBuffer != UINT_MAX &&
          curNode.producesBuffer < loop.buffers.size() &&
          loop.buffers[curNode.producesBuffer].kind ==
              ttg::MemoryKind::SMEM)
        upstreamBufs.push_back(curNode.producesBuffer);
      for (const auto &edge : loop.edges) {
        if (edge.dstId != cur || edge.distance > 0)
          continue;
        const auto &pred = loop.nodes[edge.srcId];
        if (pred.pipeline != ttg::HWPipeline::NONE &&
            pred.pipeline != ttg::HWPipeline::MEM)
          continue;
        if (visited.insert(edge.srcId).second)
          worklist.push_back(edge.srcId);
      }
    }
    if (upstreamBufs.size() <= 1)
      continue;
    // Union all upstream buffers into the same group. Collect all
    // existing group IDs, pick the smallest, and rewrite all members
    // of every touched group to use that ID (transitive merge).
    llvm::DenseSet<unsigned> existingGroups;
    for (unsigned bufId : upstreamBufs) {
      auto it = bufToGroup.find(bufId);
      if (it != bufToGroup.end())
        existingGroups.insert(it->second);
    }
    unsigned mergedGroupId;
    if (existingGroups.empty()) {
      mergedGroupId = nextGroup++;
    } else {
      mergedGroupId = *std::min_element(existingGroups.begin(),
                                        existingGroups.end());
      // Rewrite all buffers in the other groups to the merged ID.
      if (existingGroups.size() > 1) {
        for (auto &[bufId, gid] : bufToGroup) {
          if (existingGroups.count(gid))
            gid = mergedGroupId;
        }
      }
    }
    for (unsigned bufId : upstreamBufs)
      bufToGroup[bufId] = mergedGroupId;
  }

  // Collect groups.
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned>> groupMap;
  for (auto &[bufId, gid] : bufToGroup)
    groupMap[gid].push_back(bufId);
  llvm::SmallVector<llvm::SmallVector<unsigned>> groups;
  for (auto &[gid, members] : groupMap)
    groups.push_back(std::move(members));
  return groups;
}

/// Reduce all buffers in a co-consumed group to the given depth.
static void reduceGroupToDepth(ttg::ScheduleLoop &loop,
                               const llvm::SmallVector<unsigned> &group,
                               unsigned newDepth) {
  for (unsigned bufId : group) {
    if (loop.buffers[bufId].count > newDepth) {
      loop.buffers[bufId].count = newDepth;
      unsigned barId = loop.buffers[bufId].pairedBufferId;
      if (barId != UINT_MAX)
        loop.buffers[barId].count = newDepth;
    }
  }
}

/// Step 4.6: If buffer allocation exceeds SMEM/TMEM budget, greedily reduce
/// buffer depths using the kernel_time_cost metric from the design doc.
/// Co-consumed buffers (feeding the same pipeline op) are reduced together.
/// After reduction, recompute II from the tightest buffer constraint:
///   new_II = max over reduced buffers of ceil(lifetime / new_depth).
/// The schedule (op placement) stays fixed — only II and buffer depths change.
static bool reduceBuffersForBudget(ttg::ScheduleLoop &loop,
                                   int64_t smemReserved = 0) {
  // Precompute buffer lifetimes (from the original schedule, before reduction).
  llvm::DenseMap<unsigned, int> bufLifetimes;
  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    for (const auto &node : loop.nodes) {
      if (node.producesBuffer == buf.id) {
        bufLifetimes[i] = computeBufferLifetime(loop, node.id);
        break;
      }
    }
  }

  // Build co-consumed groups so we reduce them together.
  auto coGroups = buildCoConsumedGroups(loop);
  // Map bufId → group index for quick lookup.
  llvm::DenseMap<unsigned, unsigned> bufToGroupIdx;
  for (unsigned g = 0; g < coGroups.size(); ++g)
    for (unsigned bufId : coGroups[g])
      bufToGroupIdx[bufId] = g;

  int originalII = loop.II;

  int64_t effectiveSmemBudget = kSmemBudgetBytes - smemReserved;
  if (smemReserved > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] SMEM reserved by other regions: " << smemReserved
               << " B, effective budget: " << effectiveSmemBudget << " B\n");
  }

  // SMEM reduction: greedily reduce the cheapest buffer first.
  // When a buffer is in a co-consumed group, reduce the entire group.
  while (computeTotalSmem(loop) > effectiveSmemBudget) {
    int bestIdx = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      const auto &buf = loop.buffers[i];
      if (buf.kind != ttg::MemoryKind::SMEM || buf.count <= 1)
        continue;
      double cost = kernelTimeCost(loop, buf);
      if (cost < bestCost) {
        bestCost = cost;
        bestIdx = i;
      }
    }
    if (bestIdx < 0)
      break;
    unsigned newDepth = loop.buffers[bestIdx].count - 1;
    // If this buffer is in a co-consumed group, reduce the whole group.
    auto groupIt = bufToGroupIdx.find(bestIdx);
    if (groupIt != bufToGroupIdx.end()) {
      reduceGroupToDepth(loop, coGroups[groupIt->second], newDepth);
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.6] Reduced co-consumed group (buf" << bestIdx
                 << " + partners) to count=" << newDepth << "\n");
    } else {
      loop.buffers[bestIdx].count = newDepth;
      unsigned barId = loop.buffers[bestIdx].pairedBufferId;
      if (barId != UINT_MAX)
        loop.buffers[barId].count = newDepth;
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.6] Reduced SMEM buf" << bestIdx << " to count="
                 << newDepth << "\n");
    }
  }

  // TMEM reduction
  while (computeTotalTmem(loop) > kTmemBudgetBytes) {
    int bestIdx = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < loop.buffers.size(); ++i) {
      auto &buf = loop.buffers[i];
      if (buf.kind != ttg::MemoryKind::TMEM || buf.count <= 1)
        continue;
      double cost = kernelTimeCost(loop, buf);
      if (cost < bestCost) {
        bestCost = cost;
        bestIdx = i;
      }
    }
    if (bestIdx < 0)
      break;
    loop.buffers[bestIdx].count--;
    unsigned barId = loop.buffers[bestIdx].pairedBufferId;
    if (barId != UINT_MAX)
      loop.buffers[barId].count = loop.buffers[bestIdx].count;
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] Reduced TMEM buf" << bestIdx << " to count="
               << loop.buffers[bestIdx].count << "\n");
  }

  // Recompute II from reduced buffer depths.
  // new_II = max over all buffers of ceil(lifetime / depth).
  int newII = originalII;
  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    auto it = bufLifetimes.find(i);
    if (it == bufLifetimes.end() || buf.count <= 0)
      continue;
    int requiredII = (it->second + buf.count - 1) / buf.count;
    if (requiredII > newII) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.6] buf" << i << " lifetime=" << it->second
                 << " depth=" << buf.count
                 << " → requires II=" << requiredII << "\n");
      newII = requiredII;
    }
  }

  if (newII != originalII) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] Raising II from " << originalII
               << " to " << newII
               << " due to buffer depth reduction\n");
    loop.II = newII;
    loop.maxStage = 0;
    for (const auto &node : loop.nodes) {
      int stage = node.cycle / newII;
      loop.maxStage = std::max(loop.maxStage, stage);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[Step4.6] New maxStage=" << loop.maxStage << "\n");
  }

  int64_t smemUsed = computeTotalSmem(loop);
  int64_t tmemUsed = computeTotalTmem(loop);
  bool smemOk = smemUsed <= kSmemBudgetBytes;
  bool tmemOk = tmemUsed <= kTmemBudgetBytes;
  LLVM_DEBUG(llvm::dbgs() << "[Step4.6] Budget: SMEM " << smemUsed << "/"
                          << kSmemBudgetBytes << (smemOk ? " OK" : " EXCEEDED")
                          << ", TMEM " << tmemUsed << "/" << kTmemBudgetBytes
                          << (tmemOk ? " OK" : " EXCEEDED") << "\n");
  if (!smemOk || !tmemOk) {
    llvm::errs() << "[Step4.6] WARNING: memory budget exceeded"
                 << " (all reducible buffers at count=1). "
                 << "SMEM: " << smemUsed << "/" << kSmemBudgetBytes
                 << ", TMEM: " << tmemUsed << "/" << kTmemBudgetBytes << "\n";
  }
  return smemOk && tmemOk;
}

// ============================================================================
// Step 4.5: Lifetime-Aware Buffer Merging
// ============================================================================

/// Faithful port of design doc §1156-1177 `intervals_overlap_modular`:
/// project each interval onto [0, II), split if it wraps, then test all
/// (a-half, b-half) pairs for plain interval overlap.
static bool intervalsOverlapModularSingle(int aStart, int aEnd, int bStart,
                                          int bEnd, int II) {
  if (II <= 0)
    return true;
  // Empty intervals can't overlap anything.
  if (aEnd <= aStart || bEnd <= bStart)
    return false;

  auto mod = [II](int x) {
    int r = x % II;
    return r < 0 ? r + II : r;
  };
  int aS = mod(aStart);
  int aE = mod(aEnd);
  int bS = mod(bStart);
  int bE = mod(bEnd);
  // A live interval whose duration is >= II covers the entire ring.
  if (aEnd - aStart >= II || bEnd - bStart >= II)
    return true;

  llvm::SmallVector<std::pair<int, int>, 2> aHalves;
  if (aS < aE)
    aHalves.push_back({aS, aE});
  else if (aS > aE) {
    aHalves.push_back({aS, II});
    aHalves.push_back({0, aE});
  } else {
    // aS == aE with non-empty original ⇒ wraps fully.
    return true;
  }
  llvm::SmallVector<std::pair<int, int>, 2> bHalves;
  if (bS < bE)
    bHalves.push_back({bS, bE});
  else if (bS > bE) {
    bHalves.push_back({bS, II});
    bHalves.push_back({0, bE});
  } else {
    return true;
  }
  for (auto [s1, e1] : aHalves)
    for (auto [s2, e2] : bHalves)
      if (s1 < e2 && s2 < e1)
        return true;
  return false;
}

/// Faithful port of design doc §1180-1203 `any_instances_overlap`.
/// For each (d1, d2) pair of in-flight buffer instances, shift interval B
/// by (d2 - d1) * II and test for modular overlap. Two resources can share
/// a physical buffer only if NO (d1, d2) pair produces overlap.
static bool anyInstancesOverlap(int aStart, int aEnd, int bStart, int bEnd,
                                unsigned aDepth, unsigned bDepth, int II) {
  if (II <= 0)
    return true;
  for (unsigned d1 = 0; d1 < std::max(1u, aDepth); ++d1) {
    for (unsigned d2 = 0; d2 < std::max(1u, bDepth); ++d2) {
      int offset = (static_cast<int>(d2) - static_cast<int>(d1)) * II;
      if (intervalsOverlapModularSingle(aStart, aEnd, bStart + offset,
                                        bEnd + offset, II))
        return true;
    }
  }
  return false;
}

/// Compute and store [liveStart, liveEnd) for every data buffer in the loop.
/// Lifetime is producer cycle → max(consumer.cycle + consumer.selfLatency)
/// across direct consumer edges, with loop-carried edges adjusted by
/// distance × II. Paired barriers inherit the data buffer's interval
/// (per design doc §215).
static void computeBufferLifetimes(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;
  for (auto &buf : loop.buffers) {
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    for (const auto &node : loop.nodes) {
      if (node.producesBuffer != buf.id)
        continue;
      buf.liveStart = node.cycle;
      int lastEnd = node.cycle;
      for (const auto &edge : loop.edges) {
        if (edge.srcId != node.id)
          continue;
        const auto &consumer = loop.getNode(edge.dstId);
        // Use selfLatency (occupancy) over latency (result-ready) for
        // the consumer's hold time on the resource.
        int hold =
            consumer.selfLatency ? consumer.selfLatency : consumer.latency;
        int end =
            consumer.cycle + hold + static_cast<int>(edge.distance) * loop.II;
        lastEnd = std::max(lastEnd, end);
      }
      buf.liveEnd = lastEnd;
      break;
    }
  }
  // Mirror data-buffer intervals onto their paired barriers.
  for (auto &bar : loop.buffers) {
    if (bar.kind != ttg::MemoryKind::BARRIER)
      continue;
    if (bar.pairedBufferId == UINT_MAX)
      continue;
    const auto &data = loop.buffers[bar.pairedBufferId];
    bar.liveStart = data.liveStart;
    bar.liveEnd = data.liveEnd;
  }
}

/// Cycle-freedom check (design doc §1129-1137 / §1216): merging buffers A
/// and B adds an implicit edge "last_consumer_of_A happens-before
/// producer_of_B". Reject the merge if it would create a cycle in the
/// node-level dependency graph.
///
/// We model the merge as a candidate edge (last_consumer(B'), producer(A))
/// added per pair, where (A, B') ranges over (existing group members,
/// candidate). Run a forward reachability from producer(A) over all real
/// edges PLUS the prospective merge edges; if producer(B') is reachable
/// before the new edge is added the other direction, we'd close a cycle.
static bool mergeIntroducesCycle(const ttg::ScheduleLoop &loop,
                                 llvm::ArrayRef<unsigned> groupMembers,
                                 unsigned candidate) {
  // Collect (producer, lastConsumer) per buffer in {groupMembers + candidate}.
  auto findProducer = [&](unsigned bufId) -> unsigned {
    for (const auto &n : loop.nodes)
      if (n.producesBuffer == bufId)
        return n.id;
    return UINT_MAX;
  };
  auto lastConsumers = [&](unsigned bufId) {
    llvm::SmallVector<unsigned, 4> result;
    unsigned prodId = findProducer(bufId);
    if (prodId == UINT_MAX)
      return result;
    for (const auto &e : loop.edges)
      if (e.srcId == prodId)
        result.push_back(e.dstId);
    return result;
  };

  // Build adjacency for plain DDG (intra-iteration edges only — cross-
  // iteration edges close their own loops, which is fine).
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> adj;
  for (const auto &e : loop.edges)
    if (e.distance == 0)
      adj[e.srcId].push_back(e.dstId);

  // Collect candidate-induced edges: for every existing member M and the
  // candidate C, both directions of "last_consumer happens-before producer"
  // are added as additional edges to test. Coloring will pick a serial
  // order, but for the cycle test, both possibilities are checked.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 8> proposed;
  auto addPair = [&](unsigned aBuf, unsigned bBuf) {
    unsigned bProd = findProducer(bBuf);
    if (bProd == UINT_MAX)
      return;
    for (unsigned cons : lastConsumers(aBuf))
      proposed.push_back({cons, bProd});
  };
  for (unsigned m : groupMembers) {
    addPair(m, candidate);
    addPair(candidate, m);
  }

  // BFS from each proposed edge's source over (real edges + all proposed
  // edges except itself); a cycle exists iff we can reach back to itself.
  for (size_t i = 0; i < proposed.size(); ++i) {
    auto [src, dst] = proposed[i];
    llvm::DenseSet<unsigned> visited;
    llvm::SmallVector<unsigned, 16> stack{dst};
    while (!stack.empty()) {
      unsigned u = stack.pop_back_val();
      if (!visited.insert(u).second)
        continue;
      if (u == src)
        return true;
      for (unsigned v : adj.lookup(u))
        stack.push_back(v);
      for (size_t j = 0; j < proposed.size(); ++j)
        if (j != i && proposed[j].first == u)
          stack.push_back(proposed[j].second);
    }
  }
  return false;
}

/// Cost guard (design doc §1418-1429): merging is only beneficial when
/// max(size) × max(count) < sum(size × count). Otherwise, the physical
/// buffer (sized to the largest member with the deepest count) wastes
/// more memory than separate allocations.
static bool shouldMerge(const ttg::ScheduleLoop &loop,
                        llvm::ArrayRef<unsigned> groupMembers,
                        unsigned candidate) {
  int64_t separateCost = 0;
  int64_t maxSize = 0;
  unsigned maxCount = 0;
  auto accum = [&](unsigned bufId) {
    const auto &b = loop.buffers[bufId];
    int64_t sz = b.sizeBytes();
    separateCost += sz * static_cast<int64_t>(b.count);
    maxSize = std::max(maxSize, sz);
    maxCount = std::max(maxCount, b.count);
  };
  for (unsigned m : groupMembers)
    accum(m);
  accum(candidate);
  int64_t mergedCost = maxSize * static_cast<int64_t>(maxCount);
  return mergedCost < separateCost;
}

/// Materialize PhysicalBuffer entries from each merge group. Per design
/// doc §1140-1147: physical size = max(member.sizeBytes), physical count =
/// max(member.count).
static void buildPhysicalBuffers(ttg::ScheduleLoop &loop) {
  loop.physicalBuffers.clear();
  llvm::DenseMap<unsigned, unsigned> groupToPhys;
  for (const auto &buf : loop.buffers) {
    if (buf.mergeGroupId == UINT_MAX)
      continue;
    auto it = groupToPhys.find(buf.mergeGroupId);
    if (it == groupToPhys.end()) {
      ttg::PhysicalBuffer pb;
      pb.id = buf.mergeGroupId;
      pb.kind = buf.kind;
      pb.sizeBytes = buf.sizeBytes();
      pb.count = buf.count;
      pb.memberBufferIds.push_back(buf.id);
      groupToPhys[buf.mergeGroupId] = loop.physicalBuffers.size();
      loop.physicalBuffers.push_back(std::move(pb));
    } else {
      auto &pb = loop.physicalBuffers[it->second];
      pb.sizeBytes = std::max(pb.sizeBytes, buf.sizeBytes());
      pb.count = std::max(pb.count, buf.count);
      pb.memberBufferIds.push_back(buf.id);
    }
  }
}

/// Step 4.5: Merge buffers with non-overlapping lifetimes.
/// Greedy interval-graph coloring with three guards:
///   1. Same storage kind (SMEM only merges with SMEM).
///   2. No modular interval overlap across all (d1, d2) buffer instances.
///   3. should_merge cost guard — never inflate memory by merging.
///   4. Cycle-freedom — never introduce a deadlock-prone dependency.
static void mergeNonOverlappingBuffers(ttg::ScheduleLoop &loop) {
  if (loop.II <= 0)
    return;
  computeBufferLifetimes(loop);

  unsigned nextGroupId = 0;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> groups;

  for (unsigned i = 0; i < loop.buffers.size(); ++i) {
    auto &buf = loop.buffers[i];
    if (buf.kind == ttg::MemoryKind::BARRIER ||
        buf.kind == ttg::MemoryKind::Register)
      continue;
    // Skip buffers with zero-length lifetime — they have no producer/
    // consumer pattern we can reason about and shouldn't be merged blindly.
    if (buf.liveEnd == buf.liveStart)
      continue;

    bool merged = false;
    for (unsigned g = 0; g < groups.size(); ++g) {
      bool canMerge = true;
      for (unsigned memberIdx : groups[g]) {
        const auto &member = loop.buffers[memberIdx];
        if (member.kind != buf.kind) {
          canMerge = false;
          break;
        }
        if (anyInstancesOverlap(member.liveStart, member.liveEnd, buf.liveStart,
                                buf.liveEnd, member.count, buf.count,
                                loop.II)) {
          canMerge = false;
          break;
        }
      }
      if (!canMerge)
        continue;
      if (!shouldMerge(loop, groups[g], i)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Step4.5] Skip merge buf" << i << " into group " << g
                   << " (cost guard: would inflate)\n");
        continue;
      }
      if (mergeIntroducesCycle(loop, groups[g], i)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Step4.5] Skip merge buf" << i << " into group " << g
                   << " (would create dependency cycle)\n");
        continue;
      }
      buf.mergeGroupId = g;
      groups[g].push_back(i);
      merged = true;
      LLVM_DEBUG(llvm::dbgs()
                 << "[Step4.5] Merged buf" << i << " into group " << g
                 << " (live=[" << buf.liveStart << "," << buf.liveEnd << "), "
                 << (buf.kind == ttg::MemoryKind::SMEM ? "SMEM" : "TMEM")
                 << ")\n");
      break;
    }
    if (!merged) {
      buf.mergeGroupId = nextGroupId;
      groups.push_back({i});
      nextGroupId++;
    }
  }

  buildPhysicalBuffers(loop);

  LLVM_DEBUG(llvm::dbgs() << "[Step4.5] " << loop.buffers.size()
                          << " buffers -> " << loop.physicalBuffers.size()
                          << " physical groups\n");
}

/// Top-level: build a ScheduleGraph from DDG + schedule result.
/// Includes Phase 0 (DDG→nodes/edges), Step 2.5 (clusters),
/// Step 3 (buffer allocation), Step 4.5 (merging), Step 4.6 (budget).
static ttg::ScheduleGraph
buildScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                   const ttg::ModuloScheduleResult &sched,
                   const ttg::LatencyModel &model,
                   int64_t smemReserved = 0) {
  ttg::ScheduleGraph graph;
  buildScheduleLoop(loop, ddg, sched, graph, model);

  for (auto &schedLoop : graph.loops) {
    // Step 3: Buffer allocation (cycle-based depths)
    allocateBuffersForLoop(schedLoop);

    // Step 4.5: Merge non-overlapping buffers (before budget check —
    // merging reduces memory footprint, giving tighter budget).
    mergeNonOverlappingBuffers(schedLoop);

    // Step 4.6: Global memory budget check and reduction.
    // smemReserved accounts for SMEM used by other simultaneously-live
    // regions (e.g., inner loop buffers during outer loop execution).
    reduceBuffersForBudget(schedLoop, smemReserved);
  }

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

  /// DDG transformation hooks for iterative refinement.
  /// Return true if any DDG was modified (triggers re-scheduling).

  /// Pass A.5: Data partitioning — split underutilized loop ops into sub-tiles.
  /// TODO: Implement when needed.
  bool applyDataPartitioning(ModuleOp moduleOp,
                             const ttg::LatencyModel &model) {
    return false;
  }

  /// Pass A.7: Epilogue subtiling — split monolithic TMA stores into
  /// independent sub-chains for better pipeline interleaving.
  ///
  /// The actual IR splitting (tensor extract_slice + sub-stores) requires
  /// encoding-aware tensor operations that are better handled at a higher
  /// level (Python frontend or dedicated TTGIR pass). This hook identifies
  /// candidate stores and returns true if subtiling would be beneficial,
  /// allowing the iterative loop to signal that the DDG should be refined.
  ///
  /// For now, this is a stub that returns false. The epilogue subtiling
  /// concept is demonstrated by the list scheduler test
  /// (epilogue-subtiling.mlir) which shows interleaving of pre-split
  /// independent store chains.
  /// TODO: Implement tensor splitting with proper TTGIR encoding handling.
  bool applyEpilogueSubtiling(ModuleOp moduleOp,
                              const ttg::LatencyModel &model) {
    return false;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    // ================================================================
    // Iterative scheduling loop (design doc Pass A orchestrator)
    //
    // Each iteration: schedule → derive depths → check budget →
    // apply DDG transformations → re-run if any DDG changed.
    // Converges in 1-2 iterations.
    // ================================================================
    constexpr int kMaxIterations = 3;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      LDBG("=== Iterative scheduling: iteration " << iteration << " ===");

      // Step 1: Find loops that directly contain MMA ops in their body
      // (not nested). Unlike the original innermost-loop filter, this
      // handles deeply nested kernels like FA backward where MMAs are at
      // depth 3-4 with epilogue loops nested deeper.
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
             << " CUDA=" << nCUDA << " SFU=" << nSFU << " NONE=" << nNONE
             << ")");
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
            llvm::dbgs() << "[PASS-A]   N" << node.idx << "  cycle=" << cycle
                         << "  stage=" << stage << "  "
                         << ttg::getPipelineName(node.pipeline)
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
          llvm::dbgs() << "[PASS-A] === Inner Loop ScheduleGraph ===\n";
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

        LDBG("Outer schedule: II="
             << outerSched->II << " ResMII=" << outerDDG.computeResMII()
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
            llvm::dbgs() << "[PASS-A]   N" << node.idx << "  cycle=" << cycle
                         << "  stage=" << stage << "  "
                         << ttg::getPipelineName(node.pipeline)
                         << "  selfLat=" << node.selfLatency << "  "
                         << node.op->getName().getStringRef() << "\n";
          }
        });

        // Build ScheduleGraph for outer loop.
        auto outerGraph =
            buildScheduleGraph(outerLoop, outerDDG, *outerSched, model);

        LLVM_DEBUG({
          llvm::dbgs() << "[PASS-A] === Outer Loop ScheduleGraph ===\n";
          outerGraph.dump();
        });
        if (printScheduleGraph) {
          llvm::errs()
              << "[PASS-A] === Outer Loop ScheduleGraph (BEFORE expand) ===\n";
          outerGraph.dump(llvm::errs());
        }

        // Kernel-wide SMEM budget check: the inner loop's buffers and
        // the outer loop's buffers are live simultaneously. If their
        // combined SMEM exceeds the budget, re-run the inner loop's
        // budget check with the outer loop's SMEM reserved.
        int64_t outerSmem = computeTotalSmem(outerGraph.getLoop(0));
        if (outerSmem > 0 && !innerLoops.empty()) {
          LDBG("Kernel-wide check: outer SMEM=" << outerSmem);
          for (auto innerLoop : innerLoops) {
            auto innerDDG =
                ttg::DataDependenceGraph::build(innerLoop, model);
            if (innerDDG.getNumNodes() == 0)
              continue;
            auto innerSched = ttg::runModuloScheduling(innerDDG);
            if (failed(innerSched))
              continue;

            auto innerGraph = buildScheduleGraph(innerLoop, innerDDG,
                                                 *innerSched, model,
                                                 outerSmem);
            auto &innerSchedLoop = innerGraph.getLoop(0);

            if (innerSchedLoop.II != innerSched->II) {
              LDBG("Kernel-wide: inner II adjusted "
                   << innerSched->II << " → " << innerSchedLoop.II
                   << ", maxStage=" << innerSchedLoop.maxStage
                   << " (outer SMEM reserved " << outerSmem << " B)");
              auto adjustedResult = *innerSched;
              adjustedResult.II = innerSchedLoop.II;
              emitScheduleAttributes(innerLoop, innerDDG, adjustedResult);
            }

            LLVM_DEBUG({
              llvm::dbgs()
                  << "[PASS-A] === Inner Loop ScheduleGraph (kernel-wide) "
                     "===\n";
              innerGraph.dump();
            });

            for (auto &op : innerLoop.getBody()->without_terminator())
              op.removeAttr("tt.modulo_cycle");
          }
        }

        // Emit outer loop schedule attrs for downstream passes.
        emitScheduleAttributes(outerLoop, outerDDG, *outerSched);

        // Clean up tt.modulo_cycle — internal attr, not needed downstream.
        for (auto &op : outerLoop.getBody()->without_terminator())
          op.removeAttr("tt.modulo_cycle");
      }

      // ================================================================
      // Iterative refinement: apply DDG transformations and check if
      // we need to re-schedule.
      // ================================================================
      bool ddgChanged = false;
      ddgChanged |= applyDataPartitioning(moduleOp, model);
      ddgChanged |= applyEpilogueSubtiling(moduleOp, model);

      if (!ddgChanged) {
        LDBG("Converged after " << iteration + 1 << " iteration(s)");
        break;
      }

      // Don't strip attrs on the last iteration — preserve the valid
      // schedule from this iteration rather than leaving the loop
      // unscheduled.
      if (iteration + 1 >= kMaxIterations) {
        LDBG("Hit iteration limit (" << kMaxIterations
                                     << ") — keeping last valid schedule");
        break;
      }

      LDBG("DDG changed by transformation — re-scheduling");

      // Strip OUTPUT schedule attrs before re-running. Do NOT strip
      // INPUT attrs like tt.num_stages (user-provided pipeline depth).
      moduleOp.walk([](Operation *op) {
        op->removeAttr("loop.stage");
        op->removeAttr("loop.cluster");
        op->removeAttr("tt.modulo_cycle");
        op->removeAttr("tt.modulo_ii");
        op->removeAttr("tt.num_buffers");
        op->removeAttr("tt.self_latency");
        op->removeAttr("tt.scheduled_max_stage");
      });
    } // end iterative loop
  }
};

// ============================================================================
// Pass A.6: List scheduling for non-loop regions
// ============================================================================
//
// Degenerate Rau's algorithm — no modulo wrap, no loop-carried edges. All
// ops get stage 0; goal is minimum makespan instead of minimum II. Lives
// here (not its own file) so the ScheduleGraph is constructed in one place
// alongside the modulo case. DEBUG_TYPE is redefined for this section so
// debug output is gated by `-debug-only=nvgpu-list-schedule` per reviewer
// feedback (was previously leaking under `-debug-only=modulo-scheduling-rau`).

#undef DEBUG_TYPE
#undef DBGS
#undef LDBG
#define DEBUG_TYPE "nvgpu-list-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

/// Per-pipeline occupancy tracker without modulo wrap. Each pipeline has
/// a "next free" cycle — no fixed II, no wrap-around. Mirrors the modulo
/// reservation table for the linear (no-wrap) case.
struct PipelineTracker {
  llvm::DenseMap<ttg::HWPipeline, int> nextFree;

  /// Earliest cycle the pipeline is available. The `duration` parameter
  /// is the prospective op's hold time and is unused here (the tracker
  /// only records when the previously placed op's hold ends); kept for
  /// API symmetry with the modulo case.
  int findFreeSlot(int earliest, ttg::HWPipeline pipeline,
                   int /*duration*/) const {
    if (pipeline == ttg::HWPipeline::NONE)
      return earliest;
    auto it = nextFree.find(pipeline);
    int pipeReady = (it != nextFree.end()) ? it->second : 0;
    return std::max(earliest, pipeReady);
  }

  void reserve(int cycle, ttg::HWPipeline pipeline, int duration) {
    if (pipeline == ttg::HWPipeline::NONE)
      return;
    nextFree[pipeline] = std::max(nextFree.lookup(pipeline), cycle + duration);
  }
};

/// Earliest cycle a node may start, given predecessors already placed.
/// Predecessor result-ready time is `pred.cycle + edge.latency`; the DDG
/// builder records the producer's `latency` (result-ready) on outgoing
/// edges, so we don't add `pred.selfLatency` separately.
static int listEarliestStart(unsigned nodeIdx,
                             const ttg::DataDependenceGraph &ddg,
                             const llvm::DenseMap<unsigned, int> &scheduled) {
  int earliest = 0;
  for (const auto *edge : ddg.getInEdges(nodeIdx)) {
    auto it = scheduled.find(edge->srcIdx);
    if (it == scheduled.end())
      continue;
    earliest = std::max(earliest, it->second + edge->latency);
  }
  return earliest;
}

/// Priority-based list scheduling on the DDG. Minimises makespan rather
/// than II. Critical-path height is the priority (highest first).
static FailureOr<ttg::ListScheduleResult>
runListScheduling(const ttg::DataDependenceGraph &ddg) {
  if (ddg.getNumNodes() == 0)
    return failure();

  auto heights = ddg.computeCriticalPathHeights();

  llvm::SmallVector<unsigned> order;
  for (unsigned i = 0; i < ddg.getNumNodes(); ++i)
    order.push_back(i);
  llvm::sort(order, [&](unsigned a, unsigned b) {
    if (heights[a] != heights[b])
      return heights[a] > heights[b];
    return a < b;
  });

  PipelineTracker tracker;
  llvm::DenseMap<unsigned, int> scheduled;

  for (unsigned nodeIdx : order) {
    const auto &node = ddg.getNode(nodeIdx);
    int duration = std::max(node.selfLatency, 1);
    if (node.pipeline == ttg::HWPipeline::NONE)
      duration = 1;

    int earliest = listEarliestStart(nodeIdx, ddg, scheduled);
    int slot = tracker.findFreeSlot(earliest, node.pipeline, duration);

    tracker.reserve(slot, node.pipeline, duration);
    scheduled[nodeIdx] = slot;

    LLVM_DEBUG(DBGS() << "  List placed N" << nodeIdx << " ("
                      << ttg::getPipelineName(node.pipeline)
                      << " dur=" << duration << ") at cycle=" << slot << "\n");
  }

  // makespan = max(start + occupancy) across all nodes.
  int makespan = 0;
  for (auto &[idx, cycle] : scheduled) {
    const auto &node = ddg.getNode(idx);
    makespan = std::max(makespan, cycle + std::max(node.selfLatency, 1));
  }

  LLVM_DEBUG(DBGS() << "List schedule: makespan=" << makespan
                    << " nodes=" << ddg.getNumNodes() << "\n");

  ttg::ListScheduleResult result;
  result.makespan = makespan;
  result.nodeToCycle = std::move(scheduled);
  return result;
}

/// Build a ScheduleGraph from a list-scheduled loop. All ops get stage 0,
/// cluster from cycle rank.
static ttg::ScheduleGraph
buildListScheduleGraph(scf::ForOp loop, const ttg::DataDependenceGraph &ddg,
                       const ttg::ListScheduleResult &result) {
  ttg::ScheduleGraph graph;
  unsigned loopId = graph.addLoop(loop);
  auto &schedLoop = graph.getLoop(loopId);
  schedLoop.II = result.makespan; // For non-loop regions, "II" = makespan
  schedLoop.maxStage = 0;

  for (const auto &ddgNode : ddg.getNodes()) {
    ttg::ScheduleNode sn;
    sn.id = schedLoop.nodes.size();
    sn.op = ddgNode.op;
    sn.pipeline = ddgNode.pipeline;
    sn.latency = ddgNode.latency;
    sn.selfLatency = ddgNode.selfLatency;
    sn.stage = 0;

    auto cycleIt = result.nodeToCycle.find(ddgNode.idx);
    if (cycleIt != result.nodeToCycle.end())
      sn.cycle = cycleIt->second;

    schedLoop.nodes.push_back(sn);
    schedLoop.opToNodeId[ddgNode.op] = sn.id;
  }

  llvm::DenseMap<unsigned, unsigned> ddgToPipe;
  for (unsigned i = 0; i < ddg.getNodes().size(); ++i)
    ddgToPipe[ddg.getNodes()[i].idx] = i;

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

  // Cluster IDs (same logic as Step 2.5, all stage 0).
  SmallVector<int> cycles;
  for (const auto &node : schedLoop.nodes)
    cycles.push_back(node.cycle);
  llvm::sort(cycles);
  cycles.erase(llvm::unique(cycles), cycles.end());
  llvm::DenseMap<int, int> cycleToCluster;
  for (int i = 0, e = cycles.size(); i < e; ++i)
    cycleToCluster[cycles[i]] = i;
  for (auto &node : schedLoop.nodes)
    node.cluster = cycleToCluster[node.cycle];

  return graph;
}

struct ListSchedulePass
    : public PassWrapper<ListSchedulePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ListSchedulePass)

  StringRef getArgument() const override { return "nvgpu-list-schedule"; }

  StringRef getDescription() const override {
    return "List scheduling for non-loop regions (Pass A.6)";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    ttg::LatencyModel model;

    moduleOp.walk([&](scf::ForOp loop) {
      if (loop->hasAttr("tt.modulo_ii"))
        return;

      bool hasPipelineOps = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<tt::DescriptorLoadOp, tt::DescriptorStoreOp,
                ttng::AsyncTMACopyGlobalToLocalOp, ttng::TCGen5MMAOp,
                ttng::TCGen5MMAScaledOp, ttng::TMEMLoadOp>(op))
          hasPipelineOps = true;
      });
      if (!hasPipelineOps)
        return;

      auto ddg = ttg::DataDependenceGraph::build(loop, model);
      if (ddg.getNumNodes() == 0)
        return;

      LDBG("List scheduling loop with " << ddg.getNumNodes() << " nodes");

      auto result = runListScheduling(ddg);
      if (failed(result)) {
        LDBG("List scheduling FAILED");
        return;
      }

      LDBG("List schedule: makespan=" << result->makespan);

      auto schedGraph = buildListScheduleGraph(loop, ddg, *result);

      LLVM_DEBUG({
        llvm::dbgs() << "[A.6] === List ScheduleGraph ===\n";
        schedGraph.dump();
      });

      auto ctx = loop.getContext();
      for (const auto &schedLoop : schedGraph.loops) {
        for (const auto &node : schedLoop.nodes) {
          if (!node.op)
            continue;
          node.op->setAttr(tt::kLoopStageAttrName,
                           IntegerAttr::get(IntegerType::get(ctx, 32), 0));
          node.op->setAttr(
              tt::kLoopClusterAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), node.cluster));
        }
      }

      // Default unscheduled ops to stage 0, max cluster.
      int maxCluster = 0;
      for (const auto &schedLoop : schedGraph.loops)
        for (const auto &node : schedLoop.nodes)
          maxCluster = std::max(maxCluster, node.cluster);
      for (auto &op : loop.getBody()->without_terminator()) {
        if (!op.hasAttr(tt::kLoopStageAttrName))
          op.setAttr(tt::kLoopStageAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), 0));
        if (!op.hasAttr(tt::kLoopClusterAttrName))
          op.setAttr(tt::kLoopClusterAttrName,
                     IntegerAttr::get(IntegerType::get(ctx, 32), maxCluster));
      }

      // Mark the loop scheduled so downstream `processScheduledLoop`
      // (which gates on `tt.modulo_ii`) preserves the schedule attrs.
      // `tt.list_schedule_makespan` distinguishes list-scheduled loops
      // from true modulo-scheduled ones for any consumer that cares.
      loop->setAttr("tt.modulo_ii", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                     result->makespan));
      loop->setAttr(
          "tt.list_schedule_makespan",
          IntegerAttr::get(IntegerType::get(ctx, 32), result->makespan));
    });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNVGPUModuloSchedule() {
  return std::make_unique<ModuloSchedulePass>();
}

void registerNVGPUModuloSchedule() { PassRegistration<ModuloSchedulePass>(); }

std::unique_ptr<Pass> createNVGPUListSchedule() {
  return std::make_unique<ListSchedulePass>();
}

void registerNVGPUListSchedule() { PassRegistration<ListSchedulePass>(); }
} // namespace mlir
